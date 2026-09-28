"""Where TYPE_TEXT values come from. Jev chooses; it never writes free text.

1. Prepared text slots (from the caller, or quoted strings, phrases and URLs in the goal). Jev picks the slot.
2. When the goal asks for new text ("write a poem"), a writer composes it (`GENERATE`): Apple's on-device model or an
   optional OpenAI-compatible model (see jevosx.writer). Jev still picks the field and whether to type.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import TextUnavailableError
from ..types import Observation, UIElement, clean_text
from ..writer.base import TextWriter
from ..writer.openai import LLMTextWriter

__all__ = ["GENERATE", "LLMTextWriter", "ResolvedText", "TextSlot", "TextSource", "slots_from_goal"]

GENERATE = "GENERATE"
SECRET_NAME = re.compile(r"secret|password|passcode|passwd|token|\bpin\b|otp", re.IGNORECASE)
_QUOTED = re.compile(r'"([^"\n]{1,500})"|“([^”\n]{1,500})”|`([^`\n]{1,500})`')


# Unquoted phrases that are clearly meant to be typed: "search the web for red flowers", "look up the weather",
# "type hello". Anything after these verbs, minus a trailing "in Safari"/"on Google"-style clause, becomes a slot.
_PHRASE = re.compile(
    r"\b(?:search(?:\s+(?:the\s+web|the\s+internet|online|google|the\s+browser))?(?:\s+for)?|look\s+(?:up|for)"
    r"|google|type|enter|say)\s+(?P<text>.+)",
    re.IGNORECASE,
)
_TRAILING_CLAUSE = re.compile(
    r"\s+(?:in|on|using|with|into|from|via)\s+(?:a\s+new\s+(?:window|tab)|the\s+(?:browser|web|internet)|google"
    r"|safari|chrome|google\s+chrome|firefox|arc|edge|brave|textedit|notes|finder|(?:the\s+)?search\s+(?:bar|field|box))"
    r"\b.*$",
    re.IGNORECASE,
)
_URL = re.compile(r"\b(?:https?://)?(?:[a-z0-9-]+\.)+(?:com|org|net|io|ai|dev|app|co|edu|gov|uk|de|fr)(?:/[^\s\"]*)?",
                  re.IGNORECASE)  # fmt: skip


def slots_from_goal(goal: str) -> dict[str, str]:
    """Text the goal clearly asks to type, offered to Jev as choosable slots (Jev never writes text itself).

    - quoted literals: 'type "hello world" into Notes' → {"quote_1": "hello world"}
    - the phrase after search/look up/look for/google/type/enter/say: "look for pictures of red flowers in Safari"
      → {"phrase_1": "pictures of red flowers"}
    - URLs and domains: "go to apple.com" → {"url_1": "apple.com"}
    """
    quoted = [next(g for g in match.groups() if g is not None) for match in _QUOTED.finditer(goal)]
    slots = {f"quote_{i}": value for i, value in enumerate(dict.fromkeys(quoted), start=1)}
    unquoted = _QUOTED.sub(" ", goal)
    phrases: list[str] = []
    for match in _PHRASE.finditer(unquoted):
        text = re.split(r"(?:^|\s+)(?:and|then)(?:\s+|$)|[,;]", match.group("text").strip(), maxsplit=1)[0]
        text = _TRAILING_CLAUSE.sub("", text).strip(" .!?:\"'")
        if 1 <= len(text) <= 200 and not _URL.fullmatch(text):
            phrases.append(text)
    for i, value in enumerate(dict.fromkeys(p for p in phrases if p not in quoted), start=1):
        slots[f"phrase_{i}"] = value
    urls = [m.group(0).rstrip(".,") for m in _URL.finditer(unquoted)]
    for i, value in enumerate(dict.fromkeys(urls), start=1):
        slots[f"url_{i}"] = value
    return slots


@dataclass(frozen=True, slots=True)
class TextSlot:
    name: str
    value: str
    # Credential slots (see jevosx.logins): bound to one site, never shown in history, passwords only into
    # password fields.
    host: str | None = None
    label: str | None = None  # shown instead of the value in previews, history and logs
    secure_only: bool = False

    @property
    def secret(self) -> bool:
        return self.secure_only or bool(SECRET_NAME.search(self.name))

    @property
    def preview(self) -> str:
        if self.secret:
            return "••••••"
        return self.label or clean_text(self.value, 60)


@dataclass(frozen=True, slots=True)
class ResolvedText:
    """What TYPE_TEXT will type, and where it came from."""

    text: str
    secret: bool
    source: str  # slot:<name> | model:<writer>
    label: str | None = None  # history/log display instead of the text
    host: str | None = None  # the web page host that must still be on screen when typing
    secure_only: bool = False


class TextSource:
    def __init__(
        self,
        slots: Mapping[str, str] | None = None,
        writer: TextWriter | None = None,
        *,
        generate: bool = True,
    ):
        self.slots = {name: TextSlot(name, value) for name, value in (slots or {}).items() if value}
        self.writer = writer
        # GENERATE is offered only when a writer exists and the goal asks for new text (see writer.wants_generation).
        self.generate = writer is not None and generate
        self._generated: dict[str, str] = {}

    @property
    def available(self) -> bool:
        return bool(self.slots) or self.generate

    def options(self) -> dict[str, dict[str, Any]]:
        """Criteria for the `text_slot` question."""
        options: dict[str, dict[str, Any]] = {
            name: {"text_name": name, "preview": slot.preview} for name, slot in self.slots.items()
        }
        if self.generate:
            options[GENERATE] = {"text_name": "generate", "preview": "the writer composes the text the goal asks for"}
        return options

    def default_option(self) -> str | None:
        options = list(self.options())
        return options[0] if len(options) == 1 else None

    def resolve(
        self,
        option: str | None,
        *,
        goal: str,
        element: UIElement,
        obs: Observation,
        history: list[dict[str, Any]],
    ) -> ResolvedText:
        """The text for the chosen option (a prepared slot, or the writer's composition)."""
        if option in self.slots:
            slot = self.slots[option]
            return ResolvedText(slot.value, slot.secret, f"slot:{slot.name}", slot.label, slot.host, slot.secure_only)
        if self.generate and self.writer is not None and (option == GENERATE or not self.slots):
            if element.secure:
                raise TextUnavailableError("refusing to generate text for a password field")
            # One composition per field and run: a retry types the same poem instead of writing a new one.
            key = element.signature
            if key not in self._generated:
                context = {
                    "goal": goal,
                    "field": {"label": element.label, "role": element.role_name, "current_value": element.value},
                    "app": obs.app.name,
                    "window": obs.window.title if obs.window else None,
                    "screen_text": obs.text[:4000],
                    "recent_actions": [h.get("action") for h in history[-6:]],
                }
                self._generated[key] = self.writer.write(context)
            return ResolvedText(self._generated[key], False, f"model:{self.writer.model}")
        raise TextUnavailableError("TYPE_TEXT was chosen but no text slot or writer can supply a value")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
