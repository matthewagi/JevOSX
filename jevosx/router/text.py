"""Where TYPE_TEXT values come from. Jev chooses; it never writes free text.

1. Prepared text slots (from the caller, or quoted strings in the goal). Jev picks the slot in the same request.
2. Optionally, a small OpenAI-compatible model writes a value when no slot fits (`GENERATE`). Off by default.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from ..errors import TextUnavailableError
from ..types import Observation, UIElement, clean_text
from .prompts import TEXT_WRITER

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

    @property
    def secret(self) -> bool:
        return bool(SECRET_NAME.search(self.name))

    @property
    def preview(self) -> str:
        return "••••••" if self.secret else clean_text(self.value, 60)


class LLMTextWriter:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 15.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self._http = httpx.Client(
            timeout=timeout_s, headers={"Authorization": f"Bearer {api_key}"}, transport=transport
        )

    def write(self, context: Mapping[str, Any]) -> str:
        response = self._http.post(
            self.url,
            json={
                "model": self.model,
                "max_tokens": 512,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": TEXT_WRITER},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ],
            },
        )
        if response.is_error:
            raise TextUnavailableError(f"text model returned HTTP {response.status_code}; nothing typed")
        try:
            output = json.loads(response.json()["choices"][0]["message"]["content"])
            value = output["text"]
            if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 4000:
                raise ValueError
        except (ValueError, KeyError, TypeError, IndexError):
            raise TextUnavailableError("text model returned no usable value; nothing typed") from None
        return value

    def close(self) -> None:
        self._http.close()


class TextSource:
    def __init__(self, slots: Mapping[str, str] | None = None, writer: LLMTextWriter | None = None):
        self.slots = {name: TextSlot(name, value) for name, value in (slots or {}).items() if value}
        self.writer = writer

    @property
    def available(self) -> bool:
        return bool(self.slots) or self.writer is not None

    def options(self) -> dict[str, dict[str, Any]]:
        """Criteria for the `text_slot` question."""
        options: dict[str, dict[str, Any]] = {
            name: {"text_name": name, "preview": slot.preview} for name, slot in self.slots.items()
        }
        if self.writer is not None:
            options[GENERATE] = {"text_name": "generate", "preview": "compose new text for this field from the goal"}
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
    ) -> tuple[str, bool, str]:
        """Return (text, is_secret, source) for the chosen option."""
        if option in self.slots:
            slot = self.slots[option]
            return slot.value, slot.secret, f"slot:{slot.name}"
        if self.writer is not None and (option == GENERATE or not self.slots):
            if element.secure:
                raise TextUnavailableError("refusing to generate text for a password field")
            context = {
                "goal": goal,
                "field": {"label": element.label, "role": element.role_name, "current_value": element.value},
                "app": obs.app.name,
                "window": obs.window.title if obs.window else None,
                "screen_text": obs.text[:4000],
                "recent_actions": [h.get("action") for h in history[-6:]],
            }
            return self.writer.write(context), False, f"model:{self.writer.model}"
        raise TextUnavailableError("TYPE_TEXT was chosen but no text slot or text model can supply a value")

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
