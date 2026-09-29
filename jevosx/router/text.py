"""Where TYPE_TEXT values come from. Jev chooses; it never writes free text.

1. Prepared text slots (from the caller, or quoted strings, phrases and URLs in the goal). Jev picks the slot.
2. When the goal asks for new text ("write a poem"), a writer composes it (`GENERATE`): Apple's on-device model or an
   optional OpenAI-compatible model (see jevosx.writer). Jev still picks the field and whether to type.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
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

# Well-known sites that people name without a domain ("go to facebook", "on youtube"): offered as addresses to type.
KNOWN_SITES = {
    "facebook": "facebook.com",
    "marketplace": "facebook.com/marketplace",
    "instagram": "instagram.com",
    "youtube": "youtube.com",
    "gmail": "mail.google.com",
    "google": "google.com",
    "wikipedia": "wikipedia.org",
    "amazon": "amazon.com",
    "ebay": "ebay.com",
    "twitter": "x.com",
    "linkedin": "linkedin.com",
    "reddit": "reddit.com",
    "netflix": "netflix.com",
    "github": "github.com",
    "whatsapp": "web.whatsapp.com",
    "chatgpt": "chatgpt.com",
    "outlook": "outlook.live.com",
    "spotify": "open.spotify.com",
    "tiktok": "tiktok.com",
    "pinterest": "pinterest.com",
    "airbnb": "airbnb.com",
    "booking": "booking.com",
    "etsy": "etsy.com",
    "vinted": "vinted.com",
    "craigslist": "craigslist.org",
    "maltapark": "maltapark.com",
}
_SITE_NAMED = re.compile(
    r"\b(?:go\s+to|goto|open|visit|browse|on|at|to|in|into|log\s*in\s+to|sign\s*in\s+to)\s+(?:the\s+)?(?P<site>[a-z][\w-]*)",
    re.IGNORECASE,
)
_SELLING = re.compile(r"\b(?:sell|selling|listing|list\s+(?:a|an|my|the)|post\s+(?:a|an)|advert)\b", re.IGNORECASE)
MARKETPLACE_NEW_ITEM = "facebook.com/marketplace/create/item"

# "a plastic welding gun for 40 euros" → item "plastic welding gun", price "40".
_AMOUNT = r"\d{1,7}(?:[.,]\d{1,2})?"
_PRICE = re.compile(
    rf"\bfor\s+(?:only\s+|just\s+)?(?:[€$£]\s*(?P<a>{_AMOUNT})|(?P<b>{_AMOUNT})\s*(?P<cur>euros?|eur|€|dollars?|usd|\$"
    rf"|pounds?|gbp|£)?(?![\w.,]))",
    re.IGNORECASE,
)
_ARTICLES = frozenset({"a", "an", "my", "the", "this", "our", "some", "one"})


def _item_and_price(goal: str) -> tuple[str | None, str | None]:
    for match in _PRICE.finditer(goal):
        amount = match.group("a") or match.group("b")
        if not (match.group("a") or match.group("cur") or _SELLING.search(goal)):
            continue  # a bare "for 10" is only a price when the goal is about selling
        tail = goal[: match.start()].split()[-8:]
        starts = [i for i, word in enumerate(tail) if word.lower() in _ARTICLES]
        words = tail[starts[-1] + 1 :] if starts else tail[-3:]
        item = " ".join(words).strip(" ,.;:!?\"'")
        return (item if 2 <= len(item) <= 80 else None), amount
    return None, None


def slots_from_goal(goal: str) -> dict[str, str]:
    """Text the goal clearly asks to type, offered to Jev as choosable slots (Jev never writes text itself).

    - quoted literals: 'type "hello world" into Notes' → {"quote_1": "hello world"}
    - the phrase after search/look up/look for/google/type/enter/say: "look for pictures of red flowers in Safari"
      → {"phrase_1": "pictures of red flowers"}
    - URLs and domains: "go to apple.com" → {"url_1": "apple.com"}; well-known sites: "go to facebook" →
      {"url_1": "facebook.com"}
    - an item and its price: "sell a plastic welding gun for 40 euros" → {"title": "Plastic welding gun",
      "price": "40"} (named after the "Title" field that listing forms use)
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
    lowered = unquoted.lower()
    if "marketplace" in lowered and _SELLING.search(unquoted):
        urls.append(MARKETPLACE_NEW_ITEM)  # straight to the "item for sale" form
    for match in _SITE_NAMED.finditer(unquoted):
        site = KNOWN_SITES.get(match.group("site").lower())
        if site and not any(site in url for url in urls):
            urls.append(site)
    for i, value in enumerate(dict.fromkeys(urls), start=1):
        slots[f"url_{i}"] = value
    item, price = _item_and_price(unquoted)
    if item:
        slots["title"] = item[:1].upper() + item[1:]
    if price:
        slots["price"] = price
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
    fetch: Callable[[], str | None] | None = field(default=None, compare=False, repr=False)  # read at typing time

    @property
    def secret(self) -> bool:
        return self.secure_only or bool(SECRET_NAME.search(self.name))

    @property
    def preview(self) -> str:
        if self.secret:
            return f"•••••• ({self.label})" if self.label else "••••••"
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

    def set_credentials(self, slots: Iterable[TextSlot]) -> None:
        """Replace the site-bound login slots (they follow the page on screen, see jevosx.logins)."""
        for name in [name for name, slot in self.slots.items() if slot.host is not None]:
            del self.slots[name]
        for slot in slots:
            self.slots[slot.name] = slot

    def sensitive_values(self) -> list[str]:
        """Values of site-bound slots that are not secret (usernames): masked wherever the state shows them."""
        return [s.value for s in self.slots.values() if s.host is not None and not s.secret and len(s.value) >= 3]

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
            value = slot.fetch() if slot.fetch is not None else slot.value
            if not value:
                raise TextUnavailableError(f"{slot.label or slot.name} is not available (missing from the Keychain?)")
            return ResolvedText(value, slot.secret, f"slot:{slot.name}", slot.label, slot.host, slot.secure_only)
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
