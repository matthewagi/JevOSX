"""Reading the goal with a language model: the steps it implies and the exact text it needs typed, once per run.

Pattern matching cannot read "go to facebook and prepare a product to sell on marketplace a plastic welding gun for
40 euros generic text" the way a person does. The writer's model (Apple's on-device model by default) reads it once
and answers in a fixed format:

    STEPS:
    1. Open facebook.com/marketplace in the browser
    2. Create a new listing and fill in title, price and description
    VALUES:
    website: facebook.com/marketplace
    title: Plastic welding gun
    price: 40
    description: Plastic welding gun in good working order, ideal for repairing bumpers and tanks.

The values become text slots that Jev can choose for the fields it picks; the steps become the plan (a hint for
ordering). Nothing here acts: every click and every field is still a Jev choice among observed ids. When no model is
available, or its answer cannot be used, the pattern-based slots in router/text.py are all there is.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from .errors import JevOSXError, TextUnavailableError
from .router.text import SECRET_NAME
from .writer.base import TextWriter

log = logging.getLogger("jevosx.planner")

READER_INSTRUCTIONS = """You read a request that a person gave to an assistant that operates their Mac.
Answer in exactly this format and nothing else:
STEPS:
1. <first step>
2. <next step>
VALUES:
<name>: <text to type>
ASK:
<name>: <question for the person>

STEPS: the concrete steps in order, at most 6, one short line each. Name the app or website when it is clear.
Only what the request asks for: no extra tasks, warnings or questions.
VALUES: every piece of text the assistant will have to type, one per line, each with a short lowercase name:
- a website as a bare address, for example website: facebook.com
- search words, for example search: population of Malta
- names, titles, file names, recipients and dates exactly as the request gives them
- amounts as numbers only, for example price: 40
- when the request asks for new text (a description, a message, generic text, a poem), write that text in full as
  the value, on one line, for example description: ...
- when the request is to sell or list an item, also the everyday category it belongs in, even if the request does
  not name it, for example category: Tools
- when the request is to save pictures or files, the folder as a path under ~, for example folder: ~/Desktop/dogs
  (a folder the request does not name goes in ~/Pictures, named after what the pictures show), and how many as
  count: 5 (5 when the request does not say)
Never include passwords or codes. Never invent personal details (names, emails, phone numbers, addresses) or facts
only the person knows (an item's condition, age or size) that are not in the request. If nothing needs typing,
write: VALUES: none
ASK: before starting, what the task cannot be finished without that the request does not give and only the person
can know or provide: files to upload (for example photos of an item to sell), an item's condition, which account,
a recipient. At most 3, one short question per line, each with a short lowercase name, for example
photos: Where are the photos of the item saved?
Not what you can decide yourself (a category, wording, a title, a folder, a file name, how many), not
preferences, never passwords. If nothing is
missing, write: ASK: none"""

_MULTI_PART = re.compile(r",|;|\bthen\b|\band\b|\bafter(?:wards)?\b|\bnext\b|\bfinally\b", re.IGNORECASE)
_STEP = re.compile(r"^\s*(?:\d{1,2}\s*[.)]|[-•*])\s*(?P<text>.+?)\s*$")
_VALUE = re.compile(r"^\s*(?:[-•*]\s*)?(?P<name>[A-Za-z][A-Za-z0-9 _-]{0,30}?)\s*[:=]\s*(?P<value>.+?)\s*$")
_EMPTY = frozenset({"none", "n/a", "na", "-", "nothing", "null", "(none)"})
MAX_STEPS = 6
MAX_VALUES = 12
MIN_WORDS = 4  # "Open Notes" needs no reading
PLAN_TIMEOUT_S = 20.0  # optional help: never hold a run up for long


def needs_plan(goal: str) -> bool:
    """Only goals with several parts benefit from a list of steps."""
    return len(goal.split()) >= 5 and bool(_MULTI_PART.search(goal))


def parse_plan(text: str) -> list[str]:
    steps: list[str] = []
    for line in text.splitlines():
        match = _STEP.match(line)
        if not match:
            continue
        step = re.sub(r"\s+", " ", match.group("text")).strip(" *_")
        if step and len(step) <= 160 and step.lower() not in (s.lower() for s in steps):
            steps.append(step)
        if len(steps) == MAX_STEPS:
            break
    return steps if len(steps) >= 2 else []


def _slot_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_")[:30]


def parse_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _VALUE.match(line.replace("**", ""))
        if not match:
            continue
        name = _slot_name(match.group("name"))
        value = match.group("value").strip().strip("\"'“”`").strip()
        if not name or name in ("steps", "values") or value.lower() in _EMPTY or SECRET_NAME.search(name):
            continue  # passwords come from the Keychain (jevosx login), never from a model
        if name.startswith("website") or name in ("url", "address", "site"):
            value = re.sub(r"^https?://", "", value).rstrip("/")
        if re.search(r"price|amount|cost|quantity", name):
            number = re.search(r"\d+(?:[.,]\d+)?", value)
            value = number.group(0) if number else value  # "40 euros" → "40": price fields take numbers
        if value and len(value) <= 2000 and name not in values:
            values[name] = value
        if len(values) == MAX_VALUES:
            break
    return values


MAX_QUESTIONS = 3


def parse_reading(text: str) -> tuple[list[str], dict[str, str]]:
    """Split the model's answer into steps and values. Tolerates missing headers and markdown decoration."""
    steps, values, _ = parse_sections(text)
    return steps, values


def parse_sections(text: str) -> tuple[list[str], dict[str, str], list[tuple[str, str]]]:
    """Steps, values and the questions for the person (name, question)."""
    lines = [line.replace("**", "").replace("#", "") for line in text.splitlines()]
    parts: dict[str, list[str]] = {"steps": [], "values": [], "ask": []}
    section = "steps"
    for line in lines:
        head = line.strip().upper()
        found = next((name for name in parts if head == name.upper() or head.startswith(name.upper() + ":")), None)
        if found is not None:
            section = found
            rest = line.split(":", 1)[1] if ":" in line else ""
            if rest.strip() and found != "steps":
                parts[found].append(rest)
            continue
        parts[section].append(line)
    values = parse_values("\n".join(parts["values"]))
    questions = [
        (name, question if question.endswith("?") else question + "?")
        for name, question in parse_values("\n".join(parts["ask"])).items()
        if len(question) >= 8 and name not in values  # the request already answers it
    ][:MAX_QUESTIONS]
    return parse_plan("\n".join(parts["steps"])), values, questions


@dataclass
class GoalReading:
    steps: list[str] = field(default_factory=list)
    values: dict[str, str] = field(default_factory=dict)
    ms: float = 0.0
    questions: list[tuple[str, str]] = field(default_factory=list)  # (name, question) for the person, before starting

    def summary(self) -> str:
        parts = []
        if self.steps:
            parts.append(" · ".join(f"{i}. {step}" for i, step in enumerate(self.steps, start=1)))
        if self.values:
            shown = [
                f"{name} ({len(value)} characters)" if len(value) > 60 else f"{name} “{value}”"
                for name, value in self.values.items()
            ]
            parts.append("to type: " + ", ".join(shown))
        return " · ".join(parts)


class Planner:
    """Reads a goal with the writer's model. Never raises: no model, a failure or an unusable answer → nothing."""

    def __init__(self, writer: TextWriter):
        self.writer = writer

    def read(self, goal: str) -> GoalReading:
        if len(goal.split()) < MIN_WORDS:
            return GoalReading()
        started = time.perf_counter()
        try:
            text = self.writer.generate(
                READER_INSTRUCTIONS, f"Request: {goal}", max_tokens=400, temperature=0.2, timeout_s=PLAN_TIMEOUT_S
            )
        except (TextUnavailableError, JevOSXError, OSError) as exc:
            log.info("goal not read: %s", exc)
            return GoalReading()
        steps, values, questions = parse_sections(text)
        return GoalReading(steps, values, round((time.perf_counter() - started) * 1000, 1), questions)

    def plan(self, goal: str) -> list[str]:
        """Just the ordered steps (multi-part goals only)."""
        return self.read(goal).steps if needs_plan(goal) else []


_ADDRESS = re.compile(r"(?:https?://)?(?:www\.)?(?P<site>(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/\S*)?)", re.IGNORECASE)


def _address(value: str) -> str | None:
    """ "https://www.facebook.com/marketplace/" → "facebook.com/marketplace"; None when it is not an address."""
    match = _ADDRESS.fullmatch(value.strip())
    return match.group("site").lower().rstrip("/") if match else None


def merge_slots(model: dict[str, str], patterns: dict[str, str]) -> dict[str, str]:
    """The model's values first; pattern-based slots only add what the model did not already cover.

    Near-identical text ("population of Malta" / "the population of Malta") and two addresses on the same site where
    one leads further ("facebook.com/marketplace" / "facebook.com/marketplace/create/item") count as covered: the
    more specific address is kept, under the model's name. Offering both would split Jev's choice between them and
    leave neither sure enough to type."""
    merged = dict(model)
    for name, value in patterns.items():
        if name in merged or any(_same_text(value, other) for other in merged.values()):
            continue
        address = _address(value)
        related = [other for other, text in merged.items() if address and _same_site(address, _address(text))]
        if not related:
            merged[name] = value
        for other in related:
            if address and address.startswith(f"{_address(merged[other])}/"):
                merged[other] = value  # the pattern's address goes further on the same site
    return merged


_ARTICLES = frozenset({"a", "an", "the"})


def _same_text(a: str, b: str) -> bool:
    def words(text: str) -> list[str]:
        return [w for w in re.findall(r"\w+", text.lower()) if w not in _ARTICLES]

    return words(a) == words(b)


def _same_site(a: str, b: str | None) -> bool:
    return b is not None and (a == b or a.startswith(b + "/") or b.startswith(a + "/"))
