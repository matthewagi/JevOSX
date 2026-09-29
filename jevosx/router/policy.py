"""Jev router: observation + goal + history + memory hints → one Jev request → one validated Decision.

The operation question and every target question are answered in the same round trip ("speculative heads").
Only the head selected by the operation is validated and allowed to execute; a malformed answer on an unused head
cannot cause an action.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast
from urllib.parse import urlsplit

from ..errors import JevResponseError, RouterContractError
from ..executor.keys import KeyBinding
from ..types import CLICK, TYPE_TEXT, Observation, clean_text, is_console_window
from .client import ChoiceAnswer, JevClient, JevResponse, choice_question
from .prompts import MEMORY, NEXT_ACTION, PLAN, TARGET, TEXT_FOR_FIELD, TEXT_SLOT
from .space import HEADS, ActionSpace, Target
from .text import GENERATE, TextSource

T = TypeVar("T")
TEXT_SLOT_HEAD = "text_slot"
CONSOLE_NOTE = (
    "The focused window is the JevOSX console that sends you commands; never act inside it. For web tasks open a "
    "new browser window (PRESS_KEY CMD_N or the New Window menu command); otherwise OPEN_APP the app the goal needs."
)
ELEMENT_HEADS = frozenset({HEADS[CLICK], HEADS[TYPE_TEXT]})
# The text question is answered before Jev knows which field it is for. Below this, once the field is chosen, Jev is
# asked again with the field in view.
TEXT_FOLLOW_UP_BELOW = 0.8


@dataclass
class Decision:
    operation: str
    operation_answer: ChoiceAnswer
    target: Target | None = None
    target_answer: ChoiceAnswer | None = None
    text_option: str | None = None
    text_answer: ChoiceAnswer | None = None
    model: str = ""
    latency_ms: float = 0.0
    usage: Mapping[str, Any] | None = None
    text_candidates: list[str] = field(default_factory=list)  # the text options that fit the chosen field
    text_follow_up: bool = False  # the text was chosen by a second question that showed Jev the field

    @property
    def confidence(self) -> float:
        return self.operation_answer.confidence

    @property
    def gate_confidence(self) -> float:
        """The weakest confidence among the answers that would drive execution (operation, target, text slot)."""
        values = [self.operation_answer.confidence]
        values += [a.confidence for a in (self.target_answer, self.text_answer) if a is not None]
        return min(values)

    @property
    def probability(self) -> float:
        """Joint probability of the executed (operation, target) pair."""
        p = self.operation_answer.probability
        if self.target_answer is not None:
            p *= self.target_answer.probability
        return p

    def describe(self) -> str:
        return f"{self.operation} {self.target.describe()}".strip() if self.target else self.operation

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "operation": self.operation,
            "target": self.target.describe() if self.target else None,
            "p_operation": round(self.operation_answer.probability, 4),
            "confidence": round(self.confidence, 4),
            "gate_confidence": round(self.gate_confidence, 4),
            "top_operations": [(k, round(v, 3)) for k, v in self.operation_answer.ranked(3)],
            "latency_ms": self.latency_ms,
            "model": self.model,
        }
        if self.target_answer is not None:
            out["p_target"] = round(self.target_answer.probability, 4)
            out["top_targets"] = [(k, round(v, 3)) for k, v in self.target_answer.ranked(3)]
        if self.text_option is not None:
            out["text_option"] = self.text_option
        if self.text_answer is not None:
            out["text_confidence"] = round(self.text_answer.confidence, 4)
            out["top_texts"] = [(k, round(v, 3)) for k, v in self.text_answer.ranked(3)]
        if self.text_follow_up:
            out["text_follow_up"] = True
        return out


def _certain(choice: str) -> ChoiceAnswer:
    return ChoiceAnswer(choice, {choice: 1.0}, 1.0)


def restrict(answer: ChoiceAnswer, allowed: Sequence[str]) -> ChoiceAnswer:
    """Jev's distribution over only the options that fit (renormalized): ruling out options that cannot be right
    is not doubt about the rest. The confidence keeps Jev's own calibration (its confidence relative to its
    probability for the top option)."""
    if set(allowed) == set(answer.probabilities):
        return answer
    mass = sum(answer.probabilities.get(option, 0.0) for option in allowed)
    if mass <= 1e-9:
        return ChoiceAnswer(allowed[0], {option: 1 / len(allowed) for option in allowed}, 0.0)
    probabilities = {option: answer.probabilities.get(option, 0.0) / mass for option in allowed}
    choice = max(probabilities, key=lambda option: probabilities[option])
    return ChoiceAnswer(choice, probabilities, min(1.0, _calibration(answer) * probabilities[choice]))


def _calibration(answer: ChoiceAnswer) -> float:
    return answer.confidence / answer.probability if answer.probability > 1e-9 else 1.0


def same_destination(answer: ChoiceAnswer, targets: Mapping[str, Target]) -> ChoiceAnswer:
    """Search results list the same page several times (title, breadcrumb, sitelink). When Jev spreads its
    probability over links that all lead to the chosen link's page, the outcome is not in doubt: the confidence
    becomes their combined probability (never lower than Jev's own)."""
    chosen = targets[answer.choice].element
    url = link_target(chosen.url) if chosen is not None else None
    if not url:
        return answer
    same = [tid for tid, t in targets.items() if t.element is not None and link_target(t.element.url) == url]
    if len(same) < 2:
        return answer
    combined = min(1.0, sum(answer.probabilities.get(tid, 0.0) for tid in same))
    return ChoiceAnswer(answer.choice, answer.probabilities, max(answer.confidence, combined))


class JevRouter:
    def __init__(
        self,
        client: JevClient,
        *,
        keys: Mapping[str, KeyBinding],
        max_choices: int = 200,
        max_state_bytes: int = 48_000,
        include_disabled: bool = False,
        offer_installed_apps: str = "mentioned",
    ):
        self.client = client
        self.keys = keys
        self.max_choices = max_choices
        self.max_state_bytes = max_state_bytes
        self.include_disabled = include_disabled
        self.offer_installed_apps = offer_installed_apps

    @classmethod
    def from_settings(cls, client: JevClient, settings: Any, keys: Mapping[str, KeyBinding]) -> JevRouter:
        return cls(
            client,
            keys=keys,
            max_choices=settings.max_choices,
            max_state_bytes=settings.max_state_bytes,
            include_disabled=settings.include_disabled,
            offer_installed_apps=settings.offer_installed_apps,
        )

    def space(self, obs: Observation, text_source: TextSource, goal: str = "", *, handoff: bool = False) -> ActionSpace:
        return ActionSpace.build(
            obs,
            keys=self.keys,
            text_available=text_source.available,
            max_choices=self.max_choices,
            goal=goal,
            offer_installed_apps=self.offer_installed_apps,
            handoff=handoff,
        )

    def build_request(
        self,
        goal: str,
        obs: Observation,
        space: ActionSpace,
        *,
        text_source: TextSource,
        history: Sequence[Mapping[str, Any]] = (),
        hints: Sequence[Mapping[str, Any]] = (),
        plan: Sequence[str] = (),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state = build_state(
            obs,
            history=history,
            hints=hints,
            text_source=text_source,
            include_disabled=self.include_disabled,
            plan=plan,
        )
        dropped = compact_state(state, self.max_state_bytes)
        if dropped:
            space.drop_elements(dropped)
        rules = NEXT_ACTION + ("\n" + MEMORY if hints else "") + ("\n" + PLAN if plan and "plan" in state else "")
        questions: dict[str, Any] = {
            "operation": choice_question(space.operations, {"goal": goal, "rules": rules}),
        }
        seen_heads: set[str] = set()
        for operation in space.operations:
            head = space.head_for(operation)
            if head is None or head in seen_heads:
                continue
            seen_heads.add(head)
            targets = space.heads[head]
            if len(targets) < 2:
                continue  # a single compatible target needs no question
            ops = [op for op, h in HEADS.items() if h == head]
            questions[head] = choice_question(
                {t.id: t.criterion for t in targets.values()},
                {"goal": goal, "operation": " or ".join(ops), "rules": [NEXT_ACTION, TARGET]},
            )
        if TYPE_TEXT in space.operations:
            options = text_source.options()
            if len(options) >= 2:
                questions[TEXT_SLOT_HEAD] = choice_question(options, {"goal": goal, "rules": TEXT_SLOT})
        redact(questions, text_source.sensitive_values())
        validate_request(state, questions, space)
        return state, questions

    def decode(
        self, response: JevResponse, space: ActionSpace, text_source: TextSource, obs: Observation | None = None
    ) -> Decision:
        operation_answer = response.choice("operation", space.operations)
        decision = Decision(
            operation=operation_answer.choice,
            operation_answer=operation_answer,
            model=response.model,
            latency_ms=response.latency_ms,
            usage=response.usage,
        )
        targets = space.targets_for(decision.operation)
        if targets:
            head = space.head_for(decision.operation)
            assert head is not None
            if len(targets) == 1:
                decision.target_answer = _certain(next(iter(targets)))
            else:
                decision.target_answer = same_destination(response.choice(head, targets), targets)
            decision.target = targets[decision.target_answer.choice]
            self._same_field(decision, response, space)
        if decision.operation == TYPE_TEXT:
            options = text_source.options()
            field_element = decision.target.element if decision.target is not None else None
            fitting = text_source.compatible(field_element, obs.app) if obs is not None else list(options)
            decision.text_candidates = fitting
            if len(options) == 1:
                decision.text_option = next(iter(options))
            elif len(fitting) == 1:
                decision.text_answer = _certain(fitting[0])  # the only text that fits this field
                decision.text_option = fitting[0]
            elif len(options) > 1:
                decision.text_answer = restrict(response.choice(TEXT_SLOT_HEAD, options), fitting)
                decision.text_option = decision.text_answer.choice
        return decision

    @staticmethod
    def _same_field(decision: Decision, response: JevResponse, space: ActionSpace) -> None:
        """Clicking a text field and typing into it are one intent (TYPE_TEXT focuses the field itself). When Jev
        splits its operation probability between the two for the same field, the combined probability counts."""
        element = decision.target.element if decision.target is not None else None
        if decision.operation not in (CLICK, TYPE_TEXT) or element is None or element.kind != "text_input":
            return
        other = TYPE_TEXT if decision.operation == CLICK else CLICK
        targets = space.targets_for(other)
        head = space.head_for(other)
        if not targets or head is None or other not in decision.operation_answer.probabilities:
            return
        try:
            chosen = next(iter(targets)) if len(targets) == 1 else response.choice(head, targets).choice
        except JevResponseError:
            return
        other_element = targets[chosen].element
        if other_element is None or other_element.index != element.index:
            return
        answer = decision.operation_answer
        combined = min(1.0, _calibration(answer) * (answer.probability + answer.probabilities[other]))
        decision.operation_answer = ChoiceAnswer(answer.choice, answer.probabilities, max(answer.confidence, combined))

    def decide(
        self,
        goal: str,
        obs: Observation,
        space: ActionSpace,
        *,
        text_source: TextSource,
        history: Sequence[Mapping[str, Any]] = (),
        hints: Sequence[Mapping[str, Any]] = (),
        plan: Sequence[str] = (),
    ) -> Decision:
        state, questions = self.build_request(
            goal, obs, space, text_source=text_source, history=history, hints=hints, plan=plan
        )
        response = self.client.evaluate(state, questions)
        decision = self.decode(response, space, text_source, obs)
        answer = decision.text_answer
        if (
            decision.operation == TYPE_TEXT
            and answer is not None
            and answer.confidence < TEXT_FOLLOW_UP_BELOW
            and len(decision.text_candidates) >= 2
        ):
            self._ask_text_for_field(goal, state, decision, text_source)
        return decision

    def _ask_text_for_field(
        self, goal: str, state: Mapping[str, Any], decision: Decision, text_source: TextSource
    ) -> None:
        """Second question, only when the first text answer was unsure: which text for THIS field, shown to Jev."""
        assert decision.target is not None and decision.target.element is not None
        options = text_source.options()
        criteria = {option: options[option] for option in decision.text_candidates}
        where = element_state(decision.target.element)
        question = choice_question(criteria, {"goal": goal, "field": where, "rules": TEXT_FOR_FIELD})
        questions = {TEXT_SLOT_HEAD: question}
        redact(questions, text_source.sensitive_values())
        response = self.client.evaluate(state, questions)
        decision.text_answer = response.choice(TEXT_SLOT_HEAD, criteria)
        decision.text_option = decision.text_answer.choice
        decision.text_follow_up = True
        decision.latency_ms = round(decision.latency_ms + response.latency_ms, 1)


def build_state(
    obs: Observation,
    *,
    history: Sequence[Mapping[str, Any]] = (),
    hints: Sequence[Mapping[str, Any]] = (),
    text_source: TextSource | None = None,
    include_disabled: bool = False,
    plan: Sequence[str] = (),
) -> dict[str, Any]:
    """The JSON state Jev sees. Labels, roles, values and states only: no coordinates, handles or screenshots.

    Only interactive elements (at least one supported operation) are listed unless `include_disabled` is set;
    hidden, off-screen and zero-size elements never reach this point (see TreeWalker)."""
    focused = obs.focused_element
    desktop: dict[str, Any] = {
        "frontmost_app": obs.app.name,
        "bundle_id": obs.app.bundle_id,
        "window": obs.window.title if obs.window else None,
        "focused_element": focused.describe() if focused else None,
    }
    page = page_summary(obs.page_url)
    if page:
        desktop["page"] = page
    others = [w.title for w in obs.windows if not w.focused][:8]
    if others:
        desktop["other_windows"] = others
    if obs.stats.get("truncated"):
        desktop["note"] = "element list truncated; scroll or use menu commands to reach more"
    console = obs.window is not None and is_console_window(obs.window.title)
    if console:
        desktop["note"] = CONSOLE_NOTE
    state: dict[str, Any] = {
        "desktop": desktop,
        "elements": [] if console else [element_state(e) for e in obs.elements if e.ops or include_disabled],
        "visible_text": "" if console else obs.text,
        "recent_actions": list(history),
    }
    if hints:
        state["memory_hints"] = list(hints)
    if plan and not console:
        state["plan"] = [f"{i}. {step}" for i, step in enumerate(plan, start=1)]
    if text_source is not None and (text_source.slots or text_source.generate):
        slots = {name: slot.preview for name, slot in text_source.slots.items()}
        if text_source.generate:
            slots[GENERATE] = "a writer composes the new text the goal asks for, for the field TYPE_TEXT chooses"
        state["text_slots"] = slots
    if text_source is not None:
        redact(state, text_source.sensitive_values())
    return state


def page_summary(url: str | None) -> str | None:
    """Scheme, host and path of the page on screen; query strings and fragments (tokens, ids) are dropped."""
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https", "file") or not (parts.netloc or parts.path):
        return None
    return clean_text(f"{parts.scheme}://{parts.netloc}{parts.path}", 120)


def redact(payload: T, values: Sequence[str]) -> T:
    """Mask saved-login usernames in every string of a state or question payload (in place for dicts/lists)."""
    if not values:
        return payload
    if isinstance(payload, str):
        text: str = payload
        for value in values:
            text = text.replace(value, "•••")
        return cast(T, text)
    if isinstance(payload, dict):
        for key, item in payload.items():
            payload[key] = redact(item, values)
    elif isinstance(payload, list):
        payload[:] = [redact(item, values) for item in payload]
    return payload


def state_size(state: Mapping[str, Any]) -> int:
    return len(json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode())


def compact_state(state: dict[str, Any], max_bytes: int) -> set[int]:
    """Shrink the state in place until it fits `max_bytes`. Returns element indices that had to be dropped.

    Order: visible text → older history → container names → trailing elements (never the focused one)."""
    dropped: set[int] = set()
    if state_size(state) <= max_bytes:
        return dropped
    text = state.get("visible_text") or ""
    while text and state_size(state) > max_bytes:
        text = text[: len(text) // 2] if len(text) > 200 else ""
        state["visible_text"] = text + ("…" if text else "")
    if state_size(state) > max_bytes and len(state.get("recent_actions", [])) > 4:
        state["recent_actions"] = state["recent_actions"][-4:]
    if state_size(state) > max_bytes:
        for item in state["elements"]:
            item.pop("in", None)
    elements = state["elements"]
    if state_size(state) > max_bytes:
        state["desktop"]["note"] = "element list truncated to fit the context budget; scroll or use menu commands"
    while elements and state_size(state) > max_bytes:
        victim = next((e for e in reversed(elements) if "focused" not in e.get("state", ())), None)
        if victim is None:
            break
        elements.remove(victim)
        dropped.add(victim["index"])
    return dropped


def validate_request(state: Mapping[str, Any], questions: Mapping[str, Any], space: ActionSpace) -> None:
    """Enforce the strict-choice contract before anything is sent.

    Every question is a discrete one-of-N choice; element target ids are exactly `index` values present in the
    element table; every offered id resolves to an observed target."""
    indices = {str(e["index"]) for e in state.get("elements", [])}
    for name, question in questions.items():
        if question.get("type") != "choice" or not isinstance(question.get("criteria"), Mapping):
            raise RouterContractError(f"question {name!r} is not a discrete choice question")
        ids = set(question["criteria"])
        if name == "operation":
            if ids != set(space.operations):
                raise RouterContractError("operation options differ from the action space")
            continue
        if name == TEXT_SLOT_HEAD:
            continue
        if name not in space.heads or ids != set(space.heads[name]):
            raise RouterContractError(f"{name!r} offers ids that are not observed targets")
        if name in ELEMENT_HEADS and not ids <= indices:
            raise RouterContractError(f"{name!r} offers element ids missing from the element table: {ids - indices}")
    for head in ELEMENT_HEADS & set(space.heads):
        missing = set(space.heads[head]) - indices
        if missing:
            raise RouterContractError(f"{head!r} targets are missing from the element table: {sorted(missing)}")


def element_state(element: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"index": element.index, "role": element.role_name, "label": element.label}
    if element.value and not element.secure:
        out["value"] = element.value
    if element.container:
        out["in"] = element.container
    states = element.states()
    if states:
        out["state"] = states
    link = link_target(getattr(element, "url", None))
    if link:
        out["to"] = link
    if element.ops:
        out["ops"] = list(element.ops)
    return out


def link_target(url: str | None) -> str | None:
    """Host and path of a link, no scheme, query or fragment: "en.wikipedia.org/wiki/Malta"."""
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return clean_text(f"{parts.netloc}{parts.path}".rstrip("/"), 80)
