"""Jev router: observation + goal + history + memory hints → one Jev request → one validated Decision.

The operation question and every target question are answered in the same round trip ("speculative heads").
Only the head selected by the operation is validated and allowed to execute; a malformed answer on an unused head
cannot cause an action.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import RouterContractError
from ..executor.keys import KeyBinding
from ..types import CLICK, TYPE_TEXT, Observation, is_console_window
from .client import ChoiceAnswer, JevClient, JevResponse, choice_question
from .prompts import MEMORY, NEXT_ACTION, TARGET, TEXT_SLOT
from .space import HEADS, ActionSpace, Target
from .text import GENERATE, TextSource

TEXT_SLOT_HEAD = "text_slot"
CONSOLE_NOTE = (
    "The focused window is the JevOSX console that sends you commands; never act inside it. For web tasks open a "
    "new browser window (PRESS_KEY CMD_N or the New Window menu command); otherwise OPEN_APP the app the goal needs."
)
ELEMENT_HEADS = frozenset({HEADS[CLICK], HEADS[TYPE_TEXT]})


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
        return out


def _certain(choice: str) -> ChoiceAnswer:
    return ChoiceAnswer(choice, {choice: 1.0}, 1.0)


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

    def space(self, obs: Observation, text_source: TextSource, goal: str = "") -> ActionSpace:
        return ActionSpace.build(
            obs,
            keys=self.keys,
            text_available=text_source.available,
            max_choices=self.max_choices,
            goal=goal,
            offer_installed_apps=self.offer_installed_apps,
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
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state = build_state(
            obs, history=history, hints=hints, text_source=text_source, include_disabled=self.include_disabled
        )
        dropped = compact_state(state, self.max_state_bytes)
        if dropped:
            space.drop_elements(dropped)
        rules = NEXT_ACTION + ("\n" + MEMORY if hints else "")
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
        validate_request(state, questions, space)
        return state, questions

    def decode(self, response: JevResponse, space: ActionSpace, text_source: TextSource) -> Decision:
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
                decision.target_answer = response.choice(head, targets)
            decision.target = targets[decision.target_answer.choice]
        if decision.operation == TYPE_TEXT:
            options = text_source.options()
            if len(options) == 1:
                decision.text_option = next(iter(options))
            elif len(options) > 1:
                decision.text_answer = response.choice(TEXT_SLOT_HEAD, options)
                decision.text_option = decision.text_answer.choice
        return decision

    def decide(
        self,
        goal: str,
        obs: Observation,
        space: ActionSpace,
        *,
        text_source: TextSource,
        history: Sequence[Mapping[str, Any]] = (),
        hints: Sequence[Mapping[str, Any]] = (),
    ) -> Decision:
        state, questions = self.build_request(goal, obs, space, text_source=text_source, history=history, hints=hints)
        response = self.client.evaluate(state, questions)
        return self.decode(response, space, text_source)


def build_state(
    obs: Observation,
    *,
    history: Sequence[Mapping[str, Any]] = (),
    hints: Sequence[Mapping[str, Any]] = (),
    text_source: TextSource | None = None,
    include_disabled: bool = False,
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
    if text_source is not None and (text_source.slots or text_source.generate):
        slots = {name: slot.preview for name, slot in text_source.slots.items()}
        if text_source.generate:
            slots[GENERATE] = "a writer composes the new text the goal asks for, for the field TYPE_TEXT chooses"
        state["text_slots"] = slots
    return state


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
    if element.ops:
        out["ops"] = list(element.ops)
    return out
