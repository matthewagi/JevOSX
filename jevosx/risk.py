"""How much a wrong step could cost, and so how sure Jev must be before the step runs without asking.

One floor for every step made the agent ask about opening a browser window as often as about publishing a listing.
Steps are sorted into three tiers instead:

- safe: undone in a moment, or changes nothing on its own: open or switch apps and windows, a new window or tab,
  scroll, put the cursor in a field, type into a single-line or empty field, navigation keys, save a picture as a
  new file, hand a step to the person.
- routine: clicks on buttons, links and checkboxes, Return, menu commands, replacing text that is already there,
  DONE.
- careful: what the safety policy wants confirmed (Delete, Send, Publish, Pay…) and keys that close or quit, and
  a second sign-in attempt in a run: failed logins can lock an account, so that one always asks the person.

Each tier has its own floor (agent.safe_confidence, agent.routine_confidence, agent.min_confidence), never above
agent.min_confidence, so raising the floor still makes every step more careful. A step that matches one that worked
in a similar earlier run counts as safe (careful ones stay careful): a step you approved once is not asked again.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from .config import AgentSettings
from .executor.safety import SafetyPolicy
from .memory.retriever import Hint
from .types import (
    ASK_USER,
    CLICK,
    CONSOLE_SAFE_MENU,
    DONE,
    FOCUS_WINDOW,
    MENU,
    OPEN_APP,
    PRESS_KEY,
    SAVE_IMAGE,
    SCROLL_DOWN,
    SCROLL_UP,
    TYPE_TEXT,
    Action,
    Observation,
)

SAFE, ROUTINE, CAREFUL = "safe", "routine", "careful"

# Keys that move focus or the view, open a window or tab, or can be undone.
SAFE_KEYS = frozenset(
    {"ESCAPE", "TAB", "SHIFT_TAB", "UP", "DOWN", "LEFT", "RIGHT", "PAGE_DOWN", "PAGE_UP", "CMD_A", "CMD_C", "CMD_F",
     "CMD_L", "CMD_T", "CMD_N", "CMD_Z", "CMD_SHIFT_G"}
)  # fmt: skip
CAREFUL_KEYS = frozenset({"CMD_W", "CMD_Q"})  # close a tab or window, quit: unsaved work can be lost
# Clicking these opens a list, moves the cursor or selects: nothing is changed yet.
SAFE_CLICK_ROLES = frozenset(
    {"AXTextField", "AXTextArea", "AXComboBox", "AXPopUpButton", "AXMenuButton", "AXDisclosureTriangle",
     "AXRadioButton", "AXRow", "AXTab"}
)  # fmt: skip
REMEMBERED_SCORE = 0.6  # how similar an earlier run must be for its step to count as remembered
SIGN_IN = re.compile(r"\b(?:log ?in|sign ?in|log on)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class StepRisk:
    tier: str
    floor: float
    reason: str
    confirm: bool = False  # ask the person whatever Jev's confidence (a repeated sign-in attempt)

    def describe(self) -> str:
        return f"{self.tier} step: {self.reason}"


def floors(settings: AgentSettings) -> dict[str, float]:
    top = settings.min_confidence
    return {SAFE: min(settings.safe_confidence, top), ROUTINE: min(settings.routine_confidence, top), CAREFUL: top}


def assess(
    action: Action,
    obs: Observation,
    *,
    settings: AgentSettings,
    safety: SafetyPolicy,
    hints: Sequence[Hint] = (),
    target_id: str | None = None,
    sign_ins: int = 0,
) -> StepRisk:
    """The tier and floor for a decided step (`action` is its preview: no text resolved yet). `sign_ins` counts the
    sign-in attempts already made in this run with a typed password."""
    if sign_ins and is_sign_in(action, obs):
        why = "another sign-in attempt: if the password is wrong again, the account could be locked"
        return StepRisk(CAREFUL, floors(settings)[CAREFUL], why, confirm=True)
    tier, reason = _tier(action, obs, safety)
    if tier != CAREFUL and _remembered(action.operation, target_id, hints):
        tier, reason = SAFE, "worked in a similar earlier run"
    return StepRisk(tier, floors(settings)[tier], reason)


def _tier(action: Action, obs: Observation, safety: SafetyPolicy) -> tuple[str, str]:
    op = action.operation
    if op in (OPEN_APP, FOCUS_WINDOW):
        return SAFE, "switches apps or windows"
    if op in (SCROLL_UP, SCROLL_DOWN):
        return SAFE, "scrolls"
    if op == ASK_USER:
        return SAFE, "hands the step to you"
    if op == SAVE_IMAGE:
        return SAFE, "saves a copy of a picture as a new file"
    if op == DONE:
        return ROUTINE, "ends the run"
    element = action.element
    # The preview has no text yet: for a password field, check the step as the secret slot that would fill it.
    probe = replace(action, text_is_secret=True) if op == TYPE_TEXT and element and element.secure else action
    verdict = safety.check(probe, obs.app, window_title=obs.window.title if obs.window else None)
    if verdict.verdict != "allow":
        return CAREFUL, verdict.reason
    if op == PRESS_KEY and action.key is not None:
        if action.key.id in CAREFUL_KEYS:
            return CAREFUL, f"{action.key.description}"
        if action.key.id in SAFE_KEYS:
            return SAFE, f"{action.key.description}"
        return ROUTINE, f"{action.key.description}"
    if op == MENU and element is not None:
        if CONSOLE_SAFE_MENU.search(element.label):
            return SAFE, "opens a new window or tab"
        return ROUTINE, "runs a menu command"
    if op == TYPE_TEXT and element is not None:
        if element.kind == "keyboard":
            return ROUTINE, "types at the cursor"
        if element.secure:
            return ROUTINE, "fills a password field"
        if element.role == "AXTextArea" and (element.value or "").strip():
            return ROUTINE, "replaces text that is already there"
        return SAFE, "types into a field"
    if op == CLICK and element is not None:
        if element.kind in ("text_input", "row") or element.role in SAFE_CLICK_ROLES:
            return SAFE, "puts the cursor in a field or opens a list"
        if element.role == "AXLink":
            return ROUTINE, "follows a link"
        return ROUTINE, "clicks a control"
    return ROUTINE, "acts"


def is_sign_in(action: Action, obs: Observation) -> bool:
    """A step that submits a sign-in: a Sign in / Log in control, or Return in a password field."""
    if action.operation in (CLICK, MENU) and action.element is not None:
        return bool(SIGN_IN.search(action.element.label or ""))
    if action.operation == PRESS_KEY and action.key is not None and action.key.id == "RETURN":
        focused = obs.focused_element
        return focused is not None and focused.secure
    return False


def _remembered(operation: str, target_id: str | None, hints: Sequence[Hint]) -> bool:
    for hint in hints:
        if hint.score < REMEMBERED_SCORE or hint.operation != operation:
            continue
        if operation == DONE and hint.kind == "finished":
            return True
        if hint.kind == "worked" and target_id is not None and hint.target_id == target_id:
            return True
    return False
