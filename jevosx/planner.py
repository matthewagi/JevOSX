"""On-device plan for multi-part goals: the writer's language model splits the goal into ordered steps, once per run.

"Write a poem about autumn, save it as poem.rtf, then open it in Pages" is three tasks in one sentence. A System-One
model decides fast from the current screen but has to infer from history which part is already done. The plan gives
it the order. It is context only: a suggested outline from a small model, never a command. Every action is still a
Jev choice among observed ids, and a failed or empty plan simply means no plan.
"""

from __future__ import annotations

import logging
import re

from .errors import JevOSXError, TextUnavailableError
from .writer.base import TextWriter

log = logging.getLogger("jevosx.planner")

PLANNER_INSTRUCTIONS = """Split the user's request for their Mac into the concrete steps a person would take, in order.
Write one short line per step, numbered 1., 2., 3. Use at most 6 steps. Name the app when it is clear.
Only restate what the request asks for: do not add extra tasks, explanations, warnings or questions."""

_MULTI_PART = re.compile(r",|;|\bthen\b|\band\b|\bafter(?:wards)?\b|\bnext\b|\bfinally\b", re.IGNORECASE)
_STEP = re.compile(r"^\s*(?:\d{1,2}\s*[.)]|[-•*])\s*(?P<text>.+?)\s*$")
MAX_STEPS = 6
PLAN_TIMEOUT_S = 20.0  # a plan is optional: never hold a run up for long


def needs_plan(goal: str) -> bool:
    """Only goals with several parts benefit; "Open Notes" does not need a plan."""
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


class Planner:
    def __init__(self, writer: TextWriter):
        self.writer = writer

    def plan(self, goal: str) -> list[str]:
        """Ordered steps for a multi-part goal, or [] (single-part goal, or the model failed). Never raises."""
        if not needs_plan(goal):
            return []
        try:
            text = self.writer.generate(
                PLANNER_INSTRUCTIONS, f"Request: {goal}", max_tokens=200, temperature=0.2, timeout_s=PLAN_TIMEOUT_S
            )
        except (TextUnavailableError, JevOSXError, OSError) as exc:
            log.info("no plan: %s", exc)
            return []
        return parse_plan(text)
