"""Demo-mode writer: deterministic canned text so the console can show "write a poem" without any model."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_ABOUT = re.compile(r"\babout\s+(?P<topic>.+?)(?=\s+(?:in|on|into|to|and|then|using|with)\b|[,.;!?\"“]|$)", re.I)


def _topic(goal: str) -> str:
    match = _ABOUT.search(goal)
    return match.group("topic").strip() if match else "a quiet afternoon"


class SimulatedWriter:
    model = "writer-demo (simulated)"

    def write(self, context: Mapping[str, Any]) -> str:
        goal = str(context.get("goal", ""))
        topic = _topic(goal)
        if re.search(r"\bhaiku\b", goal, re.I):
            return f"{topic.capitalize()} arrives,\nsoft light across the keyboard,\nthe cursor blinks on"
        if re.search(r"\b(poem|poetry|sonnet|verse)\b", goal, re.I):
            return (
                f"A poem about {topic}\n\n"
                f"Slow light falls on {topic},\n"
                "the window hums a patient tune,\n"
                "and every line I meant to write\n"
                "arrives as softly as the moon."
            )
        if re.search(r"\b(reply|email|message|letter)\b", goal, re.I):
            return f"Hi,\n\nThanks for your note about {topic}. I'll take a look and get back to you today.\n\nBest,"
        return f"Notes on {topic}: a short draft written by the simulated writer."

    def generate(
        self,
        instructions: str,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout_s: float | None = None,
    ) -> str:
        """Stands in for the goal reader: numbered steps, then the values the pattern rules find (plus a description
        when the request asks for new text), in the same format the real model is asked for."""
        from ..router.text import slots_from_goal

        request = next((line[9:] for line in prompt.splitlines() if line.startswith("Request: ")), prompt)
        parts = [p.strip() for p in re.split(r",\s*|\s+(?:and then|then|and)\s+", request) if p.strip()]
        lines = ["STEPS:", *(f"{i}. {part[0].upper()}{part[1:]}" for i, part in enumerate(parts, start=1)), "VALUES:"]
        slots = slots_from_goal(request)
        values = {"website" if name.startswith("url") else name: value for name, value in slots.items()}
        if re.search(r"\b(generic|some|short)\s+text\b|\bdescription\b", request, re.I) and "title" in values:
            values["description"] = (
                f"{values['title']} in good working order. Works as it should and is ready to use. "
                "Pick-up or delivery can be arranged; message me with any questions."
            )
        lines += [f"{name}: {value}" for name, value in values.items()] or ["none"]
        return "\n".join(lines)

    def close(self) -> None:
        return None
