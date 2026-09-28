"""Similarity retrieval over past trajectories → hints Jev can use now.

For the current goal and screen, find finished episodes with similar goals, then steps taken from similar screens.
A remembered step only becomes a hint if its target exists in the current action space, so hints are always
actionable and always point at a real, currently offered id. Steps that changed the UI in successful runs become
"worked" hints; steps that changed nothing become "no_effect" hints; the final screen of a successful run becomes a
"finished" hint (evidence for DONE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import MemorySettings
from ..router.space import ActionSpace
from ..types import DONE, Observation
from .embedding import cosine_many
from .store import MemoryStore

STATUS_WEIGHT = {"success": 1.0, "done": 0.6}
POSITIVE_OUTCOMES = frozenset({"changed"})
NEGATIVE_OUTCOMES = frozenset({"unchanged", "failed"})
FINISHED_STATE_SIMILARITY = 0.8


def state_summary(obs: Observation, limit: int = 80) -> str:
    """What a screen 'is', independent of volatile values: app, window, and the roles/labels on it."""
    parts = [obs.app.name, obs.window.title if obs.window else ""]
    parts.extend(f"{e.role_name} {e.label}" for e in obs.elements[:limit])
    return " ; ".join(p for p in parts if p)


@dataclass
class Hint:
    kind: str  # worked | no_effect | finished
    operation: str
    target_id: str | None
    target: str | None
    score: float
    episodes: set[int] = field(default_factory=set)

    @property
    def support(self) -> int:
        return len(self.episodes)

    def to_state(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "operation": self.operation, "similarity": round(self.score, 2),
                               "past_runs": self.support}  # fmt: skip
        if self.target_id is not None:
            out["target_id"] = self.target_id
            out["target"] = self.target
        return out

    def note(self) -> str:
        runs = f"{self.support} similar past run" + ("s" if self.support != 1 else "")
        if self.kind == "worked":
            return f"used successfully in {runs} (similarity {self.score:.2f})"
        return f"had no visible effect in {runs} (similarity {self.score:.2f})"


class MemoryRetriever:
    def __init__(self, store: MemoryStore, settings: MemorySettings | None = None):
        self.store = store
        self.settings = settings or MemorySettings()

    def hints(
        self, goal: str, obs: Observation, space: ActionSpace, *, exclude_episode: int | None = None
    ) -> list[Hint]:
        s = self.settings
        index = self.store.goal_index()
        if not len(index):
            return []
        embedder = self.store.embedder
        goal_sims = cosine_many(embedder.embed(goal), index.matrix)
        order = np.argsort(-goal_sims)
        episode_sims: dict[int, tuple[float, str]] = {}
        for i in order[: s.top_episodes * 3]:
            similarity = float(goal_sims[i])
            if similarity < s.min_goal_similarity or len(episode_sims) >= s.top_episodes:
                break
            episode_id = int(index.ids[i])
            if episode_id != exclude_episode:
                episode_sims[episode_id] = (similarity, index.statuses[i])
        if not episode_sims:
            return []

        state_vec = embedder.embed(state_summary(obs))
        merged: dict[tuple[str, str, str | None], Hint] = {}
        for step in self.store.steps_for(episode_sims):
            if step.state_vec is None:
                continue
            state_sim = float(step.state_vec @ state_vec)
            if state_sim < s.min_state_similarity:
                continue
            goal_sim, status = episode_sims[step.episode_id]
            score = 0.5 * goal_sim + 0.5 * state_sim + (0.05 if step.app and step.app == obs.app.bundle_id else 0.0)
            weight = STATUS_WEIGHT.get(status)
            if step.operation == DONE:
                if weight is None or step.outcome != "final" or state_sim < FINISHED_STATE_SIMILARITY:
                    continue
                kind, target_id, target_text = "finished", None, None
                score *= weight
            else:
                if not step.memory_key:
                    continue
                target = space.resolve(step.operation, step.memory_key)
                if target is None:
                    continue
                if step.outcome in POSITIVE_OUTCOMES and weight is not None:
                    kind = "worked"
                    score *= weight
                elif step.outcome in NEGATIVE_OUTCOMES:
                    kind = "no_effect"
                else:
                    continue
                target_id, target_text = target.id, target.describe()
            key = (kind, step.operation, target_id)
            hint = merged.get(key)
            if hint is None:
                merged[key] = Hint(kind, step.operation, target_id, target_text, score, {step.episode_id})
            else:
                hint.score = max(hint.score, score)
                hint.episodes.add(step.episode_id)
        ranked = sorted(merged.values(), key=lambda h: h.score + 0.03 * (h.support - 1), reverse=True)
        return ranked[: s.max_hints]
