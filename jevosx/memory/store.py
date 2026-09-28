"""SQLite trajectory store: every episode (goal → steps → outcome) is saved locally and never leaves the Mac.

One file, WAL mode, schema versioned with PRAGMA user_version. Goal and state embeddings are stored next to the
rows (float16) so retrieval needs no recomputation; JSON-lines export/import keeps trajectories portable.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .embedding import HashingEmbedder

SCHEMA_VERSION = 1
FINAL_STATUSES = frozenset({"success", "done", "blocked", "low_confidence", "failed", "max_steps", "aborted", "error"})
LABEL_STATUSES = frozenset({"success", "failed"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY,
    goal        TEXT NOT NULL,
    goal_vec    BLOB NOT NULL,
    app         TEXT,
    status      TEXT NOT NULL DEFAULT 'running',
    steps       INTEGER NOT NULL DEFAULT 0,
    model       TEXT,
    meta        TEXT,
    started_at  REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS steps (
    id            INTEGER PRIMARY KEY,
    episode_id    INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    app           TEXT,
    window        TEXT,
    state_summary TEXT NOT NULL,
    state_vec     BLOB NOT NULL,
    operation     TEXT NOT NULL,
    memory_key    TEXT,
    target_text   TEXT,
    text          TEXT,
    probability   REAL,
    confidence    REAL,
    outcome       TEXT NOT NULL DEFAULT 'pending',
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS steps_episode ON steps(episode_id);
CREATE INDEX IF NOT EXISTS episodes_status ON episodes(status);
"""


@dataclass
class EpisodeRecord:
    id: int
    goal: str
    app: str | None
    status: str
    steps: int
    model: str | None
    started_at: float
    finished_at: float | None


@dataclass
class StepRecord:
    id: int
    episode_id: int
    idx: int
    app: str | None
    window: str | None
    state_summary: str
    operation: str
    memory_key: str | None
    target_text: str | None
    text: str | None
    probability: float | None
    confidence: float | None
    outcome: str
    created_at: float
    state_vec: np.ndarray | None = None


@dataclass
class GoalIndex:
    ids: np.ndarray
    statuses: list[str]
    matrix: np.ndarray

    def __len__(self) -> int:
        return len(self.statuses)


class MemoryStore:
    def __init__(self, path: str | Path, embedder: HashingEmbedder | None = None):
        self.path = Path(path).expanduser()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder()
        self._db = sqlite3.connect(str(self.path), isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.execute("PRAGMA synchronous = NORMAL")
        self._migrate()
        self._goal_index: GoalIndex | None = None

    def _migrate(self) -> None:
        version = self._db.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"memory database {self.path} is from a newer JevOSX (schema {version})")
        self._db.executescript(_SCHEMA)
        dim = self._db.execute("SELECT length(goal_vec) FROM episodes LIMIT 1").fetchone()
        if dim is not None and dim[0] != self.embedder.dim * 2:
            raise RuntimeError(
                f"memory database {self.path} uses {dim[0] // 2}-d embeddings but memory.dim is {self.embedder.dim}"
            )
        self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ---- writes -------------------------------------------------------------------------------------------------
    def begin_episode(
        self, goal: str, *, app: str | None = None, model: str | None = None, meta: dict | None = None
    ) -> int:
        cursor = self._db.execute(
            "INSERT INTO episodes (goal, goal_vec, app, model, meta, started_at) VALUES (?, ?, ?, ?, ?, ?)",
            (goal, self.embedder.to_bytes(self.embedder.embed(goal)), app, model, json.dumps(meta or {}), time.time()),
        )
        self._goal_index = None
        return int(cursor.lastrowid or 0)

    def record_step(
        self,
        episode_id: int,
        *,
        idx: int,
        app: str | None,
        window: str | None,
        state_summary: str,
        operation: str,
        memory_key: str | None,
        target_text: str | None,
        probability: float | None = None,
        confidence: float | None = None,
        outcome: str = "pending",
        text: str | None = None,
    ) -> int:
        vec = self.embedder.to_bytes(self.embedder.embed(state_summary))
        cursor = self._db.execute(
            """INSERT INTO steps (episode_id, idx, app, window, state_summary, state_vec, operation, memory_key,
                                  target_text, text, probability, confidence, outcome, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (episode_id, idx, app, window, state_summary, vec, operation, memory_key, target_text, text,
             probability, confidence, outcome, time.time()),
        )  # fmt: skip
        self._db.execute("UPDATE episodes SET steps = steps + 1 WHERE id = ?", (episode_id,))
        return int(cursor.lastrowid or 0)

    def set_outcome(self, step_id: int, outcome: str) -> None:
        self._db.execute("UPDATE steps SET outcome = ? WHERE id = ?", (outcome, step_id))

    def finish_episode(self, episode_id: int, status: str) -> None:
        if status not in FINAL_STATUSES:
            raise ValueError(f"unknown episode status {status!r}")
        self._db.execute(
            "UPDATE episodes SET status = ?, finished_at = ? WHERE id = ?", (status, time.time(), episode_id)
        )
        self._db.execute(
            "UPDATE steps SET outcome = 'unknown' WHERE episode_id = ? AND outcome = 'pending'", (episode_id,)
        )
        self._goal_index = None

    def label_episode(self, episode_id: int, status: str) -> None:
        """Human (or verifier) feedback: promote a run to 'success' or demote it to 'failed'."""
        if status not in LABEL_STATUSES:
            raise ValueError(f"label must be one of {sorted(LABEL_STATUSES)}")
        self._db.execute("UPDATE episodes SET status = ? WHERE id = ?", (status, episode_id))
        self._goal_index = None

    def delete_episode(self, episode_id: int) -> bool:
        deleted = self._db.execute("DELETE FROM episodes WHERE id = ?", (episode_id,)).rowcount > 0
        self._goal_index = None
        return deleted

    def prune(self, keep: int) -> int:
        """Keep the newest `keep` finished episodes (successful runs are kept preferentially)."""
        rows = self._db.execute(
            "SELECT id FROM episodes WHERE status != 'running' "
            "ORDER BY (status = 'success') DESC, started_at DESC LIMIT -1 OFFSET ?",
            (max(0, keep),),
        ).fetchall()
        with self._transaction():
            for row in rows:
                self._db.execute("DELETE FROM episodes WHERE id = ?", (row["id"],))
        self._goal_index = None
        return len(rows)

    # ---- reads --------------------------------------------------------------------------------------------------
    def goal_index(self) -> GoalIndex:
        if self._goal_index is None:
            rows = self._db.execute("SELECT id, status, goal_vec FROM episodes WHERE status != 'running'").fetchall()
            if rows:
                matrix = np.vstack([self.embedder.from_bytes(r["goal_vec"]) for r in rows])
            else:
                matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
            self._goal_index = GoalIndex(
                ids=np.array([r["id"] for r in rows], dtype=np.int64),
                statuses=[r["status"] for r in rows],
                matrix=matrix,
            )
        return self._goal_index

    def steps_for(self, episode_ids: Iterable[int], *, with_vectors: bool = True) -> list[StepRecord]:
        ids = [int(i) for i in episode_ids]
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self._db.execute(
            f"SELECT * FROM steps WHERE episode_id IN ({marks}) ORDER BY episode_id, idx", ids
        ).fetchall()
        return [self._step(row, with_vectors) for row in rows]

    def episode(self, episode_id: int) -> EpisodeRecord | None:
        row = self._db.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        return self._episode(row) if row else None

    def episodes(self, limit: int = 20) -> list[EpisodeRecord]:
        rows = self._db.execute("SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._episode(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        by_status = {
            r["status"]: r["n"]
            for r in self._db.execute("SELECT status, COUNT(*) AS n FROM episodes GROUP BY status").fetchall()
        }
        steps = self._db.execute("SELECT COUNT(*) FROM steps").fetchone()[0]
        size = self.path.stat().st_size if self.path.exists() else 0
        return {"path": str(self.path), "episodes": sum(by_status.values()), "by_status": by_status, "steps": steps,
                "bytes": size}  # fmt: skip

    # ---- portability --------------------------------------------------------------------------------------------
    def export_jsonl(self, path: str | Path) -> int:
        count = 0
        with Path(path).expanduser().open("w", encoding="utf-8") as out:
            for row in self._db.execute("SELECT * FROM episodes ORDER BY id").fetchall():
                episode = asdict(self._episode(row))
                episode["steps"] = [
                    {k: v for k, v in asdict(s).items() if k not in ("state_vec", "id", "episode_id")}
                    for s in self.steps_for([row["id"]], with_vectors=False)
                ]
                out.write(json.dumps(episode, ensure_ascii=False) + "\n")
                count += 1
        return count

    def import_jsonl(self, path: str | Path) -> int:
        count = 0
        with Path(path).expanduser().open(encoding="utf-8") as src, self._transaction():
            for line in src:
                if not line.strip():
                    continue
                data = json.loads(line)
                episode_id = self.begin_episode(data["goal"], app=data.get("app"), model=data.get("model"))
                for step in data.get("steps", []):
                    self.record_step(
                        episode_id,
                        idx=step["idx"],
                        app=step.get("app"),
                        window=step.get("window"),
                        state_summary=step["state_summary"],
                        operation=step["operation"],
                        memory_key=step.get("memory_key"),
                        target_text=step.get("target_text"),
                        probability=step.get("probability"),
                        confidence=step.get("confidence"),
                        outcome=step.get("outcome", "unknown"),
                        text=step.get("text"),
                    )
                status = data.get("status", "done")
                self.finish_episode(episode_id, status if status in FINAL_STATUSES else "done")
                count += 1
        return count

    def close(self) -> None:
        self._db.close()

    # ---- helpers ------------------------------------------------------------------------------------------------
    def _transaction(self) -> _Transaction:
        return _Transaction(self._db)

    def _episode(self, row: sqlite3.Row) -> EpisodeRecord:
        return EpisodeRecord(
            id=row["id"], goal=row["goal"], app=row["app"], status=row["status"], steps=row["steps"],
            model=row["model"], started_at=row["started_at"], finished_at=row["finished_at"],
        )  # fmt: skip

    def _step(self, row: sqlite3.Row, with_vectors: bool) -> StepRecord:
        return StepRecord(
            id=row["id"], episode_id=row["episode_id"], idx=row["idx"], app=row["app"], window=row["window"],
            state_summary=row["state_summary"], operation=row["operation"], memory_key=row["memory_key"],
            target_text=row["target_text"], text=row["text"], probability=row["probability"],
            confidence=row["confidence"], outcome=row["outcome"], created_at=row["created_at"],
            state_vec=self.embedder.from_bytes(row["state_vec"]) if with_vectors else None,
        )  # fmt: skip


class _Transaction:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def __enter__(self) -> None:
        self.db.execute("BEGIN")

    def __exit__(self, exc_type: object, *_: object) -> None:
        self.db.execute("ROLLBACK" if exc_type else "COMMIT")
