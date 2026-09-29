"""Talking to the console that is already open (`jevosx ui`) from another process: `jevosx ask`, and the connector
that lets the Claude app on this Mac drive JevOSX (`jevosx mcp`).

Runs started this way appear in the console like any other: the person sees every step there and answers its
questions and approvals there. Nothing here can approve anything.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .ui.server import console_file


class ConsoleNotRunning(Exception):
    pass


class ConsoleClient:
    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token

    @classmethod
    def find(cls) -> ConsoleClient:
        """The console this user has open, from the address and token it left in ~/.jevosx/console.json."""
        try:
            data = json.loads(console_file().read_text(encoding="utf-8"))
            client = cls(str(data["url"]), str(data["token"]))
        except (OSError, ValueError, KeyError) as exc:
            raise ConsoleNotRunning("the console is not running: start it with jevosx ui") from exc
        if not client.alive():
            raise ConsoleNotRunning("the console is not answering: is jevosx ui still running?")
        return client

    def call(self, path: str, body: dict[str, Any] | None = None, *, timeout_s: float = 10) -> Any:
        request = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-JevOSX-Token": self.token},
            method="GET" if body is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read() or b"{}")

    def alive(self) -> bool:
        try:
            self.call("/api/status", timeout_s=3)
        except (OSError, ValueError):
            return False
        return True

    # ---- runs ---------------------------------------------------------------------------------------------------
    def start_run(
        self, goal: str, *, texts: dict[str, str] | None = None, max_steps: int | None = None, origin: str = "person"
    ) -> str:
        body: dict[str, Any] = {"goal": goal, "origin": origin}
        if texts:
            body["slots"] = texts
        if max_steps:
            body["max_steps"] = max_steps
        try:
            return str(self.call("/api/run", body).get("run_id", ""))
        except urllib.error.HTTPError as exc:
            detail = json.loads(exc.read() or b"{}").get("error", exc.reason)
            raise RuntimeError(f"the console refused: {detail}") from exc

    def run(self, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The run (its events so far, and its result once finished) and what the console is waiting on."""
        state = self.call("/api/state")
        current = state.get("current") or {}
        run = current if current.get("id") == run_id else None
        run = run or next((r for r in state.get("history", []) if r.get("id") == run_id), {"id": run_id})
        return run, list(state.get("pending_approvals", []))

    def follow(
        self,
        run_id: str,
        timeout_s: float,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        on_pending: Callable[[dict[str, Any]], None] | None = None,
        poll_s: float = 0.5,
    ) -> dict[str, Any]:
        """Wait for a run to finish (or `timeout_s`), reporting new events and new questions as they come."""
        shown, asked = 0, set()
        deadline = time.monotonic() + timeout_s
        while True:
            run, pending = self.run(run_id)
            events = run.get("events", [])
            for event in events[shown:]:
                if on_event is not None:
                    on_event(event)
            shown = len(events)
            for item in pending:
                if item["request_id"] not in asked:
                    asked.add(item["request_id"])
                    if on_pending is not None:
                        on_pending(item)
            if run.get("result") is not None or time.monotonic() > deadline:
                return run
            time.sleep(poll_s)

    def look(self) -> dict[str, Any]:
        return dict(self.call("/api/look"))

    def recent(self, count: int) -> list[dict[str, Any]]:
        episodes = self.call("/api/memory").get("episodes", [])
        return [{k: e.get(k) for k in ("goal", "status", "steps")} for e in episodes[:count]]


def ensure_console(*, wait_s: float = 30.0, log: Path | None = None) -> ConsoleClient:
    """The open console, or a new one started in the background (it opens its page in the browser, where the person
    watches the runs and answers questions)."""
    try:
        return ConsoleClient.find()
    except ConsoleNotRunning:
        pass
    log = log or Path("~/.jevosx/console.log").expanduser()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as out:
        subprocess.Popen(  # noqa: S603 - our own module, no shell
            [sys.executable, "-m", "jevosx", "ui"],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=out,
            start_new_session=True,
            env={**os.environ},
        )
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        time.sleep(0.5)
        try:
            return ConsoleClient.find()
        except ConsoleNotRunning:
            continue
    raise ConsoleNotRunning(f"started the console, but it did not come up; see {log}")
