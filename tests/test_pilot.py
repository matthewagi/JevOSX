"""Claude in the console: the conversation loop, its tools, and the request it sends to the API."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from types import SimpleNamespace as NS
from typing import Any

import pytest

from jevosx.config import PilotSettings
from jevosx.pilot import SYSTEM, TOOLS, AnthropicModel, Pilot, summarize_run
from tests.test_ui import make_manager


def text(words: str) -> Any:
    return NS(type="text", text=words)


def use(name: str, args: dict[str, Any], id: str = "t1") -> Any:
    return NS(type="tool_use", id=id, name=name, input=args)


class Scripted:
    """Replies in order; records what it was sent."""

    name = "scripted"

    def __init__(self, *replies: Any):
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def complete(self, request, on_text):
        self.requests.append({**request, "messages": list(request["messages"])})
        reply = self.replies.pop(0)
        for block in reply.content:
            if block.type == "text":
                on_text(block.text)
        return reply


class Host:
    def __init__(self):
        self.calls: list[tuple[str, Any]] = []

    def look(self):
        self.calls.append(("look", None))
        return {"app": "Chrome", "window": "Marketplace"}

    def run_task(self, goal, max_steps, texts):
        self.calls.append(("run_task", (goal, max_steps, texts)))
        return {"status": "done", "message": "", "steps": 3, "log": []}

    def recent_runs(self, count):
        self.calls.append(("recent_runs", count))
        return []


def pilot_with(*replies, settings=None):
    events: list[tuple[str, dict[str, Any]]] = []
    host = Host()
    model = Scripted(*replies)
    pilot = Pilot(settings or PilotSettings(), host, lambda kind, **data: events.append((kind, data)), model=model)
    return pilot, host, model, events


def talk(pilot, words):
    pilot.send(words)
    pilot.wait(5)


def test_claude_hands_a_task_to_jevosx_and_reports_back():
    pilot, host, model, events = pilot_with(
        NS(
            content=[text("I'll open the listing form."), use("run_task", {"goal": "open the listing form"})],
            stop_reason="tool_use",
        ),  # fmt: skip
        NS(content=[text("The form is open. Do you have photos?")], stop_reason="end_turn"),
    )
    talk(pilot, "sell my welding gun")
    assert host.calls == [("run_task", ("open the listing form", 15, {}))]
    roles = [m["role"] for m in pilot.messages]
    assert roles == ["user", "assistant", "user", "assistant"]  # append-only: nothing earlier is rewritten
    result = pilot.messages[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"
    assert json.loads(result["content"])["status"] == "done"
    kinds = [k for k, _ in events]
    assert kinds.count("pilot_text") == 2 and kinds[-1] == "pilot_reply"
    assert events[-1][1]["text"] == "The form is open. Do you have photos?"
    starts = [d for k, d in events if k == "pilot_tool" and d["phase"] == "start"]
    assert starts[0]["name"] == "run_task" and starts[0]["input"]["goal"] == "open the listing form"
    assert model.requests[0]["system"] == SYSTEM and model.requests[0]["tools"] == TOOLS


def test_bad_tool_input_is_answered_with_an_error_and_nothing_runs():
    pilot, host, _, _ = pilot_with(
        NS(content=[use("run_task", {"goal": ""}), use("teleport", {}, id="t2")], stop_reason="tool_use"),
        NS(content=[text("Sorry.")], stop_reason="end_turn"),
    )
    talk(pilot, "go")
    results = pilot.messages[2]["content"]
    assert all(r["is_error"] for r in results) and host.calls == []
    assert "goal" in results[0]["content"] and "teleport" in results[1]["content"]


def test_a_cut_off_or_declined_reply_never_runs_its_tools():
    pilot, host, _, events = pilot_with(
        NS(content=[use("run_task", {"goal": "open"})], stop_reason="max_tokens"),
        NS(content=[text("Trying again.")], stop_reason="end_turn"),
    )
    talk(pilot, "go")
    assert host.calls == [] and pilot.messages[2]["content"][0]["is_error"]

    pilot, host, _, events = pilot_with(NS(content=[use("run_task", {"goal": "x"})], stop_reason="refusal"))
    talk(pilot, "go")
    assert host.calls == [] and events[-1][0] == "pilot_error"


def test_a_runaway_loop_stops_at_the_cap():
    settings = PilotSettings(max_tool_calls=2)
    looping = [NS(content=[use("look_at_screen", {}, id=f"t{i}")], stop_reason="tool_use") for i in range(3)]
    pilot, host, _, _ = pilot_with(*looping, NS(content=[text("Stuck.")], stop_reason="end_turn"), settings=settings)
    talk(pilot, "go")
    assert len(host.calls) == 2 and pilot.messages[-2]["content"][0]["is_error"]


def test_stop_and_busy():
    pilot, host, _, events = pilot_with()

    def complete(request, on_text):  # Stop is pressed while Claude is still answering
        pilot.stop()
        return NS(content=[use("look_at_screen", {})], stop_reason="tool_use")

    pilot._model = NS(name="stopping", complete=complete)
    talk(pilot, "go")
    assert host.calls == [] and events[-1] == ("pilot_reply", {"text": "Stopped."})

    slow = NS(name="slow", complete=lambda request, on_text: time.sleep(0.3) or NS(content=[], stop_reason="end_turn"))
    pilot._model = slow
    pilot.send("one")
    with pytest.raises(RuntimeError):
        pilot.send("two")
    pilot.wait(5)


def test_status_says_what_is_missing():
    events: list[Any] = []
    host = Host()
    assert Pilot(PilotSettings(enabled=False), host, events.append).status()["available"] is False
    assert Pilot(PilotSettings(), host, events.append).status()["available"] is None  # no key yet
    assert Pilot(PilotSettings(), host, events.append, api_key="sk-test").status()["available"] is True


def test_run_summary_for_claude():
    run = {
        "result": {"status": "done", "message": ""},
        "events": [
            {"step": 0, "status": "plan", "message": "1. Open it"},
            {"step": 0, "status": "ask", "message": "What condition is it in?"},
            {"step": 1, "status": "acted", "action": "OPEN_APP Safari"},
            {"step": 2, "status": "done", "action": "DONE"},
        ],
    }
    summary = summarize_run(run, {"app": "Safari"})
    assert summary["status"] == "done" and summary["steps"] == 1 and summary["plan"] == "1. Open it"
    assert summary["asked_the_person"] == ["What condition is it in?"] and summary["screen_after"] == {"app": "Safari"}


def test_the_api_request_follows_the_current_claude_api():
    captured: dict[str, Any] = {}

    class Stream:
        def __iter__(self):
            yield NS(type="text", text="Hi")

        def get_final_message(self):
            return NS(content=[text("Hi")], stop_reason="end_turn")

    @contextmanager
    def stream(**kwargs):
        captured.update(kwargs)
        yield Stream()

    model = object.__new__(AnthropicModel)
    model.settings = PilotSettings(effort="high")
    model.name = "claude-opus-5-5"
    model.client = NS(beta=NS(messages=NS(stream=stream)))
    streamed: list[str] = []
    reply = model.complete({"system": SYSTEM, "tools": TOOLS, "messages": [{"role": "user", "content": "hi"}]},
                           streamed.append)  # fmt: skip
    assert reply.stop_reason == "end_turn" and streamed == ["Hi"]
    assert captured["model"] == "claude-opus-5-5" and captured["thinking"] == {"type": "adaptive"}
    assert captured["output_config"] == {"effort": "high"} and captured["cache_control"] == {"type": "ephemeral"}
    assert captured["betas"] == ["server-side-fallback-2026-07-01"] and captured["fallbacks"] == "default"
    custom = [t for t in captured["tools"] if "input_schema" in t]
    assert len(custom) == 3 and all(t["eager_input_streaming"] for t in custom)
    assert {"type": "web_search_20260209", "name": "web_search", "max_uses": 3} in captured["tools"]


def test_the_demo_console_talks_it_through():
    manager, desktop, _ = make_manager()
    try:
        events: list[tuple[str, Any]] = []
        manager.pilot.publish = lambda kind, **data: events.append((kind, data))
        manager.pilot.send("Open Notes")
        manager.pilot.wait(10)
        assert desktop.front == "Notes"
        run = manager.history[0]
        assert run["options"]["origin"] == "claude" and run["goal"] == "Open Notes"
        assert events[-1][0] == "pilot_reply" and events[-1][1]["text"].startswith("Done")
        assert manager.pilot.status()["available"] is True
    finally:
        manager.close()


def test_claude_prepares_the_exact_text_and_never_passwords():
    pilot, host, _, _ = pilot_with(
        NS(
            content=[
                use("run_task", {"goal": "fill in the form", "texts": {"title": "Welding gun", "price": "40"}}),
                use("run_task", {"goal": "log in", "texts": {"password": "hunter2"}}, id="t2"),
            ],
            stop_reason="tool_use",
        ),  # fmt: skip
        NS(content=[text("Done.")], stop_reason="end_turn"),
    )
    talk(pilot, "list it")
    assert host.calls == [("run_task", ("fill in the form", 15, {"title": "Welding gun", "price": "40"}))]
    assert "Keychain" in pilot.messages[2]["content"][1]["content"]


def test_texts_reach_the_run_as_text_slots():
    manager, desktop, _ = make_manager()
    try:
        summary = manager.run_task('In TextEdit, write "x"', 10, {"body": "Hello from Claude"})
        run = manager.history[0]
        assert run["options"]["slots"] == {"body": "Hello from Claude"} and run["options"]["origin"] == "claude"
        assert summary["status"] and "screen_after" in summary
    finally:
        manager.close()
