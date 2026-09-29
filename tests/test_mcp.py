"""The connector that lets the Claude app on the Mac drive JevOSX from the chat (`jevosx mcp`)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from typing import Any

import pytest

from jevosx.console_client import ConsoleClient, ConsoleNotRunning
from jevosx.mcp_server import INSTRUCTIONS, TOOLS, Tools, handle, install_desktop, serve
from jevosx.ui.server import Speaker, UIServer, announce
from tests.fakes import FB_GOAL
from tests.test_ui import demo_settings, make_manager


def rpc(method: str, params: dict[str, Any] | None = None, msg_id: int | None = 1) -> dict[str, Any]:
    message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if msg_id is not None:
        message["id"] = msg_id
    if params is not None:
        message["params"] = params
    return message


class FakeTools:
    def __init__(self, fail: Exception | None = None):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.fail = fail

    def call(self, name, arguments):
        self.calls.append((name, arguments))
        if self.fail:
            raise self.fail
        return {"status": "done"}


def test_the_handshake_and_tool_list():
    reply = handle(rpc("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}}), FakeTools())
    assert reply["result"]["protocolVersion"] == "2025-03-26"
    assert reply["result"]["serverInfo"]["name"] == "jevosx" and reply["result"]["instructions"] == INSTRUCTIONS
    newer = handle(rpc("initialize", {"protocolVersion": "2099-01-01"}), FakeTools())
    assert newer["result"]["protocolVersion"] == "2025-06-18"  # the newest this server speaks
    assert handle(rpc("notifications/initialized", msg_id=None), FakeTools()) is None
    names = [t["name"] for t in handle(rpc("tools/list"), FakeTools())["result"]["tools"]]
    assert names == ["run_task", "wait_for_run", "look_at_screen", "recent_runs"]
    assert all(t["inputSchema"]["type"] == "object" for t in TOOLS)
    assert handle(rpc("resources/list"), FakeTools())["error"]["code"] == -32601
    assert handle(rpc("ping"), FakeTools())["result"] == {}


def test_tool_calls_and_their_failures():
    tools = FakeTools()
    reply = handle(rpc("tools/call", {"name": "run_task", "arguments": {"goal": "open Notes"}}), tools)
    assert reply["result"]["isError"] is False
    assert json.loads(reply["result"]["content"][0]["text"]) == {"status": "done"}
    failing = FakeTools(ConsoleNotRunning("the console is not running: start it with jevosx ui"))
    reply = handle(rpc("tools/call", {"name": "look_at_screen"}), failing)
    assert reply["result"]["isError"] is True and "jevosx ui" in reply["result"]["content"][0]["text"]


def test_stdio_carries_one_message_per_line_and_nothing_else():
    lines = [
        json.dumps(rpc("initialize", {"protocolVersion": "2025-06-18"})),
        "not json",
        "",
        json.dumps(rpc("notifications/initialized", msg_id=None)),
        json.dumps(rpc("tools/list", msg_id=2)),
    ]
    out = io.StringIO()
    serve(io.StringIO("\n".join(lines) + "\n"), out, FakeTools())
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 2] and replies[1]["error"]["code"] == -32700


def test_the_real_command_speaks_only_json_on_stdout(tmp_path):
    messages = [rpc("initialize", {"protocolVersion": "2025-06-18"}), rpc("tools/list", msg_id=2)]
    done = subprocess.run(
        [sys.executable, "-m", "jevosx", "mcp"],
        input="".join(json.dumps(m) + "\n" for m in messages),
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PYTHONPATH": str(__import__("pathlib").Path.cwd())},
    )
    replies = [json.loads(line) for line in done.stdout.splitlines()]
    assert [r["id"] for r in replies] == [1, 2] and len(replies[1]["result"]["tools"]) == 4


@pytest.fixture
def console(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    manager, _, _ = make_manager()
    ui = UIServer(demo_settings(), demo=True, port=0, manager=manager)
    ui.speaker = Speaker("")
    ui.start_background()
    announce(ui.url, ui.token)
    yield ui
    ui.shutdown()


def test_claude_runs_a_task_in_the_console_and_sees_the_result(console):
    tools = Tools(connect=ConsoleClient.find, wait_s=10)
    outcome = tools.call("run_task", {"goal": 'In TextEdit, write "x"', "texts": {"body": "Hello from the chat"}})
    assert outcome["status"] == "done" and outcome["screen_after"]["app"] == "TextEdit"
    run = console.manager.history[0]
    assert run["options"]["origin"] == "claude" and run["options"]["slots"] == {"body": "Hello from the chat"}
    assert tools.call("recent_runs", {"count": 3})[0]["goal"] == 'In TextEdit, write "x"'
    screen = tools.call("look_at_screen", {})
    assert screen["app"] == "TextEdit" and any("Body" in line for line in screen["elements"])
    with pytest.raises(ValueError):
        tools.call("run_task", {"goal": "sign in", "texts": {"password": "hunter2"}})


def test_a_long_task_is_handed_back_to_wait_on(console):
    tools = Tools(connect=ConsoleClient.find, wait_s=0)
    outcome = tools.call("run_task", {"goal": FB_GOAL})
    if outcome["status"] == "running":  # still going: the demo asks the person first
        assert outcome["run_id"] and outcome["waiting_for_the_person"] is not None
        deadline = time.monotonic() + 10
        while not console.manager._approvals and time.monotonic() < deadline:
            time.sleep(0.02)
        for approval in list(console.manager._approvals.values()):  # the person answers in the console
            console.manager.approve(approval.info["request_id"], True, answers={"condition": "Used - good"})
        finished = Tools(connect=ConsoleClient.find, wait_s=10).call("wait_for_run", {"run_id": outcome["run_id"]})
        assert finished["status"] == "done"


def test_without_a_console_the_tools_say_how_to_start_one(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ConsoleNotRunning):
        ConsoleClient.find()


def test_install_adds_jevosx_and_keeps_the_other_connectors(tmp_path):
    path = tmp_path / "Claude" / "claude_desktop_config.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"mcpServers": {"notes": {"command": "notes-mcp"}}, "theme": "dark"}))
    install_desktop(str(path))
    config = json.loads(path.read_text())
    assert config["theme"] == "dark" and config["mcpServers"]["notes"] == {"command": "notes-mcp"}
    assert config["mcpServers"]["jevosx"] == {"command": sys.executable, "args": ["-m", "jevosx", "mcp"]}
    assert json.loads((tmp_path / "Claude" / "claude_desktop_config.json.bak").read_text())["theme"] == "dark"
    fresh = tmp_path / "new" / "claude_desktop_config.json"
    install_desktop(str(fresh))
    assert "jevosx" in json.loads(fresh.read_text())["mcpServers"]
