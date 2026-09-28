import http.client
import json
import time

import pytest

from jevosx.config import Settings
from jevosx.executor.keys import key_vocabulary
from jevosx.memory import MemoryStore
from jevosx.router.policy import JevRouter
from jevosx.ui.demo import DemoDesktop, SimulatedJev
from jevosx.ui.server import Components, EventBus, RunManager, RunOptions, UIServer


def demo_settings():
    settings = Settings()
    settings.executor.wait_s = 0
    settings.executor.settle_poll_s = 0.001
    settings.agent.fallback_log = ""
    return settings


def make_manager():
    desktop = DemoDesktop(latency=False)

    def builder(settings, demo):
        router = JevRouter.from_settings(SimulatedJev(latency=False).client(), settings.jev, key_vocabulary())
        return Components(desktop, desktop, router, MemoryStore(":memory:"), None, demo=True)

    bus = EventBus()
    return RunManager(demo_settings(), demo=True, bus=bus, builder=builder), desktop, bus


def run_to_end(manager, goal, *, approve=None, timeout=10, **options):
    manager.start(RunOptions(goal=goal, **options))
    deadline = time.monotonic() + timeout
    while manager.running and time.monotonic() < deadline:
        for approval in list(manager._approvals.values()):
            manager.approve(approval.info["request_id"], bool(approve))
        time.sleep(0.01)
    assert not manager.running, "run did not finish"
    return manager.history[0]


@pytest.mark.parametrize(
    ("goal", "check"),
    [
        (
            'Open TextEdit, create a new document and type "Hello from JevOSX"',
            lambda d: d.apps["TextEdit"].body == "Hello from JevOSX",
        ),
        ('In Safari, search for "accessibility API"', lambda d: d.apps["Safari"].page == "accessibility API - Search"),
        (
            'In TextEdit, write "Shopping list" and save it as "groceries"',
            lambda d: d.apps["TextEdit"].document == "groceries" and d.apps["TextEdit"].body == "Shopping list",
        ),
        ("Open Downloads in Finder", lambda d: d.apps["Finder"].folder == "Downloads"),
        ("Open Notes", lambda d: d.front == "Notes"),
        ("look for pictures of flowers red", lambda d: d.apps["Safari"].page == "pictures of flowers red - Search"),
    ],
)
def test_demo_scenarios_complete(goal, check):
    manager, desktop, _ = make_manager()
    try:
        run = run_to_end(manager, goal)
        assert run["status"] == "done", run
        assert check(desktop)
        assert all(e["status"] in ("acted", "done") for e in run["events"])
    finally:
        manager.close()


def test_consequential_step_waits_for_approval():
    manager, desktop, _ = make_manager()
    try:
        denied = run_to_end(manager, 'Delete the "Trip ideas" note in Notes', approve=False)
        assert denied["status"] == "blocked" and "Trip ideas" in desktop.apps["Notes"].notes
        assert any(e["status"] == "declined" for e in denied["events"])
        allowed = run_to_end(manager, 'Delete the "Trip ideas" note in Notes', approve=True)
        assert allowed["status"] == "done" and "Trip ideas" not in desktop.apps["Notes"].notes
    finally:
        manager.close()


def test_vague_request_is_withheld_by_the_confidence_gate():
    manager, desktop, _ = make_manager()
    try:
        run = run_to_end(manager, "Do something interesting")
        assert run["status"] == "low_confidence"
        assert {e["status"] for e in run["events"]} == {"low_confidence"}
        assert all(e["decision"]["gate_confidence"] < 0.65 for e in run["events"])
    finally:
        manager.close()


def test_second_run_gets_memory_hints_and_stop_works():
    manager, desktop, _ = make_manager()
    try:
        goal = 'Open TextEdit, create a new document and type "Hello"'
        run_to_end(manager, goal)
        desktop.apps["TextEdit"].__init__()
        desktop.front = "Finder"
        second = run_to_end(manager, goal)
        assert any(e["hints"] for e in second["events"])
        assert manager.memory()["stats"]["episodes"] == 2
    finally:
        manager.close()


def test_run_options_validation():
    with pytest.raises(ValueError):
        RunOptions.from_json({"goal": "  "})
    with pytest.raises(ValueError):
        RunOptions.from_json({"goal": "x", "min_confidence": 3})
    with pytest.raises(ValueError):
        RunOptions.from_json({"goal": "x", "low_confidence_policy": "yolo"})
    options = RunOptions.from_json({"goal": " x ", "slots": {"a": "b"}, "max_steps": 5, "min_confidence": 0.7})
    assert options.goal == "x" and options.slots == {"a": "b"} and options.min_confidence == 0.7


@pytest.fixture
def server():
    manager, _, bus = make_manager()
    ui = UIServer(demo_settings(), demo=True, port=0, manager=manager)
    ui.start_background()
    yield ui
    ui.shutdown()


def request(ui, method, path, body=None, *, token=True, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", ui.port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{ui.port}"}
    if token:
        headers["X-JevOSX-Token"] = ui.token
    payload = None
    if body is not None:
        payload = json.dumps(body)
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def test_http_security_and_run_flow(server):
    status, page = request(server, "GET", "/", token=False)
    assert status == 200 and server.token.encode() in page and b"JevOSX" in page
    assert request(server, "GET", "/api/status", token=False)[0] == 401
    assert request(server, "GET", "/api/status", host="evil.example:80")[0] == 403
    assert request(server, "POST", "/api/run", {"goal": "Open Notes"}, token=False)[0] == 401
    assert request(server, "POST", "/api/run", {"goal": ""})[0] == 400

    status, data = request(server, "GET", "/api/status")
    assert status == 200 and json.loads(data)["demo"] is True

    status, data = request(server, "POST", "/api/run", {"goal": 'In Safari, search for "cats"'})
    assert status == 202 and json.loads(data)["run_id"]
    assert request(server, "POST", "/api/run", {"goal": "again"})[0] in (202, 409)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = json.loads(request(server, "GET", "/api/state")[1])
        if not state["running"]:
            break
        time.sleep(0.02)
    assert state["history"][0]["status"] == "done"
    observed = json.loads(request(server, "GET", "/api/observe")[1])
    assert observed["app"]["name"] == "Safari" and observed["window"] == "cats - Search"
    memory = json.loads(request(server, "GET", "/api/memory")[1])
    episode = memory["episodes"][0]["id"]
    assert request(server, "POST", "/api/memory/label", {"episode_id": episode, "status": "success"})[0] == 200


def test_event_stream_delivers_run_events(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
    conn.request("GET", f"/api/events?token={server.token}", headers={"Host": f"127.0.0.1:{server.port}"})
    response = conn.getresponse()
    assert response.status == 200 and response.getheader("Content-Type") == "text/event-stream"
    request(server, "POST", "/api/run", {"goal": "Open Notes"})
    kinds = set()
    deadline = time.monotonic() + 5
    while "run_finished" not in kinds and time.monotonic() < deadline:
        line = response.fp.readline().decode()
        if line.startswith("event: "):
            kinds.add(line.split(": ", 1)[1].strip())
    conn.close()
    assert {"run_started", "step", "run_finished"} <= kinds
