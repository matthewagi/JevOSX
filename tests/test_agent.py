import json

import pytest

from jevosx.agent import Agent
from jevosx.agent import expect_text_verifier as text_verifier
from jevosx.config import Settings
from jevosx.errors import StaleElementError
from jevosx.executor.keys import key_vocabulary
from jevosx.memory import HashingEmbedder, MemoryStore
from jevosx.router.policy import JevRouter
from jevosx.types import AppInfo
from tests.fakes import FakeDesktop, element, find_id, observation, scripted_client


def screens():
    return {
        "start": lambda: observation([element(1, "AXButton", "New Document"), element(2, "AXButton", "Delete All")]),
        "doc": lambda: observation(
            [element(1, "AXTextArea", "Body", kind="text_input", ops=("TYPE_TEXT", "CLICK"), focused=True)],
            window="Untitled 2",
        ),
        "typed": lambda: observation(
            [element(1, "AXTextArea", "Body", value="hello", kind="text_input", ops=("TYPE_TEXT", "CLICK"))],
            window="Untitled 2",
            text="hello",
        ),
    }


def policy(body):
    """A stand-in for Jev: reads the state it was sent and picks like a sensible model would."""
    labels = [e["label"] for e in body["state"]["elements"]]
    values = [e.get("value") for e in body["state"]["elements"]]
    if "New Document" in labels:
        return {"operation": "CLICK", "click_target": find_id(body, "click_target", "New Document")}
    if "hello" not in values:
        return {"operation": "TYPE_TEXT"}
    return {"operation": "DONE"}


def make_agent(tmp_path, answer=policy, *, desktop=None, memory=True, confirm=None, settings=None, requests=None):
    desktop = desktop or FakeDesktop(
        screens(), "start", {("start", "New Document"): "doc", ("doc", "TYPE_TEXT"): "typed"}
    )
    settings = settings or Settings()
    settings.executor.wait_s = 0
    settings.agent.fallback_log = str(tmp_path / "fallbacks.jsonl")
    store = MemoryStore(tmp_path / "memory.db", HashingEmbedder(settings.memory.dim)) if memory else None
    agent = Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(answer, requests), keys=key_vocabulary()),
        settings=settings,
        memory=store,
        confirm=confirm,
        sleep=lambda _s: None,
    )
    return agent, desktop, store


def test_full_run_types_goal_literal_verifies_and_learns(tmp_path):
    agent, desktop, store = make_agent(tmp_path)
    with agent:
        result = agent.run('Create a new document and type "hello"', verifier=text_verifier("hello"))
        assert result.status == "success" and result.steps == 2
        assert desktop.executed == ['CLICK [1] button "New Document"', 'TYPE_TEXT [1] textarea "Body" <- hello']
        assert [e.status for e in result.events] == ["acted", "acted", "done"]
        steps = store.steps_for([result.episode_id], with_vectors=False)
        assert [(s.operation, s.outcome) for s in steps] == [
            ("CLICK", "changed"),
            ("TYPE_TEXT", "changed"),
            ("DONE", "final"),
        ]
        assert store.episode(result.episode_id).status == "success"


def test_second_run_receives_memory_hints_from_the_first(tmp_path):
    requests = []
    agent, desktop, store = make_agent(tmp_path, requests=requests)
    with agent:
        agent.run('Create a new document and type "hello"', verifier=text_verifier("hello"))
        requests.clear()
        desktop.screen = "start"
        agent.run('Create a new document and type "hello there"', max_steps=1)
    first = requests[0]
    hints = first["state"]["memory_hints"]
    assert hints[0]["kind"] == "worked" and hints[0]["operation"] == "CLICK"
    target = first["questions"]["click_target"]["criteria"][hints[0]["target_id"]]
    assert "New Document" in target["element"] and "used successfully" in target["memory"]
    assert "memory_hints" in first["questions"]["operation"]["instructions"]["rules"]


def test_consequential_controls_need_confirmation(tmp_path):
    def answer(body):
        return {"operation": "CLICK", "click_target": find_id(body, "click_target", "Delete All")}

    asked = []
    agent, desktop, _ = make_agent(tmp_path, answer, confirm=lambda action, reason: asked.append(reason) or False)
    with agent:
        result = agent.run("clean up", max_steps=2)
    assert desktop.executed == [] and len(asked) == 2
    assert [e.status for e in result.events] == ["declined", "declined"] and result.status == "max_steps"


def test_stuck_detection_blocks_after_no_visible_change(tmp_path):
    def answer(body):
        return {"operation": "CLICK", "click_target": find_id(body, "click_target", "New Document")}

    desktop = FakeDesktop(screens(), "start", {})  # clicking changes nothing
    agent, _, _ = make_agent(tmp_path, answer, desktop=desktop)
    with agent:
        result = agent.run("make a document")
    assert result.status == "blocked" and "no visible change" in result.message
    assert len(desktop.executed) == 3


def test_verifier_rejects_premature_done(tmp_path):
    agent, _, _ = make_agent(tmp_path, lambda body: {"operation": "DONE"})
    with agent:
        result = agent.run("do it", verifier=lambda obs: False)
    assert result.status == "failed" and result.steps == 3


def fallback_records(tmp_path):
    path = tmp_path / "fallbacks.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def unsure(body):
    """Jev answers CLICK, but with only 0.25 confidence."""
    return {"operation": ("CLICK", 0.25), "click_target": find_id(body, "click_target", "New Document")}


def test_default_confidence_floor_is_065():
    assert Settings().agent.min_confidence == 0.65


def test_low_confidence_is_withheld_logged_and_retried_then_stops(tmp_path):
    agent, desktop, _ = make_agent(tmp_path, unsure)
    with agent:
        result = agent.run("anything")
    assert result.status == "low_confidence" and desktop.executed == []
    assert [e.status for e in result.events] == ["low_confidence"] * 3
    records = fallback_records(tmp_path)
    assert len(records) == 3 and records[0]["resolution"] == "retry" and records[0]["confidence"] == 0.25
    assert records[0]["floor"] == 0.3 and records[0]["decision"]["operation"] == "CLICK"  # a click is routine
    assert records[0]["risk"] == "routine step: clicks a control"


def test_low_target_confidence_is_gated_even_when_operation_is_confident(tmp_path):
    def answer(body):
        return {"operation": ("CLICK", 0.95), "click_target": (find_id(body, "click_target", "New Document"), 0.2)}

    settings = Settings()
    settings.agent.low_confidence_policy = "stop"
    agent, desktop, _ = make_agent(tmp_path, answer, settings=settings)
    with agent:
        result = agent.run("anything")
    assert result.status == "low_confidence" and desktop.executed == [] and len(fallback_records(tmp_path)) == 1


def test_low_confidence_done_does_not_end_the_run(tmp_path):
    settings = Settings()
    settings.agent.low_confidence_policy = "stop"
    agent, _, _ = make_agent(tmp_path, lambda body: {"operation": ("DONE", 0.25)}, settings=settings)
    with agent:
        assert agent.run("anything").status == "low_confidence"


def test_ask_policy_executes_only_with_human_approval(tmp_path):
    settings = Settings()
    settings.agent.low_confidence_policy = "ask"
    reasons = []
    agent, desktop, _ = make_agent(
        tmp_path, unsure, settings=settings, confirm=lambda action, reason: reasons.append(reason) or True
    )
    with agent:
        agent.run("anything", max_steps=1)
    assert desktop.executed == ['CLICK [1] button "New Document"'] and "below the floor" in reasons[0]
    # It looks again once (a page may still be loading) and only then asks.
    assert [r["resolution"] for r in fallback_records(tmp_path)] == ["retry", "execute"] and len(reasons) == 1


def test_custom_low_confidence_handler(tmp_path):
    seen = []
    agent, desktop, _ = make_agent(tmp_path, unsure)
    agent.on_low_confidence = lambda exc, obs: seen.append(exc.confidence) or "stop"
    with agent:
        assert agent.run("anything").status == "low_confidence"
    assert seen == [0.25] and desktop.executed == []


def test_stale_targets_are_never_executed(tmp_path):
    desktop = FakeDesktop(screens(), "start", {})

    def stale(action, obs):
        raise StaleElementError("moved")

    desktop.validate = stale
    agent, _, _ = make_agent(tmp_path, desktop=desktop)
    with agent:
        result = agent.run("make a document")
    assert result.status == "blocked" and desktop.executed == []


def test_requested_app_is_opened_first_and_missing_text_blocks(tmp_path):
    def answer(body):
        offered = body["questions"]["operation"]["criteria"]
        return {"operation": "TYPE_TEXT" if "TYPE_TEXT" in offered else "BLOCKED"}

    agent, desktop, _ = make_agent(tmp_path, answer, memory=False)
    desktop.screen = "doc"
    with agent:
        result = agent.run("write something", app="Notes")
    assert desktop.opened == ["Notes"]
    assert result.status == "blocked"  # no quoted text and no text model → TYPE_TEXT is never offered

    agent, desktop, _ = make_agent(tmp_path, memory=False)
    with agent:
        assert agent.run("x", app="Nonexistent").status == "error"


def test_empty_goal_is_rejected(tmp_path):
    agent, _, _ = make_agent(tmp_path, memory=False)
    with pytest.raises(ValueError):
        agent.run("   ")


def test_safety_denies_password_typing_without_secret_slot(tmp_path):
    from jevosx.executor.safety import SafetyPolicy
    from jevosx.types import Action

    field = element(1, "AXTextField", "Password", subrole="AXSecureTextField", secure=True, ops=("TYPE_TEXT",))
    policy = SafetyPolicy()
    frontmost = AppInfo("Safari", "com.apple.Safari", pid=1)
    assert policy.check(Action("TYPE_TEXT", element=field, text="x"), frontmost).verdict == "deny"
    assert policy.check(Action("TYPE_TEXT", element=field, text="x", text_is_secret=True), frontmost).allowed
    keychain = AppInfo("Keychain Access", "com.apple.keychainaccess")
    assert policy.check(Action("OPEN_APP", app=keychain), frontmost).verdict == "deny"


def test_safety_never_answers_a_macos_permission_prompt():
    """Seen live: the "Accessibility Access" prompt was in front and Jev clicked Deny at confidence 0.36."""
    from jevosx.executor.safety import SafetyPolicy
    from jevosx.types import Action

    policy = SafetyPolicy()
    prompt = AppInfo("universalAccessAuthWarn", "com.apple.accessibility.universalAccessAuthWarn", pid=9)
    deny = element(2, "AXButton", "Deny", ops=("CLICK",))
    assert policy.check(Action("CLICK", element=deny), prompt).verdict == "deny"
    assert policy.check(Action("PRESS_KEY"), prompt).verdict == "deny"
    textedit = AppInfo("TextEdit", "com.apple.TextEdit")
    assert policy.check(Action("OPEN_APP", app=textedit), prompt).allowed  # leaving the prompt is fine


def test_safety_never_acts_inside_the_console_window():
    from jevosx.executor.keys import key_vocabulary
    from jevosx.executor.safety import SafetyPolicy
    from jevosx.types import Action, UIElement

    policy = SafetyPolicy()
    safari = AppInfo("Safari", "com.apple.Safari", pid=1)
    keys = key_vocabulary()
    title = "JevOSX Console"
    field = element(1, "AXTextField", "Search or enter website name", ops=("TYPE_TEXT",))
    assert policy.check(Action("TYPE_TEXT", element=field, text="x"), safari, window_title=title).verdict == "deny"
    assert policy.check(Action("PRESS_KEY", key=keys["CMD_L"]), safari, window_title=title).verdict == "deny"
    assert policy.check(Action("PRESS_KEY", key=keys["CMD_N"]), safari, window_title=title).allowed
    close = UIElement(2, "AXMenuItem", None, "File › Close Window", kind="menu_item", ops=("MENU",))
    new = UIElement(1, "AXMenuItem", None, "File › New Window", kind="menu_item", ops=("MENU",))
    assert policy.check(Action("MENU", element=close), safari, window_title=title).verdict == "deny"
    assert policy.check(Action("MENU", element=new), safari, window_title=title).allowed
    assert policy.check(Action("TYPE_TEXT", element=field, text="x"), safari, window_title="Apple").allowed


def console_desktop():
    from jevosx.types import AppInfo as App

    chrome = App("Google Chrome", "com.google.Chrome", pid=300)
    screens = {
        "console": lambda: observation(
            [element(1, "AXTextField", "Address and search bar", kind="text_input", ops=("TYPE_TEXT", "CLICK"))],
            app=chrome,
            window="JevOSX Console - Google Chrome",
            running=[chrome],
        ),
        "new": lambda: observation(
            [
                element(
                    1,
                    "AXTextField",
                    "Address and search bar",
                    kind="text_input",
                    ops=("TYPE_TEXT", "CLICK"),
                    focused=True,
                )
            ],
            app=chrome,
            window="New Tab - Google Chrome",
            running=[chrome],
        ),  # fmt: skip
    }
    return FakeDesktop(screens, "console", {("console", "CMD_N"): "new"})


def test_leaving_the_console_is_not_held_back_but_content_actions_are(tmp_path):
    def answer(body):
        if "never act inside it" in body["state"]["desktop"].get("note", ""):
            return {"operation": ("PRESS_KEY", 0.55), "key_target": ("CMD_N", 0.9)}
        return {"operation": ("TYPE_TEXT", 0.15)}

    desktop = console_desktop()
    agent, _, _ = make_agent(tmp_path, answer, desktop=desktop, memory=False)
    with agent:
        result = agent.run("look for pictures of flowers red")
    assert desktop.executed[0] == "PRESS_KEY CMD_N (cmd+n)"
    assert result.events[0].status == "acted" and "not gated" in result.events[0].message
    assert all("TYPE_TEXT" not in a for a in desktop.executed)  # typing at 0.15 stays withheld
    assert result.status == "low_confidence"

    settings = Settings()
    settings.agent.gate_console_navigation = True
    settings.agent.safe_confidence = 0.6  # a new window is a safe step: gated at the safe floor
    desktop = console_desktop()
    agent, _, _ = make_agent(tmp_path, answer, desktop=desktop, memory=False, settings=settings)
    with agent:
        assert agent.run("look for pictures of flowers red").status == "low_confidence"
    assert desktop.executed == []


def test_sending_needs_confirmation_by_default():
    from jevosx.executor.safety import SafetyPolicy
    from jevosx.types import Action

    send = element(3, "AXButton", "Send")
    mail = AppInfo("Mail", "com.apple.mail", pid=9)
    assert SafetyPolicy().check(Action("CLICK", element=send), mail).verdict == "confirm"
