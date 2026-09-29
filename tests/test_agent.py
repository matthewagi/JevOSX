import json

import pytest

from jevosx.agent import Agent
from jevosx.agent import expect_text_verifier as text_verifier
from jevosx.config import Settings
from jevosx.errors import StaleElementError
from jevosx.executor.keys import key_vocabulary
from jevosx.memory import HashingEmbedder, MemoryStore
from jevosx.router.policy import JevRouter
from jevosx.types import ActionResult, AppInfo
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


# ---- waiting for a page or an app instead of judging too early ---------------------------------------------------
CHROME = AppInfo("Google Chrome", "com.google.Chrome", pid=300)
SEARCHED = "the weather in Valletta tomorrow"


def chrome_page(title, url, value=""):
    def build():
        field = element(1, "AXTextField", "Address and search bar", kind="text_input", ops=("TYPE_TEXT", "CLICK"))
        field.value = value
        obs = observation([field], app=CHROME, window=f"{title} - Google Chrome")
        obs.page_url = url
        return obs

    return build


class LoadingChrome(FakeDesktop):
    """A fresh Chrome window whose results appear `delay` reads after Return (never, when `delay` is None)."""

    def __init__(self, delay):
        screens = {
            "blank": chrome_page("about:blank", "about:blank"),
            "loading": chrome_page("about:blank", "about:blank", SEARCHED),
            "results": chrome_page(f"{SEARCHED} - Google Search", "https://www.google.com/search?q=x", SEARCHED),
        }
        super().__init__(screens, "blank", {("blank", "TYPE_TEXT"): "loading"})
        self.delay, self.loading_reads = delay, 0

    def observe(self):
        if self.screen == "loading" and self.delay is not None:
            if self.loading_reads >= self.delay:
                self.screen = "results"
            self.loading_reads += 1
        return super().observe()


def search_then_done(body):
    values = [e.get("value") for e in body["state"]["elements"]]
    return {"operation": "TYPE_TEXT"} if SEARCHED not in values else {"operation": "DONE"}


def test_done_waits_for_the_page_after_return_in_the_address_bar(tmp_path):
    """Seen live: DONE (conf 0.66) was decided on "about:blank", before Google's results had loaded."""
    requests = []
    desktop = LoadingChrome(delay=3)
    agent, _, _ = make_agent(tmp_path, search_then_done, desktop=desktop, requests=requests)
    with agent:
        result = agent.run(f"search Google for {SEARCHED}", text_slots={"phrase_1": SEARCHED})
    assert result.status == "done" and result.steps == 1  # DONE is not counted as a step
    assert desktop.executed == [f'TYPE_TEXT [1] textfield "Address and search bar" <- {SEARCHED}']
    assert agent.last_observation.window.title.endswith("Google Search - Google Chrome")
    assert len(requests) == 2  # Jev was not asked while the page loaded


def test_done_is_rejected_while_the_page_stays_blank(tmp_path):
    agent, _, _ = make_agent(tmp_path, search_then_done, desktop=LoadingChrome(delay=None))
    with agent:
        result = agent.run(f"search Google for {SEARCHED}", text_slots={"phrase_1": SEARCHED})
    assert result.status == "failed" and result.message == "the page did not load"
    assert [e.message for e in result.events if e.action == "DONE"][0] == "DONE rejected: the page has not loaded yet"


NOTES = AppInfo("Notes", "com.apple.Notes", pid=200)


class SlowNotes(FakeDesktop):
    """Notes launches, but only comes to the front `arrives_after` reads later (never, when None)."""

    def __init__(self, arrives_after):
        screens = {
            "start": lambda: observation([element(1, "AXButton", "Bold")], installed=[NOTES]),
            "notes": lambda: observation([element(1, "AXButton", "New Note")], app=NOTES, window="All iCloud"),
        }
        super().__init__(screens, "start", {})
        self.arrives_after, self.reads = arrives_after, None

    def execute(self, action, obs):
        if action.operation != "OPEN_APP":
            return super().execute(action, obs)
        self.opened.append(action.app.name)
        self.reads = 0
        return ActionResult(False, "launch", f"{action.app.name} did not become frontmost in time", unconfirmed=True)

    def observe(self):
        if self.reads is not None and self.arrives_after is not None:
            self.reads += 1
            if self.reads > self.arrives_after:
                self.screen = "notes"
        return super().observe()


def open_notes(body):
    labels = [e["label"] for e in body["state"]["elements"]]
    if "New Note" in labels:
        return {"operation": "DONE"}
    if "app_target" not in body["questions"]:  # the apps are offered once OPEN_APP is chosen
        return {"operation": "OPEN_APP"}
    return {"operation": "OPEN_APP", "app_target": find_id(body, "app_target", "Notes")}


def test_a_slow_app_is_waited_for_instead_of_opened_again(tmp_path):
    """Seen live: OPEN_APP Notes gave up after 8 s twice in a row while a slow Notes was still coming forward."""
    requests = []
    desktop = SlowNotes(arrives_after=4)
    agent, _, store = make_agent(tmp_path, open_notes, desktop=desktop, requests=requests)
    with agent:
        result = agent.run("open Notes")
        steps = store.steps_for([result.episode_id], with_vectors=False)
    assert result.status == "done" and result.steps == 1 and desktop.opened == ["Notes"]
    assert len(requests) == 2 and requests[1]["state"]["desktop"]["frontmost_app"] == "Notes"  # none while it launched
    assert [(s.operation, s.outcome) for s in steps] == [("OPEN_APP", "changed"), ("DONE", "final")]


def test_an_app_that_never_comes_forward_is_left_to_jev_with_the_failure(tmp_path):
    requests = []
    desktop = SlowNotes(arrives_after=None)
    agent, _, store = make_agent(tmp_path, open_notes, desktop=desktop, requests=requests)
    with agent:
        result = agent.run("open Notes", max_steps=2)
        first = store.steps_for([result.episode_id], with_vectors=False)[0]
    assert first.outcome == "failed"
    assert result.status == "max_steps" and desktop.opened == ["Notes", "Notes"]
    assert "failed: Notes did not become frontmost in time" in json.dumps(requests[1]["state"])


class ErrorOnPress(FakeDesktop):
    """The press works, but the app answers it with AXError -25205 (seen live: "New Note" in Notes)."""

    def execute(self, action, obs):
        result = super().execute(action, obs)
        if action.operation == "CLICK":
            return ActionResult(False, "ax-error", "perform AXPress failed with AXError -25205", unconfirmed=True)
        return result


def test_a_press_reported_as_an_error_counts_when_the_ui_changed(tmp_path):
    desktop = ErrorOnPress(screens(), "start", {("start", "New Document"): "doc", ("doc", "TYPE_TEXT"): "typed"})
    agent, _, store = make_agent(tmp_path, desktop=desktop)
    with agent:
        result = agent.run('Create a new document and type "hello"', verifier=text_verifier("hello"))
        steps = store.steps_for([result.episode_id], with_vectors=False)
    assert result.status == "success" and result.steps == 2
    assert [(s.operation, s.outcome) for s in steps][0] == ("CLICK", "changed")


def test_a_press_reported_as_an_error_fails_when_nothing_changed(tmp_path):
    def press(body):
        return {"operation": "CLICK", "click_target": find_id(body, "click_target", "New Document")}

    requests = []
    agent, _, store = make_agent(tmp_path, press, desktop=ErrorOnPress(screens(), "start", {}), requests=requests)
    with agent:
        result = agent.run("make a document", max_steps=2)
        first = store.steps_for([result.episode_id], with_vectors=False)[0]
    assert first.outcome == "failed"
    assert "failed: perform AXPress failed with AXError -25205" in json.dumps(requests[1]["state"])


def test_a_goal_to_write_is_not_done_before_anything_was_typed(tmp_path):
    """Seen live: "write a shopping list" in Notes ended DONE right after OPEN_APP, on the list an earlier run wrote."""
    earlier = observation(
        [element(1, "AXTextArea", "Body", value="milk eggs bread", kind="text_input", ops=("TYPE_TEXT", "CLICK"))],
        text="milk eggs bread",
    )
    desktop = FakeDesktop({"note": lambda: earlier}, "note", {})

    def done_unless_told(body):
        rejected = "nothing written yet" in json.dumps(body["state"]["recent_actions"])
        typed = "TYPE_TEXT" in json.dumps(body["state"]["recent_actions"])
        return {"operation": "TYPE_TEXT"} if rejected and not typed else {"operation": "DONE"}

    agent, _, _ = make_agent(tmp_path, done_unless_told, desktop=desktop)
    with agent:
        result = agent.run("write a shopping list: milk, eggs, bread", text_slots={"list": "milk, eggs, bread"})
    assert [e.status for e in result.events] == ["failed", "acted", "done"] and result.status == "done"
    assert len(desktop.executed) == 1 and desktop.executed[0].startswith("TYPE_TEXT")
