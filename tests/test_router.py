import json

import pytest

from jevosx.errors import JevResponseError, RouterContractError, TextUnavailableError
from jevosx.executor.keys import key_vocabulary
from jevosx.router.client import JevResponse
from jevosx.router.policy import JevRouter, build_state, state_size, validate_request
from jevosx.router.space import ActionSpace
from jevosx.router.text import GENERATE, TextSource, slots_from_goal
from jevosx.types import AppInfo, UIElement, WindowInfo
from tests.fakes import distribution, element, observation, scripted_client


def sample_obs():
    return observation(
        [
            element(1, "AXButton", "New Document"),
            element(2, "AXTextArea", "Body", kind="text_input", ops=("TYPE_TEXT", "CLICK"), focused=True),
            element(3, "AXButton", "Send", enabled=False, ops=()),
        ],
        menu_items=[
            UIElement(1, "AXMenuItem", None, "File › Save…", kind="menu_item", ops=("MENU",), shortcut="⌘S"),
            UIElement(2, "AXMenuItem", None, "File › Close", kind="menu_item", ops=("MENU",)),
        ],
        running=[AppInfo("TextEdit", "com.apple.TextEdit", pid=100), AppInfo("Safari", "com.apple.Safari", pid=5)],
        installed=[AppInfo("Notes", "com.apple.Notes"), AppInfo("Safari", "com.apple.Safari")],
        windows=[WindowInfo(1, "Untitled", focused=True), WindowInfo(2, "Draft", minimized=True)],
    )


def router(answer=lambda body: {}, requests=None):
    return JevRouter(scripted_client(answer, requests), keys=key_vocabulary())


def test_space_offers_only_supported_operations_and_compatible_targets():
    space = ActionSpace.build(sample_obs(), keys=key_vocabulary(), text_available=False, goal="open notes")
    assert "TYPE_TEXT" not in space.operations  # no text source → never offered
    assert list(space.targets_for("CLICK")) == ["2", "1"]  # focused first; disabled "Send" excluded
    assert set(space.targets_for("MENU")) == {"m1", "m2"}
    apps = {t.app.name: t.criterion["status"] for t in space.targets_for("OPEN_APP").values()}
    assert apps == {"Safari": "running", "Notes": "not running"}  # frontmost and duplicates removed
    assert [t.window.title for t in space.targets_for("FOCUS_WINDOW").values()] == ["Draft"]
    assert space.targets_for("SCROLL_DOWN") == {}
    assert list(space.operations)[-3:] == ["WAIT", "DONE", "BLOCKED"]
    assert space.resolve("CLICK", space.targets_for("CLICK")["1"].memory_key).id == "1"


def test_request_contains_state_and_one_question_per_head():
    requests = []
    r = router(requests=requests)
    obs = sample_obs()
    text = TextSource({"subject": "Hello", "password": "hunter2"})
    space = r.space(obs, text)
    state, questions = r.build_request("write hello", obs, space, text_source=text, history=[{"step": 1}])
    assert set(questions) == {
        "operation",
        "click_target",
        "menu_target",
        "key_target",
        "text_slot",
    }  # type_text/app/window heads have a single candidate each → no question needed
    assert questions["operation"]["instructions"]["goal"] == "write hello"
    assert state["desktop"]["frontmost_app"] == "TextEdit"
    assert state["elements"][1] == {
        "index": 2,
        "role": "textarea",
        "label": "Body",
        "state": ["focused"],
        "ops": ["TYPE_TEXT", "CLICK"],
    }
    assert state["text_slots"] == {"subject": "Hello", "password": "••••••"}
    assert "hunter2" not in json.dumps([state, questions])
    assert "AXPosition" not in json.dumps(state) and "node" not in json.dumps(state)


def test_decode_uses_only_the_selected_head():
    obs = sample_obs()
    text = TextSource({"a": "x", "b": "y"})
    r = router()
    space = r.space(obs, text)
    ops = list(space.operations)
    response = JevResponse(
        model="m",
        answers={
            "operation": distribution(ops, "MENU"),
            "menu_target": distribution(["m1", "m2"], "m2"),
            "click_target": {"garbage": True},  # unused speculative head: ignored
        },
    )
    decision = r.decode(response, space, text)
    assert decision.operation == "MENU" and decision.target.element.label == "File › Close"
    assert decision.probability == pytest.approx(0.9 * 0.9)

    bad = JevResponse(model="m", answers={"operation": distribution(ops, "CLICK"), "click_target": {"garbage": 1}})
    with pytest.raises(JevResponseError):
        r.decode(bad, space, text)


def test_single_target_and_text_slot_decisions_are_deterministic():
    obs = sample_obs()
    text = TextSource({"only": "Hello"})
    r = router(lambda body: {"operation": "TYPE_TEXT"})
    space = r.space(obs, text)
    decision = r.decide("type hello", obs, space, text_source=text)
    assert decision.operation == "TYPE_TEXT" and decision.target.id == "2"
    assert decision.target_answer.probability == 1.0 and decision.text_option == "only"


def test_disabled_and_non_interactive_elements_are_not_sent():
    obs = sample_obs()
    r = router()
    text = TextSource({})
    space = r.space(obs, text)
    state, _ = r.build_request("g", obs, space, text_source=text)
    assert [e["label"] for e in state["elements"]] == ["New Document", "Body"]  # disabled "Send" filtered out
    r.include_disabled = True
    state, _ = r.build_request("g", obs, r.space(obs, text), text_source=text)
    assert "Send" in [e["label"] for e in state["elements"]]


def test_state_budget_trims_text_then_elements_and_keeps_the_contract():
    many = [element(i, "AXButton", f"Button number {i} with a long label") for i in range(1, 121)]
    many[5].focused = True
    obs = observation(many, text="lorem ipsum " * 2000)
    r = router()
    r.max_state_bytes = 6_000
    text = TextSource({})
    space = r.space(obs, text)
    state, questions = r.build_request("g", obs, space, text_source=text)
    assert state_size(state) <= 6_000 and len(state["visible_text"]) < 300
    kept = {str(e["index"]) for e in state["elements"]}
    assert "6" in kept and len(kept) < 120  # focused element survives; trailing ones dropped
    assert set(questions["click_target"]["criteria"]) == kept  # every offered id is in the element table
    assert "truncated" in state["desktop"]["note"]


def test_contract_rejects_open_ended_or_unmapped_questions():
    obs = sample_obs()
    text = TextSource({})
    space = router().space(obs, text)
    state = build_state(obs)
    with pytest.raises(RouterContractError):
        validate_request(state, {"operation": {"type": "text", "prompt": "what next?"}}, space)
    bogus = {"type": "choice", "criteria": {"1": "a", "99": "not on screen"}}
    with pytest.raises(RouterContractError):
        validate_request(state, {"click_target": bogus}, space)
    with pytest.raises(RouterContractError):
        scripted_client(lambda body: {}).evaluate({}, {"q": {"type": "scale", "criteria": {"a": 1, "b": 2}}})


def test_installed_apps_are_only_offered_when_the_goal_names_them():
    obs = sample_obs()
    keys = key_vocabulary()
    names = lambda space: sorted(t.app.name for t in space.targets_for("OPEN_APP").values())  # noqa: E731
    assert names(ActionSpace.build(obs, keys=keys, text_available=False, goal="save the file")) == ["Safari"]
    assert names(ActionSpace.build(obs, keys=keys, text_available=False, goal="Open Notes and add a note")) == [
        "Notes",
        "Safari",
    ]
    everything = ActionSpace.build(obs, keys=keys, text_available=False, offer_installed_apps="all")
    assert names(everything) == ["Notes", "Safari"]


def test_slots_from_goal_and_text_resolution():
    assert slots_from_goal('Type "hello world" then `ls -la` and “café”') == {
        "quote_1": "hello world",
        "quote_2": "ls -la",
        "quote_3": "café",
    }
    obs = sample_obs()
    field = obs.elements[1]
    source = TextSource({"password": "s3cret"})
    resolved = source.resolve("password", goal="g", element=field, obs=obs, history=[])
    assert (resolved.text, resolved.secret, resolved.source) == ("s3cret", True, "slot:password")
    with pytest.raises(TextUnavailableError):
        TextSource({}).resolve(GENERATE, goal="g", element=field, obs=obs, history=[])


def test_console_window_only_offers_new_window_or_app_switch():
    obs = observation(
        [element(1, "AXTextField", "Search or enter website name", kind="text_input", ops=("TYPE_TEXT", "CLICK"))],
        app=AppInfo("Safari", "com.apple.Safari", pid=5),
        window="JevOSX Console",
        text="12:24 run search the web for pictures of red flowers",
        menu_items=[
            UIElement(1, "AXMenuItem", None, "File › New Window", kind="menu_item", ops=("MENU",), shortcut="⌘N"),
            UIElement(2, "AXMenuItem", None, "File › Close Window", kind="menu_item", ops=("MENU",), shortcut="⌘W"),
        ],
        running=[AppInfo("Safari", "com.apple.Safari", pid=5), AppInfo("TextEdit", "com.apple.TextEdit", pid=6)],
    )
    r = router()
    text = TextSource({"q": "red flowers"})
    space = r.space(obs, text, "search the web for pictures of red flowers")
    assert "CLICK" not in space.operations and "TYPE_TEXT" not in space.operations
    assert set(space.targets_for("PRESS_KEY")) == {"CMD_N", "CMD_T"}
    assert [t.element.label for t in space.targets_for("MENU").values()] == ["File › New Window"]
    state, questions = r.build_request("search the web", obs, space, text_source=text)
    assert state["elements"] == [] and state["visible_text"] == "" and "never act inside it" in state["desktop"]["note"]
    assert "click_target" not in questions and "type_text_target" not in questions


@pytest.mark.parametrize(
    ("goal", "slots"),
    [
        ("look for pictures of flowers red", {"phrase_1": "pictures of flowers red"}),
        ("search the web for pictures of red flowers", {"phrase_1": "pictures of red flowers"}),
        ("Search for red flowers in Safari", {"phrase_1": "red flowers"}),
        ('In Safari, search for "accessibility API"', {"quote_1": "accessibility API"}),
        ("Open TextEdit, create a new document and type hello world", {"phrase_1": "hello world"}),
        ("go to apple.com and look up the store hours", {"phrase_1": "the store hours", "url_1": "apple.com"}),
        ("google best pizza near me, then open the first result", {"phrase_1": "best pizza near me"}),
        ("Open Notes", {}),
        ("Open Downloads in Finder", {}),
        ("Delete the Trip ideas note in Notes", {}),
    ],
)
def test_goal_phrases_become_choosable_text(goal, slots):
    assert slots_from_goal(goal) == slots


def test_idle_apps_that_fit_the_goal_are_offered_without_being_named():
    chrome = AppInfo("Google Chrome", "com.google.Chrome", pid=5)
    installed = [AppInfo(n, f"com.apple.{n}") for n in ("TextEdit", "Safari", "Mail", "Calculator")]
    obs = observation([], app=chrome, running=[chrome], installed=installed, window="JevOSX Console")

    def offered(goal):
        return {t.app.name for t in router().space(obs, TextSource({}), goal).targets_for("OPEN_APP").values()}

    assert offered("write a poem about autumn") == {"TextEdit"}
    assert offered("search the web for red flowers") == set()  # a browser is already running
    assert offered("what is 17 times 23? calculate it") == {"Calculator"}
    assert offered("Open Mail") == {"Mail"}  # named apps still count
