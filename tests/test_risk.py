"""Risk tiers: easily undone steps need less confidence than clicks, and consequential ones keep the full floor."""

from __future__ import annotations

import pytest

from jevosx.agent import Agent
from jevosx.config import AgentSettings, Settings
from jevosx.executor.keys import key_vocabulary
from jevosx.executor.safety import SafetyPolicy
from jevosx.memory.retriever import Hint
from jevosx.risk import CAREFUL, ROUTINE, SAFE, assess, floors
from jevosx.router.policy import JevRouter
from jevosx.types import (
    ASK_USER,
    CLICK,
    DONE,
    MENU,
    OPEN_APP,
    PRESS_KEY,
    SCROLL_DOWN,
    TYPE_TEXT,
    Action,
    AppInfo,
    UIElement,
)
from tests.fakes import FakeDesktop, element, find_id, observation, scripted_client

KEYS = key_vocabulary()
CHROME = AppInfo("Google Chrome", "com.google.Chrome", pid=300)


def tier(action: Action) -> str:
    obs = observation([], app=CHROME, window="New Tab - Google Chrome")
    return assess(action, obs, settings=AgentSettings(), safety=SafetyPolicy()).tier


address = element(1, "AXTextField", "Address and search bar", kind="text_input", ops=("TYPE_TEXT", "CLICK"))
document = element(2, "AXTextArea", "Body", kind="text_input", value="Dear Anna, …", ops=("TYPE_TEXT", "CLICK"))
link = element(3, "AXLink", "Marketplace")
publish = element(4, "AXButton", "Publish")
reload = element(6, "AXButton", "Reload")
new_window = UIElement(5, "AXMenuItem", None, "File › New Window", kind="menu_item", ops=("MENU",))


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (Action(OPEN_APP, app=CHROME), SAFE),
        (Action(PRESS_KEY, key=KEYS["CMD_L"]), SAFE),
        (Action(PRESS_KEY, key=KEYS["CMD_N"]), SAFE),
        (Action(CLICK, element=address), SAFE),  # puts the cursor in the field
        (Action(TYPE_TEXT, element=address), SAFE),  # a single-line field is retyped in a moment
        (Action(SCROLL_DOWN, element=link), SAFE),
        (Action(MENU, element=new_window), SAFE),
        (Action(ASK_USER), SAFE),
        (Action(CLICK, element=link), ROUTINE),
        (Action(PRESS_KEY, key=KEYS["RETURN"]), ROUTINE),
        (Action(TYPE_TEXT, element=document), ROUTINE),  # would replace what is already written
        (Action(DONE), ROUTINE),
        (Action(CLICK, element=publish), CAREFUL),  # the safety policy asks about publishing
        (Action(PRESS_KEY, key=KEYS["CMD_W"]), CAREFUL),
        (Action(PRESS_KEY, key=KEYS["CMD_Q"]), CAREFUL),
    ],
)
def test_steps_are_sorted_by_what_a_mistake_would_cost(action, expected):
    assert tier(action) == expected


def test_no_tier_floor_is_above_the_main_floor():
    assert floors(AgentSettings()) == {SAFE: 0.35, ROUTINE: 0.5, CAREFUL: 0.65}
    assert floors(AgentSettings(min_confidence=0.3)) == {SAFE: 0.3, ROUTINE: 0.3, CAREFUL: 0.3}
    assert floors(AgentSettings(min_confidence=0.9))[CAREFUL] == 0.9


def test_a_step_that_worked_before_is_not_asked_again_unless_it_is_careful():
    obs = observation([link, publish], app=CHROME)
    settings, safety = AgentSettings(), SafetyPolicy()
    worked = [Hint("worked", CLICK, "3", link.describe(), 0.8, {1}), Hint("worked", CLICK, "4", "Publish", 0.8, {1})]
    remembered = assess(Action(CLICK, element=link), obs, settings=settings, safety=safety, hints=worked, target_id="3")
    assert remembered.tier == SAFE and "earlier run" in remembered.reason
    careful = assess(Action(CLICK, element=publish), obs, settings=settings, safety=safety, hints=worked, target_id="4")
    assert careful.tier == CAREFUL
    faint = [Hint("worked", CLICK, "3", link.describe(), 0.4, {1})]  # too different from this run
    unsure = assess(Action(CLICK, element=link), obs, settings=settings, safety=safety, hints=faint, target_id="3")
    assert unsure.tier == ROUTINE


def run(tmp_path, answer, screens, *, confirm):
    desktop = FakeDesktop(screens, "form", {})
    settings = Settings()
    settings.agent.fallback_log = str(tmp_path / "fallbacks.jsonl")
    settings.agent.low_confidence_policy = "ask"
    agent = Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(answer), keys=KEYS),
        settings=settings,
        confirm=confirm,
        sleep=lambda _s: None,
    )
    result = agent.run('go to "facebook.com"', max_steps=1)
    return result, desktop


def test_an_unsure_safe_step_runs_without_asking(tmp_path):
    asked: list[str] = []
    screens = {"form": lambda: observation([address], app=CHROME)}
    result, desktop = run(
        tmp_path,
        lambda body: {"operation": ("TYPE_TEXT", 0.45)},
        screens,
        confirm=lambda action, reason: asked.append(reason) or True,
    )
    assert desktop.executed == ['TYPE_TEXT [1] textfield "Address and search bar" <- facebook.com'] and asked == []
    assert result.events[0].decision["risk"] == SAFE and result.events[0].decision["floor"] == 0.35


def test_an_unsure_click_is_asked_about(tmp_path):
    asked: list[str] = []
    screens = {"form": lambda: observation([link, reload], app=CHROME)}
    run(
        tmp_path,
        lambda body: {"operation": ("CLICK", 0.45), "click_target": find_id(body, "click_target", "Marketplace")},
        screens,
        confirm=lambda action, reason: asked.append(reason) or True,
    )
    assert len(asked) == 1 and "for routine steps" in asked[0]


def test_an_unsure_consequential_step_is_asked_about_once(tmp_path):
    asked: list[str] = []
    screens = {"form": lambda: observation([publish, reload], app=CHROME)}
    result, desktop = run(
        tmp_path,
        lambda body: {"operation": ("CLICK", 0.6), "click_target": find_id(body, "click_target", "Publish")},
        screens,
        confirm=lambda action, reason: asked.append(reason) or True,
    )
    assert len(asked) == 1 and "for careful steps" in asked[0] and "looks consequential" in asked[0]
    assert desktop.executed == ['CLICK [4] button "Publish"']


def test_a_sure_consequential_step_still_needs_your_ok(tmp_path):
    asked: list[str] = []
    screens = {"form": lambda: observation([publish, reload], app=CHROME)}
    _, desktop = run(
        tmp_path,
        lambda body: {"operation": ("CLICK", 0.95), "click_target": (find_id(body, "click_target", "Publish"), 0.95)},
        screens,
        confirm=lambda action, reason: asked.append(reason) or False,
    )
    assert asked == ["'Publish' looks consequential"] and desktop.executed == []
