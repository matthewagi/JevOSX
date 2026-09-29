"""Working behind the person's window (agent.background): the agent keeps its own work window, reads it wherever it
is, and gives the keyboard back to the window the person was using."""

from __future__ import annotations

from typing import Any

import pytest

from jevosx.agent import Agent
from jevosx.config import Settings
from jevosx.errors import StaleElementError
from jevosx.executor.keys import key_vocabulary
from jevosx.observer.base import BackgroundObserver, WindowRef
from jevosx.router.policy import JevRouter
from jevosx.types import CLICK, PRESS_KEY, TYPE_TEXT, Action, ActionResult, AppInfo, Observation, WindowInfo
from tests.fakes import element, observation, scripted_client

CONSOLE = "JevOSX Console - Google Chrome"
NEW_TAB = "New Tab - Google Chrome"


class Win:
    """Stands in for an AX window handle (compared by identity, like CFEqual on the same window)."""

    def __init__(self, title: str):
        self.title = title


class WindowDesktop:
    """Chrome (with the console window) and Terminal. Keyboard actions bring the observed window forward first, as
    MacExecutor does; `on_execute` lets a test play the person switching windows."""

    def __init__(self) -> None:
        self.chrome = AppInfo("Google Chrome", "com.google.Chrome", pid=300)
        self.terminal = AppInfo("Terminal", "com.apple.Terminal", pid=50)
        self.console = Win(CONSOLE)
        self.shell = Win("zsh")
        self.windows: dict[int, list[Win]] = {300: [self.console], 50: [self.shell]}
        self.front_pid, self.front_window = 300, self.console
        self.typed: dict[str, str] = {}
        self.observed: list[str] = []
        self.fronts: list[str] = []  # every bring_forward, by window title
        self.on_execute: Any = None

    # Observer + BackgroundObserver
    def front(self) -> WindowRef:
        return WindowRef(self.front_pid, self.front_window, self.front_window.title)

    def observe(self) -> Observation:
        return self._observe(self.front_pid, self.front_window)

    def observe_window(self, target: WindowRef) -> Observation:
        window = target.window or self.windows[target.pid][-1]
        if window not in self.windows[target.pid]:
            raise StaleElementError(f"the work window “{target.title}” is gone")
        return self._observe(target.pid, window)

    def _observe(self, pid: int, window: Win) -> Observation:
        self.observed.append(window.title)
        app = self.chrome if pid == 300 else self.terminal
        if pid == 300:
            field = element(
                1,
                "AXTextField",
                "Address and search bar",
                kind="text_input",
                ops=("TYPE_TEXT", "CLICK"),
                value=self.typed.get(window.title, ""),
                in_web_area=False,
            )
            elements = [field, element(2, "AXButton", "Reload")]
        else:
            elements = [element(1, "AXTextArea", "shell", kind="text_input", ops=("TYPE_TEXT", "CLICK"))]
        obs = observation(elements, app=app, window=window.title, running=[self.chrome, self.terminal])
        assert obs.window is not None
        obs.window.node = window
        obs.windows = [WindowInfo(i, w.title, w is window, node=w) for i, w in enumerate(self.windows[pid], 1)]
        return obs

    def quick_signature(self) -> str:
        return self.front_window.title

    def quick_signature_of(self, target: WindowRef) -> str:
        return target.title

    def find_app(self, query: str) -> AppInfo | None:
        return next((a for a in (self.chrome, self.terminal) if a.name.lower() == query.lower()), None)

    # Executor + WindowFocuser
    def validate(self, action: Action, obs: Observation) -> None:
        return None

    def execute(self, action: Action, obs: Observation) -> ActionResult:
        assert obs.window is not None and obs.app.pid is not None
        window = obs.window.node
        if action.operation in (PRESS_KEY, TYPE_TEXT):
            self.bring_forward(WindowRef(obs.app.pid, window, window.title))
        if action.operation == PRESS_KEY and action.key is not None and action.key.id == "CMD_N":
            new = Win(NEW_TAB)
            self.windows[300].append(new)
            self.front_pid, self.front_window = 300, new
        elif action.operation == TYPE_TEXT and action.text is not None:
            self.typed[window.title] = action.text
        if self.on_execute is not None:
            self.on_execute(action)
        return ActionResult(True, "fake")

    def bring_forward(self, target: WindowRef) -> bool:
        if target.pid == self.front_pid and target.window in (None, self.front_window):
            return True
        self.fronts.append(target.title)
        self.front_pid = target.pid
        self.front_window = target.window or self.windows[target.pid][-1]
        return True

    def open_app(self, app: AppInfo) -> ActionResult:
        assert app.pid is not None
        self.front_pid, self.front_window = app.pid, self.windows[app.pid][-1]
        return ActionResult(True, "fake")


def browse(body: dict[str, Any]) -> dict[str, Any]:
    """Jev in the console: open a window, type the address, press Reload, then done."""
    desktop = body["state"]["desktop"]
    values = [e.get("value") for e in body["state"]["elements"]]
    if desktop.get("window") == CONSOLE:
        return {"operation": ("PRESS_KEY", 0.9), "key_target": "CMD_N"}
    if "facebook.com" not in values:
        return {"operation": "TYPE_TEXT"}
    if not any(h["action"].startswith("CLICK") for h in body["state"].get("recent_actions", [])):
        return {"operation": "CLICK", "click_target": "2"}
    return {"operation": "DONE"}


def make_agent(desktop: WindowDesktop, *, background: bool = True) -> Agent:
    settings = Settings()
    settings.agent.fallback_log = ""
    settings.agent.background = background
    return Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(browse), keys=key_vocabulary()),
        settings=settings,
        sleep=lambda _s: None,
    )


def test_the_agent_works_in_its_own_window_and_gives_the_console_back():
    desktop = WindowDesktop()
    assert isinstance(desktop, BackgroundObserver)
    agent = make_agent(desktop)
    result = agent.run("go to facebook")
    assert result.status == "done", result.message
    assert desktop.typed == {NEW_TAB: "facebook.com"}  # never into the console
    # The first look is at the console; every later one is at the agent's own window, although the console is in
    # front again after each step.
    assert desktop.observed[0] == CONSOLE and set(desktop.observed[1:]) == {NEW_TAB}
    assert desktop.front_window is desktop.console
    assert desktop.fronts == [CONSOLE, NEW_TAB, CONSOLE]  # after Cmd-N, and around the typing; the click needs none


def test_approving_in_the_console_or_switching_to_terminal_does_not_move_the_work():
    desktop = WindowDesktop()

    def person_switches(action: Action) -> None:
        if action.operation == PRESS_KEY:  # right after the new window opens, the person goes to Terminal
            desktop.front_pid, desktop.front_window = 50, desktop.shell

    desktop.on_execute = person_switches
    agent = make_agent(desktop)
    result = agent.run("go to facebook")
    assert result.status == "done", result.message
    assert desktop.typed == {NEW_TAB: "facebook.com"}
    assert "zsh" not in desktop.observed[1:]  # Terminal is the person's, not work
    assert desktop.front_window is desktop.shell  # typing borrowed the keyboard and gave Terminal back


def test_a_switch_the_person_makes_during_a_background_click_is_not_adopted():
    desktop = WindowDesktop()

    def person_switches(action: Action) -> None:
        if action.operation == CLICK:
            desktop.front_pid, desktop.front_window = 50, desktop.shell

    desktop.on_execute = person_switches
    agent = make_agent(desktop)
    assert agent.run("go to facebook").status == "done"
    assert agent.work is not None and agent.work.title == NEW_TAB


def test_without_background_the_agent_follows_the_front_window_as_before():
    desktop = WindowDesktop()
    agent = make_agent(desktop, background=False)
    assert agent.run("go to facebook").status == "done"
    assert desktop.typed == {NEW_TAB: "facebook.com"}
    assert CONSOLE not in desktop.fronts and desktop.front_window.title == NEW_TAB


def test_a_closed_work_window_falls_back_to_the_front_window():
    desktop = WindowDesktop()
    agent = make_agent(desktop)
    agent.work = WindowRef(300, Win("closed"), "closed")
    assert agent._observe(True).window is not None and agent.work is None
    assert desktop.observed == [CONSOLE]


# ---- MacExecutor: keys only go to the observed window ---------------------------------------------------------------
class Node:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def set(self, attribute: str, value: Any) -> None:
        self.values[attribute] = value

    def get(self, attribute: str, default: Any = None) -> Any:
        return self.values.get(attribute, default)

    def perform(self, action: str) -> None:
        self.values[action] = True


def mac_executor(monkeypatch: pytest.MonkeyPatch, front: list[int]):
    from jevosx.config import ExecutorSettings
    from jevosx.executor import input as keyboard
    from jevosx.executor.mac import MacExecutor

    sent: list[str] = []
    monkeypatch.setattr(keyboard, "post_chord", lambda chord, delay_s=0: sent.append(str(chord)))
    monkeypatch.setattr(keyboard, "type_text", lambda text, delay_s=0: sent.append(text))
    executor = object.__new__(MacExecutor)
    executor.settings = ExecutorSettings(focus_timeout_s=0.05)
    executor._frontmost_pid = lambda: front[0]
    executor._activate = lambda pid: front.__setitem__(0, pid) if front[1:] == ["obeys"] else None
    return executor, sent


def test_keys_are_never_sent_when_the_window_cannot_come_forward(monkeypatch):
    front = [50]  # Terminal stays in front: the activation request is ignored
    executor, sent = mac_executor(monkeypatch, front)
    obs = observation([], app=AppInfo("Google Chrome", "com.google.Chrome", pid=300))
    result = executor.execute(Action(PRESS_KEY, key=key_vocabulary()["RETURN"]), obs)
    assert not result.ok and result.method == "focus" and sent == []


def test_keys_follow_once_the_window_is_in_front(monkeypatch):
    front: list[Any] = [50, "obeys"]
    executor, sent = mac_executor(monkeypatch, front)
    chrome = AppInfo("Google Chrome", "com.google.Chrome", pid=300)
    obs = observation([], app=chrome)
    assert executor.execute(Action(PRESS_KEY, key=key_vocabulary()["RETURN"]), obs).ok and front[0] == 300
    field = element(1, "AXTextField", "Address and search bar", kind="text_input", in_web_area=True, node=Node())
    front[0] = 50
    result = executor.execute(Action(TYPE_TEXT, element=field, text="facebook.com"), obs)
    assert front[0] == 300 and sent[-2:] == ["cmd+a", "facebook.com"]
    assert field.node.values["AXFocused"] is True and not result.ok  # the fake field does not echo keystrokes


def test_field_writes_need_no_focus(monkeypatch):
    front = [50]  # the person stays in Terminal
    executor, sent = mac_executor(monkeypatch, front)
    executor.settings.typing_mode = "ax"
    notes = element(1, "AXTextArea", "Body", kind="text_input", value_settable=True, node=Node())
    obs = observation([notes], app=AppInfo("Notes", "com.apple.Notes", pid=70))
    result = executor.execute(Action(TYPE_TEXT, element=notes, text="hello"), obs)
    assert result.ok and result.method == "AXValue" and sent == [] and front == [50]


def test_the_mac_observer_and_executor_can_work_behind():
    from jevosx.executor.base import WindowFocuser
    from jevosx.executor.mac import MacExecutor
    from jevosx.observer.desktop import MacDesktopObserver

    assert issubclass(MacDesktopObserver, BackgroundObserver) and issubclass(MacExecutor, WindowFocuser)


def test_the_person_is_never_pulled_back_from_a_window_they_chose():
    desktop = WindowDesktop()
    agent = make_agent(desktop)
    tab = Win(NEW_TAB)
    desktop.windows[300].append(tab)
    agent.work = WindowRef(300, tab, NEW_TAB)
    in_terminal = WindowRef(50, desktop.shell, "zsh")
    desktop.front_pid, desktop.front_window = 300, desktop.console  # during a background click they opened the console
    agent._after_action(in_terminal, acted_pid=300, known={300, 50})
    assert desktop.fronts == [] and agent.work.window is tab
