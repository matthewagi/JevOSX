"""Test doubles: fake AX nodes, a scripted fake desktop, and a mocked Jev endpoint."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from jevosx.errors import StaleElementError
from jevosx.router.client import JevClient
from jevosx.types import Action, ActionResult, AppInfo, Observation, UIElement, WindowInfo


class FakeNode:
    """Implements the AXNodeLike protocol over plain Python data."""

    def __init__(
        self,
        role: str,
        *,
        children: Sequence[FakeNode] = (),
        actions: Sequence[str] = (),
        settable: Sequence[str] = (),
        frame: tuple[float, float, float, float] | None = None,
        stale: bool = False,
        **attrs: Any,
    ):
        self.attrs: dict[str, Any] = {"AXRole": role}
        for key, value in attrs.items():
            self.attrs[key if key.startswith("AX") else "AX" + key] = value
        if frame is not None:
            self.attrs["AXPosition"] = (frame[0], frame[1])
            self.attrs["AXSize"] = (frame[2], frame[3])
        self.kids = list(children)
        self._actions = tuple(actions)
        self._settable = set(settable)
        self.stale = stale
        self.performed: list[str] = []
        self.writes: list[tuple[str, Any]] = []

    def _check(self) -> None:
        if self.stale:
            raise StaleElementError("fake node is stale")

    def get(self, attribute: str, default: Any = None) -> Any:
        self._check()
        if attribute == "AXChildren":
            return list(self.kids)
        return self.attrs.get(attribute, default)

    def get_many(self, attributes: Sequence[str]) -> dict[str, Any]:
        self._check()
        return {name: self.get(name) for name in attributes}

    def children(self, attribute: str = "AXChildren", limit: int | None = None) -> list[Any]:
        self._check()
        kids = self.kids if attribute == "AXChildren" else list(self.attrs.get(attribute) or [])
        return kids[:limit] if limit is not None else kids

    def actions(self) -> tuple[str, ...]:
        self._check()
        return self._actions

    def settable(self, attribute: str) -> bool:
        self._check()
        return attribute in self._settable

    def perform(self, action: str) -> None:
        self._check()
        self.performed.append(action)

    def set(self, attribute: str, value: Any) -> None:
        self._check()
        self.writes.append((attribute, value))
        self.attrs[attribute] = value


def element(index: int, role: str, label: str, *, ops: tuple[str, ...] = ("CLICK",), **kwargs: Any) -> UIElement:
    return UIElement(index=index, role=role, subrole=kwargs.pop("subrole", None), label=label, ops=ops, **kwargs)


def observation(
    elements: list[UIElement],
    *,
    app: AppInfo | None = None,
    window: str = "Untitled",
    text: str = "",
    menu_items: list[UIElement] | None = None,
    running: list[AppInfo] | None = None,
    installed: list[AppInfo] | None = None,
    windows: list[WindowInfo] | None = None,
) -> Observation:
    app = app or AppInfo("TextEdit", "com.apple.TextEdit", pid=100)
    focused = WindowInfo(1, window, focused=True)
    return Observation(
        app=app,
        window=focused,
        windows=windows or [focused],
        elements=elements,
        menu_items=menu_items or [],
        scroll_areas=[],
        text=text,
        running_apps=running if running is not None else [app],
        installed_apps=installed or [],
    )


class FakeDesktop:
    """Observer + Executor over a tiny state machine: `screens[name]` builds an observation; `transitions` maps
    (screen, action description substring) → next screen."""

    def __init__(
        self, screens: dict[str, Callable[[], Observation]], start: str, transitions: dict[tuple[str, str], str]
    ):
        self.screens = screens
        self.screen = start
        self.transitions = transitions
        self.executed: list[str] = []
        self.opened: list[str] = []
        self.apps = [AppInfo("TextEdit", "com.apple.TextEdit", pid=100), AppInfo("Notes", "com.apple.Notes", pid=None)]

    # Observer
    def observe(self) -> Observation:
        return self.screens[self.screen]()

    def quick_signature(self) -> str:
        return self.screen

    def find_app(self, query: str) -> AppInfo | None:
        return next((a for a in self.apps if a.name.lower() == query.lower()), None)

    # Executor
    def validate(self, action: Action, obs: Observation) -> None:
        return None

    def execute(self, action: Action, obs: Observation) -> ActionResult:
        description = action.describe()
        if action.text is not None:
            description += f" <- {action.text}"
        self.executed.append(description)
        for (screen, needle), target in self.transitions.items():
            if screen == self.screen and needle in description:
                self.screen = target
                break
        return ActionResult(True, "fake")

    def open_app(self, app: AppInfo) -> ActionResult:
        self.opened.append(app.name)
        return ActionResult(True, "fake")


Answerer = Callable[[dict[str, Any]], dict[str, Any]]


def distribution(ids: Sequence[str], choice: str, p: float = 0.9, confidence: float | None = None) -> dict[str, Any]:
    rest = (1 - p) / (len(ids) - 1) if len(ids) > 1 else 0.0
    probabilities = {i: (p if i == choice else rest) for i in ids}
    if len(ids) == 1:
        probabilities = {choice: 1.0}
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": probabilities,
        "confidence": p if confidence is None else confidence,
    }


def scripted_client(answer: Answerer, requests: list[dict[str, Any]] | None = None) -> JevClient:
    """A JevClient whose HTTP transport calls `answer(request_body)` → {question: choice_id}."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if requests is not None:
            requests.append(body)
        picks = answer(body)
        answers = {}
        for name, question in body["questions"].items():
            ids = list(question["criteria"])
            pick = picks.get(name, ids[0])
            choice, confidence = pick if isinstance(pick, tuple) else (pick, 0.9)  # ("CLICK", 0.4) → confidence
            answers[name] = distribution(ids, choice, 0.9, confidence)
        return httpx.Response(200, json={"model": "jev-test", "answers": answers, "usage": {"input_tokens": 1}})

    return JevClient("test-key", transport=httpx.MockTransport(handler), http2=False)


def find_id(body: dict[str, Any], head: str, needle: str) -> str:
    """Return the criterion id in `head` whose description contains `needle`."""
    for key, criterion in body["questions"][head]["criteria"].items():
        if needle in json.dumps(criterion, ensure_ascii=False):
            return key
    raise AssertionError(f"{needle!r} not offered in {head}")
