"""The dynamic action space: which operations are possible right now, and which targets each one may use.

Every observation yields a fresh space. Only supported operations are offered, each operation has its own target
question containing only compatible targets, and every target id resolves to a handle observed on this Mac.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..executor.keys import KeyBinding
from ..images import wants_images
from ..types import (
    ASK_USER,
    BLOCKED,
    CLICK,
    CONSOLE_SAFE_KEYS,
    CONSOLE_SAFE_MENU,
    DONE,
    FOCUS_WINDOW,
    MENU,
    OPEN_APP,
    PRESS_KEY,
    SAVE_IMAGE,
    SCROLL_DOWN,
    SCROLL_UP,
    TYPE_TEXT,
    WAIT,
    AppInfo,
    Observation,
    UIElement,
    WindowInfo,
    is_console_window,
    normalize_key,
)

HEADS = {
    CLICK: "click_target",
    TYPE_TEXT: "type_text_target",
    MENU: "menu_target",
    PRESS_KEY: "key_target",
    SCROLL_UP: "scroll_target",
    SCROLL_DOWN: "scroll_target",
    OPEN_APP: "app_target",
    FOCUS_WINDOW: "window_target",
    ASK_USER: "handoff_reason",
    SAVE_IMAGE: "image_target",
}
OPERATION_TEXT = {
    CLICK: "Press or click an on-screen element: button, link, checkbox, tab, row, open-menu item, or focus a field.",
    TYPE_TEXT: "Replace the text in an editable field with prepared text.",
    MENU: "Run a command from the app's menu bar (File, Edit, View, …) without opening the menu.",
    PRESS_KEY: "Press a keyboard key or shortcut in the frontmost app.",
    SCROLL_DOWN: "Scroll a scrollable area down to reveal more content.",
    SCROLL_UP: "Scroll a scrollable area up.",
    OPEN_APP: "Open or switch to another application.",
    FOCUS_WINDOW: "Bring another window of the frontmost app to the front.",
    ASK_USER: "Hand control to the user for a step only they can do here (a verification code, a CAPTCHA, a passkey "
    "or Touch ID prompt, or information the goal does not give). The run continues after they finish.",
    SAVE_IMAGE: "Save a picture shown on the page into the folder the goal asks for (one step: no dialogs).",
    WAIT: "Wait briefly for loading or an animation to finish.",
    DONE: "Every part of the goal is visibly complete.",
    BLOCKED: "No offered operation can make progress (missing information, permission, or control).",
}

# Why the user is needed (ASK_USER). A typed choice: Jev picks the reason, it never writes the message.
HANDOFF_REASONS = {
    "code": "type a verification, two-factor or one-time code",
    "captcha": "solve a CAPTCHA or an 'are you human' check",
    "approve": "approve a passkey, Touch ID, security key or system password prompt",
    "login": "sign in (no saved login fits this site)",
    "info": "provide information the goal does not include",
    "other": "do something else only the user can do",
}

_WORD = re.compile(r"\w+")

# Idle apps offered because they fit what the goal asks for, although it does not name them ("write a poem" → TextEdit).
# A group only adds apps when none of its apps is already running, so a running browser is not joined by others.
INTENT_APPS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), apps)
    for pattern, apps in (
        (r"\b(?:write|poem|haiku|story|essay|letter|document|draft|type up)\b", ("TextEdit", "Pages", "Notes")),
        (r"\b(?:note|notes|jot)\b", ("Notes",)),
        (r"\b(?:e-?mail|inbox)\b", ("Mail",)),
        (
            r"\b(?:search|browse|website|google|look up|look for|online|log ?in|sign ?in)\b|\.(?:com|org|net|io)\b",
            ("Safari", "Google Chrome", "Firefox", "Arc", "Microsoft Edge", "Brave Browser"),
        ),
        (r"\b(?:calendar|meeting|appointment)\b", ("Calendar",)),
        (r"\b(?:remind me|reminders?|to-?do)\b", ("Reminders",)),
        (r"\b(?:spreadsheet|budget)\b", ("Numbers",)),
        (r"\b(?:presentation|slides|keynote)\b", ("Keynote",)),
        (r"\b(?:song|playlist|album)\b", ("Music",)),
        (r"\b(?:imessage|text message)\b", ("Messages",)),
        (r"\b(?:calculate|calculator)\b", ("Calculator",)),
        (r"\b(?:directions|map of)\b", ("Maps",)),
        (r"\b(?:folder|downloads|finder)\b", ("Finder",)),
        (r"\b(?:terminal|shell command|command line)\b", ("Terminal",)),
        (r"\b(?:wi-?fi|bluetooth|system settings|dark mode)\b", ("System Settings",)),
        (r"\b(?:timer|alarm|stopwatch)\b", ("Clock",)),
    )
)


def intent_apps(goal: str, running: set[str]) -> set[str]:
    """Names of idle apps worth offering for this goal (see INTENT_APPS)."""
    wanted: set[str] = set()
    pictures = wants_images(goal)
    for pattern, apps in INTENT_APPS:
        if pictures and apps == ("Finder",):
            continue  # pictures are saved into their folder directly (SAVE_IMAGE): seen live opening Finder twice
        if pattern.search(goal) and not running & {a.lower() for a in apps}:
            wanted.update(a.lower() for a in apps)
    return wanted


@dataclass(slots=True)
class Target:
    id: str
    criterion: dict[str, Any]
    memory_key: str
    element: UIElement | None = None
    app: AppInfo | None = None
    window: WindowInfo | None = None
    key: KeyBinding | None = None

    def describe(self) -> str:
        if self.element is not None:
            prefix = "m" if self.element.kind == "menu_item" else "s" if self.element.kind == "scroll_area" else ""
            return self.element.describe(prefix)
        if self.app is not None:
            return self.app.name
        if self.window is not None:
            return f'window "{self.window.title}"'
        if self.key is not None:
            return f"{self.key.id} ({self.key.chord})"
        if "need" in self.criterion:
            return f"to {self.criterion['need']}"
        return self.id


@dataclass
class ActionSpace:
    operations: dict[str, str]
    heads: dict[str, dict[str, Target]]
    _by_memory_key: dict[tuple[str, str], Target] = field(default_factory=dict, repr=False)

    def head_for(self, operation: str) -> str | None:
        head = HEADS.get(operation)
        return head if head in self.heads else None

    def targets_for(self, operation: str) -> dict[str, Target]:
        head = self.head_for(operation)
        return self.heads[head] if head else {}

    def resolve(self, operation: str, memory_key: str) -> Target | None:
        """Map a remembered target onto what is on screen now (None when it is not currently offered)."""
        head = HEADS.get(operation)
        return self._by_memory_key.get((head, memory_key)) if head else None

    def drop_elements(self, indices: set[int]) -> None:
        """Remove element targets whose rows were cut from the state (keeps ids == element table indices)."""
        keep = {str(i) for i in indices}
        for op in (CLICK, TYPE_TEXT, SAVE_IMAGE):
            head = HEADS[op]
            if head in self.heads:
                self.heads[head] = {k: t for k, t in self.heads[head].items() if k not in keep}
                if not self.heads[head]:
                    del self.heads[head]
                    self.operations.pop(op, None)
        self._by_memory_key = {k: t for k, t in self._by_memory_key.items() if not (t.element and t.id in keep)}

    def annotate(self, operation: str, target_id: str, note: str) -> None:
        target = self.targets_for(operation).get(target_id)
        if target is not None:
            target.criterion["memory"] = note

    @classmethod
    def build(
        cls,
        obs: Observation,
        *,
        keys: Mapping[str, KeyBinding],
        text_available: bool,
        max_choices: int = 200,
        goal: str = "",
        offer_installed_apps: str = "mentioned",
        handoff: bool = False,
        images: Collection[str] | None = None,
    ) -> ActionSpace:
        """`images`: the addresses of pictures already saved in this run, or None when the goal saves no pictures
        (then SAVE_IMAGE is not offered)."""
        heads: dict[str, dict[str, Target]] = {}

        def add(head: str, target: Target) -> None:
            bucket = heads.setdefault(head, {})
            if len(bucket) < max_choices:
                bucket[target.id] = target

        # In the JevOSX console window only "new window/tab" and app switching are offered (see types.py).
        console = obs.window is not None and is_console_window(obs.window.title)
        # Focused element first so truncation never drops it.
        ordered = [] if console else sorted(obs.elements, key=lambda e: not e.focused)
        for element in ordered:
            if SAVE_IMAGE in element.ops and images is not None and element.url not in images:
                images = {*images, element.url or ""}  # the same picture shown twice is offered once
                picture: dict[str, Any] = {"picture": element.label}
                if element.frame is not None:
                    picture["size"] = f"{round(element.frame.w)}x{round(element.frame.h)}"
                target = Target(str(element.index), picture, "img:" + element.signature, element=element)
                add(HEADS[SAVE_IMAGE], target)
            if CLICK in element.ops:
                add(HEADS[CLICK], _element_target(str(element.index), element))
            if TYPE_TEXT in element.ops and text_available:
                add(HEADS[TYPE_TEXT], _element_target(str(element.index), element))
        for item in obs.menu_items:
            if console and not CONSOLE_SAFE_MENU.search(item.label):
                continue
            criterion: dict[str, Any] = {"command": item.label}
            if item.shortcut:
                criterion["shortcut"] = item.shortcut
            if item.checked:
                criterion["checked"] = True
            add(HEADS[MENU], Target(f"m{item.index}", criterion, "menu:" + normalize_key(item.label), element=item))
        for area in [] if console else obs.scroll_areas:
            criterion = {"area": area.label}
            if area.container:
                criterion["in"] = area.container
            add(
                HEADS[SCROLL_DOWN],
                Target(f"s{area.index}", criterion, "scroll:" + normalize_key(area.label), element=area),
            )
        for binding in keys.values():
            if console and binding.id not in CONSOLE_SAFE_KEYS:
                continue
            add(
                HEADS[PRESS_KEY],
                Target(
                    binding.id,
                    {"key": str(binding.chord), "effect": binding.description},
                    "key:" + binding.id,
                    key=binding,
                ),
            )
        position = 0
        running_names = {a.name.lower() for a in obs.running_apps}
        suggested = intent_apps(goal, running_names) if offer_installed_apps == "mentioned" else set()
        for app in [*obs.running_apps, *obs.installed_apps]:
            if app.key == obs.app.key or (app.pid is not None and app.pid == obs.app.pid):
                continue
            if not app.running and not (
                _offer_installed(app, goal, offer_installed_apps) or app.name.lower() in suggested
            ):
                continue  # context guardrail: dozens of idle apps would only cost tokens
            if any(t.app is not None and t.app.key == app.key for t in heads.get(HEADS[OPEN_APP], {}).values()):
                continue
            position += 1
            status = "running" if app.running else "not running"
            add(HEADS[OPEN_APP], Target(f"a{position}", {"app": app.name, "status": status}, "app:" + app.key, app=app))
        for window in obs.windows:
            same_as_focused = obs.window is not None and window.node is not None and window.node == obs.window.node
            if window.focused or same_as_focused:
                continue
            criterion = {"window": window.title}
            if window.minimized:
                criterion["minimized"] = True
            add(
                HEADS[FOCUS_WINDOW],
                Target(f"w{window.index}", criterion, "win:" + normalize_key(window.title), window=window),
            )

        if handoff and not console:
            for reason, need in HANDOFF_REASONS.items():
                add(HEADS[ASK_USER], Target(reason, {"need": need}, "ask:" + reason))

        operations: dict[str, str] = {}
        for op in (SAVE_IMAGE, CLICK, TYPE_TEXT, MENU, PRESS_KEY, SCROLL_DOWN, SCROLL_UP, OPEN_APP, FOCUS_WINDOW):
            if heads.get(HEADS[op]):
                operations[op] = OPERATION_TEXT[op]
        if heads.get(HEADS[ASK_USER]):
            operations[ASK_USER] = OPERATION_TEXT[ASK_USER]
        for op in (WAIT, DONE, BLOCKED):
            operations[op] = OPERATION_TEXT[op]

        index: dict[tuple[str, str], Target] = {}
        for head, bucket in heads.items():
            for target in bucket.values():
                index.setdefault((head, target.memory_key), target)  # first (focused / earliest) wins
        return cls(operations=operations, heads=heads, _by_memory_key=index)


def _offer_installed(app: AppInfo, goal: str, mode: str) -> bool:
    """Installed-but-idle apps are offered only when the goal names them (mode "mentioned")."""
    if mode == "all":
        return True
    if mode == "none" or not goal:
        return False
    words = f" {' '.join(_WORD.findall(goal.lower()))} "
    name = " ".join(_WORD.findall(app.name.lower()))
    return bool(name) and f" {name} " in words or bool(app.bundle_id and app.bundle_id.lower() in goal.lower())


def _element_target(target_id: str, element: UIElement) -> Target:
    criterion: dict[str, Any] = {"element": element.describe()}
    if element.container:
        criterion["in"] = element.container
    states = element.states()
    if states:
        criterion["state"] = ", ".join(states)
    return Target(target_id, criterion, "el:" + element.signature, element=element)
