"""Demo mode: a simulated Mac and a simulated decision model, so the UI can be tried on any OS without an API key.

Everything here is a stand-in and is labelled as such in the UI. The simulated desktop implements the same
Observer/Executor protocols as the real macOS backends, and the simulated "Jev" answers the same typed choice
questions over HTTP (via an in-process transport) with the same response format. Only the decision logic is
a small keyword heuristic instead of the real model. That makes the whole pipeline (router contract,
confidence gate, safety approvals, memory, UI streaming) behave exactly as it does live.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import StaleElementError
from ..observer.vision import KEYBOARD_ROLE, VISION_CONTAINER, VISION_ROLE
from ..router.client import JevClient
from ..router.text import GENERATE, slots_from_goal
from ..types import (
    CLICK,
    MENU,
    OPEN_APP,
    PRESS_KEY,
    SCROLL_DOWN,
    SCROLL_UP,
    TYPE_TEXT,
    Action,
    ActionResult,
    AppInfo,
    Observation,
    UIElement,
    WindowInfo,
)
from ..writer import wants_generation

TEXT_OPS = (TYPE_TEXT, CLICK)


@dataclass
class Spec:
    """Declarative element description for the simulated apps."""

    role: str
    label: str
    value: str | None = None
    container: str | None = None
    subrole: str | None = None
    focused: bool = False
    enabled: bool = True
    selected: bool | None = None
    web: bool = False  # inside the web page (not the browser's toolbar)

    def build(self, index: int) -> UIElement:
        text_input = self.role in ("AXTextField", "AXTextArea", "AXComboBox")
        ops = TEXT_OPS if text_input else ((CLICK,) if self.role != "AXStaticText" else ())
        secure = self.subrole == "AXSecureTextField"
        kind = "text_input" if text_input else "row" if self.role == "AXRow" else "control"
        if self.role == VISION_ROLE:  # stands in for OCR'd text in an app that draws its own interface
            kind, ops = "visual", (CLICK,)
        elif self.role == KEYBOARD_ROLE:
            kind, ops = "keyboard", (TYPE_TEXT,)
        return UIElement(
            index=index,
            role=self.role,
            subrole=self.subrole,
            label=self.label,
            value=None if secure else self.value,
            kind=kind,
            ops=ops if self.enabled else (),
            enabled=self.enabled,
            focused=self.focused,
            selected=self.selected,
            secure=secure,
            filled=bool(self.value) if secure else None,
            container=self.container,
            in_web_area=self.web,
            value_settable=text_input and not secure,
        )


class DemoApp:
    name = "App"
    bundle_id = "demo.app"

    def title(self) -> str:
        return self.name

    def url(self) -> str | None:
        return None

    def specs(self) -> list[Spec]:
        return []

    def menu(self) -> list[tuple[str, str | None]]:
        return [(f"{self.name} › Quit {self.name}", "⌘Q")]

    def text(self) -> str:
        return ""

    def click(self, label: str) -> str:
        return "clicked"

    def type(self, label: str, text: str) -> str:
        return "typed"

    def key(self, key_id: str) -> str:
        return "no effect"

    def command(self, path: str) -> str:
        return "no effect"


class Finder(DemoApp):
    name, bundle_id = "Finder", "com.apple.finder"
    FOLDERS = {
        "Applications": ["Notes.app", "Safari.app", "TextEdit.app", "System Settings.app"],
        "Desktop": ["Screenshot 2026-09-28.png", "Todo.txt"],
        "Documents": ["Budget 2026.numbers", "Project plan.pages", "Resume.pdf"],
        "Downloads": ["installer.dmg", "photo.jpg", "report.pdf"],
    }

    def __init__(self) -> None:
        self.folder = "Documents"
        self.query = ""

    def title(self) -> str:
        return self.folder

    def specs(self) -> list[Spec]:
        sidebar = [
            Spec("AXRow", name, container='outline "Sidebar"', selected=name == self.folder) for name in self.FOLDERS
        ]
        files = [f for f in self.FOLDERS[self.folder] if self.query.lower() in f.lower()]
        return [
            Spec("AXButton", "Back", container="toolbar"),
            Spec("AXButton", "New Folder", container="toolbar"),
            Spec("AXTextField", "Search", value=self.query, subrole="AXSearchField", container="toolbar"),
            *sidebar,
            *(Spec("AXRow", f, container=f'list "{self.folder}"') for f in files),
        ]

    def menu(self) -> list[tuple[str, str | None]]:
        return [("File › New Finder Window", "⌘N"), ("File › New Folder", "⇧⌘N"), ("Go › Downloads", "⌥⌘L")]

    def text(self) -> str:
        return f"{len(self.FOLDERS[self.folder])} items"

    def click(self, label: str) -> str:
        if label in self.FOLDERS:
            self.folder, self.query = label, ""
            return f"opened {label}"
        return f"selected {label}"

    def type(self, label: str, text: str) -> str:
        self.query = text
        return "filtered"

    def command(self, path: str) -> str:
        if path.endswith("Downloads"):
            self.folder = "Downloads"
        return "ok"


class TextEdit(DemoApp):
    name, bundle_id = "TextEdit", "com.apple.TextEdit"

    def __init__(self) -> None:
        self.document: str | None = None  # None → the Open panel is showing
        self.body = ""
        self.saving = False
        self.save_name = "Untitled"
        self.bold = False

    def title(self) -> str:
        return "Open" if self.document is None else self.document

    def specs(self) -> list[Spec]:
        if self.document is None:
            return [
                Spec("AXRow", "Notes.rtf", container='list "iCloud Drive"'),
                Spec("AXButton", "New Document"),
                Spec("AXButton", "Open", enabled=False),
                Spec("AXButton", "Cancel"),
            ]
        specs = [
            Spec("AXPopUpButton", "Paragraph Styles", value="Body", container="toolbar"),
            Spec("AXCheckBox", "Bold", container="toolbar", selected=self.bold),
            Spec("AXCheckBox", "Italic", container="toolbar"),
            Spec("AXTextArea", "Body", value=self.body, focused=not self.saving),
        ]
        if self.saving:
            specs += [
                Spec("AXTextField", "Save As", value=self.save_name, container='sheet "Save"', focused=True),
                Spec("AXButton", "Save", container='sheet "Save"'),
                Spec("AXButton", "Cancel", container='sheet "Save"'),
            ]
        return specs

    def menu(self) -> list[tuple[str, str | None]]:
        return [
            ("File › New", "⌘N"),
            ("File › Open…", "⌘O"),
            ("File › Save…", "⌘S"),
            ("Edit › Select All", "⌘A"),
            ("Format › Font › Bold", "⌘B"),
            ("Format › Make Plain Text", "⇧⌘T"),
        ]

    def text(self) -> str:
        return self.body

    def _new(self) -> str:
        self.document, self.body, self.save_name = "Untitled", "", "Untitled"
        return "new document"

    def click(self, label: str) -> str:
        if label == "New Document":
            return self._new()
        if label == "Bold":
            self.bold = not self.bold
            return "toggled bold"
        if label == "Save" and self.saving:
            self.saving, self.document = False, self.save_name
            return f"saved as {self.save_name}"
        if label == "Cancel":
            if self.saving:
                self.saving = False
            return "cancelled"
        return "clicked"

    def type(self, label: str, text: str) -> str:
        if label == "Save As":
            self.save_name = text
        else:
            self.body = text
        return "typed"

    def key(self, key_id: str) -> str:
        if key_id == "CMD_S":
            return self.command("File › Save…")
        if key_id == "CMD_N":
            return self._new()
        if key_id == "RETURN" and self.saving:
            return self.click("Save")
        return "no effect"

    def command(self, path: str) -> str:
        if path == "File › New":
            return self._new()
        if path == "File › Save…" and self.document is not None:
            self.saving = True
            return "save sheet"
        if path == "Format › Font › Bold":
            return self.click("Bold")
        return "no effect"


class Safari(DemoApp):
    name, bundle_id = "Safari", "com.apple.Safari"
    FAVORITES = ("Apple", "Wikipedia", "GitHub")

    DEMO_LOGIN = ("octocat", "demo-password-123")  # matches the demo console's simulated Keychain entry

    def __init__(self) -> None:
        self.address = ""
        self.page = "Start Page"
        self.results: list[str] = []
        self.login = ""  # "" | form | 2fa | done: the simulated github.com sign-in
        self.username = ""
        self.password = ""
        self.error = ""
        self.listing: dict[str, str] | None = None  # the simulated Marketplace "item for sale" form
        self.published = False

    def title(self) -> str:
        return self.page

    def url(self) -> str | None:
        if self.listing is not None:
            return "https://www.facebook.com/marketplace/create/item"
        if self.login == "form":
            return "https://github.com/login"
        if self.login == "2fa":
            return "https://github.com/sessions/two-factor/app"
        if self.login == "done":
            return "https://github.com/"
        if self.results:
            return f"https://www.google.com/search?q={self.address.replace(' ', '+')}"
        return None if self.page == "Start Page" else f"https://{self.page}/"

    def _sign_in_specs(self) -> list[Spec]:
        page = f'web page "{self.page}"'
        if self.login == "form":
            return [
                Spec("AXTextField", "Username or email address", value=self.username, container=page, web=True),
                Spec(
                    "AXTextField",
                    "Password",
                    subrole="AXSecureTextField",
                    value=self.password,
                    container=page,
                    web=True,
                ),
                Spec("AXButton", "Sign in", container=page, web=True),
                Spec("AXLink", "Forgot password?", container=page, web=True),
            ]
        if self.login == "2fa":
            return [
                Spec("AXTextField", "Authentication code", container=page, web=True),
                Spec("AXButton", "Verify", container=page, web=True),
            ]
        return [Spec("AXLink", name, container=page, web=True) for name in ("Repositories", "Pull requests", "Issues")]

    def _listing_specs(self) -> list[Spec]:
        page, form = f'web page "{self.page}"', self.listing or {}
        return [
            Spec("AXButton", "Add photos", container=page, web=True),
            Spec("AXTextField", "Title", value=form.get("Title", ""), container=page, web=True),
            Spec("AXTextField", "Price", value=form.get("Price", ""), container=page, web=True),
            Spec("AXComboBox", "Category", value=form.get("Category", ""), container=page, web=True),
            Spec("AXPopUpButton", "Condition", value="New", container=page, web=True),
            Spec("AXTextArea", "Description", value=form.get("Description", ""), container=page, web=True),
            Spec("AXButton", "Publish", container=page, web=True),
        ]

    def specs(self) -> list[Spec]:
        links = self.results or [f"Favorites: {f}" for f in self.FAVORITES]
        if self.listing is not None:
            content = self._listing_specs()
        else:
            content = self._sign_in_specs() if self.login else [
            Spec("AXLink", link, container=f'web page "{self.page}"', web=True) for link in links
            ]  # fmt: skip
        return [
            Spec("AXButton", "Back", container="toolbar", enabled=self.page != "Start Page"),
            Spec(
                "AXTextField",
                "Search or enter website name",
                value=self.address,
                container="toolbar",
                subrole="AXSearchField",
                focused=self.page == "Start Page",
            ),  # fmt: skip
            Spec("AXButton", "Share", container="toolbar"),
            *content,
        ]

    def menu(self) -> list[tuple[str, str | None]]:
        return [("File › New Tab", "⌘T"), ("File › New Window", "⌘N"), ("History › Home", "⇧⌘H")]

    def text(self) -> str:
        if self.listing is not None:
            state = "Published" if self.published else "Draft, not published"
            return f"Marketplace · Item for sale · {state} · Photos are required before publishing"
        if self.login == "form":
            return "Sign in to GitHub" + (f" · {self.error}" if self.error else "")
        if self.login == "2fa":
            return "Two-factor authentication · Open your authenticator app and enter the 6-digit code"
        if self.login == "done":
            return f"Signed in as {self.username} · Dashboard · Recent activity"
        if self.page == "Start Page":
            return "Favorites · Frequently Visited · Privacy Report"
        if self.results:
            return f"Results for {self.address} · About 1,240,000 results"
        return f"Welcome to {self.page}"

    def _load(self) -> str:
        query = self.address.strip()
        if not query:
            return "nothing to load"
        if re.fullmatch(r"(?:https?://)?(?:www\.)?facebook\.com/marketplace(?:/\S*)?", query):
            self.page, self.results, self.listing = "Marketplace – Item for sale | Facebook", [], {}
        elif re.fullmatch(r"(?:https?://)?(?:www\.)?github\.com(?:/\S*)?", query):
            self.page, self.results, self.login, self.error = "Sign in to GitHub · GitHub", [], "form", ""
        elif re.fullmatch(r"[\w-]+(\.[\w-]+)+(/\S*)?", query):
            self.page, self.results = query, []
        else:
            self.page = f"{query} - Search"
            self.results = [f"{query} - Wikipedia", f"{query} - News", f"Images for {query}"]
        return f"loaded {self.page}"

    def click(self, label: str) -> str:
        if label == "Back":
            self.__init__()  # type: ignore[misc]
            return "back"
        if label == "Publish" and self.listing is not None:
            self.published = True
            return "published the listing"
        if label == "Sign in" and self.login == "form":
            if (self.username, self.password) == self.DEMO_LOGIN:
                self.login, self.page, self.error = "2fa", "Two-factor authentication · GitHub", ""
                return "asked for a two-factor code"
            self.error = "Incorrect username or password."
            return "sign-in failed"
        if self.login:
            return "clicked"
        name = label.removeprefix("Favorites: ")
        self.page, self.results, self.address = name, [], name.lower().replace(" ", "") + ".com"
        return f"opened {name}"

    def type(self, label: str, text: str) -> str:
        if self.listing is not None and label in ("Title", "Price", "Category", "Description"):
            self.listing[label] = text
        elif label == "Username or email address":
            self.username = text
        elif label == "Password":
            self.password = text
        elif label == "Authentication code":
            return "typed (the code is only on your phone)"
        else:
            self.address = text
        return "typed"

    def key(self, key_id: str) -> str:
        return self._load() if key_id == "RETURN" else "no effect"

    def user_enters_code(self) -> None:
        if self.login == "2fa":
            self.login, self.page = "done", "GitHub"


class Notes(DemoApp):
    name, bundle_id = "Notes", "com.apple.Notes"

    def __init__(self) -> None:
        self.notes = ["Groceries", "Trip ideas"]
        self.current = 0
        self.bodies = {"Groceries": "eggs, milk, coffee", "Trip ideas": "Lisbon in spring"}

    def title(self) -> str:
        return "Notes"

    def specs(self) -> list[Spec]:
        rows = [
            Spec("AXRow", n, container='list "iCloud"', selected=i == self.current) for i, n in enumerate(self.notes)
        ]
        body = self.bodies.get(self.notes[self.current], "") if self.notes else ""
        return [
            Spec("AXButton", "New Note", container="toolbar"),
            Spec("AXButton", "Delete", container="toolbar", enabled=bool(self.notes)),
            Spec("AXTextField", "Search", subrole="AXSearchField", container="toolbar"),
            *rows,
            Spec("AXTextArea", "Note", value=body, focused=True),
        ]

    def menu(self) -> list[tuple[str, str | None]]:
        return [("File › New Note", "⌘N"), ("Edit › Delete", "⌫")]

    def click(self, label: str) -> str:
        if label == "New Note":
            self.notes.insert(0, "New Note")
            self.bodies["New Note"] = ""
            self.current = 0
            return "new note"
        if label == "Delete" and self.notes:
            removed = self.notes.pop(self.current)
            self.current = 0
            return f"deleted {removed}"
        if label in self.notes:
            self.current = self.notes.index(label)
            return f"opened {label}"
        return "clicked"

    def type(self, label: str, text: str) -> str:
        if label == "Note" and self.notes:
            old = self.notes[self.current]
            title = text.splitlines()[0][:40] or "New Note"
            self.bodies.pop(old, None)
            self.notes[self.current] = title
            self.bodies[title] = text
        return "typed"

    def command(self, path: str) -> str:
        return self.click("New Note") if path == "File › New Note" else "no effect"


class SpaceBlocks(DemoApp):
    """A game that draws its own interface: Accessibility sees nothing but the window. On a real Mac the observer
    OCRs such windows (observer/vision.py); here the recognized text is simulated."""

    name, bundle_id = "Space Blocks", "com.example.SpaceBlocks"
    SCREENS = {
        "title": ("SPACE BLOCKS", "New Game", "Continue", "Options", "Quit"),
        "difficulty": ("Choose difficulty", "Easy", "Normal", "Hard", "Back"),
    }

    def __init__(self) -> None:
        self.screen = "title"  # title | difficulty | playing
        self.difficulty = ""
        self.score = 0

    def title(self) -> str:
        return "Space Blocks"

    def texts(self) -> tuple[str, ...]:
        if self.screen == "playing":
            return (f"Level 1 · {self.difficulty}", f"Score: {self.score}", "Lives: 3", "Press SPACE to launch")
        return self.SCREENS[self.screen]

    def specs(self) -> list[Spec]:
        visual = [Spec(VISION_ROLE, text, container=f"{VISION_CONTAINER}, simulated") for text in self.texts()]
        return [*visual, Spec(KEYBOARD_ROLE, "type at the cursor", container="keyboard")]

    def menu(self) -> list[tuple[str, str | None]]:
        return [("Space Blocks › Quit Space Blocks", "⌘Q")]

    def text(self) -> str:
        return "\n".join(self.texts())

    def click(self, label: str) -> str:
        if self.screen == "title" and label == "New Game":
            self.screen = "difficulty"
            return "difficulty menu"
        if self.screen == "difficulty" and label in ("Easy", "Normal", "Hard"):
            self.screen, self.difficulty, self.score = "playing", label, 0
            return f"started level 1 ({label})"
        if label == "Back":
            self.screen = "title"
            return "back to the title screen"
        return "nothing happened"

    def key(self, key_id: str) -> str:
        if self.screen == "playing" and key_id == "SPACE":
            self.score += 10
            return "launched a block"
        return "no effect"


APP_CLASSES: tuple[type[DemoApp], ...] = (Finder, TextEdit, Safari, Notes, SpaceBlocks)
DEMO_APP_NAMES = tuple(cls.name for cls in APP_CLASSES)


class DemoDesktop:
    """Simulated Mac implementing the Observer and Executor protocols."""

    def __init__(self, *, latency: bool = True, rng: random.Random | None = None):
        self.apps: dict[str, DemoApp] = {cls.name: cls() for cls in APP_CLASSES}
        self.running = {"Finder", "Safari"}
        self.front = "Finder"
        self.latency = latency
        self.rng = rng or random.Random(7)
        self._pids = {name: 400 + i * 17 for i, name in enumerate(self.apps)}

    # ---- helpers ------------------------------------------------------------------------------------------------
    def _sleep(self, low: float, high: float) -> None:
        if self.latency:
            time.sleep(self.rng.uniform(low, high))

    def _info(self, name: str, *, installed: bool = False) -> AppInfo:
        app = self.apps[name]
        running = name in self.running and not installed
        return AppInfo(
            app.name, app.bundle_id, pid=self._pids[name] if running else None, path=f"/Applications/{name}.app"
        )

    # ---- Observer -----------------------------------------------------------------------------------------------
    def observe(self) -> Observation:
        started = time.perf_counter()
        self._sleep(0.012, 0.035)
        app = self.apps[self.front]
        elements = [spec.build(i) for i, spec in enumerate(app.specs(), start=1)]
        menu_items = [
            UIElement(i, "AXMenuItem", None, path, kind="menu_item", ops=(MENU,), shortcut=shortcut)
            for i, (path, shortcut) in enumerate(app.menu(), start=1)
        ]
        window = WindowInfo(1, app.title(), focused=True)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        obs = Observation(
            app=self._info(self.front),
            window=window,
            windows=[window],
            elements=elements,
            menu_items=menu_items,
            scroll_areas=[],
            text=app.text(),
            running_apps=[self._info(n) for n in self.apps if n in self.running],
            installed_apps=[self._info(n, installed=True) for n in self.apps],
            stats={
                "visited": len(elements) * 7 + 12,
                "elements": len(elements),
                "menu_items": len(menu_items),
                "truncated": False,
                "walk_ms": elapsed,
                "observe_ms": elapsed,
                "simulated": True,
            },  # fmt: skip
            captured_at=time.time(),
            page_url=app.url(),
        )
        if isinstance(app, SpaceBlocks):
            visual = sum(e.kind == "visual" for e in elements)
            obs.stats["vision"] = {"ran": True, "elements": visual, "texts": visual, "ms": 0.0, "simulated": True}
        return obs

    def user_completes_handoff(self) -> None:
        """What the person does during ASK_USER in the demo: types the two-factor code from their phone."""
        safari = self.apps["Safari"]
        if isinstance(safari, Safari):
            safari.user_enters_code()

    def quick_signature(self) -> str:
        app = self.apps[self.front]
        return json.dumps([self.front, app.title(), [(s.label, s.value) for s in app.specs()]])

    def find_app(self, query: str) -> AppInfo | None:
        needle = query.strip().lower()
        for name in self.apps:
            if name.lower() == needle or name.lower().startswith(needle):
                return self._info(name) if name in self.running else self._info(name, installed=True)
        return None

    # ---- Executor -----------------------------------------------------------------------------------------------
    def validate(self, action: Action, obs: Observation) -> None:
        if obs.app.name != self.front:
            raise StaleElementError("frontmost app changed")
        if action.element is not None and action.element.kind != "menu_item":
            labels = {s.label for s in self.apps[self.front].specs()}
            if action.element.label not in labels:
                raise StaleElementError(f"{action.element.describe()} is gone")

    def execute(self, action: Action, obs: Observation) -> ActionResult:
        started = time.perf_counter()
        self._sleep(0.02, 0.06)
        app = self.apps[self.front]
        op = action.operation
        if op == CLICK and action.element is not None:
            pointer = action.element.kind == "visual"
            detail = app.click(action.element.label)
            method = "pointer at the recognized text (simulated)" if pointer else "AXPress (simulated)"
        elif op == TYPE_TEXT and action.element is not None and action.text is not None:
            keys = action.element.kind == "keyboard" or action.element.secure
            detail = app.type(action.element.label, action.text)
            method = "keystrokes (simulated)" if keys else "AXValue (simulated)"
            if action.submit:  # an address typed into the address bar: Return opens it
                detail = f"{detail}; {app.key('RETURN')}"
        elif op == MENU and action.element is not None:
            detail, method = app.command(action.element.label), "AXPress menu item (simulated)"
        elif op == PRESS_KEY and action.key is not None:
            detail, method = app.key(action.key.id), "CGEvent key (simulated)"
        elif op in (SCROLL_UP, SCROLL_DOWN):
            detail, method = "scrolled", "scroll bar (simulated)"
        elif op == OPEN_APP and action.app is not None:
            return self.open_app(action.app)
        else:
            return ActionResult(False, "none", f"cannot execute {action.describe()}")
        return ActionResult(True, method, detail, round((time.perf_counter() - started) * 1000, 1))

    def open_app(self, app: AppInfo) -> ActionResult:
        self._sleep(0.05, 0.12)
        if app.name not in self.apps:
            return ActionResult(False, "open", f"{app.name} is not part of the demo")
        launched = app.name not in self.running
        self.running.add(app.name)
        self.front = app.name
        return ActionResult(True, "launch (simulated)" if launched else "activate (simulated)", app.name)


# ---- simulated decision model ------------------------------------------------------------------------------------
_QUOTE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
_WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "then",
        "to",
        "in",
        "on",
        "of",
        "for",
        "it",
        "into",
        "with",
        "my",
        "me",
        "please",
        "can",
        "you",
        "could",
        "i",
        "want",
        "now",
        "also",
        "this",
        "that",
        "type",
        "write",
        "enter",
        "put",
        "click",
        "press",
        "open",
        "launch",
        "go",
        "start",
        "app",
        "application",
        "document",
        "new",
    ]
)
Pick = tuple[str, float]


def _label(criterion: Any) -> str:
    """The human label inside a choice criterion (element label, menu path, app, key or slot preview)."""
    if not isinstance(criterion, dict):
        return str(criterion)
    if "element" in criterion:
        match = re.search(r'"(.*?)"', str(criterion["element"]))
        return match.group(1) if match else str(criterion["element"])
    for key in ("command", "app", "window", "key", "preview"):
        if key in criterion:
            return str(criterion[key])
    return ""


def _find(question: dict[str, Any] | None, predicate: Callable[[str], bool]) -> str | None:
    if not question:
        return None
    return next((cid for cid, crit in question["criteria"].items() if predicate(_label(crit).lower())), None)


def _labelled(question: dict[str, Any] | None, label: str) -> str | None:
    return _find(question, lambda text: text == label)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1}


def _infer_app(lowered: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if re.search(r"\b(document|poem|haiku|story|essay|letter)\b", lowered):
        return "TextEdit"
    if re.search(r"\bnotes?\b", lowered):
        return "Notes"
    if re.search(
        r"\b(search|website|browse|visit|look (?:for|up)|google|web|online|pictures?|images?|facebook|marketplace)\b"
        r"|\.com\b",
        lowered,
    ):
        return "Safari"
    if re.search(r"\b(folder|downloads|documents|desktop|applications)\b", lowered):
        return "Finder"
    return None


def simulated_decisions(body: dict[str, Any]) -> dict[str, Pick]:
    """A transparent keyword policy standing in for Jev in demo mode. Returns {question: (choice_id, confidence)}."""
    questions = body["questions"]
    ops = questions["operation"]["criteria"]
    goal = questions["operation"]["instructions"]["goal"]
    lowered = goal.lower()
    state = body["state"]
    desktop = state["desktop"]
    front, window = desktop.get("frontmost_app") or "", (desktop.get("window") or "").lower()
    elements = state.get("elements", [])
    history = state.get("recent_actions", [])
    effective = [h["action"] for h in history if str(h.get("result", "")).startswith(("ok", "ui changed"))]
    quotes = list(slots_from_goal(goal).values())  # the same typeable text the real agent offers
    seen = " ".join([window, state.get("visible_text", ""), *(str(e.get("value", "")) for e in elements)]).lower()
    selected = {e["label"].lower() for e in elements if "selected" in e.get("state", [])}

    def answer(op: str, confidence: float, **heads: Pick) -> dict[str, Pick]:
        return {"operation": (op, confidence), **heads}

    def click(predicate: Callable[[str], bool], confidence: float = 0.9) -> dict[str, Pick] | None:
        if CLICK not in ops:
            return None
        question = questions.get("click_target")
        if question is None:  # a single clickable element: no target question was needed
            only = [e for e in elements if CLICK in e.get("ops", [])]
            return answer(CLICK, confidence) if only and predicate(only[0]["label"].lower()) else None
        cid = _find(question, predicate)
        return answer(CLICK, confidence, click_target=(cid, confidence)) if cid else None

    if history and str(history[-1].get("result", "")).startswith(("declined", "refused")):
        return answer("BLOCKED", 0.9)  # respect a human "no" instead of asking again

    mentioned = next((n for n in DEMO_APP_NAMES if re.search(rf"\b{n.lower()}\b", lowered)), None)
    wanted = _infer_app(lowered, mentioned)
    wants_save = bool(re.search(r"\bsave\b", lowered))
    wants_login = bool(re.search(r"\b(?:log\s*in|sign\s*in)\b", lowered))
    wants_submit = bool(
        re.search(r"\b(search|go to|visit|look (?:up|for)|google|browse|log\s*in|sign\s*in)\b", lowered)
    )
    wants_delete = bool(re.search(r"\b(delete|remove|trash)\b", lowered))
    typing = bool(
        re.search(r"\b(type|write|enter|search|look|google|fill|put|say|add|go to|visit|log\s*in|sign\s*in)\b", lowered)
    )
    named = re.search(r"\bsave\s+(?:it\s+|this\s+|the\s+\w+\s+)?as\s+[\"“]([^\"”]+)", goal, re.IGNORECASE)
    file_name = named.group(1) if wants_save and named else quotes[-1] if wants_save and len(quotes) > 1 else None
    body_quotes = [q for q in quotes if q != file_name] if typing else []
    pending = [q for q in body_quotes if q.lower() not in seen]
    # "write a poem about the sea": nothing to copy from the goal, the writer composes it (GENERATE).
    wants_writing = wants_generation(goal)
    compose = wants_writing and not any(a.startswith("TYPE_TEXT") and '"save as"' not in a.lower() for a in effective)
    saving = any("sheet" in str(e.get("in", "")) for e in elements)
    saved = not saving and window not in ("untitled", "open") and (file_name is None or file_name.lower() == window)
    page = str(desktop.get("page") or "").lower()
    submitted = (
        not wants_submit or any(q.lower() in window or q.lower() in page for q in quotes) or "results for" in seen
    )
    signed_in = "signed in as" in seen
    deleted = any(a.startswith("CLICK") and '"delete"' in a.lower() for a in effective)
    touched = any(a.startswith(("CLICK", "MENU", "PRESS_KEY")) for a in effective)
    only_open = wanted is not None and not (_words(goal) - {wanted.lower()})

    # 1. Switch to (or launch) the app the goal is about.
    if (
        wanted
        and wanted != front
        and OPEN_APP in ops
        and not any(a.startswith(f"OPEN_APP {wanted}") for a in effective)
    ):
        app_id = _find(questions.get("app_target"), lambda label: label == wanted.lower())
        return answer(OPEN_APP, 0.95, **({"app_target": (app_id, 0.96)} if app_id else {}))

    # 2a. A Marketplace listing: open the "item for sale" form, fill title, price and description. Never publish.
    if "marketplace" in lowered and front == "Safari":
        by_label = {e["label"].lower(): e for e in elements}
        offered = questions.get("text_slot", {}).get("criteria", {})
        fields = questions.get("type_text_target")
        if "title" in by_label:
            wanted_slots = {
                "title": ("title",),
                "price": ("price",),
                "category": ("category",),
                "description": ("description",),
            }
            for label, slot_names in wanted_slots.items():
                if by_label[label].get("value") or TYPE_TEXT not in ops:
                    continue
                slot = next((n for n in slot_names if n in offered), GENERATE if GENERATE in offered else None)
                field_id = _labelled(fields, label)
                if slot and field_id:
                    return answer(TYPE_TEXT, 0.9, type_text_target=(field_id, 0.92), text_slot=(slot, 0.93))
            return answer("DONE", 0.9)  # filled in; publishing (and photos) stay with the person
        address = by_label.get("search or enter website name", {})
        site = next((n for n, crit in offered.items() if "marketplace" in _label(crit).lower()), None)
        if site and "marketplace" not in str(address.get("value", "")) and TYPE_TEXT in ops:
            address_id = _labelled(fields, "search or enter website name")  # None: it is the only field
            address_heads: dict[str, Pick] = {"text_slot": (site, 0.93)}
            if address_id:
                address_heads["type_text_target"] = (address_id, 0.92)
            return answer(TYPE_TEXT, 0.9, **address_heads)
        if PRESS_KEY in ops:
            key = _find(questions.get("key_target"), lambda text: text == "return")
            return answer(PRESS_KEY, 0.9, **({"key_target": (key, 0.94)} if key else {}))

    # 2. Finished?
    finished = not pending and not compose and submitted and (not wants_save or saved) and (not wants_delete or deleted)
    finished = finished and (not wants_login or signed_in)
    game = front == SpaceBlocks.name
    if (
        finished
        and not game
        and wanted in (None, front)
        and (body_quotes or wants_writing or wants_save or wants_submit or wants_delete or touched or only_open)
    ):
        return answer("DONE", 0.93)

    # 2b. Sign in: saved username and password (offered only on the right site), then hand 2FA to the person.
    if wants_login and submitted and not signed_in:
        by_label = {e["label"].lower(): e for e in elements}
        slots_offered = questions.get("text_slot", {}).get("criteria", {})
        if "authentication code" in by_label and "ASK_USER" in ops:
            return answer("ASK_USER", 0.9, handoff_reason=("code", 0.93))
        if "password" in by_label and TYPE_TEXT in ops and slots_offered:
            fields = questions.get("type_text_target")
            username = by_label.get("username or email address", {})
            user_id = _find(fields, lambda label: label.startswith("username"))
            if not username.get("value") and "login_username" in slots_offered and user_id:
                return answer(TYPE_TEXT, 0.9, type_text_target=(user_id, 0.92), text_slot=("login_username", 0.95))
            password_id = _find(fields, lambda label: label == "password")
            typed_password = "filled" in by_label["password"].get("state", [])
            if not typed_password and "login_password" in slots_offered and password_id:
                return answer(TYPE_TEXT, 0.9, type_text_target=(password_id, 0.92), text_slot=("login_password", 0.95))
            pick = click(lambda label: label == "sign in")
            if pick:
                return pick

    # 2c. A game that draws its own interface: click the on-screen text the goal names until a level runs.
    if game and "level 1" in seen:
        return answer("DONE", 0.9)

    # 3. Delete: select the named item first, then press Delete (the safety policy asks the human).
    if wants_delete and not deleted:
        names = {q.lower() for q in quotes} or _words(goal)
        if not names & selected:
            pick = click(lambda label: label in names)
            if pick:
                return pick
        pick = click(lambda label: label == "delete", 0.88)
        if pick:
            return pick

    # 4. Save: open the Save sheet, name the file, confirm.
    if wants_save and not pending and not compose and not saved:
        if saving:
            name_field: dict[str, Any] = next((e for e in elements if e["label"].lower() == "save as"), {})
            if file_name and str(name_field.get("value", "")).lower() != file_name.lower() and TYPE_TEXT in ops:
                field_id = _find(questions.get("type_text_target"), lambda label: label == "save as")
                slot = _find(questions.get("text_slot"), lambda label: file_name.lower().startswith(label.rstrip("…")))
                save_heads = {"type_text_target": (field_id, 0.9)} if field_id else {}
                return answer(TYPE_TEXT, 0.88, **save_heads, **({"text_slot": (slot, 0.9)} if slot else {}))
            pick = click(lambda label: label == "save")
            if pick:
                return pick
        command = _find(questions.get("menu_target"), lambda label: label.endswith("save…"))
        if command:
            return answer(MENU, 0.89, menu_target=(command, 0.92))

    # 5. Submit a typed search or address.
    if wants_submit and not pending and not submitted and PRESS_KEY in ops:
        key = _find(questions.get("key_target"), lambda label: label == "return")
        return answer(PRESS_KEY, 0.9, **({"key_target": (key, 0.94)} if key else {}))

    # 6. Type the next quoted text, or have the writer compose it.
    if (pending or compose) and TYPE_TEXT in ops:
        goal_words = _words(goal)
        target = questions.get("type_text_target")
        field_id = _find(target, lambda label: bool(_words(label) & goal_words) and label != "save as") or _find(
            target, lambda label: label in ("body", "note", "search or enter website name")
        )
        if pending:
            slot = _find(questions.get("text_slot"), lambda label: pending[0].lower().startswith(label.rstrip("…")))
        else:
            slot = GENERATE if GENERATE in questions.get("text_slot", {}).get("criteria", {}) else None
        heads: dict[str, Pick] = {}
        if field_id:
            heads["type_text_target"] = (field_id, 0.9)
        if slot:
            heads["text_slot"] = (slot, 0.92)
        return answer(TYPE_TEXT, 0.9, **heads)

    # 7. Nowhere to type yet: New Document / New Note.
    if pending or compose:
        pick = click(lambda label: label.startswith("new "), 0.88)
        if pick:
            return pick
        command = _find(questions.get("menu_target"), lambda label: label.startswith("file › new"))
        if command:
            return answer(MENU, 0.84, menu_target=(command, 0.88))

    # 8. Click what the goal names (buttons, rows, links), skipping what was already clicked.
    goal_words = _words(goal) - set().union(*(_words(n) for n in DEMO_APP_NAMES))
    question = questions.get("click_target")
    best: tuple[int, float, str] | None = None
    for cid, crit in (question or {}).get("criteria", {}).items():
        label = _label(crit)
        if any(f'"{label}"' in a for a in effective):
            continue
        words = _words(label)
        overlap = len(words & goal_words)
        if overlap:
            score = (overlap, overlap / max(1, len(words)), cid)
            best = score if best is None or score[:2] > best[:2] else best
    if best is not None:
        confidence = round(0.62 + 0.3 * best[1], 3)
        return answer(CLICK, confidence, click_target=(best[2], confidence))
    if goal_words and question and not touched:
        # Nothing on screen matches the request: a low-confidence guess that the confidence gate should withhold.
        return answer(CLICK, 0.24, click_target=(next(iter(question["criteria"])), 0.15))
    return answer("BLOCKED", 0.82)


def distribution(ids: list[str], choice: str, confidence: float) -> dict[str, Any]:
    n = len(ids)
    top = max(confidence, 1 / n + 0.01) if n > 1 else 1.0
    rest = (1 - top) / (n - 1) if n > 1 else 0.0
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": {i: round(top if i == choice else rest, 6) for i in ids},
        "confidence": round(confidence, 3),
    }


@dataclass
class SimulatedJev:
    """In-process HTTP transport that speaks the Jev response format with simulated decisions."""

    latency: bool = True
    rng: random.Random = field(default_factory=lambda: random.Random(11))
    calls: int = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        picks = simulated_decisions(body)
        answers = {}
        for name, question in body["questions"].items():
            ids = list(question["criteria"])
            choice, confidence = picks.get(name, (ids[0], 0.5))
            if choice not in ids:
                choice = ids[0]
            answers[name] = distribution(ids, choice, confidence)
        if self.latency:
            time.sleep(self.rng.uniform(0.09, 0.22))
        self.calls += 1
        tokens = len(request.content) // 4
        return httpx.Response(
            200, json={"model": "jev-demo (simulated)", "answers": answers, "usage": {"input_tokens": tokens}}
        )

    def client(self) -> JevClient:
        return JevClient("demo-key-not-used", transport=httpx.MockTransport(self.handle), http2=False, model="jev-demo")
