"""Accessibility-tree walker: turns a live AX subtree into an indexed element table plus visible text.

Written against the small `AXNodeLike` protocol rather than pyobjc, so the traversal, pruning and labelling rules
are unit-tested on any OS with fake nodes. The walk is bounded by node count, element count, depth and wall time,
and prunes subtrees that lie outside the visible window/scroll clip (so off-screen web content does not flood Jev).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..errors import StaleElementError
from ..types import (
    CLICK,
    SCROLL_DOWN,
    SCROLL_UP,
    TYPE_TEXT,
    Rect,
    UIElement,
    clean_text,
    friendly_role,
)


class AXNodeLike(Protocol):
    def get(self, attribute: str, default: Any = None) -> Any: ...

    def get_many(self, attributes: Sequence[str]) -> dict[str, Any]: ...

    def children(self, attribute: str = "AXChildren", limit: int | None = None) -> list[Any]: ...

    def actions(self) -> tuple[str, ...]: ...

    def settable(self, attribute: str) -> bool: ...


BATCH_ATTRS = (
    "AXRole",
    "AXSubrole",
    "AXTitle",
    "AXDescription",
    "AXValue",
    "AXEnabled",
    "AXFocused",
    "AXSelected",
    "AXExpanded",
    "AXPosition",
    "AXSize",
    "AXIdentifier",
    "AXPlaceholderValue",
    "AXHelp",
    "AXHidden",
)

# Roles that are actionable by their nature. Their AX actions are read lazily at execution time.
CLICKABLE_ROLES = frozenset(
    {
        "AXButton",
        "AXCheckBox",
        "AXRadioButton",
        "AXPopUpButton",
        "AXMenuButton",
        "AXLink",
        "AXMenuItem",
        "AXMenuBarItem",
        "AXDisclosureTriangle",
        "AXColorWell",
        "AXDockItem",
        "AXRow",
        "AXDateField",
    }
)
TEXT_INPUT_ROLES = frozenset({"AXTextField", "AXTextArea", "AXComboBox"})
# Generic roles that are only indexed when they expose AXPress (costs one extra IPC, so only when labelled).
PROBE_ROLES = frozenset({"AXImage", "AXGroup", "AXCell"})
TEXT_ROLES = frozenset({"AXStaticText"})
# AXUnknown is deliberately not skipped: Chrome and Electron use it for plain containers that hold real controls.
SKIP_ROLES = frozenset({"AXScrollBar", "AXValueIndicator", "AXSplitter", "AXGrowArea", "AXMenuBar"})
SKIP_SUBROLES = frozenset({"AXMinimizeButton", "AXZoomButton", "AXFullScreenButton"})
LEAF_ROLES = frozenset(
    {
        "AXStaticText",
        "AXCheckBox",
        "AXRadioButton",
        "AXIncrementor",
        "AXColorWell",
        "AXSlider",
        "AXImage",
        "AXBusyIndicator",
        "AXProgressIndicator",
        "AXLevelIndicator",
        "AXRelevanceIndicator",
    }
)
# Clickable roles whose open menu/list shows up as an AX child, so they are always descended.
DESCEND_CLICKABLES = frozenset({"AXPopUpButton", "AXMenuButton", "AXComboBox", "AXRow", "AXDisclosureTriangle"})
CONTAINER_ROLES = frozenset(
    {
        "AXSheet",
        "AXToolbar",
        "AXTabGroup",
        "AXPopover",
        "AXMenu",
        "AXWebArea",
        "AXTable",
        "AXOutline",
        "AXList",
        "AXGroup",
        "AXScrollArea",
        "AXSplitGroup",
        "AXBrowser",
        "AXRadioGroup",
    }
)
ALWAYS_NAMED_CONTAINERS = frozenset({"AXSheet", "AXToolbar", "AXPopover", "AXMenu"})
ROW_ROLES = frozenset({"AXRow"})
CHECKABLE_SUBROLES = frozenset({"AXToggle", "AXSwitch"})


@dataclass
class WalkLimits:
    max_nodes: int = 3000
    max_elements: int = 180
    max_depth: int = 64
    max_children: int = 400
    time_budget_s: float = 1.5
    max_text_chars: int = 3000
    probe_generic: bool = True
    min_scroll_height: float = 40.0


@dataclass
class WalkResult:
    elements: list[UIElement]
    scroll_areas: list[UIElement]
    text: str
    visited: int
    truncated: bool
    elapsed_ms: float
    notes: list[str] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)  # why subtrees were pruned (diagnostics)


@dataclass(slots=True)
class _Frame:
    node: Any
    depth: int
    parent_id: int
    container: str | None
    clip: Rect | None
    owner: int | None  # position in `elements` whose label is built from descendant text
    in_web: bool
    in_row: bool


def frame_of(attrs: dict[str, Any]) -> Rect | None:
    pos, size = attrs.get("AXPosition"), attrs.get("AXSize")
    if not pos or not size:
        return None
    try:
        return Rect(float(pos[0]), float(pos[1]), float(size[0]), float(size[1]))
    except (TypeError, ValueError, IndexError):
        return None


def is_secure(role: str, subrole: str | None) -> bool:
    return subrole == "AXSecureTextField" or role == "AXSecureTextField"


def checked_state(role: str, subrole: str | None, value: Any) -> bool | None:
    if role in ("AXCheckBox", "AXRadioButton") or subrole in CHECKABLE_SUBROLES:
        if value in (1, True):
            return True
        if value in (0, False):
            return False
    return None


def display_value(role: str, subrole: str | None, value: Any, *, secure: bool, limit: int = 200) -> str | None:
    if secure or value is None or isinstance(value, bool) or checked_state(role, subrole, value) is not None:
        return None
    if isinstance(value, list | tuple) or hasattr(value, "get_many"):
        return None
    text = clean_text(value, limit)
    return text or None


def own_label(attrs: dict[str, Any], *, text_input: bool) -> str:
    for key in ("AXTitle", "AXDescription"):
        text = clean_text(attrs.get(key), 120)
        if text:
            return text
    if text_input:
        text = clean_text(attrs.get("AXPlaceholderValue"), 120)
        if text:
            return text
    return ""


class TreeWalker:
    def __init__(self, limits: WalkLimits | None = None, clock: Callable[[], float] = time.perf_counter):
        self.limits = limits or WalkLimits()
        self.clock = clock

    def walk(self, roots: Sequence[tuple[Any, str | None]], *, clip: Rect | None = None) -> WalkResult:
        limits = self.limits
        started = self.clock()
        deadline = started + limits.time_budget_s
        elements: list[UIElement] = []
        scroll_areas: list[UIElement] = []
        scroll_by_node: dict[int, UIElement] = {}
        inner_text: dict[int, list[str]] = {}
        pending_fallback: dict[int, str] = {}
        last_text_by_parent: dict[int, str] = {}
        text_lines: list[str] = []
        text_chars = 0
        visited = 0
        truncated = False
        notes: list[str] = []
        skipped: dict[str, int] = {}

        def skip(reason: str) -> None:
            skipped[reason] = skipped.get(reason, 0) + 1

        stack: list[_Frame] = [
            _Frame(node, 0, -1, container, clip, None, False, False) for node, container in reversed(roots)
        ]
        while stack:
            if visited >= limits.max_nodes or self.clock() > deadline:
                truncated = True
                notes.append("node budget reached" if visited >= limits.max_nodes else "time budget reached")
                break
            frame = stack.pop()
            visited += 1
            node_id = visited
            try:
                attrs = frame.node.get_many(BATCH_ATTRS)
            except StaleElementError as exc:
                skip("no_response" if "did not respond" in str(exc) else "vanished")
                continue
            role = str(attrs.get("AXRole") or "")
            subrole = attrs.get("AXSubrole")
            subrole = str(subrole) if subrole else None
            if role in SKIP_ROLES or subrole in SKIP_SUBROLES or attrs.get("AXHidden") is True:
                skip("hidden" if attrs.get("AXHidden") is True else "skipped_role")
                continue  # hidden or chrome-only: prune the whole subtree
            rect = frame_of(attrs)
            if rect is not None and not rect.empty and frame.clip is not None and not rect.intersects(frame.clip):
                skip("offscreen")
                continue  # entirely outside the visible region: prune the whole subtree
            zero_size = rect is not None and rect.empty  # collapsed/invisible: never indexed as a control

            # Name scroll areas after their first visible child ("scroll area · web page 'Apple'").
            parent_scroll = scroll_by_node.pop(frame.parent_id, None)
            if parent_scroll is not None and not parent_scroll.label:
                child_name = clean_text(attrs.get("AXTitle") or attrs.get("AXDescription"), 60)
                parent_scroll.label = friendly_role(role, subrole) + (f' "{child_name}"' if child_name else "")

            enabled = attrs.get("AXEnabled") is not False

            if role in TEXT_ROLES:
                text = clean_text(attrs.get("AXValue") or attrs.get("AXTitle") or attrs.get("AXDescription"), 300)
                if text:
                    last_text_by_parent[frame.parent_id] = text
                    if frame.owner is not None:
                        inner_text.setdefault(frame.owner, []).append(text)
                    elif text_chars < limits.max_text_chars and (not text_lines or text_lines[-1] != text):
                        text_lines.append(text)
                        text_chars += len(text) + 1
                continue

            secure = is_secure(role, subrole)
            element: UIElement | None = None
            if role == "AXScrollArea" and rect is not None and rect.h >= limits.min_scroll_height:
                element = UIElement(
                    index=len(scroll_areas) + 1,
                    role=role,
                    subrole=subrole,
                    label=clean_text(attrs.get("AXDescription") or attrs.get("AXTitle"), 80),
                    kind="scroll_area",
                    ops=(SCROLL_UP, SCROLL_DOWN),
                    container=frame.container,
                    frame=rect,
                    in_web_area=frame.in_web,
                    node=frame.node,
                )
                scroll_areas.append(element)
                scroll_by_node[node_id] = element
                element = None
            elif not zero_size and (
                role in TEXT_INPUT_ROLES
                or role in CLICKABLE_ROLES
                or (limits.probe_generic and role in PROBE_ROLES and own_label(attrs, text_input=False))
            ):
                element = self._make_element(frame, attrs, role, subrole, rect, enabled, secure)
                if element is None and frame.owner is not None:
                    # Read-only text inside a row or unlabeled control: contributes to its label.
                    text = clean_text(attrs.get("AXValue") or attrs.get("AXTitle"), 200)
                    if text:
                        inner_text.setdefault(frame.owner, []).append(text)
                elif element is None and role in TEXT_INPUT_ROLES:
                    text = clean_text(attrs.get("AXValue"), 300)
                    if text and text_chars < limits.max_text_chars:
                        text_lines.append(text)
                        text_chars += len(text) + 1

            owner = frame.owner
            if element is not None:
                if len(elements) >= limits.max_elements:
                    if not truncated:
                        notes.append("element budget reached")
                    truncated = True
                else:
                    element.index = len(elements) + 1
                    elements.append(element)
                    position = len(elements) - 1
                    if not element.label:
                        # Inputs and leaves are named now; containers (links, web buttons) prefer their inner text.
                        fallback = self._fallback_label(frame, attrs, last_text_by_parent)
                        if role in TEXT_INPUT_ROLES or role in LEAF_ROLES:
                            element.label = fallback
                        elif fallback:
                            pending_fallback[position] = fallback
                    if role in ROW_ROLES or not element.label:
                        owner = position

            # Decide whether to descend.
            if role in LEAF_ROLES or frame.depth >= limits.max_depth:
                continue
            if element is not None and role not in DESCEND_CLICKABLES and element.label and role != "AXTextArea":
                continue
            if role in TEXT_INPUT_ROLES and role != "AXComboBox":
                continue

            container = frame.container
            label = clean_text(attrs.get("AXTitle") or attrs.get("AXDescription"), 60)
            if role in ALWAYS_NAMED_CONTAINERS or (role in CONTAINER_ROLES and label):
                name = friendly_role(role, subrole)
                container = f'{name} "{label}"' if label else name
            child_clip = frame.clip
            if role in ("AXScrollArea", "AXWindow") and rect is not None and not rect.empty:
                child_clip = rect if child_clip is None else (child_clip.intersection(rect) or rect)

            try:
                kids = self._children(frame.node, role)
            except StaleElementError:
                skip("children_unreadable")
                continue
            if len(kids) >= limits.max_children:
                truncated = True
                notes.append(f"{friendly_role(role, subrole)} has more than {limits.max_children} children")
            in_web = frame.in_web or role == "AXWebArea"
            in_row = frame.in_row or role in ROW_ROLES
            for kid in reversed(kids):
                stack.append(_Frame(kid, frame.depth + 1, node_id, container, child_clip, owner, in_web, in_row))

        for position, texts in inner_text.items():
            element = elements[position]
            joined = clean_text(" · ".join(dict.fromkeys(texts)), 160)
            if element.role in ROW_ROLES or not element.label:
                element.label = joined or element.label
        for position, fallback in pending_fallback.items():
            if not elements[position].label:
                elements[position].label = fallback
        for element in elements:
            if not element.label:
                element.label = element.role_name
        for area in scroll_areas:
            if not area.label:
                area.label = area.container or "content"

        return WalkResult(
            elements=elements,
            scroll_areas=scroll_areas,
            text="\n".join(text_lines),
            visited=visited,
            truncated=truncated,
            elapsed_ms=round((self.clock() - started) * 1000, 1),
            notes=notes,
            skipped=skipped,
        )

    # ------------------------------------------------------------------------------------------------------------
    def _make_element(
        self,
        frame: _Frame,
        attrs: dict[str, Any],
        role: str,
        subrole: str | None,
        rect: Rect | None,
        enabled: bool,
        secure: bool,
    ) -> UIElement | None:
        node = frame.node
        ops: tuple[str, ...]
        kind = "control"
        value_settable = False
        actions: tuple[str, ...] = ()
        if role in TEXT_INPUT_ROLES:
            try:
                value_settable = bool(node.settable("AXValue"))
            except StaleElementError:
                return None
            # Some real inputs (e.g. Chrome's address bar) do not report AXValue as settable, but a field that has
            # keyboard focus, or is a search field, can still be typed into with keystrokes.
            typeable = value_settable or secure or attrs.get("AXFocused") is True or subrole == "AXSearchField"
            if not typeable and role != "AXComboBox" and not frame.in_row:
                try:  # a field the cursor can be put in is an input even if its value is not settable
                    typeable = bool(node.settable("AXFocused"))
                except StaleElementError:
                    return None
            if not typeable and role != "AXComboBox":
                return None  # read-only text (labels, table cells): treat as text, not a control
            kind = "text_input"
            ops = (TYPE_TEXT, CLICK) if typeable else (CLICK,)
        elif role in PROBE_ROLES:
            try:
                actions = tuple(node.actions())
            except StaleElementError:
                return None
            if "AXPress" not in actions:
                return None
            ops = (CLICK,)
        else:
            ops = (CLICK,)
            if role in ROW_ROLES:
                kind = "row"
        if frame.in_row and role in TEXT_INPUT_ROLES and not value_settable:
            return None
        if not enabled:
            ops = ()
        selected = attrs.get("AXSelected")
        expanded = attrs.get("AXExpanded")
        identifier = attrs.get("AXIdentifier")
        identifier = str(identifier) if identifier and not str(identifier).startswith("_NS:") else None
        return UIElement(
            index=0,
            role=role,
            subrole=subrole,
            label=own_label(attrs, text_input=role in TEXT_INPUT_ROLES),
            value=display_value(role, subrole, attrs.get("AXValue"), secure=secure),
            kind=kind,
            ops=ops,
            enabled=enabled,
            focused=attrs.get("AXFocused") is True,
            selected=bool(selected) if isinstance(selected, bool | int) else None,
            checked=checked_state(role, subrole, attrs.get("AXValue")),
            expanded=bool(expanded) if isinstance(expanded, bool | int) else None,
            secure=secure,
            container=frame.container,
            identifier=identifier,
            in_web_area=frame.in_web,
            value_settable=value_settable,
            frame=rect,
            actions=actions,
            node=node,
        )

    def _fallback_label(self, frame: _Frame, attrs: dict[str, Any], last_text_by_parent: dict[int, str]) -> str:
        try:
            title_element = frame.node.get("AXTitleUIElement")
            if title_element is not None and hasattr(title_element, "get"):
                text = clean_text(title_element.get("AXValue") or title_element.get("AXTitle"), 120)
                if text:
                    return text
        except StaleElementError:
            pass
        sibling = last_text_by_parent.get(frame.parent_id)
        if sibling:
            return clean_text(sibling.rstrip(":"), 120)
        return clean_text(attrs.get("AXHelp"), 120)

    def _children(self, node: Any, role: str) -> list[Any]:
        limit = self.limits.max_children
        if role in ("AXTable", "AXOutline"):
            kids = node.children("AXVisibleRows")
            if kids:
                return kids[:limit]
        elif role in ("AXList", "AXGrid"):
            kids = node.children("AXVisibleChildren")
            if kids:
                return kids[:limit]
        return node.children("AXChildren", limit)
