"""Menu-bar walker. Cocoa exposes every menu item through AX even while menus are closed, and AXPress on an item
runs its command without opening the menu: a deterministic, coordinate-free path to most app features."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from ..errors import StaleElementError
from ..types import MENU, UIElement, clean_text

MENU_ATTRS = (
    "AXRole",
    "AXTitle",
    "AXEnabled",
    "AXChildren",
    "AXMenuItemCmdChar",
    "AXMenuItemCmdModifiers",
    "AXMenuItemMarkChar",
)
# kAXMenuItemModifier* bit flags. Command is implied unless the NoCommand bit is set.
_MOD_SHIFT, _MOD_OPTION, _MOD_CONTROL, _MOD_NO_COMMAND = 1, 2, 4, 8
PATH_SEPARATOR = " › "
DEFAULT_SKIP_SUBMENUS = ("Services",)


def shortcut_text(char: Any, modifiers: Any) -> str | None:
    if not char or not str(char).strip():
        return None
    mods = int(modifiers or 0)
    text = ""
    if mods & _MOD_CONTROL:
        text += "⌃"
    if mods & _MOD_OPTION:
        text += "⌥"
    if mods & _MOD_SHIFT:
        text += "⇧"
    if not mods & _MOD_NO_COMMAND:
        text += "⌘"
    return text + str(char).upper()


def walk_menu_bar(
    menu_bar: Any,
    *,
    max_items: int = 200,
    max_depth: int = 3,
    skip_apple_menu: bool = True,
    skip_submenus: Sequence[str] = DEFAULT_SKIP_SUBMENUS,
    time_budget_s: float = 0.8,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list[UIElement], bool]:
    """Return enabled leaf menu items (with their full path) and whether the walk was truncated."""
    items: list[UIElement] = []
    deadline = clock() + time_budget_s
    truncated = False

    def visit(menu: Any, path: list[str], depth: int) -> None:
        nonlocal truncated
        try:
            entries = menu.children()
        except StaleElementError:
            return
        for entry in entries:
            if len(items) >= max_items or clock() > deadline:
                truncated = True
                return
            try:
                attrs = entry.get_many(MENU_ATTRS)
            except StaleElementError:
                continue
            title = clean_text(attrs.get("AXTitle"), 80)
            if not title:
                continue  # separators
            submenus = [c for c in (attrs.get("AXChildren") or []) if hasattr(c, "get_many")]
            if submenus:
                if depth < max_depth and title not in skip_submenus:
                    visit(submenus[0], [*path, title], depth + 1)
                continue
            if attrs.get("AXEnabled") is False:
                continue
            items.append(
                UIElement(
                    index=len(items) + 1,
                    role="AXMenuItem",
                    subrole=None,
                    label=PATH_SEPARATOR.join([*path, title]),
                    kind="menu_item",
                    ops=(MENU,),
                    checked=True if clean_text(attrs.get("AXMenuItemMarkChar")) else None,
                    shortcut=shortcut_text(attrs.get("AXMenuItemCmdChar"), attrs.get("AXMenuItemCmdModifiers")),
                    node=entry,
                )
            )

    try:
        bar_items = menu_bar.children()
    except StaleElementError:
        return items, True
    for position, bar_item in enumerate(bar_items):
        if skip_apple_menu and position == 0:
            continue  # Apple menu: Shut Down, Restart, Log Out… never offered
        try:
            attrs = bar_item.get_many(("AXTitle", "AXChildren"))
        except StaleElementError:
            continue
        title = clean_text(attrs.get("AXTitle"), 60)
        menus = [c for c in (attrs.get("AXChildren") or []) if hasattr(c, "get_many")]
        if title and menus:
            visit(menus[0], [title], 1)
        if truncated:
            break
    return items, truncated
