"""Keyboard vocabulary: named, pre-approved key chords the router may offer as PRESS_KEY targets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# ANSI-US virtual key codes (HIToolbox/Events.h).
KEYCODES: dict[str, int] = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12,
    "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23,
    "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34,
    "p": 35, "return": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44,
    "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49, "`": 50, "delete": 51, "escape": 53,
    "f5": 96, "f6": 97, "f7": 98, "f3": 99, "f8": 100, "f9": 101, "f11": 103, "f10": 109, "f12": 111,
    "home": 115, "pageup": 116, "forwarddelete": 117, "f4": 118, "end": 119, "f2": 120, "pagedown": 121,
    "f1": 122, "left": 123, "right": 124, "down": 125, "up": 126,
}  # fmt: skip
ALIASES = {
    "enter": "return", "esc": "escape", "backspace": "delete", "del": "forwarddelete", "pgup": "pageup",
    "pgdn": "pagedown", "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "comma": ",", "period": ".", "slash": "/", "minus": "-", "equal": "=", "backtick": "`",
}  # fmt: skip
# CGEventFlags masks.
MODIFIER_FLAGS = {"shift": 1 << 17, "ctrl": 1 << 18, "alt": 1 << 19, "cmd": 1 << 20, "fn": 1 << 23}
MODIFIER_ALIASES = {
    "command": "cmd", "⌘": "cmd", "control": "ctrl", "⌃": "ctrl", "option": "alt", "opt": "alt", "⌥": "alt",
    "⇧": "shift",
}  # fmt: skip
_MOD_ORDER = ("ctrl", "alt", "shift", "cmd", "fn")


@dataclass(frozen=True, slots=True)
class KeyChord:
    key: str
    modifiers: frozenset[str] = frozenset()

    @classmethod
    def parse(cls, spec: str) -> KeyChord:
        parts = [p.strip().lower() for p in spec.replace(" ", "").split("+") if p.strip()]
        if not parts:
            raise ValueError("empty key chord")
        *mods, key = parts
        key = ALIASES.get(key, key)
        if key not in KEYCODES:
            raise ValueError(f"unknown key {key!r} in {spec!r}")
        modifiers = set()
        for mod in mods:
            mod = MODIFIER_ALIASES.get(mod, mod)
            if mod not in MODIFIER_FLAGS:
                raise ValueError(f"unknown modifier {mod!r} in {spec!r}")
            modifiers.add(mod)
        return cls(key, frozenset(modifiers))

    @property
    def keycode(self) -> int:
        return KEYCODES[self.key]

    @property
    def flags(self) -> int:
        flags = 0
        for mod in self.modifiers:
            flags |= MODIFIER_FLAGS[mod]
        return flags

    def __str__(self) -> str:
        return "+".join([*(m for m in _MOD_ORDER if m in self.modifiers), self.key])


@dataclass(frozen=True, slots=True)
class KeyBinding:
    id: str
    chord: KeyChord
    description: str


DEFAULT_KEYS: tuple[tuple[str, str, str], ...] = (
    ("RETURN", "return", "Return: submit, confirm, or open the selection"),
    ("ESCAPE", "escape", "Escape: cancel, close a popover, menu, or dialog"),
    ("TAB", "tab", "Tab: move focus to the next field"),
    ("SHIFT_TAB", "shift+tab", "Shift-Tab: move focus to the previous field"),
    ("SPACE", "space", "Space: toggle the focused control or page down"),
    ("UP", "up", "Up arrow: previous item"),
    ("DOWN", "down", "Down arrow: next item or open suggestions"),
    ("LEFT", "left", "Left arrow"),
    ("RIGHT", "right", "Right arrow"),
    ("PAGE_DOWN", "pagedown", "Page Down"),
    ("PAGE_UP", "pageup", "Page Up"),
    ("BACKSPACE", "delete", "Delete the character before the cursor"),
    ("CMD_A", "cmd+a", "Select all"),
    ("CMD_C", "cmd+c", "Copy"),
    ("CMD_V", "cmd+v", "Paste"),
    ("CMD_Z", "cmd+z", "Undo"),
    ("CMD_F", "cmd+f", "Find"),
    ("CMD_L", "cmd+l", "Focus the address/location bar"),
    ("CMD_T", "cmd+t", "New tab"),
    ("CMD_N", "cmd+n", "New window or document"),
    ("CMD_W", "cmd+w", "Close the current tab or window"),
    ("CMD_S", "cmd+s", "Save"),
    ("CMD_Q", "cmd+q", "Quit the frontmost app"),
)


def key_vocabulary(custom: Mapping[str, str] | None = None, disabled: Iterable[str] = ()) -> dict[str, KeyBinding]:
    blocked = {d.upper() for d in disabled}
    keys = {k: KeyBinding(k, KeyChord.parse(spec), desc) for k, spec, desc in DEFAULT_KEYS if k not in blocked}
    for key_id, spec in (custom or {}).items():
        key_id = key_id.upper()
        if key_id not in blocked:
            keys[key_id] = KeyBinding(key_id, KeyChord.parse(spec), f"Custom shortcut {spec}")
    return keys
