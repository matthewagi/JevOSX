import plistlib

import pytest

from jevosx.executor.input import utf16_chunks
from jevosx.executor.keys import KeyChord, key_vocabulary
from jevosx.observer.apps import choose_frontmost, find_app, merge_apps, pick_frontmost, scan_installed_apps
from jevosx.observer.menus import shortcut_text, walk_menu_bar
from jevosx.types import AppInfo
from tests.fakes import FakeNode


def menu_bar():
    def item(title, **kw):
        return FakeNode("AXMenuItem", Title=title, **kw)

    file_menu = FakeNode(
        "AXMenu",
        children=[
            item("New", MenuItemCmdChar="n", MenuItemCmdModifiers=0),
            item(""),  # separator
            item("Save As…", MenuItemCmdChar="S", MenuItemCmdModifiers=1),
            item("Revert", Enabled=False),
            item("Share", children=[FakeNode("AXMenu", children=[item("Mail")])]),
            item("Services", children=[FakeNode("AXMenu", children=[item("Huge list")])]),
        ],
    )
    return FakeNode(
        "AXMenuBar",
        children=[
            FakeNode("AXMenuBarItem", Title="Apple", children=[FakeNode("AXMenu", children=[item("Shut Down…")])]),
            FakeNode("AXMenuBarItem", Title="File", children=[file_menu]),
        ],
    )


def test_menu_walk_keeps_enabled_leaves_with_paths_and_shortcuts():
    items, truncated = walk_menu_bar(menu_bar())
    assert not truncated
    assert [(i.label, i.shortcut) for i in items] == [
        ("File › New", "⌘N"),
        ("File › Save As…", "⇧⌘S"),
        ("File › Share › Mail", None),
    ]
    assert all(i.ops == ("MENU",) for i in items)


def test_menu_walk_budget():
    items, truncated = walk_menu_bar(menu_bar(), max_items=1)
    assert len(items) == 1 and truncated


def test_shortcut_text_modifiers():
    assert shortcut_text("q", 0) == "⌘Q"
    assert shortcut_text("x", 2 | 4) == "⌃⌥⌘X"
    assert shortcut_text("f", 8) == "F"
    assert shortcut_text("", 0) is None


def test_key_chords_parse_and_flag():
    chord = KeyChord.parse("Cmd+Shift+S")
    assert str(chord) == "shift+cmd+s" and chord.keycode == 1 and chord.flags == (1 << 20) | (1 << 17)
    assert KeyChord.parse("enter").key == "return"
    with pytest.raises(ValueError):
        KeyChord.parse("cmd+nope")
    keys = key_vocabulary({"send": "cmd+shift+d"}, disabled=["CMD_Q"])
    assert "CMD_Q" not in keys and keys["SEND"].chord == KeyChord.parse("cmd+shift+d")


def test_utf16_chunks_never_split_surrogate_pairs():
    text = "a" * 19 + "😀" + "b"
    chunks = utf16_chunks(text)
    assert "".join(chunks) == text
    assert all(len(c.encode("utf-16-le")) // 2 <= 20 for c in chunks)
    assert chunks[0] == "a" * 19


def test_installed_app_scan_and_lookup(tmp_path):
    def bundle(name, info):
        contents = tmp_path / f"{name}.app" / "Contents"
        contents.mkdir(parents=True)
        (contents / "Info.plist").write_bytes(plistlib.dumps(info))

    bundle("Notes", {"CFBundleName": "Notes", "CFBundleIdentifier": "com.apple.Notes"})
    bundle("Helper", {"CFBundleName": "Helper", "CFBundleIdentifier": "x.helper", "LSUIElement": True})
    (tmp_path / "Broken.app").mkdir()
    installed = scan_installed_apps([str(tmp_path), str(tmp_path / "missing")])
    assert [a.bundle_id for a in installed] == ["com.apple.Notes"]

    running = [AppInfo("Safari", "com.apple.Safari", pid=7)]
    apps = merge_apps(running, [*installed, AppInfo("Safari", "com.apple.Safari")])
    assert [a.name for a in apps] == ["Safari", "Notes"]
    assert find_app("com.apple.notes", apps).name == "Notes"
    assert find_app("saf", apps).pid == 7
    assert find_app("nothing", apps) is None


def test_pick_frontmost_from_window_list():
    windows = [
        {"kCGWindowLayer": 25, "kCGWindowOwnerPID": 90, "kCGWindowOwnerName": "Control Center"},  # menu extra
        {"kCGWindowLayer": 0, "kCGWindowOwnerPID": 77, "kCGWindowOwnerName": "Python"},  # ourselves
        {"kCGWindowLayer": 0, "kCGWindowOwnerPID": 55, "kCGWindowAlpha": 0},  # invisible
        {"kCGWindowLayer": 0, "kCGWindowOwnerPID": 44, "kCGWindowBounds": {"Width": 10, "Height": 10}},  # sliver
        {"kCGWindowLayer": 0, "kCGWindowOwnerPID": 12, "kCGWindowOwnerName": "Safari",
         "kCGWindowBounds": {"X": 0, "Y": 25, "Width": 1200, "Height": 800}},
        {"kCGWindowLayer": 0, "kCGWindowOwnerPID": 13, "kCGWindowOwnerName": "Terminal"},
    ]  # fmt: skip
    assert pick_frontmost(windows, exclude={77}) == 12
    assert pick_frontmost([], exclude=set()) is None


def test_choose_frontmost_trusts_workspace_for_an_active_app_without_windows():
    """Seen live: TextEdit was frontmost with no document open, the window list said Finder, OPEN_APP kept failing."""
    finder = {"kCGWindowLayer": 0, "kCGWindowOwnerPID": 20, "kCGWindowBounds": {"Width": 900, "Height": 600}}
    notes = {"kCGWindowLayer": 0, "kCGWindowOwnerPID": 30, "kCGWindowBounds": {"Width": 900, "Height": 600}}
    assert choose_frontmost([finder, notes], active=40) == 40  # TextEdit, active and windowless
    assert choose_frontmost([finder, notes], active=30) == 20  # active app has a window: the list's order wins
    assert choose_frontmost([finder, notes], active=None) == 20
    assert choose_frontmost([finder], active=77, exclude={77}) == 20  # never ourselves
    assert choose_frontmost([], active=40) == 40
