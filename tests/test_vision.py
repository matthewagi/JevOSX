import subprocess

import pytest

from jevosx.executor.mac import MacExecutor
from jevosx.observer.vision import (
    KEYBOARD_ROLE,
    VISION_ROLE,
    TextBox,
    VisionReader,
    keyboard_element,
    needs_vision,
    pick_window,
    reading_order,
    to_screen,
    visual_elements,
)
from jevosx.router.policy import build_state
from jevosx.types import CLICK, TYPE_TEXT, Action, Rect
from tests.fakes import FakeNode, NoFocusAX, element, observation

WINDOW = Rect(100, 50, 800, 600)


def box(text, x, y, w=0.1, h=0.04, confidence=0.9):
    return TextBox(text, confidence, x, y, w, h)


def test_boxes_map_to_screen_points_inside_the_window():
    frame = to_screen(box("Play", 0.5, 0.25, 0.1, 0.05), WINDOW)
    assert frame == Rect(500, 200, 80, 30) and frame.center == (540, 215)


def test_reading_order_groups_lines():
    boxes = [box("right", 0.6, 0.101), box("second line", 0.1, 0.3), box("left", 0.1, 0.1)]
    assert [b.text for b in reading_order(boxes)] == ["left", "right", "second line"]


def test_visual_elements_are_clickable_ids_and_skip_what_accessibility_already_has():
    close = element(1, "AXButton", "Close", frame=Rect(110, 60, 20, 20))
    boxes = [
        box("SPACE BLOCKS", 0.3, 0.1, 0.4, 0.08),
        box("New Game", 0.4, 0.4),
        box("New Game", 0.401, 0.401),  # the same text twice at the same spot
        box("Close", 0.0125, 0.0166, 0.01, 0.01),  # repeats the AX close button in its own frame
        box("Level 1 of 3", 0.4, 0.8),  # already in the AX text: context, not a control
        box("•••", 0.1, 0.9),  # no letters or digits
        box("blurry", 0.5, 0.9, confidence=0.1),
    ]
    elements, lines = visual_elements(boxes, WINDOW, existing=[close], ax_text="Level 1 of 3", start_index=2)
    assert [(e.index, e.label) for e in elements] == [(2, "SPACE BLOCKS"), (3, "New Game")]
    assert all(e.kind == "visual" and e.ops == (CLICK,) and e.role == VISION_ROLE for e in elements)
    assert lines == ["Close", "SPACE BLOCKS", "New Game", "Level 1 of 3"]  # reading order, duplicates once
    assert elements[1].role_name == "on-screen text"
    capped, _ = visual_elements(boxes, WINDOW, max_items=1)
    assert len(capped) == 1


def test_needs_vision_only_for_windows_accessibility_cannot_see():
    game = [element(1, "AXButton", "Close")]
    assert needs_vision("auto", game, "")
    assert not needs_vision("auto", game, "Are you sure you want to quit? Unsaved changes will be lost.")
    rich = [element(i, "AXButton", f"b{i}") for i in range(1, 8)]
    assert not needs_vision("auto", rich, "")
    browser_canvas = [*rich, element(9, "AXLink", "Home", in_web_area=True)]  # toolbar + a bare canvas page
    assert needs_vision("auto", browser_canvas, "")
    assert needs_vision("always", rich, "lots of text here " * 5) and not needs_vision("off", game, "")


def test_pick_window_prefers_the_focused_title_and_skips_panels():
    windows = [
        {"kCGWindowOwnerPID": 9, "kCGWindowLayer": 0, "kCGWindowNumber": 1,
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 800, "Height": 600}},
        {"kCGWindowOwnerPID": 7, "kCGWindowLayer": 3, "kCGWindowNumber": 2,
         "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 300, "Height": 300}},
        {"kCGWindowOwnerPID": 7, "kCGWindowLayer": 0, "kCGWindowNumber": 3, "kCGWindowName": "Inspector",
         "kCGWindowBounds": {"X": 5, "Y": 5, "Width": 200, "Height": 400}},
        {"kCGWindowOwnerPID": 7, "kCGWindowLayer": 0, "kCGWindowNumber": 4, "kCGWindowName": "Level 1",
         "kCGWindowBounds": {"X": 100, "Y": 50, "Width": 800, "Height": 600}},
    ]  # fmt: skip
    assert pick_window(windows, 7, "Level 1") == (4, WINDOW)
    assert pick_window(windows, 7, None)[0] == 3
    assert pick_window(windows, 8, None) is None


class Recorder:
    def __init__(self, image=b"frame-1"):
        self.image = image
        self.ocr_calls = 0

    def capture(self, window_id, path):
        path.write_bytes(self.image)

    def ocr(self, path):
        self.ocr_calls += 1
        return [box("Play", 0.4, 0.4)]


def test_reader_reuses_ocr_for_an_unchanged_window_and_explains_failures():
    rec = Recorder()
    reader = VisionReader(allowed=lambda: True, locate=lambda pid, title: (4, WINDOW), capture=rec.capture,
                          ocr=rec.ocr)  # fmt: skip
    first, second = reader.read(7, "Game"), reader.read(7, "Game")
    assert first.ok and second.ok and second.cached and rec.ocr_calls == 1 and second.boxes == first.boxes
    rec.image = b"frame-2"
    assert not reader.read(7, "Game").cached and rec.ocr_calls == 2
    denied = VisionReader(allowed=lambda: False, locate=lambda pid, title: (4, WINDOW))
    assert not denied.read(7, "Game").ok and "Screen Recording" in denied.read(7, "Game").note

    def broken(window_id, path):
        raise subprocess.CalledProcessError(1, "screencapture")

    failing = VisionReader(allowed=lambda: True, locate=lambda pid, title: (4, WINDOW), capture=broken, ocr=rec.ocr)
    result = failing.read(7, "Game")
    assert not result.ok and "vision failed" in result.note


def test_jev_sees_on_screen_text_as_ordinary_choices():
    visual, _ = visual_elements([box("New Game", 0.4, 0.4)], WINDOW, start_index=2)
    elements = [element(1, "AXButton", "Close"), *visual, keyboard_element(3)]
    state = build_state(observation(elements, window="Space Blocks"))
    assert state["elements"][1] == {"index": 2, "role": "on-screen text", "label": "New Game",
                                    "in": "screen text (OCR)", "ops": ["CLICK"]}  # fmt: skip
    assert state["elements"][2]["role"] == "keyboard" and state["elements"][2]["ops"] == ["TYPE_TEXT"]
    assert "frame" not in str(state) and "540" not in str(state)  # no coordinates reach Jev


def test_executor_clicks_the_centre_of_recognized_text_and_types_at_the_cursor(monkeypatch):
    from jevosx.config import ExecutorSettings
    from jevosx.executor import input as keyboard

    clicks, typed = [], []
    monkeypatch.setattr(keyboard, "click_at", lambda x, y: clicks.append((x, y)))
    monkeypatch.setattr(keyboard, "type_text", lambda text, delay_s=0: typed.append(text))
    executor = object.__new__(MacExecutor)
    executor.settings = ExecutorSettings()
    visual, _ = visual_elements([box("Play", 0.5, 0.25, 0.1, 0.05)], WINDOW)
    obs = observation(visual)
    executor._frontmost_pid = lambda: obs.app.pid  # the game window is in front
    result = executor.execute(Action(CLICK, element=visual[0]), obs)
    assert result.ok and result.method == "pointer" and clicks == [(540.0, 215.0)]
    keys = keyboard_element(2)
    assert keys.role == KEYBOARD_ROLE
    assert executor.execute(Action(TYPE_TEXT, element=keys, text="hello"), obs).ok and typed == ["hello"]
    secret = Action(TYPE_TEXT, element=keys, text="hunter2", text_is_secret=True)
    assert not executor.execute(secret, obs).ok and typed == ["hello"]


def test_typed_text_that_shows_up_late_in_the_field_counts_as_typed(monkeypatch):
    """Seen live: a fresh Chrome window's address bar held the typed text only ~30 ms after the last key event."""
    from jevosx.config import ExecutorSettings
    from jevosx.executor import input as keyboard

    class LateField(FakeNode):
        reads = 0

        def get(self, attribute, default=None):
            if attribute == "AXValue":
                self.reads += 1
                return "population of Gozo" if self.reads > 2 else ""
            return super().get(attribute, default)

    monkeypatch.setattr(keyboard, "post_chord", lambda chord, delay_s=0: None)
    monkeypatch.setattr(keyboard, "type_text", lambda text, delay_s=0: None)
    executor = object.__new__(MacExecutor)
    executor.settings = ExecutorSettings(settle_poll_s=0.001, typed_timeout_s=0.5)
    executor._AXNode = NoFocusAX
    field = element(1, "AXTextField", "Address and search bar", ops=("TYPE_TEXT",), in_web_area=False,
                    node=LateField("AXTextField"))  # fmt: skip
    obs = observation([field])
    executor._frontmost_pid = lambda: obs.app.pid
    result = executor._type(field, "population of Gozo", obs, secret=False, prefer_keys=True)
    assert result.ok and result.method == "keystrokes" and field.node.reads == 3
    executor.settings = ExecutorSettings(settle_poll_s=0.001, typed_timeout_s=0.01)
    never = element(2, "AXTextField", "Search", ops=("TYPE_TEXT",), node=FakeNode("AXTextField", Value=""))
    assert not executor._type(never, "Gozo", obs, secret=False, prefer_keys=True).ok  # it never showed up


def test_typed_text_gets_seconds_to_show_up_on_a_busy_mac(monkeypatch):
    """Seen live: with a load average of 6, Chrome's address bar held "facebook.com" only after the 0.8 s read-back
    had given up; the next observation found it there. The default wait is longer and ends as soon as it shows."""
    from jevosx.config import ExecutorSettings
    from jevosx.executor import input as keyboard

    class SlowField(FakeNode):
        reads = 0

        def get(self, attribute, default=None):
            if attribute == "AXValue":
                self.reads += 1
                return "facebook.com" if self.reads > 25 else ""
            return super().get(attribute, default)

    monkeypatch.setattr(keyboard, "post_chord", lambda chord, delay_s=0: None)
    monkeypatch.setattr(keyboard, "type_text", lambda text, delay_s=0: None)
    executor = object.__new__(MacExecutor)
    executor.settings = ExecutorSettings()  # the defaults: 25 polls of 0.05 s is 1.25 s
    executor._AXNode = NoFocusAX
    field = element(1, "AXTextField", "Address and search bar", ops=("TYPE_TEXT",), in_web_area=False,
                    node=SlowField("AXTextField"))  # fmt: skip
    obs = observation([field])
    executor._frontmost_pid = lambda: obs.app.pid
    assert executor._type(field, "facebook.com", obs, secret=False, prefer_keys=True).ok


@pytest.mark.parametrize("mode", ["auto", "always", "off"])
def test_vision_mode_is_validated(mode):
    from jevosx.config import Settings

    settings = Settings()
    settings.observer.vision = mode
    settings.validate()
