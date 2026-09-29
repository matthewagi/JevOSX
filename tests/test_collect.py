"""`jevosx collect`: scroll a results page to its end and keep every result, without a model."""

import httpx

from jevosx.collect import (
    Collector,
    link_key,
    price_summary,
    price_value,
    result_shape,
    save_photos,
    split_price,
    url_shape,
)
from jevosx.executor.keys import key_vocabulary
from jevosx.images import ImageSaver
from jevosx.types import ActionResult, AppInfo, Rect
from tests.fakes import element, observation

CHROME = AppInfo("Google Chrome", "com.google.Chrome", pid=300)
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def card(n, title, price="€50", town="Mosta", row=0):
    frame = Rect(20 + (n % 4) * 200, 200 + row * 260, 180, 240)
    link = element(
        0, "AXLink", f"{price} · {title} · {town}", in_web_area=True, frame=frame,
        url=f"https://www.facebook.com/marketplace/item/{1000 + n}/?ref=search&tracking={n}",
    )  # fmt: skip
    photo = element(
        0, "AXImage", title, ops=("SAVE_IMAGE",), kind="image", in_web_area=True,
        url=f"https://scontent.fbcdn.net/{n}.jpg", frame=Rect(frame.x, frame.y, 180, 180),
    )  # fmt: skip
    return [link, photo]


class Feed:
    """A results page: `per_screen` cards visible at a time; `loaded` cards exist until the next batch arrives
    `late` reads after the reader has reached the end of what is loaded."""

    def __init__(self, titles, *, per_screen=8, step=4, batch=12, late=0, scroll_area=True):
        self.titles = titles
        self.per_screen, self.step, self.batch, self.late = per_screen, step, batch, late
        self.loaded = min(batch, len(titles))
        self.top = 0
        self.waiting = 0
        self.scroll_area = scroll_area
        self.actions = []

    def observe(self):
        if self.top + self.per_screen >= self.loaded and self.loaded < len(self.titles):
            self.waiting += 1
            if self.waiting > self.late:
                self.loaded, self.waiting = min(len(self.titles), self.loaded + self.batch), 0
        elements = [element(0, "AXLink", "Marketplace", in_web_area=True, url="https://www.facebook.com/marketplace/",
                            frame=Rect(0, 0, 200, 30))]  # fmt: skip
        elements += [element(0, "AXLink", "Tools", in_web_area=True, frame=Rect(0, 40, 200, 90),
                             url="https://www.facebook.com/marketplace/category/tools")]  # fmt: skip
        for n in range(self.top, min(self.top + self.per_screen, self.loaded)):
            title, price = self.titles[n]
            elements += card(n, title, price, row=(n - self.top) // 4)
        for i, e in enumerate(elements, start=1):
            e.index = i
        obs = observation(elements, app=CHROME, window="Marketplace")
        if self.scroll_area:
            obs.scroll_areas = [element(1, "AXScrollArea", "web page", ops=("SCROLL_UP", "SCROLL_DOWN"),
                                        kind="scroll_area", in_web_area=True, frame=Rect(0, 0, 1000, 800))]  # fmt: skip
        return obs

    def execute(self, action, obs):
        self.actions.append(action.operation + (f" {action.key.id}" if action.key else ""))
        if self.top + self.per_screen >= len(self.titles):
            return ActionResult(False, "scrollbar", "already at the bottom")
        self.top = min(self.top + self.step, max(0, self.loaded - self.per_screen // 2))
        return ActionResult(True, "scrollbar")


def collector(feed, **kwargs):
    kwargs.setdefault("patience", 3)
    return Collector(feed, feed, page_down=key_vocabulary()["PAGE_DOWN"], sleep=lambda _s: None, **kwargs)


DRILLS = [(f"Cordless drill {n}", f"€{20 + n}") for n in range(30)]


def test_collects_every_result_once_with_price_and_photo():
    feed = Feed(DRILLS)
    result = collector(feed).run()
    assert [i.title for i in result.items] == [f"Cordless drill {n} · Mosta" for n in range(30)]
    assert len({i.key for i in result.items}) == 30  # overlapping screens and tracking parameters count once
    first = result.items[0]
    assert first.price == "€20" and first.photo_url == "https://scontent.fbcdn.net/0.jpg"
    assert first.key == "www.facebook.com/marketplace/item/1000"
    assert result.shape == "facebook.com/marketplace/item/{n}"
    assert "does not scroll further" in result.stop_reason
    assert all(a == "SCROLL_DOWN" for a in feed.actions)


def test_waits_for_a_batch_that_arrives_late():
    """Seen live: two scrolls brought nothing, and the next one brought 24 cards."""
    feed = Feed(DRILLS, late=2)
    result = collector(feed, patience=5).run()
    assert len(result.items) == 30


def test_stops_when_results_stop_matching_the_query():
    """Seen live: far down "angle grinder" came coffee grinders, lamps and table saws."""
    titles = [(f"Angle grinder {n}", "€40") for n in range(25)] + [(f"Coffee machine {n}", "€15") for n in range(60)]
    result = collector(Feed(titles), query="angle grinder", patience=10).run()
    assert "mention the query" in result.stop_reason
    assert len(result.items) < 50


def test_limits_and_page_down_without_a_scroll_area():
    feed = Feed(DRILLS, scroll_area=False)
    result = collector(feed, max_items=10).run()
    assert len(result.items) == 10 and "maximum of 10" in result.stop_reason
    assert feed.actions and all(a == "PRESS_KEY PAGE_DOWN" for a in feed.actions)


def test_match_overrides_the_found_shape():
    result = collector(Feed(DRILLS[:6]), match=r"/item/100[0-2]/").run()
    assert [i.key[-4:] for i in result.items] == ["1000", "1001", "1002"]


def test_shapes_prices_and_summary():
    assert url_shape("https://www.facebook.com/marketplace/item/123/?ref=x") == "facebook.com/marketplace/item/{n}"
    assert link_key("https://www.facebook.com/marketplace/item/123/?a=1#b") == "www.facebook.com/marketplace/item/123"
    assert result_shape(Feed(DRILLS[:2]).observe().elements) is None  # fewer than 3 cards: no shape yet
    assert split_price("€1,250 · Deca welder · Fgura") == ("€1,250", "Deca welder · Fgura")
    assert split_price("Free · Old drill") == ("Free", "Old drill")
    assert split_price("Drill, no price") == (None, "Drill, no price")
    assert [price_value(p) for p in ("€1,250", "€47.50", "40 €", "Free", None)] == [1250.0, 47.5, 40.0, None, None]
    feed = Feed([("a", "€1"), ("b", "€10"), ("c", "€30"), ("d", "Free"), ("e", "€50")])
    items = collector(feed).run().items
    assert price_summary(items) == {"priced": 3, "min": 10.0, "median": 30.0, "max": 50.0}


def test_saves_photos(tmp_path):
    items = collector(Feed(DRILLS[:5])).run().items
    items[2].photo_url = "https://bad.example/x"

    def serve(request):
        if "bad.example" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    saver = ImageSaver(httpx.Client(transport=httpx.MockTransport(serve)))
    assert save_photos(items, tmp_path, saver) == 4
    assert items[0].photo and (tmp_path / items[0].photo).read_bytes() == PNG
    assert items[2].photo is None


def test_collect_command_is_offered():
    from jevosx.cli import build_parser

    args = build_parser().parse_args(["collect", "--query", "cordless drill", "--max", "200"])
    assert args.max == 200 and args.query == "cordless drill" and args.patience == 5


def test_stops_on_patience_or_at_the_bottom():
    one = collector(Feed(DRILLS[:8]), patience=1).run()  # everything fits on one screen
    assert len(one.items) == 8 and one.reads == 2 and "1 scroll in a row" in one.stop_reason
    two = collector(Feed(DRILLS[:8]), patience=5).run()
    assert two.reads == 3 and "does not scroll further" in two.stop_reason
