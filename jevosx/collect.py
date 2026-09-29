"""Collecting every result of a long, endlessly scrolling page: `jevosx collect`.

Seen live: collecting Marketplace results by asking Jev to "scroll down the results" took one model decision per
scroll (5 to 30 s each). A "stop after 2 scrolls without new cards" rule quit early, because Facebook loads results in
batches, and a cap of 80 cut the drills and grinders short. Further down, Facebook keeps loading looser matches
(coffee grinders under "angle grinder"), so "the end" is where the results stop matching, not where the page stops.

The collector needs no model. It reads the page, keeps every result card it has not seen, scrolls one page and reads
again:

- A result is a link whose address has the shape most cards share ("/marketplace/item/<n>/"), found on its own or
  given with --match. It is keyed by that address, so a card seen twice counts once.
- A card's picture is the photo-sized image inside the card's frame; its price is read from the card's text.
- It waits longer after a scroll that brought nothing (batches arrive late), and stops when several scrolls in a row
  bring nothing (--patience), the page cannot scroll further, --max results are kept, or (with --query) most of the
  latest results no longer mention what was searched for.

Nothing is clicked or typed into the page; the only input is scrolling.
"""

from __future__ import annotations

import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from .executor.keys import KeyBinding
from .images import MIN_SIDE, ImageSaveError, ImageSaver
from .types import PRESS_KEY, SCROLL_DOWN, Action, ActionResult, Observation, UIElement

DEFAULT_MAX = 500
DEFAULT_PATIENCE = 5
SETTLE_S = 1.2  # after each scroll, before reading; longer after a scroll that brought nothing
RELEVANCE_WINDOW = 20  # --query: judged on the latest this many results
MIN_RELEVANCE = 0.25  # --query: stop when fewer than this share of them mention the query
_PRICE = re.compile(
    r"(?:[€$£]\s?\d[\d.,]*|\d[\d.,]*\s?(?:€|EUR|eur)\b|\bfree\b)",
    re.IGNORECASE,
)
_DIGITS = re.compile(r"\d+")


class _Observer(Protocol):
    def observe(self) -> Observation: ...


class _Executor(Protocol):
    def execute(self, action: Action, obs: Observation) -> ActionResult: ...


@dataclass
class Item:
    key: str
    url: str
    text: str
    title: str
    price: str | None = None
    photo_url: str | None = None
    photo: str | None = None  # file name, when photos are saved


@dataclass
class CollectResult:
    items: list[Item] = field(default_factory=list)
    reads: int = 0
    scrolls: int = 0
    stop_reason: str = ""
    elapsed_s: float = 0.0
    shape: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "reads": self.reads,
            "scrolls": self.scrolls,
            "elapsed_s": round(self.elapsed_s, 1),
            "shape": self.shape,
            "items": [asdict(item) for item in self.items],
        }


def link_key(url: str) -> str:
    """A result's identity: its address without query or fragment (tracking parameters differ per view)."""
    parts = urlsplit(url)
    return f"{parts.netloc.lower()}{parts.path.rstrip('/')}"


def url_shape(url: str) -> str:
    """ "facebook.com/marketplace/item/123/?ref=x" → "facebook.com/marketplace/item/{n}": what result links share."""
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.")
    return host + _DIGITS.sub("{n}", parts.path.rstrip("/"))


def is_card(element: UIElement) -> bool:
    return (
        element.role == "AXLink"
        and element.in_web_area
        and bool(element.url)
        and element.frame is not None
        and min(element.frame.w, element.frame.h) >= MIN_SIDE
    )


def result_shape(elements: Sequence[UIElement], minimum: int = 3) -> str | None:
    """The address shape most card links share (at least `minimum` of them), with a number in it."""
    counts = Counter(url_shape(e.url or "") for e in elements if is_card(e))
    for shape, count in counts.most_common():
        if count >= minimum and "{n}" in shape:
            return shape
    return None


def split_price(text: str) -> tuple[str | None, str]:
    """ "€50 · Cordless drill · Żabbar" → ("€50", "Cordless drill · Żabbar")."""
    match = _PRICE.search(text)
    if not match:
        return None, text.strip(" ·")
    price = match.group(0).strip()
    rest = (text[: match.start()] + text[match.end() :]).strip(" ·")
    return price, re.sub(r"\s*·\s*·\s*", " · ", rest).strip(" ·")


def price_value(price: str | None) -> float | None:
    """ "€1,250" → 1250.0; "Free" and unreadable prices → None."""
    if not price:
        return None
    digits = re.search(r"\d[\d.,]*", price)
    if not digits:
        return None
    raw = digits.group(0)
    raw = raw.replace(",", "") if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", raw) else raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def photo_for(card: UIElement, elements: Sequence[UIElement]) -> str | None:
    """The largest picture whose centre lies inside the card."""
    if card.frame is None:
        return None
    frame = card.frame
    best: tuple[float, str] | None = None
    for element in elements:
        if element.kind != "image" or not element.url or element.frame is None:
            continue
        cx, cy = element.frame.center
        if frame.x <= cx <= frame.x + frame.w and frame.y <= cy <= frame.y + frame.h:
            area = element.frame.w * element.frame.h
            if best is None or area > best[0]:
                best = (area, element.url)
    return best[1] if best else None


def mentions(text: str, words: Sequence[str]) -> bool:
    """Whether the text mentions any query word (its first five letters, so "drills" matches "drill")."""
    lowered = text.lower()
    return any(word[:5] in lowered for word in words)


class Collector:
    def __init__(
        self,
        observer: _Observer,
        executor: _Executor,
        *,
        page_down: KeyBinding,
        match: str | None = None,
        query: str | None = None,
        max_items: int = DEFAULT_MAX,
        patience: int = DEFAULT_PATIENCE,
        settle_s: float = SETTLE_S,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        progress: Callable[[str], None] | None = None,
    ):
        self.observer = observer
        self.executor = executor
        self.page_down = page_down
        self.pattern = re.compile(match) if match else None
        self.words = [w.lower() for w in re.findall(r"\w{3,}", query or "")]
        self.max_items = max(1, max_items)
        self.patience = max(1, patience)
        self.settle_s = settle_s
        self.sleep = sleep
        self.clock = clock
        self.progress = progress or (lambda _message: None)

    def run(self) -> CollectResult:
        result = CollectResult()
        started = self.clock()
        seen: dict[str, Item] = {}
        empty = at_bottom = 0
        while True:
            obs = self.observer.observe()
            result.reads += 1
            if result.shape is None and self.pattern is None:
                result.shape = result_shape(obs.elements)
            added = self._harvest(obs, seen, result)
            empty = 0 if added else empty + 1
            self.progress(f"read {result.reads}: +{added} (total {len(result.items)})")
            reason = self._stop(result, empty, at_bottom)
            if reason:
                result.stop_reason = reason
                break
            scrolled = self._scroll(obs)
            result.scrolls += 1
            at_bottom = at_bottom + 1 if not scrolled.ok and "bottom" in scrolled.detail else 0
            self.sleep(self.settle_s * (1 + min(empty, 3)))  # batches arrive late: wait longer when nothing came
        result.elapsed_s = self.clock() - started
        return result

    def _stop(self, result: CollectResult, empty: int, at_bottom: int) -> str:
        if len(result.items) >= self.max_items:
            return f"kept the maximum of {self.max_items} results"
        if at_bottom >= 2 and empty:
            return "the page does not scroll further"
        if empty >= self.patience:
            return f"{empty} scroll{'' if empty == 1 else 's'} in a row brought no new results"
        if self.words and len(result.items) >= RELEVANCE_WINDOW:
            latest = result.items[-RELEVANCE_WINDOW:]
            share = sum(mentions(item.text, self.words) for item in latest) / len(latest)
            if share < MIN_RELEVANCE:
                return f"only {share:.0%} of the latest {len(latest)} results mention the query"
        return ""

    def _harvest(self, obs: Observation, seen: dict[str, Item], result: CollectResult) -> int:
        added = 0
        for element in obs.elements:
            if not is_card(element) or not self._wanted(element.url or "", result.shape):
                continue
            key = link_key(element.url or "")
            item = seen.get(key)
            if item is not None:
                if item.photo_url is None:  # the picture may load after the card
                    item.photo_url = photo_for(element, obs.elements)
                continue
            price, title = split_price(element.label)
            item = Item(key, element.url or "", element.label, title, price, photo_for(element, obs.elements))
            seen[key] = item
            result.items.append(item)
            added += 1
            if len(result.items) >= self.max_items:
                break
        return added

    def _wanted(self, url: str, shape: str | None) -> bool:
        if self.pattern is not None:
            return bool(self.pattern.search(url))
        return shape is not None and url_shape(url) == shape

    def _scroll(self, obs: Observation) -> ActionResult:
        """Scroll the page's own scroll area by a page; without one, press Page Down in the window."""
        areas = [a for a in obs.scroll_areas if a.in_web_area and a.frame is not None]
        if areas:
            area = max(areas, key=lambda a: a.frame.h if a.frame else 0.0)
            return self.executor.execute(Action(SCROLL_DOWN, element=area), obs)
        return self.executor.execute(Action(PRESS_KEY, key=self.page_down), obs)


def save_photos(items: Sequence[Item], folder: Path, saver: ImageSaver | None = None) -> int:
    """Download each item's picture as <n>.jpg-ish into `folder`; returns how many were saved."""
    saver = saver or ImageSaver()
    saved = 0
    for number, item in enumerate(items, start=1):
        if not item.photo_url:
            continue
        try:
            path = saver.save(item.photo_url, folder, f"{number:03d}")
        except ImageSaveError:
            continue
        item.photo = path.name
        saved += 1
    return saved


def price_summary(items: Sequence[Item]) -> dict[str, float | int] | None:
    """Count, min, median and max of real prices (€0/€1 "ask" prices and "Free" left out)."""
    values = sorted(v for v in (price_value(i.price) for i in items) if v is not None and v > 1)
    if not values:
        return None
    return {"priced": len(values), "min": values[0], "median": statistics.median(values), "max": values[-1]}
