"""Saving pictures from the web into a folder: "find 5 photos of golden retrievers and save them in a folder".

Doing it by hand takes about seven steps per picture: open the picture, right-click it, choose "Save Image As…",
go to the folder in the dialog, create it when it is missing, name the file, click Save. Each one was a separate Jev
decision, and each one could go wrong. JevOSX does it in one:

- The observer offers pictures on web pages that are big enough to be photos (not icons) and whose address the page
  gives (Accessibility reports it as the image's AXURL).
- Jev picks which picture with SAVE_IMAGE. Which of several equally good pictures is a matter of taste, so that
  choice is not gated (see router/policy.py).
- This module downloads it into the folder and never replaces a file. The folder must be inside your home folder.

The folder and how many pictures come from the request ("a folder called dogs on my desktop", "5 photos"). When the
request does not say, pictures go to ~/Pictures/<what they show> and five are saved. When enough are saved, the run
is done: no model has to notice that.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from glob import escape as glob_escape
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, quote_plus, unquote, urlsplit

import httpx

from .errors import JevOSXError

if TYPE_CHECKING:
    from .types import UIElement

MIN_SIDE = 60  # points: smaller pictures are icons, avatars and logos
MAX_BYTES = 25 * 1024 * 1024
TIMEOUT_S = 15.0
DEFAULT_COUNT = 5
MAX_COUNT = 50
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko)"
EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/heic": ".heic",
    "image/avif": ".avif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
}
_PICTURES = r"(?:photos?|pictures?|pics?|images?|wallpapers?|photographs?)"
WANTS_IMAGES = re.compile(
    rf"\b(?:save|download|collect|grab|keep|store|get)\b.*\b{_PICTURES}\b|\b{_PICTURES}\b.*\b(?:save|download|folder)\b",
    re.IGNORECASE,
)
_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "a single": 1, "two": 2, "a couple of": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a few": 3, "some": DEFAULT_COUNT, "a dozen": 12, "twenty": 20,
}  # fmt: skip
_COUNT = re.compile(
    rf"\b(?P<n>\d{{1,2}}|{'|'.join(sorted((re.escape(k) for k in _NUMBERS), key=len, reverse=True))})\s+"
    rf"(?:(?:\w+[- ]){{0,3}}?){_PICTURES}\b",
    re.IGNORECASE,
)
_TOPIC = re.compile(
    rf"\b{_PICTURES}\s+(?:of|about|showing|with)\s+(?P<topic>.+?)"
    r"(?=\s+(?:and|then|to|into|in|on|from|at|for)\b|[,.;!]|$)",
    re.IGNORECASE,
)
_TOPIC_BEFORE = re.compile(rf"\b(?P<topic>[\w-]+(?:\s+[\w-]+)?)\s+{_PICTURES}\b", re.IGNORECASE)
_FOLDER_NAME = re.compile(
    r"\bfolder\s+(?:called|named|titled)\s+[\"'“]?(?P<name>[^\"'”,;.]+?)[\"'”]?"
    r"(?=\s+(?:on|in|at|inside|under)\b|[,;.!]|$)",
    re.IGNORECASE,
)
_FOLDER_QUOTED = re.compile(r"[\"“'](?P<name>[^\"”']{1,60})[\"”']\s+folder\b", re.IGNORECASE)
_FOLDER_PATH = re.compile(r"(?P<path>~/[^\s,;\"']+)")
_PLACES = (
    (re.compile(r"\b(?:on|to)\s+(?:my|the)\s+desktop\b|\bdesktop\b", re.IGNORECASE), "~/Desktop"),
    (re.compile(r"\bdownloads?\b", re.IGNORECASE), "~/Downloads"),
    (re.compile(r"\bdocuments\b", re.IGNORECASE), "~/Documents"),
    (re.compile(r"\bpictures\s+folder\b|\bin\s+(?:my\s+)?pictures\b", re.IGNORECASE), "~/Pictures"),
)
_STOP = frozenset({"some", "the", "a", "an", "few", "my", "nice", "good", "cool", "best", "any", "random", "more"})
FOLDER_NAMES = ("folder", "directory", "save_folder", "save_to", "destination", "save_location", "location")
COUNT_NAMES = ("count", "number", "how_many", "quantity")


class ImageSaveError(JevOSXError):
    """A picture could not be saved (not an image, too large, unreachable, or a folder outside your home)."""


@dataclass
class ImageTask:
    """What an image-saving request asks for, and what has been saved so far in this run."""

    folder: Path
    count: int
    topic: str = ""
    saved: list[Path] = field(default_factory=list)
    saved_urls: set[str] = field(default_factory=set)
    pages: set[str] = field(default_factory=set)  # pages a picture was saved from: sources for the rest
    scrolls: int = 0  # scrolled for more pictures since the last one was saved
    loading: int = 0  # reads left to wait for a page of pictures that is still loading or being rebuilt
    loading_results: bool = False  # the picture results' address was typed: wait until they are in front

    def is_source(self, page_url: str | None) -> bool:
        """A page to keep saving from: the picture results for the topic, or a page Jev already saved from."""
        return is_results_page(page_url, self.topic) or (bool(page_url) and page_key(page_url) in self.pages)

    @property
    def complete(self) -> bool:
        return len(self.saved) >= self.count

    def progress(self) -> str:
        return f"saved {len(self.saved)} of {self.count} pictures into {display_path(self.folder)}"


def wants_images(goal: str) -> bool:
    return bool(WANTS_IMAGES.search(goal))


def image_task(goal: str, values: dict[str, str] | None = None, *, home: Path | None = None) -> ImageTask | None:
    """The folder, count and topic of an image-saving goal; None when the goal does not ask to save pictures.
    Planner values (folder: ~/Desktop/dogs, count: 3) win over what the patterns find."""
    if not wants_images(goal):
        return None
    values = values or {}
    home = home or Path.home()
    topic = image_topic(goal) or next((v for k, v in values.items() if k in ("search", "query", "topic")), "")
    folder_value = next((values[k] for k in FOLDER_NAMES if values.get(k)), None)
    folder = resolve_folder(folder_value, home) if folder_value else None
    return ImageTask(folder=folder or folder_from_goal(goal, topic, home), count=image_count(goal, values), topic=topic)


def image_count(goal: str, values: dict[str, str] | None = None) -> int:
    for name in COUNT_NAMES:
        raw = (values or {}).get(name, "")
        if raw.strip().isdigit():
            return max(1, min(MAX_COUNT, int(raw)))
    match = _COUNT.search(goal)
    if not match:
        return DEFAULT_COUNT
    n = match.group("n").lower()
    return max(1, min(MAX_COUNT, int(n) if n.isdigit() else _NUMBERS.get(n, DEFAULT_COUNT)))


def image_topic(goal: str) -> str:
    """What the pictures should show: "photos of golden retrievers" → "golden retrievers", "cat pictures" → "cat"."""
    match = _TOPIC.search(goal)
    if match:
        topic = match.group("topic")
    else:
        before = [m for m in _TOPIC_BEFORE.finditer(goal) if m.group("topic").split()[-1].lower() not in _STOP]
        topic = before[0].group("topic") if before else ""
    words = [w for w in re.findall(r"[\w'-]+", topic) if w.lower() not in _STOP | _NUMBERS.keys() and not w.isdigit()]
    return " ".join(words)[:60]


def folder_from_goal(goal: str, topic: str, home: Path) -> Path:
    path = _FOLDER_PATH.search(goal)
    if path:
        resolved = resolve_folder(path.group("path"), home)
        if resolved is not None:
            return resolved
    base = next((where for pattern, where in _PLACES if pattern.search(goal)), "~/Pictures")
    named = _FOLDER_NAME.search(goal) or _FOLDER_QUOTED.search(goal)
    name = named.group("name").strip() if named else (topic.title() if topic else "JevOSX Pictures")
    return resolve_folder(f"{base}/{safe_name(name)}", home) or home / "Pictures" / "JevOSX Pictures"


def resolve_folder(value: str, home: Path) -> Path | None:
    """A folder inside the home folder, or None. "~/Desktop/dogs", "Desktop/dogs" and "/Users/me/dogs" all work."""
    raw = value.strip().strip("\"'")
    if not raw:
        return None
    if raw.startswith("~"):
        path = home / raw[1:].lstrip("/")
    elif raw.startswith("/"):
        path = Path(raw)
    else:
        path = home / raw
    path = Path(*[part for part in path.parts if part not in ("", ".")])
    try:
        resolved = path.resolve()
        home_resolved = home.resolve()
    except OSError:
        return None
    if resolved == home_resolved or home_resolved not in resolved.parents or ".." in path.parts:
        return None
    return resolved


def safe_name(text: str) -> str:
    name = re.sub(r"[/\\:*?\"<>|\x00-\x1f]+", " ", text).strip(" .")
    return re.sub(r"\s+", " ", name)[:60] or "JevOSX Pictures"


def display_path(path: Path, home: Path | None = None) -> str:
    home = home or Path.home()
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def search_address(topic: str) -> str:
    """Google's picture results for the topic, so the browser lands on pictures in one step."""
    return f"https://www.google.com/search?q={quote_plus(topic)}&udm=2"


def image_plan(task: ImageTask) -> list[str]:
    """The steps for every picture-saving goal. The on-device reader was seen live writing "1. open Finder ·
    2. search: dogs · 3. count: 3" for one, and Jev followed it into Finder twice."""
    return [
        "Open the web browser",
        "Type the picture_search address into the browser's address bar",
        f"SAVE_IMAGE one picture per step until {task.count} are saved (no Finder, menus or dialogs)",
    ]


def is_results_page(url: str | None, topic: str) -> bool:
    """Google's picture results for this topic (the page picture_search opens): every picture there fits."""
    if not url or not topic:
        return False
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if not (host.startswith(("www.google.", "google.")) and parts.path == "/search"):
        return False
    query = parse_qs(parts.query)
    if "2" not in query.get("udm", []) and "isch" not in query.get("tbm", []):
        return False
    searched = " ".join(query.get("q", [])).lower()
    return all(word in searched for word in topic.lower().split())


def page_key(url: str | None) -> str:
    """A page without its fragment: scrolling or a gallery overlay must not make it another page."""
    return (url or "").split("#", 1)[0]


def is_thumbnail(url: str | None, page_url: str | None = None) -> bool:
    """Google's small copy of a picture on its results: encrypted-tbn0.gstatic.com (about 500 pixels wide), or,
    for the first tiles while the page loads, an inline data: picture (seen live: 246 by 164 pixels)."""
    if (url or "").startswith("data:"):
        page = urlsplit(page_url or "")
        return page.netloc.lower().startswith(("www.google.", "google.")) and page.path == "/search"
    return bool(re.fullmatch(r"encrypted-tbn\d*\.gstatic\.com", urlsplit(url or "").netloc.lower()))


_WIKIMEDIA_THUMB = re.compile(
    r"(?P<base>/[^/]+/[^/]+)/thumb/(?P<file>[0-9a-f]/[0-9a-f]{2}/(?P<name>[^/]+))/[^/]*?\d+px-[^/]+"
)
_WIKIMEDIA_KEEPS = (".jpg", ".jpeg", ".png", ".gif", ".webp")  # not .svg, .pdf or .tif: the thumbnail is the picture


def wikimedia_original(url: str | None) -> str | None:
    """The original file behind a Wikimedia thumbnail: upload.wikimedia.org/wikipedia/commons/thumb/a/a8/X.jpg/
    330px-X.jpg → upload.wikimedia.org/wikipedia/commons/a/a8/X.jpg. None for any other address. Seen live: a
    picture saved from Wikipedia was its 330 by 550 thumbnail, served from thumb.wikimedia.org with a ?utm_source
    query when it came through Google's preview."""
    parts = urlsplit(url or "")
    if parts.netloc.lower() not in ("upload.wikimedia.org", "thumb.wikimedia.org") or parts.scheme not in (
        "http",
        "https",
    ):
        return None
    match = _WIKIMEDIA_THUMB.fullmatch(parts.path)
    if not match or not unquote(match.group("name")).lower().endswith(_WIKIMEDIA_KEEPS):
        return None
    return f"https://upload.wikimedia.org{match.group('base')}/{match.group('file')}"


def imgres_target(url: str | None) -> str | None:
    """The picture a Google /imgres link points at (its imgurl), or None for any other link."""
    parts = urlsplit(url or "")
    if not (parts.netloc.lower().startswith(("www.google.", "google.")) and parts.path == "/imgres"):
        return None
    return next(iter(parse_qs(parts.query).get("imgurl", [])), None)


def original_of(thumbnail: UIElement, elements: Iterable[UIElement], skip: Iterable[str] = ()) -> str | None:
    """The full-size picture behind one of Google's thumbnails. Once its tile is pressed, the tile's link becomes
    /imgres?imgurl=<original> and the preview beside the results shows the original, both with the tile's label."""
    label = thumbnail.label
    if not label or label == "image":
        return None
    skipped = set(skip) | {thumbnail.url}
    linked: list[str] = []
    shown: list[tuple[float, str]] = []
    for element in elements:
        if element.label != label or not element.url:
            continue
        if element.role == "AXLink":
            target = imgres_target(element.url)
            if target:
                linked.append(target)
        elif element.kind == "image":
            area = element.frame.w * element.frame.h if element.frame is not None else 0.0
            shown.append((area, element.url))
    candidates = linked + [url for _, url in sorted(shown, reverse=True)]
    return next(
        (u for u in candidates if u not in skipped and not is_thumbnail(u) and urlsplit(u).scheme in ("http", "https")),
        None,
    )


def savable_url(url: str | None) -> bool:
    if not url:
        return False
    if url.startswith("data:image/"):
        return ";base64," in url[:80]
    scheme = urlsplit(url).scheme
    return scheme in ("http", "https")


class ImageSaver:
    """Downloads one picture into a folder. `client` is injectable for tests (an httpx.Client)."""

    def __init__(self, client: httpx.Client | None = None, *, max_bytes: int = MAX_BYTES):
        self._client = client
        self.max_bytes = max_bytes

    def save(self, url: str, folder: Path, stem: str, *, referer: str | None = None) -> Path:
        if not savable_url(url):
            raise ImageSaveError("this picture has no address that can be downloaded")
        data, content_type = self._data_uri(url) if url.startswith("data:") else self._download(url, referer)
        extension = EXTENSIONS.get(content_type) or _sniff(data)
        if extension is None:
            raise ImageSaveError(f"not a picture ({content_type or 'unknown type'})")
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ImageSaveError(f"cannot create {display_path(folder)}: {exc.strerror or exc}") from exc
        path = _free_path(folder, safe_name(stem) or "picture", extension)
        try:
            with path.open("xb") as out:  # "x": never replace a file that is already there
                out.write(data)
        except OSError as exc:
            raise ImageSaveError(f"cannot write {path.name}: {exc.strerror or exc}") from exc
        return path

    def _data_uri(self, url: str) -> tuple[bytes, str]:
        header, _, payload = url.partition(",")
        content_type = header[5:].split(";")[0].lower()
        try:
            data = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise ImageSaveError("the picture's data is damaged") from exc
        if len(data) > self.max_bytes:
            raise ImageSaveError("the picture is too large")
        return data, content_type

    def _download(self, url: str, referer: str | None) -> tuple[bytes, str]:
        headers = {"User-Agent": USER_AGENT, "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"}
        if referer:
            headers["Referer"] = referer
        client = self._client or httpx.Client(follow_redirects=True, timeout=TIMEOUT_S)
        try:
            with client.stream("GET", url, headers=headers) as response:
                if response.status_code >= 400:
                    raise ImageSaveError(f"the site answered {response.status_code}")
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > self.max_bytes:
                        raise ImageSaveError("the picture is too large")
                    chunks.append(chunk)
        except httpx.HTTPError as exc:
            raise ImageSaveError(f"could not download the picture: {type(exc).__name__}") from exc
        finally:
            if self._client is None:
                client.close()
        return b"".join(chunks), content_type


def _sniff(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return None


def _free_path(folder: Path, stem: str, extension: str) -> Path:
    for n in range(1, 10_000):  # numbered by name only: "red tulips 1.jpg" and "red tulips 2.webp", not two 1s
        path = folder / f"{stem} {n}{extension}"
        if not path.exists() and not any(folder.glob(f"{glob_escape(stem)} {n}.*")):
            return path
    raise ImageSaveError(f"too many files named {stem} in {display_path(folder)}")


def file_stem(task: ImageTask, url: str) -> str:
    """Files are named after what they show ("golden retrievers 1.jpg"), else after the address."""
    if task.topic:
        return task.topic
    name = Path(unquote(urlsplit(url).path)).stem if not url.startswith("data:") else ""
    return name[:40] or "picture"
