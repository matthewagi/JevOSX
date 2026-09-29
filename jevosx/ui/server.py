"""Local web console for JevOSX: type commands, watch decisions stream in, approve risky or low-confidence steps.

Standard library only (http.server + Server-Sent Events). Security model:
- binds to 127.0.0.1 by default;
- every /api call needs the per-session token (header `X-JevOSX-Token`, or `?token=` for the event stream), so
  other web pages in your browser cannot drive your Mac through it (custom headers force a CORS preflight that
  this server never approves);
- the Host header must be the loopback address the server listens on (blocks DNS-rebinding);
- JSON bodies are size-limited and strictly validated.
"""

from __future__ import annotations

import contextlib
import copy
import hmac
import json
import logging
import os
import queue
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .. import __version__
from ..agent import Agent, expect_text_verifier
from ..config import Settings
from ..errors import JevOSXError
from ..executor.base import DryRunExecutor, Executor
from ..logins import LoginStore
from ..memory.store import MemoryStore
from ..planner import Planner
from ..router.policy import JevRouter, element_state
from ..types import Action, Observation
from ..writer import TextWriter, WriterStatus, create_writer

log = logging.getLogger("jevosx.ui")
STATIC = Path(__file__).with_name("static")
MAX_BODY = 64 * 1024
APPROVAL_TIMEOUT_S = 300.0


# ---- events -------------------------------------------------------------------------------------------------------
class EventBus:
    """Fan-out of UI events to every connected event stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[dict[str, Any]]] = []
        self._next_id = 0

    def publish(self, kind: str, **data: Any) -> dict[str, Any]:
        with self._lock:
            self._next_id += 1
            event = {"id": self._next_id, "kind": kind, "ts": time.time(), **data}
            for subscriber in list(self._subscribers):
                try:
                    subscriber.put_nowait(event)
                except queue.Full:
                    self._subscribers.remove(subscriber)  # a stalled client: drop it, it will resync on reconnect
        return event

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        subscriber: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)


# ---- runs ---------------------------------------------------------------------------------------------------------
@dataclass
class RunOptions:
    goal: str
    app: str | None = None
    expect_text: str | None = None
    dry_run: bool = False
    max_steps: int | None = None
    min_confidence: float | None = None
    low_confidence_policy: str | None = None
    slots: dict[str, str] = field(default_factory=dict)
    use_memory: bool = True

    @classmethod
    def from_json(cls, data: Any) -> RunOptions:
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        goal = data.get("goal")
        if not isinstance(goal, str) or not goal.strip() or len(goal) > 2000:
            raise ValueError("goal must be a non-empty string (max 2000 characters)")

        def text(name: str) -> str | None:
            value = data.get(name)
            if value in (None, ""):
                return None
            if not isinstance(value, str) or len(value) > 500:
                raise ValueError(f"{name} must be a string")
            return value.strip() or None

        max_steps = data.get("max_steps")
        if max_steps not in (None, "") and (not isinstance(max_steps, int) or not 1 <= max_steps <= 200):
            raise ValueError("max_steps must be an integer between 1 and 200")
        floor = data.get("min_confidence")
        if floor not in (None, "") and (not isinstance(floor, int | float) or not 0 <= floor <= 1):
            raise ValueError("min_confidence must be between 0 and 1")
        policy = data.get("low_confidence_policy") or None
        if policy not in (None, "retry", "ask", "stop"):
            raise ValueError("low_confidence_policy must be retry, ask or stop")
        slots = data.get("slots") or {}
        if not isinstance(slots, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in slots.items()):
            raise ValueError("slots must map names to text")
        return cls(
            goal=goal.strip(),
            app=text("app"),
            expect_text=text("expect_text"),
            dry_run=bool(data.get("dry_run", False)),
            max_steps=max_steps or None,
            min_confidence=float(floor) if floor not in (None, "") else None,
            low_confidence_policy=policy,
            slots={k.strip(): v for k, v in slots.items() if k.strip()},
            use_memory=bool(data.get("use_memory", True)),
        )


@dataclass
class Components:
    observer: Any
    executor: Executor
    router: JevRouter
    memory: MemoryStore | None
    text_writer: TextWriter | None
    demo: bool
    writer_status: WriterStatus | None = None
    logins: LoginStore | None = None

    def close(self) -> None:
        self.router.client.close()
        if self.text_writer is not None:
            self.text_writer.close()
        if self.memory is not None:
            self.memory.close()


def build_components(settings: Settings, *, demo: bool) -> Components:
    from ..executor import key_vocabulary
    from ..memory.embedding import HashingEmbedder
    from ..router.client import JevClient

    keys = key_vocabulary(settings.keys.custom, settings.keys.disabled)
    embedder = HashingEmbedder(settings.memory.dim)
    if demo:
        from ..writer.simulated import SimulatedWriter
        from .demo import DemoDesktop, SimulatedJev

        desktop = DemoDesktop()
        router = JevRouter.from_settings(SimulatedJev().client(), settings.jev, keys)
        memory = MemoryStore(":memory:", embedder) if settings.memory.enabled else None
        status = WriterStatus("simulated", True, "available", "demo writer with canned text")
        return Components(
            desktop,
            desktop,
            router,
            memory,
            SimulatedWriter(),
            demo=True,
            writer_status=status,
            logins=demo_logins(),
        )

    from ..executor import create_executor
    from ..observer import create_observer
    from ..observer.ax import require_trusted

    observer = create_observer(settings.observer)
    require_trusted()
    executor = create_executor(settings.executor, observer.frontmost_pid)
    router = JevRouter.from_settings(JevClient.from_settings(settings.jev), settings.jev, keys)
    memory = MemoryStore(settings.memory_path, embedder) if settings.memory.enabled else None
    writer, status = create_writer(settings, notify=lambda message: log.warning("%s", message))
    logins = LoginStore(settings.logins.index_path) if settings.logins.enabled else None
    return Components(observer, executor, router, memory, writer, demo=False, writer_status=status, logins=logins)


@dataclass
class _Approval:
    event: threading.Event = field(default_factory=threading.Event)
    allowed: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    answer: str = ""  # a hand-off answered in words instead of on screen
    answers: dict[str, str] = field(default_factory=dict)  # the questions asked before starting


def code_version() -> float:
    """Newest modification time of the installed JevOSX code (a `git pull` changes it)."""
    root = Path(__file__).resolve().parents[1]
    newest = 0.0
    for pattern in ("*.py", "*.swift", "*.html"):
        for path in root.rglob(pattern):
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
    return newest


class RunManager:
    """Owns the long-lived components and runs one agent at a time on a worker thread."""

    def __init__(
        self,
        settings: Settings,
        *,
        demo: bool,
        bus: EventBus,
        builder: Callable[[Settings, bool], Components] | None = None,
    ):
        self.settings = settings
        self.demo = demo
        self.bus = bus
        self._builder = builder or (lambda s, d: build_components(s, demo=d))
        self._components: Components | None = None
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._approvals: dict[str, _Approval] = {}
        self._agent: Agent | None = None
        self._last_obs: Observation | None = None
        self.current: dict[str, Any] | None = None
        self.history: deque[dict[str, Any]] = deque(maxlen=20)
        self._started_code = code_version()

    # ---- lifecycle ----------------------------------------------------------------------------------------------
    def components(self) -> Components:
        with self._lock:
            if self._components is None:
                self._components = self._builder(self.settings, self.demo)
            return self._components

    def close(self) -> None:
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._components is not None:
            self._components.close()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- commands -----------------------------------------------------------------------------------------------
    def start(self, options: RunOptions) -> str:
        with self._lock:
            if self.running:
                raise RuntimeError("a run is already in progress; stop it first")
            components = self.components()
            settings = copy.deepcopy(self.settings)
            if options.min_confidence is not None:
                settings.agent.min_confidence = options.min_confidence
            if options.low_confidence_policy:
                settings.agent.low_confidence_policy = options.low_confidence_policy
            if self.demo:
                settings.agent.fallback_log = ""
            settings.validate()
            writer = components.text_writer
            planner = Planner(writer) if writer is not None and settings.agent.plan == "auto" else None
            agent = Agent(
                observer=components.observer,
                executor=DryRunExecutor() if options.dry_run else components.executor,
                router=components.router,
                settings=settings,
                memory=components.memory if options.use_memory else None,
                text_writer=components.text_writer,
                planner=planner,
                logins=components.logins,
                confirm=self._confirm,
                handoff=self._handoff,
                clarify=self._clarify,
            )
            run_id = secrets.token_hex(4)
            self.current = {
                "id": run_id,
                "goal": options.goal,
                "options": options.__dict__,
                "status": "running",
                "events": [],
                "result": None,
                "started_at": time.time(),
            }
            self._stop.clear()
            self._agent = agent
            self._thread = threading.Thread(target=self._run, args=(run_id, options, agent), daemon=True)
            self.bus.publish("run_started", run=self._public(self.current))
            self._thread.start()
            return run_id

    def stop(self) -> None:
        self._stop.set()
        for approval in list(self._approvals.values()):
            approval.allowed = False
            approval.event.set()

    def approve(
        self, request_id: str, allowed: bool, *, answer: str = "", answers: dict[str, str] | None = None
    ) -> bool:
        approval = self._approvals.get(request_id)
        if approval is None:
            return False
        approval.allowed = allowed
        approval.answer = answer.strip()[:2000]
        approval.answers = {k: v.strip()[:2000] for k, v in (answers or {}).items()}
        approval.event.set()
        return True

    # ---- queries ------------------------------------------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "current": self._public(self.current) if self.current else None,
            "history": [self._public(r) for r in self.history],
            "pending_approvals": [a.info for a in self._approvals.values()],
            "running": self.running,
        }

    def observe(self) -> dict[str, Any]:
        obs = self._last_obs if self.running else None
        if obs is None:
            if self.running:
                return {"available": False, "message": "waiting for the first observation of this run"}
            obs = self.components().observer.observe()
            self._last_obs = obs
        return observation_json(obs)

    def status(self) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        if self.demo:
            checks.append(
                {"name": "Simulated Mac", "ok": True, "detail": "Demo: simulated apps and simulated decisions"}
            )
        else:
            checks.append({"name": "macOS", "ok": sys.platform == "darwin", "detail": sys.platform})
            try:
                from ..observer.ax import AX_AVAILABLE, is_trusted

                trusted = AX_AVAILABLE and is_trusted(prompt=False)
                detail = "granted" if trusted else "System Settings › Privacy & Security › Accessibility"
                checks.append({"name": "Accessibility", "ok": trusted, "detail": detail})
            except JevOSXError as exc:
                checks.append({"name": "Accessibility", "ok": False, "detail": str(exc)})
            has_key = bool(self.settings.jev.api_key())
            checks.append(
                {"name": "Jev API key", "ok": has_key, "detail": "set" if has_key else "set TYPESAFE_API_KEY in .env"}
            )
        writer = self._components.writer_status if self._components is not None else None
        if writer is not None:
            # Optional feature: an unavailable writer is shown as neutral (None), not as a failure.
            detail = writer.detail + (f" → {writer.hint}" if writer.hint and not writer.available else "")
            checks.append({"name": "Writer", "ok": True if writer.available else None, "detail": detail})
        # A running console keeps the code it started with: say so after an update instead of silently
        # running the old version (this happened on a real Mac after `git pull`).
        stale = code_version() > self._started_code + 1
        if stale:
            checks.insert(
                0,
                {
                    "name": "Restart to update",
                    "ok": False,
                    "detail": "JevOSX was updated after this console started. Stop it with Ctrl+C in Terminal and "
                    "run jevosx ui again.",
                },
            )
        memory: dict[str, Any] = {"enabled": self.settings.memory.enabled}
        if self._components is not None and self._components.memory is not None:
            stats = self._components.memory.stats()
            memory.update(episodes=stats["episodes"], steps=stats["steps"])
        return {
            "version": __version__,
            "demo": self.demo,
            "model": "jev-demo (simulated)" if self.demo else self.settings.jev.model,
            "checks": checks,
            "memory": memory,
            "defaults": {
                "min_confidence": self.settings.agent.min_confidence,
                "low_confidence_policy": self.settings.agent.console_low_confidence_policy,
                "max_steps": self.settings.agent.max_steps,
                "background": self.settings.agent.background,
            },
            "running": self.running,
            "stale": stale,
        }

    def memory(self, limit: int = 30) -> dict[str, Any]:
        store = self.components().memory
        if store is None:
            return {"enabled": False, "episodes": []}
        return {
            "enabled": True,
            "stats": store.stats(),
            "episodes": [e.__dict__ for e in store.episodes(limit)],
        }

    def label(self, episode_id: int, status: str) -> None:
        store = self.components().memory
        if store is None:
            raise ValueError("memory is disabled")
        store.label_episode(episode_id, status)
        self.bus.publish("memory_changed")

    # ---- worker -------------------------------------------------------------------------------------------------
    def _run(self, run_id: str, options: RunOptions, agent: Agent) -> None:
        verifier = expect_text_verifier(options.expect_text) if options.expect_text else None
        max_steps = options.max_steps or (1 if options.dry_run else None)
        try:
            stream = agent.iter_run(
                options.goal, app=options.app, text_slots=options.slots, max_steps=max_steps, verifier=verifier
            )
            for event in stream:
                if agent.last_observation is not None:
                    self._last_obs = agent.last_observation
                payload = event.to_dict()
                if self.current is not None:
                    self.current["events"].append(payload)
                self.bus.publish("step", run_id=run_id, event=payload)
                if self._stop.is_set():
                    stream.close()
                    break
        except Exception as exc:  # noqa: BLE001 - surface every failure in the UI instead of killing the thread
            log.exception("run failed")
            self.bus.publish("run_error", run_id=run_id, message=f"{type(exc).__name__}: {exc}")
        result = agent.last_result.to_dict() if agent.last_result else {"status": "error", "steps": 0, "ok": False}
        if self.current is not None and self.current["id"] == run_id:
            self.current["status"] = result["status"]
            self.current["result"] = result
            self.history.appendleft(self.current)
        self.bus.publish("run_finished", run_id=run_id, result=result)
        if agent.memory is not None:
            self.bus.publish("memory_changed")

    def _confirm(self, action: Action, reason: str) -> bool:
        """Called on the worker thread by the safety policy or the low-confidence 'ask' fallback."""
        category = "low_confidence" if "below the floor" in reason else "safety"
        return self._ask(action.describe(), reason, category).allowed

    def _handoff(self, request: str, _obs: Observation) -> bool | str:
        """ASK_USER: the human does what only they can (2FA code, CAPTCHA…), then presses Continue (or Stop). Or
        they type the answer, which the agent can then type itself."""
        reply = self._ask(
            f"Please {request}.",
            "Do it on the Mac and continue, or type the answer here. Jev looks at the screen again.",
            "handoff",
        )
        observer = self._components.observer if self._components is not None else None
        simulate = getattr(observer, "user_completes_handoff", None)
        if reply.allowed and not reply.answer and self.demo and simulate is not None:
            simulate()  # demo: the simulated person types the code
        if not reply.allowed:
            return False
        return reply.answer or True

    def _clarify(self, questions: list[tuple[str, str]]) -> dict[str, str] | None:
        """Before starting: what only the person can provide. None stops the run; skipped questions stay empty."""
        reply = self._ask(
            "Before I start",
            "A few things only you know. Answer what you can; empty answers are skipped.",
            "questions",
            questions=[{"name": name, "question": question} for name, question in questions],
        )
        return reply.answers if reply.allowed else None

    def _ask(self, action: str, reason: str, category: str, **extra: Any) -> _Approval:
        request_id = secrets.token_hex(4)
        info = {"request_id": request_id, "action": action, "reason": reason, "category": category, **extra}
        approval = _Approval(info=info)
        self._approvals[request_id] = approval
        self.bus.publish("approval", run_id=self.current["id"] if self.current else None, **approval.info)
        deadline = time.monotonic() + APPROVAL_TIMEOUT_S
        while not approval.event.wait(0.25):
            if self._stop.is_set() or time.monotonic() > deadline:
                break
        self._approvals.pop(request_id, None)
        self.bus.publish("approval_resolved", request_id=request_id, allowed=approval.allowed)
        return approval

    @staticmethod
    def _public(run: dict[str, Any]) -> dict[str, Any]:
        return {**run, "events": list(run["events"])}  # snapshot: the worker keeps appending


def observation_json(obs: Observation) -> dict[str, Any]:
    return {
        "available": True,
        "app": {"name": obs.app.name, "bundle_id": obs.app.bundle_id, "pid": obs.app.pid},
        "window": obs.window.title if obs.window else None,
        "windows": [w.title for w in obs.windows],
        "elements": [element_state(e) for e in obs.elements],
        "menu_items": [{"id": f"m{m.index}", "label": m.label, "shortcut": m.shortcut} for m in obs.menu_items],
        "scroll_areas": [{"id": f"s{s.index}", "label": s.label} for s in obs.scroll_areas],
        "text": obs.text[:3000],
        "stats": {k: v for k, v in obs.stats.items() if isinstance(v, int | float | str | bool)},
        "captured_at": obs.captured_at,
    }


class Speaker:
    """Speaks the console's questions and results with the Mac's own voice (`say`: the system voice, which can be a
    Siri voice, set in System Settings › Accessibility › Spoken Content). One sentence at a time; a new one cuts off
    the previous. Returns once the sentence is spoken, so the page can listen right after."""

    MAX_CHARS = 400

    def __init__(self, command: str | None = None) -> None:
        self.command = command if command is not None else (shutil.which("say") if sys.platform == "darwin" else None)
        self._lock = threading.Lock()
        self._current: subprocess.Popen[bytes] | None = None

    @property
    def available(self) -> bool:
        return bool(self.command)

    def say(self, text: str, *, timeout_s: float = 60.0) -> bool:
        text = " ".join(text.split())[: self.MAX_CHARS]
        if not self.command or not text:
            return False
        with self._lock:
            if self._current is not None and self._current.poll() is None:
                self._current.terminate()
            process = subprocess.Popen([self.command, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._current = process
        try:
            process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            process.terminate()
        return process.returncode == 0


def console_file() -> Path:
    """Where a running console leaves its address and token for `jevosx ask` (readable by this user only)."""
    return Path("~/.jevosx/console.json").expanduser()


# ---- HTTP ---------------------------------------------------------------------------------------------------------
class _QuietServer(ThreadingHTTPServer):
    """A browser closing a connection (reload, event-stream reconnect) is normal, not an error to print."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        if isinstance(sys.exc_info()[1], ConnectionError | TimeoutError):
            return
        super().handle_error(request, client_address)


class UIServer:
    def __init__(
        self,
        settings: Settings,
        *,
        demo: bool = False,
        host: str = "127.0.0.1",
        port: int = 8765,
        manager: RunManager | None = None,
    ):
        self.token = secrets.token_urlsafe(18)
        self.demo = demo
        self.speaker = Speaker()
        self.bus = manager.bus if manager else EventBus()
        self.manager = manager or RunManager(settings, demo=demo, bus=self.bus)
        server = self

        class Handler(_Handler):
            ui = server

        self.httpd = _QuietServer((host, port), Handler)
        self.httpd.daemon_threads = True
        address = self.httpd.server_address
        self.host, self.port = str(address[0]), int(address[1])

    @property
    def url(self) -> str:
        host = "127.0.0.1" if self.host in ("0.0.0.0", "") else self.host
        return f"http://{host}:{self.port}/"

    def serve_forever(self) -> None:
        self.httpd.serve_forever(poll_interval=0.25)

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        self.manager.close()
        self.httpd.shutdown()
        self.httpd.server_close()


class _Handler(BaseHTTPRequestHandler):
    ui: UIServer
    server_version = f"jevosx/{__version__}"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature from the base class
        log.debug("%s %s", self.address_string(), format % args)

    # ---- guards -------------------------------------------------------------------------------------------------
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        allowed = {f"127.0.0.1:{self.ui.port}", f"localhost:{self.ui.port}", f"[::1]:{self.ui.port}"}
        if self.ui.host not in ("127.0.0.1", "localhost", "::1", "0.0.0.0", ""):
            allowed.add(f"{self.ui.host}:{self.ui.port}".lower())
        return host in allowed

    def _token_ok(self, query: dict[str, list[str]]) -> bool:
        supplied = self.headers.get("X-JevOSX-Token") or (query.get("token") or [""])[0]
        return hmac.compare_digest(supplied.encode(), self.ui.token.encode())

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status)

    def _body(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    # ---- routes -------------------------------------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if not self._host_ok():
            return self._error(HTTPStatus.FORBIDDEN, "unexpected Host header")
        if url.path in ("/", "/index.html"):
            return self._index()
        if url.path == "/favicon.svg":
            return self._static("favicon.svg", "image/svg+xml")
        if not url.path.startswith("/api/"):
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        if not self._token_ok(query):
            return self._error(HTTPStatus.UNAUTHORIZED, "missing or invalid token")
        manager = self.ui.manager
        try:
            if url.path == "/api/events":
                return self._events()
            if url.path == "/api/status":
                return self._send_json({**manager.status(), "speech": self.ui.speaker.available})
            if url.path == "/api/state":
                return self._send_json(manager.state())
            if url.path == "/api/observe":
                return self._send_json(manager.observe())
            if url.path == "/api/memory":
                return self._send_json(manager.memory())
        except JevOSXError as exc:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        return self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        url = urlparse(self.path)
        if not self._host_ok():
            return self._error(HTTPStatus.FORBIDDEN, "unexpected Host header")
        if not self._token_ok({}):  # POSTs must carry the header: blocks cross-site form posts
            return self._error(HTTPStatus.UNAUTHORIZED, "missing or invalid token")
        manager = self.ui.manager
        try:
            data = self._body()
            if url.path == "/api/run":
                run_id = manager.start(RunOptions.from_json(data))
                return self._send_json({"run_id": run_id}, HTTPStatus.ACCEPTED)
            if url.path == "/api/stop":
                manager.stop()
                return self._send_json({"stopping": manager.running})
            if url.path == "/api/say":
                text = data.get("text")
                if not isinstance(text, str):
                    raise ValueError("text is required")
                return self._send_json({"ok": self.ui.speaker.say(text)})
            if url.path == "/api/approve":
                answer, answers = data.get("answer") or "", data.get("answers") or {}
                if not isinstance(answer, str) or not isinstance(answers, dict):
                    raise ValueError("answer must be text and answers a map of text")
                answers = {str(k): str(v) for k, v in answers.items() if isinstance(v, str | int | float)}
                ok = manager.approve(
                    str(data.get("request_id", "")), bool(data.get("allow", False)), answer=answer, answers=answers
                )
                return self._send_json({"ok": ok}, HTTPStatus.OK if ok else HTTPStatus.NOT_FOUND)
            if url.path == "/api/memory/label":
                episode, status = data.get("episode_id"), data.get("status")
                if not isinstance(episode, int) or status not in ("success", "failed"):
                    raise ValueError("episode_id (int) and status (success|failed) are required")
                manager.label(episode, status)
                return self._send_json({"ok": True})
        except (ValueError, json.JSONDecodeError) as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except RuntimeError as exc:
            return self._error(HTTPStatus.CONFLICT, str(exc))
        except JevOSXError as exc:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        return self._error(HTTPStatus.NOT_FOUND, "not found")

    # ---- responses ----------------------------------------------------------------------------------------------
    def _index(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        boot = json.dumps({"token": self.ui.token, "demo": self.ui.demo, "version": __version__})
        body = html.replace("/*__JEVOSX_BOOT__*/{}", boot).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'",
        )
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name: str, content_type: str) -> None:
        body = (STATIC / name).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _events(self) -> None:
        subscriber = self.ui.bus.subscribe()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.wfile.write(b"retry: 1500\n\n")
            self.wfile.flush()
            while True:
                try:
                    event = subscriber.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                data = json.dumps(event, ensure_ascii=False, default=str)
                self.wfile.write(f"id: {event['id']}\nevent: {event['kind']}\ndata: {data}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.ui.bus.unsubscribe(subscriber)


def serve(settings: Settings, *, demo: bool, host: str, port: int, open_browser: bool) -> int:
    server = UIServer(settings, demo=demo, host=host, port=port)
    link = server.url
    print(f"JevOSX console {'(DEMO: simulated Mac) ' if demo else ''}running at {link}")
    print("Press Ctrl+C to stop.")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: listening beyond loopback; anyone who can reach this port and has the page can drive your Mac")
    if open_browser:
        import webbrowser

        threading.Timer(0.4, lambda: webbrowser.open(link)).start()
    if not demo:
        # Connect to the Mac and prepare the writer (a one-time Swift build) while the page loads, so the first
        # command does not wait for it. Failures are reported again when a run starts.
        threading.Thread(target=_warm_up, args=(server.manager,), daemon=True).start()
    announce(link, server.token)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        server.shutdown()
        with contextlib.suppress(OSError):
            console_file().unlink()
    return 0


def announce(url: str, token: str) -> None:
    """Leave the console's address and token for `jevosx ask` (so a Siri Shortcut can start runs)."""
    path = console_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump({"url": url, "token": token, "pid": os.getpid()}, out)
    except OSError as exc:
        log.info("cannot write %s: %s", path, exc)


def _warm_up(manager: RunManager) -> None:
    try:
        components = manager.components()
    except Exception as exc:  # noqa: BLE001 - only a head start; the first run builds (and reports) again
        log.info("warm-up skipped: %s", exc)
        return
    if components.writer_status is not None:
        print(f"writer {components.writer_status.describe()}")


def demo_logins() -> LoginStore:
    """A throwaway login for the simulated sign-in page (never the real Keychain)."""
    import tempfile

    from ..logins import MemorySecrets

    folder = Path(tempfile.mkdtemp(prefix="jevosx-demo-"))
    store = LoginStore(folder / "logins.json", MemorySecrets())
    store.add("github.com", "octocat", "demo-password-123")
    return store
