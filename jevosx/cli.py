"""Command-line interface: `jevosx run | observe | ui | diagnose | doctor | memory`."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from . import __version__
from .config import Settings
from .errors import JevOSXError
from .types import Action, clean_text


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = Settings.load(args.config)
        return int(args.handler(args, settings) or 0)
    except JevOSXError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jevosx", description="Coordinate-free macOS automation driven by Jev.")
    parser.add_argument("--version", action="version", version=f"jevosx {__version__}")
    parser.add_argument(
        "--config", help="path to a TOML config file (default: ./jevosx.toml or ~/.config/jevosx/config.toml)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="pursue a goal on this Mac")
    run.add_argument("goal", help="what to do, e.g. 'Open TextEdit, create a document and type \"hello\"'")
    run.add_argument("--app", help="open/switch to this app before the first decision")
    run.add_argument("--slot", action="append", default=[], metavar="NAME=TEXT", help="prepared text for TYPE_TEXT")
    run.add_argument("--max-steps", type=int, help="step budget (default from config)")
    run.add_argument("--expect-text", help="only accept DONE when this text is visible on screen")
    run.add_argument("--dry-run", action="store_true", help="decide but never touch the Mac (implies --max-steps 1)")
    run.add_argument("--step", action="store_true", help="confirm every action interactively")
    run.add_argument("--yes", action="store_true", help="auto-approve actions that would need confirmation")
    run.add_argument("--min-confidence", type=float, help="confidence floor for the gate (default 0.65)")
    run.add_argument(
        "--on-low-confidence",
        choices=["retry", "ask", "stop"],
        help="fallback when Jev is below the floor: re-observe, ask you, or stop (default from config)",
    )
    run.add_argument("--no-memory", action="store_true", help="neither read nor write trajectory memory")
    run.add_argument("--feedback", action="store_true", help="ask whether the run succeeded and store the label")
    run.add_argument("--trace", type=Path, help="write step events as JSON lines to this file")
    run.add_argument("--delay", type=float, default=0.0, help="seconds to wait before starting (switch apps)")
    run.set_defaults(handler=cmd_run)

    observe = sub.add_parser("observe", help="print the structured text map of the frontmost window")
    observe.add_argument("--delay", type=float, default=0.0, help="seconds to wait first (switch to the target app)")
    observe.add_argument("--menus", action="store_true", help="also list menu-bar commands")
    observe.add_argument("--json", action="store_true", help="print the exact Jev state and questions instead")
    observe.add_argument("--goal", default="(inspect only)", help="goal to embed in --json questions")
    observe.set_defaults(handler=cmd_observe)

    ui = sub.add_parser("ui", help="open the local web console: type commands, watch and approve steps")
    ui.add_argument("--demo", action="store_true", help="simulated Mac + simulated decisions (any OS, no API key)")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only)")
    ui.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    ui.set_defaults(handler=cmd_ui)

    diagnose = sub.add_parser("diagnose", help="report exactly what the agent can read from the frontmost window")
    diagnose.add_argument("--delay", type=float, default=3.0, help="seconds to switch to the app first (default 3)")
    diagnose.add_argument("--app", help='check this running app directly, e.g. --app "Google Chrome" (no clicking)')
    diagnose.add_argument("--depth", type=int, default=12, help="raw tree depth to print (default 12)")
    diagnose.set_defaults(handler=cmd_diagnose)

    doctor = sub.add_parser("doctor", help="check permissions, dependencies and configuration")
    doctor.add_argument("--live", action="store_true", help="also send one tiny Jev request to measure latency")
    doctor.set_defaults(handler=cmd_doctor)

    memory = sub.add_parser("memory", help="inspect and manage local trajectory memory")
    msub = memory.add_subparsers(dest="memory_command", required=True)
    msub.add_parser("stats", help="counts and database size").set_defaults(handler=cmd_memory)
    listing = msub.add_parser("list", help="recent episodes")
    listing.add_argument("-n", type=int, default=20)
    listing.set_defaults(handler=cmd_memory)
    show = msub.add_parser("show", help="steps of one episode")
    show.add_argument("episode", type=int)
    show.set_defaults(handler=cmd_memory)
    label = msub.add_parser("label", help="mark an episode as success or failed")
    label.add_argument("episode", type=int)
    label.add_argument("status", choices=["success", "failed"])
    label.set_defaults(handler=cmd_memory)
    forget = msub.add_parser("forget", help="delete one episode")
    forget.add_argument("episode", type=int)
    forget.set_defaults(handler=cmd_memory)
    prune = msub.add_parser("prune", help="keep only the newest N episodes")
    prune.add_argument("--keep", type=int, required=True)
    prune.set_defaults(handler=cmd_memory)
    export = msub.add_parser("export", help="write trajectories as JSON lines")
    export.add_argument("path", type=Path)
    export.set_defaults(handler=cmd_memory)
    load = msub.add_parser("import", help="load trajectories from JSON lines")
    load.add_argument("path", type=Path)
    load.set_defaults(handler=cmd_memory)
    return parser


# ---- run ---------------------------------------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace, settings: Settings) -> int:
    from .agent import Agent, expect_text_verifier

    slots = parse_slots(args.slot)
    interactive = sys.stdin.isatty()
    if args.step:
        settings.safety.confirm_all = True
    if args.min_confidence is not None:
        settings.agent.min_confidence = args.min_confidence
    if args.on_low_confidence:
        settings.agent.low_confidence_policy = args.on_low_confidence
    settings.validate()

    def confirm(action: Action, reason: str) -> bool:
        if args.yes:
            return True
        if not interactive:
            return False
        answer = input(f"  ? {action.describe()}  [{reason}]  allow? [y/N] ").strip().lower()
        return answer in ("y", "yes")

    verifier = None
    if args.expect_text:
        verifier = expect_text_verifier(args.expect_text)

    if args.delay:
        print(f"starting in {args.delay:.1f}s…")
        time.sleep(args.delay)
    agent = Agent.from_settings(settings, dry_run=args.dry_run, use_memory=not args.no_memory, confirm=confirm)
    max_steps = 1 if args.dry_run and not args.max_steps else args.max_steps
    trace = args.trace.open("w", encoding="utf-8") if args.trace else None
    try:
        with agent:
            print(f"▶ {args.goal}")
            for event in agent.iter_run(
                args.goal, app=args.app, text_slots=slots, max_steps=max_steps, verifier=verifier
            ):
                print(format_event(event))
                if trace:
                    trace.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            result = agent.last_result
            assert result is not None
            print(
                f"■ {result.status} after {result.steps} step(s) in {result.elapsed_ms / 1000:.1f}s"
                + (f" · {result.message}" if result.message else "")
            )
            if args.feedback and interactive and result.episode_id is not None and agent.memory is not None:
                answer = input("  did the run achieve the goal? [y/n/skip] ").strip().lower()
                if answer in ("y", "yes", "n", "no"):
                    agent.feedback(result.episode_id, answer.startswith("y"))
                    print("  saved to memory")
            return 0 if result.ok else 1
    finally:
        if trace:
            trace.close()


def parse_slots(values: list[str]) -> dict[str, str]:
    slots = {}
    for value in values:
        name, sep, text = value.partition("=")
        if not sep or not name.strip():
            raise JevOSXError(f"--slot expects NAME=TEXT, got {value!r}")
        slots[name.strip()] = text
    return slots


def format_event(event: Any) -> str:
    decision = event.decision or {}
    parts = [f"  {event.step:>2} {event.status:<14} {event.action or ''}"]
    if decision:
        parts.append(f"p={decision.get('p_operation', 0):.2f} conf={decision.get('gate_confidence', 0):.2f}")
    if event.timings:
        parts.append(" ".join(f"{k[:-3]}={v:.0f}ms" for k, v in event.timings.items()))
    if event.hints:
        parts.append(f"{len(event.hints)} memory hint(s)")
    if event.message:
        parts.append(event.message)
    return " · ".join(parts)


# ---- observe -----------------------------------------------------------------------------------------------------
def cmd_observe(args: argparse.Namespace, settings: Settings) -> int:
    from .executor.keys import key_vocabulary
    from .observer import create_observer, render_text_map
    from .observer.ax import require_trusted
    from .router.policy import JevRouter, state_size
    from .router.text import TextSource

    observer = create_observer(settings.observer)
    require_trusted()
    if args.delay:
        time.sleep(args.delay)
    obs = observer.observe()
    if not args.json:
        print(render_text_map(obs, menus=args.menus))
        return 0

    class _NoClient:
        model = settings.jev.model

    router = JevRouter.from_settings(
        _NoClient(),  # type: ignore[arg-type]
        settings.jev,
        key_vocabulary(settings.keys.custom, settings.keys.disabled),
    )
    text_source = TextSource({})
    space = router.space(obs, text_source, args.goal)
    state, questions = router.build_request(args.goal, obs, space, text_source=text_source)
    payload = {"model": settings.jev.model, "state": state, "questions": questions}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"// state {state_size(state)} bytes · {len(questions)} choice questions", file=sys.stderr)
    return 0


# ---- ui ----------------------------------------------------------------------------------------------------------
def cmd_ui(args: argparse.Namespace, settings: Settings) -> int:
    from .ui import serve

    return serve(settings, demo=args.demo, host=args.host, port=args.port, open_browser=not args.no_browser)


# ---- diagnose ----------------------------------------------------------------------------------------------------
def cmd_diagnose(args: argparse.Namespace, settings: Settings) -> int:
    """One pasteable report: frontmost detection, app AX flags, the raw window tree with error codes and timings,
    and what the walker extracted. Only roles, labels and sizes are printed (no field values)."""
    from .observer import create_observer
    from .observer.apps import app_for_pid
    from .observer.ax import AXNode, require_trusted

    observer = create_observer(settings.observer)
    require_trusted()
    print(f"jevosx {__version__} diagnose · Python {platform.python_version()} · {platform.platform()}")
    if args.delay and not args.app:
        print(f"click on the app to check within {args.delay:.0f} s…", flush=True)
        time.sleep(args.delay)

    def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
        started = time.perf_counter()
        return fn(), (time.perf_counter() - started) * 1000

    def show(value: Any) -> str:
        if isinstance(value, AXNode):
            return "<element>"
        if isinstance(value, list):
            return f"[{len(value)} items]"
        return repr(value)[:60]

    (pid, how), ms = timed(observer.detect_frontmost)
    print(f"frontmost: pid {pid} via {how} ({ms:.0f} ms)")
    if args.app:
        target = observer.find_app(args.app)
        if target is None or target.pid is None:
            print(f"{args.app!r} is not running")
            return 1
        pid = target.pid
        print(f"checking {target.name} (pid {pid}) as requested")
    if pid is None:
        return 1
    app = app_for_pid(pid)
    print(f"app: {app.name} ({app.bundle_id})")
    node = observer.app_node(pid)
    observer.enable_web_accessibility(pid, app, node)
    window: Any = None
    for attribute in ("AXRole", "AXEnhancedUserInterface", "AXManualAccessibility", "AXFocusedWindow",
                      "AXMainWindow", "AXWindows", "AXFocusedUIElement"):  # fmt: skip
        (err, value), ms = timed(partial(node.read, attribute))
        print(f"  app.{attribute}: err={err} {show(value)} ({ms:.0f} ms)")
        if attribute in ("AXFocusedWindow", "AXMainWindow") and window is None and isinstance(value, AXNode):
            window = value
        if attribute == "AXWindows" and window is None and value:
            window = next((w for w in value if isinstance(w, AXNode)), None)
    if window is None:
        print("no window found")
        return 1

    print(f"raw window tree (depth ≤ {args.depth}):")
    lines = 0
    attrs = ("AXRole", "AXSubrole", "AXTitle", "AXDescription", "AXSize", "AXHidden", "AXEnabled")

    def dump(element: Any, depth: int) -> None:
        nonlocal lines
        if lines >= 150:
            return
        (err, values), ms = timed(lambda: element.read_many(attrs))
        (kerr, kids), kms = timed(lambda: element.read("AXChildren"))
        kids = [k for k in (kids or []) if isinstance(k, AXNode)]
        size = values.get("AXSize")
        size_text = f"{size[0]:.0f}x{size[1]:.0f}" if isinstance(size, tuple) and len(size) == 2 else "?"
        label = clean_text(values.get("AXTitle") or values.get("AXDescription"), 40)
        if values.get("AXRole") == "AXGroup" and not label and len(kids) == 1 and not err and depth:
            dump(kids[0], depth)  # unlabeled single-child wrapper: print its content at the same level
            return
        flags = " hidden" if values.get("AXHidden") else ""
        flags += " disabled" if values.get("AXEnabled") is False else ""
        print(
            f"  {'  ' * depth}{values.get('AXRole') or '?'}/{values.get('AXSubrole') or '-'} {label!r} {size_text}"
            f"{flags} kids={len(kids)} err={err}/{kerr} {ms + kms:.0f}ms"
        )
        lines += 1
        if depth < args.depth:
            for kid in kids[:15]:
                dump(kid, depth + 1)

    dump(window, 0)
    try:
        obs, ms = timed(partial(observer.observe, pid))
    except JevOSXError as exc:
        print(f"observe failed: {exc}")
        return 1
    keys = ("visited", "elements", "menu_items", "truncated", "menus_truncated", "walk_ms", "notes")
    stats = {k: obs.stats.get(k) for k in keys}
    print(f"walker: {stats} (observe {ms:.0f} ms)")
    focused = obs.focused_element.describe() if obs.focused_element else None
    print(f"window: {obs.window.title if obs.window else None!r} · focused: {focused}")
    for element in obs.elements[:40]:
        print(f"  {element.describe()}  {{{','.join(element.ops) or '-'}}}")
    return 0


# ---- doctor ------------------------------------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace, settings: Settings) -> int:
    failures = 0

    def check(ok: bool, label: str, hint: str = "") -> None:
        nonlocal failures
        failures += not ok
        print(f"  {'✓' if ok else '✗'} {label}" + (f"\n      → {hint}" if hint and not ok else ""))

    print(f"jevosx {__version__} · Python {platform.python_version()} · {platform.platform()}")
    check(sys.platform == "darwin", "running on macOS", "the observer and executor need macOS; tests run anywhere")
    try:
        from .observer.ax import AX_AVAILABLE, is_trusted

        check(AX_AVAILABLE, "pyobjc Accessibility bindings", "pip install -r requirements.txt")
        trusted = AX_AVAILABLE and is_trusted(prompt=True)
        if AX_AVAILABLE:
            check(
                trusted,
                "Accessibility permission",
                "System Settings › Privacy & Security › Accessibility → enable your terminal/IDE, then restart it",
            )
        if trusted:
            from .observer import create_observer
            from .observer.apps import app_for_pid

            pid, how = create_observer(settings.observer).detect_frontmost()
            name = app_for_pid(pid).name if pid else "none"
            check(pid is not None, f"frontmost app detected: {name} (via {how})", "report this line to the developers")
    except JevOSXError as exc:
        check(False, "Accessibility bindings", str(exc))
    from .executor.input import QUARTZ_AVAILABLE

    check(QUARTZ_AVAILABLE, "pyobjc Quartz (keyboard events)", "pip install pyobjc-framework-Quartz")
    key = settings.jev.api_key()
    from .config import dotenv_candidates

    places = ", ".join(str(p).replace(str(Path.home()), "~") for p in dotenv_candidates())
    check(
        bool(key),
        f"{settings.jev.api_key_env} is set",
        f"export {settings.jev.api_key_env}=… or add {settings.jev.api_key_env}=… to one of: {places}",
    )
    try:
        import h2  # noqa: F401

        check(True, "HTTP/2 support (h2)")
    except ImportError:
        check(False, "HTTP/2 support (h2)", "pip install 'httpx[http2]' (falls back to HTTP/1.1)")
    try:
        from .memory.store import MemoryStore

        if settings.memory.enabled:
            store = MemoryStore(settings.memory_path)
            stats = store.stats()
            store.close()
            check(True, f"memory database {stats['path']} ({stats['episodes']} episodes)")
    except Exception as exc:  # noqa: BLE001
        check(False, "memory database", str(exc))
    if settings.text_model.model:
        check(
            bool(settings.text_model.api_key()),
            f"text model {settings.text_model.model} key",
            settings.text_model.api_key_env,
        )

    if args.live and key:
        from .router.client import JevClient, choice_question

        with JevClient.from_settings(settings.jev) as client:
            latencies = []
            try:
                for _ in range(3):
                    response = client.evaluate(
                        {"light": "green"},
                        {"go": choice_question({"yes": "The light allows driving", "no": "It does not"})},
                    )
                    response.choice("go", ["yes", "no"])
                    latencies.append(response.latency_ms)
                check(True, f"Jev round trip ({response.model}): " + ", ".join(f"{ms:.0f}ms" for ms in latencies))
            except JevOSXError as exc:
                check(False, "Jev round trip", str(exc))
    print("all checks passed" if not failures else f"{failures} check(s) failed")
    return 0 if not failures else 1


# ---- memory ------------------------------------------------------------------------------------------------------
def cmd_memory(args: argparse.Namespace, settings: Settings) -> int:
    from .memory.embedding import HashingEmbedder
    from .memory.store import MemoryStore

    store = MemoryStore(settings.memory_path, HashingEmbedder(settings.memory.dim))
    try:
        command = args.memory_command
        if command == "stats":
            print(json.dumps(store.stats(), indent=2))
        elif command == "list":
            for episode in store.episodes(args.n):
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(episode.started_at))
                print(f"  #{episode.id:<5} {when}  {episode.status:<9} {episode.steps:>3} steps  {episode.goal[:80]}")
        elif command == "show":
            record = store.episode(args.episode)
            if record is None:
                print(f"no episode #{args.episode}")
                return 1
            print(f"#{record.id} {record.status} · {record.goal}")
            for step in store.steps_for([record.id], with_vectors=False):
                print(f"  {step.idx:>3} {step.operation:<12} {step.target_text or '':<50} {step.outcome}")
        elif command == "label":
            store.label_episode(args.episode, args.status)
            print(f"episode #{args.episode} labelled {args.status}")
        elif command == "forget":
            print("deleted" if store.delete_episode(args.episode) else "not found")
        elif command == "prune":
            print(f"removed {store.prune(args.keep)} episode(s)")
        elif command == "export":
            print(f"exported {store.export_jsonl(args.path)} episode(s) to {args.path}")
        elif command == "import":
            print(f"imported {store.import_jsonl(args.path)} episode(s)")
        return 0
    finally:
        store.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
