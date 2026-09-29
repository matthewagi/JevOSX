"""Command-line interface: `jevosx run | observe | ui | write | login | diagnose | doctor | report | memory`."""

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
    run.add_argument(
        "--min-confidence",
        type=float,
        help="confidence floor for consequential steps (default 0.65); easier steps need less, never more",
    )
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

    mcp = sub.add_parser("mcp", help="connect the Claude app on this Mac to JevOSX (Model Context Protocol, stdio)")
    mcp.add_argument("--install", action="store_true", help="add JevOSX to the Claude desktop app and print the rest")
    mcp.set_defaults(handler=cmd_mcp)

    ask = sub.add_parser("ask", help="give the running console a command (for a Siri Shortcut: Hey Siri, Ask JevOSX)")
    ask.add_argument("goal", nargs="+", help="what to do, e.g. sell the welding gun on Marketplace")
    ask.add_argument("--wait", action="store_true", help="print the steps as they happen and wait for the result")
    ask.add_argument("--timeout", type=float, default=600.0, help="with --wait: give up waiting after this many s")
    ask.set_defaults(handler=cmd_ask)

    diagnose = sub.add_parser("diagnose", help="report exactly what the agent can read from the frontmost window")
    diagnose.add_argument("--delay", type=float, default=3.0, help="seconds to switch to the app first (default 3)")
    diagnose.add_argument("--app", help='check this running app directly, e.g. --app "Google Chrome" (no clicking)')
    diagnose.add_argument("--depth", type=int, default=12, help="raw tree depth to print (default 12)")
    diagnose.set_defaults(handler=cmd_diagnose)

    write = sub.add_parser("write", help="compose text with the writer (Apple's on-device model by default)")
    write.add_argument("request", nargs="?", help='what to write, e.g. "a haiku about the sea"')
    write.add_argument("--check", action="store_true", help="only report whether the writer is available and why")
    write.add_argument("--rebuild", action="store_true", help="recompile the Apple on-device helper")
    write.add_argument("--backend", choices=["auto", "apple", "openai"], help="override [writer] backend")
    write.set_defaults(handler=cmd_write)

    login = sub.add_parser("login", help="save website logins in the macOS Keychain for sign-in tasks")
    lsub = login.add_subparsers(dest="login_command", required=True)
    add = lsub.add_parser("add", help="save a login: jevosx login add github.com")
    add.add_argument("site", help="the sign-in page's site, e.g. github.com (covers its subdomains)")
    add.add_argument("--username", help="asked for when omitted")
    add.add_argument("--password-stdin", action="store_true", help="read the password from stdin (for scripts)")
    add.set_defaults(handler=cmd_login)
    lsub.add_parser("list", help="saved sites and usernames (never passwords)").set_defaults(handler=cmd_login)
    remove = lsub.add_parser("remove", help="forget a saved login (and delete it from the Keychain)")
    remove.add_argument("site")
    remove.add_argument("--username", help="only this username (default: every login for the site)")
    remove.set_defaults(handler=cmd_login)
    match = lsub.add_parser("match", help="which saved login would be offered on this page URL")
    match.add_argument("url", help="e.g. https://github.com/login")
    match.set_defaults(handler=cmd_login)

    doctor = sub.add_parser("doctor", help="check permissions, dependencies and configuration")
    doctor.add_argument("--live", action="store_true", help="also send one tiny Jev request to measure latency")
    doctor.set_defaults(handler=cmd_doctor)

    report = sub.add_parser("report", help="print the last run(s) step by step, for troubleshooting")
    report.add_argument("-n", type=int, default=1, help="how many recent runs (default 1)")
    report.set_defaults(handler=cmd_report)

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

    def handoff(request: str, _obs: Any) -> bool | str:
        print(f"  ⏸ Jev needs you to {request}.")
        answer = input("    Do it on the Mac and press Enter, or type the answer (or stop): ").strip()
        if answer.lower() in ("stop", "s", "q", "quit", "n", "no"):
            return False
        return answer or True

    def clarify(questions: list[tuple[str, str]]) -> dict[str, str] | None:
        print("  Before I start (press Enter to skip a question, type stop to cancel):")
        answers: dict[str, str] = {}
        for name, question in questions:
            answer = input(f"    {question} ").strip()
            if answer.lower() == "stop":
                return None
            answers[name] = answer
        return answers

    agent = Agent.from_settings(
        settings,
        dry_run=args.dry_run,
        use_memory=not args.no_memory,
        confirm=confirm,
        handoff=handoff if interactive else None,
        clarify=clarify if interactive else None,
        notify=_notify,
    )
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


def _notify(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


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


def cmd_mcp(args: argparse.Namespace, settings: Settings) -> int:
    """Serve JevOSX's tools to the Claude app over stdio, or with --install, register it with the Claude apps."""
    from . import mcp_server

    if not args.install:
        return mcp_server.serve()
    path = mcp_server.install_desktop()
    command = " ".join(mcp_server.server_command())
    print(f"added JevOSX to the Claude desktop app: {path}")
    print("quit and reopen Claude, then ask it in a chat, for example: use JevOSX to open Notes")
    print("for Claude Code, run this once:")
    print(f"  claude mcp add --scope user jevosx -- {command}")
    return 0


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    """Start a run in the console that is already open (`jevosx ui`), which then talks it through with you. With
    --wait, print its steps as they happen and exit with its outcome (0 when done)."""
    from .console_client import ConsoleClient, ConsoleNotRunning

    goal = " ".join(args.goal).strip()
    try:
        console = ConsoleClient.find()
        run_id = console.start_run(goal)
    except ConsoleNotRunning as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"started: {goal}", flush=True)
    if not args.wait:
        return 0

    def event(e: dict[str, Any]) -> None:
        message = f" · {e['message']}" if e.get("message") else ""
        print(f"  {e.get('step')} {e.get('status')} {e.get('action') or ''}{message}", flush=True)

    def pending(p: dict[str, Any]) -> None:
        print(f"  … waiting for you in the console: {p['action']} ({p['reason']})", flush=True)

    run = console.follow(run_id, args.timeout, on_event=event, on_pending=pending)
    result = run.get("result")
    if result is None:
        print("still running; stopped waiting (the run goes on in the console)", file=sys.stderr)
        return 3
    print(f"{result.get('status')} after {result.get('steps')} step(s) {result.get('message') or ''}".rstrip())
    return 0 if result.get("status") in ("done", "success") else 1


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
    attrs = ("AXRole", "AXSubrole", "AXTitle", "AXDescription", "AXPosition", "AXSize", "AXHidden", "AXEnabled")

    def dump(element: Any, depth: int) -> None:
        nonlocal lines
        if lines >= 150:
            return
        (err, values), ms = timed(lambda: element.read_many(attrs))
        (kerr, kids), kms = timed(lambda: element.read("AXChildren"))
        kids = [k for k in (kids or []) if isinstance(k, AXNode)]
        size = values.get("AXSize")
        size_text = f"{size[0]:.0f}x{size[1]:.0f}" if isinstance(size, tuple) and len(size) == 2 else "?"
        position = values.get("AXPosition")
        if isinstance(position, tuple) and len(position) == 2:
            size_text += f"@{position[0]:.0f},{position[1]:.0f}"
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
    keys = ("visited", "elements", "menu_items", "truncated", "menus_truncated", "walk_ms", "notes", "skipped")
    stats = {k: obs.stats.get(k) for k in keys}
    print(f"walker: {stats} (observe {ms:.0f} ms)")
    focused = obs.focused_element.describe() if obs.focused_element else None
    print(f"window: {obs.window.title if obs.window else None!r} · focused: {focused}")
    for element in obs.elements[:40]:
        print(f"  {element.describe()}  {{{','.join(element.ops) or '-'}}}")
    return 0


# ---- write -------------------------------------------------------------------------------------------------------
def cmd_write(args: argparse.Namespace, settings: Settings) -> int:
    """Try the writer on its own: `jevosx write "a haiku about rain"`, or `--check` for why it is unavailable."""
    from .errors import TextUnavailableError
    from .writer import create_writer

    if not args.check and not args.request:
        print('usage: jevosx write "a haiku about the sea"   (or --check)', file=sys.stderr)
        return 2
    writer, status = create_writer(settings, notify=_notify, rebuild=args.rebuild, backend=args.backend)
    if args.check or writer is None:
        print(f"writer {status.describe()}")
        if status.hint and not status.available:
            print(f"  → {status.hint}")
        return 0 if status.available else 1
    started = time.perf_counter()
    try:
        text = writer.write(
            {"goal": f"Write {args.request}", "field": {"label": "Document", "role": "textarea", "current_value": ""}}
        )
    except TextUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        writer.close()
    print(text)
    print(f"— {writer.model}, {time.perf_counter() - started:.1f}s", file=sys.stderr)
    return 0


# ---- login -------------------------------------------------------------------------------------------------------
def cmd_login(args: argparse.Namespace, settings: Settings) -> int:
    import getpass

    from .logins import LoginStore, mask
    from .sites import page_host

    store = LoginStore(settings.logins.index_path)
    if args.login_command == "list":
        logins = store.saved()
        if not logins:
            print("no saved logins. Add one with: jevosx login add github.com")
        for login in logins:
            print(f"  {login.host:<32} {login.username}")
        return 0
    if args.login_command == "remove":
        gone = store.remove(args.site, args.username)
        print(f"removed {len(gone)} login(s)" if gone else "nothing saved for that site")
        return 0 if gone else 1
    if args.login_command == "match":
        host = page_host(args.url)
        if host is None:
            print("no: saved logins are only used on https pages (or http on this Mac)")
            return 1
        logins = store.for_url(args.url)
        for login in logins:
            print(f"  {login.host}: {mask(login.username)}")
        if not logins:
            print(f"no saved login matches {host}")
        return 0 if logins else 1
    username = args.username or input("username or email: ").strip()
    if args.password_stdin:
        password = sys.stdin.readline().rstrip("\n")
    else:
        password = getpass.getpass("password (hidden): ")
        if getpass.getpass("again: ") != password:
            print("the passwords differ; nothing saved", file=sys.stderr)
            return 1
    login = store.add(args.site, username, password)
    print(f"saved {login.username} for {login.host} in the macOS Keychain (service {login.service})")
    print(f'  try: jevosx run "log in to {login.host}"   (you approve before the password is typed)')
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
    from .writer import create_writer

    writer, status = create_writer(settings, notify=_notify)
    if writer is not None:
        writer.close()
    # Optional: without a writer JevOSX still types quoted text; it only cannot compose new text ("write a poem").
    mark = "✓" if status.available else "–"
    print(f"  {mark} writer for free-form text: {status.describe()}")
    if status.hint and not status.available:
        print(f"      → optional: {status.hint}")
    # Optional: Claude in the console (the "Claude" switch in jevosx ui).
    has_key = bool(settings.pilot.api_key())
    pilot_state = "off" if not settings.pilot.enabled else settings.pilot.model if has_key else "no ANTHROPIC_API_KEY"
    print(f"  {'✓' if settings.pilot.enabled and has_key else '–'} Claude in the console: {pilot_state}")
    if settings.pilot.enabled and not has_key:
        print("      → optional: add ANTHROPIC_API_KEY=... to ~/JevOSX/.env to talk to Claude in jevosx ui")
    # Optional: the Claude app on this Mac drives JevOSX from its own chat (no API key), through jevosx mcp.
    from .mcp_server import desktop_config_path

    try:
        connected = "jevosx" in json.loads(Path(desktop_config_path()).read_text()).get("mcpServers", {})
    except (OSError, ValueError, AttributeError):
        connected = False
    print(f"  {'✓' if connected else '–'} Claude app connector: {'installed' if connected else 'not installed'}")
    if not connected:
        print("      → optional: jevosx mcp --install lets the Claude app chat drive JevOSX (no API key)")
    if settings.observer.vision != "off" and sys.platform == "darwin":
        from .observer.vision import screen_recording_allowed

        try:
            import Vision  # noqa: F401

            ocr = True
        except ImportError:
            ocr = False
        allowed = screen_recording_allowed(prompt=True)
        mark = "✓" if allowed and ocr else "–"
        print(f"  {mark} vision for apps that draw their own interface (Screen Recording + on-device OCR)")
        if not ocr:
            print("      → optional: pip install pyobjc-framework-Vision")
        elif not allowed:
            print(
                "      → optional: System Settings › Privacy & Security › Screen & System Audio Recording → enable"
                " your terminal/IDE, then restart it"
            )
    if settings.logins.enabled:
        from .logins import LoginError, LoginStore, keychain

        try:
            keychain()
            saved = LoginStore(settings.logins.index_path).saved()
            hosts = ", ".join(sorted({x.host for x in saved})) or "none yet (jevosx login add github.com)"
            print(f"  ✓ website logins in the macOS Keychain: {hosts}")
        except LoginError as exc:
            print(f"  – website logins: {exc}")

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


# ---- report ------------------------------------------------------------------------------------------------------
def cmd_report(args: argparse.Namespace, settings: Settings) -> int:
    """Recent runs with their executed steps and every withheld (low-confidence) decision, in one paste."""
    from datetime import datetime

    from .memory.embedding import HashingEmbedder
    from .memory.store import MemoryStore

    withheld: list[dict[str, Any]] = []
    log_path = Path(settings.agent.fallback_log).expanduser() if settings.agent.fallback_log else None
    if log_path is not None and log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines()[-200:]:
            try:
                record = json.loads(line)
                record["_t"] = datetime.strptime(record["ts"], "%Y-%m-%dT%H:%M:%S%z").timestamp()
                withheld.append(record)
            except (ValueError, KeyError):
                continue
    store = MemoryStore(settings.memory_path, HashingEmbedder(settings.memory.dim))
    try:
        episodes = store.episodes(args.n)
        if not episodes:
            print("no runs recorded yet")
            return 0
        print(f"jevosx {__version__} report · Python {platform.python_version()} · {platform.platform()}")
        for episode in reversed(episodes):
            started = time.strftime("%H:%M:%S", time.localtime(episode.started_at))
            print(f"\n#{episode.id} {started} {episode.status} · {episode.steps} step(s) · {episode.goal!r}")
            if episode.meta.get("plan"):
                print("  plan: " + " · ".join(f"{i}. {s}" for i, s in enumerate(episode.meta["plan"], start=1)))
            if episode.meta.get("values"):
                print("  to type: " + ", ".join(f"{k}={v!r}" for k, v in episode.meta["values"].items()))
            for step in store.steps_for([episode.id], with_vectors=False):
                probability = f" p={step.probability:.2f}" if step.probability is not None else ""
                confidence = f" conf={step.confidence:.2f}" if step.confidence is not None else ""
                print(f"  step {step.idx} {step.operation} {step.target_text or ''}{probability}{confidence}")
                print(f"         in {step.app} · {step.window!r} → {step.outcome}")
            end = (episode.finished_at or time.time()) + 1
            for record in (r for r in withheld if episode.started_at - 1 <= r["_t"] <= end):
                decision = record.get("decision", {})
                print(
                    f"  withheld {record['ts'][11:19]} {decision.get('operation')} {decision.get('target') or ''}"
                    f" conf={record.get('confidence')} (floor {record.get('floor')}) in {record.get('window')!r}"
                )
                if record.get("risk"):
                    print(f"         {record['risk']} · {record.get('resolution')}")
                print(f"         top: {decision.get('top_operations')} · targets: {decision.get('top_targets')}")
                if "offered" in record:
                    print(f"         offered: {record['offered']} · focused: {record.get('focused')}")
                    print(f"         seen ({len(record.get('elements', []))}): {record.get('elements', [])[:12]}")
                if record.get("observe"):
                    print(f"         observe: {record['observe']}")
        return 0
    finally:
        store.close()


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
