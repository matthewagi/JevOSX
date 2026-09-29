"""The agent loop: observe → recall → decide → guard → act → settle → learn.

One Jev request per decision. Every executed target was observed on this Mac, re-validated immediately before
input, and logged to local memory. Episode outcomes feed back into retrieval, so repeated tasks get better hints.

Working behind you (agent.background): once an action of the agent's own has put a window in front (a new browser
window, an app it opened), that window becomes its work window. It keeps reading that window wherever it is, so you
can go back to the console or Terminal. Clicks and field writes reach it through Accessibility in the background;
key presses bring it forward briefly, and then the window you were using is brought back.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Generator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

from .config import Settings
from .errors import (
    JevAuthError,
    JevError,
    JevOSXError,
    LowConfidenceError,
    StaleElementError,
    TextUnavailableError,
)
from .executor.base import DryRunExecutor, Executor, WindowFocuser
from .executor.safety import SafetyPolicy, SafetyVerdict
from .images import (
    COUNT_NAMES,
    FOLDER_NAMES,
    ImageSaveError,
    ImageSaver,
    ImageTask,
    display_path,
    file_stem,
    image_plan,
    image_task,
    is_thumbnail,
    original_of,
    page_key,
    search_address,
    wikimedia_original,
)
from .logins import LoginStore, credential_slots
from .memory.retriever import Hint, MemoryRetriever, state_summary
from .memory.store import MemoryStore
from .observer.base import BackgroundObserver, Observer, WindowRef
from .planner import GoalReading, Planner, drop_detours, merge_slots
from .risk import CAREFUL, StepRisk, assess, is_sign_in
from .router.client import ChoiceAnswer
from .router.policy import Decision, JevRouter, redact
from .router.space import ActionSpace, Target
from .router.text import ADDRESS, TextSource, slot_kind, slots_from_goal, template_slots
from .types import (
    ASK_USER,
    BLOCKED,
    BROWSER_BUNDLES,
    CLICK,
    DONE,
    FOCUS_WINDOW,
    MENU,
    NEW_DOCUMENT,
    OPEN_APP,
    OPEN_DOCUMENT,
    PRESS_KEY,
    SAVE_IMAGE,
    SCROLL_DOWN,
    TYPE_TEXT,
    WAIT,
    Action,
    ActionResult,
    AppInfo,
    Observation,
    UIElement,
    clean_text,
    is_address_bar,
    is_console_window,
    is_document_body,
)
from .writer import TextWriter, create_writer, wants_generation

log = logging.getLogger("jevosx")
T = TypeVar("T")
MAX_IMAGE_FAILURES = 3
MAX_PICTURE_SCROLLS = 4  # scrolls in a row for more pictures before Jev decides again
ORIGINAL_READS = 8  # reads of the page after pressing a thumbnail, waiting for the full picture beside it
ORIGINAL_WAIT_S = 0.3
RESULTS_READS = 10  # reads while the picture results load, before Jev decides on whatever is there
RESULTS_WAIT_S = 0.3
PAGE_READS = 10  # reads after Return in a browser's address bar, waiting for the new page before Jev judges it
PAGE_WAIT_S = 0.4
LAUNCH_READS = 10  # reads after an app was slow to come forward, before Jev decides on whatever is in front
LAUNCH_WAIT_S = 1.0
BLANK_PAGE = re.compile(r"^(?:about:(?:blank|newtab|home)|chrome://new-?tab|edge://newtab|favorites://)", re.IGNORECASE)
BLANK_TITLE = re.compile(r"^(?:about:blank|new tab|start page)\b", re.IGNORECASE)
# Questions an image-saving goal never needs answered first: the folder and the number of pictures have defaults.
IMAGE_DEFAULTS = re.compile(
    r"\b(?:folder|directory|where|location|destination|save|path|how many|count|number|quantity)\b", re.IGNORECASE
)

ConfirmFn = Callable[[Action, str], bool]
# ASK_USER: tell the human what only they can do (e.g. "type a verification code (Safari)"). True = done on screen,
# continue; False = stop the run; a string = their typed answer, which becomes text the agent can type.
HandoffFn = Callable[[str, Observation], "bool | str"]
# Before starting: the questions the goal reader found (name, question) → the person's answers by name, or None to stop.
ClarifyFn = Callable[[list[tuple[str, str]]], "dict[str, str] | None"]
Verifier = Callable[[Observation], bool]
# Returns "retry" (re-observe, nothing executed), "execute" (explicitly approve this decision) or "stop".
LowConfidenceHandler = Callable[[LowConfidenceError, Observation], str]
FALLBACK_RESOLUTIONS = frozenset({"retry", "execute", "stop"})


@dataclass(frozen=True)
class ConfidenceGate:
    """Raises LowConfidenceError when the weakest answer that would drive execution is below the floor: the step's
    own floor from its risk tier (see jevosx/risk.py), or `floor` when none is given."""

    floor: float

    def check(self, decision: Decision, risk: StepRisk | None = None) -> None:
        confidence = decision.gate_confidence
        floor = risk.floor if risk is not None else self.floor
        if confidence < floor:
            raise LowConfidenceError(decision, confidence, floor, risk.tier if risk is not None else "")


@dataclass
class StepEvent:
    step: int
    status: str  # acted | wait | done | blocked | denied | declined | stale | low_confidence | failed
    action: str | None = None
    decision: dict[str, Any] | None = None
    result: ActionResult | None = None
    hints: list[dict[str, Any]] = field(default_factory=list)
    observation: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "status": self.status,
            "action": self.action,
            "decision": self.decision,
            "result": None if self.result is None else asdict(self.result),
            "hints": self.hints,
            "observation": self.observation,
            "timings": self.timings,
            "message": self.message,
        }


@dataclass
class RunResult:
    status: str  # success | done | blocked | low_confidence | failed | max_steps | aborted | error
    goal: str
    steps: int
    episode_id: int | None
    elapsed_ms: float
    events: list[StepEvent] = field(default_factory=list)
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("success", "done")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "goal": self.goal,
            "steps": self.steps,
            "episode_id": self.episode_id,
            "elapsed_ms": self.elapsed_ms,
            "message": self.message,
            "ok": self.ok,
        }


def expect_text_verifier(expected: str) -> Verifier:
    """DONE is only accepted when `expected` is visible (text, window title, element labels or values)."""
    needle = expected.lower()

    def verify(obs: Observation) -> bool:
        haystack = [obs.text, obs.window.title if obs.window else ""]
        haystack += [f"{e.label} {e.value or ''}" for e in obs.elements]
        return any(needle in (h or "").lower() for h in haystack)

    return verify


@dataclass
class _Pending:
    step_id: int | None
    entry: dict[str, Any]
    fingerprint: str
    operation: str
    reported: str | None = None  # the executor reported this failure, but the action may have happened anyway


class Agent:
    def __init__(
        self,
        *,
        observer: Observer,
        executor: Executor,
        router: JevRouter,
        settings: Settings | None = None,
        safety: SafetyPolicy | None = None,
        memory: MemoryStore | None = None,
        text_writer: TextWriter | None = None,
        planner: Planner | None = None,
        logins: LoginStore | None = None,
        confirm: ConfirmFn | None = None,
        handoff: HandoffFn | None = None,
        clarify: ClarifyFn | None = None,
        on_low_confidence: LowConfidenceHandler | None = None,
        image_saver: ImageSaver | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.observer = observer
        self.executor = executor
        self.router = router
        self.settings = settings or Settings()
        self.safety = safety or SafetyPolicy(self.settings.safety)
        self.memory = memory
        self.retriever = MemoryRetriever(memory, self.settings.memory) if memory is not None else None
        self.text_writer = text_writer
        self.planner = planner
        self.logins = logins
        self.confirm = confirm
        self.handoff = handoff
        self.clarify = clarify
        self.gate = ConfidenceGate(self.settings.agent.min_confidence)
        self.on_low_confidence = on_low_confidence
        self.image_saver = image_saver or ImageSaver()
        self.images: ImageTask | None = None  # this run's pictures to save (folder, how many, saved so far)
        self.sleep = sleep
        self.clock = clock
        self.last_result: RunResult | None = None
        self.last_observation: Observation | None = None
        self._sensitive: list[str] = []  # saved-login usernames on screen: masked in events, memory and logs
        self.last_plan: list[str] = []
        self.work: WindowRef | None = None  # the agent's own work window (agent.background)
        self._behind = False  # this run reads its work window behind others (see works_behind)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        dry_run: bool = False,
        use_memory: bool = True,
        confirm: ConfirmFn | None = None,
        handoff: HandoffFn | None = None,
        clarify: ClarifyFn | None = None,
        on_low_confidence: LowConfidenceHandler | None = None,
        notify: Callable[[str], None] | None = None,
    ) -> Agent:
        """Wire the real macOS observer/executor, the Jev client, the text writer and local memory from settings."""
        from .executor import create_executor, key_vocabulary
        from .memory.embedding import HashingEmbedder
        from .observer import create_observer
        from .observer.ax import require_trusted
        from .router.client import JevClient

        observer = create_observer(settings.observer)
        require_trusted()
        executor: Executor = DryRunExecutor() if dry_run else create_executor(settings.executor, observer.frontmost_pid)
        router = JevRouter.from_settings(
            JevClient.from_settings(settings.jev),
            settings.jev,
            key_vocabulary(settings.keys.custom, settings.keys.disabled),
        )
        memory = None
        if use_memory and settings.memory.enabled:
            memory = MemoryStore(settings.memory_path, HashingEmbedder(settings.memory.dim))
        writer, status = create_writer(settings, notify=notify)
        log.info("text writer: %s", status.describe())
        logins = LoginStore(settings.logins.index_path) if settings.logins.enabled else None
        planner = Planner(writer) if writer is not None and settings.agent.plan == "auto" else None
        return cls(
            observer=observer,
            executor=executor,
            router=router,
            settings=settings,
            memory=memory,
            text_writer=writer,
            planner=planner,
            logins=logins,
            confirm=confirm,
            handoff=handoff,
            clarify=clarify,
            on_low_confidence=on_low_confidence,
        )

    # ------------------------------------------------------------------------------------------------------------
    def run(self, goal: str, **kwargs: Any) -> RunResult:
        for _ in self.iter_run(goal, **kwargs):
            pass
        assert self.last_result is not None
        return self.last_result

    def iter_run(
        self,
        goal: str,
        *,
        app: str | None = None,
        text_slots: Mapping[str, str] | None = None,
        max_steps: int | None = None,
        verifier: Verifier | None = None,
    ) -> Generator[StepEvent, None, None]:
        goal = goal.strip()
        if not goal:
            raise ValueError("goal must not be empty")
        cfg = self.settings.agent
        max_steps = max_steps or cfg.max_steps
        # The writer's model reads the goal once (steps + exact values to type); patterns fill in when it cannot.
        reading = self.planner.read(goal) if self.planner is not None else GoalReading()
        reading.steps = drop_detours(reading.steps, goal, self._is_app)
        plan = reading.steps
        self.last_plan = plan
        slots = merge_slots(reading.values, slots_from_goal(goal))
        images = self.images = image_task(goal, reading.values)
        if images is not None:
            # Folder and count have defaults: never a reason to stop and ask before starting.
            reading.questions = [(n, q) for n, q in reading.questions if not IMAGE_DEFAULTS.search(f"{n} {q}")]
            for name in (*FOLDER_NAMES, *COUNT_NAMES):
                slots.pop(name, None)  # settings of the task, not text to type
            if images.topic and not any(slot_kind(n, v) == ADDRESS for n, v in slots.items()):
                slots["picture_search"] = search_address(images.topic)  # straight to picture results
            plan = self.last_plan = reading.steps = image_plan(images)
        if self.text_writer is None:
            slots.update(template_slots(goal, slots))  # no model to compose with: plain filler text where asked
        text_source = TextSource(
            {**slots, **(text_slots or {})},
            self.text_writer,
            generate=self.settings.writer.offer == "always" or wants_generation(goal),
        )
        history: list[dict[str, Any]] = []
        events: list[StepEvent] = []
        started = self.clock()
        meta = {"plan": reading.steps, "values": {k: clean_text(v, 60) for k, v in reading.values.items()}}
        episode_id = (
            self.memory.begin_episode(goal, app=app, model=self.router.client.model, meta=self._mask(meta))
            if self.memory
            else None
        )
        steps = 0
        status = "aborted"
        message = ""
        pending: _Pending | None = None
        low_confidence = stale = done_rejections = no_change = text_failures = handoffs = image_failures = 0
        password_typed = False  # since the last sign-in attempt
        sign_ins = 0  # sign-in attempts made after typing a password
        risk: StepRisk | None = None
        navigation: tuple[str, str] | None = None  # the page Return was pressed on in the address bar
        by_link = False  # that navigation was a clicked link: only a new title shows the new page
        page_reads = 0
        launching: tuple[AppInfo, _Pending] | None = None  # an app asked to come forward, not in front yet
        launch_reads = 0
        asks_writing, typed_any = wants_generation(goal), False  # a goal to write text is not done before typing
        fresh_document = bool(OPEN_DOCUMENT.search(goal))  # a new note or document to write in (or the person's own)

        def emit(event: StepEvent) -> StepEvent:
            events.append(event)
            self._log(event)
            return event

        self.work = None
        behind = self._behind = self.works_behind
        try:
            if reading.steps or reading.values:
                yield emit(StepEvent(step=0, status="plan", action="PLAN", message=reading.summary()))
            if images is not None:
                wanted = f"save {images.count} pictures into {display_path(images.folder)}"
                yield emit(StepEvent(step=0, status="plan", action="PLAN", message=wanted))
            if reading.questions and self.clarify is not None and cfg.ask_first:
                asked = " · ".join(question for _, question in reading.questions)
                yield emit(StepEvent(step=0, status="ask", action="ASK", message=asked))
                answers = self.clarify(reading.questions)
                if answers is None:
                    status, message = "aborted", "stopped before starting"
                    return
                given = [text_source.add(name, value.strip()) for name, value in answers.items() if value.strip()]
                history.append({"step": 0, "action": "ASK (before starting)", "result": f"answered: {given or 'none'}"})
            if app:
                before = self._front() if behind else None
                result = self._open_requested_app(app)
                if behind:
                    requested = self.observer.find_app(app)
                    self._after_action(before, acted_pid=requested.pid if requested else None, known=None)
                history.append({"step": 0, "action": f"OPEN_APP {app} (requested)", "result": result.detail or "ok"})
                if not result.ok:
                    status, message = "error", f"could not open {app}: {result.detail}"
                    return
            while True:
                if steps >= max_steps:
                    status, message = "max_steps", f"stopped after {max_steps} steps"
                    break
                t0 = self.clock()
                try:
                    obs = self._observe(behind)
                    if images is None:
                        without_pictures(obs)
                    self.last_observation = obs
                except StaleElementError as exc:
                    stale += 1
                    if stale > cfg.max_stale_retries:
                        status, message = "blocked", f"cannot observe the desktop: {exc}"
                        break
                    self.sleep(self.settings.executor.settle_poll_s)
                    continue
                t1 = self.clock()

                # Resolve the previous action's effect now that we have a fresh observation.
                if pending is not None:
                    changed = obs.fingerprint != pending.fingerprint
                    if pending.reported is not None:  # the screen says whether it happened after all
                        reported = pending.reported
                        pending.entry["result"] = (
                            f"ui changed (reported: {reported})" if changed else f"failed: {reported}"
                        )
                        seen = "changed" if changed else "failed"
                    else:
                        pending.entry["result"] = "ui changed" if changed else "no visible change"
                        seen = "changed" if changed else "unchanged"
                    if self.memory is not None and pending.step_id is not None:
                        self.memory.set_outcome(pending.step_id, seen)
                    if changed or pending.operation in (WAIT, ASK_USER):
                        no_change = 0
                    elif pending.reported is None:
                        no_change += 1
                    pending = None
                    if no_change >= cfg.stuck_after:
                        status, message = "blocked", f"{no_change} consecutive actions produced no visible change"
                        break

                if launching is not None:
                    slow_app, waiting = launching
                    if self._came_forward(slow_app, obs):
                        launching = None
                        waiting.entry["result"] = "ok (slow to come to the front)"
                        if self.memory is not None and waiting.step_id is not None:
                            self.memory.set_outcome(waiting.step_id, "changed")
                        if not same_app(obs.app, slow_app):
                            continue  # working behind: read the app that came forward
                    elif launch_reads > 0:
                        launch_reads -= 1  # asking Jev now would only open it again (seen live: twice, 8 s each)
                        self.sleep(LAUNCH_WAIT_S)
                        continue
                    else:
                        launching = None
                        waiting.entry["result"] = f"failed: {waiting.reported}"
                        if self.memory is not None and waiting.step_id is not None:
                            self.memory.set_outcome(waiting.step_id, "failed")

                if navigation is not None:
                    if page_arrived(obs, navigation, new_title=by_link):
                        navigation = None
                    elif page_reads > 0:
                        page_reads -= 1  # Jev would judge the page the address was typed on
                        self.sleep(PAGE_WAIT_S)
                        continue

                if images is not None and images.loading > 0 and not pictures_ready(obs, images):
                    images.loading -= 1  # Jev would only see a blank or half-built page
                    self.sleep(RESULTS_WAIT_S)
                    continue

                if self.logins is not None:
                    text_source.set_credentials(credential_slots(self.logins, obs, goal))
                    self._sensitive = text_source.sensitive_values()
                can_hand_off = self.handoff is not None and handoffs < cfg.max_handoffs
                saved = images.saved_urls if images is not None else None
                space = self.router.space(obs, text_source, goal, handoff=can_hand_off, images=saved)
                if asks_writing and not typed_any:
                    # Seen live: "write a shopping list" in Notes ended DONE right after OPEN_APP, three times after
                    # being told nothing was written, because the list an earlier run had written was open.
                    space.operations.pop(DONE, None)
                hints: list[Hint] = []
                if self.retriever is not None:
                    hints = self.retriever.hints(goal, obs, space, exclude_episode=episode_id)
                    for hint in hints:
                        if hint.target_id is not None:
                            space.annotate(hint.operation, hint.target_id, hint.note())
                t2 = self.clock()
                decision = next_picture(space, obs, images) or self.router.decide(
                    goal,
                    obs,
                    space,
                    text_source=text_source,
                    history=history[-cfg.history_size :],
                    hints=[h.to_state() for h in hints],
                    plan=plan,
                )
                t3 = self.clock()
                if images is not None and decision.operation == TYPE_TEXT and decision.text_option == "picture_search":
                    images.loading, images.loading_results = RESULTS_READS, True
                event = StepEvent(
                    step=steps + 1,
                    status="decided",
                    action=self._mask(decision.describe()),
                    decision=self._mask(decision.summary()),
                    hints=[h.to_state() for h in hints],
                    observation={
                        "app": obs.app.name,
                        "window": obs.window.title if obs.window else None,
                        "elements": len(obs.elements),
                        "menu_items": len(obs.menu_items),
                        "truncated": bool(obs.stats.get("truncated")),
                    },
                    timings={
                        "observe_ms": _ms(t0, t1),
                        "recall_ms": _ms(t1, t2),
                        "decide_ms": _ms(t2, t3),
                        "jev_ms": decision.latency_ms,
                    },
                )
                op = decision.operation

                if op == BLOCKED:
                    self._record(episode_id, steps + 1, obs, decision, outcome="final")
                    status, message = "blocked", "Jev reported that no offered operation can make progress"
                    event.status = "blocked"
                    yield emit(event)
                    break

                # Confidence gate: nothing (not even DONE) is acted on below its floor. WAIT is harmless.
                person_approved = False
                risk = None
                if op != WAIT and self._console_navigation(decision, obs):
                    event.message = (
                        f"moving away from the console (confidence {decision.gate_confidence:.2f}; not gated)"
                    )
                elif op != WAIT:
                    risk = assess(
                        self._preview_action(decision),
                        obs,
                        settings=cfg,
                        safety=self.safety,
                        hints=hints,
                        target_id=decision.target.id if decision.target else None,
                        sign_ins=sign_ins,
                        goal=goal,
                    )
                    if event.decision is not None:
                        event.decision.update(risk=risk.tier, floor=risk.floor, risk_reason=risk.reason)
                    try:
                        self.gate.check(decision, risk)
                        low_confidence = 0
                    except LowConfidenceError as exc:
                        resolution = self._handle_low_confidence(exc, goal, obs, risk, attempt=low_confidence)
                        by_person = self.on_low_confidence is None and cfg.low_confidence_policy == "ask"
                        person_approved = resolution == "execute" and by_person
                        event.message = f"{exc} → {resolution}"
                        if resolution == "stop":
                            event.status = "low_confidence"
                            yield emit(event)
                            status, message = "low_confidence", str(exc) + _typing_tip(text_source)
                            break
                        if resolution == "retry":
                            low_confidence += 1
                            event.status = "low_confidence"
                            yield emit(event)
                            if low_confidence > cfg.max_low_confidence_retries:
                                status = "low_confidence"
                                message = f"Jev stayed below the {exc.floor:.2f} confidence floor"
                                message += _typing_tip(text_source)
                                break
                            self.sleep(self.settings.executor.wait_s)
                            continue
                        low_confidence = 0  # "execute": explicitly approved by the fallback handler

                if op == ASK_USER and self.handoff is not None:
                    steps += 1
                    handoffs += 1
                    need = decision.target.criterion.get("need", "help") if decision.target else "help"
                    where = obs.app.name + (f", window “{obs.window.title}”" if obs.window else "")
                    event.status, event.message = "handoff", f"waiting for you to {need}"
                    entry = {"step": steps, "action": decision.describe(), "result": "waiting for the user"}
                    history.append(entry)
                    step_id = self._record(episode_id, steps, obs, decision, outcome="pending")
                    yield emit(event)
                    if behind and self.work is not None:
                        cast(WindowFocuser, self.executor).bring_forward(self.work)  # the person acts in it
                    outcome = self.handoff(f"{need} ({where})", obs)
                    if outcome is False:
                        entry["result"] = "the user stopped the run"
                        status, message = "aborted", "stopped by you during a hand-off"
                        break
                    if isinstance(outcome, str) and outcome.strip():
                        slot = text_source.add("answer", outcome.strip())
                        entry["result"] = f"the user answered; their answer is text slot {slot}"
                    else:
                        entry["result"] = "the user finished; check the screen"
                    pending = _Pending(step_id, entry, obs.fingerprint, op)
                    continue

                if op == DONE and images is not None and not images.complete:
                    # Seen live: Jev said DONE (confidence 0.36) with 2 of 3 pictures saved.
                    done_rejections += 1
                    steps += 1
                    history.append({"step": steps, "action": "DONE", "result": f"rejected: {images.progress()}"})
                    event.status, event.message = "failed", f"DONE rejected: {images.progress()}"
                    yield emit(event)
                    if done_rejections > cfg.max_done_rejections:
                        status, message = "failed", f"stopped early: {images.progress()}"
                        break
                    continue
                if op == DONE and navigation is not None and blank_page(obs):
                    # Seen live: DONE (confidence 0.66) on "about:blank" right after searching, before the results.
                    done_rejections += 1
                    steps += 1
                    history.append({"step": steps, "action": "DONE", "result": "rejected: the page has not loaded"})
                    event.status, event.message = "failed", "DONE rejected: the page has not loaded yet"
                    yield emit(event)
                    if done_rejections > cfg.max_done_rejections:
                        status, message = "failed", "the page did not load"
                        break
                    page_reads = PAGE_READS
                    continue
                if op == DONE:
                    if verifier is not None and not verifier(obs):
                        done_rejections += 1
                        steps += 1
                        history.append({"step": steps, "action": "DONE", "result": "rejected: goal not verified yet"})
                        event.status, event.message = "failed", "DONE rejected by the verifier"
                        yield emit(event)
                        if done_rejections > cfg.max_done_rejections:
                            status, message = "failed", "the verifier rejected DONE repeatedly"
                            break
                        continue
                    self._record(episode_id, steps + 1, obs, decision, outcome="final")
                    status = "success" if verifier is not None else "done"
                    event.status = "done"
                    yield emit(event)
                    break
                if op == WAIT:
                    steps += 1
                    self.sleep(self.settings.executor.wait_s)
                    entry = {"step": steps, "action": "WAIT", "result": "waited"}
                    history.append(entry)
                    step_id = self._record(episode_id, steps, obs, decision, outcome="pending")
                    pending = _Pending(step_id, entry, obs.fingerprint, op)
                    event.status = "wait"
                    yield emit(event)
                    continue

                if op == SAVE_IMAGE and images is not None and decision.target is not None:
                    steps += 1
                    ok, detail = self._save_picture(decision, images, obs)
                    image_failures = 0 if ok else image_failures + 1
                    entry = {"step": steps, "action": decision.describe(), "result": detail}
                    history.append(entry)
                    self._record(episode_id, steps, obs, decision, outcome="changed" if ok else "failed")
                    event.status, event.message = ("acted" if ok else "failed"), detail
                    yield emit(event)
                    if images.complete:
                        status, message = "done", images.progress()
                        break
                    if image_failures >= MAX_IMAGE_FAILURES:
                        status, message = "blocked", f"{image_failures} pictures in a row could not be saved ({detail})"
                        break
                    continue

                if (
                    asks_writing
                    and not (typed_any or fresh_document)
                    and op == TYPE_TEXT
                    and decision.target is not None
                    and is_document_body(decision.target.element, obs.app)
                    and getattr(decision.target.element, "value", None) != ""  # empty: already new; unknown: not
                ):
                    # Seen live: "open Notes and write a shopping list" typed over the note that was open. Something
                    # new goes into a new note or document; the open one is kept (unless the goal says "this note").
                    fresh_document = True
                    steps += 1
                    made, detail = self._new_document(obs)
                    history.append({"step": steps, "action": made, "result": detail})
                    event.action, event.status, event.message = made, "acted", detail
                    yield emit(event)
                    continue

                try:
                    action = self._to_action(decision, text_source, goal, obs, history)
                    text_failures = 0
                except TextUnavailableError as exc:
                    text_failures += 1
                    steps += 1
                    history.append({"step": steps, "action": decision.describe(), "result": f"failed: {exc}"})
                    event.status, event.message = "failed", str(exc)
                    yield emit(event)
                    if text_failures >= 2:
                        status, message = "blocked", str(exc)
                        break
                    continue

                verdict = self.safety.check(
                    action, obs.app, window_title=obs.window.title if obs.window else None, page_url=obs.page_url
                )
                if risk is not None and risk.confirm and verdict.verdict == "allow":
                    verdict = SafetyVerdict("confirm", risk.reason)
                # A step you just approved is not asked about again, unless it types text you have not seen yet.
                asked_already = person_approved and action.operation != TYPE_TEXT
                if verdict.verdict == "deny" or (
                    verdict.verdict == "confirm"
                    and not asked_already
                    and not (self.confirm and self.confirm(action, verdict.reason))
                ):
                    steps += 1
                    outcome = "refused by safety policy" if verdict.verdict == "deny" else "declined by the user"
                    history.append(
                        {"step": steps, "action": action.describe(), "result": f"{outcome}: {verdict.reason}"}
                    )
                    self._record(episode_id, steps, obs, decision, outcome="denied")
                    event.status = "denied" if verdict.verdict == "deny" else "declined"
                    event.message = verdict.reason
                    yield emit(event)
                    continue

                try:
                    self.executor.validate(action, obs)
                except StaleElementError as exc:
                    stale += 1
                    event.status, event.message = "stale", str(exc)
                    yield emit(event)
                    if stale > cfg.max_stale_retries:
                        status, message = "blocked", f"the UI kept changing before execution ({exc})"
                        break
                    continue
                stale = 0

                t4 = self.clock()
                before = self._front() if behind else None
                attempt = password_typed and is_sign_in(action, obs)
                result = self.executor.execute(action, obs)
                if result.ok and action.operation == TYPE_TEXT and action.text_is_secret:
                    password_typed = True
                if result.ok and attempt:
                    sign_ins, password_typed = sign_ins + 1, False
                t5 = self.clock()
                if (result.ok or result.unconfirmed) and action.operation != OPEN_APP:
                    self._settle()
                if behind:
                    if op == FOCUS_WINDOW and result.ok and action.window is not None and obs.app.pid is not None:
                        self.work = WindowRef(obs.app.pid, action.window.node, action.window.title)
                    acted_pid = action.app.pid if op == OPEN_APP and action.app is not None else obs.app.pid
                    self._after_action(before, acted_pid=acted_pid, known={a.pid for a in obs.running_apps})
                t6 = self.clock()
                steps += 1
                entry = {
                    "step": steps,
                    "action": _history_action(action),
                    "result": "ok" if result.ok else f"failed: {result.detail}",
                }
                if result.unconfirmed and op == OPEN_APP:
                    entry["result"] = f"not in front yet ({result.detail}); waiting for it"
                history.append(entry)
                step_id = self._record(
                    episode_id,
                    steps,
                    obs,
                    decision,
                    outcome="pending" if result.ok or result.unconfirmed else "failed",
                    text=action.text
                    if self.settings.memory.store_typed_text and not (action.text_is_secret or action.text_label)
                    else None,
                )
                if result.ok:
                    pending = _Pending(step_id, entry, obs.fingerprint, op)
                elif result.unconfirmed and op == OPEN_APP and action.app is not None:
                    waiting = _Pending(step_id, entry, obs.fingerprint, op, reported=result.detail)
                    launching, launch_reads = (action.app, waiting), LAUNCH_READS
                elif result.unconfirmed:
                    pending = _Pending(step_id, entry, obs.fingerprint, op, reported=result.detail)
                typed_any = typed_any or (result.ok and op == TYPE_TEXT)
                fresh_document = fresh_document or ((result.ok or result.unconfirmed) and makes_document(action))
                by_link = result.ok and follows_link(action, obs)
                if by_link or (
                    result.ok and op == TYPE_TEXT and action.submit and is_address_bar(action.element, obs.app)
                ):
                    navigation, page_reads = page_where(obs), PAGE_READS
                event.status = "acted" if result.ok else "failed"
                event.result = result
                written = ""
                if action.text_source.startswith("model:") and action.text is not None:
                    written = f"{len(action.text)} characters written by {action.text_source[6:]}"
                event.message = " · ".join(m for m in (event.message, written, result.detail) if m)
                event.timings.update(act_ms=_ms(t4, t5), settle_ms=_ms(t5, t6))
                yield emit(event)
        except KeyboardInterrupt:
            status, message = "aborted", "interrupted"
        except JevAuthError as exc:
            status, message = "error", str(exc)
        except JevError as exc:
            status, message = "error", str(exc)
        except JevOSXError as exc:
            status, message = "error", str(exc)
        except Exception as exc:
            status, message = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            for unresolved in (pending, launching[1] if launching is not None else None):
                if unresolved is not None and self.memory is not None and unresolved.step_id is not None:
                    self.memory.set_outcome(unresolved.step_id, "unknown")
            if self.memory is not None and episode_id is not None:
                self.memory.finish_episode(episode_id, status)
            self.last_result = RunResult(
                status=status,
                goal=goal,
                steps=steps,
                episode_id=episode_id,
                elapsed_ms=_ms(started, self.clock()),
                events=events,
                message=message,
            )
            log.info("run finished: %s after %d steps (%s)", status, steps, message or "no message")

    def _mask(self, value: T) -> T:
        return redact(value, self._sensitive) if self._sensitive else value

    def _handle_low_confidence(
        self, exc: LowConfidenceError, goal: str, obs: Observation, risk: StepRisk | None = None, *, attempt: int = 0
    ) -> str:
        """Fallback for a withheld decision: custom handler, else the configured policy. Always logged.
        `attempt` counts the decisions withheld in a row before this one."""
        policy = self.settings.agent.low_confidence_policy
        decision: Decision = exc.decision  # type: ignore[assignment]
        if self.on_low_confidence is not None:
            resolution = self.on_low_confidence(exc, obs)
        elif policy == "ask" and attempt < self.settings.agent.ask_after_retries:
            resolution = "retry"  # look again before asking: the page may still have been loading
        elif policy == "ask":
            preview = self._preview_action(decision)
            why = str(exc) + (f" ({risk.reason})" if risk is not None and risk.tier == CAREFUL else "")
            approved = self.confirm is not None and self.confirm(preview, why)
            resolution = "execute" if approved else "retry"
        else:
            resolution = policy
        if resolution not in FALLBACK_RESOLUTIONS:
            raise ValueError(f"low-confidence handler returned {resolution!r}")
        log.warning("low confidence: %s (%s) → %s", decision.describe(), exc, resolution)
        self._write_fallback_log(
            {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "goal": goal,
                "app": obs.app.name,
                "window": obs.window.title if obs.window else None,
                "decision": self._mask(decision.summary()),
                "confidence": round(exc.confidence, 4),
                "floor": exc.floor,
                "risk": risk.describe() if risk is not None else None,
                "policy": policy,
                "resolution": resolution,
                "offered": list(decision.operation_answer.probabilities),
                "focused": obs.focused_element.describe() if obs.focused_element else None,
                "elements": self._mask([e.describe() for e in obs.elements[:30]]),
                "observe": {k: obs.stats.get(k) for k in ("visited", "walk_ms", "truncated", "notes", "skipped")},
            }
        )
        return resolution

    def _write_fallback_log(self, record: dict[str, Any]) -> None:
        path = self.settings.agent.fallback_log
        if not path:
            return
        try:
            file = Path(path).expanduser()
            file.parent.mkdir(parents=True, exist_ok=True)
            with file.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("cannot write fallback log %s: %s", path, exc)

    def _console_navigation(self, decision: Decision, obs: Observation) -> bool:
        """New window/tab or app/window switch while the web console is in front: harmless, so not gated.
        The action space only offers these there, and the safety policy re-checks them before execution."""
        if self.settings.agent.gate_console_navigation or obs.window is None:
            return False
        return is_console_window(obs.window.title) and decision.operation in CONSOLE_NAVIGATION

    @staticmethod
    def _preview_action(decision: Decision) -> Action:
        target = decision.target
        return Action(
            decision.operation,
            element=target.element if target else None,
            app=target.app if target else None,
            window=target.window if target else None,
            key=target.key if target else None,
        )

    def feedback(self, episode_id: int, success: bool) -> None:
        """Label an episode after the fact (human or external verifier). Successful runs weigh most in retrieval."""
        if self.memory is not None:
            self.memory.label_episode(episode_id, "success" if success else "failed")

    def close(self) -> None:
        self.router.client.close()
        if self.text_writer is not None:
            self.text_writer.close()
        if self.memory is not None:
            self.memory.close()

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------------------------------------------------
    @property
    def works_behind(self) -> bool:
        return (
            self.settings.agent.background
            and isinstance(self.observer, BackgroundObserver)
            and isinstance(self.executor, WindowFocuser)
        )

    def _front(self) -> WindowRef | None:
        try:
            return cast(BackgroundObserver, self.observer).front()
        except StaleElementError:
            return None

    def _observe(self, behind: bool) -> Observation:
        if not behind or self.work is None:
            return self.observer.observe()
        try:
            obs = cast(BackgroundObserver, self.observer).observe_window(self.work)
        except StaleElementError as exc:
            log.info("%s; reading the front window again", exc)
            self.work = None
            return self.observer.observe()
        window = obs.window
        pin = self.work.window is None and window is not None and window.node is not None
        if pin and window is not None and not is_console_window(window.title):  # the app's window that was read
            self.work = WindowRef(self.work.pid, window.node, window.title)
        return obs

    def _after_action(self, before: WindowRef | None, *, acted_pid: int | None, known: set[int | None] | None) -> None:
        """Follow the agent's own window changes, then give the keyboard back to where the person was.

        The window in front after an action becomes the work window only when the action put it there: a window of
        the app the agent acted in (a new browser window), or an app the action launched. A switch the person makes to
        an app that was already running stays theirs, and the console window is never work. When nothing new came
        forward, the app the agent acted in is its work (whichever of its windows it has focused)."""
        after = self._front()
        if after is None or before is None:
            return
        watching = self.work is not None and before.same(self.work)
        moved = not after.same(before)
        ours = after.pid == acted_pid or (known is not None and after.pid not in known)
        if moved and ours and not after.console:
            self.work = after
        elif self.work is None and acted_pid is not None:
            self.work = WindowRef(acted_pid)
        if moved and not watching and self.work is not None and self.work.same(after):  # only a window it put there
            cast(WindowFocuser, self.executor).bring_forward(before)

    def _came_forward(self, app: AppInfo, obs: Observation) -> bool:
        """Whether an app that was slow to open is in front now. Seen live: Notes, its AppleEvents timing out, took
        about 16 s. Working behind, the observation shows the work window, so the front window is checked too and
        becomes the work window when it is that app's."""
        if same_app(obs.app, app):
            return True
        if not self._behind:
            return False
        front = self._front()
        if front is None or front.console:
            return False
        found = self.observer.find_app(app.name)
        if found is None or found.pid != front.pid:
            return False
        self.work = front
        return True

    def _save_picture(self, decision: Decision, task: ImageTask, obs: Observation) -> tuple[bool, str]:
        """SAVE_IMAGE: download the chosen picture into the run's folder. No dialogs, no keys, nothing on screen."""
        element = decision.target.element if decision.target is not None else None
        url = element.url if element is not None else None
        if not url:
            return False, "failed: this picture has no address"
        task.saved_urls.add(url)  # tried: never offered again, whether it saves or not
        if isinstance(self.executor, DryRunExecutor):
            return True, f"dry run: would save it into {display_path(task.folder)}"
        original = (
            self._original(element, obs, task) if element is not None and is_thumbnail(url, obs.page_url) else None
        )
        # Seen live: Google's link for a Wikipedia picture is Wikimedia's 330-pixel thumbnail, so both are looked at.
        full = wikimedia_original(original or url)
        sources = list(dict.fromkeys(s for s in (full, original, url) if s))
        failure: ImageSaveError | None = None
        for source in sources:  # a site that refuses or a file too large: the smaller copy is still good
            try:
                path = self.image_saver.save(source, task.folder, file_stem(task, source), referer=obs.page_url)
                break
            except ImageSaveError as exc:
                failure = exc
        else:
            return False, f"failed: {failure}"
        task.saved_urls.update(sources)
        task.saved.append(path)
        if obs.page_url:
            task.pages.add(page_key(obs.page_url))
        task.scrolls = 0
        task.loading, task.loading_results = RESULTS_READS, False  # pressing the tile makes Chrome rebuild the page
        small = " · Google's small copy" if is_thumbnail(source, obs.page_url) else ""
        return True, f"saved {path.name}{small} ({task.progress()})"

    def _original(self, thumbnail: UIElement, obs: Observation, task: ImageTask) -> str | None:
        """The full picture behind a Google thumbnail: press the tile and read it from the preview that opens.
        Seen live: every picture saved from Google's results was its 500-pixel thumbnail."""
        found = original_of(thumbnail, obs.elements, task.saved_urls)
        try:
            if not found and self.executor.execute(Action(CLICK, element=thumbnail), obs).ok:
                for _ in range(ORIGINAL_READS):
                    self.sleep(ORIGINAL_WAIT_S)
                    obs = self._observe(self._behind)
                    found = original_of(thumbnail, obs.elements, task.saved_urls)
                    if found:
                        break
        except JevOSXError as exc:
            log.info("no full picture for %s: %s", thumbnail.describe(), exc)
        if found:  # the preview shows the same picture: never offer it again
            task.saved_urls.update(
                e.url for e in obs.elements if e.kind == "image" and e.label == thumbnail.label and e.url
            )
        return found

    def _new_document(self, obs: Observation) -> tuple[str, str]:
        """Start a new note or document in the front app: its own "New Note" / "New Document" button when it shows
        one (an AX press needs no keyboard), else cmd+N. Returns what was done and how it went, for the history."""
        button = next(
            (e for e in obs.elements if e.enabled and CLICK in e.ops and NEW_DOCUMENT.fullmatch(e.label or "")), None
        )
        key = self.router.keys.get("CMD_N")
        tries = [Action(CLICK, element=button)] if button is not None else []
        tries += [Action(PRESS_KEY, key=key)] if key is not None else []
        detail = "no way to start a new document here"
        for action in tries:
            before = self._front() if self._behind else None
            try:
                result = self.executor.execute(action, obs)
            except JevOSXError as exc:
                detail = f"failed: {exc}"
                continue
            if result.ok or result.unconfirmed:
                self._settle()
            if self._behind:
                self._after_action(before, acted_pid=obs.app.pid, known={a.pid for a in obs.running_apps})
            if result.ok or result.unconfirmed:
                return _history_action(action), "ok: a new note or document to write in, the open one is kept"
            detail = f"failed: {result.detail}"
        return "NEW_DOCUMENT", detail

    def _is_app(self, name: str) -> bool:
        """An app with exactly this name is running or installed ("Notes", not "Downloads")."""
        try:
            found = self.observer.find_app(name)
        except JevOSXError:
            return False
        return found is not None and found.name.lower() == name.lower()

    def _open_requested_app(self, query: str) -> ActionResult:
        target = self.observer.find_app(query)
        if target is None:
            return ActionResult(False, "open", f"no running or installed app matches {query!r}")
        return self.executor.open_app(target)

    def _to_action(
        self, decision: Decision, text_source: TextSource, goal: str, obs: Observation, history: list[dict[str, Any]]
    ) -> Action:
        action = self._preview_action(decision)
        if decision.operation == TYPE_TEXT:
            assert action.element is not None
            resolved = text_source.resolve(
                decision.text_option, goal=goal, element=action.element, obs=obs, history=history
            )
            action.text, action.text_is_secret, action.text_source = resolved.text, resolved.secret, resolved.source
            action.text_label, action.require_host = resolved.label, resolved.host
            action.secure_only = resolved.secure_only
            action.submit = text_source.submits(decision.text_option, action.element, obs.app)
        return action

    def _record(
        self,
        episode_id: int | None,
        idx: int,
        obs: Observation,
        decision: Decision,
        *,
        outcome: str,
        text: str | None = None,
    ) -> int | None:
        if self.memory is None or episode_id is None:
            return None
        target = decision.target
        return self.memory.record_step(
            episode_id,
            idx=idx,
            app=obs.app.bundle_id or obs.app.name,
            window=self._mask(obs.window.title if obs.window else None),
            state_summary=self._mask(state_summary(obs)),
            operation=decision.operation,
            memory_key=target.memory_key if target else None,
            target_text=self._mask(target.describe() + _slot_note(decision)) if target else None,
            probability=decision.probability,
            confidence=decision.confidence,
            outcome=outcome,
            text=text,
        )

    def _settle(self) -> None:
        """Wait until a cheap UI signature stops changing (bounded), so the next observation is not mid-animation."""
        cfg = self.settings.executor
        deadline = self.clock() + cfg.settle_timeout_s
        work = self.work if self.works_behind else None
        signature = (
            (lambda: cast(BackgroundObserver, self.observer).quick_signature_of(work))
            if work is not None
            else self.observer.quick_signature
        )
        previous = signature()
        while self.clock() < deadline:
            self.sleep(cfg.settle_poll_s)
            current = signature()
            if current == previous:
                return
            previous = current

    @staticmethod
    def _log(event: StepEvent) -> None:
        timings = " · ".join(f"{k[:-3]} {v:.0f}ms" for k, v in event.timings.items())
        confidence = event.decision.get("confidence") if event.decision else None
        log.info(
            "step %d %-14s %s%s%s",
            event.step,
            event.status,
            event.action or "",
            f" · conf {confidence:.2f}" if confidence is not None else "",
            f" · {timings}" if timings else "",
        )


CONSOLE_NAVIGATION = frozenset({PRESS_KEY, MENU, OPEN_APP, FOCUS_WINDOW})


def pictures_ready(obs: Observation, images: ImageTask) -> bool:
    """Whether the page can be judged yet. Seen live: while the results loaded, and again while Chrome rebuilt
    them after a tile was pressed, a read had only the page's header (40 elements, no pictures, nothing to scroll),
    and Jev clicked "Search" and "Search by image"."""
    if images.loading_results and not images.is_source(obs.page_url):
        return False
    if not images.is_source(obs.page_url):
        return True
    pictures = any(SAVE_IMAGE in e.ops and e.url not in images.saved_urls for e in obs.elements)
    return pictures or any(area.in_web_area for area in obs.scroll_areas)


def makes_document(action: Action) -> bool:
    """Jev's own step that starts a new note or document (the button, the menu item or cmd+N)."""
    if action.operation == PRESS_KEY:
        return action.key is not None and action.key.id == "CMD_N"
    return action.operation in (CLICK, MENU) and bool(action.element and NEW_DOCUMENT.fullmatch(action.element.label))


def same_app(current: AppInfo, wanted: AppInfo) -> bool:
    if wanted.bundle_id and current.bundle_id:
        return current.bundle_id == wanted.bundle_id
    return current.name == wanted.name


def page_where(obs: Observation) -> tuple[str, str]:
    return obs.page_url or "", obs.window.title if obs.window else ""


def blank_page(obs: Observation) -> bool:
    """A browser showing an empty page (about:blank, a new tab), where nothing can have been found yet."""
    if obs.app.bundle_id not in BROWSER_BUNDLES:
        return False
    if obs.page_url:  # the title can lag behind the address
        return bool(BLANK_PAGE.match(obs.page_url))
    return bool(BLANK_TITLE.match(obs.window.title if obs.window else ""))


def follows_link(action: Action, obs: Observation) -> bool:
    """A click on a link in a web page, which opens another page. Seen live: after clicking Facebook's "Marketplace",
    the settle read matched at once (nothing had changed yet), so Jev chose the home page's "Search Facebook" box and
    searched all of Facebook instead of Marketplace."""
    element = action.element
    return (
        action.operation == CLICK
        and element is not None
        and element.role == "AXLink"
        and element.in_web_area
        and obs.app.bundle_id in BROWSER_BUNDLES
    )


def page_arrived(obs: Observation, before: tuple[str, str], *, new_title: bool = False) -> bool:
    """After Return in a browser's address bar or a clicked link: whether the new page is there to be judged. Seen
    live: the read right after searching still showed "about:blank", and Jev said DONE on it before the results had
    loaded. After a link (`new_title`), the address alone is not enough: Facebook changed it 0.3 s after the click
    and still showed the old page, under the old title, until 0.9 s. Nor is the title: facebook.com had its title
    before anything of the page could be read, and Jev, seeing an empty page, typed the address again."""
    if obs.app.bundle_id not in BROWSER_BUNDLES:
        return True  # the agent is looking at something else now
    if not any(element.in_web_area for element in obs.elements):
        return False
    now = page_where(obs)
    if new_title:
        return now[1] != before[1]
    return not blank_page(obs) and now != before


def next_picture(space: ActionSpace, obs: Observation, images: ImageTask | None) -> Decision | None:
    """On a page that is a source of pictures (the picture results for the topic, or a page Jev already saved
    from), every picture fits: the next one is saved without asking Jev, and when all visible ones are saved the
    page is scrolled for more. Seen live: on the results Jev followed a link to another site instead of saving, and
    after two saves on Unsplash it clicked a carousel's left and right buttons in turn until it ran out of steps."""
    if images is None or not images.is_source(obs.page_url):
        return None
    targets = space.targets_for(SAVE_IMAGE)
    if targets:
        first = min(targets.values(), key=lambda t: t.element.index if t.element is not None else 0)
        return _certain_decision(SAVE_IMAGE, first)
    areas = [t for t in space.targets_for(SCROLL_DOWN).values() if t.element is not None and t.element.in_web_area]
    if not areas or images.scrolls >= MAX_PICTURE_SCROLLS:
        return None  # nothing to scroll, or scrolling found nothing new: Jev decides
    images.scrolls += 1
    page = max(areas, key=lambda t: t.element.frame.h if t.element and t.element.frame else 0)
    return _certain_decision(SCROLL_DOWN, page)


def _certain_decision(operation: str, target: Target) -> Decision:
    certain = ChoiceAnswer(operation, {operation: 1.0}, 1.0)
    pick = ChoiceAnswer(target.id, {target.id: 1.0}, 1.0)
    return Decision(operation, certain, target=target, target_answer=pick, model="picture source page")


def without_pictures(obs: Observation) -> None:
    """Pictures are only offered to runs that save pictures; elsewhere they would only be noise in Jev's state."""
    for element in obs.elements:
        if SAVE_IMAGE in element.ops:
            element.ops = tuple(op for op in element.ops if op != SAVE_IMAGE)


def _typing_tip(text_source: TextSource) -> str:
    if text_source.available:
        return ""
    return '. Nothing was available to type: put the text in quotes, e.g. search for "red flowers"'


def _history_action(action: Action) -> str:
    text = action.describe()
    if action.operation == TYPE_TEXT and action.text is not None:
        if action.text_label:
            shown = f"({action.text_label})"
        elif action.text_is_secret:
            shown = "(secret)"
        else:
            shown = repr(clean_text(action.text, 60))
        text += " ← " + shown
        if action.submit:
            text += " + RETURN"
    return text


def _slot_note(decision: Decision) -> str:
    """Which prepared text a TYPE_TEXT step chose (its name, never the text)."""
    if decision.operation != TYPE_TEXT or decision.text_option is None:
        return ""
    return f" ← {decision.text_option}" + (" (asked for this field)" if decision.text_follow_up else "")


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 1)
