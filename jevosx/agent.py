"""The agent loop: observe → recall → decide → guard → act → settle → learn.

One Jev request per decision. Every executed target was observed on this Mac, re-validated immediately before
input, and logged to local memory. Episode outcomes feed back into retrieval, so repeated tasks get better hints.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Generator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import (
    JevAuthError,
    JevError,
    JevOSXError,
    LowConfidenceError,
    StaleElementError,
    TextUnavailableError,
)
from .executor.base import DryRunExecutor, Executor
from .executor.safety import SafetyPolicy
from .memory.retriever import Hint, MemoryRetriever, state_summary
from .memory.store import MemoryStore
from .observer.base import Observer
from .router.policy import Decision, JevRouter
from .router.text import LLMTextWriter, TextSource, slots_from_goal
from .types import (
    BLOCKED,
    DONE,
    OPEN_APP,
    TYPE_TEXT,
    WAIT,
    Action,
    ActionResult,
    Observation,
    clean_text,
)

log = logging.getLogger("jevosx")

ConfirmFn = Callable[[Action, str], bool]
Verifier = Callable[[Observation], bool]
# Returns "retry" (re-observe, nothing executed), "execute" (explicitly approve this decision) or "stop".
LowConfidenceHandler = Callable[[LowConfidenceError, Observation], str]
FALLBACK_RESOLUTIONS = frozenset({"retry", "execute", "stop"})


@dataclass(frozen=True)
class ConfidenceGate:
    """Raises LowConfidenceError when the weakest answer that would drive execution is below `floor`."""

    floor: float

    def check(self, decision: Decision) -> None:
        confidence = decision.gate_confidence
        if confidence < self.floor:
            raise LowConfidenceError(decision, confidence, self.floor)


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
        text_writer: LLMTextWriter | None = None,
        confirm: ConfirmFn | None = None,
        on_low_confidence: LowConfidenceHandler | None = None,
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
        self.confirm = confirm
        self.gate = ConfidenceGate(self.settings.agent.min_confidence)
        self.on_low_confidence = on_low_confidence
        self.sleep = sleep
        self.clock = clock
        self.last_result: RunResult | None = None
        self.last_observation: Observation | None = None

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        dry_run: bool = False,
        use_memory: bool = True,
        confirm: ConfirmFn | None = None,
        on_low_confidence: LowConfidenceHandler | None = None,
    ) -> Agent:
        """Wire the real macOS observer/executor, the Jev client and local memory from settings."""
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
        writer = None
        text_key = settings.text_model.api_key()
        if settings.text_model.model and text_key:
            writer = LLMTextWriter(
                base_url=settings.text_model.base_url,
                api_key=text_key,
                model=settings.text_model.model,
                timeout_s=settings.text_model.timeout_s,
            )
        return cls(
            observer=observer,
            executor=executor,
            router=router,
            settings=settings,
            memory=memory,
            text_writer=writer,
            confirm=confirm,
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
        text_source = TextSource({**slots_from_goal(goal), **(text_slots or {})}, self.text_writer)
        history: list[dict[str, Any]] = []
        events: list[StepEvent] = []
        started = self.clock()
        episode_id = self.memory.begin_episode(goal, app=app, model=self.router.client.model) if self.memory else None
        steps = 0
        status = "aborted"
        message = ""
        pending: _Pending | None = None
        low_confidence = stale = done_rejections = no_change = text_failures = 0

        def emit(event: StepEvent) -> StepEvent:
            events.append(event)
            self._log(event)
            return event

        try:
            if app:
                result = self._open_requested_app(app)
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
                    obs = self.observer.observe()
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
                    pending.entry["result"] = "ui changed" if changed else "no visible change"
                    if self.memory is not None and pending.step_id is not None:
                        self.memory.set_outcome(pending.step_id, "changed" if changed else "unchanged")
                    no_change = 0 if changed or pending.operation == WAIT else no_change + 1
                    pending = None
                    if no_change >= cfg.stuck_after:
                        status, message = "blocked", f"{no_change} consecutive actions produced no visible change"
                        break

                space = self.router.space(obs, text_source, goal)
                hints: list[Hint] = []
                if self.retriever is not None:
                    hints = self.retriever.hints(goal, obs, space, exclude_episode=episode_id)
                    for hint in hints:
                        if hint.target_id is not None:
                            space.annotate(hint.operation, hint.target_id, hint.note())
                t2 = self.clock()
                decision = self.router.decide(
                    goal,
                    obs,
                    space,
                    text_source=text_source,
                    history=history[-cfg.history_size :],
                    hints=[h.to_state() for h in hints],
                )
                t3 = self.clock()
                event = StepEvent(
                    step=steps + 1,
                    status="decided",
                    action=decision.describe(),
                    decision=decision.summary(),
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

                # Confidence gate: nothing (not even DONE) is acted on below the floor. WAIT is harmless.
                if op != WAIT:
                    try:
                        self.gate.check(decision)
                        low_confidence = 0
                    except LowConfidenceError as exc:
                        resolution = self._handle_low_confidence(exc, goal, obs)
                        event.message = f"{exc} → {resolution}"
                        if resolution == "stop":
                            event.status = "low_confidence"
                            yield emit(event)
                            status, message = "low_confidence", str(exc)
                            break
                        if resolution == "retry":
                            low_confidence += 1
                            event.status = "low_confidence"
                            yield emit(event)
                            if low_confidence > cfg.max_low_confidence_retries:
                                status = "low_confidence"
                                message = f"Jev stayed below the {self.gate.floor:.2f} confidence floor"
                                break
                            self.sleep(self.settings.executor.wait_s)
                            continue
                        low_confidence = 0  # "execute": explicitly approved by the fallback handler

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

                verdict = self.safety.check(action, obs.app, window_title=obs.window.title if obs.window else None)
                if verdict.verdict == "deny" or (
                    verdict.verdict == "confirm" and not (self.confirm and self.confirm(action, verdict.reason))
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
                result = self.executor.execute(action, obs)
                t5 = self.clock()
                if result.ok and action.operation != OPEN_APP:
                    self._settle()
                t6 = self.clock()
                steps += 1
                entry = {
                    "step": steps,
                    "action": _history_action(action),
                    "result": "ok" if result.ok else f"failed: {result.detail}",
                }
                history.append(entry)
                step_id = self._record(
                    episode_id,
                    steps,
                    obs,
                    decision,
                    outcome="pending" if result.ok else "failed",
                    text=action.text if self.settings.memory.store_typed_text and not action.text_is_secret else None,
                )
                if result.ok:
                    pending = _Pending(step_id, entry, obs.fingerprint, op)
                event.status = "acted" if result.ok else "failed"
                event.result = result
                event.message = result.detail
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
            if pending is not None and self.memory is not None and pending.step_id is not None:
                self.memory.set_outcome(pending.step_id, "unknown")
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

    def _handle_low_confidence(self, exc: LowConfidenceError, goal: str, obs: Observation) -> str:
        """Fallback for a withheld decision: custom handler, else the configured policy. Always logged."""
        policy = self.settings.agent.low_confidence_policy
        decision: Decision = exc.decision  # type: ignore[assignment]
        if self.on_low_confidence is not None:
            resolution = self.on_low_confidence(exc, obs)
        elif policy == "ask":
            preview = self._preview_action(decision)
            approved = self.confirm is not None and self.confirm(preview, str(exc))
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
                "decision": decision.summary(),
                "confidence": round(exc.confidence, 4),
                "floor": exc.floor,
                "policy": policy,
                "resolution": resolution,
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
            text, secret, _source = text_source.resolve(
                decision.text_option, goal=goal, element=action.element, obs=obs, history=history
            )
            action.text, action.text_is_secret = text, secret
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
            window=obs.window.title if obs.window else None,
            state_summary=state_summary(obs),
            operation=decision.operation,
            memory_key=target.memory_key if target else None,
            target_text=target.describe() if target else None,
            probability=decision.probability,
            confidence=decision.confidence,
            outcome=outcome,
            text=text,
        )

    def _settle(self) -> None:
        """Wait until a cheap UI signature stops changing (bounded), so the next observation is not mid-animation."""
        cfg = self.settings.executor
        deadline = self.clock() + cfg.settle_timeout_s
        previous = self.observer.quick_signature()
        while self.clock() < deadline:
            self.sleep(cfg.settle_poll_s)
            current = self.observer.quick_signature()
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


def _history_action(action: Action) -> str:
    text = action.describe()
    if action.operation == TYPE_TEXT and action.text is not None:
        text += " ← " + ("(secret)" if action.text_is_secret else repr(clean_text(action.text, 60)))
    return text


def _ms(start: float, end: float) -> float:
    return round((end - start) * 1000, 1)
