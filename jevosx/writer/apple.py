"""Apple's on-device language model (Foundation Models, macOS 26+ with Apple Intelligence) as the text writer.

Apple ships no Python binding that installs without the full Xcode, so JevOSX compiles `apple_writer.swift` once
with the Command Line Tools (`xcrun swiftc`) into `~/.jevosx/bin/` and talks to it over JSON on stdin/stdout.
Free, private and offline: prompts and screen text never leave the Mac.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..errors import TextUnavailableError
from .base import WRITER_INSTRUCTIONS, WriterStatus, clean_generated, writer_prompt

HELPER_SOURCE = Path(__file__).with_name("apple_writer.swift")
HELPER_PREFIX = "jevosx-writer-"
Runner = Callable[..., subprocess.CompletedProcess[str]]

# reason → (what it means, how to fix it)
REASONS: dict[str, tuple[str, str]] = {
    "available": ("ready", ""),
    "appleIntelligenceNotEnabled": (
        "Apple Intelligence is turned off",
        "System Settings › Apple Intelligence & Siri → turn on Apple Intelligence, then run: jevosx write --check",
    ),
    "deviceNotEligible": ("this Mac cannot run Apple Intelligence", "it needs a Mac with Apple silicon (M1 or later)"),
    "modelNotReady": (
        "the on-device model is still downloading",
        "keep the Mac online and plugged in for a while, then run: jevosx write --check",
    ),
    "osTooOld": ("macOS 26 or newer is required", "update macOS in System Settings › General › Software Update"),
    "sdkMissing": (
        "the Command Line Tools are older than macOS 26",
        "install the latest Command Line Tools (Software Update, or: xcode-select --install), "
        "then run: jevosx write --rebuild",
    ),
    "noCompiler": ("no Swift compiler (Command Line Tools) found", "run: xcode-select --install"),
    "notMacOS": ("Apple's on-device model only runs on macOS", ""),
    "buildFailed": ("the Swift helper did not compile", "run: jevosx write --rebuild  and report the error it prints"),
    "helperFailed": ("the Swift helper did not answer", "run: jevosx write --rebuild"),
    "unknown": ("Apple's model reports it is unavailable", "run: jevosx write --check"),
}


class WriterUnavailable(Exception):
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{REASONS.get(reason, REASONS['unknown'])[0]}" + (f": {detail}" if detail else ""))

    def status(self) -> WriterStatus:
        meaning, hint = REASONS.get(self.reason, REASONS["unknown"])
        return WriterStatus("apple", False, self.reason, meaning + (f": {self.detail}" if self.detail else ""), hint)


def helper_path(directory: str | Path) -> Path:
    """The compiled helper for the current source (a changed .swift file gets a new name, so it is rebuilt)."""
    digest = hashlib.sha256(HELPER_SOURCE.read_bytes()).hexdigest()[:12]
    return Path(directory).expanduser() / f"{HELPER_PREFIX}{digest}"


def build_helper(
    directory: str | Path,
    *,
    force: bool = False,
    runner: Runner = subprocess.run,
    platform: str = sys.platform,
    timeout_s: float = 300.0,
) -> Path:
    """Compile the Swift helper if needed and return its path. Raises WriterUnavailable."""
    target = helper_path(directory)
    if target.is_file() and not force:
        return target
    if platform != "darwin":
        raise WriterUnavailable("notMacOS")
    # `xcrun` without the Command Line Tools opens an install dialog, so ask xcode-select first (it never prompts).
    try:
        if runner(["xcode-select", "-p"], capture_output=True, text=True, timeout=10, check=False).returncode != 0:
            raise WriterUnavailable("noCompiler")
        sdk = runner(
            ["xcrun", "--sdk", "macosx", "--show-sdk-path"], capture_output=True, text=True, timeout=30, check=False
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WriterUnavailable("noCompiler", str(exc)) from None
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".build-", dir=target.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    command = ["xcrun", "--sdk", "macosx", "swiftc", "-O", "-parse-as-library", "-o", str(tmp), str(HELPER_SOURCE)]
    if sdk and (Path(sdk) / "System/Library/Frameworks/FoundationModels.framework").exists():
        # Weak link: a helper built with a newer SDK still starts (and reports osTooOld) on an older macOS.
        command += ["-Xlinker", "-weak_framework", "-Xlinker", "FoundationModels"]
    try:
        completed = runner(command, capture_output=True, text=True, timeout=timeout_s, check=False)
        if completed.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            errors = (completed.stderr or completed.stdout or "").strip().splitlines()
            raise WriterUnavailable("buildFailed", " / ".join(errors[-6:])[:600])
        tmp.chmod(0o755)
        os.replace(tmp, target)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WriterUnavailable("buildFailed", str(exc)) from None
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()
    for old in target.parent.glob(HELPER_PREFIX + "*"):
        if old != target:
            with contextlib.suppress(OSError):
                old.unlink()
    return target


def _last_json(output: str) -> dict[str, Any] | None:
    for line in reversed((output or "").strip().splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


class AppleWriter:
    """Text writer backed by the compiled helper. One short-lived process per request."""

    model = "apple-on-device"

    def __init__(
        self,
        helper: str | Path,
        *,
        temperature: float = 0.7,
        max_tokens: int = 800,
        timeout_s: float = 60.0,
        runner: Runner = subprocess.run,
    ):
        self.helper = Path(helper)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.runner = runner

    @classmethod
    def prepare(
        cls,
        directory: str | Path,
        *,
        rebuild: bool = False,
        runner: Runner = subprocess.run,
        platform: str = sys.platform,
        notify: Callable[[str], None] | None = None,
        **options: Any,
    ) -> tuple[AppleWriter | None, WriterStatus]:
        """Build (first use only) and check the helper. Returns the writer only when the model is available."""
        target = helper_path(directory)
        if notify is not None and platform == "darwin" and (rebuild or not target.is_file()):
            notify("Preparing Apple's on-device writer (one-time Swift build, up to a minute)…")
        try:
            helper = build_helper(directory, force=rebuild, runner=runner, platform=platform)
        except WriterUnavailable as exc:
            return None, exc.status()
        writer = cls(helper, runner=runner, **options)
        status = writer.status()
        return (writer if status.available else None), status

    def status(self) -> WriterStatus:
        try:
            completed = self.runner([str(self.helper), "--check"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return WriterUnavailable("helperFailed", str(exc)).status()
        answer = _last_json(completed.stdout)
        if answer is None:
            detail = (completed.stderr or "").strip().splitlines()[-1:] or [f"exit code {completed.returncode}"]
            return WriterUnavailable("helperFailed", detail[0][:300]).status()
        reason = str(answer.get("reason") or "unknown")
        if answer.get("available") is True:
            os_version = answer.get("os")
            return WriterStatus("apple", True, "available", f"Apple on-device model, macOS {os_version}")
        return WriterUnavailable(reason if reason in REASONS else "unknown").status()

    def generate(self, instructions: str, prompt: str, *, max_tokens: int | None = None) -> str:
        request = {
            "instructions": instructions,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        try:
            completed = self.runner(
                [str(self.helper)],
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise TextUnavailableError(f"the on-device writer took longer than {self.timeout_s:.0f}s") from None
        except OSError as exc:
            raise TextUnavailableError(f"cannot run the on-device writer: {exc}") from None
        answer = _last_json(completed.stdout)
        if answer is None:
            raise TextUnavailableError(f"the on-device writer gave no answer (exit code {completed.returncode})")
        if answer.get("ok") is True and isinstance(answer.get("text"), str):
            return str(answer["text"])
        code = str(answer.get("error") or "failed")
        if code == "exceededContextWindowSize":
            raise _ContextTooLong()
        if code == "guardrailViolation":
            raise TextUnavailableError("Apple's on-device model declined this request (its safety guardrails)")
        meaning = REASONS.get(code, (str(answer.get("message") or code), ""))[0]
        raise TextUnavailableError(f"the on-device writer failed: {meaning}")

    def write(self, context: Mapping[str, Any]) -> str:
        try:
            raw = self.generate(WRITER_INSTRUCTIONS, writer_prompt(context))
        except _ContextTooLong:
            try:  # the model has a small context window: retry once without the screen text
                raw = self.generate(WRITER_INSTRUCTIONS, writer_prompt(context, screen_chars=0))
            except _ContextTooLong:
                raise TextUnavailableError("the request is too long for the on-device model") from None
        text = clean_generated(raw)
        if not text:
            raise TextUnavailableError("the on-device writer returned no text; nothing typed")
        return text

    def close(self) -> None:
        return None


class _ContextTooLong(TextUnavailableError):
    pass
