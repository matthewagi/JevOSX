import json
import subprocess
from pathlib import Path

import pytest

from jevosx.agent import Agent
from jevosx.config import Settings
from jevosx.errors import TextUnavailableError
from jevosx.executor.keys import key_vocabulary
from jevosx.router.policy import JevRouter
from jevosx.router.text import GENERATE, TextSource
from jevosx.writer import clean_generated, create_writer, wants_generation, writer_prompt
from jevosx.writer.apple import AppleWriter, WriterUnavailable, build_helper, helper_path
from jevosx.writer.openai import LLMTextWriter
from tests.fakes import FakeDesktop, element, observation, scripted_client


@pytest.mark.parametrize(
    ("goal", "wanted"),
    [
        ("write a poem about autumn in TextEdit", True),
        ("Open Notes and jot down a haiku about rain", True),
        ("reply to Anna's email saying I'll be late", True),
        ("summarize this page into a new note", True),
        ('In TextEdit, write "Shopping list" and save it as "groceries"', False),
        ('write the text "hello" in Notes', False),
        ('write a poem called "Rain"', True),
        ("search the web for pictures of red flowers", False),
        ("Open Downloads in Finder", False),
        ('Create a new document and type "hello"', False),
    ],
)
def test_wants_generation(goal, wanted):
    assert wants_generation(goal) is wanted


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  Roses are red\nViolets are blue  ", "Roses are red\nViolets are blue"),
        ("Here is a poem about autumn:\n\nLeaves fall\nslowly", "Leaves fall\nslowly"),
        ("Sure! Here's your haiku:\nsoft rain", "soft rain"),
        ('"A quoted title"', "A quoted title"),
        ("“Curly quoted”", "Curly quoted"),
        ('"Hi" she said, "bye"', '"Hi" she said, "bye"'),
        ("```\nline one\nline two\n```", "line one\nline two"),
        ("Here is the plan. We start at nine.", "Here is the plan. We start at nine."),
    ],
)
def test_clean_generated(raw, clean):
    assert clean_generated(raw) == clean


def test_writer_prompt_names_the_field_and_bounds_screen_text():
    context = {
        "goal": "write a poem about autumn",
        "field": {"label": "Body", "role": "textarea", "current_value": "old"},
        "app": "TextEdit",
        "window": "Untitled",
        "screen_text": "x" * 5000,
    }
    prompt = writer_prompt(context)
    assert "Request: write a poem about autumn" in prompt and "Body (textarea) in TextEdit" in prompt
    assert "currently contains" in prompt and prompt.count("x") <= 1600
    assert "Screen text" not in writer_prompt(context, screen_chars=0)


# ---- Apple helper build ------------------------------------------------------------------------------------------
class FakeRunner:
    """Stands in for subprocess.run: records commands and answers per executable."""

    def __init__(self, *, xcode=0, sdk="", compile_rc=0, compile_err="", check=None, answers=None):
        self.calls: list[list[str]] = []
        self.inputs: list[str | None] = []
        self.xcode, self.sdk, self.compile_rc, self.compile_err = xcode, sdk, compile_rc, compile_err
        self.check = check or {"available": True, "reason": "available", "framework": True, "os": "26.3.0"}
        self.answers = list(answers or [])
        self.timeout = False

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        self.inputs.append(kwargs.get("input"))
        if cmd[:2] == ["xcode-select", "-p"]:
            return subprocess.CompletedProcess(cmd, self.xcode, "/Library/Developer/CommandLineTools\n", "")
        if "--show-sdk-path" in cmd:
            return subprocess.CompletedProcess(cmd, 0, self.sdk + "\n", "")
        if "swiftc" in cmd:
            if self.compile_rc == 0:
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\xcf\xfa\xed\xfe binary")
            return subprocess.CompletedProcess(cmd, self.compile_rc, "", self.compile_err)
        if cmd[-1] == "--check":
            return subprocess.CompletedProcess(cmd, 0, json.dumps(self.check) + "\n", "")
        if self.timeout:
            raise subprocess.TimeoutExpired(cmd, 60)
        answer = self.answers.pop(0)
        return subprocess.CompletedProcess(cmd, 0, answer if isinstance(answer, str) else json.dumps(answer), "")


def test_build_helper_compiles_once_with_weak_linking(tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "System/Library/Frameworks/FoundationModels.framework").mkdir(parents=True)
    stale = tmp_path / "bin" / "jevosx-writer-000000000000"
    stale.parent.mkdir()
    stale.write_text("old")
    runner = FakeRunner(sdk=str(sdk))
    path = build_helper(tmp_path / "bin", runner=runner, platform="darwin")
    assert path == helper_path(tmp_path / "bin") and path.is_file() and path.stat().st_mode & 0o111
    compile_cmd = next(c for c in runner.calls if "swiftc" in c)
    assert compile_cmd[:4] == ["xcrun", "--sdk", "macosx", "swiftc"] and "-parse-as-library" in compile_cmd
    assert compile_cmd[-4:] == ["-Xlinker", "-weak_framework", "-Xlinker", "FoundationModels"]
    assert not stale.exists() and not list((tmp_path / "bin").glob(".build-*"))
    runner.calls.clear()
    assert build_helper(tmp_path / "bin", runner=runner, platform="darwin") == path and runner.calls == []


def test_build_helper_without_the_framework_in_the_sdk_skips_the_weak_link(tmp_path):
    runner = FakeRunner(sdk=str(tmp_path / "old-sdk"))
    build_helper(tmp_path, runner=runner, platform="darwin")
    assert "-weak_framework" not in next(c for c in runner.calls if "swiftc" in c)


@pytest.mark.parametrize(
    ("kwargs", "platform", "reason"),
    [
        ({}, "linux", "notMacOS"),
        ({"xcode": 2}, "darwin", "noCompiler"),
        ({"compile_rc": 1, "compile_err": "error: cannot find 'SystemLanguageModel'"}, "darwin", "buildFailed"),
    ],
)
def test_build_helper_failures_have_reasons(tmp_path, kwargs, platform, reason):
    with pytest.raises(WriterUnavailable) as raised:
        build_helper(tmp_path, runner=FakeRunner(**kwargs), platform=platform)
    assert raised.value.reason == reason
    status = raised.value.status()
    assert not status.available and status.reason == reason
    if reason == "buildFailed":
        assert "SystemLanguageModel" in status.detail
    assert not list(tmp_path.glob("jevosx-writer-*"))


def test_prepare_reports_why_the_model_is_unavailable(tmp_path):
    runner = FakeRunner(check={"available": False, "reason": "appleIntelligenceNotEnabled", "framework": True})
    notes: list[str] = []
    writer, status = AppleWriter.prepare(tmp_path, runner=runner, platform="darwin", notify=notes.append)
    assert writer is None and status.reason == "appleIntelligenceNotEnabled"
    assert "Apple Intelligence" in status.hint and notes and "one-time" in notes[0]
    runner.check = {"available": True, "reason": "available", "framework": True, "os": "26.3.0"}
    writer, status = AppleWriter.prepare(tmp_path, runner=runner, platform="darwin", notify=notes.append)
    assert writer is not None and status.available and len(notes) == 1  # already built: no second notice


def test_status_survives_a_broken_helper(tmp_path):
    writer = AppleWriter(tmp_path / "missing")
    status = writer.status()
    assert not status.available and status.reason == "helperFailed"


# ---- Apple writer requests ---------------------------------------------------------------------------------------
CONTEXT = {
    "goal": "Open TextEdit and write a haiku about rain",
    "field": {"label": "Body", "role": "textarea", "current_value": ""},
    "app": "TextEdit",
    "window": "Untitled",
    "screen_text": "Untitled — Edited",
}


def test_write_cleans_the_answer_and_sends_instructions(tmp_path):
    runner = FakeRunner(answers=[{"ok": True, "text": "Here is a haiku:\n\nsoft rain on glass\nthe kettle sings"}])
    writer = AppleWriter(tmp_path / "helper", runner=runner, temperature=0.4, max_tokens=300)
    assert writer.write(CONTEXT) == "soft rain on glass\nthe kettle sings"
    request = json.loads(runner.inputs[-1])
    assert request["temperature"] == 0.4 and request["max_tokens"] == 300
    assert "exact text that will be typed" in request["instructions"] and "haiku about rain" in request["prompt"]


def test_write_retries_without_screen_text_when_the_context_is_too_long(tmp_path):
    runner = FakeRunner(
        answers=[{"ok": False, "error": "exceededContextWindowSize", "message": "too long"}, {"ok": True, "text": "ok"}]
    )
    assert AppleWriter(tmp_path / "helper", runner=runner).write(CONTEXT) == "ok"
    first, second = (json.loads(i)["prompt"] for i in runner.inputs)
    assert "Screen text" in first and "Screen text" not in second


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ({"ok": False, "error": "guardrailViolation", "message": "unsafe"}, "declined"),
        ({"ok": False, "error": "appleIntelligenceNotEnabled", "message": "off"}, "Apple Intelligence is turned off"),
        ("not json at all", "no answer"),
        ({"ok": True, "text": "   "}, "no text"),
    ],
)
def test_write_failures_become_text_unavailable(tmp_path, answer, message):
    writer = AppleWriter(tmp_path / "helper", runner=FakeRunner(answers=[answer]))
    with pytest.raises(TextUnavailableError, match=message):
        writer.write(CONTEXT)


def test_write_timeout(tmp_path):
    runner = FakeRunner()
    runner.timeout = True
    with pytest.raises(TextUnavailableError, match="longer than"):
        AppleWriter(tmp_path / "helper", runner=runner, timeout_s=5).write(CONTEXT)


# ---- backend selection ---------------------------------------------------------------------------------------------
def test_create_writer_backends(tmp_path, monkeypatch):
    monkeypatch.delenv("TEXT_MODEL_API_KEY", raising=False)
    settings = Settings()
    settings.writer.helper_dir = str(tmp_path)
    settings.writer.backend = "off"
    assert create_writer(settings)[0] is None
    settings.writer.backend = "auto"
    writer, status = create_writer(settings, platform="linux")
    assert writer is None and status.backend == "apple" and status.reason == "notMacOS"
    writer, status = create_writer(settings, backend="openai")
    assert writer is None and status.reason == "notConfigured"
    settings.text_model.model = "some/model"
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "k")
    writer, status = create_writer(settings, platform="linux")
    assert isinstance(writer, LLMTextWriter) and status.available and status.backend == "openai"
    writer.close()


# ---- text source + agent -------------------------------------------------------------------------------------------
class FakeWriter:
    model = "fake-writer"

    def __init__(self, text="soft rain on glass"):
        self.text = text
        self.contexts: list[dict] = []

    def write(self, context):
        self.contexts.append(dict(context))
        return self.text

    def generate(self, instructions, prompt, *, max_tokens=None):
        return "1. step"

    def close(self):
        return None


def test_generate_is_offered_only_when_asked_and_composed_once_per_field():
    writer = FakeWriter()
    assert GENERATE not in TextSource({}, writer, generate=False).options()
    assert not TextSource({}, None, generate=True).available
    source = TextSource({}, writer, generate=True)
    assert list(source.options()) == [GENERATE] and source.available
    body = element(1, "AXTextArea", "Body", kind="text_input", ops=("TYPE_TEXT", "CLICK"))
    obs = observation([body])
    first = source.resolve(GENERATE, goal="write a haiku", element=body, obs=obs, history=[])
    again = source.resolve(GENERATE, goal="write a haiku", element=body, obs=obs, history=[])
    assert first == again and first.source == "model:fake-writer" and len(writer.contexts) == 1
    secure = element(2, "AXTextField", "Password", kind="text_input", ops=("TYPE_TEXT",), secure=True)
    with pytest.raises(TextUnavailableError, match="password"):
        source.resolve(GENERATE, goal="write a haiku", element=secure, obs=obs, history=[])


def test_agent_types_what_the_writer_composed(tmp_path):
    screens = {
        "doc": lambda: observation(
            [element(1, "AXTextArea", "Body", kind="text_input", ops=("TYPE_TEXT", "CLICK"), focused=True)],
            window="Untitled",
        ),
        "typed": lambda: observation(
            [element(1, "AXTextArea", "Body", value="soft rain on glass", kind="text_input", ops=("TYPE_TEXT",))],
            window="Untitled",
            text="soft rain on glass",
        ),
    }
    desktop = FakeDesktop(screens, "doc", {("doc", "TYPE_TEXT"): "typed"})
    requests: list[dict] = []

    def jev(body):
        values = [e.get("value") for e in body["state"]["elements"]]
        return {"operation": "DONE" if "soft rain on glass" in values else "TYPE_TEXT"}

    settings = Settings()
    settings.agent.fallback_log = ""
    writer = FakeWriter()
    agent = Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(jev, requests), keys=key_vocabulary()),
        settings=settings,
        text_writer=writer,
        sleep=lambda _s: None,
    )
    result = agent.run("Write a haiku about rain in this document")
    assert result.status == "done"
    assert desktop.executed == ['TYPE_TEXT [1] textarea "Body" <- soft rain on glass']
    assert requests[0]["state"]["text_slots"] == {
        GENERATE: "a writer composes the new text the goal asks for, for the field TYPE_TEXT chooses"
    }
    assert "text_slot" not in requests[0]["questions"]  # the only option needs no question
    assert "written by fake-writer" in result.events[0].message
    assert writer.contexts[0]["field"]["label"] == "Body" and writer.contexts[0]["app"] == "TextEdit"
