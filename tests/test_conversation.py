"""Talking it through: questions before starting, typed answers to hand-offs, spoken replies, and `jevosx ask`."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import pytest

from jevosx.agent import Agent
from jevosx.config import Settings
from jevosx.executor.keys import key_vocabulary
from jevosx.planner import Planner, parse_sections
from jevosx.router.policy import JevRouter
from jevosx.ui.server import Speaker, UIServer, announce, console_file
from tests.fakes import FB_GOAL, FakeDesktop, element, observation, scripted_client
from tests.test_ui import demo_settings, make_manager, request

READING = """STEPS:
1. Open facebook.com/marketplace/create/item
2. Fill in the listing
VALUES:
website: facebook.com/marketplace/create/item
title: Plastic welding gun
price: 40
category: Tools
ASK:
photos: Where are the photos of the welding gun saved
condition: What condition is the welding gun in?
password: What is your Facebook password?
title: What should the title be?
extra: One more question?
"""


def test_the_reader_lists_only_what_the_person_must_provide():
    steps, values, questions = parse_sections(READING)
    assert steps[0].startswith("Open") and values["category"] == "Tools"
    names = [name for name, _ in questions]
    assert "password" not in names  # never asked for: passwords come from the Keychain
    assert questions[0] == ("photos", "Where are the photos of the welding gun saved?")
    assert len(questions) == 3  # at most three
    assert parse_sections("STEPS:\n1. a\n2. b\nVALUES: none\nASK: none")[2] == []


class Reader:
    model = "reader"

    def generate(self, instructions, prompt, **kwargs):
        return READING

    def write(self, context):
        return "text"

    def close(self):
        return None


def test_questions_the_request_already_answers_are_not_asked():
    reading = Planner(Reader()).read(FB_GOAL)
    assert [name for name, _ in reading.questions] == ["photos", "condition", "extra"]  # "title" is known


def form():
    fields = [
        element(1, "AXTextField", "Condition", kind="text_input", ops=("TYPE_TEXT", "CLICK"), in_web_area=True),
        element(2, "AXButton", "Next"),
    ]
    return FakeDesktop({"form": lambda: observation(fields)}, "form", {})


def done(body):
    return {"operation": "DONE"}


def agent_with(desktop, requests, *, clarify=None, handoff=None, answer=None, settings=None):
    settings = settings or Settings()
    settings.agent.fallback_log = ""
    reader = Reader()
    return Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(answer or done, requests), keys=key_vocabulary()),
        settings=settings,
        text_writer=reader,
        planner=Planner(reader),
        clarify=clarify,
        handoff=handoff,
        sleep=lambda _s: None,
    )


def test_answers_given_before_starting_become_text_the_agent_can_type():
    asked: list[list[tuple[str, str]]] = []
    requests: list[dict[str, Any]] = []

    def clarify(questions):
        asked.append(questions)
        return {"condition": "Used - good", "photos": "", "extra": "  "}

    result = agent_with(form(), requests, clarify=clarify).run(FB_GOAL)
    assert asked and asked[0][1][0] == "condition"
    slots = requests[0]["state"]["text_slots"]
    assert slots["condition"] == "Used - good" and "photos" not in slots  # skipped answers stay out
    assert [e.status for e in result.events][:2] == ["plan", "ask"]


def test_stopping_at_the_questions_stops_the_run_and_asking_first_can_be_turned_off():
    requests: list[dict[str, Any]] = []
    result = agent_with(form(), requests, clarify=lambda questions: None).run(FB_GOAL)
    assert result.status == "aborted" and requests == []
    settings = Settings()
    settings.agent.ask_first = False
    asked: list[Any] = []
    agent_with(form(), requests, clarify=lambda q: asked.append(q) or {}, settings=settings).run(FB_GOAL)
    assert asked == []


def test_a_hand_off_can_be_answered_in_words():
    requests: list[dict[str, Any]] = []

    def answer(body):
        if any("answered" in str(h.get("result")) for h in body["state"].get("recent_actions", [])):
            return {"operation": "DONE"}
        return {"operation": ("ASK_USER", 0.9), "handoff_reason": "info"}

    agent = agent_with(form(), requests, answer=answer, handoff=lambda need, obs: "Used - like new")
    agent.run("list my drill")
    assert requests[-1]["state"]["text_slots"]["answer"] == "Used - like new"


def test_the_mac_voice_speaks_one_sentence_at_a_time(tmp_path):
    spoken = tmp_path / "spoken.txt"
    script = tmp_path / "say"
    script.write_text(f'#!/bin/sh\nprintf "%s\\n" "$1" >> "{spoken}"\n')
    script.chmod(0o755)
    speaker = Speaker(str(script))
    assert speaker.available and speaker.say("Should I   publish the listing?")
    assert spoken.read_text() == "Should I publish the listing?\n"  # whitespace tidied
    assert not speaker.say("   ") and not Speaker("").say("hello")
    assert len(Speaker.MAX_CHARS * "x") == Speaker.MAX_CHARS


@pytest.fixture
def console(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    manager, _, _ = make_manager()
    ui = UIServer(demo_settings(), demo=True, port=0, manager=manager)
    ui.speaker = Speaker("")  # no speech in tests
    ui.start_background()
    announce(ui.url, ui.token)
    yield ui
    ui.shutdown()


def test_ask_starts_a_run_in_the_open_console(console, capsys):
    from jevosx.cli import main

    assert json.loads(console_file().read_text())["token"] == console.token
    assert oct(console_file().stat().st_mode & 0o777) == "0o600"
    assert main(["ask", "Open", "Notes"]) == 0 and "started: Open Notes" in capsys.readouterr().out
    deadline = time.monotonic() + 10
    while console.manager.running and time.monotonic() < deadline:
        time.sleep(0.02)
    assert console.manager.history[0]["goal"] == "Open Notes"
    status = json.loads(request(console, "GET", "/api/status")[1])
    assert status["speech"] is False
    assert request(console, "POST", "/api/say", {"text": "hello"})[0] == 200
    assert request(console, "POST", "/api/say", {})[0] == 400


def test_ask_without_a_console_says_how_to_start_one(tmp_path, monkeypatch, capsys):
    from jevosx.cli import main

    monkeypatch.setenv("HOME", str(tmp_path))
    assert main(["ask", "Open Notes"]) == 2 and "jevosx ui" in capsys.readouterr().err


def test_the_console_takes_answers_to_questions_over_http(console):
    status, data = request(console, "POST", "/api/run", {"goal": FB_GOAL})
    assert status == 202
    deadline = time.monotonic() + 10
    pending: list[dict[str, Any]] = []
    while not pending and time.monotonic() < deadline:
        pending = json.loads(request(console, "GET", "/api/state")[1])["pending_approvals"]
        time.sleep(0.02)
    assert pending[0]["category"] == "questions" and pending[0]["questions"][0]["name"] == "condition"
    reply = {"request_id": pending[0]["request_id"], "allow": True, "answers": {"condition": "Used - fair"}}
    assert request(console, "POST", "/api/approve", reply)[0] == 200
    assert request(console, "POST", "/api/approve", {"request_id": "x", "allow": True, "answers": "no"})[0] == 400
    while console.manager.running and time.monotonic() < deadline:
        time.sleep(0.02)
    listing = console.manager.components().observer.apps["Safari"].listing
    assert listing["Condition"] == "Used - fair"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_console_file_is_private(console):
    assert console_file().stat().st_mode & 0o077 == 0
