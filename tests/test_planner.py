import pytest

from jevosx.agent import Agent
from jevosx.config import Settings
from jevosx.errors import TextUnavailableError
from jevosx.executor.keys import key_vocabulary
from jevosx.planner import Planner, merge_slots, needs_plan, parse_plan, parse_reading
from jevosx.router.policy import JevRouter
from tests.fakes import FB_GOAL, FakeDesktop, element, observation, scripted_client


@pytest.mark.parametrize(
    ("goal", "wanted"),
    [
        ("Write a poem about autumn, save it as poem.rtf, then open it in Pages", True),
        ("Open TextEdit and write a short poem about the sea", True),
        ("Open Notes", False),
        ("Delete the Trip ideas note", False),
        ("search for red flowers and cats", True),
    ],
)
def test_needs_plan(goal, wanted):
    assert needs_plan(goal) is wanted


def test_parse_plan_keeps_ordered_distinct_steps():
    text = "Sure!\n1. Open TextEdit\n2) Write a poem about autumn\n- Save it as poem.rtf\n2. Write a poem about autumn"
    assert parse_plan(text) == ["Open TextEdit", "Write a poem about autumn", "Save it as poem.rtf"]
    assert parse_plan("\n".join(f"{i}. step {i}" for i in range(1, 10))) == [f"step {i}" for i in range(1, 7)]
    assert parse_plan("1. only one step") == [] and parse_plan("no list at all") == []


class Writer:
    model = "fake"

    def __init__(self, answer="1. Open TextEdit\n2. Write the poem", error=None):
        self.answer, self.error, self.calls = answer, error, []

    def write(self, context):
        return "text"

    def generate(self, instructions, prompt, *, max_tokens=None, temperature=None, timeout_s=None):
        self.calls.append((instructions, prompt, temperature))
        if self.error:
            raise self.error
        return self.answer

    def close(self):
        return None


def test_planner_is_quiet_on_failure_and_skips_simple_goals():
    writer = Writer()
    assert Planner(writer).plan("Open Notes") == [] and writer.calls == []
    assert Planner(writer).plan("Open TextEdit and write a poem") == ["Open TextEdit", "Write the poem"]
    assert writer.calls[0][2] == 0.2 and "Request: Open TextEdit and write a poem" in writer.calls[0][1]
    failing = Writer(error=TextUnavailableError("model declined"))
    assert Planner(failing).plan("Open TextEdit and write a poem") == []


def test_agent_shows_the_plan_and_sends_it_to_jev_as_a_hint():
    desktop = FakeDesktop({"doc": lambda: observation([element(1, "AXButton", "New Document")])}, "doc", {})
    requests: list[dict] = []
    settings = Settings()
    settings.agent.fallback_log = ""
    agent = Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(lambda body: {"operation": "DONE"}, requests), keys=key_vocabulary()),
        settings=settings,
        planner=Planner(Writer()),
        sleep=lambda _s: None,
    )
    result = agent.run("Open TextEdit and write a poem about rain")
    plan_event = result.events[0]
    assert (plan_event.step, plan_event.status, plan_event.action) == (0, "plan", "PLAN")
    assert plan_event.message == "1. Open TextEdit · 2. Write the poem"
    assert requests[0]["state"]["plan"] == ["1. Open TextEdit", "2. Write the poem"]
    assert "suggested order of steps" in str(requests[0]["questions"]["operation"]["instructions"]["rules"])
    requests.clear()
    agent.run("Open Notes")
    assert "plan" not in requests[0]["state"]


MODEL_ANSWER = """**STEPS:**
1. Open facebook.com/marketplace in the browser
2. Create a new listing: item for sale
3. Fill in the title, price and description
VALUES:
- Website: https://www.facebook.com/marketplace/
- Title: "Plastic welding gun"
- Price: 40 euros
- Description: Plastic welding gun in good condition, heats up fast, ideal for bumper and tank repairs.
- Password: hunter2
- Notes: none"""


def test_reading_turns_a_free_form_request_into_steps_and_values():
    steps, values = parse_reading(MODEL_ANSWER)
    assert steps[0] == "Open facebook.com/marketplace in the browser" and len(steps) == 3
    assert values == {
        "website": "www.facebook.com/marketplace",
        "title": "Plastic welding gun",
        "price": "40",
        "description": "Plastic welding gun in good condition, heats up fast, ideal for bumper and tank repairs.",
    }  # the password line is dropped: passwords only ever come from the Keychain


def test_reading_tolerates_missing_headers_and_empty_values():
    assert parse_reading("1. Open Notes\n2. Make a new note") == (["Open Notes", "Make a new note"], {})
    assert parse_reading("STEPS:\n1. a\n2. b\nVALUES: none") == (["a", "b"], {})
    assert parse_reading("VALUES: search: population of Malta")[1] == {"search": "population of Malta"}


def test_model_values_come_first_and_patterns_fill_gaps():
    model = {"title": "Plastic welding gun", "price": "40"}
    patterns = {"item_name": "plastic welding gun", "price": "41", "url_1": "facebook.com"}
    assert merge_slots(model, patterns) == {"title": "Plastic welding gun", "price": "40", "url_1": "facebook.com"}


def test_agent_offers_the_values_the_model_read_from_the_goal():
    field = element(1, "AXTextField", "Title", kind="text_input", ops=("TYPE_TEXT", "CLICK"), in_web_area=True)
    desktop = FakeDesktop({"form": lambda: observation([field, element(2, "AXButton", "Publish")])}, "form", {})
    requests: list[dict] = []
    settings = Settings()
    settings.agent.fallback_log = ""
    writer = Writer(answer=MODEL_ANSWER)
    agent = Agent(
        observer=desktop,
        executor=desktop,
        router=JevRouter(scripted_client(lambda body: {"operation": "DONE"}, requests), keys=key_vocabulary()),
        settings=settings,
        text_writer=writer,
        planner=Planner(writer),
        sleep=lambda _s: None,
    )
    result = agent.run(FB_GOAL)
    slots = requests[0]["state"]["text_slots"]
    assert slots["title"] == "Plastic welding gun" and slots["price"] == "40" and "description" in slots
    assert "password" not in slots and "GENERATE" in slots
    assert "title “Plastic welding gun”" in result.events[0].message and "description (" in result.events[0].message
