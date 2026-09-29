"""Typed text fits the field it goes into: addresses into a browser's address bar (followed by Return), listing text
into the listing form. Seen live: "plastic welding gun" was typed into Chrome's address bar and became a Google
search, because the text was chosen before Jev knew which field it was for."""

from __future__ import annotations

from typing import Any

import pytest

from jevosx.executor.keys import key_vocabulary
from jevosx.observer.walker import dedupe_browser_fields
from jevosx.router.client import ChoiceAnswer, JevResponse
from jevosx.router.policy import JevRouter, restrict
from jevosx.router.text import ADDRESS, CONTENT, SEARCH, TextSource, slot_kind
from jevosx.types import TYPE_TEXT, Action, AppInfo
from tests.fakes import distribution, element, observation, scripted_client

CHROME = AppInfo("Google Chrome", "com.google.Chrome", pid=300)
LISTING = {"website": "facebook.com/marketplace/create/item", "title": "Plastic welding gun", "price": "40"}


def address_bar(**kwargs: Any):
    return element(5, "AXTextField", "Address and search bar", kind="text_input", ops=("TYPE_TEXT", "CLICK"), **kwargs)


def title_field():
    return element(17, "AXTextField", "Title", kind="text_input", ops=("TYPE_TEXT", "CLICK"), in_web_area=True)


@pytest.mark.parametrize(
    ("name", "value", "kind"),
    [
        ("website", "facebook.com", ADDRESS),
        ("url_1", "facebook.com/marketplace/create/item", ADDRESS),
        ("quote_1", "https://example.com/a", ADDRESS),
        ("quote_1", "report.pdf", CONTENT),  # a file name is not a site
        ("search", "population of Malta", SEARCH),
        ("phrase_1", "red flowers", SEARCH),
        ("title", "Plastic welding gun", CONTENT),
        ("GENERATE", "", CONTENT),
    ],
)
def test_slots_know_what_they_hold(name, value, kind):
    assert slot_kind(name, value) == kind


def test_each_field_is_offered_only_text_that_fits():
    text = TextSource(LISTING)
    assert text.compatible(address_bar(), CHROME) == ["website"]
    assert text.compatible(title_field(), CHROME) == ["title", "price"]
    website_field = element(9, "AXTextField", "Website", kind="text_input", in_web_area=True)
    assert "website" in text.compatible(website_field, CHROME)  # a page field that asks for an address
    notes = AppInfo("Notes", "com.apple.Notes", pid=7)
    assert text.compatible(address_bar(), notes) == ["title", "price"]  # not a browser: an ordinary field
    nothing_fits = TextSource({"title": "x", "price": "1"})
    assert nothing_fits.compatible(address_bar(), CHROME) == ["title", "price"]  # then everything stays offered


def test_addresses_and_searches_typed_into_the_address_bar_are_submitted():
    text = TextSource({**LISTING, "search": "red flowers"})
    assert text.submits("website", address_bar(), CHROME) and text.submits("search", address_bar(), CHROME)
    assert not text.submits("title", title_field(), CHROME)


def test_restricting_keeps_jev_calibration():
    answer = ChoiceAnswer("title", {"title": 0.5, "website": 0.4, "price": 0.1}, 0.4)
    fit = restrict(answer, ["website", "price"])
    assert fit.choice == "website" and fit.probabilities["website"] == pytest.approx(0.8)
    assert fit.confidence == pytest.approx(0.8 * 0.4 / 0.5)


def form_obs():
    return observation([address_bar(), title_field(), element(18, "AXButton", "Next")], app=CHROME)


def test_the_only_fitting_text_is_certain_and_an_unsure_one_is_asked_again_for_the_field():
    requests: list[dict[str, Any]] = []

    def answer(body):
        if "field" in body["questions"]["text_slot"]["instructions"]:  # the second question shows Jev the field
            return {"text_slot": ("title", 0.95)}
        return {"operation": ("TYPE_TEXT", 0.9), "type_text_target": "17", "text_slot": ("website", 0.5)}

    r = JevRouter(scripted_client(answer, requests), keys=key_vocabulary())
    obs = form_obs()
    text = TextSource(LISTING)
    decision = r.decide("sell it", obs, r.space(obs, text), text_source=text)
    assert decision.target.element.label == "Title" and decision.text_option == "title"
    assert decision.text_follow_up and len(requests) == 2
    assert set(requests[1]["questions"]["text_slot"]["criteria"]) == {"title", "price"}
    assert requests[1]["questions"]["text_slot"]["instructions"]["field"]["label"] == "Title"

    requests.clear()

    def into_bar(body):
        return {"operation": ("TYPE_TEXT", 0.9), "type_text_target": "5", "text_slot": ("title", 0.6)}

    r = JevRouter(scripted_client(into_bar, requests), keys=key_vocabulary())
    decision = r.decide("sell it", obs, r.space(obs, text), text_source=text)
    assert decision.text_option == "website" and decision.text_answer.confidence == 1.0 and len(requests) == 1


def test_clicking_a_field_and_typing_into_it_are_one_intent():
    obs = form_obs()
    text = TextSource(LISTING)
    r = JevRouter(scripted_client(lambda body: {}), keys=key_vocabulary())
    space = r.space(obs, text)
    ops = list(space.operations)
    probabilities = {op: 0.0 for op in ops} | {"CLICK": 0.4, "TYPE_TEXT": 0.4, "WAIT": 0.2}
    response = JevResponse(
        model="m",
        answers={
            "operation": {"choice": "CLICK", "probabilities": probabilities, "confidence": 0.3},
            "click_target": distribution(list(space.targets_for("CLICK")), "17"),
            "type_text_target": distribution(list(space.targets_for("TYPE_TEXT")), "17"),
            "text_slot": distribution(list(text.options()), "title"),
        },
    )
    decision = r.decode(response, space, text, obs)
    assert decision.operation == "CLICK" and decision.confidence == pytest.approx(0.6)  # 0.3 x (0.4 + 0.4) / 0.4
    elsewhere = dict(response.answers, type_text_target=distribution(list(space.targets_for("TYPE_TEXT")), "5"))
    assert r.decode(JevResponse(model="m", answers=elsewhere), space, text, obs).confidence == 0.3


def test_the_mac_executor_presses_return_after_an_address(monkeypatch):
    from jevosx.config import ExecutorSettings
    from jevosx.executor import input as keyboard
    from jevosx.executor.mac import MacExecutor

    class Field:
        def __init__(self) -> None:
            self.values: dict[str, Any] = {}

        def set(self, attribute: str, value: Any) -> None:
            self.values[attribute] = value

        def get(self, attribute: str, default: Any = None) -> Any:
            return self.values.get(attribute, default)

    node = Field()
    sent: list[str] = []
    monkeypatch.setattr(keyboard, "post_chord", lambda chord, delay_s=0: sent.append(str(chord)))
    monkeypatch.setattr(keyboard, "type_text", lambda text, delay_s=0: (sent.append(text), node.set("AXValue", text)))
    executor = object.__new__(MacExecutor)
    executor.settings = ExecutorSettings()
    executor._frontmost_pid = lambda: 300
    obs = observation([], app=CHROME)
    bar = address_bar(node=node)
    result = executor.execute(Action(TYPE_TEXT, element=bar, text="facebook.com", submit=True), obs)
    assert result.ok and result.detail == "typed and pressed Return"
    assert sent == ["cmd+a", "facebook.com", "forwarddelete", "return"]


def test_an_address_is_opened_without_chrome_s_inline_completion(monkeypatch):
    """Seen live: "facebook.com" + Return opened the completion Chrome had selected after it, the create-listing
    page from earlier runs. Forward Delete drops the selected completion before Return."""
    from jevosx.config import ExecutorSettings
    from jevosx.executor import input as keyboard
    from jevosx.executor.mac import MacExecutor

    class Bar:
        def __init__(self) -> None:
            self.typed = ""
            self.completion = ""

        def set(self, attribute: str, value: Any) -> None:
            pass

        def get(self, attribute: str, default: Any = None) -> Any:
            return self.typed + self.completion if attribute == "AXValue" else default

    node = Bar()
    opened: list[str] = []

    def chord(chord: Any, delay_s: float = 0) -> None:
        if str(chord) == "forwarddelete":
            node.completion = ""
        elif str(chord) == "return":
            opened.append(node.typed + node.completion)

    def type_text(text: str, delay_s: float = 0) -> None:
        node.typed, node.completion = text, "/marketplace/create/item"

    monkeypatch.setattr(keyboard, "post_chord", chord)
    monkeypatch.setattr(keyboard, "type_text", type_text)
    executor = object.__new__(MacExecutor)
    executor.settings = ExecutorSettings()
    executor._frontmost_pid = lambda: 300
    executor.execute(
        Action(TYPE_TEXT, element=address_bar(node=node), text="facebook.com", submit=True), observation([], app=CHROME)
    )
    assert opened == ["facebook.com"]


@pytest.mark.parametrize(("focused_label", "ok"), [("Search Marketplace", True), ("Search Facebook", False)])
def test_typed_text_is_found_in_the_field_that_replaced_the_observed_one(monkeypatch, focused_label, ok):
    """Seen live: Facebook re-rendered "Search Marketplace" while its page loaded. The keys went to the new combobox,
    the observed node kept reading "", and the agent typed the search a second time. A focused field with the same
    role and label counts; any other field does not."""
    from jevosx.config import ExecutorSettings
    from jevosx.executor import input as keyboard
    from jevosx.executor.mac import MacExecutor

    class Node:
        def __init__(self, **values: Any) -> None:
            self.values = values

        def set(self, attribute: str, value: Any) -> None:
            pass

        def get(self, attribute: str, default: Any = None) -> Any:
            return self.values.get(attribute, default)

        def get_many(self, attributes: tuple[str, ...]) -> dict[str, Any]:
            return {a: self.values[a] for a in attributes if a in self.values}

    replacement = Node(AXRole="AXComboBox", AXDescription=focused_label)
    app = Node(AXFocusedUIElement=replacement)
    monkeypatch.setattr(keyboard, "post_chord", lambda chord, delay_s=0: None)
    monkeypatch.setattr(keyboard, "type_text", lambda text, delay_s=0: replacement.values.update(AXValue=text))
    executor = object.__new__(MacExecutor)
    executor.settings = ExecutorSettings(settle_timeout_s=0.1, settle_poll_s=0.01)
    executor._frontmost_pid = lambda: 300
    executor._AXNode = type("AX", (), {"application": staticmethod(lambda pid: app)})
    combobox = element(
        40, "AXComboBox", "Search Marketplace", kind="text_input", ops=("TYPE_TEXT",), in_web_area=True, node=Node()
    )
    result = executor.execute(Action(TYPE_TEXT, element=combobox, text="welding machine"), observation([], app=CHROME))
    assert result.ok is ok


def test_a_browser_window_offers_one_address_bar():
    shown = address_bar(value="google.com/search?q=plastic+welding+gun")
    edited = address_bar(value="plastic welding gun", focused=True)
    edited.index = 41
    page_search = element(7, "AXTextField", "Address and search bar", kind="text_input", in_web_area=True)
    kept = dedupe_browser_fields([shown, element(6, "AXButton", "Reload"), edited, page_search])
    assert [(e.index, e.label, e.value) for e in kept] == [
        (1, "Address and search bar", "plastic welding gun"),  # the one being edited
        (2, "Reload", None),
        (3, "Address and search bar", None),  # a field inside the page is another field
    ]
