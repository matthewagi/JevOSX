import httpx
import pytest

from jevosx.errors import JevAuthError, JevError, JevResponseError
from jevosx.router.client import JevClient, choice_question, validate_choice


def client(handler, **kwargs):
    return JevClient("k" * 16, transport=httpx.MockTransport(handler), http2=False, **kwargs)


def ok_answer():
    return {"choice": "a", "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.6}


def test_request_shape_and_answer_validation():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["url"] = str(request.url)
        seen["body"] = request.read()
        return httpx.Response(200, json={"model": "jev-1", "answers": {"q": ok_answer()}, "usage": {"input_tokens": 9}})

    with client(handler) as jev:
        response = jev.evaluate({"x": 1}, {"q": choice_question({"a": "A", "b": "B"}, {"goal": "g"})})
    assert seen["auth"] == "Bearer " + "k" * 16
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert b'"model":"jev-latest"' in seen["body"] and b'"type":"choice"' in seen["body"]
    answer = response.choice("q", ["a", "b"])
    assert answer.choice == "a" and answer.probability == 0.7 and response.usage == {"input_tokens": 9}


@pytest.mark.parametrize(
    "raw",
    [
        {"choice": "z", "probabilities": {"a": 0.5, "b": 0.5}, "confidence": 0.5},  # not offered
        {"choice": "a", "probabilities": {"a": 0.5}, "confidence": 0.5},  # missing id
        {"choice": "b", "probabilities": {"a": 0.7, "b": 0.3}, "confidence": 0.5},  # not the argmax
        {"choice": "a", "probabilities": {"a": 0.9, "b": 0.9}, "confidence": 0.5},  # does not sum to 1
        {"choice": "a", "probabilities": {"a": 1.0, "b": 0.0}, "confidence": float("nan")},
        {"choice": "a"},
        "garbage",
    ],
)
def test_invalid_answers_never_validate(raw):
    with pytest.raises(JevResponseError):
        validate_choice(raw, ["a", "b"])


def test_retries_transient_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) < 3:
            return httpx.Response(529, headers={"retry-after": "0"})
        return httpx.Response(200, json={"model": "m", "answers": {"q": ok_answer()}})

    with client(handler, max_retries=2) as jev:
        jev.evaluate({}, {"q": choice_question({"a": "A", "b": "B"})})
    assert len(calls) == 3


def test_auth_and_hard_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(401)

    with client(handler) as jev, pytest.raises(JevAuthError):
        jev.evaluate({}, {"q": choice_question({"a": "A", "b": "B"})})
    assert len(calls) == 1

    with client(lambda r: httpx.Response(400, text="bad")) as jev, pytest.raises(JevError, match="HTTP 400"):
        jev.evaluate({}, {"q": choice_question({"a": "A", "b": "B"})})
    with client(lambda r: httpx.Response(200, json={"nope": 1})) as jev, pytest.raises(JevResponseError):
        jev.evaluate({}, {"q": choice_question({"a": "A", "b": "B"})})


def test_question_limits_and_missing_key():
    with pytest.raises(ValueError):
        choice_question({"only": "one"})
    with pytest.raises(ValueError):
        choice_question({str(i): str(i) for i in range(256)})
    with pytest.raises(JevAuthError):
        JevClient("")
