"""TypeSafe Jev client (System One endpoint).

Jev answers typed questions about a JSON state in one pass: for a `choice` question it returns the chosen id,
a probability for every offered id, and a confidence. This client keeps one pooled HTTP/2 connection alive (the
TLS handshake would otherwise dominate a sub-300 ms decision), retries only transient failures, and validates
every answer before anything downstream may act on it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..errors import JevAuthError, JevError, JevResponseError, RouterContractError

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
MAX_CHOICES = 255
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504, 529})


def choice_question(criteria: Mapping[str, Any], instructions: Any = None) -> dict[str, Any]:
    """Build a one-of-N question. `criteria` maps each option id to a description (string or object)."""
    if not 2 <= len(criteria) <= MAX_CHOICES:
        raise ValueError(f"a choice question needs 2..{MAX_CHOICES} options, got {len(criteria)}")
    question: dict[str, Any] = {"type": "choice", "criteria": dict(criteria)}
    if instructions is not None:
        question["instructions"] = instructions
    return question


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float

    @property
    def probability(self) -> float:
        return self.probabilities[self.choice]

    def ranked(self, limit: int | None = None) -> list[tuple[str, float]]:
        ranked = sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:limit] if limit else ranked


def validate_choice(raw: Any, ids: Collection[str], *, name: str = "answer") -> ChoiceAnswer:
    """Reject anything that is not a well-formed distribution over exactly the offered ids."""
    try:
        choice = raw["choice"]
        probabilities = {str(k): v for k, v in raw["probabilities"].items()}
        confidence = raw["confidence"]
        numbers = [*probabilities.values(), confidence]
        valid = (
            isinstance(choice, str)
            and choice in ids
            and set(probabilities) == {str(i) for i in ids}
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[choice] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, AttributeError, ValueError):
        valid = False
    if not valid:
        raise JevResponseError(f"Jev returned an invalid {name!r} answer; no action executed")
    return ChoiceAnswer(choice, {k: float(v) for k, v in probabilities.items()}, float(confidence))


@dataclass
class JevResponse:
    model: str
    answers: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def choice(self, name: str, ids: Collection[str]) -> ChoiceAnswer:
        if name not in self.answers:
            raise JevResponseError(f"Jev response has no answer for {name!r}; no action executed")
        return validate_choice(self.answers[name], ids, name=name)


class JevClient:
    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout_s: float = 10.0,
        max_retries: int = 2,
        http2: bool = True,
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key:
            raise JevAuthError("No TypeSafe API key. Set TYPESAFE_API_KEY (see README › Configuration).")
        self.endpoint = endpoint
        self.model = model
        self.max_retries = max(0, max_retries)
        if http2:
            try:
                import h2  # noqa: F401
            except ImportError:
                http2 = False
        self._http = httpx.Client(
            http2=http2,
            timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            transport=transport,
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=120),
        )

    @classmethod
    def from_settings(cls, settings: Any, *, transport: httpx.BaseTransport | None = None) -> JevClient:
        return cls(
            settings.api_key() or "",
            endpoint=settings.endpoint,
            model=settings.model,
            timeout_s=settings.timeout_s,
            max_retries=settings.max_retries,
            http2=settings.http2,
            transport=transport,
        )

    def evaluate(self, state: Mapping[str, Any], questions: Mapping[str, Mapping[str, Any]]) -> JevResponse:
        """One round trip: every question is answered against the same state. Safe to retry (no side effects)."""
        for name, question in questions.items():
            criteria = question.get("criteria") if isinstance(question, Mapping) else None
            if (
                not isinstance(question, Mapping)
                or question.get("type") != "choice"
                or not isinstance(criteria, Mapping)
                or not 2 <= len(criteria) <= MAX_CHOICES
            ):
                raise RouterContractError(
                    f"question {name!r} is not a discrete choice question; only typed one-of-N questions are sent"
                )
        body = {"model": self.model, "state": state, "questions": questions}
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                response = self._http.post(self.endpoint, json=body)
            except httpx.TimeoutException:
                if attempt < self.max_retries:
                    continue
                raise JevError("Jev request timed out; no action executed") from None
            except httpx.HTTPError as exc:
                if attempt < self.max_retries:
                    time.sleep(0.2 * 2**attempt)
                    continue
                raise JevError(f"Jev connection failed ({type(exc).__name__}); no action executed") from None
            if response.status_code in (401, 403):
                raise JevAuthError(
                    f"Jev rejected the API key (HTTP {response.status_code})", status=response.status_code
                )
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                time.sleep(_retry_delay(response, attempt))
                continue
            if response.is_error:
                raise JevError(
                    f"Jev returned HTTP {response.status_code}: {response.text[:200]}", status=response.status_code
                )
            try:
                data = response.json()
            except ValueError:
                raise JevResponseError("Jev returned non-JSON content; no action executed") from None
            if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
                raise JevResponseError("Jev response has no answers; no action executed")
            return JevResponse(
                model=str(data.get("model", self.model)),
                answers=data["answers"],
                usage=data.get("usage") or {},
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
            )
        raise JevError("Jev unavailable after retries; no action executed")

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> JevClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    try:
        if header is not None:
            return min(2.0, max(0.0, float(header)))
    except ValueError:
        pass
    return 0.25 * 2**attempt
