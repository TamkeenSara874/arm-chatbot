"""Thin HTTP client for driving the live API in the LLM-as-judge eval harness.

Not a pytest module itself -- imported by test_eval_fixture.py. Kept separate
from src/ because it duplicates just enough of the frontend's SSE handling
(see frontend/src/hooks/useSSE.ts) to drive the same /chat/query contract
from Python, and has no reason to ship in the application image.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx

BASE_URL = os.environ.get("EVAL_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("EVAL_API_KEY", "change-me-local-dev-key")


@dataclass
class ChatTurnResult:
    answer: str
    evidence: list[dict[str, Any]]
    confidence: float
    caveats: str | None
    entity_counts: dict[str, int]
    source_breakdown: dict[str, int]
    complexity: str
    model_used: str
    cached: bool
    latency_ms: int
    cost_usd: float
    session_id: str
    message_id: str


async def get_jwt(client: httpx.AsyncClient, restaurant_id: int) -> str:
    resp = await client.post(
        "/api/v1/auth/token",
        json={"restaurant_id": restaurant_id},
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    resp.raise_for_status()
    return str(resp.json()["access_token"])


async def create_session(client: httpx.AsyncClient, jwt: str, restaurant_id: int) -> str:
    resp = await client.post(
        "/api/v1/chat/sessions",
        json={"restaurant_id": restaurant_id},
        headers={"Authorization": f"Bearer {jwt}"},
    )
    resp.raise_for_status()
    return str(resp.json()["session_id"])


def _parse_sse(raw: str) -> list[tuple[str, str]]:
    """Parse a raw SSE text stream into (event, data) pairs.

    Mirrors the subset of the SSE spec sse-starlette emits: 'event: <name>'
    and 'data: <payload>' lines, terminated by a blank line per message.
    """
    events: list[tuple[str, str]] = []
    event_name = "message"
    data_lines: list[str] = []
    for line in raw.splitlines():
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line == "" and data_lines:
            events.append((event_name, "\n".join(data_lines)))
            event_name, data_lines = "message", []
    if data_lines:
        events.append((event_name, "\n".join(data_lines)))
    return events


async def send_query(
    client: httpx.AsyncClient,
    jwt: str,
    session_id: str,
    restaurant_id: int,
    message: str,
) -> ChatTurnResult:
    async with client.stream(
        "POST",
        "/api/v1/chat/query",
        json={"session_id": session_id, "restaurant_id": restaurant_id, "message": message},
        headers={"Authorization": f"Bearer {jwt}", "Accept": "text/event-stream"},
        timeout=60.0,
    ) as resp:
        resp.raise_for_status()
        raw = await resp.aread()

    events = _parse_sse(raw.decode("utf-8"))
    done_payload: dict[str, Any] | None = None
    for event, data in events:
        if event == "done":
            done_payload = json.loads(data)
        elif event == "error":
            raise RuntimeError(f"chat query errored: {json.loads(data)}")

    if done_payload is None:
        raise RuntimeError("no 'done' event received from /chat/query stream")

    response = done_payload["response"]
    return ChatTurnResult(
        answer=response["answer"],
        evidence=response["evidence"],
        confidence=response["confidence"],
        caveats=response.get("caveats"),
        entity_counts=response.get("entity_counts", {}),
        source_breakdown=response.get("source_breakdown", {}),
        complexity=done_payload["complexity"],
        model_used=done_payload["model_used"],
        cached=done_payload["cached"],
        latency_ms=done_payload.get("latency_ms", 0),
        cost_usd=done_payload.get("cost_usd", 0.0),
        session_id=done_payload["session_id"],
        message_id=done_payload["message_id"],
    )


async def submit_correction(
    client: httpx.AsyncClient,
    jwt: str,
    session_id: str,
    message_id: str,
    corrected_response: str,
) -> dict[str, Any]:
    resp = await client.post(
        "/api/v1/chat/correct",
        json={
            "session_id": session_id,
            "message_id": message_id,
            "corrected_response": corrected_response,
        },
        headers={"Authorization": f"Bearer {jwt}"},
    )
    resp.raise_for_status()
    return dict(resp.json())


async def get_report(
    client: httpx.AsyncClient,
    jwt: str,
    session_id: str,
    restaurant_id: int,
    message: str,
) -> dict[str, Any]:
    resp = await client.post(
        "/api/v1/chat/report",
        json={"session_id": session_id, "restaurant_id": restaurant_id, "message": message},
        headers={"Authorization": f"Bearer {jwt}"},
        timeout=60.0,
    )
    resp.raise_for_status()
    return dict(resp.json())


JUDGE_SYSTEM_PROMPT = (
    "You are a strict QA judge for a restaurant-review RAG chatbot. You are given a user "
    "question, the chatbot's answer, the evidence it retrieved, and a list of assertions the "
    "answer must satisfy. For each assertion, decide pass or fail and give a one-sentence reason. "
    "Be skeptical -- an assertion only passes if the answer clearly and unambiguously satisfies it. "
    'Respond with JSON only: {"results": [{"assertion": str, "pass": bool, "reason": str}]}'
)


async def judge_assertions(
    openai_client: Any,
    question: str,
    answer: str,
    evidence: list[dict[str, Any]],
    assertions: list[str],
    model: str = "gpt-4o-mini",
) -> list[dict[str, Any]]:
    """Grade an answer against a list of natural-language assertions via LLM-as-judge."""
    user_prompt = (
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        f"Evidence retrieved (may be empty if none was found): {json.dumps(evidence)[:3000]}\n\n"
        "Assertions to check:\n" + "\n".join(f"- {a}" for a in assertions)
    )
    resp = await openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=600,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    results = parsed.get("results", [])
    return list(results) if isinstance(results, list) else []
