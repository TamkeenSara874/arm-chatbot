from __future__ import annotations

import time

import structlog
from groq import AsyncGroq

from src.services.stt.base import BaseSTTClient
from src.utils.circuit_breaker import groq_stt_breaker
from src.utils.metrics import llm_request_latency, llm_request_total
from src.utils.retry import fetch_with_retry

logger = structlog.get_logger()


class GroqSTTClient(BaseSTTClient):
    """Groq-hosted Whisper transcription. Free up to 2,000 requests/day as of
    writing -- see docs/evaluation_report.md style cost notes elsewhere in
    this project for the reasoning behind picking Groq over OpenAI's Whisper
    endpoint (same tradeoff already made for decomposition: same-quality
    result, no cost, and one fewer provider's key to manage).
    """

    def __init__(self, api_key: str, model: str) -> None:
        self.client = AsyncGroq(api_key=api_key)
        self.model = model

    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        async def _call() -> str:
            response = await self.client.audio.transcriptions.create(
                model=self.model,
                file=(filename, audio_bytes),
            )
            return response.text

        start = time.perf_counter()
        try:
            result = await fetch_with_retry(
                lambda: groq_stt_breaker.call_async(_call), label="groq.transcribe"
            )
            llm_request_total.labels(provider="groq", model=self.model, intent="stt").inc()
            return result.strip()
        finally:
            llm_request_latency.labels(provider="groq", model=self.model).observe(
                time.perf_counter() - start
            )
