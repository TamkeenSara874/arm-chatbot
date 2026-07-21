"""Unit tests for the Groq Whisper speech-to-text client and its factory."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings
from src.services.stt.factory import create_stt_client
from src.services.stt.groq_whisper import GroqSTTClient


class TestGroqSTTClient:
    @pytest.mark.asyncio
    async def test_transcribe_returns_stripped_text(self) -> None:
        client = GroqSTTClient(api_key="test-key", model="whisper-large-v3-turbo")
        fake_response = MagicMock(text="  what do customers say about the food?  ")
        client.client.audio.transcriptions.create = AsyncMock(return_value=fake_response)

        result = await client.transcribe(b"fake-audio-bytes", "clip.webm")

        assert result == "what do customers say about the food?"

    @pytest.mark.asyncio
    async def test_transcribe_passes_model_and_file_to_groq(self) -> None:
        client = GroqSTTClient(api_key="test-key", model="whisper-large-v3-turbo")
        fake_response = MagicMock(text="hello")
        create_mock = AsyncMock(return_value=fake_response)
        client.client.audio.transcriptions.create = create_mock

        await client.transcribe(b"raw-bytes", "clip.webm")

        _, call_kwargs = create_mock.call_args
        assert call_kwargs["model"] == "whisper-large-v3-turbo"
        assert call_kwargs["file"] == ("clip.webm", b"raw-bytes")

    @pytest.mark.asyncio
    async def test_transcribe_propagates_failure_after_retries(self) -> None:
        # Nice-to-know it doesn't swallow the error -- the route layer
        # (test_voice_route.py) is what verifies this becomes a clean 502
        # instead of a raw exception reaching the client. Patches asyncio.sleep
        # so the real exponential backoff between retry attempts doesn't slow
        # the suite down.
        client = GroqSTTClient(api_key="test-key", model="whisper-large-v3-turbo")
        client.client.audio.transcriptions.create = AsyncMock(
            side_effect=RuntimeError("groq unavailable")
        )

        with patch("asyncio.sleep", new_callable=AsyncMock), pytest.raises(RuntimeError):
            await client.transcribe(b"raw-bytes", "clip.webm")


class TestCreateSTTClient:
    def test_groq_provider_returns_groq_client(self) -> None:
        settings = Settings(
            stt_provider="groq", groq_api_key="k", groq_stt_model="whisper-large-v3-turbo"
        )
        client = create_stt_client(settings)
        assert isinstance(client, GroqSTTClient)
        assert client.model == "whisper-large-v3-turbo"

    def test_unknown_provider_raises(self) -> None:
        settings = Settings(stt_provider="nonexistent")
        with pytest.raises(ValueError, match="Unsupported STT provider"):
            create_stt_client(settings)
