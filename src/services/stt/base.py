from abc import ABC, abstractmethod


class BaseSTTClient(ABC):
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        """Transcribe an audio clip to text."""
