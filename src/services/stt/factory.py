from src.config import Settings
from src.services.stt.base import BaseSTTClient
from src.services.stt.groq_whisper import GroqSTTClient


def create_stt_client(settings: Settings) -> BaseSTTClient:
    if settings.stt_provider == "groq":
        return GroqSTTClient(api_key=settings.groq_api_key, model=settings.groq_stt_model)
    raise ValueError(f"Unsupported STT provider: {settings.stt_provider}")
