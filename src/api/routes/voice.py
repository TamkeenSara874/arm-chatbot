"""Voice-mode dictation: audio in, transcribed text out."""

# NOTE: deliberately no `from __future__ import annotations` -- this file has
# an UploadFile parameter combined with @limiter.limit(), which crashes
# FastAPI's dependant analysis under postponed evaluation. Same issue and
# fix as src/api/routes/ingest.py.

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status

from src.api.dependencies import RestaurantId, get_stt_client
from src.api.rate_limit import limiter
from src.config import get_settings
from src.models.schemas import VoiceTranscribeResponse
from src.services.stt.base import BaseSTTClient
from src.utils.security import check_audio_upload

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])

settings = get_settings()

STTClient = Annotated[BaseSTTClient, Depends(get_stt_client)]


@router.post("/transcribe", response_model=VoiceTranscribeResponse)
@limiter.limit(settings.rate_limit_voice)
async def transcribe(
    request: Request,
    file: UploadFile,
    restaurant_id: RestaurantId,
    stt_client: STTClient,
) -> VoiceTranscribeResponse:
    audio_bytes = await file.read()

    try:
        check_audio_upload(content_type=file.content_type or "", size_bytes=len(audio_bytes))
    except ValueError as exc:
        status_code = (
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            if "too large" in str(exc)
            else status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    try:
        text = await stt_client.transcribe(audio_bytes, file.filename or "audio.webm")
    except Exception as exc:
        logger.error("voice_transcription_failed", restaurant_id=restaurant_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Transcription failed. Please try again or type your question.",
        ) from exc

    return VoiceTranscribeResponse(text=text)
