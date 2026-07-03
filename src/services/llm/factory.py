from __future__ import annotations

from src.config import Settings
from src.services.llm.base import BaseLLMClient, FallbackLLMClient
from src.services.llm.groq_client import GroqClient
from src.services.llm.groq_rotation import RotatingGroqClient
from src.services.llm.openai_client import OpenAIClient


def create_decomposition_client(settings: Settings) -> BaseLLMClient:
    """Groq primary (free tier, fast), GPT-4o-mini fallback if all Groq keys are rate-limited.

    With 2+ keys configured (groq_api_key + groq_api_keys), rotates across
    them on a 429 instead of immediately falling back to paid OpenAI --
    each free-tier key has its own daily quota, so N keys give ~N x the
    free daily decomposition budget before the paid fallback is ever needed.
    """
    groq_keys = settings.groq_api_key_list
    groq_client: BaseLLMClient
    if len(groq_keys) > 1:
        groq_client = RotatingGroqClient(api_keys=groq_keys, model=settings.groq_decomp_model)
    else:
        groq_client = GroqClient(
            api_key=groq_keys[0] if groq_keys else settings.groq_api_key,
            model=settings.groq_decomp_model,
        )
    return FallbackLLMClient(
        [
            groq_client,
            OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_simple_model),
        ]
    )


def create_simple_client(settings: Settings) -> BaseLLMClient:
    """GPT-4o-mini for simple query generation."""
    return OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_simple_model,
    )


def create_complex_client(settings: Settings) -> BaseLLMClient:
    """GPT-4o primary, GPT-4o-mini safety net for complex generation."""
    return FallbackLLMClient(
        [
            OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_complex_model),
            OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_simple_model),
        ]
    )


def create_summary_client(settings: Settings) -> BaseLLMClient:
    """GPT-4o-mini for session summarization."""
    return OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_simple_model,
    )
