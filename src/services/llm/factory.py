from __future__ import annotations

from src.config import Settings
from src.services.llm.anthropic_client import AnthropicClient
from src.services.llm.base import BaseLLMClient, FallbackLLMClient
from src.services.llm.groq_client import GroqClient
from src.services.llm.openai_client import OpenAIClient


def create_decomposition_client(settings: Settings) -> BaseLLMClient:
    """Groq primary, GPT-4o-mini fallback — used for query decomposition."""
    return FallbackLLMClient([
        GroqClient(api_key=settings.groq_api_key, model=settings.groq_decomp_model),
        OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_simple_model),
    ])


def create_simple_client(settings: Settings) -> BaseLLMClient:
    """GPT-4o-mini — used for simple query generation."""
    return OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_simple_model,
    )


def create_complex_client(settings: Settings) -> BaseLLMClient:
    """GPT-4o primary, Anthropic fallback, GPT-4o-mini safety net — complex generation."""
    return FallbackLLMClient([
        OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_complex_model),
        AnthropicClient(api_key=settings.anthropic_api_key, model=settings.anthropic_fallback_model),
        OpenAIClient(api_key=settings.openai_api_key, model=settings.openai_simple_model),
    ])


def create_summary_client(settings: Settings) -> BaseLLMClient:
    """GPT-4o-mini — used for session summarization (compression task)."""
    return OpenAIClient(
        api_key=settings.openai_api_key,
        model=settings.openai_simple_model,
    )
