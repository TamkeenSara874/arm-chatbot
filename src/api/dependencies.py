from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Settings, get_settings
from src.services.cache import RedisCache
from src.services.database import get_db
from src.services.embedding.base import BaseEmbedder
from src.services.embedding.factory import create_embedder
from src.services.llm.base import BaseLLMClient
from src.services.llm.factory import (
    create_complex_client,
    create_decomposition_client,
    create_simple_client,
    create_summary_client,
)
from src.services.vector.base import BaseVectorStore
from src.services.vector.factory import create_vector_store

DbSession = Annotated[AsyncSession, Depends(get_db)]

_bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Security(_bearer)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> str:
    if credentials is None or credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


AuthToken = Annotated[str, Depends(require_api_key)]


@lru_cache(maxsize=1)
def _decomp_client_singleton() -> BaseLLMClient:
    return create_decomposition_client(get_settings())


@lru_cache(maxsize=1)
def _simple_client_singleton() -> BaseLLMClient:
    return create_simple_client(get_settings())


@lru_cache(maxsize=1)
def _complex_client_singleton() -> BaseLLMClient:
    return create_complex_client(get_settings())


@lru_cache(maxsize=1)
def _summary_client_singleton() -> BaseLLMClient:
    return create_summary_client(get_settings())


@lru_cache(maxsize=1)
def _embedder_singleton() -> BaseEmbedder:
    return create_embedder(get_settings())


@lru_cache(maxsize=1)
def _vector_store_singleton() -> BaseVectorStore:
    return create_vector_store(get_settings())


@lru_cache(maxsize=1)
def _cache_singleton() -> RedisCache:
    s = get_settings()
    return RedisCache(url=s.redis_url, ttl_seconds=s.cache_ttl_seconds)


@lru_cache(maxsize=1)
def _openai_singleton() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=get_settings().openai_api_key)


def get_decomp_client() -> BaseLLMClient:
    return _decomp_client_singleton()


def get_simple_client() -> BaseLLMClient:
    return _simple_client_singleton()


def get_complex_client() -> BaseLLMClient:
    return _complex_client_singleton()


def get_summary_client() -> BaseLLMClient:
    return _summary_client_singleton()


def get_embedder() -> BaseEmbedder:
    return _embedder_singleton()


def get_vector_store() -> BaseVectorStore:
    return _vector_store_singleton()


def get_cache() -> RedisCache:
    return _cache_singleton()


def get_openai_client() -> AsyncOpenAI:
    return _openai_singleton()
