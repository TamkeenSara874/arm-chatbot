from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from src.utils.metrics import cache_hit_total

logger = structlog.get_logger()


class RedisCache:
    def __init__(self, url: str, ttl_seconds: int = 3600) -> None:
        self.client: aioredis.Redis = aioredis.from_url(url, decode_responses=True)
        self.ttl = ttl_seconds

    def _key(self, restaurant_id: int, query: str) -> str:
        raw = f"{restaurant_id}:{query.strip().lower()}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return f"chat:response:{digest}"

    async def get(self, restaurant_id: int, query: str) -> dict[str, Any] | None:
        if self.ttl <= 0:
            return None
        key = self._key(restaurant_id, query)
        try:
            value = await self.client.get(key)
            if value:
                cache_hit_total.labels(result="hit").inc()
                return json.loads(value)
            cache_hit_total.labels(result="miss").inc()
            return None
        except Exception as exc:
            logger.warning("cache_get_failed", key=key, error=str(exc))
            return None

    async def set(self, restaurant_id: int, query: str, value: dict[str, Any]) -> None:
        if self.ttl <= 0:
            return
        key = self._key(restaurant_id, query)
        try:
            await self.client.setex(key, self.ttl, json.dumps(value))
        except Exception as exc:
            logger.warning("cache_set_failed", key=key, error=str(exc))

    async def close(self) -> None:
        await self.client.aclose()
