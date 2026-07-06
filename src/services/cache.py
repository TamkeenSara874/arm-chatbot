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
        digest = hashlib.sha256(query.strip().lower().encode()).hexdigest()
        return f"chat:response:{restaurant_id}:{digest}"

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

    async def invalidate_query(self, restaurant_id: int, query: str) -> bool:
        """Delete the cached response for one exact query, if present.

        Called after a correction is submitted for that query -- otherwise a
        cache hit on the identical query text would keep serving the
        pre-correction answer for the rest of the TTL, silently ignoring the
        correction the user just made.
        """
        key = self._key(restaurant_id, query)
        try:
            deleted = await self.client.delete(key)
            return bool(deleted)
        except Exception as exc:
            logger.warning("cache_invalidate_query_failed", key=key, error=str(exc))
            return False

    async def invalidate_restaurant(self, restaurant_id: int) -> int:
        """Delete all cached responses for a restaurant. Returns the number of keys deleted."""
        pattern = f"chat:response:{restaurant_id}:*"
        try:
            keys: list[str] = []
            async for key in self.client.scan_iter(match=pattern, count=100):
                keys.append(key)
            if keys:
                deleted = await self.client.delete(*keys)
                logger.info("cache_invalidated", restaurant_id=restaurant_id, keys_deleted=deleted)
                return deleted
        except Exception as exc:
            logger.warning("cache_invalidate_failed", restaurant_id=restaurant_id, error=str(exc))
        return 0

    async def get_json(self, key: str) -> Any | None:
        """Generic get for an arbitrary (non query-scoped) key, e.g. anomaly detection results."""
        try:
            value = await self.client.get(key)
            return json.loads(value) if value else None
        except Exception as exc:
            logger.warning("cache_get_json_failed", key=key, error=str(exc))
            return None

    async def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Generic set for an arbitrary (non query-scoped) key with an explicit TTL."""
        try:
            await self.client.setex(key, ttl_seconds, json.dumps(value))
        except Exception as exc:
            logger.warning("cache_set_json_failed", key=key, error=str(exc))

    async def close(self) -> None:
        await self.client.aclose()
