"""Unit tests for RedisCache, in particular query-scoped invalidation."""

from unittest.mock import AsyncMock, patch

import pytest

from src.services.cache import RedisCache


@pytest.fixture
def cache() -> RedisCache:
    with patch("src.services.cache.aioredis.from_url"):
        return RedisCache(url="redis://localhost:6379", ttl_seconds=3600)


class TestInvalidateQuery:
    async def test_deletes_the_exact_query_key(self, cache: RedisCache) -> None:
        cache.client.delete = AsyncMock(return_value=1)
        result = await cache.invalidate_query(1, "What do people say about the pasta?")
        assert result is True
        expected_key = cache._key(1, "What do people say about the pasta?")
        cache.client.delete.assert_awaited_once_with(expected_key)

    async def test_returns_false_when_no_key_existed(self, cache: RedisCache) -> None:
        cache.client.delete = AsyncMock(return_value=0)
        result = await cache.invalidate_query(1, "unseen query")
        assert result is False

    async def test_swallows_redis_errors(self, cache: RedisCache) -> None:
        cache.client.delete = AsyncMock(side_effect=ConnectionError("redis down"))
        result = await cache.invalidate_query(1, "some query")
        assert result is False

    async def test_key_matches_the_one_set_get_uses(self, cache: RedisCache) -> None:
        """A correction on a query must invalidate the same key .get()/.set() use for it."""
        query = "How many positive reviews do I have?"
        assert cache._key(1, query) == cache._key(1, query.upper())


class TestGetSetJson:
    async def test_get_json_returns_parsed_value(self, cache: RedisCache) -> None:
        cache.client.get = AsyncMock(return_value='{"detected": true}')
        result = await cache.get_json("anomaly:1")
        assert result == {"detected": True}

    async def test_get_json_returns_none_when_missing(self, cache: RedisCache) -> None:
        cache.client.get = AsyncMock(return_value=None)
        result = await cache.get_json("anomaly:1")
        assert result is None

    async def test_get_json_swallows_redis_errors(self, cache: RedisCache) -> None:
        cache.client.get = AsyncMock(side_effect=ConnectionError("redis down"))
        result = await cache.get_json("anomaly:1")
        assert result is None

    async def test_set_json_stores_with_explicit_ttl(self, cache: RedisCache) -> None:
        cache.client.setex = AsyncMock()
        await cache.set_json("anomaly:1", {"detected": False}, 3600)
        cache.client.setex.assert_awaited_once_with("anomaly:1", 3600, '{"detected": false}')

    async def test_set_json_swallows_redis_errors(self, cache: RedisCache) -> None:
        cache.client.setex = AsyncMock(side_effect=ConnectionError("redis down"))
        await cache.set_json("anomaly:1", {"detected": False}, 3600)
