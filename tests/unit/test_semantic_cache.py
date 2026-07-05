"""Unit tests for the semantic cache tier (find/store/invalidate) — no I/O required."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.semantic_cache import (
    find_cached_response,
    invalidate_cached_response,
    store_cached_response,
)

COLLECTION = "chat_cache"


def _search_result(query: str) -> MagicMock:
    result = MagicMock()
    result.payload = {"query": query}
    return result


class TestFindCachedResponse:
    @pytest.mark.asyncio
    async def test_returns_none_when_no_semantic_match(self) -> None:
        embedder = MagicMock(embed_one=AsyncMock(return_value=[0.1, 0.2]))
        vector_store = MagicMock(search=AsyncMock(return_value=[]))
        redis_cache = MagicMock(get=AsyncMock())

        value, vector = await find_cached_response(
            "how is the food?", 1, embedder, vector_store, redis_cache, COLLECTION
        )

        assert value is None
        assert vector == [0.1, 0.2]
        redis_cache.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_value_from_redis_on_semantic_match(self) -> None:
        embedder = MagicMock(embed_one=AsyncMock(return_value=[0.1, 0.2]))
        vector_store = MagicMock(search=AsyncMock(return_value=[_search_result("how's the food?")]))
        redis_cache = MagicMock(get=AsyncMock(return_value={"answer": "cached"}))

        value, _ = await find_cached_response(
            "how is the food?", 1, embedder, vector_store, redis_cache, COLLECTION
        )

        assert value == {"answer": "cached"}
        redis_cache.get.assert_called_once_with(1, "how's the food?")

    @pytest.mark.asyncio
    async def test_returns_none_when_embedding_fails(self) -> None:
        embedder = MagicMock(embed_one=AsyncMock(side_effect=RuntimeError("embed down")))
        vector_store = MagicMock(search=AsyncMock())
        redis_cache = MagicMock(get=AsyncMock())

        value, vector = await find_cached_response(
            "how is the food?", 1, embedder, vector_store, redis_cache, COLLECTION
        )

        assert value is None
        assert vector == []
        vector_store.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_none_when_search_fails(self) -> None:
        embedder = MagicMock(embed_one=AsyncMock(return_value=[0.1]))
        vector_store = MagicMock(search=AsyncMock(side_effect=RuntimeError("qdrant down")))
        redis_cache = MagicMock(get=AsyncMock())

        value, vector = await find_cached_response(
            "how is the food?", 1, embedder, vector_store, redis_cache, COLLECTION
        )

        assert value is None
        assert vector == [0.1]


class TestStoreCachedResponse:
    @pytest.mark.asyncio
    async def test_writes_redis_and_qdrant_point(self) -> None:
        embedder = MagicMock(embed_one=AsyncMock(return_value=[0.5, 0.6]))
        vector_store = MagicMock(upsert=AsyncMock())
        redis_cache = MagicMock(set=AsyncMock())

        await store_cached_response(
            "how is the food?", 1, {"answer": "ok"}, embedder, vector_store, redis_cache, COLLECTION
        )

        redis_cache.set.assert_called_once_with(1, "how is the food?", {"answer": "ok"})
        vector_store.upsert.assert_called_once()
        args, _ = vector_store.upsert.call_args
        assert args[0] == COLLECTION
        assert args[1][0]["payload"] == {"restaurant_id": 1, "query": "how is the food?"}

    @pytest.mark.asyncio
    async def test_reuses_precomputed_vector_without_reembedding(self) -> None:
        embedder = MagicMock(embed_one=AsyncMock())
        vector_store = MagicMock(upsert=AsyncMock())
        redis_cache = MagicMock(set=AsyncMock())

        await store_cached_response(
            "how is the food?",
            1,
            {"answer": "ok"},
            embedder,
            vector_store,
            redis_cache,
            COLLECTION,
            precomputed_vector=[0.9],
        )

        embedder.embed_one.assert_not_called()

    @pytest.mark.asyncio
    async def test_swallows_qdrant_failure_after_redis_write(self) -> None:
        embedder = MagicMock(embed_one=AsyncMock(return_value=[0.1]))
        vector_store = MagicMock(upsert=AsyncMock(side_effect=RuntimeError("qdrant down")))
        redis_cache = MagicMock(set=AsyncMock())

        await store_cached_response(
            "how is the food?", 1, {"answer": "ok"}, embedder, vector_store, redis_cache, COLLECTION
        )

        redis_cache.set.assert_called_once()


class TestInvalidateCachedResponse:
    @pytest.mark.asyncio
    async def test_invalidates_redis_key_and_deletes_qdrant_point_by_deterministic_id(self) -> None:
        vector_store = MagicMock(delete=AsyncMock())
        redis_cache = MagicMock(invalidate_query=AsyncMock(return_value=True))

        await invalidate_cached_response("how is the food?", 1, vector_store, redis_cache, COLLECTION)

        redis_cache.invalidate_query.assert_called_once_with(1, "how is the food?")
        vector_store.delete.assert_called_once()
        args, _ = vector_store.delete.call_args
        assert args[0] == COLLECTION
        assert len(args[1]) == 1

    @pytest.mark.asyncio
    async def test_same_query_and_restaurant_produce_same_point_id_as_store(self) -> None:
        """Regression guard: invalidation must target the exact point store_cached_response wrote."""
        vector_store_store = MagicMock(upsert=AsyncMock())
        redis_cache = MagicMock(set=AsyncMock(), invalidate_query=AsyncMock())
        embedder = MagicMock(embed_one=AsyncMock(return_value=[0.1]))

        await store_cached_response(
            "how is the food?", 1, {"answer": "ok"}, embedder, vector_store_store, redis_cache, COLLECTION
        )
        stored_point_id = vector_store_store.upsert.call_args[0][1][0]["id"]

        vector_store_delete = MagicMock(delete=AsyncMock())
        await invalidate_cached_response(
            "how is the food?", 1, vector_store_delete, redis_cache, COLLECTION
        )
        deleted_id = vector_store_delete.delete.call_args[0][1][0]

        assert deleted_id == stored_point_id

    @pytest.mark.asyncio
    async def test_swallows_qdrant_delete_failure(self) -> None:
        vector_store = MagicMock(delete=AsyncMock(side_effect=RuntimeError("qdrant down")))
        redis_cache = MagicMock(invalidate_query=AsyncMock(return_value=True))

        await invalidate_cached_response("how is the food?", 1, vector_store, redis_cache, COLLECTION)

        redis_cache.invalidate_query.assert_called_once()
