"""Verify cross-restaurant data isolation at the retrieval layer."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.vector.base import SearchResult


def _mock_embedder() -> MagicMock:
    embedder = MagicMock()
    embedder.embed_one = AsyncMock(return_value=[0.1] * 3072)
    return embedder


def _sparse_patch():
    """Patch compute_sparse_vector so tests need no fastembed install."""
    from src.services.embedding.sparse_embedder import SparseVector

    mock = AsyncMock(return_value=SparseVector(indices=[0, 1], values=[0.5, 0.5]))
    return patch("src.core.retrieval.compute_sparse_vector", mock)


class TestMultitenantIsolation:
    @pytest.mark.asyncio
    async def test_vector_store_receives_correct_restaurant_filter(self) -> None:
        """Hybrid search must always be scoped to the requested restaurant_id."""
        from src.core.retrieval import hybrid_retrieve

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=[])

        with _sparse_patch():
            await hybrid_retrieve(
                query="best dish",
                restaurant_id=42,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
            )

        assert vector_store.hybrid_search.called
        _, call_kwargs = vector_store.hybrid_search.call_args
        filters = call_kwargs.get("filters", {})
        assert filters.get("restaurant_id") == 42, (
            "Vector store search must filter by restaurant_id; "
            "a missing filter would expose data across tenants"
        )

    @pytest.mark.asyncio
    async def test_different_restaurant_ids_produce_separate_filter_calls(self) -> None:
        """Two queries for different restaurants must each pass their own restaurant_id."""
        from src.core.retrieval import hybrid_retrieve

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=[])

        with _sparse_patch():
            for restaurant_id in [1, 2]:
                await hybrid_retrieve(
                    query="food quality",
                    restaurant_id=restaurant_id,
                    embedder=_mock_embedder(),
                    vector_store=vector_store,
                    collection="review_chunks",
                )

        calls = vector_store.hybrid_search.call_args_list
        assert len(calls) == 2
        seen_ids = {c[1].get("filters", {}).get("restaurant_id") for c in calls}
        assert seen_ids == {1, 2}, "Each restaurant must have its own isolated search call"

    @pytest.mark.asyncio
    async def test_results_carry_no_other_restaurant_data(self) -> None:
        """If the vector store correctly enforces filters, results are always single-tenant."""
        from src.core.retrieval import hybrid_retrieve

        restaurant_1_chunks = [
            SearchResult(id="r1c1", score=0.9, payload={"text": "food", "restaurant_id": 1}),
            SearchResult(id="r1c2", score=0.8, payload={"text": "service", "restaurant_id": 1}),
        ]
        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=restaurant_1_chunks)

        with _sparse_patch():
            results = await hybrid_retrieve(
                query="food quality",
                restaurant_id=1,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
            )

        for result in results:
            rid = result.payload.get("restaurant_id")
            if rid is not None:
                assert rid == 1, f"Found result from restaurant {rid} when querying restaurant 1"
