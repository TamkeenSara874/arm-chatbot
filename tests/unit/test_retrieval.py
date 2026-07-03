"""Unit tests for hybrid_retrieve -- mocked vector store and sparse embedder."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.vector.base import SearchResult


def _sr(chunk_id: str, score: float, restaurant_id: int = 1) -> SearchResult:
    return SearchResult(
        id=chunk_id,
        score=score,
        payload={"text": "review text", "rating": 4.0, "restaurant_id": restaurant_id},
    )


def _mock_embedder(vector: list[float] | None = None) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_one = AsyncMock(return_value=vector or [0.1] * 3072)
    return embedder


def _sparse_patch(indices: list[int] | None = None, values: list[float] | None = None):
    from src.services.embedding.sparse_embedder import SparseVector

    sv = SparseVector(indices=indices or [0, 1], values=values or [0.5, 0.5])
    return patch("src.core.retrieval.compute_sparse_vector", AsyncMock(return_value=sv))


class TestHybridRetrieve:
    @pytest.mark.asyncio
    async def test_passes_restaurant_id_filter(self) -> None:
        from src.core.retrieval import hybrid_retrieve

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=[])

        with _sparse_patch():
            await hybrid_retrieve(
                query="food quality",
                restaurant_id=7,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
            )

        _, kw = vector_store.hybrid_search.call_args
        assert kw["filters"]["restaurant_id"] == 7

    @pytest.mark.asyncio
    async def test_passes_dense_and_sparse_vectors(self) -> None:
        from src.core.retrieval import hybrid_retrieve

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=[])
        dense = [0.42] * 3072

        with _sparse_patch(indices=[5, 10], values=[0.8, 0.2]):
            await hybrid_retrieve(
                query="biryani",
                restaurant_id=1,
                embedder=_mock_embedder(dense),
                vector_store=vector_store,
                collection="review_chunks",
            )

        _, kw = vector_store.hybrid_search.call_args
        assert kw["dense_vector"] == dense
        assert kw["sparse_indices"] == [5, 10]
        assert kw["sparse_values"] == [0.8, 0.2]

    @pytest.mark.asyncio
    async def test_respects_top_k(self) -> None:
        from src.core.retrieval import hybrid_retrieve

        chunks = [_sr(f"c{i}", 1.0 / (i + 1)) for i in range(20)]
        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=chunks)

        with _sparse_patch():
            results = await hybrid_retrieve(
                query="service",
                restaurant_id=1,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
                top_k=4,
            )

        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_passes_date_and_rating_filters(self) -> None:
        from src.core.retrieval import hybrid_retrieve

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=[])

        with _sparse_patch():
            await hybrid_retrieve(
                query="recent reviews",
                restaurant_id=1,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
                date_from=1700000000.0,
                date_to=1800000000.0,
                rating_min=3.0,
                rating_max=5.0,
            )

        _, kw = vector_store.hybrid_search.call_args
        f = kw["filters"]
        assert f["date_from"] == 1700000000.0
        assert f["date_to"] == 1800000000.0
        assert f["rating_min"] == 3.0
        assert f["rating_max"] == 5.0

    @pytest.mark.asyncio
    async def test_returns_empty_on_hybrid_search_failure(self) -> None:
        from src.core.retrieval import hybrid_retrieve

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(side_effect=RuntimeError("Qdrant down"))

        with _sparse_patch():
            results = await hybrid_retrieve(
                query="anything",
                restaurant_id=1,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
            )

        assert results == []

    @pytest.mark.asyncio
    async def test_reranker_candidate_pool_capped_regardless_of_top_k(self) -> None:
        """Aggregation queries (top_k=20) must not balloon the rerank candidate pool.

        Reranking is CPU-bound cross-encoder scoring; sending it 80 candidates
        (top_k*4 uncapped) instead of a capped pool was the dominant cost in a
        live reproduction that took 52s end-to-end for one query.
        """
        from src.core.retrieval import hybrid_retrieve

        chunks = [_sr(f"c{i}", 1.0 / (i + 1)) for i in range(100)]
        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=chunks)

        with (
            _sparse_patch(),
            patch("src.core.reranker.rerank", AsyncMock(return_value=chunks[:20])) as mock_rerank,
        ):
            await hybrid_retrieve(
                query="what should we improve",
                restaurant_id=1,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
                top_k=20,
                reranker_model="BAAI/bge-reranker-base",
            )

        candidates_passed = mock_rerank.call_args.args[1]
        assert len(candidates_passed) <= 30

    @pytest.mark.asyncio
    async def test_returns_empty_when_embedding_fails(self) -> None:
        from src.core.retrieval import hybrid_retrieve

        embedder = MagicMock()
        embedder.embed_one = AsyncMock(side_effect=RuntimeError("OpenAI down"))

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=[])

        with _sparse_patch():
            results = await hybrid_retrieve(
                query="anything",
                restaurant_id=1,
                embedder=embedder,
                vector_store=vector_store,
                collection="review_chunks",
            )

        assert results == []
        assert not vector_store.hybrid_search.called
