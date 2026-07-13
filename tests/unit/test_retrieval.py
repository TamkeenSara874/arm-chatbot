"""Unit tests for hybrid_retrieve -- mocked vector store and sparse embedder."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.retrieval import RetrievalTiming, build_retrieval_params
from src.models.schemas import DateFilter, DecomposedQuery, RatingFilter
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

    @pytest.mark.asyncio
    async def test_observes_retrieval_latency_metric(self) -> None:
        """Regression test: retrieval_latency was declared in metrics.py but
        never actually observed anywhere -- a dead metric."""
        from src.core.retrieval import hybrid_retrieve

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=[_sr("c1", 0.9)])

        with _sparse_patch(), patch("src.core.retrieval.retrieval_latency") as mock_hist:
            await hybrid_retrieve(
                query="food quality",
                restaurant_id=1,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
            )

        mock_hist.observe.assert_called_once()
        (observed_seconds,) = mock_hist.observe.call_args.args
        assert observed_seconds >= 0


class TestRetrievalTiming:
    @pytest.mark.asyncio
    async def test_timing_populated_when_no_reranker(self) -> None:
        from src.core.retrieval import hybrid_retrieve

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=[_sr("a", 0.5)])
        timing = RetrievalTiming()

        with _sparse_patch():
            await hybrid_retrieve(
                query="food",
                restaurant_id=1,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
                timing=timing,
            )

        assert timing.embed_ms >= 0.0
        assert timing.search_ms >= 0.0
        assert timing.rerank_ms == 0.0

    @pytest.mark.asyncio
    async def test_timing_populated_on_empty_results(self) -> None:
        from src.core.retrieval import hybrid_retrieve

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=[])
        timing = RetrievalTiming()

        with _sparse_patch():
            await hybrid_retrieve(
                query="food",
                restaurant_id=1,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
                timing=timing,
            )

        assert timing.embed_ms >= 0.0
        assert timing.search_ms >= 0.0
        assert timing.rerank_ms == 0.0

    @pytest.mark.asyncio
    async def test_timing_includes_rerank_ms_when_reranked(self) -> None:
        from src.core.retrieval import hybrid_retrieve

        chunks = [_sr(f"c{i}", 1.0 / (i + 1)) for i in range(10)]
        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=chunks)
        timing = RetrievalTiming()

        with (
            _sparse_patch(),
            patch("src.core.reranker.rerank", AsyncMock(return_value=chunks[:6])),
        ):
            await hybrid_retrieve(
                query="food",
                restaurant_id=1,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
                reranker_model="mock-model",
                timing=timing,
            )

        assert timing.rerank_ms >= 0.0

    @pytest.mark.asyncio
    async def test_no_timing_arg_does_not_error(self) -> None:
        """timing is optional -- existing callers that don't pass it must be unaffected."""
        from src.core.retrieval import hybrid_retrieve

        vector_store = MagicMock()
        vector_store.hybrid_search = AsyncMock(return_value=[_sr("a", 0.5)])

        with _sparse_patch():
            results = await hybrid_retrieve(
                query="food",
                restaurant_id=1,
                embedder=_mock_embedder(),
                vector_store=vector_store,
                collection="review_chunks",
            )

        assert len(results) == 1


class TestBuildRetrievalParams:
    def test_non_aggregation_uses_top_k_6(self) -> None:
        decomposed = DecomposedQuery(intent="factual", needs_aggregation=False)
        params = build_retrieval_params(decomposed)
        assert params.top_k == 6
        assert params.is_aggregation is False

    def test_aggregation_uses_top_k_20(self) -> None:
        decomposed = DecomposedQuery(intent="aggregation", needs_aggregation=True)
        params = build_retrieval_params(decomposed)
        assert params.top_k == 20
        assert params.is_aggregation is True

    def test_improvement_intent_forces_top_k_20_even_if_flag_false(self) -> None:
        decomposed = DecomposedQuery(intent="improvement", needs_aggregation=False)
        params = build_retrieval_params(decomposed)
        assert params.top_k == 20
        assert params.is_aggregation is True

    def test_compound_count_query_forces_top_k_20_even_if_flag_false(self) -> None:
        # A count_query with sub_queries is a compound question (e.g. "how many
        # positive reviews do I have and how can I improve?") -- the count half
        # is answered exactly via direct SQL regardless of top_k, but the
        # sub_queries half needs the same wide evidence pool an improvement
        # query gets. Confirmed live that needs_aggregation isn't reliably set
        # true by the model for these compound questions. This is one specific
        # instance of the general bool(sub_queries) rule tested below.
        decomposed = DecomposedQuery(
            intent="count_query",
            needs_aggregation=False,
            sub_queries=["What are the most common complaints?"],
        )
        params = build_retrieval_params(decomposed)
        assert params.top_k == 20
        assert params.is_aggregation is True

    def test_any_compound_question_forces_top_k_20_regardless_of_intent(self) -> None:
        # The widening rule checks bool(sub_queries) directly rather than
        # enumerating specific intents, so a compound sentiment_overview
        # question ("what % of my reviews are negative and how worried
        # should I be") gets the same wide evidence pool for its reasoning
        # half as a compound count_query does -- without needing its own
        # special case here.
        decomposed = DecomposedQuery(
            intent="sentiment_overview",
            needs_aggregation=False,
            wants_overall_stats=True,
            sub_queries=["How worried should I be?"],
        )
        params = build_retrieval_params(decomposed)
        assert params.top_k == 20
        assert params.is_aggregation is True

    def test_pure_count_query_without_sub_queries_uses_top_k_6(self) -> None:
        # A pure count_query (no sub_queries) never reaches build_retrieval_params
        # in practice -- it's answered by the direct-COUNT fast path before any
        # retrieval happens. This just documents that the sub_queries check
        # above doesn't accidentally widen top_k for every count_query.
        decomposed = DecomposedQuery(intent="count_query", needs_aggregation=False, sub_queries=[])
        params = build_retrieval_params(decomposed)
        assert params.top_k == 6
        assert params.is_aggregation is False

    def test_no_filters_leaves_none(self) -> None:
        decomposed = DecomposedQuery(intent="factual")
        params = build_retrieval_params(decomposed)
        assert params.date_from is None
        assert params.date_to is None
        assert params.rating_min is None
        assert params.rating_max is None

    def test_valid_date_filter_parsed_to_timestamps(self) -> None:
        decomposed = DecomposedQuery(
            intent="factual",
            date_filter=DateFilter(from_date="2024-01-01", to_date="2024-12-31"),
        )
        params = build_retrieval_params(decomposed)
        assert params.date_from is not None
        assert params.date_to is not None
        assert params.date_from < params.date_to

    def test_malformed_date_suppressed_to_none(self) -> None:
        decomposed = DecomposedQuery(
            intent="factual",
            date_filter=DateFilter(from_date="not-a-date"),
        )
        params = build_retrieval_params(decomposed)
        assert params.date_from is None

    def test_rating_filter_passed_through(self) -> None:
        decomposed = DecomposedQuery(
            intent="factual",
            rating_filter=RatingFilter(min=3.0, max=5.0),
        )
        params = build_retrieval_params(decomposed)
        assert params.rating_min == 3.0
        assert params.rating_max == 5.0
