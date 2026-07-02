"""Unit tests for the cross-encoder reranker."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.reranker import _sigmoid, rerank
from src.services.vector.base import SearchResult


def _sr(chunk_id: str, score: float, text: str = "generic review text") -> SearchResult:
    return SearchResult(id=chunk_id, score=score, payload={"text": text})


class TestSigmoid:
    def test_zero_maps_to_half(self) -> None:
        assert abs(_sigmoid(0.0) - 0.5) < 1e-9

    def test_large_positive_approaches_one(self) -> None:
        assert _sigmoid(10.0) > 0.99

    def test_large_negative_approaches_zero(self) -> None:
        assert _sigmoid(-10.0) < 0.01

    def test_always_in_unit_interval(self) -> None:
        for x in [-50.0, -5.0, 0.0, 5.0, 50.0]:
            s = _sigmoid(x)
            assert 0.0 <= s <= 1.0, f"sigmoid({x}) = {s} out of [0,1]"

    def test_monotonically_increasing(self) -> None:
        assert _sigmoid(-1.0) < _sigmoid(0.0) < _sigmoid(1.0)


class TestRerank:
    @pytest.mark.asyncio
    async def test_sorts_by_cross_encoder_score(self) -> None:
        results = [_sr("a", 0.5, "stale bread"), _sr("b", 0.3, "excellent biryani")]
        with patch("src.core.reranker.load_reranker", new_callable=AsyncMock) as mock_load:
            mock_model = MagicMock()
            mock_model.predict = MagicMock(return_value=[-3.0, 6.0])
            mock_load.return_value = mock_model
            ranked = await rerank("best food?", results, model_name="mock-model")
        assert ranked[0].id == "b", "Higher CE score should rank first"
        assert ranked[1].id == "a"

    @pytest.mark.asyncio
    async def test_scores_are_sigmoid_normalized(self) -> None:
        results = [_sr("x", 0.5)]
        with patch("src.core.reranker.load_reranker", new_callable=AsyncMock) as mock_load:
            mock_model = MagicMock()
            mock_model.predict = MagicMock(return_value=[0.0])
            mock_load.return_value = mock_model
            ranked = await rerank("query", results, model_name="mock-model")
        assert abs(ranked[0].score - 0.5) < 1e-6

    @pytest.mark.asyncio
    async def test_top_k_limits_output_length(self) -> None:
        results = [_sr(str(i), 0.5) for i in range(10)]
        with patch("src.core.reranker.load_reranker", new_callable=AsyncMock) as mock_load:
            mock_model = MagicMock()
            mock_model.predict = MagicMock(return_value=list(range(10)))
            mock_load.return_value = mock_model
            ranked = await rerank("query", results, model_name="mock-model", top_k=4)
        assert len(ranked) == 4

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self) -> None:
        ranked = await rerank("query", [], model_name="mock-model")
        assert ranked == []

    @pytest.mark.asyncio
    async def test_model_failure_falls_back_to_original_order(self) -> None:
        results = [_sr("first", 0.9), _sr("second", 0.5)]
        with patch("src.core.reranker.load_reranker", new_callable=AsyncMock) as mock_load:
            mock_model = MagicMock()
            mock_model.predict = MagicMock(side_effect=RuntimeError("GPU OOM"))
            mock_load.return_value = mock_model
            ranked = await rerank("query", results, model_name="mock-model", top_k=2)
        assert [r.id for r in ranked] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_payload_preserved_after_rerank(self) -> None:
        results = [
            SearchResult(
                id="a", score=0.5, payload={"text": "food", "rating": 4.5, "source": "Google"}
            ),
        ]
        with patch("src.core.reranker.load_reranker", new_callable=AsyncMock) as mock_load:
            mock_model = MagicMock()
            mock_model.predict = MagicMock(return_value=[2.0])
            mock_load.return_value = mock_model
            ranked = await rerank("food quality", results, model_name="mock-model")
        assert ranked[0].payload["rating"] == 4.5
        assert ranked[0].payload["source"] == "Google"

    @pytest.mark.asyncio
    async def test_high_ce_score_beats_high_rrf_score(self) -> None:
        rrf_winner = _sr("rrf", 0.99, "only slightly related content")
        ce_winner = _sr("ce", 0.10, "directly answers the user question about biryani")
        results = [rrf_winner, ce_winner]
        with patch("src.core.reranker.load_reranker", new_callable=AsyncMock) as mock_load:
            mock_model = MagicMock()
            mock_model.predict = MagicMock(return_value=[-1.0, 7.0])
            mock_load.return_value = mock_model
            ranked = await rerank("tell me about biryani", results, model_name="mock-model")
        assert ranked[0].id == "ce", "Cross-encoder should override RRF ordering"
