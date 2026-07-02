"""Unit tests for retrieval module -- pure functions only (no I/O)."""

from src.core.retrieval import _fuse_and_rank, _tokenize, invalidate_bm25_cache
from src.services.vector.base import SearchResult


def _sr(chunk_id: str, score: float, text: str = "review text") -> SearchResult:
    return SearchResult(
        id=chunk_id,
        score=score,
        payload={"text": text, "rating": 4.0, "source": "Google"},
    )


class TestInvalidateBM25Cache:
    def test_invalidate_nonexistent_restaurant_is_safe(self) -> None:
        invalidate_bm25_cache(99999)

    def test_invalidate_clears_cache_entry(self) -> None:
        from rank_bm25 import BM25Okapi

        from src.core.retrieval import _bm25_cache

        _bm25_cache[1234] = (BM25Okapi([["hello"]]), [{"id": "x", "text": "hello"}])
        invalidate_bm25_cache(1234)
        assert 1234 not in _bm25_cache


class TestTokenize:
    def test_lowercases_words(self) -> None:
        tokens = _tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        tokens = _tokenize("food, drinks!")
        assert "food" in tokens
        assert "drinks" in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == []


class TestFuseAndRank:
    def test_rrf_scores_merged_across_lists(self) -> None:
        dense = [_sr("a", 0.9), _sr("b", 0.8)]
        bm25 = [_sr("b", 8.0), _sr("c", 7.0)]
        results = _fuse_and_rank(dense, bm25, top_k=3)
        ids = [r.id for r in results]
        assert "b" in ids
        assert "a" in ids
        assert "c" in ids

    def test_chunk_appearing_in_both_lists_ranks_higher(self) -> None:
        dense = [_sr("shared", 0.9), _sr("dense_only", 0.85)]
        bm25 = [_sr("shared", 7.0), _sr("bm25_only", 6.5)]
        results = _fuse_and_rank(dense, bm25, top_k=3, rrf_k=60)
        assert results[0].id == "shared", "Chunk in both lists should rank first"

    def test_dense_payload_takes_priority_over_bm25(self) -> None:
        dense = [
            SearchResult(
                id="x", score=1.0, payload={"text": "dense payload", "food_entities": ["naan"]}
            )
        ]
        bm25 = [
            SearchResult(id="x", score=5.0, payload={"text": "bm25 payload", "food_entities": []})
        ]
        results = _fuse_and_rank(dense, bm25, top_k=1)
        assert results[0].payload["text"] == "dense payload"
        assert results[0].payload["food_entities"] == ["naan"]

    def test_top_k_limit_respected(self) -> None:
        dense = [_sr(f"d{i}", 1.0 / (i + 1)) for i in range(10)]
        bm25 = [_sr(f"b{i}", 10.0 / (i + 1)) for i in range(10)]
        results = _fuse_and_rank(dense, bm25, top_k=5)
        assert len(results) == 5

    def test_score_field_is_rrf_score(self) -> None:
        dense = [_sr("a", 0.9)]
        results = _fuse_and_rank(dense, [], top_k=1, rrf_k=60)
        assert results[0].id == "a"
        expected_rrf = 1.0 / (60 + 1)
        assert abs(results[0].score - expected_rrf) < 1e-9

    def test_empty_dense_uses_bm25_only(self) -> None:
        bm25 = [_sr("bm25_chunk", 5.0)]
        results = _fuse_and_rank([], bm25, top_k=2)
        assert len(results) == 1
        assert results[0].id == "bm25_chunk"

    def test_empty_both_returns_empty(self) -> None:
        results = _fuse_and_rank([], [], top_k=5)
        assert results == []
