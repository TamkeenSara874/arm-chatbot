"""Unit tests for the cross-encoder reranker."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.reranker import (
    _load_cross_encoder,
    _quantized_export_dir,
    _quantized_file_name,
    _sigmoid,
    load_reranker,
    rerank,
)
from src.services.vector.base import SearchResult


def _sr(chunk_id: str, score: float, text: str = "generic review text") -> SearchResult:
    return SearchResult(id=chunk_id, score=score, payload={"text": text})


class TestLoadCrossEncoder:
    def test_returns_cross_encoder_instance(self) -> None:
        mock_ce_cls = MagicMock()
        mock_instance = MagicMock()
        mock_ce_cls.return_value = mock_instance

        with patch.dict(
            "sys.modules", {"sentence_transformers": MagicMock(CrossEncoder=mock_ce_cls)}
        ):
            result = _load_cross_encoder("BAAI/bge-reranker-base")

        mock_ce_cls.assert_called_once_with("BAAI/bge-reranker-base")
        assert result is mock_instance


class TestQuantizedFileName:
    def test_matches_export_dynamic_quantized_onnx_model_output_naming(self) -> None:
        # Confirmed empirically against the real installed library: for the
        # "avx2" config specifically the exported file is named
        # "model_quint8_avx2.onnx", not "model_qint8_avx2.onnx" as the
        # avx512_vnni-family naming (pre-published upstream) would suggest.
        assert _quantized_file_name("avx2") == "model_quint8_avx2.onnx"


class TestQuantizedExportDir:
    def test_path_under_hf_home_keyed_by_model_and_config(self) -> None:
        with patch("huggingface_hub.constants.HF_HOME", "/fake/hf/home"):
            result = _quantized_export_dir("cross-encoder/ms-marco-MiniLM-L6-v2", "avx2")
        assert result == Path(
            "/fake/hf/home/onnx-quantized/cross-encoder__ms-marco-MiniLM-L6-v2/avx2"
        )

    def test_different_quantization_configs_get_different_dirs(self) -> None:
        with patch("huggingface_hub.constants.HF_HOME", "/fake/hf/home"):
            avx2_dir = _quantized_export_dir("some/model", "avx2")
            arm64_dir = _quantized_export_dir("some/model", "arm64")
        assert avx2_dir != arm64_dir


class TestEnsureQuantizedExport:
    def _fake_sentence_transformers_modules(self, mock_ce_cls, mock_export_fn):
        backend_module = MagicMock(export_dynamic_quantized_onnx_model=mock_export_fn)
        st_module = MagicMock(CrossEncoder=mock_ce_cls)
        st_module.backend = backend_module
        return {"sentence_transformers": st_module, "sentence_transformers.backend": backend_module}

    def test_skips_export_when_quantized_file_already_exists(self, tmp_path) -> None:
        from src.core.reranker import _ensure_quantized_export

        export_dir = tmp_path / "export"
        onnx_dir = export_dir / "onnx"
        onnx_dir.mkdir(parents=True)
        (onnx_dir / "model_quint8_avx2.onnx").write_bytes(b"fake weights")

        mock_export_fn = MagicMock()
        with patch.dict(
            "sys.modules", self._fake_sentence_transformers_modules(MagicMock(), mock_export_fn)
        ):
            _ensure_quantized_export("some/model", "avx2", export_dir)

        mock_export_fn.assert_not_called()

    def test_exports_and_copies_sibling_files_when_missing(self, tmp_path) -> None:
        from src.core.reranker import _ensure_quantized_export

        export_dir = tmp_path / "export"
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        (snapshot_dir / "config.json").write_text('{"model_type": "bert"}')
        (snapshot_dir / "vocab.txt").write_text("fake vocab")

        mock_ce_cls = MagicMock()
        mock_export_fn = MagicMock()

        def _fake_export(base_model, quantization_config, model_name_or_path) -> None:
            # Mimic the real library's behavior: only the .onnx weights file
            # is written, no config/tokenizer siblings.
            out = Path(model_name_or_path) / "onnx"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"model_quint8_{quantization_config}.onnx").write_bytes(b"fake weights")

        mock_export_fn.side_effect = _fake_export

        with (
            patch.dict(
                "sys.modules", self._fake_sentence_transformers_modules(mock_ce_cls, mock_export_fn)
            ),
            patch("huggingface_hub.snapshot_download", return_value=str(snapshot_dir)),
        ):
            _ensure_quantized_export("some/model", "avx2", export_dir)

        mock_export_fn.assert_called_once()
        assert (export_dir / "onnx" / "model_quint8_avx2.onnx").exists()
        assert (export_dir / "config.json").read_text() == '{"model_type": "bert"}'
        assert (export_dir / "vocab.txt").exists()
        # model.safetensors is deliberately not copied -- the onnx backend
        # doesn't need the original FP32 weights.
        assert not (export_dir / "model.safetensors").exists()


class TestLoadCrossEncoderOnnxQuantized:
    def test_loads_from_export_dir_with_quantized_file_name(self, tmp_path) -> None:
        from src.core.reranker import _load_cross_encoder_onnx_quantized

        mock_ce_cls = MagicMock()
        mock_instance = MagicMock()
        mock_ce_cls.return_value = mock_instance

        with (
            patch.dict(
                "sys.modules", {"sentence_transformers": MagicMock(CrossEncoder=mock_ce_cls)}
            ),
            patch("src.core.reranker._quantized_export_dir", return_value=tmp_path),
            patch("src.core.reranker._ensure_quantized_export") as mock_ensure,
        ):
            result = _load_cross_encoder_onnx_quantized("some/model", "avx2")

        mock_ensure.assert_called_once_with("some/model", "avx2", tmp_path)
        mock_ce_cls.assert_called_once_with(
            str(tmp_path), backend="onnx", model_kwargs={"file_name": "model_quint8_avx2.onnx"}
        )
        assert result is mock_instance


class TestLoadReranker:
    @pytest.mark.asyncio
    async def test_first_call_loads_and_caches_model(self) -> None:
        import src.core.reranker as reranker_module

        reranker_module._model_cache.clear()

        mock_model = MagicMock()
        mock_settings = MagicMock(
            reranker_onnx_quantized=False, reranker_onnx_quantization_config="avx2"
        )

        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("src.core.reranker._load_cross_encoder", return_value=mock_model),
        ):
            result = await load_reranker("test-model-unique")

        assert result is mock_model
        assert reranker_module._model_cache.get("test-model-unique") is mock_model

    @pytest.mark.asyncio
    async def test_second_call_returns_cached_model(self) -> None:
        import src.core.reranker as reranker_module

        mock_model = MagicMock()
        reranker_module._model_cache["cached-model"] = mock_model

        with patch("src.core.reranker._load_cross_encoder") as mock_load:
            result = await load_reranker("cached-model")
            mock_load.assert_not_called()

        assert result is mock_model

    @pytest.mark.asyncio
    async def test_uses_plain_loader_when_onnx_quantized_disabled(self) -> None:
        import src.core.reranker as reranker_module

        reranker_module._model_cache.clear()
        mock_model = MagicMock()
        mock_settings = MagicMock(
            reranker_onnx_quantized=False, reranker_onnx_quantization_config="avx2"
        )

        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("src.core.reranker._load_cross_encoder", return_value=mock_model) as mock_plain,
            patch("src.core.reranker._load_cross_encoder_onnx_quantized") as mock_quantized,
        ):
            result = await load_reranker("some-model-a")

        mock_plain.assert_called_once_with("some-model-a")
        mock_quantized.assert_not_called()
        assert result is mock_model

    @pytest.mark.asyncio
    async def test_uses_quantized_loader_when_onnx_quantized_enabled(self) -> None:
        import src.core.reranker as reranker_module

        reranker_module._model_cache.clear()
        mock_model = MagicMock()
        mock_settings = MagicMock(
            reranker_onnx_quantized=True, reranker_onnx_quantization_config="avx2"
        )

        with (
            patch("src.config.get_settings", return_value=mock_settings),
            patch("src.core.reranker._load_cross_encoder") as mock_plain,
            patch(
                "src.core.reranker._load_cross_encoder_onnx_quantized", return_value=mock_model
            ) as mock_quantized,
        ):
            result = await load_reranker("some-model-b")

        mock_quantized.assert_called_once_with("some-model-b", "avx2")
        mock_plain.assert_not_called()
        assert result is mock_model


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
    async def test_degenerate_scores_fall_back_to_original_order(self) -> None:
        # Near-identical logits across every candidate (e.g. a broad "summarize
        # the negative reviews" style question the cross-encoder wasn't trained
        # to discriminate) should keep the pre-rerank order/scores instead of a
        # meaningless wall of near-0% sigmoid scores.
        results = [_sr("first", 0.9), _sr("second", 0.5), _sr("third", 0.3)]
        with patch("src.core.reranker.load_reranker", new_callable=AsyncMock) as mock_load:
            mock_model = MagicMock()
            mock_model.predict = MagicMock(return_value=[-11.44, -11.45, -11.46])
            mock_load.return_value = mock_model
            ranked = await rerank(
                "summarize the negative reviews", results, model_name="mock-model"
            )
        assert [r.id for r in ranked] == ["first", "second", "third"]
        assert [r.score for r in ranked] == [0.9, 0.5, 0.3]

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
