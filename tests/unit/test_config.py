"""Unit tests for Settings — no I/O required."""

from src.config import Settings


def test_allowed_origins_list_parses_csv() -> None:
    s = Settings(
        database_url="postgresql+asyncpg://x:x@localhost/x",
        allowed_origins="http://localhost:5173,http://localhost:3000",
    )
    assert s.allowed_origins_list == ["http://localhost:5173", "http://localhost:3000"]


def test_allowed_origins_list_strips_whitespace() -> None:
    s = Settings(
        database_url="postgresql+asyncpg://x:x@localhost/x",
        allowed_origins=" http://a.com , http://b.com ",
    )
    assert s.allowed_origins_list == ["http://a.com", "http://b.com"]


def test_model_names_configurable() -> None:
    s = Settings(
        database_url="postgresql+asyncpg://x:x@localhost/x",
        openai_simple_model="gpt-5-mini",
        openai_complex_model="gpt-5",
    )
    assert s.openai_simple_model == "gpt-5-mini"
    assert s.openai_complex_model == "gpt-5"


def test_defaults_are_sane() -> None:
    s = Settings(database_url="postgresql+asyncpg://x:x@localhost/x")
    assert s.embedding_dim == 3072
    assert s.correction_sim_threshold == 0.85
    assert s.data_staleness_days == 365
    assert s.entity_extraction_batch_size == 10
    # Ranking weights sum to 1.0
    total = s.ranking_weight_rrf + s.ranking_weight_recency + s.ranking_weight_rating
    assert abs(total - 1.0) < 1e-9
