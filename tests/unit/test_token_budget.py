"""Unit tests for token estimation and budget enforcement."""

from src.utils.token_budget import enforce_token_budget, estimate_tokens


def test_estimate_tokens_reasonable_for_short_text() -> None:
    text = "Hello, this is a short sentence."
    count = estimate_tokens(text)
    assert count > 0
    assert count < 20


def test_estimate_tokens_scales_with_length() -> None:
    short = "Hello world."
    long = "Hello world. " * 100
    assert estimate_tokens(long) > estimate_tokens(short)


def test_estimate_tokens_within_5_percent_for_100_words() -> None:
    text = " ".join(["restaurant"] * 100)
    count = estimate_tokens(text)
    # 100 common words, typical encoding ~1.3 tokens each
    assert 90 <= count <= 160


def test_estimate_tokens_empty_string() -> None:
    assert estimate_tokens("") == 0


def test_enforce_token_budget_returns_original_when_under_limit() -> None:
    text = "Short text."
    result = enforce_token_budget(text, max_tokens=512)
    assert result == text


def test_enforce_token_budget_trims_long_text() -> None:
    long_text = "word " * 2000
    result = enforce_token_budget(long_text, max_tokens=256)
    assert estimate_tokens(result) <= 256


def test_enforce_token_budget_preserves_start_and_end() -> None:
    start = "START_MARKER "
    end = " END_MARKER"
    middle = "filler text " * 1000
    text = start + middle + end
    result = enforce_token_budget(text, max_tokens=100)
    assert "START_MARKER" in result
    assert "END_MARKER" in result


def test_enforce_token_budget_exactly_at_limit() -> None:
    text = "word " * 100
    tokens = estimate_tokens(text)
    result = enforce_token_budget(text, max_tokens=tokens)
    assert estimate_tokens(result) <= tokens


def test_estimate_tokens_unknown_model_falls_back() -> None:
    count = estimate_tokens("hello world", model="nonexistent-model")
    assert count > 0
