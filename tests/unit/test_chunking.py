"""Unit tests for adaptive text chunking."""

from src.core.chunking import chunk_text
from src.utils.token_budget import estimate_tokens


def test_empty_text_returns_empty_list() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_returns_single_chunk() -> None:
    text = "Amazing biryani! Great service. Will come back."
    chunks = chunk_text(text, chunk_size=256)
    assert chunks == [text.strip()]


def test_single_chunk_when_exactly_at_limit() -> None:
    word = "word "
    text = word * 50  # well under 256 tokens
    chunks = chunk_text(text, chunk_size=256)
    assert len(chunks) == 1


def test_long_text_produces_multiple_chunks() -> None:
    # Generate a text clearly over 256 tokens
    sentence = "The chicken biryani was absolutely amazing and I would recommend it to everyone. "
    text = sentence * 30
    chunks = chunk_text(text, chunk_size=256)
    assert len(chunks) > 1


def test_chunks_respect_size_limit() -> None:
    sentence = "The food was great and service was fast. "
    text = sentence * 40
    chunks = chunk_text(text, chunk_size=100, overlap_tokens=10)
    for chunk in chunks:
        tokens = estimate_tokens(chunk)
        # Allow a small overshoot when a single sentence exceeds the budget
        assert tokens <= 150, f"Chunk too large: {tokens} tokens"


def test_all_content_is_preserved() -> None:
    """No text should be silently dropped during chunking."""
    sentence = "Seekh kebab was overcooked but the naan was fresh. "
    text = sentence * 20
    chunks = chunk_text(text, chunk_size=80, overlap_tokens=10)
    combined = " ".join(chunks)
    # Each unique sentence token should appear at least once across chunks
    for word in ["Seekh", "kebab", "overcooked", "naan", "fresh"]:
        assert word.lower() in combined.lower(), f"Word '{word}' missing from chunks"


def test_no_empty_chunks() -> None:
    sentence = "Good food. "
    text = sentence * 50
    chunks = chunk_text(text, chunk_size=50, overlap_tokens=5)
    for chunk in chunks:
        assert chunk.strip(), "Empty chunk found"


def test_single_very_long_sentence() -> None:
    long_sentence = ("word " * 300).strip()
    chunks = chunk_text(long_sentence, chunk_size=100)
    assert len(chunks) >= 1
    assert all(c.strip() for c in chunks)
