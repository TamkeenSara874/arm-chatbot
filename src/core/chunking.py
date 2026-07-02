from __future__ import annotations

import structlog

from src.utils.token_budget import estimate_tokens

logger = structlog.get_logger()


def chunk_text(
    text: str,
    chunk_size: int = 256,
    overlap_tokens: int = 32,
) -> list[str]:
    """Return a list of text chunks using an adaptive strategy.

    Short texts (at most chunk_size tokens) are returned as a single chunk.
    Long texts use a sentence-level sliding window that respects chunk_size
    boundaries and carries approximately overlap_tokens tokens of context
    from the previous chunk into the next.
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if estimate_tokens(text) <= chunk_size:
        return [text]

    sentences = _sentence_split(text)
    if not sentences:
        return [text]

    chunks: list[str] = []
    window: list[str] = []
    window_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        if window_tokens + sentence_tokens > chunk_size and window:
            chunk = " ".join(window).strip()
            if chunk:
                chunks.append(chunk)

            # Carry the last overlap_tokens worth of sentences forward
            carry: list[str] = []
            carry_count = 0
            for s in reversed(window):
                s_count = estimate_tokens(s)
                if carry_count + s_count <= overlap_tokens:
                    carry.insert(0, s)
                    carry_count += s_count
                else:
                    break
            window = carry
            window_tokens = carry_count

        window.append(sentence)
        window_tokens += sentence_tokens

    if window:
        last = " ".join(window).strip()
        if last:
            chunks.append(last)

    return chunks or [text]


def _sentence_split(text: str) -> list[str]:
    """Split text into sentences, falling back to punctuation split on NLTK failure."""
    try:
        import nltk

        return nltk.sent_tokenize(text)
    except LookupError:
        logger.warning("nltk_punkt_unavailable", action="falling back to punctuation split")
    except Exception as exc:
        logger.warning("nltk_sent_tokenize_failed", error=str(exc))

    # Minimal fallback: split on sentence-ending punctuation
    import re

    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
