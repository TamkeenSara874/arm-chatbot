from __future__ import annotations

import re

import structlog

from src.utils.metrics import injection_attempts_detected_total, output_validation_failed_total

logger = structlog.get_logger()

INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions",
    r"ignore\s+all\s+instructions",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"(system|assistant|user)\s*:\s*",
    r"<\|im_(start|end)\|>",
    r"<\|endoftext\|>",
    r"forget\s+(?:your|all|previous)\s+(?:previous\s+)?(?:instructions|context|training)",
    r"new\s+(instructions|persona|role)\s*:",
    r"do\s+not\s+(follow|obey)\s+(your|the)\s+(previous|prior|original)",
    r"\[\[.*?injection.*?\]\]",
]

SYSTEM_LEAK_PATTERNS: list[str] = [
    "system_prompt",
    "ignore previous",
    "instructions:",
    "as an ai",
    "i have been freed",
    "without restrictions",
    "my true",
    "actually i am",
    "----begin review",
    "----end review",
    "restaurant_id",
    "embedding",
    "vector",
    "qdrant",
]


def scan_for_injection(text: str) -> bool:
    """Return True if text matches any known prompt injection pattern."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in INJECTION_PATTERNS)


def flag_injection(text: str, restaurant_id: int) -> bool:
    """Scan and increment metric counter if a pattern is found."""
    found = scan_for_injection(text)
    if found:
        injection_attempts_detected_total.labels(restaurant_id=str(restaurant_id)).inc()
        logger.warning(
            "injection_attempt_detected",
            restaurant_id=restaurant_id,
            text_preview=text[:80],
        )
    return found


def sanitize_input(text: str, max_length: int = 2000) -> str:
    """Truncate and remove known injection patterns from user-submitted text."""
    text = text[:max_length]
    for pattern in INJECTION_PATTERNS:
        text = re.sub(pattern, "[removed]", text, flags=re.IGNORECASE)
    return text.strip()


def validate_llm_output(response: object) -> object:
    """Detect and replace outputs that contain system leak or injection markers.

    Returns the original response if clean, or a safe fallback if suspicious
    content is detected. Importing ChatResponseSchema here avoids a circular
    import since schemas.py has no dependency on this module.
    """
    from src.models.schemas import ChatResponseSchema

    if not isinstance(response, ChatResponseSchema):
        return response

    combined = (response.answer + " " + (response.caveats or "")).lower()
    matches = [p for p in SYSTEM_LEAK_PATTERNS if p in combined]

    if matches:
        output_validation_failed_total.inc()
        logger.error("llm_output_validation_failed", detected_patterns=matches)
        return ChatResponseSchema(
            answer="I was not able to produce a reliable answer for that question. Please try rephrasing.",
            evidence=[],
            confidence=0.0,
            caveats="Response validation failed.",
        )

    return response


def check_file_upload(filename: str, content_type: str, size_bytes: int) -> None:
    """Raise ValueError for invalid uploads before any parsing begins."""
    from src.config import get_settings

    max_bytes = get_settings().ingest_max_file_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValueError(f"File too large: {size_bytes} bytes exceeds {max_bytes} byte limit")

    allowed_types = {"text/csv", "application/json", "text/plain"}
    if content_type not in allowed_types:
        raise ValueError(
            f"Unsupported content type: {content_type}. Accepted: {sorted(allowed_types)}"
        )
