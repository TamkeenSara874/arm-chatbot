from __future__ import annotations

import re
from typing import TYPE_CHECKING

import structlog

from src.utils.metrics import injection_attempts_detected_total, output_validation_failed_total

if TYPE_CHECKING:
    from src.models.schemas import EvidenceItem

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


def _boundary_pattern(name: str) -> str:
    """re.escape(name) wrapped in \\b word boundaries, but only where the
    name's own edge is actually a word character.

    A name ending in punctuation (e.g. "Jared A.") has a non-word character
    on both sides of a trailing \\b when followed by whitespace/end-of-string
    -- \\b requires a word/non-word transition, and non-word-to-non-word
    isn't one, so a naive \\b...\\b pattern would silently never match that
    name at all. Dropping the boundary on whichever edge isn't a word
    character avoids that failure mode without weakening matching for the
    (usual) case of a name that starts and ends alphanumeric.
    """
    escaped = re.escape(name)
    prefix = r"\b" if name[0].isalnum() else ""
    suffix = r"\b" if name[-1].isalnum() else ""
    return prefix + escaped + suffix


_NAME_REQUEST_PATTERN = re.compile(
    r"\bname(s|d)?\b"
    r"|\bwho\s+(wrote|said|left|posted|is|are|complained)\b"
    # "which specific customers complained...", "which 8 reviewers mention..." --
    # asking to identify/enumerate specific people, with no "name"/"who" in
    # sight. Confirmed live as a real gap: "can you tell me which specific
    # customers complained about the hidden service fee?" is unambiguously
    # asking for identities, but got every reviewer anonymized anyway since
    # the pattern didn't recognize this phrasing at all. .{0,20} tolerates a
    # number or adjective between "which" and the noun (gap-tolerant, same
    # style as _REPLY_STATUS_PATTERNS in chat.py).
    r"|\bwhich\b.{0,20}\b(customers?|reviewers?|guests?)\b",
    re.IGNORECASE,
)


def wants_reviewer_names(query: str) -> bool:
    """True when the CURRENT question is itself explicitly asking to know
    who wrote something ("tell me their names", "who wrote these reviews",
    "name them").

    Reviewer usernames are already public on the review platform (Google,
    Yelp, ...) and a restaurant owner asking about their own reviews has a
    real reason to know who left one -- deliberately answering "who wrote
    this" when that's the literal question isn't a privacy leak. The actual
    leak this module guards against is the model *volunteering* a name
    nobody asked for (confirmed live: a username showed up attached to a
    complaint in an answer to a question that never asked who wrote it).
    """
    return bool(_NAME_REQUEST_PATTERN.search(query))


def redact_reviewer_names(answer: str, evidence: list[EvidenceItem], raw_query: str) -> str:
    """Replace any reviewer username quoted in the answer with "a reviewer",
    unless the current question is itself asking for reviewer identities.

    Deterministic backstop for the REVIEWER NAME PRIVACY prompt rule
    (chat_response_complex.yaml rule 10b / chat_response_simple.yaml) --
    confirmed live that the rule alone isn't a guarantee: a real reviewer's
    username ("Cat Huffine", "Gary Silansky") showed up attached to examples
    in an answer to a question that never asked who wrote it. A prompt
    instruction is a request the model can silently ignore; this check runs
    unconditionally on every response regardless of whether the model
    complied.

    Two cases where naming a reviewer is the actual answer, not a leak, and
    redaction is skipped entirely: (1) wants_reviewer_names(raw_query) is
    true -- an explicit "tell me their names" style request; (2) the current
    question already names that specific person (asking for their own
    opinion as a reviewer).
    """
    if wants_reviewer_names(raw_query):
        return answer

    query_lower = raw_query.lower()
    seen: set[str] = set()
    redacted = answer
    for item in evidence:
        username = item.username
        if not username or username in seen:
            continue
        seen.add(username)
        if username.lower() in query_lower:
            continue
        redacted = re.sub(_boundary_pattern(username), "a reviewer", redacted, flags=re.IGNORECASE)
    return redacted


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


# Covers what browser MediaRecorder implementations actually produce
# (webm/opus in Chrome/Firefox, mp4/aac in Safari) plus common raw formats a
# non-browser client might send.
_ALLOWED_AUDIO_TYPES = {
    "audio/webm",
    "audio/ogg",
    "audio/mp4",
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
}


def check_audio_upload(content_type: str, size_bytes: int) -> None:
    """Raise ValueError for invalid voice-dictation audio uploads."""
    from src.config import get_settings

    max_bytes = get_settings().voice_max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise ValueError(f"Audio too large: {size_bytes} bytes exceeds {max_bytes} byte limit")

    if content_type not in _ALLOWED_AUDIO_TYPES:
        raise ValueError(
            f"Unsupported audio content type: {content_type}. Accepted: {sorted(_ALLOWED_AUDIO_TYPES)}"
        )
