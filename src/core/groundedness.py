from __future__ import annotations

import re

# Star ratings ("5 stars", "4-star") and years (1900-2099) are numbers that
# frequently appear right before a count-like noun purely by coincidence
# ("5 stars... reviews mention...") and are excluded from the overclaim check.
_STAR_CONTEXT_RE = re.compile(r"\d+\s*[-\s]?stars?\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"^(19|20)\d{2}$")

# Matches a number followed, within a few words, by a count-ish noun -- e.g.
# "7 reviews", "12 of the retrieved reviews", "3 customers complained".
_COUNT_CLAIM_RE = re.compile(
    r"\b(\d+)\b(?:\s+\w+){0,3}?\s+(?:reviews?|mentions?|customers?|complaints?|times|people)\b",
    re.IGNORECASE,
)


def check_count_groundedness(
    answer: str,
    evidence_count: int,
    precomputed_count: str | None = None,
) -> bool:
    """Flag whether the generated answer's stated counts are supported by the evidence.

    A cheap, code-only heuristic (no extra LLM call): if the model states a count of
    reviews/mentions/customers strictly greater than the number of evidence chunks it
    was actually given, that count is very likely fabricated rather than genuinely
    tallied. Returns True (grounded) when no such overclaim is found, False otherwise.

    When precomputed_count is set (an exact Postgres COUNT(*) was computed for this
    query), any count is exempt -- the model is expected to restate that verified
    number, which can legitimately exceed the retrieved evidence sample size.
    """
    if precomputed_count:
        return True

    star_spans = {m.span() for m in _STAR_CONTEXT_RE.finditer(answer)}
    for match in _COUNT_CLAIM_RE.finditer(answer):
        if match.span() in star_spans:
            continue
        digits = match.group(1)
        if _YEAR_RE.match(digits):
            continue
        if int(digits) > max(evidence_count, 1):
            return False
    return True
