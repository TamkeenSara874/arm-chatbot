from __future__ import annotations

import re

from src.utils.metrics import guardrail_triggered_total

# Deterministic, code-only check for language suggesting the user themselves
# may be in crisis (self-harm/suicidal ideation) -- deliberately NOT folded
# into the intent-based guardrail (src/core/guardrail.py) or left to the
# generation prompt's general "emotional tone" rule. Both of those depend on
# an LLM classification/instruction running first and correctly noticing the
# signal buried inside an otherwise normal-looking business question (e.g.
# "I want to die, the reviews are so bad" still decomposes as a countable
# review question) -- confirmed live that it doesn't reliably happen. This
# check runs on the raw sanitized text itself, before decomposition or any
# LLM call, so it can't be missed, delayed, or out-argued by a smaller model.
#
# Patterns are specific multi-word phrases, not single words like "die" or
# "kill" alone, which appear harmlessly in ordinary business language ("my
# sales are dying," "kill the competition").
_CRISIS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bi\s*(?:'m|\bam)?\s*want(?:s|ed)?\s+to\s+die\b",
        r"\bwant(?:s|ed)?\s+to\s+kill\s+myself\b",
        r"\bkill(?:ing)?\s+myself\b",
        r"\bi\s*(?:'m|\bam)\s+suicidal\b",
        r"\bsuicidal\s+thoughts\b",
        r"\bthinking\s+about\s+suicide\b",
        r"\bend(?:ing)?\s+my\s+(?:own\s+)?life\b",
        r"\bending\s+it\s+all\b",
        r"\bdon'?t\s+want\s+to\s+(?:live|be\s+alive)\b",
        r"\bno\s+reason\s+to\s+live\b",
        r"\bnot\s+worth\s+living\b",
        r"\bhurt(?:ing)?\s+myself\b",
        r"\bself[\s-]?harm(?:ing|ed)?\b",
    ]
]

CRISIS_RESPONSE = (
    "I'm really sorry you're feeling this way -- that matters more than any review data. "
    "I'm just a restaurant-reviews assistant, so I'm not the right place to help with this, "
    "but please don't go through it alone. If you're in the US, you can call or text 988 (the "
    "Suicide & Crisis Lifeline) any time, day or night. Wherever you are, "
    "https://findahelpline.com lists crisis lines by country. I'll be here with your review "
    "data whenever you're ready to come back to it."
)


def detect_crisis_language(text: str) -> bool:
    """True if text contains language suggesting the user may be in crisis.

    Also increments guardrail_triggered_total(type="crisis_language") so this
    is trackable the same way other guardrail categories are.
    """
    triggered = any(pattern.search(text) for pattern in _CRISIS_PATTERNS)
    if triggered:
        guardrail_triggered_total.labels(type="crisis_language").inc()
    return triggered
