from __future__ import annotations

import re

from src.utils.metrics import guardrail_triggered_total

# A bare greeting or acknowledgement carries no question, so it must never reach
# review generation: with prior turns in the session context, the model answers
# "hey" by regurgitating the last topic (confirmed live -- "hey" returned a full
# atmosphere summary). Handled deterministically and *before* decomposition, so
# no session context is ever assembled for it.
GREETING_RESPONSE = (
    "Happy to help! Ask me anything about your guest reviews and I'll answer "
    "from what your customers have said."
)

# Whole-message greetings only. Deliberately exact-match against a normalised
# form so "hey" matches but "hey, what do guests say about parking?" does not --
# a real question that merely opens with a greeting must go through the pipeline.
_GREETINGS: frozenset[str] = frozenset(
    {
        "hi",
        "hey",
        "hello",
        "yo",
        "hiya",
        "heya",
        "hey there",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "thankyou",
        "ty",
        "thx",
        "cheers",
        "ok",
        "okay",
        "cool",
        "great",
        "got it",
        "sounds good",
    }
)


def detect_greeting(text: str) -> bool:
    """True when the entire message is a bare greeting or acknowledgement."""
    normalized = re.sub(r"[^a-z\s]", "", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized in _GREETINGS


GUARDRAIL_RESPONSES: dict[str, str] = {
    "out_of_scope": (
        "That is a great question, but I am only able to help with insights from your "
        "restaurant's reviews. For anything outside of that, you would need a different "
        "source. Is there anything I can help you with from your customer feedback?"
    ),
    "ui_question": (
        "I am your restaurant insights assistant, focused on what your customers are "
        "saying in their reviews. For questions about navigating or using the app itself, "
        "AIO's CareBot assistant can help you directly -- try asking CareBot. Can I help "
        "you with anything from your reviews?"
    ),
    "report_howto": (
        "You can generate a full insights report from the 'Report' button in the top "
        "navigation bar -- it puts together ratings, sentiment, and top praised/complained "
        "items into one document. Once it's generated, use the 'Download PDF' button at the "
        "top of the report to save it. Is there anything from your reviews I can help with "
        "in the meantime?"
    ),
    "manipulation_request": (
        "I am not able to help with writing, removing, or influencing reviews -- that "
        "falls outside what I am here for and could affect the authenticity of your "
        "feedback. I can help you understand what your existing reviews are saying. "
        "Would you like to explore that?"
    ),
    "multi_location": (
        "I currently have review data for one restaurant at a time, so I am not able to "
        "compare across multiple branches in a single answer. You can switch restaurants "
        "using the selector at the top and ask the same question for each one. Can I help "
        "you dig into the reviews for the current restaurant?"
    ),
    "allergen": (
        "For specific allergen or food safety questions, please check directly with your "
        "kitchen team -- I can only share what reviewers have mentioned, which is not a "
        "reliable guide for allergy or dietary decisions. Is there anything else from your "
        "reviews I can help with?"
    ),
}

GUARDRAIL_INTENTS: frozenset[str] = frozenset(GUARDRAIL_RESPONSES)


def check_guardrail(intent: str) -> str | None:
    """Return a canned response if the intent is guardrailed, or None to pass through.

    Also increments the guardrail_triggered_total Prometheus counter so supervisors
    can track how often users hit each guardrail category.
    """
    response = GUARDRAIL_RESPONSES.get(intent)
    if response is not None:
        guardrail_triggered_total.labels(type=intent).inc()
    return response
