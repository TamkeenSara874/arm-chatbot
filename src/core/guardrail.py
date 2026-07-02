from __future__ import annotations

from src.utils.metrics import guardrail_triggered_total

GUARDRAIL_RESPONSES: dict[str, str] = {
    "out_of_scope": (
        "That is a great question, but I am only able to help with insights from your "
        "restaurant's reviews. For anything outside of that, you would need a different "
        "source. Is there anything I can help you with from your customer feedback?"
    ),
    "ui_question": (
        "I am your restaurant insights assistant — I focus on what your customers are "
        "saying in their reviews. For questions about navigating or using the AIO platform "
        "itself, AIO's CareBot assistant can help you directly. Can I help you with "
        "anything from your reviews?"
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
