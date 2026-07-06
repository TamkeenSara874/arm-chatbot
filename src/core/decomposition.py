from __future__ import annotations

import structlog
from pydantic import ValidationError

from src.models.schemas import DecomposedQuery
from src.services.llm.base import BaseLLMClient, UsageCallback

logger = structlog.get_logger()


async def decompose_query(
    client: BaseLLMClient,
    prompt: str,
    system: str,
    usage_callback: UsageCallback | None = None,
) -> DecomposedQuery:
    """Classify and decompose a user query into a structured DecomposedQuery.

    Uses each client's complete_structured() -- OpenAI's native beta.parse()
    or Groq's JSON-mode + schema validation -- instead of hand-rolled
    json.loads(), so schema enforcement happens once, in one place, for every
    provider. On validation failure retries once with the error appended so
    the model can self-correct. Falls back to a safe factual intent if both
    attempts fail, ensuring no query hard-errors at the decomposition stage.
    """
    # Only visible at LOG_LEVEL=DEBUG -- the actual system_prompt/user_prompt
    # sent to the LLM, session_context included, exactly as interpolated. The
    # classified intent/complexity alone (in request_traces.jsonl) isn't
    # enough to tell whether session context caused a misclassification;
    # seeing the real prompt text is what makes that verifiable.
    logger.debug("decomposition_prompt", system=system, user=prompt)
    validation_error_msg: str = ""
    try:
        return await client.complete_structured(
            prompt,
            system,
            response_format=DecomposedQuery,
            temperature=0.0,
            max_tokens=512,
            usage_callback=usage_callback,
        )
    except (ValidationError, ValueError) as exc:
        validation_error_msg = str(exc)
        logger.warning("decomposition_validation_failed", error=validation_error_msg)

    retry_prompt = (
        f"{prompt}\n\nYour previous response failed JSON validation with error: {validation_error_msg}. "
        "Return only valid JSON matching the schema. No markdown fences or extra text."
    )
    try:
        return await client.complete_structured(
            retry_prompt,
            system,
            response_format=DecomposedQuery,
            temperature=0.0,
            max_tokens=512,
            usage_callback=usage_callback,
        )
    except (ValidationError, ValueError) as exc2:
        logger.error("decomposition_failed_after_retry", error=str(exc2))

    return DecomposedQuery(intent="factual", complexity="simple", rephrased_query=prompt)
