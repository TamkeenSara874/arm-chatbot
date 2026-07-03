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

    On validation failure retries once with the error appended so the model can
    self-correct. Falls back to a safe factual intent if both attempts fail,
    ensuring no query hard-errors at the decomposition stage.
    """
    raw = await client.complete(
        prompt, system=system, temperature=0.0, max_tokens=512, usage_callback=usage_callback
    )

    validation_error_msg: str = ""
    try:
        return DecomposedQuery.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        validation_error_msg = str(exc)
        logger.warning(
            "decomposition_validation_failed",
            error=validation_error_msg,
            raw_preview=raw[:200],
        )

    retry_prompt = (
        f"{prompt}\n\nYour previous response failed JSON validation with error: {validation_error_msg}. "
        "Return only valid JSON matching the schema. No markdown fences or extra text."
    )
    try:
        raw2 = await client.complete(
            retry_prompt,
            system=system,
            temperature=0.0,
            max_tokens=512,
            usage_callback=usage_callback,
        )
        return DecomposedQuery.model_validate_json(raw2)
    except (ValidationError, ValueError) as exc2:
        logger.error("decomposition_failed_after_retry", error=str(exc2))

    return DecomposedQuery(intent="factual", complexity="simple", rephrased_query=prompt)
