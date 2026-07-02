from __future__ import annotations

MODEL_COSTS_USD_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.10, "output": 0.0},
    "llama-3.3-70b-versatile": {"input": 0.0, "output": 0.0},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int = 0) -> float:
    """Return estimated USD cost for a single LLM or embedding call.

    Unknown models return 0.0 so missing entries never raise.
    """
    rates = MODEL_COSTS_USD_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000
