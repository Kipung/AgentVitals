from typing import Dict, Optional


# Pricing data per million tokens (input and output), in USD.
# If your provider pricing differs, override at runtime with `--model` and
# keep this table updated.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # Google
    "gemini-1.5-pro": {"input": 7.00, "output": 21.00},
    # Default / fallback
    "default": {"input": 1.00, "output": 1.00},
}

# Known context window sizes, in tokens.
MODEL_CONTEXT_WINDOWS: Dict[str, Optional[int]] = {
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "gemini-1.5-pro": 1_000_000,
    "default": None,
}


def _resolve_model_key(model_name: str, table: Dict[str, object]) -> str:
    """Resolve a model key from exact or substring match, case-insensitive."""
    normalized = (model_name or "").strip().lower()

    if not normalized:
        return "default"

    if normalized in table:
        return normalized

    # Prefer the longest key to avoid broad partial matches.
    candidate_keys = sorted((key for key in table if key != "default"), key=len, reverse=True)
    for key in candidate_keys:
        if key in normalized:
            return key

    return "default"


def get_price(model_name: str, token_type: str) -> float:
    """
    Get the USD price per million tokens for a given model and token type.

    Args:
        model_name: Provider model name (or any string containing it).
        token_type: 'input' or 'output'.

    Returns:
        The price per million tokens.
    """
    if token_type not in {"input", "output"}:
        raise ValueError("token_type must be 'input' or 'output'")

    model_key = _resolve_model_key(model_name, MODEL_PRICING)
    return MODEL_PRICING[model_key][token_type]


def get_context_window(model_name: str) -> Optional[int]:
    """
    Get the context window size in tokens for a model.

    Returns:
        The context window token size, or None if unknown.
    """
    model_key = _resolve_model_key(model_name, MODEL_CONTEXT_WINDOWS)
    return MODEL_CONTEXT_WINDOWS[model_key]
