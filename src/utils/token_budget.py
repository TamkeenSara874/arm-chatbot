import tiktoken


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Return the approximate token count for text under the given model encoding."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def enforce_token_budget(text: str, max_tokens: int = 2048, model: str = "gpt-4o") -> str:
    """Trim text to at most max_tokens by removing the middle section.

    Preserves the beginning and end of the text so both context-setting
    introduction and closing conclusions remain intact after truncation.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text

    half = max_tokens // 2
    trimmed = tokens[:half] + tokens[-half:]
    return enc.decode(trimmed)
