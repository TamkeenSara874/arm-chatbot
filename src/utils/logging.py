import logging

import structlog


def _truncate_long_values(_logger: object, _method: str, event_dict: dict) -> dict:
    """Truncate any string value over 140 chars to keep log lines readable."""
    for key, value in event_dict.items():
        if key in ("event", "timestamp", "level", "logger"):
            continue
        if isinstance(value, str) and len(value) > 140:
            event_dict[key] = value[:137] + "…"
    return event_dict


def configure_logging(log_level: str = "INFO", debug: bool = False) -> None:
    if debug:
        timestamper = structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False)
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:
        timestamper = structlog.processors.TimeStamper(fmt="iso")
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            timestamper,
            structlog.processors.StackInfoRenderer(),
            _truncate_long_values,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    for noisy in (
        "uvicorn.access",
        "httpx",
        "httpcore",
        "sentence_transformers",
        "sqlalchemy",
        "fastembed",
        "tokenizers",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
