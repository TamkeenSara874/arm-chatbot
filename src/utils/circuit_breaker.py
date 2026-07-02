import pybreaker
import structlog

logger = structlog.get_logger()


class _LoggingListener(pybreaker.CircuitBreakerListener):
    def state_change(
        self,
        cb: pybreaker.CircuitBreaker,
        old_state: pybreaker.CircuitBreakerState,
        new_state: pybreaker.CircuitBreakerState,
    ) -> None:
        logger.warning(
            "circuit_breaker_state_change",
            name=cb.name,
            old_state=old_state.name,
            new_state=new_state.name,
        )

    def failure(self, cb: pybreaker.CircuitBreaker, exc: Exception) -> None:
        logger.error(
            "circuit_breaker_failure",
            name=cb.name,
            fail_counter=cb.fail_counter,
            error=str(exc),
        )


_listener = _LoggingListener()


def _breaker(name: str, fail_max: int = 5, reset_timeout: int = 60) -> pybreaker.CircuitBreaker:
    return pybreaker.CircuitBreaker(
        fail_max=fail_max,
        reset_timeout=reset_timeout,
        listeners=[_listener],
        name=name,
    )


groq_breaker = _breaker("groq")
openai_breaker = _breaker("openai")
anthropic_breaker = _breaker("anthropic")
qdrant_breaker = _breaker("qdrant", fail_max=3, reset_timeout=30)
