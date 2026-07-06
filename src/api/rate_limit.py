"""Shared rate limiter -- one instance, imported by every route file and main.py.

Previously each route file (chat.py, ingest.py) constructed its own separate
Limiter(key_func=get_remote_address), and neither passed storage_uri -- so
the actual rate-limiting decisions ran against two independent in-memory
counters, never the Redis-backed instance main.py built and attached to
app.state.limiter. With multiple uvicorn workers, that meant each worker
enforced its own separate quota, silently multiplying the configured limit
and giving inconsistent results depending on which worker handled a request.
"""

from __future__ import annotations

from fastapi import Request
from jose import JWTError, jwt
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.config import get_settings


def rate_limit_key(request: Request) -> str:
    """Rate-limit key: restaurant_id from a valid JWT when present, else remote IP.

    Chat routes carry a per-restaurant JWT -- keying by restaurant_id means
    one restaurant's traffic can't exhaust another's quota, and a caller
    can't dodge its own limit by rotating source IPs, neither of which
    get_remote_address alone can provide. Auth/ingest routes only carry the
    shared API_KEY (no per-tenant identity in that token), so those still
    fall back to IP -- there's no better signal available there without a
    bigger auth-model change.
    """
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:]
        settings = get_settings()
        try:
            payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        except JWTError:
            payload = None
        if payload is not None:
            restaurant_id = payload.get("restaurant_id")
            if restaurant_id is not None:
                return f"restaurant:{restaurant_id}"
    return get_remote_address(request)


def _create_limiter() -> Limiter:
    settings = get_settings()
    return Limiter(
        key_func=rate_limit_key,
        storage_uri=settings.redis_url,
        # Rate limiting is a protective mechanism, not core functionality --
        # if Redis is briefly unreachable, every chat/ingest request failing
        # outright would be a worse outcome than temporarily skipping the
        # limit check. swallow_errors logs the failure and lets the request
        # through instead of raising.
        swallow_errors=True,
    )


limiter = _create_limiter()
