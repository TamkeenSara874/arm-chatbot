"""Per-restaurant credential generation and verification.

Keys are high-entropy random tokens (32 bytes via secrets.token_urlsafe), not
user-chosen passwords -- a fast, constant-time SHA-256 comparison is the
right tool here, not a deliberately-slow password hash (bcrypt/argon2 exist
to resist brute-forcing a low-entropy secret; these keys have no brute-force
risk at 256 bits of entropy, and paying bcrypt's per-call cost on every chat
request's auth exchange would be wasted work).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets


def generate_restaurant_key() -> str:
    return secrets.token_urlsafe(32)


def hash_restaurant_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def verify_restaurant_key(key: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_restaurant_key(key), stored_hash)
