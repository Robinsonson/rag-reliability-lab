"""Password hashing and opaque server-side sessions; no browser role claims."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from collections import defaultdict, deque
from threading import Lock


def hash_password(password: str) -> str:
    if not isinstance(password, str) or not 12 <= len(password) <= 256:
        raise ValueError("Password must contain 12–256 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 600_000)
    return "pbkdf2_sha256$600000$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, rounds, salt, expected = encoded.split("$")
        if algorithm != "pbkdf2_sha256" or not 1 <= int(rounds) <= 1_000_000 or len(password) > 256:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), base64.b64decode(salt), int(rounds))
        return hmac.compare_digest(actual, base64.b64decode(expected))
    except (ValueError, TypeError):
        return False


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def csrf_token(token: str) -> str:
    return hashlib.sha256(("policy-csrf:" + token).encode()).hexdigest()


def public_user(user: dict) -> dict:
    return {key: user.get(key) for key in ("id", "username", "role", "scopes", "active")}


class LoginLimiter:
    """Bounded local login limiter for the single API process Compose profile."""
    def __init__(self):
        self._entries = defaultdict(deque)
        self._lock = Lock()

    def attempt(self, identity: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if len(self._entries) > 10000:
                self._entries = defaultdict(deque, {k: v for k, v in self._entries.items() if v and v[-1] > now - 300})
            attempts = self._entries[identity]
            while attempts and attempts[0] < now - 300:
                attempts.popleft()
            if len(attempts) >= 10:
                return False
            attempts.append(now)
            return True
