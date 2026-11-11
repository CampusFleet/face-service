"""
Service-to-service HMAC authentication.

Caller sends:
  X-CF-Timestamp: unix seconds
  X-CF-Signature: hex(HMAC-SHA256(FACE_SERVICE_KEY, f"{ts}\n{METHOD}\n{path}\n{sha256_hex(body)}"))
Requests older/newer than 60 s are rejected; signatures are compared in constant time.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time

MAX_SKEW_SECONDS = 60


def get_key() -> bytes:
    key = os.getenv("FACE_SERVICE_KEY", "")
    if len(key) < 32:
        raise RuntimeError("FACE_SERVICE_KEY must be set (32+ characters)")
    return key.encode()


def sign(key: bytes, ts: str, method: str, path: str, body: bytes) -> str:
    msg = f"{ts}\n{method.upper()}\n{path}\n{hashlib.sha256(body).hexdigest()}".encode()
    return hmac.new(key, msg, hashlib.sha256).hexdigest()


def verify(key: bytes, ts: str | None, sig: str | None, method: str, path: str, body: bytes, now: float | None = None) -> str | None:
    """Returns an error string, or None when valid."""
    if not ts or not sig:
        return "missing signature headers"
    try:
        t = int(ts)
    except ValueError:
        return "bad timestamp"
    if abs((now or time.time()) - t) > MAX_SKEW_SECONDS:
        return "timestamp outside allowed window"
    expected = sign(key, ts, method, path, body)
    if not hmac.compare_digest(expected, sig):
        return "bad signature"
    return None
