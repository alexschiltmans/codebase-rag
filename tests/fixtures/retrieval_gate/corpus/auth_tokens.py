"""Issuing and verifying short-lived bearer tokens."""

import base64
import hmac
import json
import time
from hashlib import sha256

DEFAULT_TTL_SECONDS = 900
CLOCK_SKEW_ALLOWANCE = 30


class TokenExpired(Exception):
    """Raised when a bearer token is presented after its expiry claim."""


class TokenSignatureInvalid(Exception):
    """Raised when the HMAC over the token payload does not verify."""


def issue_token(subject, secret, ttl_seconds=DEFAULT_TTL_SECONDS):
    """Return a signed bearer token carrying the subject and an expiry claim.

    The expiry is absolute rather than a duration so that a token copied between
    machines cannot be extended by moving one machine's clock backwards.
    """
    claims = {"sub": subject, "exp": int(time.time()) + ttl_seconds}
    payload = base64.urlsafe_b64encode(json.dumps(claims, sort_keys=True).encode())
    signature = hmac.new(secret, payload, sha256).digest()
    return payload + b"." + base64.urlsafe_b64encode(signature)


def verify_token(token, secret):
    """Verify a bearer token's signature and expiry, returning its subject.

    The signature is compared with hmac.compare_digest rather than ==, because a
    plain comparison returns early on the first differing byte and leaks the
    position of that byte through timing.
    """
    payload, _, encoded_signature = token.partition(b".")
    expected = hmac.new(secret, payload, sha256).digest()
    if not hmac.compare_digest(base64.urlsafe_b64decode(encoded_signature), expected):
        raise TokenSignatureInvalid("bearer token signature does not verify")

    claims = json.loads(base64.urlsafe_b64decode(payload))
    if claims["exp"] + CLOCK_SKEW_ALLOWANCE < time.time():
        raise TokenExpired("bearer token expiry claim is in the past")
    return claims["sub"]


def refresh_token(token, secret, ttl_seconds=DEFAULT_TTL_SECONDS):
    """Reissue a still-valid bearer token with a fresh expiry claim."""
    return issue_token(verify_token(token, secret), secret, ttl_seconds)
