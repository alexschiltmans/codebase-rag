"""Password hashing, verification, and rehash-on-login policy."""

import hashlib
import hmac
import os

SALT_BYTES = 16
DEFAULT_ITERATIONS = 600_000
DIGEST_NAME = "sha256"


class PasswordTooShort(Exception):
    """Raised when a candidate password is below the minimum length."""


MINIMUM_LENGTH = 12


def hash_password(password, iterations=DEFAULT_ITERATIONS):
    """Derive a salted hash from a plaintext password.

    The salt is per-password and stored alongside the digest, which is what
    stops one precomputed table from covering every account: with a distinct
    salt, an attacker has to redo the work for each row.
    """
    if len(password) < MINIMUM_LENGTH:
        raise PasswordTooShort(f"password must be at least {MINIMUM_LENGTH} characters")
    salt = os.urandom(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(DIGEST_NAME, password.encode(), salt, iterations)
    return f"pbkdf2_{DIGEST_NAME}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password, encoded):
    """Check a plaintext password against a stored hash in constant time."""
    _, iterations, salt_hex, digest_hex = encoded.split("$")
    candidate = hashlib.pbkdf2_hmac(DIGEST_NAME, password.encode(), bytes.fromhex(salt_hex), int(iterations))
    return hmac.compare_digest(candidate.hex(), digest_hex)


def needs_rehash(encoded, iterations=DEFAULT_ITERATIONS):
    """Report whether a stored hash was derived with fewer iterations than current policy.

    Checked at login because that is the only moment the plaintext is available
    to derive a stronger hash from; a background job cannot upgrade a stored
    digest without it.
    """
    _, stored_iterations, _, _ = encoded.split("$")
    return int(stored_iterations) < iterations
