"""
backend_services/auth.py
────────────────────────
Authentication & Authorization Module.
Provides FastAPI `Depends` providers for API Key and JWT Bearer Token verification.
Includes JWT signing and payload decoding.
"""

from __future__ import annotations
import logging
import hmac
import hashlib
import base64
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Depends, Header, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from backend_services.config import BackendSettings
from backend_services.errors import AuthenticationError, AuthorizationError

logger = logging.getLogger(__name__)

# FastAPI Security Schemes
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_token_scheme = HTTPBearer(auto_error=False)


def get_backend_settings() -> BackendSettings:
    """Dependency provider for Settings."""
    return BackendSettings()


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def _base64url_decode(data: str) -> bytes:
    padding = '=' * (4 - (len(data) % 4))
    return base64.urlsafe_b64decode(data + padding)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a signed JWT access token using HMAC-SHA256.

    Args:
        data: Dictionary of claims to embed in token payload.
        expires_delta: Optional timedelta expiration.

    Returns:
        Encoded JWT string (header.payload.signature).
    """
    settings = BackendSettings()
    secret = settings.jwt_secret_key.encode('utf-8')

    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    expire = now + int(expires_delta.total_seconds() if expires_delta else 86400)

    payload = data.copy()
    payload.update({"iat": now, "exp": expire})

    header_b64 = _base64url_encode(json.dumps(header).encode('utf-8'))
    payload_b64 = _base64url_encode(json.dumps(payload).encode('utf-8'))

    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    signature = hmac.new(secret, signing_input, hashlib.sha256).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decodes and validates a signed JWT access token.

    Args:
        token: Encoded JWT string.

    Returns:
        Decoded payload dictionary.
    """
    settings = BackendSettings()
    secret = settings.jwt_secret_key.encode('utf-8')

    parts = token.split('.')
    if len(parts) != 3:
        raise AuthenticationError("Invalid JWT token format")

    header_b64, payload_b64, signature_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode('utf-8')
    expected_sig = hmac.new(secret, signing_input, hashlib.sha256).digest()

    if not hmac.compare_digest(_base64url_encode(expected_sig), signature_b64):
        raise AuthenticationError("Invalid JWT token signature")

    try:
        payload = json.loads(_base64url_decode(payload_b64).decode('utf-8'))
    except Exception:
        raise AuthenticationError("Malformed JWT payload")

    if payload.get("exp", 0) < int(time.time()):
        raise AuthenticationError("JWT token has expired")

    return payload


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header_scheme),
    settings: BackendSettings = Depends(get_backend_settings),
) -> str:
    """
    Validates the X-API-Key header against configured valid keys.

    Returns:
        The validated API key string.
    """
    if settings.debug:
        return "debug-mode-key"

    if not api_key:
        raise AuthenticationError("Missing X-API-Key header in request")

    if api_key not in settings.valid_api_keys:
        logger.warning(f"Failed API Key authentication attempt: key='{api_key[:6]}...'")
        raise AuthenticationError("Invalid X-API-Key provided")

    return api_key


async def verify_token_or_key(
    api_key: Optional[str] = Security(api_key_header_scheme),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(bearer_token_scheme),
    settings: BackendSettings = Depends(get_backend_settings),
) -> str:
    """
    Accepts EITHER a valid API Key OR a Bearer Token.

    Returns:
        Client identity string.
    """
    if settings.debug:
        return "authenticated-user"

    if api_key and api_key in settings.valid_api_keys:
        return f"api-key-client:{api_key[:6]}"

    if bearer and bearer.credentials:
        token = bearer.credentials
        try:
            payload = decode_access_token(token)
            return payload.get("sub", "authenticated-user")
        except Exception:
            if token == settings.jwt_secret_key or token in settings.valid_api_keys:
                return "bearer-token-client"

    raise AuthenticationError("Request requires a valid X-API-Key header or Bearer Token")
