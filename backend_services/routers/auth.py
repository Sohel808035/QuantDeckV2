"""
backend_services/routers/auth.py
─────────────────────────────────
Authentication, User Management & RBAC Router.
Provides JWT token generation, password verification, API key issuance, and identity inspection.
"""

from __future__ import annotations
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend_services.auth import create_access_token, verify_token_or_key
from db.database import get_db_connection

router = APIRouter(prefix="/auth", tags=["Authentication & Access Control"])


class LoginRequest(BaseModel):
    username: str = Field(..., example="admin")
    password: str = Field(..., example="admin123")


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int = 86400
    username: str
    role: str


class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: str


@router.post("/login", response_model=AuthTokenResponse, summary="User Login & JWT Token Generation")
async def login(request: LoginRequest) -> AuthTokenResponse:
    """Authenticates username/password and returns signed JWT access token."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (request.username,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    password_hash = hashlib.sha256(request.password.encode()).hexdigest()
    if user["hashed_password"] != password_hash:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    if not user["is_active"]:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is deactivated")

    token_data = {"sub": user["username"], "role": user["role"], "id": user["id"]}
    token = create_access_token(token_data, expires_delta=timedelta(days=1))

    return AuthTokenResponse(
        access_token=token,
        username=user["username"],
        role=user["role"],
    )


@router.get("/me", response_model=UserProfileResponse, summary="Get Current Authenticated User Profile")
async def get_current_user_profile(user_id: str = Depends(verify_token_or_key)) -> UserProfileResponse:
    """Returns profile for currently authenticated JWT token or API key."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ? OR id = ?", (user_id, user_id))
    user = cursor.fetchone()
    conn.close()

    if not user:
        # Fallback for API key / demo system identity
        return UserProfileResponse(
            id=1,
            username=user_id,
            email=f"{user_id}@quantspherex.com",
            role="admin",
            is_active=True,
            created_at=datetime.utcnow().isoformat(),
        )

    return UserProfileResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        role=user["role"],
        is_active=bool(user["is_active"]),
        created_at=user["created_at"],
    )
