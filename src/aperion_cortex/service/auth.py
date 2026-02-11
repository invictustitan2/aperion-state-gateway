"""Scoped token validation for the Cortex service."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import CortexConfig, ScopedToken


@dataclass(slots=True)
class TokenInfo:
    """Validated token information."""

    token: str
    scopes: Sequence[str]
    expires_at: datetime | None

    def has_scope(self, scope: str) -> bool:
        """Check if token has the required scope."""
        return scope in self.scopes or "*" in self.scopes

    def is_expired(self, now: datetime) -> bool:
        """Check if token has expired."""
        if self.expires_at is None:
            return False
        return now >= self.expires_at


class ScopedTokenManager:
    """In-memory token registry with scope enforcement.

    Provides token validation with:
    - Scope-based access control
    - Token expiration checking
    - Wildcard scope support ("*" grants all scopes)
    """

    def __init__(self, config: CortexConfig):
        self._tokens: dict[str, TokenInfo] = {
            token.token: TokenInfo(
                token=token.token,
                scopes=tuple(token.scopes) or tuple(config.default_scopes),
                expires_at=token.expires_at,
            )
            for token in config.tokens
        }

    def register(self, token: ScopedToken) -> None:
        """Register a new token."""
        self._tokens[token.token] = TokenInfo(
            token=token.token,
            scopes=tuple(token.scopes),
            expires_at=token.expires_at,
        )

    def revoke(self, token: str) -> bool:
        """Revoke a token. Returns True if token existed."""
        return self._tokens.pop(token, None) is not None

    def validate(
        self,
        token: str,
        required_scope: str,
        now: datetime | None = None,
    ) -> TokenInfo:
        """Validate a token against required scope.

        Args:
            token: The bearer token to validate
            required_scope: The scope required for the operation
            now: Current time (for testing)

        Returns:
            TokenInfo if valid

        Raises:
            HTTPException: 401 if invalid/expired, 403 if insufficient scope
        """
        now = now or datetime.now(timezone.utc)
        info = self._tokens.get(token)

        if info is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )

        if info.is_expired(now):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
            )

        if not info.has_scope(required_scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient scope: requires '{required_scope}'",
            )

        return info

    def all_tokens(self) -> list[TokenInfo]:
        """Return all registered tokens (for admin/debugging)."""
        return list(self._tokens.values())


# Module-level token manager storage (avoid global keyword for ruff)
_TOKEN_MANAGER_SLOT: dict[str, ScopedTokenManager | None] = {"tm": None}

bearer_scheme = HTTPBearer(auto_error=False)


def set_token_manager(manager: ScopedTokenManager) -> None:
    """Set the global token manager instance."""
    _TOKEN_MANAGER_SLOT["tm"] = manager


def get_token_manager_dependency() -> ScopedTokenManager:
    """FastAPI dependency to get the token manager."""
    tm = _TOKEN_MANAGER_SLOT.get("tm")
    if tm is None:
        raise RuntimeError("Token manager dependency not configured")
    return tm


def make_scope_dependency(required_scope: str):
    """Create a FastAPI dependency that validates bearer token.

    Supports both:
    - Authorization: Bearer <token> header
    - ?token=<token> query parameter (for SSE/EventSource)

    Args:
        required_scope: The scope required for this endpoint

    Returns:
        A FastAPI dependency function
    """

    def dependency(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
        token_manager: ScopedTokenManager = Depends(get_token_manager_dependency),
        query_token: str | None = Query(default=None, alias="token"),
    ) -> TokenInfo:
        # Try header first, then query param
        raw_token: str | None = None
        if credentials is not None and credentials.credentials:
            raw_token = credentials.credentials
        elif query_token:
            raw_token = query_token

        if not raw_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing bearer token",
            )

        return token_manager.validate(raw_token, required_scope)

    return dependency
