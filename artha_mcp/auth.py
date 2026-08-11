"""OAuth resource-server support for remote Streamable HTTP deployments."""

from __future__ import annotations

import asyncio
from typing import Any

import jwt
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings

from .settings import MCPSettings


class JWTTokenVerifier:
    """Validate issuer-, audience-, expiry-, and signature-bound JWTs."""

    def __init__(self, settings: MCPSettings) -> None:
        self.settings = settings
        self._jwks = jwt.PyJWKClient(
            settings.oauth_jwks_url,
            cache_keys=True,
            max_cached_keys=16,
            lifespan=300,
            timeout=5,
        )

    def _verify_sync(self, token: str) -> AccessToken | None:
        try:
            key = self._jwks.get_signing_key_from_jwt(token)
            claims: dict[str, Any] = jwt.decode(
                token,
                key.key,
                algorithms=list(self.settings.oauth_algorithms),
                audience=self.settings.oauth_audience,
                issuer=self.settings.oauth_issuer,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.PyJWTError:
            return None
        raw_scope = claims.get("scope") or claims.get("scp") or []
        if isinstance(raw_scope, str):
            scopes = raw_scope.split()
        elif isinstance(raw_scope, list):
            scopes = [str(value) for value in raw_scope]
        else:
            scopes = []
        client_id = str(
            claims.get("client_id") or claims.get("azp") or claims.get("sub") or ""
        )
        if not client_id:
            return None
        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(claims["exp"]),
            resource=self.settings.oauth_resource_url,
            subject=str(claims.get("sub") or ""),
            claims=claims,
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        return await asyncio.to_thread(self._verify_sync, token)


def build_auth(
    settings: MCPSettings,
) -> tuple[AuthSettings | None, JWTTokenVerifier | None]:
    if not settings.oauth_configured:
        return None, None
    auth = AuthSettings(
        issuer_url=settings.oauth_issuer,
        resource_server_url=settings.oauth_resource_url,
        required_scopes=["artha:read"],
    )
    return auth, JWTTokenVerifier(settings)
