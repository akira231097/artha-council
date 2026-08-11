"""Authorization, redaction, and fail-closed policy for MCP operations."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import AccessMode
from .settings import MCPSettings

READ_SCOPE = "artha:read"
OPERATE_SCOPE = "artha:operate"
TRADE_SCOPE = "artha:trade"

_SENSITIVE_KEY = re.compile(
    r"(access[_-]?token|refresh[_-]?token|api[_-]?key|secret|password|authorization|cookie|callback[_-]?token)",
    re.IGNORECASE,
)
_ACCOUNT_KEY = re.compile(
    r"(account[_-]?(number|id)|client[_-]?id|user[_-]?id)", re.IGNORECASE
)
_INLINE_SECRET = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password|authorization)=)[^&\s]+"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")

ROBINHOOD_ACCOUNT_PLACEHOLDER = "${ARTHA_RESOLVED_ROBINHOOD_ACCOUNT_NUMBER}"


class AuthorizationError(PermissionError):
    pass


def mask_identifier(value: Any) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    if len(raw) <= 4:
        return "****"
    return f"****{raw[-4:]}"


def redact(value: Any, *, key: str = "") -> Any:
    """Recursively remove credentials and mask account identifiers."""
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]" if value not in (None, "") else value
    if _ACCOUNT_KEY.search(key):
        return mask_identifier(value)
    if isinstance(value, Mapping):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        cleaned = _INLINE_SECRET.sub(r"\1[REDACTED]", value)
        cleaned = _BEARER.sub("Bearer [REDACTED]", cleaned)
        home = str(Path.home())
        if home:
            cleaned = cleaned.replace(f"{home}/", "~/")
            if cleaned == home:
                cleaned = "~"
        return cleaned
    return value


def robinhood_host_operation_contract(
    operation: Mapping[str, Any],
    *,
    account_number: str,
    expected_account_type: str = "",
    expected_account_nickname: str = "",
) -> dict[str, Any]:
    """Redact a Robinhood plan while leaving it executable by an MCP host.

    The public server must not return a full brokerage account number to a host
    model. A masked value is also not a valid Robinhood tool argument, so every
    exact ``account_number`` argument is replaced with an explicit placeholder.
    The host resolves that placeholder from Robinhood's own ``get_accounts``
    result and must fail closed unless exactly one allowlisted account matches.
    """

    safe = redact(operation)
    if not isinstance(safe, dict):
        return {
            "success": False,
            "operation": "blocked",
            "message": "Invalid operation contract.",
        }

    replaced = 0

    def bind(value: Any) -> Any:
        nonlocal replaced
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for raw_key, child in value.items():
                child_key = str(raw_key)
                if child_key.lower() in {"account_number", "rhs_account_number"}:
                    result[child_key] = ROBINHOOD_ACCOUNT_PLACEHOLDER
                    replaced += 1
                else:
                    result[child_key] = bind(child)
            return result
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [bind(item) for item in value]
        return value

    bound = bind(safe)
    if not isinstance(bound, dict):
        return {
            "success": False,
            "operation": "blocked",
            "message": "Invalid operation contract.",
        }
    if replaced:
        bound["contract_version"] = "artha.robinhood.host-operation.v1"
        bound["account_binding"] = {
            "placeholder": ROBINHOOD_ACCOUNT_PLACEHOLDER,
            "configured_account_masked": mask_identifier(account_number),
            "resolution_tool": "get_accounts",
            "selection_requirements": [
                "Call Robinhood get_accounts immediately before broker operations.",
                "Select exactly one active, non-deactivated, agentic-allowed account whose number ends with the configured masked suffix.",
                "Require the configured account type and nickname when those values are present below.",
                "Replace only the exact placeholder values in broker tool arguments; do not alter any order field.",
                "Abort if no account matches, multiple accounts match, or any account property is unknown.",
            ],
            "expected_account_type": expected_account_type or None,
            "expected_account_nickname": expected_account_nickname or None,
            "resolved_account_exposed_by_artha": False,
        }
    return bound


class CapabilityPolicy:
    """Local configuration is a ceiling; OAuth scopes can only narrow it."""

    def __init__(self, settings: MCPSettings) -> None:
        self.settings = settings

    @staticmethod
    def _scope_allows(required: str, scopes: set[str]) -> bool:
        if not scopes:
            return False
        if required == READ_SCOPE:
            return bool(scopes & {READ_SCOPE, OPERATE_SCOPE, TRADE_SCOPE})
        if required == OPERATE_SCOPE:
            return bool(scopes & {OPERATE_SCOPE, TRADE_SCOPE})
        return TRADE_SCOPE in scopes

    def require(self, required: str, *, oauth_scopes: set[str] | None = None) -> None:
        mode = self.settings.access_mode
        if required == READ_SCOPE:
            local_allowed = True
        elif required == OPERATE_SCOPE:
            local_allowed = (
                mode in {AccessMode.OPERATOR, AccessMode.TRADING}
                and self.settings.operations_enabled
            )
        elif required == TRADE_SCOPE:
            local_allowed = (
                mode == AccessMode.TRADING
                and self.settings.operations_enabled
                and self.settings.trading_enabled
                and not self.settings.kill_switch
            )
        else:
            raise AuthorizationError(f"Unknown capability scope: {required}")
        if not local_allowed:
            raise AuthorizationError(f"Local MCP policy does not permit {required}.")
        if oauth_scopes is not None and not self._scope_allows(required, oauth_scopes):
            raise AuthorizationError(f"OAuth token is missing scope {required}.")
