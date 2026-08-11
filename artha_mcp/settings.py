"""Environment-backed configuration for the public Artha MCP server."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from ipaddress import ip_address
from math import isfinite
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from artha.paths import DATA_DIR as ARTHA_DATA_DIR
from artha.paths import PROJECT_ROOT as ARTHA_PROJECT_ROOT

from .models import AccessMode, BrokerName, MarketCode

_ALGO_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(
        part.strip() for part in os.getenv(name, default).split(",") if part.strip()
    )


def _is_loopback_name(hostname: str | None) -> bool:
    value = str(hostname or "").strip().lower().strip("[]")
    if value == "localhost":
        return True
    try:
        return ip_address(value).is_loopback
    except ValueError:
        return False


def _secure_url(value: str, *, allow_loopback_http: bool) -> bool:
    parsed = urlparse(value)
    if not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        return False
    if parsed.scheme == "https":
        return True
    return bool(
        parsed.scheme == "http"
        and allow_loopback_http
        and _is_loopback_name(parsed.hostname)
    )


def _secure_origin(value: str, *, allow_loopback_http: bool) -> bool:
    parsed = urlparse(value)
    return bool(
        _secure_url(value, allow_loopback_http=allow_loopback_http)
        and parsed.path in {"", "/"}
        and not parsed.query
    )


def _broker_base_url(value: str) -> bool:
    """Direct broker adapters must never send credentials to an insecure URL."""
    parsed = urlparse(value)
    return bool(
        parsed.scheme == "https"
        and parsed.hostname
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
        and parsed.path in {"", "/"}
    )


def _allowed_host_name(value: str) -> str:
    raw = value.strip()
    if raw.startswith("[") and "]" in raw:
        return raw[1 : raw.index("]")]
    if raw.count(":") == 1:
        return raw.split(":", 1)[0]
    return raw


def _allowed_host_has_port(value: str) -> bool:
    raw = value.strip()
    if raw.startswith("[") and "]" in raw:
        return raw[raw.index("]") + 1 :].startswith(":")
    return raw.count(":") == 1


def _host_with_port(value: str, port: int) -> str:
    raw = value.strip()
    if ":" in raw and not raw.startswith("["):
        return f"[{raw}]:{port}"
    return f"{raw}:{port}"


@dataclass(frozen=True)
class MCPSettings:
    root: Path
    data_dir: Path
    db_path: Path
    portfolio_path: Path
    snapshot_path: Path
    transport: str
    host: str
    port: int
    http_path: str
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    access_mode: AccessMode
    operations_enabled: bool
    trading_enabled: bool
    kill_switch: bool
    market: MarketCode
    broker: BrokerName
    broker_plugin: str
    research_plugin: str
    notification: str
    research_mode: str
    max_order_value: float
    max_daily_order_value: float
    max_daily_orders: int
    max_spread_pct: float
    quote_max_age_seconds: int
    workflow_timeout_seconds: int
    max_pending_jobs: int
    oauth_issuer: str
    oauth_resource_url: str
    oauth_audience: str
    oauth_jwks_url: str
    oauth_algorithms: tuple[str, ...]
    upstox_base_url: str
    upstox_order_base_url: str
    upstox_access_token: str
    upstox_sandbox_access_token: str
    upstox_sandbox: bool
    upstox_algo_name: str
    zerodha_base_url: str
    zerodha_api_key: str
    zerodha_access_token: str
    india_static_ip_registered: bool
    india_demat_sell_authorized: bool

    @classmethod
    def from_env(cls, *, root: Path | None = None) -> MCPSettings:
        project_root = (
            (root or Path(os.getenv("ARTHA_MCP_ROOT", str(ARTHA_PROJECT_ROOT))))
            .expanduser()
            .resolve()
        )
        data_dir = (
            Path(os.getenv("ARTHA_MCP_DATA_DIR", str(ARTHA_DATA_DIR)))
            .expanduser()
            .resolve()
        )
        market_raw = os.getenv("ARTHA_MCP_MARKET", "US").strip().upper()
        broker_raw = os.getenv("ARTHA_MCP_BROKER", "snapshot").strip().lower()
        access_raw = os.getenv("ARTHA_MCP_ACCESS_MODE", "read_only").strip().lower()
        try:
            market = MarketCode(market_raw)
        except ValueError as exc:
            raise ValueError("ARTHA_MCP_MARKET must be US or IN") from exc
        try:
            broker = BrokerName(broker_raw)
        except ValueError as exc:
            raise ValueError(
                "ARTHA_MCP_BROKER must be none, snapshot, upstox, zerodha, or plugin"
            ) from exc
        try:
            access_mode = AccessMode(access_raw)
        except ValueError as exc:
            raise ValueError(
                "ARTHA_MCP_ACCESS_MODE must be read_only, operator, or trading"
            ) from exc

        operations_enabled = _bool("ARTHA_MCP_OPERATIONS_ENABLED", False)
        trading_enabled = _bool("ARTHA_MCP_TRADING_ENABLED", False)
        kill_switch = _bool("ARTHA_MCP_KILL_SWITCH", True)
        default_order_value = 25.0 if market == MarketCode.US else 2500.0
        default_daily_order_value = 50.0 if market == MarketCode.US else 5000.0
        return cls(
            root=project_root,
            data_dir=data_dir,
            db_path=Path(os.getenv("ARTHA_MCP_DB_PATH", str(data_dir / "artha.db")))
            .expanduser()
            .resolve(),
            portfolio_path=Path(
                os.getenv("ARTHA_MCP_PORTFOLIO_PATH", str(data_dir / "portfolio.json"))
            )
            .expanduser()
            .resolve(),
            snapshot_path=Path(
                os.getenv(
                    "ARTHA_MCP_BROKER_SNAPSHOT_PATH",
                    str(data_dir / "robinhood" / "latest_snapshot.json"),
                )
            )
            .expanduser()
            .resolve(),
            transport=os.getenv("ARTHA_MCP_TRANSPORT", "stdio").strip().lower(),
            host=os.getenv("ARTHA_MCP_HOST", "127.0.0.1").strip(),
            port=_int("ARTHA_MCP_PORT", 8765),
            http_path=os.getenv("ARTHA_MCP_HTTP_PATH", "/mcp").strip() or "/mcp",
            allowed_hosts=_csv("ARTHA_MCP_ALLOWED_HOSTS", "127.0.0.1,localhost"),
            allowed_origins=_csv("ARTHA_MCP_ALLOWED_ORIGINS"),
            access_mode=access_mode,
            operations_enabled=operations_enabled,
            trading_enabled=trading_enabled,
            kill_switch=kill_switch,
            market=market,
            broker=broker,
            broker_plugin=os.getenv("ARTHA_MCP_BROKER_PLUGIN", "").strip(),
            research_plugin=os.getenv("ARTHA_MCP_RESEARCH_PLUGIN", "").strip(),
            notification=os.getenv("ARTHA_MCP_NOTIFICATION", "telegram")
            .strip()
            .lower(),
            research_mode=os.getenv("ARTHA_MCP_RESEARCH_MODE", "embedded")
            .strip()
            .lower(),
            max_order_value=_float("ARTHA_MCP_MAX_ORDER_VALUE", default_order_value),
            max_daily_order_value=_float(
                "ARTHA_MCP_MAX_DAILY_ORDER_VALUE", default_daily_order_value
            ),
            max_daily_orders=_int("ARTHA_MCP_MAX_DAILY_ORDERS", 2),
            max_spread_pct=_float("ARTHA_MCP_MAX_SPREAD_PCT", 0.01),
            quote_max_age_seconds=_int("ARTHA_MCP_QUOTE_MAX_AGE_SECONDS", 30),
            workflow_timeout_seconds=_int("ARTHA_MCP_WORKFLOW_TIMEOUT_SECONDS", 10800),
            max_pending_jobs=_int("ARTHA_MCP_MAX_PENDING_JOBS", 8),
            oauth_issuer=os.getenv("ARTHA_MCP_OAUTH_ISSUER", "").strip(),
            oauth_resource_url=os.getenv("ARTHA_MCP_OAUTH_RESOURCE_URL", "").strip(),
            oauth_audience=os.getenv("ARTHA_MCP_OAUTH_AUDIENCE", "").strip(),
            oauth_jwks_url=os.getenv("ARTHA_MCP_OAUTH_JWKS_URL", "").strip(),
            oauth_algorithms=_csv("ARTHA_MCP_OAUTH_ALGORITHMS", "RS256,ES256"),
            upstox_base_url=os.getenv(
                "ARTHA_UPSTOX_BASE_URL", "https://api.upstox.com"
            ).rstrip("/"),
            upstox_order_base_url=os.getenv(
                "ARTHA_UPSTOX_ORDER_BASE_URL", "https://api-hft.upstox.com"
            ).rstrip("/"),
            upstox_access_token=os.getenv("UPSTOX_ACCESS_TOKEN", "").strip(),
            upstox_sandbox_access_token=os.getenv(
                "UPSTOX_SANDBOX_ACCESS_TOKEN", ""
            ).strip(),
            upstox_sandbox=_bool("ARTHA_UPSTOX_SANDBOX", False),
            upstox_algo_name=os.getenv("ARTHA_UPSTOX_ALGO_NAME", "").strip(),
            zerodha_base_url=os.getenv(
                "ARTHA_ZERODHA_BASE_URL", "https://api.kite.trade"
            ).rstrip("/"),
            zerodha_api_key=os.getenv("KITE_API_KEY", "").strip(),
            zerodha_access_token=os.getenv("KITE_ACCESS_TOKEN", "").strip(),
            india_static_ip_registered=_bool("ARTHA_INDIA_STATIC_IP_REGISTERED", False),
            india_demat_sell_authorized=_bool(
                "ARTHA_INDIA_DEMAT_SELL_AUTHORIZED", False
            ),
        )

    @property
    def oauth_configured(self) -> bool:
        return all(
            (
                self.oauth_issuer,
                self.oauth_resource_url,
                self.oauth_audience,
                self.oauth_jwks_url,
            )
        )

    @property
    def is_loopback(self) -> bool:
        return _is_loopback_name(self.host)

    @property
    def effective_allowed_hosts(self) -> tuple[str, ...]:
        """Expand explicit hosts with the configured bind port, without wildcards."""
        values: list[str] = []
        for allowed_host in self.allowed_hosts:
            if allowed_host not in values:
                values.append(allowed_host)
            if not _allowed_host_has_port(allowed_host):
                with_port = _host_with_port(allowed_host, self.port)
                if with_port not in values:
                    values.append(with_port)
        return tuple(values)

    def startup_findings(self) -> dict[str, list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        if self.transport not in {"stdio", "streamable-http"}:
            errors.append(
                "Transport must be stdio or streamable-http; legacy SSE is not supported."
            )
        if not self.http_path.startswith("/"):
            errors.append("ARTHA_MCP_HTTP_PATH must begin with '/'.")
        if not 1 <= self.port <= 65535:
            errors.append("ARTHA_MCP_PORT must be between 1 and 65535.")
        if self.transport == "streamable-http" and not self.allowed_hosts:
            errors.append(
                "Streamable HTTP requires at least one explicit allowed host."
            )
        for allowed_host in self.allowed_hosts:
            if "*" in allowed_host or "://" in allowed_host or "/" in allowed_host:
                errors.append(
                    "Allowed hosts must be explicit host names (optionally with a port), without wildcards or URLs."
                )
                break
        for allowed_origin in self.allowed_origins:
            if allowed_origin == "*" or not _secure_origin(
                allowed_origin, allow_loopback_http=self.is_loopback
            ):
                errors.append(
                    "Allowed origins must be explicit HTTPS origins; loopback HTTP is allowed only locally."
                )
                break
        oauth_values = (
            self.oauth_issuer,
            self.oauth_resource_url,
            self.oauth_audience,
            self.oauth_jwks_url,
        )
        if any(oauth_values) and not all(oauth_values):
            errors.append(
                "OAuth issuer, resource URL, audience, and JWKS URL must be configured together."
            )
        if (
            self.transport == "streamable-http"
            and not self.is_loopback
            and not self.oauth_configured
        ):
            errors.append(
                "Non-loopback HTTP requires OAuth issuer, resource, audience, and JWKS configuration."
            )
        if (
            self.transport == "streamable-http"
            and not self.is_loopback
            and self.allowed_hosts
            and all(
                _is_loopback_name(_allowed_host_name(value))
                for value in self.allowed_hosts
            )
        ):
            errors.append(
                "Non-loopback HTTP requires the externally requested host in ARTHA_MCP_ALLOWED_HOSTS."
            )
        if self.oauth_configured:
            for name, value in (
                ("issuer", self.oauth_issuer),
                ("resource", self.oauth_resource_url),
                ("JWKS", self.oauth_jwks_url),
            ):
                if not _secure_url(value, allow_loopback_http=self.is_loopback):
                    errors.append(
                        f"OAuth {name} must be an HTTPS URL without credentials or fragments."
                    )
            if not self.oauth_algorithms or any(
                not algorithm.startswith(("RS", "PS", "ES")) and algorithm != "EdDSA"
                for algorithm in self.oauth_algorithms
            ):
                errors.append(
                    "OAuth algorithms must use asymmetric RS, PS, ES, or EdDSA signatures."
                )
        if self.access_mode == AccessMode.READ_ONLY and (
            self.operations_enabled or self.trading_enabled
        ):
            errors.append("read_only access cannot enable operations or trading.")
        if self.access_mode == AccessMode.OPERATOR and self.trading_enabled:
            errors.append("operator access cannot enable trading.")
        if self.trading_enabled and (
            self.access_mode != AccessMode.TRADING or not self.operations_enabled
        ):
            errors.append(
                "Trading requires access_mode=trading and operations_enabled=true."
            )
        if self.trading_enabled and self.kill_switch:
            warnings.append(
                "Trading is configured but the MCP kill switch is engaged; placement remains blocked."
            )
        if (
            not isfinite(self.max_order_value)
            or not isfinite(self.max_daily_order_value)
            or self.max_order_value <= 0
            or self.max_daily_order_value <= 0
            or self.max_daily_orders <= 0
        ):
            errors.append("Order and daily trading limits must all be positive.")
        if not isfinite(self.max_spread_pct) or not 0 < self.max_spread_pct <= 0.10:
            errors.append(
                "ARTHA_MCP_MAX_SPREAD_PCT must be above 0 and no greater than 0.10."
            )
        if self.quote_max_age_seconds <= 0:
            errors.append("ARTHA_MCP_QUOTE_MAX_AGE_SECONDS must be positive.")
        if self.workflow_timeout_seconds < 30:
            errors.append(
                "ARTHA_MCP_WORKFLOW_TIMEOUT_SECONDS must be at least 30 seconds."
            )
        if not 1 <= self.max_pending_jobs <= 64:
            errors.append("ARTHA_MCP_MAX_PENDING_JOBS must be between 1 and 64.")
        if self.notification not in {"none", "telegram", "webhook"}:
            errors.append("ARTHA_MCP_NOTIFICATION must be none, telegram, or webhook.")
        if self.research_mode not in {"embedded", "host_orchestrated", "plugin"}:
            errors.append(
                "ARTHA_MCP_RESEARCH_MODE must be embedded, host_orchestrated, or plugin."
            )
        if self.broker == BrokerName.SNAPSHOT and self.market != MarketCode.US:
            errors.append("The Robinhood snapshot adapter supports US equities only.")
        if (
            self.broker in {BrokerName.UPSTOX, BrokerName.ZERODHA}
            and self.market != MarketCode.INDIA
        ):
            errors.append(
                f"The {self.broker.value} adapter supports Indian equities only."
            )
        if self.market == MarketCode.INDIA and self.broker not in {
            BrokerName.UPSTOX,
            BrokerName.ZERODHA,
            BrokerName.PLUGIN,
        }:
            warnings.append(
                "India mode has no India-capable broker adapter configured."
            )
        if self.market == MarketCode.INDIA and self.research_mode == "embedded":
            warnings.append(
                "Built-in Artha Council workflows are US-native and remain blocked in India mode."
            )
        if self.research_mode == "plugin" and ":" not in self.research_plugin:
            errors.append(
                "ARTHA_MCP_RESEARCH_PLUGIN must use module:factory syntax in plugin mode."
            )
        if self.broker == BrokerName.UPSTOX and not self.upstox_access_token:
            message = "UPSTOX_ACCESS_TOKEN is missing; Upstox account, quote, and preview calls will fail closed."
            if self.trading_enabled:
                errors.append(message)
            else:
                warnings.append(message)
        if (
            self.broker == BrokerName.UPSTOX
            and self.upstox_sandbox
            and not self.upstox_sandbox_access_token
        ):
            message = (
                "UPSTOX_SANDBOX_ACCESS_TOKEN is required for sandbox placement; "
                "Upstox sandbox tokens are separate from live account/read tokens."
            )
            if self.trading_enabled:
                errors.append(message)
            else:
                warnings.append(message)
        if self.broker == BrokerName.UPSTOX and not all(
            _broker_base_url(value)
            for value in (self.upstox_base_url, self.upstox_order_base_url)
        ):
            errors.append(
                "Upstox base URLs must be HTTPS origins without credentials, paths, queries, or fragments."
            )
        if self.upstox_algo_name and not _ALGO_NAME.fullmatch(self.upstox_algo_name):
            errors.append(
                "ARTHA_UPSTOX_ALGO_NAME must be 1-64 characters using letters, numbers, spaces, '.', '_', or '-'."
            )
        if self.broker == BrokerName.ZERODHA and not (
            self.zerodha_api_key and self.zerodha_access_token
        ):
            message = "KITE_API_KEY/KITE_ACCESS_TOKEN are missing; Zerodha calls will fail closed."
            if self.trading_enabled:
                errors.append(message)
            else:
                warnings.append(message)
        if self.broker == BrokerName.ZERODHA and not _broker_base_url(
            self.zerodha_base_url
        ):
            errors.append(
                "Zerodha base URL must be an HTTPS origin without credentials, paths, queries, or fragments."
            )
        if self.broker == BrokerName.PLUGIN and ":" not in self.broker_plugin:
            errors.append("ARTHA_MCP_BROKER_PLUGIN must use module:factory syntax.")
        if (
            self.market == MarketCode.INDIA
            and self.broker in {BrokerName.UPSTOX, BrokerName.ZERODHA}
            and not (self.broker == BrokerName.UPSTOX and self.upstox_sandbox)
            and not self.india_static_ip_registered
        ):
            message = (
                "Indian broker API order placement requires a static IP registered with the broker; "
                "set ARTHA_INDIA_STATIC_IP_REGISTERED=true only after registration is complete."
            )
            if self.trading_enabled:
                errors.append(message)
            else:
                warnings.append(message)
        if (
            self.market == MarketCode.INDIA
            and self.broker in {BrokerName.UPSTOX, BrokerName.ZERODHA}
            and not self.india_demat_sell_authorized
        ):
            warnings.append(
                "Automated Indian delivery sells remain blocked until durable DDPI/POA or a current depository authorization is explicitly attested with ARTHA_INDIA_DEMAT_SELL_AUTHORIZED=true."
            )
        return {"errors": errors, "warnings": warnings}

    def public_summary(self) -> dict[str, Any]:
        findings = self.startup_findings()
        return {
            "market": self.market.value,
            "broker": self.broker.value,
            "broker_plugin_configured": bool(self.broker_plugin),
            "notification": self.notification,
            "research_mode": self.research_mode,
            "research_plugin_configured": bool(self.research_plugin),
            "transport": self.transport,
            "http": {
                "host": self.host,
                "port": self.port,
                "path": self.http_path,
                "allowed_hosts": list(self.effective_allowed_hosts),
                "oauth_configured": self.oauth_configured,
            },
            "access": {
                "mode": self.access_mode.value,
                "operations_enabled": self.operations_enabled,
                "trading_enabled": self.trading_enabled,
                "kill_switch": self.kill_switch,
            },
            "limits": {
                "max_order_value": self.max_order_value,
                "max_daily_order_value": self.max_daily_order_value,
                "max_daily_orders": self.max_daily_orders,
                "max_spread_pct": self.max_spread_pct,
                "quote_max_age_seconds": self.quote_max_age_seconds,
                "max_pending_jobs": self.max_pending_jobs,
            },
            "credentials_present": {
                "fmp": bool(os.getenv("FMP_API_KEY")),
                "finnhub": bool(os.getenv("FINNHUB_API_KEY")),
                "gemini": bool(
                    os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                ),
                "telegram": bool(
                    os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")
                ),
                "upstox": bool(self.upstox_access_token),
                "upstox_sandbox": bool(self.upstox_sandbox_access_token),
                "zerodha": bool(self.zerodha_api_key and self.zerodha_access_token),
            },
            "india_api_compliance": {
                "static_ip_registered_attestation": self.india_static_ip_registered,
                "demat_sell_authorized_attestation": self.india_demat_sell_authorized,
                "upstox_algo_name_configured": bool(self.upstox_algo_name),
                "order_policy": "whole-share cash-delivery limit orders only",
            },
            "findings": findings,
        }
