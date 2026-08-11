"""Shared authenticated HTTP behavior for direct broker adapters."""

from __future__ import annotations

import json as json_module
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from artha_mcp.security import redact

from .base import AdapterError

MAX_BROKER_RESPONSE_BYTES = 5 * 1024 * 1024


class HTTPBrokerMixin:
    def __init__(
        self, *, client: httpx.AsyncClient | None = None, timeout: float = 15.0
    ) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: Any = None,
        json: Any = None,
        data: Any = None,
    ) -> Any:
        try:
            async with self._client.stream(
                method,
                url,
                headers=headers,
                params=params,
                json=json,
                data=data,
            ) as response:
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError:
                        declared_size = 0
                    if declared_size > MAX_BROKER_RESPONSE_BYTES:
                        raise AdapterError(
                            f"Broker response exceeded {MAX_BROKER_RESPONSE_BYTES} bytes"
                        )
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > MAX_BROKER_RESPONSE_BYTES:
                        raise AdapterError(
                            f"Broker response exceeded {MAX_BROKER_RESPONSE_BYTES} bytes"
                        )
                    body.extend(chunk)
                status_code = response.status_code
        except AdapterError:
            raise
        except httpx.HTTPError as exc:
            raise AdapterError(
                f"Broker network request failed: {type(exc).__name__}"
            ) from exc
        try:
            payload = json_module.loads(body)
        except (UnicodeDecodeError, ValueError) as exc:
            raise AdapterError(f"Broker returned non-JSON HTTP {status_code}") from exc
        if status_code < 200 or status_code >= 300:
            safe = redact(payload)
            raise AdapterError(f"Broker returned HTTP {status_code}: {str(safe)[:500]}")
        return payload

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def first_number(value: Any, keys: set[str]) -> float | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in keys:
                try:
                    return float(child)
                except (TypeError, ValueError):
                    pass
        for child in value.values():
            found = first_number(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = first_number(child, keys)
            if found is not None:
                return found
    return None


def first_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if "data" in value and isinstance(value["data"], dict):
            value = value["data"]
        if value and all(isinstance(v, dict) for v in value.values()):
            return dict(next(iter(value.values())))
        return dict(value)
    return {}


def parse_broker_timestamp(value: Any, *, default_timezone: str) -> datetime | None:
    """Parse ISO or epoch broker time and normalize it to UTC."""
    if value in (None, ""):
        return None
    raw = str(value).strip()
    try:
        if raw.replace(".", "", 1).isdigit():
            epoch = float(raw)
            if epoch > 10_000_000_000:
                epoch /= 1000.0
            return datetime.fromtimestamp(epoch, UTC)
        parsed = datetime.fromisoformat(raw)
    except (ValueError, OSError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(default_timezone))
    return parsed.astimezone(UTC)
