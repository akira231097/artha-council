"""Broker-neutral adapter interfaces."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

from artha_mcp.models import (
    BrokerCapabilities,
    InstrumentRef,
    OrderPreview,
    OrderRequest,
    OrderResult,
    PortfolioSnapshot,
    Quote,
)


class AdapterError(RuntimeError):
    """A provider or broker operation failed without exposing credentials."""


class CapabilityUnavailable(AdapterError):
    """The configured adapter does not implement the requested operation."""


def deterministic_order_tag(value: str, *, max_length: int) -> str:
    """Build a broker-safe alphanumeric tag with collision-resistant suffix."""
    raw = str(value or "")
    if max_length < 12:
        raise ValueError("Broker order-tag limit is too short")
    clean = "".join(character for character in raw if character.isalnum())
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    prefix = (clean or "artha")[: max_length - len(digest)]
    return f"{prefix}{digest}"


class BrokerAdapter(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> BrokerCapabilities:
        raise NotImplementedError

    async def health(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "broker": self.capabilities.name,
            "capabilities": self.capabilities.model_dump(mode="json"),
        }

    async def search_instruments(
        self, query: str, *, exchange: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        raise CapabilityUnavailable(
            f"{self.capabilities.name} does not provide instrument search"
        )

    @abstractmethod
    async def portfolio(self) -> PortfolioSnapshot:
        raise NotImplementedError

    @abstractmethod
    async def quote(self, instrument: InstrumentRef) -> Quote:
        raise NotImplementedError

    @abstractmethod
    async def preview(self, order: OrderRequest) -> OrderPreview:
        raise NotImplementedError

    @abstractmethod
    async def place(self, order: OrderRequest) -> OrderResult:
        raise NotImplementedError

    async def orders(self) -> list[dict[str, Any]]:
        raise CapabilityUnavailable(
            f"{self.capabilities.name} does not provide order status"
        )

    def client_order_key(self, order: OrderRequest) -> str:
        return str(order.tag or order.action_id)

    async def find_existing_order(self, order: OrderRequest) -> dict[str, Any] | None:
        """Find a broker order tagged with this immutable action id.

        Live adapters should preserve this key in the broker order tag. This
        check protects against duplicate placement after a process crash or an
        ambiguous prior network response.
        """
        wanted = self.client_order_key(order)
        for row in await self.orders():
            if not isinstance(row, dict):
                continue
            values = {
                str(row.get(name) or "")
                for name in ("tag", "order_tag", "client_order_id", "client_order_tag")
            }
            if wanted in values:
                return row
        return None

    async def close(self) -> None:
        return None
