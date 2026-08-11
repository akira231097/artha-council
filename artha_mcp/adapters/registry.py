"""Construct built-in or user-supplied broker adapters."""

from __future__ import annotations

import importlib

from artha_mcp.models import BrokerName
from artha_mcp.settings import MCPSettings

from .base import BrokerAdapter, CapabilityUnavailable
from .snapshot import SnapshotBrokerAdapter
from .upstox import UpstoxBrokerAdapter
from .zerodha import ZerodhaBrokerAdapter


class NullBrokerAdapter(BrokerAdapter):
    @property
    def capabilities(self):
        from artha_mcp.models import BrokerCapabilities

        return BrokerCapabilities(
            name="none", market="US", notes=["No broker adapter is configured."]
        )

    async def portfolio(self):
        raise CapabilityUnavailable("No broker adapter is configured")

    async def quote(self, instrument):
        raise CapabilityUnavailable("No broker adapter is configured")

    async def preview(self, order):
        raise CapabilityUnavailable("No broker adapter is configured")

    async def place(self, order):
        raise CapabilityUnavailable("No broker adapter is configured")


def _plugin(settings: MCPSettings) -> BrokerAdapter:
    module_name, factory_name = settings.broker_plugin.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    adapter = factory(settings)
    if not isinstance(adapter, BrokerAdapter):
        raise TypeError("Broker plugin factory must return a BrokerAdapter instance")
    return adapter


def build_broker_adapter(settings: MCPSettings) -> BrokerAdapter:
    if settings.broker == BrokerName.SNAPSHOT:
        return SnapshotBrokerAdapter(
            settings.snapshot_path,
            max_age_seconds=max(60, settings.quote_max_age_seconds * 20),
        )
    if settings.broker == BrokerName.UPSTOX:
        return UpstoxBrokerAdapter(settings)
    if settings.broker == BrokerName.ZERODHA:
        return ZerodhaBrokerAdapter(settings)
    if settings.broker == BrokerName.PLUGIN:
        return _plugin(settings)
    return NullBrokerAdapter()
