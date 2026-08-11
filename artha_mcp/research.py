"""Market-aware research adapters for the public MCP boundary."""

from __future__ import annotations

import asyncio
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Any

from .models import InstrumentRef, MarketCode
from .settings import MCPSettings


class ResearchAdapter(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    async def collect(self, instrument: InstrumentRef) -> dict[str, Any]:
        raise NotImplementedError


class EmbeddedUSResearchAdapter(ResearchAdapter):
    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "name": "embedded_us",
            "markets": ["US"],
            "packet": "full Artha US evidence packet",
            "embedded_workflows": True,
        }

    async def collect(self, instrument: InstrumentRef) -> dict[str, Any]:
        if instrument.market != MarketCode.US:
            raise ValueError(
                "The embedded Artha research adapter supports US equities only"
            )
        from artha.collector import DataCollector

        data = await asyncio.to_thread(
            DataCollector().collect_stock, instrument.research_symbol
        )
        return {"data": data, "completeness": "built_in_us_packet", "limitations": []}


class HostOrchestratedResearchAdapter(ResearchAdapter):
    @property
    def capabilities(self) -> dict[str, Any]:
        return {
            "name": "host_orchestrated",
            "markets": ["US", "IN"],
            "packet": "technical market packet for host-model enrichment",
            "embedded_workflows": False,
        }

    async def collect(self, instrument: InstrumentRef) -> dict[str, Any]:
        from artha.collector import YFinanceCollector

        collector = YFinanceCollector()
        quote, history = await asyncio.gather(
            asyncio.to_thread(collector.quote, instrument.research_symbol),
            asyncio.to_thread(collector.history, instrument.research_symbol, "1y"),
        )
        limitations = [
            "This is a technical market packet, not a complete fundamental diligence packet.",
            "The MCP host must add properly licensed, market-native filings, fundamentals, news, and estimates.",
            "Host-model conclusions are not an embedded Artha Council verdict.",
        ]
        return {
            "data": {
                "ticker": instrument.symbol,
                "yf_quote": quote,
                "price_history": history,
            },
            "completeness": "technical_market_packet_only",
            "limitations": limitations,
        }


class PluginResearchAdapter(ResearchAdapter):
    def __init__(self, settings: MCPSettings) -> None:
        module_name, factory_name = settings.research_plugin.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, factory_name)
        adapter = factory(settings)
        if not hasattr(adapter, "capabilities") or not hasattr(adapter, "collect"):
            raise TypeError("Research plugin must expose capabilities and collect")
        if not inspect.iscoroutinefunction(adapter.collect):
            raise TypeError("Research plugin collect method must be async")
        self.adapter = adapter

    @property
    def capabilities(self) -> dict[str, Any]:
        value = self.adapter.capabilities
        value = value() if callable(value) else value
        if not isinstance(value, dict):
            raise TypeError("Research plugin capabilities must be a dictionary")
        return value

    async def collect(self, instrument: InstrumentRef) -> dict[str, Any]:
        result = await self.adapter.collect(instrument)
        if not isinstance(result, dict):
            raise TypeError("Research plugin collect must return a dictionary")
        if not isinstance(result.get("data"), dict):
            raise TypeError("Research plugin result must contain a data dictionary")
        result.setdefault("completeness", "plugin_defined")
        result.setdefault("limitations", [])
        return result


def build_research_adapter(settings: MCPSettings) -> ResearchAdapter:
    if settings.research_mode == "embedded":
        return EmbeddedUSResearchAdapter()
    if settings.research_mode == "plugin":
        return PluginResearchAdapter(settings)
    return HostOrchestratedResearchAdapter()
