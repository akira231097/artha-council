"""Broker and notification adapter registry."""

from .base import AdapterError, BrokerAdapter, CapabilityUnavailable
from .registry import build_broker_adapter

__all__ = [
    "AdapterError",
    "BrokerAdapter",
    "CapabilityUnavailable",
    "build_broker_adapter",
]
