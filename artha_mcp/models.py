"""Portable data contracts used by the Artha MCP boundary."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_US_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_INDIA_SYMBOL = re.compile(r"^[A-Z0-9][A-Z0-9&.-]{0,29}$")


class MarketCode(StrEnum):
    US = "US"
    INDIA = "IN"


class AccessMode(StrEnum):
    READ_ONLY = "read_only"
    OPERATOR = "operator"
    TRADING = "trading"


class BrokerName(StrEnum):
    NONE = "none"
    SNAPSHOT = "snapshot"
    UPSTOX = "upstox"
    ZERODHA = "zerodha"
    PLUGIN = "plugin"


class InstrumentRef(BaseModel):
    """One security expressed without confusing provider and broker symbols."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    market: MarketCode
    symbol: str = Field(min_length=1, max_length=30)
    exchange: str = Field(min_length=1, max_length=12)
    currency: str = Field(min_length=3, max_length=3)
    research_symbol: str = Field(min_length=1, max_length=40)
    broker_instrument_id: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def validate_identity(self) -> InstrumentRef:
        if any(
            value != value.upper()
            for value in (self.symbol, self.exchange, self.currency)
        ):
            raise ValueError("symbol, exchange, and currency must be uppercase")
        if self.market == MarketCode.US:
            if not _US_SYMBOL.fullmatch(self.symbol):
                raise ValueError("invalid US equity symbol")
            if self.currency != "USD" or self.exchange not in {
                "US",
                "NYSE",
                "NASDAQ",
                "AMEX",
            }:
                raise ValueError(
                    "US instruments require USD and a supported US exchange"
                )
            if self.research_symbol != self.symbol:
                raise ValueError("US research_symbol must match symbol")
        else:
            if not _INDIA_SYMBOL.fullmatch(self.symbol):
                raise ValueError("invalid Indian equity symbol")
            expected_research = (
                f"{self.symbol}.NS" if self.exchange == "NSE" else f"{self.symbol}.BO"
            )
            if self.currency != "INR" or self.exchange not in {"NSE", "BSE"}:
                raise ValueError(
                    "Indian instruments require INR and exchange NSE or BSE"
                )
            if self.research_symbol != expected_research:
                raise ValueError(
                    "Indian research_symbol does not match exchange and symbol"
                )
        if self.broker_instrument_id and any(
            ord(character) < 32 for character in self.broker_instrument_id
        ):
            raise ValueError("broker_instrument_id contains control characters")
        if (
            self.broker_instrument_id
            and self.broker_instrument_id != self.broker_instrument_id.strip()
        ):
            raise ValueError("broker_instrument_id cannot have surrounding whitespace")
        return self


class BrokerCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    market: MarketCode
    account_read: bool = False
    portfolio_read: bool = False
    instrument_search: bool = False
    quote_read: bool = False
    order_preview: bool = False
    order_place: bool = False
    order_status: bool = False
    fractional_equities: bool = False
    sandbox: bool = False
    notes: list[str] = Field(default_factory=list)


class Quote(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    instrument: InstrumentRef
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    previous_close: float | None = None
    volume: float | None = None
    timestamp: datetime | None = None
    source: str
    fresh: bool | None = None

    @model_validator(mode="after")
    def validate_numbers(self) -> Quote:
        for name in ("bid", "ask", "last", "previous_close"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")
        if self.volume is not None and (not isfinite(self.volume) or self.volume < 0):
            raise ValueError("volume must be a finite non-negative number")
        if self.bid is not None and self.ask is not None and self.ask < self.bid:
            raise ValueError("ask cannot be below bid in an executable quote")
        return self

    @property
    def midpoint(self) -> float | None:
        if (
            self.bid is not None
            and self.ask is not None
            and self.bid > 0
            and self.ask > 0
        ):
            return (self.bid + self.ask) / 2.0
        return self.last

    @property
    def spread_pct(self) -> float | None:
        midpoint = self.midpoint
        if midpoint is None or midpoint <= 0 or self.bid is None or self.ask is None:
            return None
        return (self.ask - self.bid) / midpoint


class Position(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    instrument: InstrumentRef
    quantity: float
    average_price: float | None = None
    last_price: float | None = None
    market_value: float | None = None
    unrealized_pnl: float | None = None
    product: str | None = None

    @model_validator(mode="after")
    def validate_numbers(self) -> Position:
        if not isfinite(self.quantity):
            raise ValueError("quantity must be finite")
        for name in ("average_price", "last_price"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")
        for name in ("market_value", "unrealized_pnl"):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self


class AccountSnapshot(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    broker: str
    account_id_masked: str | None = None
    currency: str
    buying_power: float | None = None
    cash: float | None = None
    equity: float | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    fresh: bool | None = None

    @model_validator(mode="after")
    def validate_snapshot(self) -> AccountSnapshot:
        for name in ("buying_power", "cash", "equity"):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must include a timezone")
        return self


class PortfolioSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    account: AccountSnapshot
    positions: list[Position] = Field(default_factory=list)
    open_orders: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class OrderRequest(BaseModel):
    """Broker-neutral cash-equity order contract.

    The public MCP intentionally excludes options, leverage, short sales, and
    derivatives. Broker-specific instrument identifiers are mandatory for live
    placement so a display symbol can never be guessed into an order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    action_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
    )
    instrument: InstrumentRef
    side: Literal["buy", "sell"]
    order_type: Literal["market", "limit"]
    quantity: float | None = None
    notional: float | None = None
    limit_price: float | None = None
    time_in_force: Literal["day", "ioc"] = "day"
    product: Literal["cash", "delivery"] = "cash"
    max_price: float | None = None
    min_price: float | None = None
    tag: str | None = Field(
        default=None, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"
    )

    @model_validator(mode="after")
    def validate_shape(self) -> OrderRequest:
        for name in (
            "quantity",
            "notional",
            "limit_price",
            "max_price",
            "min_price",
        ):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if (self.quantity is None) == (self.notional is None):
            raise ValueError("exactly one of quantity or notional is required")
        if self.quantity is not None and self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.notional is not None and self.notional <= 0:
            raise ValueError("notional must be positive")
        if self.order_type == "limit" and (
            self.limit_price is None or self.limit_price <= 0
        ):
            raise ValueError("positive limit_price is required for a limit order")
        if self.side == "buy" and self.max_price is not None and self.max_price <= 0:
            raise ValueError("max_price must be positive")
        if self.side == "sell" and self.min_price is not None and self.min_price <= 0:
            raise ValueError("min_price must be positive")
        if self.side == "buy" and self.min_price is not None:
            raise ValueError("min_price is only valid for sell orders")
        if self.side == "sell" and self.max_price is not None:
            raise ValueError("max_price is only valid for buy orders")
        if (
            self.order_type == "market"
            and self.side == "buy"
            and self.max_price is None
        ):
            raise ValueError("max_price is required for a market buy")
        if (
            self.order_type == "market"
            and self.side == "sell"
            and self.min_price is None
        ):
            raise ValueError("min_price is required for a market sell")
        if (
            self.side == "buy"
            and self.limit_price is not None
            and self.max_price is not None
            and self.limit_price > self.max_price
        ):
            raise ValueError("buy limit_price cannot exceed max_price")
        if (
            self.side == "sell"
            and self.limit_price is not None
            and self.min_price is not None
            and self.limit_price < self.min_price
        ):
            raise ValueError("sell limit_price cannot be below min_price")
        return self


class OrderPreview(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    broker: str
    action_id: str
    passed: bool
    estimated_value: float | None = None
    estimated_fees: float | None = None
    buying_power: float | None = None
    quote: Quote | None = None
    reasons: list[str] = Field(default_factory=list)
    broker_proof: dict[str, Any] = Field(default_factory=dict)
    previewed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def validate_numbers(self) -> OrderPreview:
        for name in ("estimated_value", "estimated_fees", "buying_power"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.previewed_at.tzinfo is None:
            raise ValueError("previewed_at must include a timezone")
        return self


class OrderResult(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    broker: str
    action_id: str
    accepted: bool
    broker_order_id: str | None = None
    status: str
    message: str
    submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    broker_response: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_acceptance(self) -> OrderResult:
        if self.accepted and not self.broker_order_id:
            raise ValueError("accepted broker results require broker_order_id")
        if self.submitted_at.tzinfo is None:
            raise ValueError("submitted_at must include a timezone")
        return self


class WorkflowJob(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    workflow: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "unknown"]
    command: list[str]
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    return_code: int | None = None
    log_path: str
    message: str = ""
