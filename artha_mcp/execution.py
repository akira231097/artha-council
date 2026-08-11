"""Fail-closed preview receipts and exact-order placement for broker adapters."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from .adapters import BrokerAdapter
from .markets import get_market_profile
from .models import MarketCode, OrderPreview, OrderRequest, OrderResult
from .security import TRADE_SCOPE, CapabilityPolicy, redact
from .settings import MCPSettings
from .storage import ensure_private_dir, ensure_private_file


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )


def _hash_order(order: OrderRequest) -> str:
    return hashlib.sha256(_json(order).encode("utf-8")).hexdigest()


class ExecutionCoordinator:
    """Persisted two-stage execution coordinator.

    A preview receipt is immutable, expires quickly, and can be claimed only
    once. Placement repeats the exact broker preview after claiming the row.
    A network-ambiguous placement remains `unknown`, never automatically
    retried, because duplicate real orders are worse than a missed order.
    """

    def __init__(
        self,
        settings: MCPSettings,
        policy: CapabilityPolicy,
        broker: BrokerAdapter,
        *,
        db_path: Path | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.settings = settings
        self.policy = policy
        self.broker = broker
        self.db_path = db_path or settings.data_dir / "mcp" / "execution.db"
        ensure_private_dir(self.db_path.parent)
        self._now = now or (lambda: datetime.now(UTC))
        self._init_db()
        ensure_private_file(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS receipts (
                    receipt_id TEXT PRIMARY KEY,
                    action_id TEXT NOT NULL,
                    order_hash TEXT NOT NULL,
                    order_json TEXT NOT NULL,
                    preview_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    broker_order_id TEXT,
                    result_json TEXT,
                    message TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_receipts_action ON receipts(action_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_receipts_status ON receipts(status)"
            )

    @staticmethod
    def _parse_time(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)

    def _local_reasons(self, order: OrderRequest, preview: OrderPreview) -> list[str]:
        reasons = list(preview.reasons)
        if preview.action_id != order.action_id:
            reasons.append("Broker preview action id does not match the order.")
        if preview.broker != self.broker.capabilities.name:
            reasons.append(
                "Broker preview source does not match the configured adapter."
            )
        profile = get_market_profile(order.instrument.market)
        session = profile.session_status(self._now())
        if (
            not self.broker.capabilities.sandbox
            and not session["regular_session_estimate"]
        ):
            reasons.append(
                f"Outside the estimated regular {order.instrument.market.value} cash-equity session; "
                "the broker remains authoritative for holidays and halts."
            )
        if order.instrument.market != self.broker.capabilities.market:
            reasons.append("Order market does not match the configured broker adapter.")
        if order.instrument.market == MarketCode.INDIA:
            if order.order_type != "limit":
                reasons.append(
                    "Indian API execution is restricted to protected limit orders."
                )
            if (
                order.notional is not None
                or order.quantity is None
                or not float(order.quantity).is_integer()
            ):
                reasons.append(
                    "Indian cash-equity orders require a whole-share quantity."
                )
            if not order.instrument.broker_instrument_id:
                reasons.append(
                    "Indian placement requires a broker instrument identifier."
                )
        estimated = preview.estimated_value
        quote = preview.quote
        execution_price = None
        if quote is not None:
            execution_price = (
                quote.ask or quote.last
                if order.side == "buy"
                else quote.bid or quote.last
            )
        derived_value = order.notional
        if order.quantity is not None:
            priced_at = (
                order.limit_price if order.order_type == "limit" else execution_price
            )
            derived_value = (
                float(order.quantity) * priced_at if priced_at is not None else None
            )
        if derived_value is not None:
            estimated = max(float(estimated or 0.0), float(derived_value))
        if estimated is not None and estimated > self.settings.max_order_value:
            reasons.append(
                f"Estimated value {estimated:.2f} exceeds MCP per-order limit {self.settings.max_order_value:.2f}."
            )
        if quote is None:
            reasons.append("Broker preview did not return a quote.")
        else:
            if quote.fresh is False:
                reasons.append("Broker explicitly marked the quote as stale.")
            if quote.instrument != order.instrument:
                reasons.append("Broker quote instrument does not match the order.")
            if quote.spread_pct is None:
                reasons.append("Live bid/ask spread cannot be verified.")
            elif quote.spread_pct > self.settings.max_spread_pct:
                reasons.append(
                    f"Live spread {quote.spread_pct:.4%} exceeds limit {self.settings.max_spread_pct:.4%}."
                )
            if quote.timestamp is not None and quote.timestamp.tzinfo is not None:
                timestamp = quote.timestamp
                age = (self._now() - timestamp.astimezone(UTC)).total_seconds()
                if age < -5.0:
                    reasons.append(
                        "Broker quote timestamp is unexpectedly in the future."
                    )
                if age > self.settings.quote_max_age_seconds:
                    reasons.append(
                        f"Broker quote is {age:.1f}s old; limit is {self.settings.quote_max_age_seconds}s."
                    )
            else:
                reasons.append(
                    "Broker quote requires a timezone-aware timestamp; freshness cannot be verified."
                )
            approved_max = order.max_price or (
                order.limit_price if order.side == "buy" else None
            )
            if order.side == "buy" and approved_max is not None:
                execution_price = quote.ask or quote.last
                if execution_price is None or execution_price > approved_max:
                    reasons.append(
                        "Live buy price is unavailable or above the approved maximum price."
                    )
            approved_min = order.min_price or (
                order.limit_price if order.side == "sell" else None
            )
            if order.side == "sell" and approved_min is not None:
                execution_price = quote.bid or quote.last
                if execution_price is None or execution_price < approved_min:
                    reasons.append(
                        "Live sell price is unavailable or below the approved minimum price."
                    )
        proof = preview.broker_proof
        required_proof = ["instrument", "quote", "order_preview"]
        required_proof.append("funds" if order.side == "buy" else "position")
        if (
            order.instrument.market == MarketCode.INDIA
            and self.broker.capabilities.name in {"upstox", "zerodha"}
        ):
            required_proof.append("static_ip")
            if order.side == "sell":
                required_proof.append("demat_sell_authorized")
        for check in required_proof:
            if proof.get(check) is not True:
                reasons.append(f"Broker proof is missing or failed: {check}.")
        return list(dict.fromkeys(reasons))

    async def _evaluate(self, order: OrderRequest) -> OrderPreview:
        preview = await self.broker.preview(order)
        reasons = self._local_reasons(order, preview)
        payload = redact(preview.model_dump(mode="python"))
        quote = preview.quote
        execution_price = None
        if quote is not None:
            execution_price = (
                quote.ask or quote.last
                if order.side == "buy"
                else quote.bid or quote.last
            )
        derived_value = order.notional
        if order.quantity is not None:
            priced_at = (
                order.limit_price if order.order_type == "limit" else execution_price
            )
            derived_value = (
                float(order.quantity) * priced_at if priced_at is not None else None
            )
        if derived_value is not None:
            payload["estimated_value"] = max(
                float(payload.get("estimated_value") or 0.0), float(derived_value)
            )
        payload["passed"] = bool(preview.passed and not reasons)
        payload["reasons"] = reasons
        return OrderPreview.model_validate(payload)

    async def create_preview(self, order: OrderRequest) -> dict[str, Any]:
        preview = await self._evaluate(order)
        now = self._now()
        receipt_id = f"mcppr_{uuid4().hex}"
        expires = now + timedelta(seconds=max(5, self.settings.quote_max_age_seconds))
        status = "ready" if preview.passed else "blocked"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO receipts (
                    receipt_id, action_id, order_hash, order_json, preview_json,
                    status, created_at, expires_at, updated_at, message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt_id,
                    order.action_id,
                    _hash_order(order),
                    _json(order),
                    _json(preview),
                    status,
                    now.isoformat(),
                    expires.isoformat(),
                    now.isoformat(),
                    "Preview passed." if preview.passed else "; ".join(preview.reasons),
                ),
            )
        return {
            "status": "PASS" if preview.passed else "BLOCKED",
            "receipt_id": receipt_id,
            "expires_at": expires.isoformat(),
            "order_hash": _hash_order(order),
            "preview": preview.model_dump(mode="json"),
            "placement_contract": "Place by receipt_id only; the exact order is loaded from the immutable receipt.",
        }

    def get_receipt(self, receipt_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM receipts WHERE receipt_id = ?", (receipt_id,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        for key in ("order_json", "preview_json", "result_json"):
            if result.get(key):
                try:
                    result[key.removesuffix("_json")] = json.loads(result[key])
                except json.JSONDecodeError:
                    result[key.removesuffix("_json")] = None
            result.pop(key, None)
        return result

    def _daily_capacity(
        self, now: datetime, conn: sqlite3.Connection
    ) -> tuple[int, float]:
        profile = get_market_profile(self.settings.market)
        local = now.astimezone(ZoneInfo(profile.timezone))
        day_start = local.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + timedelta(days=1)
        rows = conn.execute(
            """
            SELECT order_json, preview_json FROM receipts
            WHERE created_at >= ? AND created_at < ?
              AND status IN ('placing', 'submitted', 'partially_filled', 'filled', 'unknown')
            """,
            (
                day_start.astimezone(UTC).isoformat(),
                day_end.astimezone(UTC).isoformat(),
            ),
        ).fetchall()
        total = 0.0
        for row in rows:
            try:
                preview = json.loads(row["preview_json"])
                order = json.loads(row["order_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            value = preview.get("estimated_value") or order.get("notional") or 0.0
            try:
                total += float(value)
            except (TypeError, ValueError):
                pass
        return len(rows), total

    def _claim(self, receipt_id: str) -> tuple[OrderRequest, dict[str, Any]]:
        now = self._now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM receipts WHERE receipt_id = ?", (receipt_id,)
            ).fetchone()
            if not row:
                raise ValueError("Preview receipt was not found.")
            if row["status"] != "ready":
                raise ValueError(f"Preview receipt is {row['status']}, not ready.")
            if self._parse_time(row["expires_at"]) < now:
                conn.execute(
                    "UPDATE receipts SET status='expired', updated_at=?, message=? WHERE receipt_id=?",
                    (now.isoformat(), "Preview receipt expired.", receipt_id),
                )
                conn.commit()
                raise ValueError("Preview receipt expired; request a fresh preview.")
            prior = conn.execute(
                """
                SELECT receipt_id, status FROM receipts
                WHERE action_id = ? AND receipt_id != ?
                  AND status IN ('placing', 'submitted', 'partially_filled', 'filled', 'unknown')
                LIMIT 1
                """,
                (row["action_id"], receipt_id),
            ).fetchone()
            if prior:
                raise ValueError(
                    f"Action already has a {prior['status']} execution receipt."
                )
            count, value = self._daily_capacity(now, conn)
            preview = json.loads(row["preview_json"])
            order_payload = json.loads(row["order_json"])
            proposed_value = float(
                preview.get("estimated_value") or order_payload.get("notional") or 0.0
            )
            if count >= self.settings.max_daily_orders:
                raise ValueError("MCP daily order-count limit is exhausted.")
            if value + proposed_value > self.settings.max_daily_order_value:
                raise ValueError("MCP daily order-value limit would be exceeded.")
            conn.execute(
                "UPDATE receipts SET status='placing', updated_at=?, message=? WHERE receipt_id=?",
                (now.isoformat(), "Receipt claimed for final recheck.", receipt_id),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        order = OrderRequest.model_validate(order_payload)
        if _hash_order(order) != row["order_hash"]:
            self._finish(receipt_id, "blocked", message="Stored order hash mismatch.")
            raise ValueError("Stored order integrity check failed.")
        return order, dict(row)

    def _finish(
        self,
        receipt_id: str,
        status: str,
        *,
        result: OrderResult | None = None,
        message: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE receipts
                SET status=?, updated_at=?, broker_order_id=?, result_json=?, message=?
                WHERE receipt_id=?
                """,
                (
                    status,
                    self._now().isoformat(),
                    result.broker_order_id if result else None,
                    _json(result) if result else None,
                    message,
                    receipt_id,
                ),
            )

    async def place(
        self, receipt_id: str, *, oauth_scopes: set[str] | None = None
    ) -> dict[str, Any]:
        self.policy.require(TRADE_SCOPE, oauth_scopes=oauth_scopes)
        order, _ = self._claim(receipt_id)
        capabilities = self.broker.capabilities
        if not capabilities.order_status and not capabilities.sandbox:
            message = (
                "Live placement is blocked because the broker adapter cannot prove "
                "duplicate orders or reconcile order status; no order call was made."
            )
            self._finish(receipt_id, "blocked", message=message)
            return {
                "status": "BLOCKED",
                "receipt_id": receipt_id,
                "message": message,
                "reconciliation_available": False,
            }
        try:
            final_preview = await self._evaluate(order)
        except Exception as exc:  # noqa: BLE001 - an unclassified adapter failure must fail closed
            message = f"Final broker recheck failed before placement ({type(exc).__name__}); no order call was made."
            self._finish(receipt_id, "blocked", message=message)
            return {"status": "BLOCKED", "receipt_id": receipt_id, "message": message}
        if not final_preview.passed:
            message = "Final broker recheck blocked: " + "; ".join(
                final_preview.reasons
            )
            self._finish(receipt_id, "blocked", message=message)
            return {
                "status": "BLOCKED",
                "receipt_id": receipt_id,
                "message": message,
                "final_preview": final_preview.model_dump(mode="json"),
            }
        existing = None
        if capabilities.order_status:
            try:
                existing = await self.broker.find_existing_order(order)
            except Exception as exc:  # noqa: BLE001 - duplicate-check uncertainty must fail closed
                message = f"Broker duplicate-order check failed before placement ({type(exc).__name__}); no order call was made."
                self._finish(receipt_id, "blocked", message=message)
                return {
                    "status": "BLOCKED",
                    "receipt_id": receipt_id,
                    "message": message,
                }
        if existing is not None:
            broker_order_id = (
                str(existing.get("order_id") or existing.get("id") or "") or None
            )
            if broker_order_id is None:
                message = (
                    "A matching broker action tag was found, but no broker order id was returned; "
                    "placement remains blocked to prevent a duplicate."
                )
                self._finish(receipt_id, "unknown", message=message)
                return {
                    "status": "UNKNOWN",
                    "receipt_id": receipt_id,
                    "message": message,
                    "duplicate_prevented": True,
                    "reconciliation_required": True,
                }
            result = OrderResult(
                broker=self.broker.capabilities.name,
                action_id=order.action_id,
                accepted=True,
                broker_order_id=broker_order_id,
                status="existing_order",
                message="An existing broker order has the same immutable action tag; duplicate placement was skipped.",
                broker_response=redact(existing),
            )
            self._finish(receipt_id, "submitted", result=result, message=result.message)
            return {
                "status": "PASS",
                "receipt_id": receipt_id,
                "final_preview": final_preview.model_dump(mode="json"),
                "result": result.model_dump(mode="json"),
                "duplicate_prevented": True,
                "reconciliation_required": True,
            }
        try:
            result = await self.broker.place(order)
        except Exception as exc:  # noqa: BLE001 - placement exceptions have an intentionally unknown outcome
            if capabilities.order_status:
                message = (
                    f"Placement outcome is unknown after {type(exc).__name__}; "
                    "do not retry until broker orders are reconciled."
                )
            else:
                message = (
                    f"Sandbox placement outcome is unknown after {type(exc).__name__}; "
                    "the receipt remains single-use and this sandbox cannot reconcile it."
                )
            self._finish(receipt_id, "unknown", message=message)
            return {
                "status": "UNKNOWN",
                "receipt_id": receipt_id,
                "message": message,
                "reconciliation_available": capabilities.order_status,
            }
        result = OrderResult.model_validate(redact(result.model_dump(mode="python")))
        status = "submitted" if result.accepted else "rejected"
        self._finish(receipt_id, status, result=result, message=result.message)
        return {
            "status": "PASS" if result.accepted else "BLOCKED",
            "receipt_id": receipt_id,
            "final_preview": final_preview.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "reconciliation_required": bool(
                result.accepted and capabilities.order_status
            ),
            "reconciliation_available": capabilities.order_status,
            "sandbox_submission_only": bool(
                result.accepted
                and capabilities.sandbox
                and not capabilities.order_status
            ),
        }

    @staticmethod
    def _broker_order_status(row: dict[str, Any]) -> str:
        try:
            filled = float(row.get("filled_quantity") or row.get("filled_qty") or 0)
            pending = float(row.get("pending_quantity") or row.get("pending_qty") or 0)
            quantity = float(row.get("quantity") or row.get("qty") or 0)
        except (TypeError, ValueError):
            filled = pending = quantity = 0.0
        if filled > 0 and (pending > 0 or (quantity > 0 and filled < quantity)):
            return "partially_filled"
        if filled > 0 and quantity > 0 and filled >= quantity:
            return "filled"
        raw = (
            str(row.get("status") or row.get("state") or row.get("order_status") or "")
            .strip()
            .lower()
        )
        normalized = raw.replace("-", "_").replace(" ", "_")
        if normalized in {"complete", "completed", "filled", "executed"}:
            return "filled"
        if normalized in {
            "partial",
            "partially_filled",
            "partially_executed",
            "open_pending_quantity",
        }:
            return "partially_filled"
        if normalized in {"cancelled", "canceled"}:
            return "cancelled"
        if normalized in {"rejected", "failed"}:
            return "rejected"
        if normalized in {
            "accepted",
            "after_market_order_req_received",
            "amo_req_received",
            "cancel_pending",
            "modify_pending",
            "modify_validation_pending",
            "open",
            "open_pending",
            "pending",
            "put_order_req_received",
            "submitted",
            "trigger_pending",
            "validation_pending",
        }:
            return "submitted"
        return "unknown"

    async def reconcile(self, receipt_id: str) -> dict[str, Any]:
        """Reconcile an accepted or ambiguous receipt without ever resubmitting it."""
        receipt = self.get_receipt(receipt_id)
        if receipt is None:
            raise ValueError("Execution receipt was not found.")
        current = str(receipt.get("status") or "")
        if current in {"filled", "partially_filled", "cancelled", "rejected"}:
            return {
                "status": "PASS",
                "receipt_id": receipt_id,
                "execution_status": current,
                "message": "Receipt is already in a reconciled broker state.",
            }
        if current not in {"placing", "submitted", "unknown"}:
            raise ValueError(
                f"Receipt status {current or 'missing'} is not eligible for broker reconciliation."
            )
        if not self.broker.capabilities.order_status:
            return {
                "status": "WARN",
                "receipt_id": receipt_id,
                "execution_status": current,
                "reconciliation_available": False,
                "order_retried": False,
                "message": (
                    "The configured broker mode does not expose order-status "
                    "reconciliation; the receipt was not changed or retried."
                ),
            }
        order = OrderRequest.model_validate(receipt["order"])
        try:
            rows = await self.broker.orders()
        except Exception as exc:  # noqa: BLE001 - unavailable status must preserve uncertainty
            return {
                "status": "WARN",
                "receipt_id": receipt_id,
                "execution_status": current,
                "message": f"Broker order reconciliation failed ({type(exc).__name__}); the receipt was not changed or retried.",
            }

        wanted_id = str(receipt.get("broker_order_id") or "")
        wanted_tag = self.broker.client_order_key(order)
        matched: dict[str, Any] | None = None
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            broker_id = str(raw.get("order_id") or raw.get("id") or "")
            tags = {
                str(raw.get(name) or "")
                for name in (
                    "tag",
                    "order_tag",
                    "client_order_id",
                    "client_order_tag",
                )
            }
            if (wanted_id and broker_id == wanted_id) or wanted_tag in tags:
                matched = raw
                break

        if matched is None:
            next_status = "unknown" if current in {"placing", "unknown"} else current
            if next_status != current:
                self._finish(
                    receipt_id,
                    next_status,
                    message="No matching broker order was found after an interrupted placement; outcome remains unknown and must not be retried.",
                )
            return {
                "status": "WARN",
                "receipt_id": receipt_id,
                "execution_status": next_status,
                "message": "No matching broker order is currently visible; no order was retried.",
            }

        broker_order_id = (
            str(matched.get("order_id") or matched.get("id") or "") or None
        )
        if broker_order_id is None:
            message = (
                "A matching broker action tag is visible, but no broker order id was returned; "
                "the outcome remains unknown and no order was retried."
            )
            self._finish(receipt_id, "unknown", message=message)
            return {
                "status": "WARN",
                "receipt_id": receipt_id,
                "execution_status": "unknown",
                "message": message,
                "order_retried": False,
            }
        broker_status = self._broker_order_status(matched)
        next_status = (
            broker_status
            if broker_status != "unknown"
            else ("unknown" if current == "unknown" else "submitted")
        )
        result = OrderResult(
            broker=self.broker.capabilities.name,
            action_id=order.action_id,
            accepted=next_status not in {"rejected"},
            broker_order_id=broker_order_id,
            status=next_status,
            message=f"Broker reconciliation observed status {next_status}.",
            broker_response=redact(matched),
        )
        self._finish(receipt_id, next_status, result=result, message=result.message)
        return {
            "status": "PASS" if broker_status != "unknown" else "WARN",
            "receipt_id": receipt_id,
            "execution_status": next_status,
            "result": result.model_dump(mode="json"),
            "order_retried": False,
        }
