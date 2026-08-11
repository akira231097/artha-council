"""Application facade that maps MCP calls onto Artha's existing services."""

from __future__ import annotations

import asyncio
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from artha import __version__
from artha.journal import DecisionJournal
from artha.paths import runtime_asset_path
from artha.portfolio_state import PortfolioStateEngine

from .adapters import (
    BrokerAdapter,
    CapabilityUnavailable,
    build_broker_adapter,
)
from .execution import ExecutionCoordinator
from .jobs import WorkflowJobManager
from .markets import get_market_profile, normalize_instrument
from .models import BrokerName, InstrumentRef, MarketCode, OrderRequest
from .notifications import NotificationHub
from .provenance import build_provenance
from .research import ResearchAdapter, build_research_adapter
from .security import (
    OPERATE_SCOPE,
    TRADE_SCOPE,
    CapabilityPolicy,
    redact,
    robinhood_host_operation_contract,
)
from .settings import MCPSettings
from .storage import atomic_write_private_text, ensure_private_dir


def _bounded(value: Any, *, depth: int = 0) -> Any:
    """Keep tool results useful without flooding a host model context window."""
    if depth > 8:
        return "[depth limited]"
    if isinstance(value, dict):
        return {
            str(key): _bounded(child, depth=depth + 1)
            for key, child in list(value.items())[:120]
        }
    if isinstance(value, list):
        return [_bounded(child, depth=depth + 1) for child in value[:50]]
    if isinstance(value, str) and len(value) > 12000:
        return value[:12000] + "...[truncated]"
    return value


class ArthaMCPService:
    def __init__(
        self,
        settings: MCPSettings,
        *,
        broker: BrokerAdapter | None = None,
        research: ResearchAdapter | None = None,
        journal: DecisionJournal | None = None,
    ) -> None:
        self.settings = settings
        self.policy = CapabilityPolicy(settings)
        self.broker = broker or build_broker_adapter(settings)
        self.research = research or build_research_adapter(settings)
        self.journal = journal or DecisionJournal(settings.db_path)
        self.portfolio_engine = PortfolioStateEngine(settings.portfolio_path)
        self.jobs = WorkflowJobManager(
            settings.root,
            settings.data_dir,
            timeout_seconds=settings.workflow_timeout_seconds,
            max_pending_jobs=settings.max_pending_jobs,
        )
        self.execution = ExecutionCoordinator(settings, self.policy, self.broker)
        self.notifications = NotificationHub(settings.notification)
        self.research_dir = settings.data_dir / "mcp" / "research"
        ensure_private_dir(self.research_dir)

    def capabilities(self) -> dict[str, Any]:
        profile = get_market_profile(self.settings.market)
        broker = self.broker.capabilities
        provenance = build_provenance()
        return {
            "name": "Artha Council MCP",
            "version": __version__,
            "protocol_surface": {
                "transports": ["stdio", "streamable-http"],
                "resources": True,
                "prompts": True,
                "long_jobs": "explicit persisted job handles",
            },
            "market": {
                "code": profile.code.value,
                "name": profile.name,
                "currency": profile.currency,
                "timezone": profile.timezone,
                "exchanges": list(profile.exchanges),
                "fractional_equities_default": profile.fractional_equities_default,
                "research_status": profile.research_status,
                "research_notes": list(profile.research_notes),
            },
            "broker": broker.model_dump(mode="json"),
            "research": self.research.capabilities,
            "models": {
                "host_orchestrated": (
                    "Any MCP host can use its own model subscription to inspect Artha resources and call tools."
                ),
                "embedded_council": (
                    "Artha's independent Council roles still require the provider credentials configured for this installation."
                ),
                "client_sampling_dependency": False,
            },
            "communications": {
                "mcp": True,
                "telegram": True,
                "webhook": True,
                "openclaw_bridge": {
                    "operation_contract": True,
                    "runner_artifact_present": runtime_asset_path(
                        "scripts", "robinhood_auto_trade_runner.mjs"
                    ).exists(),
                    "node_runtime_present": shutil.which("node") is not None,
                    "note": "The Robinhood runner remains a separately authenticated OpenClaw/Node deployment.",
                },
                "notification": self.notifications.status(),
            },
            "workflows": [
                "scheduled_scan",
                "scan",
                "analyze",
                "supervisor",
                "execution_readiness",
                "broker_router_preview",
                "sell_review",
            ],
            "safety": {
                "public_default": "read_only",
                "raw_secrets_in_tool_arguments": False,
                "exact_order_preview_receipts": True,
                "final_broker_recheck": True,
                "fail_closed": True,
            },
            "provenance": provenance,
        }

    def configuration(self) -> dict[str, Any]:
        return self.settings.public_summary()

    def sync_status(self) -> dict[str, Any]:
        """Prove that the loaded core and MCP source belong to one release."""
        return build_provenance()

    async def health(self) -> dict[str, Any]:
        findings = self.settings.startup_findings()
        provenance = build_provenance()
        try:
            broker_health = await self.broker.health()
        except Exception as exc:  # noqa: BLE001 - health must report every adapter boundary failure
            broker_health = {
                "status": "FAIL",
                "message": f"{type(exc).__name__}: {str(exc)[:300]}",
            }
        latest_supervisor = self.journal.get_latest_supervisor_run()
        status = (
            "FAIL"
            if findings["errors"] or provenance["status"] == "FAIL"
            else "WARN"
            if findings["warnings"]
            or broker_health.get("status") != "PASS"
            or provenance["status"] == "WARN"
            else "PASS"
        )
        return {
            "status": status,
            "generated_at": datetime.now(UTC).isoformat(),
            "configuration": findings,
            "provenance": provenance,
            "broker": redact(broker_health),
            "storage": {
                "data_dir_configured": True,
                "journal_reachable": self.settings.db_path.exists(),
                "portfolio_present": self.settings.portfolio_path.exists(),
                "snapshot_present": self.settings.snapshot_path.exists(),
            },
            "latest_supervisor": redact(latest_supervisor)
            if latest_supervisor
            else None,
        }

    def market_status(self) -> dict[str, Any]:
        profile = get_market_profile(self.settings.market)
        return {**profile.session_status(), "profile": self.capabilities()["market"]}

    def resolve_instrument(
        self,
        symbol: str,
        *,
        market: str | None = None,
        exchange: str | None = None,
        broker_instrument_id: str | None = None,
    ) -> InstrumentRef:
        instrument = normalize_instrument(
            symbol,
            market=market or self.settings.market,
            exchange=exchange,
            broker_instrument_id=broker_instrument_id,
        )
        if (
            instrument.market == MarketCode.INDIA
            and self.settings.broker == BrokerName.ZERODHA
            and instrument.broker_instrument_id is None
        ):
            instrument = instrument.model_copy(
                update={
                    "broker_instrument_id": f"{instrument.exchange}:{instrument.symbol}"
                }
            )
        return instrument

    async def search_instruments(
        self, query: str, *, exchange: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        rows = await self.broker.search_instruments(
            query, exchange=exchange, limit=max(1, min(limit, 30))
        )
        return _bounded(
            {
                "status": "PASS",
                "broker": self.broker.capabilities.name,
                "query": str(query)[:50],
                "count": len(rows),
                "instruments": redact(rows),
                "execution_note": "Use the exact returned instrument object; placement re-verifies broker identity.",
            }
        )

    async def portfolio(self) -> dict[str, Any]:
        try:
            snapshot = await self.broker.portfolio()
            return _bounded(
                redact(
                    {
                        "status": "PASS",
                        "source": "broker",
                        "portfolio": snapshot.model_dump(mode="json"),
                    }
                )
            )
        except CapabilityUnavailable as exc:
            state = self.portfolio_engine.compute_state()
            return {
                "status": "WARN",
                "source": "artha_local",
                "warning": str(exc),
                "portfolio": redact(state),
            }

    async def quote(self, instrument: InstrumentRef) -> dict[str, Any]:
        quote = await self.broker.quote(instrument)
        return _bounded(
            redact(
                {
                    "status": "PASS",
                    "quote": quote.model_dump(mode="json"),
                    "spread_pct": quote.spread_pct,
                    "execution_note": "A quote is information only; order preview repeats broker checks.",
                }
            )
        )

    async def broker_orders(self, *, limit: int = 100) -> dict[str, Any]:
        if not self.broker.capabilities.order_status:
            return {
                "status": "WARN",
                "count": 0,
                "orders": [],
                "message": (
                    "The configured broker mode does not expose an order book or "
                    "fill-status reconciliation."
                ),
            }
        rows = await self.broker.orders()
        bounded = [redact(row) for row in rows[: max(1, min(limit, 250))]]
        return _bounded({"status": "PASS", "count": len(bounded), "orders": bounded})

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        try:
            with self.journal._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []
        return [redact(dict(row)) for row in rows]

    def recent_decisions(
        self, *, ticker: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 100))
        if ticker:
            rows = self.journal.get_recent_recommendations(ticker, limit=limit)
        else:
            rows = self._query(
                "SELECT * FROM recommendations ORDER BY datetime(timestamp) DESC, id DESC LIMIT ?",
                (limit,),
            )
        return {"status": "PASS", "count": len(rows), "decisions": redact(rows)}

    def active_theses(self) -> dict[str, Any]:
        rows = self.journal.get_all_sell_monitored_theses()
        return {"status": "PASS", "count": len(rows), "theses": redact(rows)}

    def defer_watches(self, *, limit: int = 100) -> dict[str, Any]:
        rows = self.journal.get_defer_watches(
            status="active", limit=max(1, min(limit, 250))
        )
        return {"status": "PASS", "count": len(rows), "watches": redact(rows)}

    def execution_queue(self, *, limit: int = 50) -> dict[str, Any]:
        actions = self.journal.get_trade_actions(limit=max(1, min(limit, 100)))
        orders = self.journal.get_execution_orders(limit=max(1, min(limit, 100)))
        return redact(
            {
                "status": "PASS",
                "trade_actions": actions,
                "execution_orders": orders,
                "mcp_receipts": [
                    self.execution.get_receipt(row["receipt_id"])
                    for row in self._receipt_rows(limit)
                ],
            }
        )

    def _receipt_rows(self, limit: int) -> list[dict[str, Any]]:
        with self.execution._connect() as conn:
            rows = conn.execute(
                "SELECT receipt_id FROM receipts ORDER BY datetime(created_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def schedule_status(self) -> dict[str, Any]:
        slots_path = self.settings.data_dir / "scheduler_run_slots.json"
        capacity_path = (
            self.settings.data_dir / "robinhood" / "buy_scan_capacity_state.json"
        )

        def load(path: Path) -> Any:
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

        return redact(
            {
                "status": "PASS",
                "scheduler": {
                    "monitor_command": ["python", "-m", "run", "monitor"],
                    "buy_scan_local_time": "11:30 America/Chicago (current built-in US scheduler)",
                    "market_profile": self.settings.market.value,
                    "portable_note": "Use the market-specific deployment examples; broker/exchange status is authoritative.",
                },
                "last_run_slots": load(slots_path),
                "buy_capacity_state": load(capacity_path),
                "recent_jobs": [
                    row.model_dump(mode="json") for row in self.jobs.list(limit=10)
                ],
            }
        )

    async def collect_evidence(self, instrument: InstrumentRef) -> dict[str, Any]:
        artifact_id = f"research_{uuid4().hex}"
        target = self.research_dir / f"{artifact_id}.json"
        collected = await self.research.collect(instrument)
        data = collected["data"]
        completeness = str(collected.get("completeness") or "unknown")
        limitations = [str(value) for value in collected.get("limitations", [])]
        payload = {
            "artifact_id": artifact_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "instrument": instrument.model_dump(mode="json"),
            "completeness": completeness,
            "limitations": limitations,
            "data": redact(data),
        }
        atomic_write_private_text(
            target,
            json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        )
        return {
            "status": "PASS" if not limitations else "PARTIAL",
            "artifact_id": artifact_id,
            "resource_uri": f"artha://research/{artifact_id}",
            "completeness": completeness,
            "limitations": limitations,
            "packet": _bounded(payload),
        }

    def research_artifact(self, artifact_id: str) -> dict[str, Any]:
        if not artifact_id.startswith("research_") or not artifact_id[9:].isalnum():
            raise ValueError("Invalid research artifact id")
        path = self.research_dir / f"{artifact_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError("Research artifact was not found") from exc
        return redact(payload)

    def list_dossiers(
        self, *, ticker: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        base = self.settings.data_dir / "decision_dossiers"
        wanted = str(ticker or "").upper().strip()
        paths = (
            sorted(
                base.glob("*/*.json"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if base.exists()
            else []
        )
        rows = []
        for path in paths:
            if wanted and not path.name.upper().startswith(f"{wanted}_"):
                continue
            rows.append(
                {
                    "id": f"{path.parent.name}/{path.stem}",
                    "ticker": path.name.split("_", 1)[0].upper(),
                    "modified_at": datetime.fromtimestamp(
                        path.stat().st_mtime, UTC
                    ).isoformat(),
                    "resource_uri": f"artha://dossier/{path.parent.name}/{path.stem}",
                }
            )
            if len(rows) >= max(1, min(limit, 200)):
                break
        return {"status": "PASS", "count": len(rows), "dossiers": rows}

    def dossier(self, day: str, dossier_id: str) -> dict[str, Any]:
        if len(day) != 10 or not all(ch.isdigit() or ch == "-" for ch in day):
            raise ValueError("Invalid dossier date")
        if not dossier_id or not all(ch.isalnum() or ch in "_-" for ch in dossier_id):
            raise ValueError("Invalid dossier id")
        path = (
            self.settings.data_dir / "decision_dossiers" / day / f"{dossier_id}.json"
        ).resolve()
        base = (self.settings.data_dir / "decision_dossiers").resolve()
        if base not in path.parents:
            raise ValueError("Dossier path escaped the data directory")
        try:
            return redact(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError as exc:
            raise ValueError("Dossier was not found") from exc

    def _core_live_trading_enabled(self) -> bool:
        try:
            from artha.config import Config
            from artha.robinhood_bridge import get_trading_control

            return bool(
                Config.ROBINHOOD_AGENTIC_ENABLED
                and not Config.ROBINHOOD_DRY_RUN_ONLY
                and not Config.ROBINHOOD_REVIEW_ONLY
                and not Config.ROBINHOOD_KILL_SWITCH
                and not get_trading_control().get("trading_disabled")
            )
        except Exception:  # noqa: BLE001 - missing or broken private-runtime modules mean trading is off
            return False

    def start_workflow(
        self,
        workflow: str,
        *,
        symbols: list[str] | None = None,
        limit: int = 8,
        telegram: bool = False,
        oauth_scopes: set[str] | None = None,
    ) -> dict[str, Any]:
        self.policy.require(OPERATE_SCOPE, oauth_scopes=oauth_scopes)
        investment_workflows = {
            "scheduled_scan",
            "scan",
            "analyze",
            "broker_router_preview",
            "sell_review",
        }
        if workflow in investment_workflows and not self.research.capabilities.get(
            "embedded_workflows"
        ):
            raise ValueError(
                "Embedded investment workflows are unavailable for this research adapter. "
                "Use the MCP host to orchestrate evidence or install an audited market-native workflow integration."
            )
        if workflow in investment_workflows and self.settings.market != MarketCode.US:
            raise ValueError(
                "The built-in Artha Council and sell Council are US-native and are blocked outside US mode"
            )
        auto_trade_capable = {"scheduled_scan", "scan", "analyze", "sell_review"}
        if workflow in auto_trade_capable and (
            self.settings.trading_enabled or self._core_live_trading_enabled()
        ):
            self.policy.require(TRADE_SCOPE, oauth_scopes=oauth_scopes)
        job = self.jobs.start(workflow, symbols=symbols, limit=limit, telegram=telegram)
        return {"status": "PASS", "job": job.model_dump(mode="json")}

    def cancel_workflow(
        self, job_id: str, *, oauth_scopes: set[str] | None = None
    ) -> dict[str, Any]:
        self.policy.require(OPERATE_SCOPE, oauth_scopes=oauth_scopes)
        return {
            "status": "PASS",
            "job": self.jobs.cancel(job_id).model_dump(mode="json"),
        }

    async def preview_order(
        self, order: OrderRequest, *, oauth_scopes: set[str] | None = None
    ) -> dict[str, Any]:
        self.policy.require(OPERATE_SCOPE, oauth_scopes=oauth_scopes)
        return _bounded(await self.execution.create_preview(order))

    async def place_order(
        self, receipt_id: str, *, oauth_scopes: set[str] | None = None
    ) -> dict[str, Any]:
        return _bounded(
            await self.execution.place(receipt_id, oauth_scopes=oauth_scopes)
        )

    async def reconcile_order(
        self, receipt_id: str, *, oauth_scopes: set[str] | None = None
    ) -> dict[str, Any]:
        self.policy.require(OPERATE_SCOPE, oauth_scopes=oauth_scopes)
        return _bounded(await self.execution.reconcile(receipt_id))

    def robinhood_operation(
        self, action_id: str, *, oauth_scopes: set[str] | None = None
    ) -> dict[str, Any]:
        self.policy.require(TRADE_SCOPE, oauth_scopes=oauth_scopes)
        from artha.config import Config
        from artha.robinhood_bridge import build_auto_buy_operation

        operation = build_auto_buy_operation(action_id, journal=self.journal)
        return robinhood_host_operation_contract(
            operation,
            account_number=str(Config.ROBINHOOD_AGENTIC_ACCOUNT_NUMBER or ""),
            expected_account_type=str(Config.ROBINHOOD_EXPECTED_ACCOUNT_TYPE or ""),
            expected_account_nickname=str(
                Config.ROBINHOOD_EXPECTED_ACCOUNT_NICKNAME or ""
            ),
        )

    async def notify(
        self, message: str, *, oauth_scopes: set[str] | None = None
    ) -> dict[str, Any]:
        self.policy.require(OPERATE_SCOPE, oauth_scopes=oauth_scopes)
        return await self.notifications.send(message)

    async def close(self) -> None:
        await asyncio.to_thread(self.jobs.close)
        await self.broker.close()
