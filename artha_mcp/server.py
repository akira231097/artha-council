"""MCP tools, resources, and prompts for Artha Council."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.types import ToolAnnotations

from artha import __version__

from .auth import build_auth
from .models import OrderRequest
from .security import READ_SCOPE, redact
from .service import ArthaMCPService
from .settings import MCPSettings

READ_ONLY = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False
)
READ_OPEN = ToolAnnotations(
    readOnlyHint=True, destructiveHint=False, idempotentHint=False, openWorldHint=True
)
OPERATE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=True
)
DESTRUCTIVE = ToolAnnotations(
    readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True
)


def _scopes() -> set[str] | None:
    token = get_access_token()
    return set(token.scopes) if token else None


def create_server(
    settings: MCPSettings | None = None,
    *,
    service: ArthaMCPService | None = None,
) -> MCPServer:
    settings = settings or MCPSettings.from_env()
    service = service or ArthaMCPService(settings)
    auth, verifier = build_auth(settings)

    @asynccontextmanager
    async def lifespan(_server):
        yield {"service": service}
        await service.close()

    mcp = MCPServer(
        name="artha-council",
        title="Artha Council",
        description="Auditable equity research, portfolio monitoring, and fail-closed broker execution.",
        instructions=(
            "Start with artha_capabilities and artha_configuration. Treat research verdicts and execution "
            "proof as separate decisions. Never infer missing broker evidence. For money-moving operations, "
            "obtain an immutable preview receipt and place only that receipt. India mode requires India-native "
            "research data, broker-verified instruments, a registered static IP, and protected limit orders; "
            "do not reuse US SEC/FMP assumptions."
        ),
        website_url="https://github.com/akira231097/artha-council",
        version=__version__,
        auth=auth,
        token_verifier=verifier,
        lifespan=lifespan,
    )

    def require_read() -> None:
        service.policy.require(READ_SCOPE, oauth_scopes=_scopes())

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_capabilities() -> dict[str, Any]:
        """Describe markets, brokers, model modes, workflows, and safety boundaries."""
        require_read()
        return service.capabilities()

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_configuration() -> dict[str, Any]:
        """Return redacted configuration readiness; credentials are never returned."""
        require_read()
        return service.configuration()

    @mcp.tool(annotations=READ_OPEN, structured_output=True)
    async def artha_health() -> dict[str, Any]:
        """Check MCP storage, broker connectivity, configuration, and latest Supervisor state."""
        require_read()
        return await service.health()

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_market_status() -> dict[str, Any]:
        """Return the configured market profile and conservative regular-session estimate."""
        require_read()
        return service.market_status()

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_resolve_instrument(
        symbol: str,
        market: str | None = None,
        exchange: str | None = None,
        broker_instrument_id: str | None = None,
    ) -> dict[str, Any]:
        """Normalize a US or Indian equity without guessing a broker instrument id."""
        require_read()
        return service.resolve_instrument(
            symbol,
            market=market,
            exchange=exchange,
            broker_instrument_id=broker_instrument_id,
        ).model_dump(mode="json")

    @mcp.tool(annotations=READ_OPEN, structured_output=True)
    async def artha_broker_search_instruments(
        query: str,
        exchange: str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Resolve broker-verified Indian cash-equity identifiers without guessing."""
        require_read()
        return await service.search_instruments(query, exchange=exchange, limit=limit)

    @mcp.tool(annotations=READ_OPEN, structured_output=True)
    async def artha_portfolio() -> dict[str, Any]:
        """Read the configured broker portfolio, with a local Artha fallback when unavailable."""
        require_read()
        return await service.portfolio()

    @mcp.tool(annotations=READ_OPEN, structured_output=True)
    async def artha_broker_quote(
        symbol: str,
        market: str | None = None,
        exchange: str | None = None,
        broker_instrument_id: str | None = None,
    ) -> dict[str, Any]:
        """Fetch a current broker quote; this does not authorize an order."""
        require_read()
        instrument = service.resolve_instrument(
            symbol,
            market=market,
            exchange=exchange,
            broker_instrument_id=broker_instrument_id,
        )
        return await service.quote(instrument)

    @mcp.tool(annotations=READ_OPEN, structured_output=True)
    async def artha_broker_orders(limit: int = 100) -> dict[str, Any]:
        """Read a bounded, redacted broker order history for status verification."""
        require_read()
        return await service.broker_orders(limit=limit)

    @mcp.tool(annotations=READ_OPEN, structured_output=True)
    async def artha_collect_evidence(
        symbol: str,
        market: str | None = None,
        exchange: str | None = None,
        broker_instrument_id: str | None = None,
    ) -> dict[str, Any]:
        """Collect a bounded research packet and explicitly report market-specific data gaps."""
        require_read()
        instrument = service.resolve_instrument(
            symbol,
            market=market,
            exchange=exchange,
            broker_instrument_id=broker_instrument_id,
        )
        return await service.collect_evidence(instrument)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_recent_decisions(
        ticker: str | None = None, limit: int = 20
    ) -> dict[str, Any]:
        """Read recent Council recommendations from the audit journal."""
        require_read()
        return service.recent_decisions(ticker=ticker, limit=limit)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_active_theses() -> dict[str, Any]:
        """Read held and pending theses monitored by Artha's sell side."""
        require_read()
        return service.active_theses()

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_defer_watches(limit: int = 100) -> dict[str, Any]:
        """Read active pullback/defer watches and their re-review zones."""
        require_read()
        return service.defer_watches(limit=limit)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_execution_queue(limit: int = 50) -> dict[str, Any]:
        """Read trade actions, execution orders, and MCP preview receipt states."""
        require_read()
        return service.execution_queue(limit=limit)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_schedule_status() -> dict[str, Any]:
        """Read scheduler slots, buy-capacity state, and recent MCP workflow jobs."""
        require_read()
        return service.schedule_status()

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_list_dossiers(
        ticker: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        """List decision dossiers by stable resource URI without exposing arbitrary files."""
        require_read()
        return service.list_dossiers(ticker=ticker, limit=limit)

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_list_jobs(limit: int = 20) -> dict[str, Any]:
        """List persisted long-running workflow handles."""
        require_read()
        return {
            "status": "PASS",
            "jobs": [
                row.model_dump(mode="json") for row in service.jobs.list(limit=limit)
            ],
        }

    @mcp.tool(annotations=READ_ONLY, structured_output=True)
    def artha_job_status(job_id: str, tail_lines: int = 80) -> dict[str, Any]:
        """Read workflow status and a bounded log tail."""
        require_read()
        return {
            "status": "PASS",
            "job": service.jobs.get(job_id, tail_lines=tail_lines),
        }

    @mcp.tool(annotations=OPERATE, structured_output=True)
    def artha_start_workflow(
        workflow: str,
        symbols: list[str] | None = None,
        limit: int = 8,
        telegram: bool = False,
    ) -> dict[str, Any]:
        """Start one allowlisted Artha workflow and return a persisted job handle."""
        return service.start_workflow(
            workflow,
            symbols=symbols,
            limit=limit,
            telegram=telegram,
            oauth_scopes=_scopes(),
        )

    @mcp.tool(annotations=OPERATE, structured_output=True)
    def artha_cancel_workflow(job_id: str) -> dict[str, Any]:
        """Cancel a workflow that is running in this MCP process."""
        return service.cancel_workflow(job_id, oauth_scopes=_scopes())

    @mcp.tool(annotations=OPERATE, structured_output=True)
    async def artha_preview_order(order: OrderRequest) -> dict[str, Any]:
        """Run broker and local gates, then mint a short-lived immutable preview receipt."""
        return await service.preview_order(order, oauth_scopes=_scopes())

    @mcp.tool(annotations=DESTRUCTIVE, structured_output=True)
    async def artha_place_previewed_order(receipt_id: str) -> dict[str, Any]:
        """Repeat all gates and place only the exact order stored in a valid receipt."""
        return await service.place_order(receipt_id, oauth_scopes=_scopes())

    @mcp.tool(annotations=OPERATE, structured_output=True)
    async def artha_reconcile_execution(receipt_id: str) -> dict[str, Any]:
        """Reconcile a submitted or ambiguous receipt; this never retries an order."""
        return await service.reconcile_order(receipt_id, oauth_scopes=_scopes())

    @mcp.tool(annotations=DESTRUCTIVE, structured_output=True)
    def artha_robinhood_execution_operation(action_id: str) -> dict[str, Any]:
        """Build a trade-scoped Robinhood plan; resolve its account placeholder fail-closed."""
        return service.robinhood_operation(action_id, oauth_scopes=_scopes())

    @mcp.tool(annotations=OPERATE, structured_output=True)
    async def artha_send_notification(message: str) -> dict[str, Any]:
        """Send an operational message through the configured Telegram or HTTPS webhook adapter."""
        return await service.notify(message, oauth_scopes=_scopes())

    @mcp.resource(
        "artha://capabilities", name="Artha capabilities", mime_type="application/json"
    )
    def capabilities_resource() -> str:
        require_read()
        return json.dumps(service.capabilities(), indent=2, ensure_ascii=True)

    @mcp.resource(
        "artha://configuration",
        name="Artha configuration",
        mime_type="application/json",
    )
    def configuration_resource() -> str:
        require_read()
        return json.dumps(service.configuration(), indent=2, ensure_ascii=True)

    @mcp.resource(
        "artha://portfolio", name="Artha portfolio", mime_type="application/json"
    )
    async def portfolio_resource() -> str:
        require_read()
        return json.dumps(
            await service.portfolio(), indent=2, ensure_ascii=True, default=str
        )

    @mcp.resource(
        "artha://schedule", name="Artha schedule", mime_type="application/json"
    )
    def schedule_resource() -> str:
        require_read()
        return json.dumps(
            service.schedule_status(), indent=2, ensure_ascii=True, default=str
        )

    @mcp.resource(
        "artha://research/{artifact_id}",
        name="Research packet",
        mime_type="application/json",
    )
    def research_resource(artifact_id: str) -> str:
        require_read()
        return json.dumps(
            service.research_artifact(artifact_id),
            indent=2,
            ensure_ascii=True,
            default=str,
        )

    @mcp.resource(
        "artha://dossier/{day}/{dossier_id}",
        name="Decision dossier",
        mime_type="application/json",
    )
    def dossier_resource(day: str, dossier_id: str) -> str:
        require_read()
        return json.dumps(
            service.dossier(day, dossier_id), indent=2, ensure_ascii=True, default=str
        )

    @mcp.resource(
        "artha://job/{job_id}", name="Workflow job", mime_type="application/json"
    )
    def job_resource(job_id: str) -> str:
        require_read()
        return json.dumps(
            service.jobs.get(job_id), indent=2, ensure_ascii=True, default=str
        )

    @mcp.resource(
        "artha://execution-receipt/{receipt_id}",
        name="Execution receipt",
        mime_type="application/json",
    )
    def receipt_resource(receipt_id: str) -> str:
        require_read()
        payload = service.execution.get_receipt(receipt_id)
        if payload is None:
            raise ValueError("Execution receipt was not found")
        return json.dumps(redact(payload), indent=2, ensure_ascii=True, default=str)

    @mcp.prompt(name="artha_portfolio_review", title="Review an Artha portfolio")
    def portfolio_review_prompt() -> str:
        return (
            "Read artha://portfolio, artha_active_theses, artha_recent_decisions, and artha_health. "
            "Separate observed broker facts from Artha's theses. Identify thesis changes, concentration, "
            "missing evidence, and operational blockers. Do not recommend or place an order unless a fresh "
            "Council or sell-Council verdict and the execution path both independently pass."
        )

    @mcp.prompt(name="artha_research_symbol", title="Research one equity with Artha")
    def research_symbol_prompt(symbol: str, market: str = "US") -> str:
        return (
            f"Resolve {symbol} for market {market}, collect evidence, inspect prior decisions and active watches, "
            "then assess data completeness before forming a view. For IN, require India-native official/company "
            "fundamentals and broker instrument proof; never treat the US SEC/FMP packet as complete India data."
        )

    @mcp.prompt(name="artha_operations_audit", title="Audit Artha operations")
    def operations_audit_prompt() -> str:
        return (
            "Read artha://configuration, artha_health, artha://schedule, artha_execution_queue, and recent jobs. "
            "Report fail-closed blockers, stale state, missing credentials by capability (never ask for secret values), "
            "and whether research, monitoring, preview, placement, and reconciliation are independently proven."
        )

    @mcp.prompt(
        name="artha_india_onboarding", title="Configure Artha for Indian equities"
    )
    def india_onboarding_prompt() -> str:
        return (
            "Configure market IN with an Upstox, Zerodha, or custom broker adapter. Use INR, NSE/BSE instruments, "
            "whole-share cash-delivery limit orders, Asia/Kolkata sessions, broker-verified instrument identifiers, "
            "a broker-registered static IP, and broker/exchange "
            "market status. Add a licensed India-native research adapter before Council automation. Keep trading disabled "
            "until account, portfolio, quote depth, and preview/margin tests pass. Treat Upstox sandbox as submission-only; "
            "prove order status and reconciliation separately against a live-capable broker environment before enabling live trading."
        )

    return mcp
