"""Versioned OpenClaw handler for Artha Robinhood Telegram callbacks.

OpenClaw owns the Robinhood MCP tools, but the money-moving sequence belongs in
source control. This module is intentionally broker-client agnostic so tests can
prove the exact order:

1. Resolve Artha's opaque Telegram token from SQLite.
2. Run Robinhood tradability.
3. Run Robinhood review.
4. Record tradability + review back into Artha.
5. Only for a Place callback that passes all Artha gates, run Robinhood place.
6. Record the submission/fill back into Artha.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .journal import DecisionJournal
from .portfolio import PORTFOLIO_FILE
from .execution_officer import robinhood_review_final_clearance, run_agentic_execution_officer
from .robinhood_bridge import (
    build_action_operation,
    build_auto_buy_operation,
    record_action_review,
    record_order_submission,
    sync_snapshot_to_artha,
    write_robinhood_snapshot,
)


def _merge_agentic_result(
    journal: DecisionJournal,
    action_id: str,
    agentic: dict[str, Any],
) -> None:
    try:
        row = journal.get_trade_action(action_id) or {}
        result = row.get("result_json")
        if isinstance(result, str):
            try:
                result_json = json.loads(result)
            except Exception:
                result_json = {}
        elif isinstance(result, dict):
            result_json = dict(result)
        else:
            result_json = {}
        result_json["agentic_execution_officer"] = agentic
        journal.update_trade_action(action_id, {"result_json": result_json})
    except Exception:
        return


class RobinhoodMCPClient(Protocol):
    def get_equity_quotes(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def get_equity_tradability(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def review_equity_order(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def place_equity_order(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def get_accounts(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def get_portfolio(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def get_equity_positions(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def get_equity_orders(self, **kwargs: Any) -> dict[str, Any]:
        ...


def refresh_snapshot_from_broker(
    broker: RobinhoodMCPClient,
    *,
    account_number: str,
) -> dict[str, Any]:
    """Run the read-only snapshot sequence and import it into Artha."""
    sync_started_dt = datetime.now(timezone.utc)
    sync_started_at = sync_started_dt.isoformat()
    run_id = f"openclaw-rh-sync-{sync_started_dt.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    accounts = broker.get_accounts()
    portfolio = broker.get_portfolio(account_number=account_number)
    positions = broker.get_equity_positions(account_number=account_number)
    orders = broker.get_equity_orders(account_number=account_number)
    snapshot = {
        "run_id": run_id,
        "generated_at": sync_started_at,
        "accounts_response": accounts,
        "portfolio_response": portfolio,
        "positions_response": positions,
        "orders_response": orders,
    }
    imported = write_robinhood_snapshot(snapshot)
    synced = sync_snapshot_to_artha(imported.get("snapshot"))
    status = "PASS" if imported.get("status") == "PASS" and synced.get("status") == "PASS" else "WARN"
    return {
        "status": status,
        "run_id": run_id,
        "steps": ["accounts", "portfolio", "positions", "orders", "import_snapshot", "sync_snapshot"],
        "snapshot": {k: v for k, v in imported.items() if k != "snapshot"},
        "sync": synced,
    }


def handle_telegram_callback(
    callback_data: str,
    broker: RobinhoodMCPClient,
    *,
    journal: DecisionJournal | None = None,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Execute one Artha Telegram callback through the audited MCP sequence."""
    journal = journal or DecisionJournal()
    operation = build_action_operation(callback_data, journal=journal)
    if not operation.get("success"):
        return {"status": "BLOCKED", "operation": operation, "steps": []}

    op_name = str(operation.get("operation") or "")
    if op_name == "skip":
        return {"status": "PASS", "operation": operation, "steps": ["skip"]}

    if op_name == "tradability_then_review_equity_order":
        tradability = broker.get_equity_tradability(**operation["tradability_mcp_args"])
        review = broker.review_equity_order(**operation["review_mcp_args"])
        recorded = record_action_review(
            str(operation["action_id"]),
            review,
            tradability_response=tradability,
            journal=journal,
        )
        return {
            "status": "PASS" if recorded.get("status") == "review_clear" else "BLOCKED",
            "operation": operation,
            "steps": ["tradability", "review", "record_review"],
            "tradability_response": tradability,
            "review_response": review,
            "recorded_review": recorded,
        }

    if op_name == "tradability_then_review_then_place_equity_order":
        tradability = broker.get_equity_tradability(**operation["tradability_mcp_args"])
        review = broker.review_equity_order(**operation["review_mcp_args"])
        recorded = record_action_review(
            str(operation["action_id"]),
            review,
            tradability_response=tradability,
            journal=journal,
        )
        steps = ["tradability", "review", "record_review"]
        if recorded.get("status") != "review_clear":
            return {
                "status": "BLOCKED",
                "operation": operation,
                "steps": steps,
                "tradability_response": tradability,
                "review_response": review,
                "recorded_review": recorded,
            }

        place_operation = build_action_operation(callback_data, journal=journal)
        if not place_operation.get("success"):
            return {
                "status": "BLOCKED",
                "operation": place_operation,
                "steps": steps,
                "tradability_response": tradability,
                "review_response": review,
                "recorded_review": recorded,
            }

        place = broker.place_equity_order(**place_operation["place_mcp_args"])
        submission = record_order_submission(
            action_id=str(place_operation["action_id"]),
            place_response=place,
            journal=journal,
            portfolio_path=portfolio_path,
        )
        return {
            "status": "PASS" if submission.get("status") == "PASS" else "BLOCKED",
            "operation": place_operation,
            "steps": steps + ["place", "record_submission"],
            "tradability_response": tradability,
            "review_response": review,
            "recorded_review": recorded,
            "place_response": place,
            "recorded_submission": submission,
        }

    return {"status": "BLOCKED", "operation": operation, "steps": [], "message": f"Unsupported operation: {op_name}"}


def handle_auto_buy_action(
    action_id: str,
    broker: RobinhoodMCPClient,
    *,
    journal: DecisionJournal | None = None,
    portfolio_path: str | Path = PORTFOLIO_FILE,
) -> dict[str, Any]:
    """Execute one queued auto-buy action through the audited MCP sequence."""
    journal = journal or DecisionJournal()
    operation = build_auto_buy_operation(action_id, journal=journal)
    if not operation.get("success"):
        return {"status": "BLOCKED", "operation": operation, "steps": []}

    op_name = str(operation.get("operation") or "")
    if op_name == "auto_tradability_review_then_place_equity_order":
        action_row = journal.get_trade_action(str(operation["action_id"])) or {}
        agentic = run_agentic_execution_officer(
            action=action_row,
            operation=operation,
            broker=broker,
            journal=journal,
        )
        if agentic.get("status") != "SKIPPED":
            steps = [
                f"agent_tool:{item.get('tool_name')}"
                for item in (agentic.get("tool_trace") or [])
            ]
            if not agentic.get("allow_place"):
                return {
                    "status": "BLOCKED",
                    "operation": operation,
                    "steps": steps,
                    "agentic_execution_officer": agentic,
                }

            place_operation = build_auto_buy_operation(action_id, journal=journal)
            if not place_operation.get("success"):
                return {
                    "status": "BLOCKED",
                    "operation": place_operation,
                    "steps": steps,
                    "agentic_execution_officer": agentic,
                }

            second_tradability = broker.get_equity_tradability(**place_operation["tradability_mcp_args"])
            second_review = broker.review_equity_order(**place_operation["review_mcp_args"])
            second_recorded = record_action_review(
                str(place_operation["action_id"]),
                second_review,
                tradability_response=second_tradability,
                journal=journal,
            )
            steps.extend(["tradability", "review", "record_review"])
            if second_recorded.get("status") != "review_clear":
                return {
                    "status": "BLOCKED",
                    "operation": place_operation,
                    "steps": steps,
                    "agentic_execution_officer": agentic,
                    "tradability_response": second_tradability,
                    "review_response": second_review,
                    "recorded_review": second_recorded,
                }

            latest_action = journal.get_trade_action(str(place_operation["action_id"])) or {}
            final_clearance = robinhood_review_final_clearance(
                action=latest_action,
                review_response=second_review,
                tradability_response=second_tradability,
                recorded_review=second_recorded,
            )
            if not final_clearance.get("allow_place"):
                journal.update_trade_action(
                    str(place_operation["action_id"]),
                    {
                        "status": "review_blocked",
                        "result_json": {"execution_officer_final_clearance": final_clearance},
                        "notes": "Execution Officer blocked auto-buy after final Robinhood review.",
                    },
                )
                return {
                    "status": "BLOCKED",
                    "operation": place_operation,
                    "steps": steps,
                    "agentic_execution_officer": agentic,
                    "tradability_response": second_tradability,
                    "review_response": second_review,
                    "recorded_review": second_recorded,
                    "execution_officer_final_clearance": final_clearance,
                }

            place = broker.place_equity_order(**place_operation["place_mcp_args"])
            submission = record_order_submission(
                action_id=str(place_operation["action_id"]),
                place_response=place,
                journal=journal,
                portfolio_path=portfolio_path,
            )
            _merge_agentic_result(journal, str(place_operation["action_id"]), agentic)
            return {
                "status": "PASS" if submission.get("status") == "PASS" else "BLOCKED",
                "operation": place_operation,
                "steps": steps + ["place", "record_submission"],
                "agentic_execution_officer": agentic,
                "tradability_response": second_tradability,
                "review_response": second_review,
                "recorded_review": second_recorded,
                "execution_officer_final_clearance": final_clearance,
                "place_response": place,
                "recorded_submission": submission,
            }

        tradability = broker.get_equity_tradability(**operation["tradability_mcp_args"])
        review = broker.review_equity_order(**operation["review_mcp_args"])
        recorded = record_action_review(
            str(operation["action_id"]),
            review,
            tradability_response=tradability,
            journal=journal,
        )
        steps = ["tradability", "review", "record_review"]
        if recorded.get("status") != "review_clear":
            return {
                "status": "BLOCKED",
                "operation": operation,
                "steps": steps,
                "tradability_response": tradability,
                "review_response": review,
                "recorded_review": recorded,
            }

        place_operation = build_auto_buy_operation(action_id, journal=journal)
        if not place_operation.get("success"):
            return {
                "status": "BLOCKED",
                "operation": place_operation,
                "steps": steps,
                "tradability_response": tradability,
                "review_response": review,
                "recorded_review": recorded,
            }

        second_tradability = broker.get_equity_tradability(**place_operation["tradability_mcp_args"])
        second_review = broker.review_equity_order(**place_operation["review_mcp_args"])
        second_recorded = record_action_review(
            str(place_operation["action_id"]),
            second_review,
            tradability_response=second_tradability,
            journal=journal,
        )
        steps.extend(["tradability", "review", "record_review"])
        if second_recorded.get("status") != "review_clear":
            return {
                "status": "BLOCKED",
                "operation": place_operation,
                "steps": steps,
                "tradability_response": second_tradability,
                "review_response": second_review,
                "recorded_review": second_recorded,
            }

        latest_action = journal.get_trade_action(str(place_operation["action_id"])) or {}
        final_clearance = robinhood_review_final_clearance(
            action=latest_action,
            review_response=second_review,
            tradability_response=second_tradability,
            recorded_review=second_recorded,
        )
        if not final_clearance.get("allow_place"):
            journal.update_trade_action(
                str(place_operation["action_id"]),
                {
                    "status": "review_blocked",
                    "result_json": {"execution_officer_final_clearance": final_clearance},
                    "notes": "Execution Officer blocked auto-buy after final Robinhood review.",
                },
            )
            return {
                "status": "BLOCKED",
                "operation": place_operation,
                "steps": steps,
                "tradability_response": second_tradability,
                "review_response": second_review,
                "recorded_review": second_recorded,
                "execution_officer_final_clearance": final_clearance,
            }

        place = broker.place_equity_order(**place_operation["place_mcp_args"])
        submission = record_order_submission(
            action_id=str(place_operation["action_id"]),
            place_response=place,
            journal=journal,
            portfolio_path=portfolio_path,
        )
        return {
            "status": "PASS" if submission.get("status") == "PASS" else "BLOCKED",
            "operation": place_operation,
            "steps": steps + ["place", "record_submission"],
            "tradability_response": tradability,
            "review_response": review,
            "recorded_review": recorded,
            "second_tradability_response": second_tradability,
            "second_review_response": second_review,
            "second_recorded_review": second_recorded,
            "execution_officer_final_clearance": final_clearance,
            "place_response": place,
            "recorded_submission": submission,
        }

    if op_name == "tradability_then_review_then_place_equity_order":
        tradability = broker.get_equity_tradability(**operation["tradability_mcp_args"])
        review = broker.review_equity_order(**operation["review_mcp_args"])
        recorded = record_action_review(
            str(operation["action_id"]),
            review,
            tradability_response=tradability,
            journal=journal,
        )
        steps = ["tradability", "review", "record_review"]
        if recorded.get("status") != "review_clear":
            return {
                "status": "BLOCKED",
                "operation": operation,
                "steps": steps,
                "tradability_response": tradability,
                "review_response": review,
                "recorded_review": recorded,
            }
        latest_action = journal.get_trade_action(str(operation["action_id"])) or {}
        final_clearance = robinhood_review_final_clearance(
            action=latest_action,
            review_response=review,
            tradability_response=tradability,
            recorded_review=recorded,
        )
        if not final_clearance.get("allow_place"):
            journal.update_trade_action(
                str(operation["action_id"]),
                {
                    "status": "review_blocked",
                    "result_json": {"execution_officer_final_clearance": final_clearance},
                    "notes": "Execution Officer blocked auto-buy after final Robinhood review.",
                },
            )
            return {
                "status": "BLOCKED",
                "operation": operation,
                "steps": steps,
                "tradability_response": tradability,
                "review_response": review,
                "recorded_review": recorded,
                "execution_officer_final_clearance": final_clearance,
            }
        place = broker.place_equity_order(**operation["place_mcp_args"])
        submission = record_order_submission(
            action_id=str(operation["action_id"]),
            place_response=place,
            journal=journal,
            portfolio_path=portfolio_path,
        )
        return {
            "status": "PASS" if submission.get("status") == "PASS" else "BLOCKED",
            "operation": operation,
            "steps": steps + ["place", "record_submission"],
            "tradability_response": tradability,
            "review_response": review,
            "recorded_review": recorded,
            "execution_officer_final_clearance": final_clearance,
            "place_response": place,
            "recorded_submission": submission,
        }

    return {"status": "BLOCKED", "operation": operation, "steps": [], "message": f"Unsupported auto-buy operation: {op_name}"}


class ReplayRobinhoodBroker:
    """Replay MCP responses collected by an OpenClaw cron turn.

    Artha cannot call OpenClaw-owned Robinhood MCP tools from a plain Python
    process. The unattended runner therefore collects the live MCP responses in
    OpenClaw, then replays those exact responses into the source-controlled
    Execution Officer tool loop.
    """

    def __init__(
        self,
        *,
        quote_response: dict[str, Any],
        tradability_response: dict[str, Any],
        review_response: dict[str, Any],
    ) -> None:
        self.quote_response = quote_response
        self.tradability_response = tradability_response
        self.review_response = review_response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_equity_quotes(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("quote", kwargs))
        return self.quote_response

    def get_equity_tradability(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("tradability", kwargs))
        return self.tradability_response

    def review_equity_order(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("review", kwargs))
        return self.review_response


def run_agentic_auto_buy_clearance_from_responses(
    action_id: str,
    *,
    quote_response: dict[str, Any],
    tradability_response: dict[str, Any],
    review_response: dict[str, Any],
    journal: DecisionJournal | None = None,
) -> dict[str, Any]:
    """Run the GPT-5.5 Execution Officer using OpenClaw-collected MCP data.

    This is the bridge between an unattended OpenClaw cron and Artha's
    source-controlled auto-buy safety logic. It records the Robinhood review
    only if the agentic officer actually calls the review tool, and it blocks
    placement unless the officer used the required live broker tools.
    """
    journal = journal or DecisionJournal()
    operation = build_auto_buy_operation(action_id, journal=journal)
    if not operation.get("success"):
        return {"status": "BLOCKED", "allow_place": False, "operation": operation, "steps": []}
    if str(operation.get("operation") or "") != "auto_tradability_review_then_place_equity_order":
        return {
            "status": "BLOCKED",
            "allow_place": False,
            "operation": operation,
            "steps": [],
            "message": f"Unsupported agentic clearance operation: {operation.get('operation')}",
        }

    action_row = journal.get_trade_action(str(operation["action_id"])) or {}
    replay = ReplayRobinhoodBroker(
        quote_response=quote_response,
        tradability_response=tradability_response,
        review_response=review_response,
    )
    agentic = run_agentic_execution_officer(
        action=action_row,
        operation=operation,
        broker=replay,
        journal=journal,
    )
    _merge_agentic_result(journal, str(operation["action_id"]), agentic)
    allow_place = bool(agentic.get("allow_place"))
    if agentic.get("status") == "SKIPPED":
        allow_place = False
        agentic = {
            **agentic,
            "status": "BLOCKED",
            "reason": "Agentic Execution Officer is disabled; unattended auto-buy requires agentic clearance.",
        }
    if not allow_place:
        row = journal.get_trade_action(str(operation["action_id"])) or {}
        raw_result = row.get("result_json")
        if isinstance(raw_result, str):
            try:
                result = json.loads(raw_result)
            except Exception:
                result = {}
        elif isinstance(raw_result, dict):
            result = dict(raw_result)
        else:
            result = {}
        result["agentic_execution_officer"] = agentic
        journal.update_trade_action(
            str(operation["action_id"]),
            {
                "status": "review_blocked",
                "result_json": result,
                "notes": "Agentic Execution Officer blocked unattended auto-buy before placement.",
            },
        )
    return {
        "status": "PASS" if allow_place else "BLOCKED",
        "allow_place": allow_place,
        "operation": operation,
        "steps": [f"agent_tool:{item.get('tool_name')}" for item in (agentic.get("tool_trace") or [])],
        "agentic_execution_officer": agentic,
        "replayed_broker_calls": replay.calls,
    }
