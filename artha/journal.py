"""SQLite decision journal for recommendations, sessions, and snapshots.

This module provides append-focused persistence for council outputs and
portfolio state snapshots used by Artha context injection.
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "artha.db"


class DecisionJournal:
    """Thin SQLite wrapper for Artha context and history storage."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @staticmethod
    def _utcnow_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create required tables if they do not already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    session_id TEXT,
                    ticker TEXT,
                    action TEXT,
                    rationale TEXT,
                    confidence INTEGER,
                    price_at_recommendation REAL,
                    conditions TEXT,
                    status TEXT DEFAULT 'open',
                    outcome TEXT,
                    outcome_notes TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    total_value REAL,
                    cash REAL,
                    holdings_json TEXT,
                    summary TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    session_type TEXT,
                    tickers_analyzed TEXT,
                    report_path TEXT,
                    created_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_positions (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    ticker TEXT,
                    sector TEXT,
                    benchmark_ticker TEXT,
                    sector_benchmark_ticker TEXT,
                    thesis_type TEXT,
                    blocked_by TEXT,
                    blocked_reason TEXT,
                    hypothetical_entry REAL,
                    hypothetical_stop REAL,
                    opportunity_score REAL,
                    regime TEXT,
                    fear_greed INTEGER,
                    price_5d REAL,
                    price_20d REAL,
                    price_60d REAL,
                    return_5d REAL,
                    return_20d REAL,
                    return_60d REAL,
                    benchmark_price_entry REAL,
                    benchmark_price_5d REAL,
                    benchmark_price_20d REAL,
                    benchmark_price_60d REAL,
                    benchmark_return_5d REAL,
                    benchmark_return_20d REAL,
                    benchmark_return_60d REAL,
                    sector_benchmark_price_entry REAL,
                    sector_benchmark_price_5d REAL,
                    sector_benchmark_price_20d REAL,
                    sector_benchmark_price_60d REAL,
                    sector_benchmark_return_5d REAL,
                    sector_benchmark_return_20d REAL,
                    sector_benchmark_return_60d REAL,
                    excess_return_5d REAL,
                    excess_return_20d REAL,
                    excess_return_60d REAL,
                    mfe REAL,
                    mae REAL,
                    would_hit_stop INTEGER,
                    status TEXT DEFAULT 'tracking',
                    created_at TEXT
                )
                """
            )
            # ---------------------------------------------------------------
            # Sell-side tables (added sell-engine implementation)
            # ---------------------------------------------------------------
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS position_theses (
                    id INTEGER PRIMARY KEY,
                    thesis_id TEXT UNIQUE NOT NULL,
                    ticker TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    position_type TEXT NOT NULL,
                    council_session_id TEXT,
                    thesis_summary TEXT,
                    invalidation_conditions TEXT,
                    price_target REAL,
                    stop_loss_pct REAL,
                    stop_loss_price REAL,
                    recommended_allocation_pct REAL,
                    entry_price REAL,
                    entry_date TEXT,
                    entry_regime TEXT,
                    hard_stop_price REAL,
                    trailing_stop_price REAL,
                    trailing_stop_high REAL,
                    thesis_health_score INTEGER DEFAULT 100,
                    last_review_date TEXT,
                    next_review_date TEXT,
                    sell_cooldown_until TEXT,
                    scale_out_completed TEXT DEFAULT '[]',
                    exit_date TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    pending_expiry TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_position_theses_ticker ON position_theses(ticker)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_position_theses_status ON position_theses(status)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sell_signals (
                    id INTEGER PRIMARY KEY,
                    signal_id TEXT UNIQUE NOT NULL,
                    ticker TEXT NOT NULL,
                    thesis_id TEXT,
                    signal_type TEXT NOT NULL,
                    severity TEXT NOT NULL DEFAULT 'MEDIUM',
                    source TEXT NOT NULL,
                    message TEXT,
                    sell_score REAL,
                    action_recommended TEXT,
                    confirmed INTEGER DEFAULT 0,
                    confirmed_at TEXT,
                    actioned INTEGER DEFAULT 0,
                    actioned_at TEXT,
                    suppressed INTEGER DEFAULT 0,
                    suppressed_reason TEXT,
                    expires_at TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sell_signals_ticker ON sell_signals(ticker)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sell_signals_status ON sell_signals(actioned, suppressed)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sell_sessions (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT UNIQUE NOT NULL,
                    ticker TEXT NOT NULL,
                    thesis_id TEXT,
                    trigger_type TEXT NOT NULL,
                    fundamental_verdict TEXT,
                    fundamental_report TEXT,
                    technical_verdict TEXT,
                    technical_report TEXT,
                    contrarian_verdict TEXT,
                    contrarian_report TEXT,
                    sell_score REAL,
                    action TEXT,
                    synthesis_report TEXT,
                    next_review_date TEXT,
                    health_score_after INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sell_sessions_ticker ON sell_sessions(ticker)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS post_sell_tracking (
                    id INTEGER PRIMARY KEY,
                    tracking_id TEXT UNIQUE NOT NULL,
                    ticker TEXT NOT NULL,
                    thesis_id TEXT,
                    sell_date TEXT NOT NULL,
                    sell_price REAL NOT NULL,
                    sell_reason TEXT,
                    position_type TEXT,
                    shares REAL,
                    price_5d REAL,
                    price_20d REAL,
                    price_60d REAL,
                    return_5d REAL,
                    return_20d REAL,
                    return_60d REAL,
                    regret_score REAL,
                    grade TEXT,
                    status TEXT DEFAULT 'tracking',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_post_sell_tracking_ticker ON post_sell_tracking(ticker)"
            )
            # ---------------------------------------------------------------
            # Entry watchlist tables
            # ---------------------------------------------------------------
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defer_watchlist (
                    id INTEGER PRIMARY KEY,
                    watch_id TEXT UNIQUE NOT NULL,
                    ticker TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    source_action TEXT,
                    current_price REAL,
                    zone_low REAL,
                    zone_high REAL,
                    trigger_type TEXT DEFAULT 'zone',
                    trigger_text TEXT,
                    invalidation_conditions TEXT,
                    opportunity_score REAL,
                    confidence INTEGER,
                    entry_valid_until TEXT,
                    dossier_path TEXT,
                    trace_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    triggered_at TEXT,
                    trigger_price REAL,
                    notes TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defer_watchlist_ticker ON defer_watchlist(ticker)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defer_watchlist_status ON defer_watchlist(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defer_watchlist_expiry ON defer_watchlist(entry_valid_until)"
            )
            # ---------------------------------------------------------------
            # Point-in-time decision feature warehouse
            # ---------------------------------------------------------------
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_features (
                    id INTEGER PRIMARY KEY,
                    dossier_path TEXT UNIQUE NOT NULL,
                    generated_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    final_verdict TEXT,
                    opportunity_score REAL,
                    adjusted_score REAL,
                    confidence INTEGER,
                    price REAL,
                    market_cap REAL,
                    sector TEXT,
                    industry TEXT,
                    evidence_count INTEGER,
                    context_coverage_score REAL,
                    completeness_score REAL,
                    source_count INTEGER,
                    gap_count INTEGER,
                    valuation_signal TEXT,
                    consensus_upside_pct REAL,
                    expectation_risk_level TEXT,
                    portfolio_risk_level TEXT,
                    portfolio_sector_after_pct REAL,
                    benchmark_ticker TEXT,
                    feature_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            self._ensure_columns(
                conn,
                "shadow_positions",
                {
                    "sector": "TEXT",
                    "benchmark_ticker": "TEXT",
                    "sector_benchmark_ticker": "TEXT",
                    "benchmark_price_entry": "REAL",
                    "benchmark_price_5d": "REAL",
                    "benchmark_price_20d": "REAL",
                    "benchmark_price_60d": "REAL",
                    "benchmark_return_5d": "REAL",
                    "benchmark_return_20d": "REAL",
                    "benchmark_return_60d": "REAL",
                    "sector_benchmark_price_entry": "REAL",
                    "sector_benchmark_price_5d": "REAL",
                    "sector_benchmark_price_20d": "REAL",
                    "sector_benchmark_price_60d": "REAL",
                    "sector_benchmark_return_5d": "REAL",
                    "sector_benchmark_return_20d": "REAL",
                    "sector_benchmark_return_60d": "REAL",
                    "excess_return_5d": "REAL",
                    "excess_return_20d": "REAL",
                    "excess_return_60d": "REAL",
                },
            )
            self._ensure_columns(
                conn,
                "decision_features",
                {
                    "valuation_signal": "TEXT",
                    "consensus_upside_pct": "REAL",
                    "expectation_risk_level": "TEXT",
                    "portfolio_risk_level": "TEXT",
                    "portfolio_sector_after_pct": "REAL",
                    "benchmark_ticker": "TEXT",
                },
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decision_features_ticker ON decision_features(ticker)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decision_features_generated ON decision_features(generated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_decision_features_verdict ON decision_features(final_verdict)"
            )
            # ---------------------------------------------------------------
            # Calibration diagnostics and self-improvement audit trail
            # ---------------------------------------------------------------
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calibration_diagnostics (
                    id INTEGER PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    completed_samples INTEGER NOT NULL,
                    stage TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    report_hash TEXT NOT NULL,
                    report_text TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    sent_to_telegram INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_calibration_diagnostics_generated ON calibration_diagnostics(generated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_calibration_diagnostics_hash ON calibration_diagnostics(report_hash)"
            )
            # ---------------------------------------------------------------
            # Shadow rule engine: proposed investing-rule changes run here
            # only as private counterfactuals. These rows never change live
            # recommendations or allocation policy.
            # ---------------------------------------------------------------
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_rule_evaluations (
                    id INTEGER PRIMARY KEY,
                    evaluation_id TEXT UNIQUE NOT NULL,
                    rule_id TEXT NOT NULL,
                    rule_version TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    dossier_path TEXT,
                    decision_generated_at TEXT,
                    real_action TEXT,
                    shadow_action TEXT,
                    rule_status TEXT NOT NULL DEFAULT 'shadow_mode',
                    trigger_reason TEXT,
                    evidence_json TEXT,
                    hypothetical_entry REAL,
                    benchmark_ticker TEXT,
                    sector_benchmark_ticker TEXT,
                    price_5d REAL,
                    price_10d REAL,
                    price_20d REAL,
                    price_60d REAL,
                    return_5d REAL,
                    return_10d REAL,
                    return_20d REAL,
                    return_60d REAL,
                    benchmark_price_entry REAL,
                    benchmark_price_5d REAL,
                    benchmark_price_10d REAL,
                    benchmark_price_20d REAL,
                    benchmark_price_60d REAL,
                    benchmark_return_5d REAL,
                    benchmark_return_10d REAL,
                    benchmark_return_20d REAL,
                    benchmark_return_60d REAL,
                    sector_benchmark_price_entry REAL,
                    sector_benchmark_price_5d REAL,
                    sector_benchmark_price_10d REAL,
                    sector_benchmark_price_20d REAL,
                    sector_benchmark_price_60d REAL,
                    sector_benchmark_return_5d REAL,
                    sector_benchmark_return_10d REAL,
                    sector_benchmark_return_20d REAL,
                    sector_benchmark_return_60d REAL,
                    excess_return_5d REAL,
                    excess_return_10d REAL,
                    excess_return_20d REAL,
                    excess_return_60d REAL,
                    mfe REAL,
                    mae REAL,
                    would_hit_stop INTEGER,
                    status TEXT DEFAULT 'tracking',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_rule_eval_rule ON shadow_rule_evaluations(rule_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_rule_eval_status ON shadow_rule_evaluations(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_rule_eval_ticker ON shadow_rule_evaluations(ticker)"
            )
            # ---------------------------------------------------------------
            # Execution audit trail: Robinhood-ready order proposals and
            # dry-run/live broker responses. This table is append-focused so
            # every blocked or rehearsed order remains inspectable.
            # ---------------------------------------------------------------
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS execution_orders (
                    id INTEGER PRIMARY KEY,
                    order_intent_id TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    time_in_force TEXT,
                    quantity REAL,
                    notional REAL,
                    limit_price REAL,
                    estimated_price REAL,
                    status TEXT NOT NULL,
                    broker TEXT NOT NULL DEFAULT 'robinhood',
                    broker_order_id TEXT,
                    dry_run INTEGER DEFAULT 1,
                    decision_dossier_path TEXT,
                    recommendation_id INTEGER,
                    thesis_id TEXT,
                    supervisor_run_id INTEGER,
                    guardrail_status TEXT,
                    guardrail_json TEXT,
                    rationale TEXT,
                    evidence_json TEXT,
                    request_json TEXT,
                    response_json TEXT,
                    submitted_at TEXT,
                    filled_at TEXT,
                    canceled_at TEXT,
                    notes TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_execution_orders_created ON execution_orders(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_execution_orders_ticker ON execution_orders(ticker)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_execution_orders_status ON execution_orders(status)"
            )
            # ---------------------------------------------------------------
            # Scheduled scan broker/data router: every candidate considered
            # for a buy-now Council slot is persisted with the execution/data
            # evidence that routed it to buy-now, research/watch, or reject.
            # This router is explicitly not a company-risk judge; that remains
            # Council/risk sizing work.
            # ---------------------------------------------------------------
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_routing_decisions (
                    id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    candidate_rank INTEGER,
                    lane TEXT NOT NULL,
                    bucket TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    reason TEXT,
                    route_score REAL,
                    funnel_score REAL,
                    price REAL,
                    live_price REAL,
                    bid REAL,
                    ask REAL,
                    spread_pct REAL,
                    avg_volume REAL,
                    dollar_volume REAL,
                    liquidity_source TEXT,
                    quote_source TEXT,
                    evidence_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scan_routing_session ON scan_routing_decisions(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scan_routing_ticker ON scan_routing_decisions(ticker)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scan_routing_lane ON scan_routing_decisions(lane)"
            )
            # ---------------------------------------------------------------
            # Human approval/action queue for OpenClaw/Ammu. Artha can prepare
            # an audited order review, then publish durable Telegram callback
            # tokens. OpenClaw consumes the token, re-checks guardrails, calls
            # Robinhood MCP only after a user click, and writes the outcome back.
            # ---------------------------------------------------------------
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_actions (
                    id INTEGER PRIMARY KEY,
                    action_id TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    action_type TEXT NOT NULL,
                    ticker TEXT,
                    side TEXT,
                    execution_order_row INTEGER,
                    order_intent_id TEXT,
                    thesis_id TEXT,
                    account_number_masked TEXT,
                    token_review TEXT,
                    token_place TEXT,
                    token_skip TEXT,
                    payload_json TEXT,
                    result_json TEXT,
                    message TEXT,
                    notes TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_actions_status ON trade_actions(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trade_actions_ticker ON trade_actions(ticker)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_actions_token_review ON trade_actions(token_review)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_actions_token_place ON trade_actions(token_place)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_actions_token_skip ON trade_actions(token_skip)"
            )
            # ---------------------------------------------------------------
            # Pending market-open order rechecks. These are not orders. They
            # are durable instructions for the monitor to refresh a prior
            # buy-side council call at the next market open before any broker
            # review or user-facing trade prompt.
            # ---------------------------------------------------------------
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_order_rechecks (
                    id INTEGER PRIMARY KEY,
                    recheck_id TEXT UNIQUE NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    run_after TEXT NOT NULL,
                    expires_at TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    ticker TEXT NOT NULL,
                    original_verdict TEXT,
                    source_session_id TEXT,
                    source_recommendation_id INTEGER,
                    original_dossier_path TEXT,
                    original_action TEXT,
                    original_price REAL,
                    max_price REAL,
                    notional REAL,
                    account_number_masked TEXT,
                    last_reviewed_at TEXT,
                    last_verdict TEXT,
                    last_price REAL,
                    execution_order_row INTEGER,
                    notes TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_order_rechecks_due ON pending_order_rechecks(status, run_after)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_order_rechecks_ticker ON pending_order_rechecks(ticker)"
            )
            # ---------------------------------------------------------------
            # Supervisor health checks: one row per run for audit and
            # Telegram de-duplication.
            # ---------------------------------------------------------------
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS supervisor_runs (
                    id INTEGER PRIMARY KEY,
                    generated_at TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    report_hash TEXT NOT NULL,
                    report_text TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    sent_to_telegram INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_supervisor_runs_generated ON supervisor_runs(generated_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_supervisor_runs_hash ON supervisor_runs(report_hash)"
            )
            conn.commit()

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        """Add missing columns for lightweight SQLite migrations."""
        existing = {
            str(row[1])
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        for name, ddl in columns.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

    def save_recommendation(
        self,
        session_id: str,
        ticker: str,
        action: str,
        rationale: str,
        confidence: int,
        price_at_recommendation: float | None,
        conditions: str = "",
        status: str = "open",
        outcome: str = "unknown",
        outcome_notes: str = "",
        timestamp: str | None = None,
    ) -> int:
        """Persist a council recommendation and return inserted row id."""
        ts = timestamp or self._utcnow_iso()
        created = self._utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO recommendations (
                    timestamp, session_id, ticker, action, rationale, confidence,
                    price_at_recommendation, conditions, status, outcome, outcome_notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    (session_id or "").strip(),
                    (ticker or "").upper().strip(),
                    (action or "").upper().strip(),
                    rationale,
                    int(confidence),
                    float(price_at_recommendation) if price_at_recommendation is not None else None,
                    conditions,
                    status,
                    outcome,
                    outcome_notes,
                    created,
                ),
            )
            conn.commit()
            row_id = int(cursor.lastrowid)
        logger.info("Saved recommendation row=%s ticker=%s action=%s", row_id, ticker, action)
        return row_id

    def save_snapshot(
        self,
        total_value: float,
        cash: float,
        holdings_json: str,
        summary: str,
        timestamp: str | None = None,
    ) -> int:
        """Persist a portfolio snapshot and return inserted row id."""
        ts = timestamp or self._utcnow_iso()
        created = self._utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO portfolio_snapshots (
                    timestamp, total_value, cash, holdings_json, summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, float(total_value), float(cash), holdings_json, summary, created),
            )
            conn.commit()
            row_id = int(cursor.lastrowid)
        logger.info("Saved portfolio snapshot row=%s total_value=%.2f", row_id, total_value)
        return row_id

    def save_session(
        self,
        session_type: str,
        tickers_analyzed: str,
        report_path: str,
        timestamp: str | None = None,
    ) -> int:
        """Persist a session execution log and return inserted row id."""
        ts = timestamp or self._utcnow_iso()
        created = self._utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (
                    timestamp, session_type, tickers_analyzed, report_path, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (ts, session_type, tickers_analyzed, report_path, created),
            )
            conn.commit()
            row_id = int(cursor.lastrowid)
        logger.info("Saved session row=%s type=%s", row_id, session_type)
        return row_id

    def get_recent_recommendations(self, ticker: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get the most recent recommendations for a ticker (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, session_id, ticker, action, rationale, confidence,
                       price_at_recommendation, conditions, status, outcome, outcome_notes, created_at
                FROM recommendations
                WHERE ticker = ?
                ORDER BY datetime(timestamp) DESC, id DESC
                LIMIT ?
                """,
                ((ticker or "").upper().strip(), int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def log_shadow_trade(
        self,
        ticker: str,
        thesis_type: str,
        blocked_by: str,
        blocked_reason: str,
        hypothetical_entry: float,
        hypothetical_stop: float,
        opportunity_score: float,
        regime: str,
        fear_greed: int,
        sector: str = "",
        benchmark_ticker: str = "",
        sector_benchmark_ticker: str = "",
    ) -> int:
        """Log a trade that was blocked (WATCH override) for counterfactual tracking."""
        ts = self._utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO shadow_positions (
                    timestamp, ticker, sector, benchmark_ticker, sector_benchmark_ticker,
                    thesis_type, blocked_by, blocked_reason,
                    hypothetical_entry, hypothetical_stop, opportunity_score,
                    regime, fear_greed, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'tracking', ?)
                """,
                (
                    ts,
                    (ticker or "").upper().strip(),
                    sector,
                    (benchmark_ticker or "").upper().strip(),
                    (sector_benchmark_ticker or "").upper().strip(),
                    thesis_type,
                    blocked_by,
                    blocked_reason,
                    float(hypothetical_entry),
                    float(hypothetical_stop),
                    float(opportunity_score),
                    regime,
                    int(fear_greed),
                    ts,
                ),
            )
            conn.commit()
            row_id = int(cursor.lastrowid)
        logger.info("Logged shadow trade row=%s ticker=%s score=%.1f", row_id, ticker, opportunity_score)
        return row_id

    def get_pending_shadow_reviews(self) -> list[dict[str, Any]]:
        """Get shadow trades that need forward return updates (tracking status)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM shadow_positions
                WHERE status = 'tracking'
                ORDER BY datetime(created_at) ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def update_shadow_returns(
        self,
        shadow_id: int,
        price_5d: float | None = None,
        price_20d: float | None = None,
        price_60d: float | None = None,
        mfe: float | None = None,
        mae: float | None = None,
        would_hit_stop: bool | None = None,
        **extra_fields: Any,
    ) -> None:
        """Update forward return data for a shadow trade."""
        updates: list[str] = []
        params: list[Any] = []

        if price_5d is not None:
            updates.append("price_5d = ?")
            params.append(float(price_5d))
        if price_20d is not None:
            updates.append("price_20d = ?")
            params.append(float(price_20d))
        if price_60d is not None:
            updates.append("price_60d = ?")
            params.append(float(price_60d))
        if mfe is not None:
            updates.append("mfe = ?")
            params.append(float(mfe))
        if mae is not None:
            updates.append("mae = ?")
            params.append(float(mae))
        if would_hit_stop is not None:
            updates.append("would_hit_stop = ?")
            params.append(1 if would_hit_stop else 0)

        allowed_extra = {
            "benchmark_price_entry",
            "benchmark_price_5d",
            "benchmark_price_20d",
            "benchmark_price_60d",
            "sector_benchmark_price_entry",
            "sector_benchmark_price_5d",
            "sector_benchmark_price_20d",
            "sector_benchmark_price_60d",
        }
        for key, value in extra_fields.items():
            if key not in allowed_extra or value is None:
                continue
            updates.append(f"{key} = ?")
            params.append(float(value))

        # Compute returns from prices if we have the entry price
        # Mark as completed if 60d data is available
        with self._connect() as conn:
            if updates:
                conn.execute(
                    f"UPDATE shadow_positions SET {', '.join(updates)} WHERE id = ?",
                    (*params, int(shadow_id)),
                )
            # Compute derived returns when all three price columns will be set
            conn.execute(
                """
                UPDATE shadow_positions
                SET
                    return_5d = CASE WHEN price_5d IS NOT NULL AND hypothetical_entry > 0
                                     THEN (price_5d - hypothetical_entry) / hypothetical_entry
                                     ELSE return_5d END,
                    return_20d = CASE WHEN price_20d IS NOT NULL AND hypothetical_entry > 0
                                      THEN (price_20d - hypothetical_entry) / hypothetical_entry
                                      ELSE return_20d END,
                    return_60d = CASE WHEN price_60d IS NOT NULL AND hypothetical_entry > 0
                                      THEN (price_60d - hypothetical_entry) / hypothetical_entry
                                      ELSE return_60d END,
                    benchmark_return_5d = CASE WHEN benchmark_price_5d IS NOT NULL AND benchmark_price_entry > 0
                                     THEN (benchmark_price_5d - benchmark_price_entry) / benchmark_price_entry
                                     ELSE benchmark_return_5d END,
                    benchmark_return_20d = CASE WHEN benchmark_price_20d IS NOT NULL AND benchmark_price_entry > 0
                                      THEN (benchmark_price_20d - benchmark_price_entry) / benchmark_price_entry
                                      ELSE benchmark_return_20d END,
                    benchmark_return_60d = CASE WHEN benchmark_price_60d IS NOT NULL AND benchmark_price_entry > 0
                                      THEN (benchmark_price_60d - benchmark_price_entry) / benchmark_price_entry
                                      ELSE benchmark_return_60d END,
                    sector_benchmark_return_5d = CASE WHEN sector_benchmark_price_5d IS NOT NULL AND sector_benchmark_price_entry > 0
                                     THEN (sector_benchmark_price_5d - sector_benchmark_price_entry) / sector_benchmark_price_entry
                                     ELSE sector_benchmark_return_5d END,
                    sector_benchmark_return_20d = CASE WHEN sector_benchmark_price_20d IS NOT NULL AND sector_benchmark_price_entry > 0
                                      THEN (sector_benchmark_price_20d - sector_benchmark_price_entry) / sector_benchmark_price_entry
                                      ELSE sector_benchmark_return_20d END,
                    sector_benchmark_return_60d = CASE WHEN sector_benchmark_price_60d IS NOT NULL AND sector_benchmark_price_entry > 0
                                      THEN (sector_benchmark_price_60d - sector_benchmark_price_entry) / sector_benchmark_price_entry
                                      ELSE sector_benchmark_return_60d END,
                    excess_return_5d = CASE WHEN price_5d IS NOT NULL AND hypothetical_entry > 0
                                     THEN ((price_5d - hypothetical_entry) / hypothetical_entry)
                                          - COALESCE(
                                              CASE WHEN sector_benchmark_price_5d IS NOT NULL AND sector_benchmark_price_entry > 0
                                                   THEN (sector_benchmark_price_5d - sector_benchmark_price_entry) / sector_benchmark_price_entry END,
                                              CASE WHEN benchmark_price_5d IS NOT NULL AND benchmark_price_entry > 0
                                                   THEN (benchmark_price_5d - benchmark_price_entry) / benchmark_price_entry END,
                                              0
                                          )
                                     ELSE excess_return_5d END,
                    excess_return_20d = CASE WHEN price_20d IS NOT NULL AND hypothetical_entry > 0
                                      THEN ((price_20d - hypothetical_entry) / hypothetical_entry)
                                           - COALESCE(
                                               CASE WHEN sector_benchmark_price_20d IS NOT NULL AND sector_benchmark_price_entry > 0
                                                    THEN (sector_benchmark_price_20d - sector_benchmark_price_entry) / sector_benchmark_price_entry END,
                                               CASE WHEN benchmark_price_20d IS NOT NULL AND benchmark_price_entry > 0
                                                    THEN (benchmark_price_20d - benchmark_price_entry) / benchmark_price_entry END,
                                               0
                                           )
                                      ELSE excess_return_20d END,
                    excess_return_60d = CASE WHEN price_60d IS NOT NULL AND hypothetical_entry > 0
                                      THEN ((price_60d - hypothetical_entry) / hypothetical_entry)
                                           - COALESCE(
                                               CASE WHEN sector_benchmark_price_60d IS NOT NULL AND sector_benchmark_price_entry > 0
                                                    THEN (sector_benchmark_price_60d - sector_benchmark_price_entry) / sector_benchmark_price_entry END,
                                               CASE WHEN benchmark_price_60d IS NOT NULL AND benchmark_price_entry > 0
                                                    THEN (benchmark_price_60d - benchmark_price_entry) / benchmark_price_entry END,
                                               0
                                           )
                                      ELSE excess_return_60d END,
                    status = CASE WHEN price_60d IS NOT NULL THEN 'completed' ELSE status END
                WHERE id = ?
                """,
                (int(shadow_id),),
            )
            conn.commit()

    def get_shadow_trade_stats(self) -> dict[str, Any]:
        """Aggregate shadow trade stats: hit rate, avg returns, blocked-by breakdown."""
        with self._connect() as conn:
            all_rows = conn.execute("SELECT * FROM shadow_positions").fetchall()
        rows = [dict(r) for r in all_rows]

        if not rows:
            return {"total": 0, "completed": 0, "tracking": 0, "blocked_by": {}, "avg_returns": {}}

        completed = [r for r in rows if r.get("status") == "completed"]
        tracking = [r for r in rows if r.get("status") == "tracking"]

        # Blocked-by breakdown
        blocked_by: dict[str, int] = {}
        for r in rows:
            key = str(r.get("blocked_by") or "unknown")
            blocked_by[key] = blocked_by.get(key, 0) + 1

        # Average returns
        def _avg(lst: list, key: str) -> float | None:
            vals = [r[key] for r in lst if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        avg_returns = {
            "return_5d": _avg(completed, "return_5d"),
            "return_20d": _avg(completed, "return_20d"),
            "return_60d": _avg(completed, "return_60d"),
            "excess_return_5d": _avg(completed, "excess_return_5d"),
            "excess_return_20d": _avg(completed, "excess_return_20d"),
            "excess_return_60d": _avg(completed, "excess_return_60d"),
        }

        hit_stop_count = sum(1 for r in completed if r.get("would_hit_stop"))

        return {
            "total": len(rows),
            "completed": len(completed),
            "tracking": len(tracking),
            "blocked_by": blocked_by,
            "avg_returns": avg_returns,
            "would_hit_stop_rate": round(hit_stop_count / len(completed), 3) if completed else None,
        }

    # -----------------------------------------------------------------------
    # Sell-Side Methods
    # -----------------------------------------------------------------------

    def save_thesis(self, thesis_data: dict[str, Any]) -> None:
        """Insert or update a position thesis record (upsert by thesis_id).

        FIX 11: Optimistic locking — if the DB row's updated_at is newer than
        the version we loaded (thesis_data["updated_at"]), a concurrent update
        occurred and we skip the write to avoid overwriting it.
        """
        ts = self._utcnow_iso()
        thesis_data.setdefault("created_at", ts)

        # Optimistic lock: compare DB updated_at vs what we loaded before modifying
        thesis_id = thesis_data.get("thesis_id")
        incoming_updated_at = thesis_data.get("updated_at")  # value at load time
        if thesis_id and incoming_updated_at:
            with self._connect() as conn:
                existing = conn.execute(
                    "SELECT updated_at FROM position_theses WHERE thesis_id = ?",
                    (thesis_id,),
                ).fetchone()
            if existing and existing["updated_at"] > incoming_updated_at:
                logger.warning(
                    "Thesis %s optimistic lock conflict: DB updated_at=%s > loaded=%s — "
                    "skipping write to avoid overwriting concurrent update",
                    thesis_id[:8] if len(thesis_id) >= 8 else thesis_id,
                    existing["updated_at"],
                    incoming_updated_at,
                )
                return

        # Set new updated_at now that the lock check passed
        thesis_data["updated_at"] = ts
        cols = list(thesis_data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "thesis_id")
        sql = (
            f"INSERT INTO position_theses ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(thesis_id) DO UPDATE SET {updates}"
        )
        with self._connect() as conn:
            conn.execute(sql, [thesis_data[c] for c in cols])
            conn.commit()
        logger.debug("Saved thesis ticker=%s status=%s", thesis_data.get("ticker"), thesis_data.get("status"))

    def get_thesis(self, thesis_id: str) -> dict[str, Any] | None:
        """Fetch a thesis by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM position_theses WHERE thesis_id = ?", (thesis_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_active_thesis_for_ticker(self, ticker: str) -> dict[str, Any] | None:
        """Get the active thesis for a ticker (status=active)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM position_theses WHERE ticker = ? AND status = 'active' "
                "ORDER BY datetime(created_at) DESC LIMIT 1",
                ((ticker or "").upper().strip(),),
            ).fetchone()
        return dict(row) if row else None

    def get_pending_theses(self) -> list[dict[str, Any]]:
        """Get all non-expired pending theses."""
        ts = self._utcnow_iso()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM position_theses WHERE status = 'pending' "
                "AND (pending_expiry IS NULL OR pending_expiry > ?) "
                "ORDER BY datetime(created_at) DESC",
                (ts,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_pending_theses_raw(self) -> list[dict[str, Any]]:
        """Get ALL status='pending' theses regardless of pending_expiry.

        FIX 1: Used by expire_stale_pending() so rows with pending_expiry<=now
        can be transitioned to 'expired'. get_pending_theses() filters them out.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM position_theses WHERE status = 'pending' "
                "ORDER BY datetime(created_at) DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_active_theses(self) -> list[dict[str, Any]]:
        """Get all active theses (for monitoring/review scheduling)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM position_theses WHERE status = 'active' "
                "ORDER BY datetime(created_at) ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_due_reviews(self) -> list[dict[str, Any]]:
        """Get active theses whose next_review_date has passed."""
        ts = self._utcnow_iso()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM position_theses WHERE status = 'active' "
                "AND next_review_date IS NOT NULL AND next_review_date <= ? "
                "ORDER BY next_review_date ASC",
                (ts,),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_sell_signal(self, signal_data: dict[str, Any]) -> None:
        """Insert a sell signal record."""
        ts = self._utcnow_iso()
        signal_data.setdefault("created_at", ts)
        cols = list(signal_data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT OR IGNORE INTO sell_signals ({', '.join(cols)}) VALUES ({placeholders})"
        with self._connect() as conn:
            conn.execute(sql, [signal_data[c] for c in cols])
            conn.commit()

    def get_active_sell_signals(self, ticker: str | None = None) -> list[dict[str, Any]]:
        """Get unactioned, unsuppressed sell signals."""
        query = "SELECT * FROM sell_signals WHERE actioned = 0 AND suppressed = 0"
        params: list[Any] = []
        if ticker:
            query += " AND ticker = ?"
            params.append((ticker or "").upper().strip())
        query += " ORDER BY datetime(created_at) DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def save_sell_session(self, session_data: dict[str, Any]) -> None:
        """Insert a sell council session record."""
        ts = self._utcnow_iso()
        session_data.setdefault("created_at", ts)
        cols = list(session_data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        sql = (
            f"INSERT OR IGNORE INTO sell_sessions ({', '.join(cols)}) VALUES ({placeholders})"
        )
        with self._connect() as conn:
            conn.execute(sql, [session_data[c] for c in cols])
            conn.commit()

    def save_post_sell_tracking(self, tracking_data: dict[str, Any]) -> None:
        """Insert or update a post-sell tracking record."""
        ts = self._utcnow_iso()
        tracking_data = dict(tracking_data)
        tracking_id = str(tracking_data.get("tracking_id") or "").strip()
        with self._connect() as conn:
            if tracking_id:
                existing = conn.execute(
                    "SELECT 1 FROM post_sell_tracking WHERE tracking_id = ?",
                    (tracking_id,),
                ).fetchone()
                if existing:
                    update_data = dict(tracking_data)
                    update_data.pop("tracking_id", None)
                    update_data["updated_at"] = ts
                    cols = list(update_data.keys())
                    if cols:
                        assignments = ", ".join(f"{c} = ?" for c in cols)
                        conn.execute(
                            f"UPDATE post_sell_tracking SET {assignments} WHERE tracking_id = ?",
                            [update_data[c] for c in cols] + [tracking_id],
                        )
                    conn.commit()
                    return

            tracking_data.setdefault("created_at", ts)
            tracking_data["updated_at"] = ts
            cols = list(tracking_data.keys())
            placeholders = ", ".join(["?"] * len(cols))
            updates = ", ".join(
                f"{c} = excluded.{c}" for c in cols if c != "tracking_id"
            )
            sql = (
                f"INSERT INTO post_sell_tracking ({', '.join(cols)}) VALUES ({placeholders}) "
                f"ON CONFLICT(tracking_id) DO UPDATE SET {updates}"
            )
            conn.execute(sql, [tracking_data[c] for c in cols])
            conn.commit()

    def get_pending_post_sell_reviews(self) -> list[dict[str, Any]]:
        """Get post-sell records that still have remaining tracking days (status='tracking')."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM post_sell_tracking WHERE status = 'tracking' "
                "ORDER BY datetime(sell_date) ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_post_sell_reviews(self) -> list[dict[str, Any]]:
        """FIX 10: Get ALL post-sell tracking records including completed ones.

        Use get_pending_post_sell_reviews() for active tracking only.
        Use this method when historical post-exit data is needed.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM post_sell_tracking "
                "ORDER BY datetime(sell_date) ASC"
            ).fetchall()
        return [dict(r) for r in rows]

    # -----------------------------------------------------------------------
    # Entry Watchlist Methods
    # -----------------------------------------------------------------------

    def save_defer_watch(self, watch_data: dict[str, Any]) -> None:
        """Insert or update a monitored DEFER/WATCH entry condition."""
        ts = self._utcnow_iso()
        watch_data.setdefault("created_at", ts)
        watch_data["updated_at"] = ts
        cols = list(watch_data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "watch_id")
        sql = (
            f"INSERT INTO defer_watchlist ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(watch_id) DO UPDATE SET {updates}"
        )
        with self._connect() as conn:
            conn.execute(sql, [watch_data[c] for c in cols])
            conn.commit()

    def get_active_defer_watch_for_ticker(self, ticker: str) -> dict[str, Any] | None:
        """Return latest active DEFER/WATCH entry condition for a ticker."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM defer_watchlist
                WHERE ticker = ? AND status = 'active'
                ORDER BY datetime(updated_at) DESC, id DESC
                LIMIT 1
                """,
                ((ticker or "").upper().strip(),),
            ).fetchone()
        return dict(row) if row else None

    def get_active_defer_watches_for_ticker(self, ticker: str) -> list[dict[str, Any]]:
        """Return all active, non-expired DEFER/WATCH entry conditions for a ticker."""
        ts = self._utcnow_iso()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM defer_watchlist
                WHERE ticker = ?
                  AND status = 'active'
                  AND (entry_valid_until IS NULL OR entry_valid_until = '' OR datetime(entry_valid_until) >= datetime(?))
                ORDER BY datetime(updated_at) DESC, id DESC
                """,
                ((ticker or "").upper().strip(), ts),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_active_defer_watches(self) -> list[dict[str, Any]]:
        """Return active, non-expired entry watches."""
        ts = self._utcnow_iso()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM defer_watchlist
                WHERE status = 'active'
                  AND (entry_valid_until IS NULL OR entry_valid_until = '' OR datetime(entry_valid_until) >= datetime(?))
                ORDER BY datetime(updated_at) DESC, id DESC
                """,
                (ts,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_defer_watch(self, watch_id: str) -> dict[str, Any] | None:
        """Return one DEFER/WATCH row by watch id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM defer_watchlist WHERE watch_id = ? LIMIT 1",
                (str(watch_id or ""),),
            ).fetchone()
        return dict(row) if row else None

    def get_defer_watches(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent DEFER/WATCH rows for Supervisor and diagnostics."""
        query = "SELECT * FROM defer_watchlist WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY datetime(updated_at) DESC, id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def expire_defer_watches(self) -> int:
        """Expire active entry watches whose validity window has passed."""
        ts = self._utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE defer_watchlist
                SET status = 'expired', updated_at = ?
                WHERE status = 'active'
                  AND entry_valid_until IS NOT NULL
                  AND entry_valid_until != ''
                  AND datetime(entry_valid_until) < datetime(?)
                """,
                (ts, ts),
            )
            conn.commit()
            return int(cursor.rowcount or 0)

    def invalidate_implausible_defer_watches(
        self,
        min_ratio: float = 0.35,
        max_ratio: float = 2.5,
    ) -> int:
        """Quarantine active entry watches that are clearly parsed non-price ranges."""
        ts = self._utcnow_iso()
        min_ratio = max(0.01, float(min_ratio or 0.35))
        max_ratio = max(min_ratio, float(max_ratio or 2.5))
        note = (
            "Auto-invalidated implausible entry zone: zone was outside "
            f"{min_ratio:.0%}-{max_ratio:.0%} of recorded current price."
        )
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE defer_watchlist
                SET status = 'invalid_zone',
                    updated_at = ?,
                    notes = CASE
                        WHEN notes IS NULL OR notes = '' THEN ?
                        ELSE notes || char(10) || ?
                    END
                WHERE status = 'active'
                  AND current_price IS NOT NULL
                  AND current_price > 0
                  AND (
                    zone_high < current_price * ?
                    OR zone_low > current_price * ?
                  )
                """,
                (ts, note, note, min_ratio, max_ratio),
            )
            conn.commit()
            return int(cursor.rowcount or 0)

    def requeue_stale_defer_auto_reviews(self, max_age_minutes: int = 120) -> int:
        """Return stale in-flight DEFER auto-reviews to active with an audit note.

        A monitor restart can interrupt a triggered review after the row has been
        moved to triggered_reviewing. Requeueing lets the next watch cycle retry
        instead of leaving the watch permanently invisible to active scans.
        """
        ts = self._utcnow_iso()
        minutes = max(1, int(max_age_minutes or 1))
        note = f"Auto-review requeued after stale triggered_reviewing state exceeded {minutes} minutes."
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE defer_watchlist
                SET status = 'active',
                    updated_at = ?,
                    notes = CASE
                        WHEN notes IS NULL OR notes = '' THEN ?
                        ELSE notes || char(10) || ?
                    END
                WHERE status = 'triggered_reviewing'
                  AND datetime(updated_at) < datetime(?, '-' || ? || ' minutes')
                """,
                (ts, note, note, ts, minutes),
            )
            conn.commit()
            return int(cursor.rowcount or 0)

    def mark_defer_watch_triggered(self, watch_id: str, trigger_price: float, notes: str = "") -> None:
        """Mark a DEFER/WATCH entry condition as triggered."""
        ts = self._utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE defer_watchlist
                SET status = 'triggered',
                    triggered_at = ?,
                    trigger_price = ?,
                    updated_at = ?,
                    notes = CASE
                        WHEN notes IS NULL OR notes = '' THEN ?
                        ELSE notes || char(10) || ?
                    END
                WHERE watch_id = ?
                """,
                (ts, float(trigger_price), ts, notes, notes, watch_id),
            )
            conn.commit()

    def update_defer_watch_status(
        self,
        watch_id: str,
        status: str,
        notes: str = "",
        trigger_price: float | None = None,
        set_triggered_at: bool = False,
    ) -> None:
        """Transition a DEFER/WATCH row and append an auditable note."""
        ts = self._utcnow_iso()
        status = str(status or "").strip()
        if not status:
            raise ValueError("defer watch status is required")

        clauses = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status, ts]
        if trigger_price is not None:
            clauses.append("trigger_price = ?")
            params.append(float(trigger_price))
        if set_triggered_at:
            clauses.append("triggered_at = COALESCE(triggered_at, ?)")
            params.append(ts)
        if notes:
            clauses.append(
                """
                notes = CASE
                    WHEN notes IS NULL OR notes = '' THEN ?
                    ELSE notes || char(10) || ?
                END
                """
            )
            params.extend([notes, notes])

        params.append(str(watch_id or ""))
        with self._connect() as conn:
            conn.execute(
                f"UPDATE defer_watchlist SET {', '.join(clauses)} WHERE watch_id = ?",
                tuple(params),
            )
            conn.commit()

    def supersede_defer_watches_for_ticker(
        self,
        ticker: str,
        keep_watch_ids: list[str] | None = None,
        notes: str = "",
    ) -> int:
        """Mark older active watches for a ticker as superseded after a fresh decision."""
        symbol = (ticker or "").upper().strip()
        if not symbol:
            return 0
        keep = [str(item or "").strip() for item in (keep_watch_ids or []) if str(item or "").strip()]
        ts = self._utcnow_iso()
        note = notes or "Superseded by newer Council entry-watch zones for the same ticker."
        placeholders = ", ".join(["?"] * len(keep))
        keep_filter = f"AND watch_id NOT IN ({placeholders})" if keep else ""
        params: list[Any] = [ts, note, note, symbol, *keep]
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE defer_watchlist
                SET status = 'superseded',
                    updated_at = ?,
                    notes = CASE
                        WHEN notes IS NULL OR notes = '' THEN ?
                        ELSE notes || char(10) || ?
                    END
                WHERE ticker = ?
                  AND status = 'active'
                  {keep_filter}
                """,
                tuple(params),
            )
            conn.commit()
            return int(cursor.rowcount or 0)

    def save_decision_features(self, feature_data: dict[str, Any]) -> None:
        """Insert/update compact point-in-time features for calibration."""
        ts = self._utcnow_iso()
        feature_data.setdefault("created_at", ts)
        cols = list(feature_data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "dossier_path")
        sql = (
            f"INSERT INTO decision_features ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(dossier_path) DO UPDATE SET {updates}"
        )
        with self._connect() as conn:
            conn.execute(sql, [feature_data[c] for c in cols])
            conn.commit()

    def save_scan_routing_decisions(self, rows: list[dict[str, Any]]) -> int:
        """Persist broker/data router decisions for scheduled-scan auditability."""
        import json

        if not rows:
            return 0
        ts = self._utcnow_iso()
        allowed = {
            "session_id",
            "created_at",
            "ticker",
            "candidate_rank",
            "lane",
            "bucket",
            "reason_code",
            "reason",
            "route_score",
            "funnel_score",
            "price",
            "live_price",
            "bid",
            "ask",
            "spread_pct",
            "avg_volume",
            "dollar_volume",
            "liquidity_source",
            "quote_source",
            "evidence_json",
        }
        normalized: list[dict[str, Any]] = []
        for row in rows:
            item = {k: row.get(k) for k in allowed if k in row}
            item.setdefault("created_at", ts)
            evidence = item.get("evidence_json", row.get("evidence") or {})
            if isinstance(evidence, (dict, list)):
                evidence = json.dumps(evidence, sort_keys=True, ensure_ascii=True)
            item["evidence_json"] = str(evidence or "{}")
            normalized.append(item)

        cols = [
            "session_id",
            "created_at",
            "ticker",
            "candidate_rank",
            "lane",
            "bucket",
            "reason_code",
            "reason",
            "route_score",
            "funnel_score",
            "price",
            "live_price",
            "bid",
            "ask",
            "spread_pct",
            "avg_volume",
            "dollar_volume",
            "liquidity_source",
            "quote_source",
            "evidence_json",
        ]
        placeholders = ", ".join(["?"] * len(cols))
        with self._connect() as conn:
            conn.executemany(
                f"INSERT INTO scan_routing_decisions ({', '.join(cols)}) VALUES ({placeholders})",
                [[row.get(c) for c in cols] for row in normalized],
            )
            conn.commit()
        return len(normalized)

    def get_scan_routing_decisions(
        self,
        session_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Return recent scheduled-scan router decisions."""
        params: list[Any] = []
        query = "SELECT * FROM scan_routing_decisions WHERE 1=1"
        if session_id:
            query += " AND session_id = ?"
            params.append(str(session_id))
        query += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_decision_features(self, limit: int = 500) -> list[dict[str, Any]]:
        """Return recent point-in-time decision feature rows."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM decision_features
                ORDER BY datetime(generated_at) DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def save_calibration_diagnostic(self, diagnostic: dict[str, Any]) -> int:
        """Persist a calibration diagnosis report and return row id."""
        import json

        ts = self._utcnow_iso()
        payload = dict(diagnostic.get("payload") or diagnostic)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO calibration_diagnostics (
                    generated_at, completed_samples, stage, severity, report_hash,
                    report_text, payload_json, sent_to_telegram, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(diagnostic.get("generated_at") or ts),
                    int(diagnostic.get("completed_samples") or 0),
                    str(diagnostic.get("stage") or "unknown"),
                    str(diagnostic.get("severity") or "INFO"),
                    str(diagnostic.get("report_hash") or ""),
                    str(diagnostic.get("report_text") or ""),
                    json.dumps(payload, sort_keys=True, ensure_ascii=True),
                    1 if diagnostic.get("sent_to_telegram") else 0,
                    ts,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_latest_calibration_diagnostic(self) -> dict[str, Any] | None:
        """Return the most recent calibration diagnosis report."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM calibration_diagnostics
                ORDER BY datetime(generated_at) DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    # -----------------------------------------------------------------------
    # Shadow Rule Engine Methods
    # -----------------------------------------------------------------------

    def save_shadow_rule_evaluation(self, evaluation: dict[str, Any]) -> bool:
        """Insert a private shadow-rule evaluation.

        Returns True when a new row was inserted, False when the evaluation
        already existed. Shadow evaluations never alter live recommendations.
        """
        import json

        ts = self._utcnow_iso()
        evaluation.setdefault("created_at", ts)
        evaluation.setdefault("updated_at", ts)
        if isinstance(evaluation.get("evidence_json"), (dict, list)):
            evaluation["evidence_json"] = json.dumps(
                evaluation["evidence_json"], sort_keys=True, ensure_ascii=True
            )
        cols = list(evaluation.keys())
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT OR IGNORE INTO shadow_rule_evaluations ({', '.join(cols)}) VALUES ({placeholders})"
        with self._connect() as conn:
            cursor = conn.execute(sql, [evaluation[c] for c in cols])
            conn.commit()
            return bool(cursor.rowcount)

    def get_pending_shadow_rule_evaluations(self) -> list[dict[str, Any]]:
        """Return shadow-rule evaluations with remaining checkpoints."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM shadow_rule_evaluations
                WHERE status = 'tracking'
                ORDER BY datetime(created_at) ASC, id ASC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_shadow_rule_evaluations(self, limit: int = 500) -> list[dict[str, Any]]:
        """Return recent shadow-rule evaluations."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM shadow_rule_evaluations
                ORDER BY datetime(created_at) DESC, id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_shadow_rule_evaluation(
        self,
        evaluation_id: str,
        updates: dict[str, Any],
    ) -> None:
        """Update shadow-rule checkpoint prices and derived returns."""
        allowed = {
            "price_5d", "price_10d", "price_20d", "price_60d",
            "benchmark_price_entry", "benchmark_price_5d", "benchmark_price_10d",
            "benchmark_price_20d", "benchmark_price_60d",
            "sector_benchmark_price_entry", "sector_benchmark_price_5d",
            "sector_benchmark_price_10d", "sector_benchmark_price_20d",
            "sector_benchmark_price_60d", "mfe", "mae", "would_hit_stop",
        }
        set_clauses: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            if key not in allowed or value is None:
                continue
            set_clauses.append(f"{key} = ?")
            if key == "would_hit_stop":
                params.append(1 if value else 0)
            else:
                params.append(float(value))
        ts = self._utcnow_iso()
        with self._connect() as conn:
            if set_clauses:
                conn.execute(
                    f"""
                    UPDATE shadow_rule_evaluations
                    SET {", ".join(set_clauses)}, updated_at = ?
                    WHERE evaluation_id = ?
                    """,
                    (*params, ts, evaluation_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE shadow_rule_evaluations
                    SET updated_at = ?
                    WHERE evaluation_id = ?
                    """,
                    (ts, evaluation_id),
                )
            conn.execute(
                """
                UPDATE shadow_rule_evaluations
                SET
                    return_5d = CASE WHEN price_5d IS NOT NULL AND hypothetical_entry > 0
                        THEN (price_5d - hypothetical_entry) / hypothetical_entry ELSE return_5d END,
                    return_10d = CASE WHEN price_10d IS NOT NULL AND hypothetical_entry > 0
                        THEN (price_10d - hypothetical_entry) / hypothetical_entry ELSE return_10d END,
                    return_20d = CASE WHEN price_20d IS NOT NULL AND hypothetical_entry > 0
                        THEN (price_20d - hypothetical_entry) / hypothetical_entry ELSE return_20d END,
                    return_60d = CASE WHEN price_60d IS NOT NULL AND hypothetical_entry > 0
                        THEN (price_60d - hypothetical_entry) / hypothetical_entry ELSE return_60d END,
                    benchmark_return_5d = CASE WHEN benchmark_price_5d IS NOT NULL AND benchmark_price_entry > 0
                        THEN (benchmark_price_5d - benchmark_price_entry) / benchmark_price_entry ELSE benchmark_return_5d END,
                    benchmark_return_10d = CASE WHEN benchmark_price_10d IS NOT NULL AND benchmark_price_entry > 0
                        THEN (benchmark_price_10d - benchmark_price_entry) / benchmark_price_entry ELSE benchmark_return_10d END,
                    benchmark_return_20d = CASE WHEN benchmark_price_20d IS NOT NULL AND benchmark_price_entry > 0
                        THEN (benchmark_price_20d - benchmark_price_entry) / benchmark_price_entry ELSE benchmark_return_20d END,
                    benchmark_return_60d = CASE WHEN benchmark_price_60d IS NOT NULL AND benchmark_price_entry > 0
                        THEN (benchmark_price_60d - benchmark_price_entry) / benchmark_price_entry ELSE benchmark_return_60d END,
                    sector_benchmark_return_5d = CASE WHEN sector_benchmark_price_5d IS NOT NULL AND sector_benchmark_price_entry > 0
                        THEN (sector_benchmark_price_5d - sector_benchmark_price_entry) / sector_benchmark_price_entry ELSE sector_benchmark_return_5d END,
                    sector_benchmark_return_10d = CASE WHEN sector_benchmark_price_10d IS NOT NULL AND sector_benchmark_price_entry > 0
                        THEN (sector_benchmark_price_10d - sector_benchmark_price_entry) / sector_benchmark_price_entry ELSE sector_benchmark_return_10d END,
                    sector_benchmark_return_20d = CASE WHEN sector_benchmark_price_20d IS NOT NULL AND sector_benchmark_price_entry > 0
                        THEN (sector_benchmark_price_20d - sector_benchmark_price_entry) / sector_benchmark_price_entry ELSE sector_benchmark_return_20d END,
                    sector_benchmark_return_60d = CASE WHEN sector_benchmark_price_60d IS NOT NULL AND sector_benchmark_price_entry > 0
                        THEN (sector_benchmark_price_60d - sector_benchmark_price_entry) / sector_benchmark_price_entry ELSE sector_benchmark_return_60d END,
                    excess_return_5d = CASE WHEN price_5d IS NOT NULL AND hypothetical_entry > 0
                        THEN ((price_5d - hypothetical_entry) / hypothetical_entry)
                          - COALESCE(
                            CASE WHEN sector_benchmark_price_5d IS NOT NULL AND sector_benchmark_price_entry > 0
                              THEN (sector_benchmark_price_5d - sector_benchmark_price_entry) / sector_benchmark_price_entry END,
                            CASE WHEN benchmark_price_5d IS NOT NULL AND benchmark_price_entry > 0
                              THEN (benchmark_price_5d - benchmark_price_entry) / benchmark_price_entry END,
                            0
                          ) ELSE excess_return_5d END,
                    excess_return_10d = CASE WHEN price_10d IS NOT NULL AND hypothetical_entry > 0
                        THEN ((price_10d - hypothetical_entry) / hypothetical_entry)
                          - COALESCE(
                            CASE WHEN sector_benchmark_price_10d IS NOT NULL AND sector_benchmark_price_entry > 0
                              THEN (sector_benchmark_price_10d - sector_benchmark_price_entry) / sector_benchmark_price_entry END,
                            CASE WHEN benchmark_price_10d IS NOT NULL AND benchmark_price_entry > 0
                              THEN (benchmark_price_10d - benchmark_price_entry) / benchmark_price_entry END,
                            0
                          ) ELSE excess_return_10d END,
                    excess_return_20d = CASE WHEN price_20d IS NOT NULL AND hypothetical_entry > 0
                        THEN ((price_20d - hypothetical_entry) / hypothetical_entry)
                          - COALESCE(
                            CASE WHEN sector_benchmark_price_20d IS NOT NULL AND sector_benchmark_price_entry > 0
                              THEN (sector_benchmark_price_20d - sector_benchmark_price_entry) / sector_benchmark_price_entry END,
                            CASE WHEN benchmark_price_20d IS NOT NULL AND benchmark_price_entry > 0
                              THEN (benchmark_price_20d - benchmark_price_entry) / benchmark_price_entry END,
                            0
                          ) ELSE excess_return_20d END,
                    excess_return_60d = CASE WHEN price_60d IS NOT NULL AND hypothetical_entry > 0
                        THEN ((price_60d - hypothetical_entry) / hypothetical_entry)
                          - COALESCE(
                            CASE WHEN sector_benchmark_price_60d IS NOT NULL AND sector_benchmark_price_entry > 0
                              THEN (sector_benchmark_price_60d - sector_benchmark_price_entry) / sector_benchmark_price_entry END,
                            CASE WHEN benchmark_price_60d IS NOT NULL AND benchmark_price_entry > 0
                              THEN (benchmark_price_60d - benchmark_price_entry) / benchmark_price_entry END,
                            0
                          ) ELSE excess_return_60d END,
                    status = CASE WHEN price_60d IS NOT NULL THEN 'completed' ELSE status END
                WHERE evaluation_id = ?
                """,
                (evaluation_id,),
            )
            conn.commit()

    # -----------------------------------------------------------------------
    # Supervisor Methods
    # -----------------------------------------------------------------------

    def save_supervisor_run(self, payload: dict[str, Any]) -> int:
        """Persist a supervisor health report."""
        import json

        ts = self._utcnow_iso()
        report_payload = dict(payload.get("payload") or payload)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO supervisor_runs (
                    generated_at, severity, report_hash, report_text,
                    payload_json, sent_to_telegram, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(payload.get("generated_at") or ts),
                    str(payload.get("severity") or "INFO"),
                    str(payload.get("report_hash") or ""),
                    str(payload.get("report_text") or ""),
                    json.dumps(report_payload, sort_keys=True, ensure_ascii=True),
                    1 if payload.get("sent_to_telegram") else 0,
                    ts,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_latest_supervisor_run(self) -> dict[str, Any] | None:
        """Return the most recent supervisor health report."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM supervisor_runs
                ORDER BY datetime(generated_at) DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        return dict(row) if row else None

    # -----------------------------------------------------------------------
    # Execution Audit Methods
    # -----------------------------------------------------------------------

    def save_execution_order(self, order_data: dict[str, Any]) -> int:
        """Insert/update a Robinhood-ready order proposal audit row."""
        import json

        ts = self._utcnow_iso()
        order_data.setdefault("created_at", ts)
        order_data["updated_at"] = ts
        for key in ("guardrail_json", "evidence_json", "request_json", "response_json"):
            if isinstance(order_data.get(key), (dict, list)):
                order_data[key] = json.dumps(order_data[key], sort_keys=True, ensure_ascii=True)
        if order_data.get("ticker"):
            order_data["ticker"] = str(order_data["ticker"]).upper().strip()
        if order_data.get("side"):
            order_data["side"] = str(order_data["side"]).lower().strip()
        if order_data.get("order_type"):
            order_data["order_type"] = str(order_data["order_type"]).lower().strip()
        if "dry_run" in order_data:
            order_data["dry_run"] = 1 if order_data["dry_run"] else 0

        cols = list(order_data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "order_intent_id")
        sql = (
            f"INSERT INTO execution_orders ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(order_intent_id) DO UPDATE SET {updates}"
        )
        with self._connect() as conn:
            cursor = conn.execute(sql, [order_data[c] for c in cols])
            row = conn.execute(
                "SELECT id FROM execution_orders WHERE order_intent_id = ?",
                (str(order_data.get("order_intent_id") or ""),),
            ).fetchone()
            conn.commit()
        return int(row["id"] if row else cursor.lastrowid)

    def update_execution_order(self, order_intent_id: str, updates: dict[str, Any]) -> None:
        """Update one execution audit row."""
        import json

        allowed = {
            "status", "broker_order_id", "guardrail_status", "guardrail_json",
            "response_json", "submitted_at", "filled_at", "canceled_at", "dry_run", "notes",
            "quantity", "notional", "estimated_price",
        }
        ts = self._utcnow_iso()
        clauses: list[str] = []
        params: list[Any] = []
        for key, value in updates.items():
            if key not in allowed:
                continue
            if key == "dry_run":
                value = 1 if value else 0
            if key in {"guardrail_json", "response_json"} and isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True, ensure_ascii=True)
            clauses.append(f"{key} = ?")
            params.append(value)
        clauses.append("updated_at = ?")
        params.append(ts)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE execution_orders SET {', '.join(clauses)} WHERE order_intent_id = ?",
                (*params, order_intent_id),
            )
            conn.commit()

    def get_execution_orders(
        self,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return recent execution audit rows."""
        query = "SELECT * FROM execution_orders WHERE 1=1"
        params: list[Any] = []
        if ticker:
            query += " AND ticker = ?"
            params.append((ticker or "").upper().strip())
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY datetime(created_at) DESC, id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get_execution_order_by_id(self, row_id: int) -> dict[str, Any] | None:
        """Return one execution audit row by numeric row id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM execution_orders WHERE id = ? LIMIT 1",
                (int(row_id),),
            ).fetchone()
        return dict(row) if row else None

    def get_execution_order_by_intent_id(self, order_intent_id: str) -> dict[str, Any] | None:
        """Return one execution audit row by stable order intent id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM execution_orders WHERE order_intent_id = ? LIMIT 1",
                (str(order_intent_id or ""),),
            ).fetchone()
        return dict(row) if row else None

    def count_execution_orders_today(self, side: str | None = None) -> int:
        """Count order attempts today by UTC date."""
        today = datetime.now(timezone.utc).date().isoformat()
        query = (
            "SELECT COUNT(*) AS c FROM execution_orders "
            "WHERE date(created_at) = date(?) "
            "AND status IN ('submitted', 'filled', 'partially_filled')"
        )
        params: list[Any] = [today]
        if side:
            query += " AND side = ?"
            params.append(side.lower().strip())
        with self._connect() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
        return int(row["c"] if row else 0)

    # -----------------------------------------------------------------------
    # Trade Action Queue Methods
    # -----------------------------------------------------------------------

    def save_trade_action(self, action_data: dict[str, Any]) -> int:
        """Insert/update a Telegram/OpenClaw trade action row."""
        import json

        ts = self._utcnow_iso()
        action_data.setdefault("created_at", ts)
        action_data["updated_at"] = ts
        if action_data.get("ticker"):
            action_data["ticker"] = str(action_data["ticker"]).upper().strip()
        if action_data.get("side"):
            action_data["side"] = str(action_data["side"]).lower().strip()
        for key in ("payload_json", "result_json"):
            if isinstance(action_data.get(key), (dict, list)):
                action_data[key] = json.dumps(action_data[key], sort_keys=True, ensure_ascii=True)
        cols = list(action_data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "action_id")
        sql = (
            f"INSERT INTO trade_actions ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(action_id) DO UPDATE SET {updates}"
        )
        with self._connect() as conn:
            cursor = conn.execute(sql, [action_data[c] for c in cols])
            row = conn.execute(
                "SELECT id FROM trade_actions WHERE action_id = ?",
                (str(action_data.get("action_id") or ""),),
            ).fetchone()
            conn.commit()
        return int(row["id"] if row else cursor.lastrowid)

    def get_trade_action(
        self,
        action_id: str | None = None,
        token: str | None = None,
    ) -> dict[str, Any] | None:
        """Return one trade action by action id or any callback token."""
        if not action_id and not token:
            return None
        with self._connect() as conn:
            if action_id:
                row = conn.execute(
                    "SELECT * FROM trade_actions WHERE action_id = ? LIMIT 1",
                    (str(action_id),),
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT * FROM trade_actions
                    WHERE token_review = ? OR token_place = ? OR token_skip = ?
                    LIMIT 1
                    """,
                    (str(token), str(token), str(token)),
                ).fetchone()
        return dict(row) if row else None

    def get_trade_actions(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent trade action rows."""
        query = "SELECT * FROM trade_actions WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY datetime(updated_at) DESC, id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def update_trade_action(self, action_id: str, updates: dict[str, Any]) -> None:
        """Update one trade action row."""
        import json

        allowed = {
            "status", "expires_at", "execution_order_row", "order_intent_id",
            "thesis_id", "payload_json", "result_json", "message", "notes",
        }
        updates = {k: v for k, v in updates.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = self._utcnow_iso()
        for key in ("payload_json", "result_json"):
            if isinstance(updates.get(key), (dict, list)):
                updates[key] = json.dumps(updates[key], sort_keys=True, ensure_ascii=True)
        clauses = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE trade_actions SET {clauses} WHERE action_id = ?",
                (*updates.values(), str(action_id or "")),
            )
            conn.commit()

    # -----------------------------------------------------------------------
    # Pending Market-Open Order Recheck Methods
    # -----------------------------------------------------------------------

    def save_pending_order_recheck(self, recheck_data: dict[str, Any]) -> int:
        """Insert/update a durable market-open recheck instruction."""
        ts = self._utcnow_iso()
        recheck_data.setdefault("created_at", ts)
        recheck_data["updated_at"] = ts
        if recheck_data.get("ticker"):
            recheck_data["ticker"] = str(recheck_data["ticker"]).upper().strip()
        cols = list(recheck_data.keys())
        placeholders = ", ".join(["?"] * len(cols))
        updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "recheck_id")
        sql = (
            f"INSERT INTO pending_order_rechecks ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT(recheck_id) DO UPDATE SET {updates}"
        )
        with self._connect() as conn:
            cursor = conn.execute(sql, [recheck_data[c] for c in cols])
            row = conn.execute(
                "SELECT id FROM pending_order_rechecks WHERE recheck_id = ?",
                (str(recheck_data.get("recheck_id") or ""),),
            ).fetchone()
            conn.commit()
        return int(row["id"] if row else cursor.lastrowid)

    def get_due_pending_order_rechecks(self, now_iso: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Return pending market-open order rechecks due at or before now."""
        ts = now_iso or self._utcnow_iso()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM pending_order_rechecks
                WHERE status = 'pending'
                  AND datetime(run_after) <= datetime(?)
                  AND (expires_at IS NULL OR expires_at = '' OR datetime(expires_at) >= datetime(?))
                ORDER BY datetime(run_after) ASC, id ASC
                LIMIT ?
                """,
                (ts, ts, int(limit)),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_pending_order_rechecks(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent market-open order recheck rows."""
        query = "SELECT * FROM pending_order_rechecks WHERE 1=1"
        params: list[Any] = []
        if status:
            query += " AND status = ?"
            params.append(str(status))
        query += " ORDER BY datetime(updated_at) DESC, id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def update_pending_order_recheck(self, recheck_id: str, updates: dict[str, Any]) -> None:
        """Update one market-open order recheck row."""
        allowed = {
            "status", "updated_at", "last_reviewed_at", "last_verdict", "last_price",
            "execution_order_row", "notes",
        }
        updates = {k: v for k, v in updates.items() if k in allowed}
        if not updates:
            return
        updates["updated_at"] = self._utcnow_iso()
        clauses = ", ".join(f"{key} = ?" for key in updates)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE pending_order_rechecks SET {clauses} WHERE recheck_id = ?",
                (*updates.values(), str(recheck_id or "")),
            )
            conn.commit()

    def get_open_recommendations(self, ticker: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Get open recommendations optionally filtered by ticker."""
        query = (
            "SELECT id, timestamp, session_id, ticker, action, rationale, confidence, "
            "price_at_recommendation, conditions, status, outcome, outcome_notes, created_at "
            "FROM recommendations WHERE status = 'open'"
        )
        params: list[Any] = []
        if ticker:
            query += " AND ticker = ?"
            params.append((ticker or "").upper().strip())
        query += " ORDER BY datetime(timestamp) DESC, id DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]
