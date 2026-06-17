# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-17

Initial public, sanitized release of Artha Council — a staged, multi-model AI
investment committee that turns a broad US-stock universe into auditable,
fail-closed execution decisions.

### Added

- **Promotion funnel** (`funnel.py`): a 5-stage pipeline that builds an
  investable universe from a market-data screener, machine-ranks ~500-1000
  names on momentum, valuation, quality, liquidity, and regime fit, enriches
  the top tier with ratios/recommendations/earnings context, and returns ~6-8
  council-ready finalists — with a soft repeat/cooldown freshness penalty drawn
  from recent scan sessions.
- **Broker-aware router** (`broker_router.py`): a deliberately non-fundamental
  gate that lanes candidates into `execution_ready`, `research_watch`, or
  `hard_reject` based on quote freshness, spread, tradability, and
  data-provider conflict, preserving non-executable ideas instead of discarding
  them.
- **Agentic opportunity scout** (`opportunity_scout.py`): a pre-council agent
  that ranks which executable names deserve the scarce, expensive council slots
  without making the final call.
- **Multi-model analyst council** (`council.py`, `analysts.py`): independent
  Fundamental (GPT-5.5), Technical (Gemini), and Contrarian/Risk (GPT-5.5)
  analysts with no cross-contamination, a binary hard-risk gate, opportunity
  scoring mapped to an action label (`BUY`, `STARTER`, `TACTICAL_BUY`, `DEFER`,
  `WATCH`, `AVOID`), and a synthesis/CIO audit layer that can restrict below the
  score-mapped action but never upgrade above it.
- **Execution officer** (`execution_officer.py`): converts a buy-side label into
  a live-quote verdict (`BUY_READY` / `WAIT_FOR_SAFE_EXECUTION` / `BLOCKED`)
  under hard, non-overridable caps (no-chase cap, spread, buying power), with an
  optional agentic mode that validates required live-tool usage before clearing.
- **OpenClaw / Robinhood agentic broker bridge** (`robinhood_bridge.py`,
  `openclaw_robinhood_handler.py`): emits an exact read-only MCP call sequence
  (snapshot refresh → account identity → positions/orders reconciliation →
  quote/tradability review → order preview → final clearance → place with the
  exact approved arguments → post-fill reconciliation) and replays responses
  into source-controlled clearance logic, keeping money-moving steps
  deterministic and auditable.
- **Fail-closed safety cage**: public defaults are review-only, dry-run,
  agentic-disabled, and kill-switch-on; placement re-validates a fresh broker
  snapshot, the stored review gate, market hours, and per-order/per-day dollar
  caps, defaulting to blocked on any error.
- **Sell side** (`sell_engine.py`, `sell_council.py`, `thesis_tracker.py`,
  `trailing_stop.py`): thesis tracking, a 3-analyst sell council, trailing
  stops, regime comparison, and a portfolio circuit breaker.
- **Resilient LLM clients**: a GPT backend (`chatgpt_backend.py`) that refreshes
  OAuth tokens on 401, falls back to an alternate model on 404, retries without
  `temperature` on 400, and backs off on 503; a Gemini wrapper
  (`gemini_client.py`); and a synchronous wrapper over the Claude Agent SDK
  (`claude_sdk.py`) that runs async queries on a fresh event loop in a worker
  thread.
- **Multi-provider data collection** (`collector.py`): Financial Modeling Prep,
  Finnhub, Benzinga, Alpha Vantage, FRED, SEC EDGAR, and yfinance.
- **Auditability**: SQLite decision journal (`journal.py`), decision dossiers
  (`dossier.py`), agentic traces, fill reconciliation, and supervisor health
  checks (`supervisor.py`).
- **Telegram reporting** (`telegram.py`, `scheduler.py`) for scan progress,
  council decisions, execution decisions, and supervisor checks.
- **CLI entry point** (`run.py`) with ~30 subcommands, including `scan`,
  `analyze`, `overview`, `portfolio`, `broker-router-preview`,
  `execution-readiness`, `supervise`, and the `robinhood-*` bridge operations.
- **CI** (`.github/workflows/artha-ci.yml`): GitHub Actions pipeline that
  compiles the package and runs the enhancement and production-hardening smoke
  tests on Python 3.12.

### Security

- Sanitized public release: excludes all private runtime data (`.env`
  credentials, live portfolio state, broker snapshots, SQLite journals,
  generated reports/dossiers/traces, and chat tokens). Live trading is disabled
  by default. See `SECURITY.md` and `docs/PUBLIC_RELEASE.md`.

[1.0.0]: https://github.com/akira231097/artha-council/releases/tag/v1.0.0
