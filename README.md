# Artha Council

Artha is an AI-assisted equity research and broker-aware execution system. It scans a broad US-stock universe, promotes the most interesting candidates through a multi-stage funnel, asks specialist AI analysts to debate the best opportunities, and then applies broker execution gates before any order can be queued.

This repository is a public-safe release of the project code. It intentionally excludes private runtime data: portfolios, broker snapshots, API keys, Telegram tokens, SQLite journals, reports, and live trade logs.

## What It Does

- Builds an investable stock universe from market data providers.
- Ranks hundreds to thousands of stocks through momentum, valuation, quality, liquidity, regime fit, and repeat-cooldown signals.
- Uses a broker-aware router to separate "interesting research ideas" from "buyable today" candidates.
- Runs an agentic opportunity scout before the council to rank the best finalists with evidence.
- Sends the strongest candidates to a multi-role investment council.
- Produces decision dossiers with evidence IDs, source audits, score breakdowns, invalidation rules, and final action labels.
- Uses an execution officer and Robinhood/OpenClaw bridge for quote freshness, spread, buying power, tradability, order preview, kill-switch, and auto-buy controls.
- Sends Telegram reports for scan progress, council decisions, execution decisions, and supervisor checks.

## High-Level Pipeline

```text
Market data universe
  -> promotion funnel
  -> broker-aware router
  -> opportunity scout
  -> council analysts
  -> synthesis / CIO cross-check
  -> execution officer
  -> Robinhood/OpenClaw safety bridge
  -> journal, dossier, Telegram, supervisor
```

The most important design choice is that Artha separates investment quality from execution feasibility:

- Research layer: "Is this company worth investigating or owning?"
- Execution layer: "Can this specific order be placed safely right now?"

That prevents a good idea with a bad quote, wide spread, stale snapshot, or broker alert from becoming an unsafe trade.

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full component map.

Core modules:

- `artha/funnel.py` - narrows the broad stock universe into finalist candidates.
- `artha/broker_router.py` - checks execution/data feasibility before council slots are spent.
- `artha/opportunity_scout.py` - agentic pre-council ranking and evidence review.
- `artha/council.py` - multi-analyst decision engine and synthesis.
- `artha/execution_officer.py` - final buy-side decision and guardrail reasoning.
- `artha/robinhood_bridge.py` - broker snapshot, review, place, and auto-buy handoff contracts.
- `artha/scheduler.py` - scheduled scan orchestration and Telegram reporting.
- `artha/supervisor.py` - production readiness and health checks.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your own provider keys. The default broker settings are intentionally safe: review-only, dry-run, agentic trading disabled, and kill-switch enabled.

Smoke checks:

```bash
python -m compileall artha run.py
python -m artha.test_enhancements
python -m artha.test_production_hardening
```

Example commands:

```bash
python run.py overview
python run.py analyze AAPL MSFT V
python run.py broker-router-preview --assume-market-open --no-persist
python run.py supervisor-check
```

## Safety Model

Artha is built fail-closed. Broker-dependent actions require fresh snapshot state, live quote sanity, tradability checks, review results, configured limits, and a kill-switch check. Public defaults do not place live trades.

The auto-buy path is designed for a tightly capped pilot account, not unrestricted trading:

- long US equities only
- cash account only
- max order and daily caps
- no options, margin, shorts, or crypto
- no stale broker snapshots
- no order if Robinhood review raises blocking alerts
- Telegram notification after success, block, or failure

## Public Release Boundary

Included:

- source code
- tests
- CI workflow
- architecture and setup docs
- environment template with placeholder values

Excluded:

- `.env` and `.env.bak`
- API keys and OAuth tokens
- live portfolio state
- Robinhood account snapshots
- SQLite journals/databases
- Telegram chat IDs and bot tokens
- generated reports, dossiers, traces, and logs

## Disclaimer

Artha is personal research and automation software. It is not financial advice. Anyone using it should verify the code, configure their own data providers, keep live trading disabled until they understand every gate, and comply with their broker's terms.
