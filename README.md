# Artha Council

_A staged, multi-model AI investment committee that turns a broad US-stock universe into auditable, fail-closed execution decisions._

[![CI](https://github.com/akira231097/artha-council/actions/workflows/artha-ci.yml/badge.svg)](https://github.com/akira231097/artha-council/actions/workflows/artha-ci.yml)
![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![GPT--5.5](https://img.shields.io/badge/LLM-GPT--5.5-412991?logo=openai&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4?logo=google&logoColor=white)
![Claude Agent SDK](https://img.shields.io/badge/LLM-Claude%20Agent%20SDK-D97757)
![SQLite](https://img.shields.io/badge/store-SQLite-003B57?logo=sqlite&logoColor=white)

## Overview

Artha Council is an AI-assisted equity-research and broker-aware execution system. It scans a broad US-stock universe, promotes the most interesting candidates through a multi-stage funnel, asks specialist AI analysts to debate the best opportunities, and then applies deterministic broker-execution gates before any order can be queued.

Its defining design choice is the hard separation of **investment quality** ("is this company worth investigating or owning?") from **execution feasibility** ("can this specific order be placed safely right now?"). That separation prevents a good idea with a bad quote, wide spread, stale snapshot, or broker alert from ever becoming an unsafe trade. The system is built **fail-closed**: public defaults are review-only, dry-run, agentic-disabled, kill-switch-on, and never place live trades.

> Artha is personal research and automation software. **It is not financial advice.**

## Key features

- **Promotion funnel** that narrows 1000+ active stocks into ~6-8 council-ready finalists using momentum, valuation, quality, liquidity, regime fit, and a soft repeat/cooldown freshness penalty.
- **Broker-aware router** that lanes candidates into `execution_ready` / `research_watch` / `hard_reject` based on quote freshness, spread, tradability, and data-provider conflict — so non-executable ideas are preserved, not discarded.
- **Agentic opportunity scout** that ranks which executable names deserve the scarce, expensive council slots (without ever making the final call).
- **Multi-model analyst council**: Fundamental (GPT-5.5), Technical (Gemini), and Contrarian/Risk (GPT-5.5) run **independently**, then a synthesis/CIO layer audits scores, valuation anchors, data gaps, and invalidation rules.
- **Execution officer** that converts a buy-side label into a live-quote verdict (`BUY_READY` / `WAIT_FOR_SAFE_EXECUTION` / `BLOCKED`) under hard, non-overridable caps.
- **Replayable OpenClaw agentic broker bridge** over the Model Context Protocol (MCP): the engine emits exact read-only call sequences for the OpenClaw agentic runner and replays the responses into source-controlled clearance logic — keeping money-moving steps deterministic and auditable even though execution is agentic.
- **Sell side** with thesis tracking, a 3-analyst sell council, trailing stops, regime comparison, and a portfolio circuit breaker.
- **Auditable by design**: decision dossiers with evidence IDs, a SQLite decision journal, agentic traces, fill reconciliation, and supervisor health checks.
- **Telegram reporting** for scan progress, council decisions, execution decisions, and supervisor checks.

## Architecture

Each stage has a narrow job and emits enough evidence for later audit. Research never becomes an order without passing an independent, deterministic execution gate.

```text
Market-data universe (FMP screener)
  -> promotion funnel        (momentum / valuation / quality / liquidity / regime fit)
  -> broker-aware router     (quote freshness, spread, tradability, data conflict)
  -> opportunity scout       (agentic ranking of council slots)
  -> council analysts        (Fundamental=GPT-5.5 | Technical=Gemini | Contrarian=GPT-5.5)
  -> synthesis / CIO audit   (score, valuation anchors, invalidation, buy-score audit)
  -> execution officer       (live-quote verdict under hard caps)
  -> OpenClaw broker bridge  (read-only MCP snapshot -> review -> preview -> place)
  -> journal, dossier, Telegram, supervisor
```

**Research layer vs. execution layer.** The council answers "is this worth owning?" and produces a label (`BUY`, `STARTER`, `TACTICAL_BUY`, `DEFER`, `WATCH`, `AVOID`). Only buy-side labels advance. The execution officer and broker bridge then answer "can this exact order be placed safely at the live quote?" — and the LLM may choose among deterministic candidates but **cannot expand caps or override the guardrail engine**.

## Tech stack

| Layer | Technology |
| --- | --- |
| Language / runtime | Python 3.12 |
| Reasoning analysts | GPT-5.5 (ChatGPT/Codex backend), Google Gemini, Claude Agent SDK |
| Market & fundamental data | Financial Modeling Prep, Finnhub, Benzinga, Alpha Vantage, FRED, SEC EDGAR, yfinance |
| Numerics | pandas, numpy |
| Persistence | SQLite (decision journal, dossiers) |
| Config / parsing | python-dotenv, PyYAML |
| Delivery | Telegram Bot API |
| Broker integration | OpenClaw agentic runner over the Model Context Protocol (MCP) |
| HTTP | requests |
| CI | GitHub Actions (compile + hardening smoke tests) |

## Project structure

```text
.
├── run.py                     # CLI entry point (~30 subcommands)
├── requirements.txt
├── .env.example               # placeholder env template (no real keys)
├── README.md
├── SECURITY.md
├── docs/
│   ├── ARCHITECTURE.md        # full component map
│   └── PUBLIC_RELEASE.md      # public-release boundary
├── .github/workflows/         # CI smoke pipeline
└── artha/
    ├── funnel.py              # universe -> finalist candidates
    ├── broker_router.py       # execution/data feasibility lanes
    ├── opportunity_scout.py   # agentic pre-council ranking
    ├── council.py             # multi-analyst decision + synthesis
    ├── analysts.py            # independent per-model analysts
    ├── execution_officer.py   # final buy-side execution verdict
    ├── robinhood_bridge.py    # broker snapshot / review / place contracts
    ├── sell_engine.py         # sell council, trailing stops, circuit breaker
    ├── scheduler.py           # scan orchestration + Telegram reporting
    ├── supervisor.py          # production readiness / health checks
    ├── journal.py             # SQLite decision journal
    ├── chatgpt_backend.py     # resilient GPT client (token refresh, fallback)
    ├── gemini_client.py       # Gemini wrapper
    ├── claude_sdk.py          # sync wrapper over the Claude Agent SDK
    ├── collector.py           # multi-provider data collection
    └── ...                    # regime, valuation, dossier, prompts, tests, etc.
```

## Getting started

**Requirements:** Python 3.12, and your own API keys for the data and AI providers you want to enable.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your own provider keys. The default broker settings are intentionally safe: **review-only, dry-run, agentic trading disabled, and kill-switch enabled.**

Smoke checks:

```bash
python -m compileall artha run.py
python -m artha.test_enhancements
python -m artha.test_production_hardening
```

Example commands:

```bash
python run.py overview                      # market overview report
python run.py analyze AAPL MSFT V           # analyze specific tickers
python run.py scan 6                        # full funnel + council (6 finalists)
python run.py broker-router-preview --assume-market-open --no-persist
python run.py supervisor-check              # production readiness / health
```

## How it works

**Promotion funnel.** A 5-stage pipeline builds an investable universe from a market-data screener, machine-ranks ~500-1000 names with momentum/quality/valuation/liquidity/regime-fit signals, enriches the top tier with ratios, analyst recommendations, and earnings context, applies a quick triage, and returns ~6-8 finalists. A soft penalty queried from recent scan sessions discourages lazily recycling the same basket.

**Broker-aware router.** A deliberately non-fundamental gate. It checks whether a candidate has sane, fresh-enough price/quote/liquidity data and is realistically executable today, and it lanes accordingly. Interesting-but-non-executable ideas are routed to research/watch rather than thrown away.

**Multi-model council.** Three analysts run with no cross-contamination — each gets its own model, prompt, and data slice. A two-stage engine applies a binary hard risk gate, then opportunity scoring mapped to an action label. The synthesis/CIO layer reconciles the analysts and records valuation anchors, source conflicts, invalidation conditions, and a buy-score audit. Analyst output is parsed defensively: JSON block first, then markdown regex, then safe defaults.

**Resilient LLM clients.** The GPT backend client refreshes OAuth tokens on 401, falls back to an alternate model on 404, retries without `temperature` on 400, and backs off on 503. The Claude wrapper runs async SDK queries on a fresh event loop in a worker thread, so it works from both the synchronous CLI and the async daemon.

**Execution officer + OpenClaw broker bridge.** The officer turns a buy-side label into a live-quote verdict under hard caps (no-chase, spread, buying power). Because the engine does not call broker MCP tools directly, it emits an exact, deterministic read-only MCP sequence for the **OpenClaw** agentic runner to execute, then replays the collected responses into source-controlled clearance logic: snapshot refresh → account identity check → positions/orders reconciliation → quote/tradability review → order preview → final clearance → place only with the exact approved arguments → post-fill reconciliation. This keeps the system agentic while remaining constrained by deterministic safety rules.

**Safety model.** Broker-dependent actions require fresh snapshot state, live-quote sanity, tradability checks, review results, configured limits, and a kill-switch check. The auto-buy path is designed for a tightly capped pilot account: long US equities only, cash account only, per-order and per-day dollar caps, no options/margin/shorts/crypto, no stale snapshots, and no order if a broker review raises a blocking alert.

## Notes / limitations

- This is a sanitized public release intended to show the architecture and implementation quality. It intentionally excludes all private runtime data: `.env` credentials, live portfolio state, broker snapshots, SQLite journals, generated reports/dossiers/traces, and any chat tokens.
- Live trading is **disabled by default** and should stay disabled until you have reviewed the full broker bridge, execution officer, account caps, and your broker's terms.
- You must supply your own data/AI provider accounts; no keys are bundled.
- This software is for personal research and automation. **It is not financial advice**, and you are responsible for verifying every gate and complying with applicable rules.

## License

Released under the [MIT License](LICENSE).