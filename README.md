# Artha Council

<!-- mcp-name: io.github.akira231097/artha-council -->

[![Artha CI](https://github.com/akira231097/artha-council/actions/workflows/artha-ci.yml/badge.svg)](https://github.com/akira231097/artha-council/actions/workflows/artha-ci.yml)
[![CodeQL](https://github.com/akira231097/artha-council/actions/workflows/codeql.yml/badge.svg)](https://github.com/akira231097/artha-council/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/akira231097/artha-council/badge)](https://securityscorecards.dev/viewer/?uri=github.com/akira231097/artha-council)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Artha is an AI-assisted equity research and portfolio automation system. It
narrows a broad US-stock universe, ranks opportunities, asks independent analyst
roles to debate the strongest candidates, monitors active investment theses, and
uses deterministic broker gates before an order can proceed.

Version 1.4 hardens the MCP server with source/build provenance for portable
local or OAuth-protected remote access. It supports the existing US workflow
and market-aware broker adapters for Indian cash equities without treating US
research as India data.
India adapters provide broker-verified instrument lookup and fail-closed,
whole-share delivery limit execution with current static-IP requirements.

Created and maintained by **Sarath** ([@akira231097](https://github.com/akira231097)).

See the [design notes and architecture diagram](docs/DESIGN.md) for a visual
walkthrough of the research, Council, execution, and audit boundaries.

> Artha is research software, not financial advice. Public defaults cannot place
> live trades. Anyone enabling broker integration is responsible for reviewing
> the source, their configuration, broker terms, and applicable requirements.

## Why Artha Exists

Most stock screeners stop after ranking tickers. Artha treats investing as a
stateful operating system:

1. Find candidates from a large universe.
2. Verify data quality and execution feasibility.
3. Give scarce Council attention to the strongest candidates.
4. Separate investment judgment from order execution.
5. Monitor every held thesis and send material changes to a sell Council.
6. Reconcile intended actions against broker state.
7. Preserve evidence and state transitions for audit.

## System Pipeline

```text
Market data providers
  -> universe and promotion funnel
  -> investment sanity checks
  -> broker-aware candidate router
  -> opportunity scout
  -> fundamental / technical / risk analysts
  -> CIO synthesis
  -> execution officer
  -> deterministic broker review and placement gates
  -> reconciliation, thesis tracking, Telegram, supervisor
```

The sell side runs independently of new-buy capacity:

```text
Portfolio snapshot
  -> position and thesis monitor
  -> deterministic stop / invalidation triggers
  -> sell Council for judgment decisions
  -> execution officer
  -> broker review and exact-order placement gate
  -> post-fill reconciliation
```

## Important Design Rules

### Research is not execution

The Council answers: "Is this investment attractive?"

The execution layer answers: "Can this exact order be placed safely now?"

A good company can be temporarily unbuyable because the quote is stale, the
spread is too wide, the price moved above the approved cap, or broker review is
incomplete. Those conditions do not rewrite the investment thesis.

### Missing broker proof blocks an order

Money-moving paths fail closed. A model statement that a check passed is not
enough; Artha requires the decisive structured broker output.

### Portfolio monitoring continues when buys pause

Position limits and invested-capacity limits pause new buys only. Monitoring,
sell review, reconciliation, health checks, and alerts continue running.

## Major Components

- `artha/funnel.py` - broad-universe promotion and multi-sleeve ranking.
- `artha/broker_router.py` - quote, liquidity, tradability, and data feasibility.
- `artha/opportunity_scout.py` - agentic pre-Council evidence review and ranking.
- `artha/council.py` - analyst roles, score audit, and CIO synthesis.
- `artha/execution_officer.py` - Stage A and Stage B execution reasoning.
- `artha/alpha_shadow.py` - non-authoritative alpha experiments and outcome tracking.
- `artha/broker_capacity.py` - portfolio and daily buy-capacity calculations.
- `artha/stand_down.py` - buy-only pause and next-session reset behavior.
- `artha/sell_engine.py` - position triggers and sell-side orchestration.
- `artha/sell_council.py` - hold, trim, and exit review.
- `artha/position_classification.py` - broker-position sector and industry repair.
- `artha/execution_learning.py` - post-trade execution-quality measurements.
- `artha/fill_finalizer.py` - broker-fill accounting and idempotent state finalization.
- `artha/robinhood_bridge.py` - broker handoff, review, clearance, and reconciliation.
- `artha/scheduler.py` - scheduled scans and lifecycle orchestration.
- `artha/supervisor.py` - production health and readiness checks.
- `artha_mcp/` - MCP tools, resources, prompts, authorization, jobs,
  redaction, research adapters, exact-order receipts, and broker reconciliation.
- `dashboard/` - local operator dashboard.

See [Architecture](docs/ARCHITECTURE.md) for the complete component map.

## Quick Start

Requirements: Python 3.12 or newer and Node.js 20 or newer.

```bash
git clone https://github.com/akira231097/artha-council.git
cd artha-council
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
npm ci
cp .env.example .env
```

Add your own provider credentials to `.env`. Keep every Robinhood setting at its
safe default while evaluating the system.

```bash
python run.py overview
python run.py analyze AAPL MSFT V
python run.py broker-router-preview --assume-market-open --no-persist
python run.py supervise
artha-mcp --check
```

### MCP

Connect any MCP-compatible host through local stdio using
[`.mcp.json.example`](.mcp.json.example), or deploy stateless Streamable HTTP
behind OAuth and TLS. The server starts read-only with its kill switch engaged.

```bash
artha-mcp
```

The user supplies their own model subscriptions, data subscriptions, broker
credentials, and notification credentials. The MCP server never provides or
shares the maintainer's accounts. See [Artha MCP](docs/MCP.md) and
[India Support](docs/MCP_INDIA.md) for the exact capability and safety boundary.
See [MCP Updates](docs/MCP_UPDATES.md) for release fingerprints, rolling versus
stable channels, and the fail-closed local-source promotion path.

## Verification

```bash
python -m compileall -q artha artha_mcp dashboard run.py
python -m unittest discover -s tests -t . -v
python scripts/check_release_integrity.py
python -m artha.test_enhancements
python -m artha.test_alpha_pipeline_hardening
python -m artha.test_production_hardening
python -m artha.test_feedback_loop_hardening
```

Tests use synthetic fixtures and must not require real broker credentials.

### Docker

The included image supports the core Python research CLI:

```bash
docker build -t artha-council .
docker run --rm --env-file .env artha-council overview
```

The broker snapshot helper is intentionally outside this minimal Python image;
it requires a separately configured Node.js MCP runtime and broker access.

`Dockerfile.mcp` builds the non-root MCP image. The `main` image tag is refreshed
after each tested default-branch update; release versions remain immutable.

## Public Release Boundary

This repository contains source, tests, documentation, and safe configuration
templates. It intentionally excludes:

- API keys, OAuth tokens, and `.env` files
- broker account identifiers and snapshots
- portfolios, order history, and transaction records
- SQLite journals and runtime state
- generated dossiers, traces, reports, and logs
- Telegram tokens, chat identifiers, and callback tokens
- local OpenClaw, launchd, and workstation configuration
- third-party market-data payloads

Third-party data and services remain governed by their own licenses and terms.
The Apache license covers Artha's source code, not provider data or broker access.

## Project Documents

- [Architecture](docs/ARCHITECTURE.md)
- [Design notes and diagram](docs/DESIGN.md)
- [Public release and data boundary](docs/PUBLIC_RELEASE.md)
- [Artha MCP](docs/MCP.md)
- [MCP source alignment and updates](docs/MCP_UPDATES.md)
- [India support](docs/MCP_INDIA.md)
- [Security policy](SECURITY.md)
- [Contributing](CONTRIBUTING.md)
- [Authors](AUTHORS.md)
- [Changelog](CHANGELOG.md)
- [Apache License 2.0](LICENSE)

## License and Citation

Artha Council is licensed under [Apache License 2.0](LICENSE). Distributed copies
and modifications must retain the applicable license, copyright, and attribution
notices described in [NOTICE](NOTICE).

Use [CITATION.cff](CITATION.cff) to cite the project. GitHub exposes it through
the repository's **Cite this repository** control.
