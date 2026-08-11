# Architecture

Artha is a staged decision and portfolio-management system. Each stage has a
narrow responsibility and passes structured evidence forward.

## 1. Universe and Promotion Funnel

The universe layer starts with active US equities and removes unsupported asset
types and names below configured market-cap, price, and liquidity floors. It then
computes multi-horizon momentum, quality, valuation, pullback, revision, catalyst,
regime-fit, and repeat/cooldown features.

```text
1000+ active equities
  -> ranked universe
  -> enriched shortlist
  -> finalist cards
```

Provider failures and missing fields remain explicit. Missing data is not silently
converted into a positive signal.

## 2. Investment Sanity and Broker-Aware Routing

The sanity layer penalizes unsupported valuation, stale targets, excessive
extension, and regime-inappropriate risk. The broker-aware router then handles
execution feasibility only:

- quote presence and freshness
- bid/ask sanity and spread
- average and live liquidity
- tradability and fractional support
- duplicate/open-order constraints
- severe provider conflicts
- active watch-zone and cooldown state

The router uses three outcomes:

- `execution_ready`: eligible for a buy-now Council slot
- `research_watch`: interesting, but not executable now or awaiting a trigger
- `hard_reject`: unusable identity/data or unsupported instrument

The router does not decide whether the company is fundamentally attractive.

## 3. Opportunity Scout

The scout ranks the bounded finalist set before scarce Council slots are consumed.
It receives structured candidate cards, may collect bounded additional evidence,
and must cite the reasons for its ordering. Deterministic scores remain visible so
the agent cannot silently replace the funnel.

`alpha_shadow.py` records experimental signals separately from production
ranking. Shadow observations can be evaluated later, but cannot promote a stock
or change a live threshold on their own.

## 4. Buy Council and CIO Synthesis

Independent roles evaluate the same evidence packet:

- fundamental quality and valuation
- technical timing and trend
- risk, crowding, and contrary evidence

The CIO synthesis cross-checks their claims against source IDs, filing periods,
valuation anchors, data gaps, and score rules. Buy-side verdicts can advance;
`DEFER`, `WATCH`, and `AVOID` do not create an order.

## 5. Execution Officer

Execution is a separate two-stage decision:

- Stage A converts the Council verdict into immutable order intent: side, symbol,
  quantity/notional, order type, reference price, no-chase cap, and expiry.
- Stage B receives compact, decisive broker evidence for that exact intent and
  decides whether placement remains permitted.

The final decision requires visible proof of quote, tradability, exact-order
review, account capacity, snapshot freshness, and deterministic guardrails.

## 6. Broker Bridge and Reconciliation

The bridge produces bounded operations for the external broker tool owner. It
does not treat model prose as proof. The placement path is:

```text
fresh read-only snapshot
  -> account/position/order reconciliation
  -> live quote and tradability
  -> exact-order broker review
  -> Stage B execution clearance
  -> deterministic final clearance
  -> exact-argument placement
  -> submission/fill reconciliation
  -> idempotent fill finalization and execution-quality measurement
```

Any unknown decisive check blocks the order.

## 7. Capacity and Buy Stand-Down

`broker_capacity.py` derives buy capacity from reconciled broker state. Portfolio,
daily-trade, daily-dollar, and invested-percentage limits can pause new buys.
`stand_down.py` records a buy-only pause and its next-session reset.

Sell monitoring, reconciliation, and health supervision are never paused merely
because buy capacity is exhausted.

## 8. Position Monitoring and Sell Council

Every reconciled holding should have an active thesis containing invalidation,
review, stop, and target state. The monitor checks price, thesis age, trailing
stops, adverse evidence, news, earnings, and portfolio constraints.

Broker holdings are also classified by sector and industry so portfolio-level
concentration checks do not silently operate on missing classifications.

Deterministic emergency rules can create an immediate exit intent when configured.
Judgment triggers receive a fresh sell-Council review. The sell Council can return
hold, trim, or exit; an actionable sell still passes through exact broker review,
execution clearance, placement, and fill reconciliation.

## 9. Journals, Dossiers, Calibration, and Supervisor

Runtime deployments record evidence packets, Council decisions, action intents,
broker reviews, submissions, fills, exact fill accounting, execution-quality
observations, thesis transitions, and health checks.
Calibration and shadow rules can observe outcomes, but they cannot silently alter
live investing rules.

Runtime artifacts are private state and are excluded from this repository.
Runtime paths are centralized in `artha.paths`: a source checkout keeps the
historical repository-local `data/` tree, while an installed wheel uses the
operating system's per-user application-data directory. `ARTHA_DATA_DIR`
provides one explicit persistent-volume override shared by core Artha and all
MCP-launched workflows.

## 10. MCP Boundary

`artha_mcp/` is an interface over these services, not a second investment
engine. It exposes bounded tools, resources, prompts, and persisted workflow
handles. Local stdio is read-only by default; remote Streamable HTTP requires
OAuth when bound beyond loopback.

Direct broker adapters use immutable preview receipts and a separate local
SQLite execution journal. Robinhood remains behind its deterministic snapshot
and OpenClaw-owned MCP review/place bridge. Upstox and Zerodha implement Indian
cash-equity account, portfolio, quote, preview, order, and reconciliation
boundaries. The US-native Council is blocked in India mode so market portability
cannot silently corrupt research assumptions.

## Trust Boundaries

- Structured market and filing providers: evidence inputs, not execution truth.
- LLM analysts and scouts: bounded reasoning, never direct broker authority.
- Broker outputs: source of truth for live quote, tradability, review, and fills.
- Deterministic code: limits, state transitions, exact-order identity, and final
  permission gates.
- Operator configuration: credentials, account identity, risk limits, and whether
  any live capability is enabled.
