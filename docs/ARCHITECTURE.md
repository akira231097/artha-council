# Architecture

Artha is organized as a staged decision system. Each stage has a narrow job and writes enough evidence for later audit.

## 1. Universe And Promotion Funnel

The funnel starts with a broad US equity universe from market data providers. It filters out names that are too small, too illiquid, or unsuitable for the configured strategy. It then ranks candidates using momentum, quality, valuation, regime fit, earnings context, liquidity, and repeat/cooldown rules.

Typical shape:

```text
1000+ active stocks
  -> top ranked universe
  -> enriched finalists
  -> broker/data feasible shortlist
  -> council candidates
```

The funnel should find interesting ideas, but it should not make final buy decisions.

## 2. Broker-Aware Router

The router is not a fundamental analyst. Its job is execution and data feasibility:

- Is the quote present and fresh enough?
- Are bid/ask values sane?
- Is the spread too wide for an auto-buy?
- Is the security tradeable and fractional-capable where needed?
- Is the candidate blocked by duplicate order or account constraints?
- Are data providers in severe conflict?

Clean buy-now candidates can consume council slots. Interesting but non-executable candidates move to research/watch instead of being thrown away.

## 3. Opportunity Scout

The opportunity scout is a pre-council agent. It receives the enriched candidates and can inspect additional context before ranking which names deserve the scarce council slots.

It is meant to improve slot quality, not replace the council. It can favor names with better evidence, cleaner valuation, better regime alignment, or better execution readiness.

## 4. Council

The council uses multiple analyst roles:

- Fundamental analyst
- Technical analyst
- Risk/contrarian analyst

Each role evaluates the same evidence packet from a different perspective. The synthesis layer then audits score, valuation anchors, data gaps, source conflicts, invalidation conditions, and final action label.

Common labels:

- `BUY`
- `STARTER`
- `TACTICAL_BUY`
- `DEFER`
- `WATCH`
- `AVOID`

Only buy-side labels can advance toward order placement.

## 5. Execution Officer

The execution officer turns a buy-side council decision into an execution verdict. It checks whether the proposed order is still reasonable at the broker's live quote.

Examples:

- Council says `STARTER`, live quote is clean, spread is tight, buying power is enough: queue auto-buy.
- Council says `STARTER`, but ask has moved above the no-chase cap: no order, create watch.
- Council says `DEFER` or `AVOID`: no broker attempt.

## 6. Robinhood/OpenClaw Bridge

Artha does not assume broker state is fresh. The bridge contract requires:

- read-only broker snapshot refresh
- account identity check
- positions/orders reconciliation
- quote/tradability review
- Robinhood order preview
- final clearance
- place only with exact approved arguments
- post-fill reconciliation

This lets the system be agentic while still being constrained by deterministic safety rules.

## 7. Journal, Dossiers, And Supervisor

Artha records:

- decision dossiers
- agentic traces
- execution intents
- trade actions
- fill reconciliation
- supervisor health checks

Those runtime artifacts are private and are intentionally excluded from this public repository.
