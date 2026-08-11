# India Support

Artha MCP supports Indian cash equities without pretending that Robinhood,
fractional US orders, SEC filings, or US valuation data apply in India.

## What Works

- INR accounts and portfolio totals.
- NSE and BSE symbol normalization (`RELIANCE.NS`, `500325.BO`).
- Upstox account, funds, holdings, positions, quote depth, exact-order margin
  and brokerage estimates, delivery order, and order-book APIs.
- Zerodha Kite account, margins, holdings, positions, quote depth, margin and
  charge estimates, CNC order, and order-book APIs.
- Whole-share protected limit buy and sell requests.
- Broker-verified instrument lookup through `artha_broker_search_instruments`.
- Buy maximum-price and sell minimum-price guards.
- Delivery-holdings proof before a sell, net of same-day delivery activity and
  still-open sell orders.
- Duplicate-order tags and final broker rechecks.
- A conservative 09:15-15:30 Asia/Kolkata weekday session gate.

The weekday clock is not an exchange holiday calendar. The broker and exchange
remain authoritative for holidays, halts, circuits, auctions, and order status.

## What Is Intentionally Blocked

The built-in Artha funnel, SEC cross-checks, and Council were designed and
calibrated for US equities. They cannot be relabeled as India-ready. In `IN`
mode, embedded scan, analyze, router, and sell-Council workflows remain blocked.

Use one of these research modes:

- `host_orchestrated`: the user's MCP host gathers market-native filings,
  fundamentals, news, and estimates with its own licensed tools. Artha provides
  technical packets, portfolio state, broker tools, audit resources, and
  prompts. The result is a host-model conclusion, not an embedded Council vote.
- `plugin`: a locally installed `module:factory` research adapter supplies a
  market-native evidence packet. The plugin is trusted code and must be audited.

An evidence plugin exposes an async `collect(instrument)` method returning:

```python
{
    "data": {...},
    "completeness": "india_native_full_packet",
    "limitations": [],
}
```

This release does not claim an India-calibrated automated Council or sell
Council. That requires India-native data contracts, outcome calibration, and
regulatory review beyond broker connectivity.

## Upstox Configuration

```text
ARTHA_MCP_MARKET=IN
ARTHA_MCP_BROKER=upstox
ARTHA_MCP_RESEARCH_MODE=host_orchestrated
UPSTOX_ACCESS_TOKEN=...
UPSTOX_SANDBOX_ACCESS_TOKEN=...
ARTHA_UPSTOX_SANDBOX=true
ARTHA_INDIA_STATIC_IP_REGISTERED=false
ARTHA_INDIA_DEMAT_SELL_AUTHORIZED=false
# Optional overrides; values are INR. Omitted defaults are 2500/5000.
ARTHA_MCP_MAX_ORDER_VALUE=2500
ARTHA_MCP_MAX_DAILY_ORDER_VALUE=5000
```

Test account, funds, portfolio, quote, fee preview, and sandbox placement before
disabling the MCP kill switch. Upstox currently exposes sandbox place/modify/
cancel APIs, but not a sandbox order book. A successful sandbox response proves
the adapter's submission payload only; it does **not** prove an exchange fill or
live reconciliation. Artha reports such results as `sandbox_submission_only`
with `reconciliation_available=false`. Test live order status and reconciliation
separately under broker-approved, tightly capped conditions before enabling real
money. Sandbox portfolio reads therefore omit open-order state and say so
explicitly. Upstox sandbox tokens are exclusively for sandbox order APIs, so sandbox
mode requires a separate `UPSTOX_SANDBOX_ACCESS_TOKEN`;
`UPSTOX_ACCESS_TOKEN` remains the credential for live account, quote, margin,
and fee proof. Sandbox placement does not require the live static-IP attestation.
In production,
register the deployment's static IP with Upstox, then set
`ARTHA_INDIA_STATIC_IP_REGISTERED=true`. If the strategy has an
exchange-approved Upstox algo name, configure that exact value in
`ARTHA_UPSTOX_ALGO_NAME`; otherwise leave it empty. Search returns the required
Upstox instrument key and accepts exchange-native NSE/BSE equity series codes
only when the returned segment is `NSE_EQ` or `BSE_EQ`; Artha rechecks the
symbol/key pair before placement.
An order preview fails closed unless Upstox explicitly returns successful
responses for the exact-order margin calculation and brokerage calculation.
Available cash must cover the required margin plus estimated charges.
Any non-sandbox direct broker plugin must expose order status so Artha can check
for duplicate action tags and reconcile an ambiguous placement; otherwise live
placement is blocked before the order call.

## Zerodha Configuration

```text
ARTHA_MCP_MARKET=IN
ARTHA_MCP_BROKER=zerodha
ARTHA_MCP_RESEARCH_MODE=host_orchestrated
KITE_API_KEY=...
KITE_ACCESS_TOKEN=...
ARTHA_INDIA_STATIC_IP_REGISTERED=true
ARTHA_INDIA_DEMAT_SELL_AUTHORIZED=false
# Optional overrides; values are INR. Omitted defaults are 2500/5000.
ARTHA_MCP_MAX_ORDER_VALUE=2500
ARTHA_MCP_MAX_DAILY_ORDER_VALUE=5000
```

Zerodha access tokens are session credentials. Credential renewal is owned by
the user deployment; Artha does not persist or return them through MCP.
Zerodha cash-equity identifiers use the exact `NSE:SYMBOL` or `BSE:SYMBOL`
quote key and are broker-verified before preview. The deployment's static IP
must also be registered with Zerodha.

## Delivery Sell Authorization

Indian delivery sells can require DDPI/POA or an interactive CDSL TPIN/OTP
authorization. Artha cannot know or bypass the user's depository secret. It
therefore blocks direct Upstox/Zerodha delivery sells unless
`ARTHA_INDIA_DEMAT_SELL_AUTHORIZED=true` is explicitly set. Use that attestation
only while the required authorization is actually valid. A durable DDPI/POA can
support unattended sells; a one-session TPIN authorization may expire and can
still cause the broker to reject the order. Buys do not require this sell-only
attestation.

## Regulatory Boundary

Code portability is not regulatory permission. Anyone operating automated
trading in India must follow their broker's terms and applicable SEBI/exchange
requirements. Public defaults are read-only and live placement must be enabled
deliberately. Indian direct adapters reject API market orders and expose only
whole-share cash-delivery limit execution. This follows the current exchange
operating boundary and preserves an explicit worst acceptable price.

Official references:

- [NSE market timings](https://www.nseindia.com/resources/exchange-communication-holidays)
- [SEBI retail algorithmic trading circular](https://www.sebi.gov.in/legal/circulars/feb-2025/safer-participation-of-retail-investors-in-algorithmic-trading_91614.html)
- [NSE implementation standards](https://nsearchives.nseindia.com/content/circulars/INVG67858.pdf)
- [Upstox static-IP and algo requirements](https://upstox.com/developer/api-documentation/announcements/algo-trading-circular/)
- [Upstox developer API](https://upstox.com/developer/api-documentation/)
- [Upstox margin calculation](https://upstox.com/developer/api-documentation/margin/)
- [Zerodha Kite Connect API](https://kite.trade/docs/connect/v3/)
- [Zerodha holdings authorization](https://kite.trade/docs/connect/v3/portfolio/#holdings-authorisation)
