# ARTHA Control Room

ARTHA's read-only operations dashboard. It is designed to answer six questions
without requiring the reader to understand the trading code:

1. Is ARTHA healthy right now?
2. How is the portfolio performing?
3. What does ARTHA own, and how is each holding protected?
4. Why did Council buy, wait, or avoid, and what happened at Robinhood?
5. How did the market universe narrow into Council candidates?
6. What is genuinely learning versus still collecting evidence?

## Views

- **Overview** - alarms, portfolio value, capacity, equity curve, today's work,
  and the latest Council-to-broker journey.
- **Portfolio** - holdings, stops, latest sell review, sector concentration, and
  realized/unrealized results.
- **Decisions** - Council verdict, Execution Officer outcome, and Robinhood fill
  or block reason shown as three separate stages.
- **How ARTHA Works** - live scan coverage, funnel/router/Scout/Council counts,
  and plain-language pipeline explanations.
- **Learning** - benchmark grades, closed trade episodes, post-sell maturity,
  paper trials, calibration, Sentinel, and explicit strategy-effect gates.
- **Health** - the current trading cage, services, all Supervisor checks, and
  the operating schedule.

## Architecture and safety

- `server.py` assembles `/api/dashboard` from ARTHA's local records and SQLite
  journal in read-only mode.
- `index.html`, `styles.css`, and `app.js` provide a dependency-free responsive
  interface suitable for a phone, tablet, or desktop.
- `test_dashboard.py` protects the dashboard contract, privacy boundary,
  current trading limits, decision/execution linkage, and chart anomaly filter.
- Broker account numbers, credentials, and raw broker payloads are never sent
  to the browser.
- The service cannot review, place, cancel, or modify an order. Its only writes
  are the private dashboard token and chart samples under `data/dashboard/`.
- The API refreshes every 15 seconds during estimated regular market hours and
  every minute otherwise. Robinhood remains the authority for market/session
  and trade execution checks.

## Access

- Service: `0.0.0.0:8787` (`ARTHA_DASHBOARD_PORT` can override it).
- Authentication: `data/dashboard/token.txt`, generated with file mode `0600`.
- Open the private URL containing `?k=<token>` once. The server stores a
  same-site, HTTP-only cookie and redirects to a clean URL without the token.
- Home network: `http://192.168.1.158:8787/`
- Tailscale: `http://100.88.234.49:8787/`

Existing saved private links continue to work; the access token is unchanged.

## Service

The launchd label is `com.artha.dashboard`. Logs are written to
`data/logs/dashboard.out.log` and `data/logs/dashboard.err.log`.

## Data-quality note

Portfolio snapshots before `2026-06-16T18:00:00Z` are excluded because the
pre-fix ledger double-counted the first buy. The chart also hides isolated
single-point accounting spikes that strongly disagree with both neighboring
snapshots. The raw journal remains untouched, and the dashboard reports the
number of hidden chart points.
