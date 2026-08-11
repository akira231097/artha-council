# Artha MCP

Artha MCP exposes the existing Artha research, audit, portfolio, monitoring,
scheduling, notification, and execution boundaries to any MCP-compatible host.
It does not make credentials portable: every installation supplies its own
model, market-data, broker, Telegram, and OAuth credentials.

## What The User Supplies

There are two separate model arrangements:

1. **Host-orchestrated:** the user's MCP host uses its own model subscription to
   call Artha tools, read resources, and follow Artha prompts. No model key is
   passed to the MCP server for these host calls.
2. **Embedded Artha Council:** the installed Artha process calls its configured
   analyst providers. The user supplies those provider credentials and remains
   subject to each provider's plan and terms.

The same separation applies to data and brokerage. Artha never shares the
maintainer's subscriptions, API keys, Robinhood account, or runtime data.

## Supported Boundary

| Capability | US | India |
|---|---|---|
| Portfolio/account read | Robinhood deterministic snapshot or plugin | Upstox, Zerodha, or plugin |
| Broker instrument lookup | Broker/plugin-specific | Upstox search; exact-symbol Zerodha verification |
| Live broker quote | Robinhood/OpenClaw execution bridge or plugin | Upstox, Zerodha, or plugin |
| Exact-order preview/place | Robinhood/OpenClaw bridge | Direct Upstox/Zerodha adapter |
| Embedded Artha Council | Supported | Blocked because it is US-native |
| Host-model research | Supported | Supported with explicit data limitations |
| Cash equities | Supported | Supported |
| Options, margin, shorts, crypto | Not exposed | Not exposed |

India compatibility is real at the MCP, account, holdings, quote, preview, fee,
margin, order, and reconciliation boundaries. It is not a claim that US SEC/FMP
research is valid for Indian companies. See [India Support](MCP_INDIA.md).

## Install And Connect

Use Python 3.12 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
artha-mcp --check
```

A source checkout stores runtime state in its existing `data/` directory. A
wheel installation uses the operating system's per-user application-data
directory. Set `ARTHA_DATA_DIR` (or `ARTHA_MCP_DATA_DIR`) when a deployment
needs an explicit persistent volume. All Council, journal, scheduler, broker,
and MCP subprocess paths use that same directory.

For a local MCP host, copy `.mcp.json.example` into the host's configuration and
replace the executable with an absolute path. Stdio is the default and writes
protocol messages only to stdout; logs go to stderr.

The default server is deliberately read-only:

```text
ARTHA_MCP_ACCESS_MODE=read_only
ARTHA_MCP_OPERATIONS_ENABLED=false
ARTHA_MCP_TRADING_ENABLED=false
ARTHA_MCP_KILL_SWITCH=true
```

## Capability Levels

- `read_only`: portfolio, evidence, dossiers, decisions, watches, health, and
  schedules.
- `operator`: also starts allowlisted workflows, cancels jobs, sends configured
  notifications, and creates broker previews.
- `trading`: also places an immutable preview receipt, but only when operations,
  trading, and the kill switch are all explicitly configured. It is also
  required to build a Robinhood/OpenClaw execution contract or start a live
  auto-trade-capable scan, analysis, or sell-review workflow.

Remote OAuth scopes (`artha:read`, `artha:operate`, `artha:trade`) can only
narrow local permissions. Every remote token needs `artha:read`; operator tokens
also need `artha:operate`, and trading tokens also need `artha:trade`. An OAuth
token cannot override a local kill switch.

## Execution Contract

Artha separates investment judgment from execution:

1. A host or embedded workflow produces an investment decision.
2. `artha_preview_order` asks the broker for a fresh quote and decisive proof.
3. Local gates check market, session, spread, quote age, price cap/floor,
   instrument identity, funds or holdings, per-order cap, and order shape.
4. A short-lived receipt stores the exact order and its SHA-256 hash.
5. `artha_place_previewed_order` atomically claims that receipt.
6. Artha repeats the complete preview and checks broker orders for the same
   immutable action tag before calling placement.
7. An ambiguous network outcome becomes `UNKNOWN` and is never retried until
   broker reconciliation resolves it.
8. `artha_broker_orders` exposes bounded redacted broker status, and
   `artha_reconcile_execution` maps the matching broker order back to the
   receipt without submitting anything. Filled and partially filled receipts
   continue to count toward daily limits, and the same action ID cannot be
   placed again.

Live direct adapters must expose broker order status; Artha blocks placement if
it cannot perform duplicate detection and later reconciliation. A sandbox that
does not expose status may exercise one immutable submission receipt, but the
result is explicitly marked submission-only and is never represented as a fill.

This direct receipt path is available to direct broker adapters. Robinhood uses
the existing audited OpenClaw operation contract because Robinhood MCP owns the
live review/place calls.

The Robinhood bridge requires Node.js 20+, an authorized `robinhood-trading`
OpenClaw MCP connection, and an explicit
`ARTHA_ROBINHOOD_AGENTIC_ACCOUNT_NUMBER`. It never guesses an account. Source
installations use `npm ci`; wheel installations include the runner assets and
lockfile under the installed `share/artha-council/` directory, where the same
locked Node dependencies must be installed before using the external runner.
The Python-only OCI server does not contain Robinhood OAuth state or run that
separate OpenClaw process.

For privacy, `artha_robinhood_execution_operation` does not return the full
Robinhood account number to the host model. Broker argument fields contain an
explicit `${ARTHA_RESOLVED_ROBINHOOD_ACCOUNT_NUMBER}` placeholder plus an
`account_binding` contract. The host must call Robinhood `get_accounts`, find
exactly one active and agentic-allowed account matching Artha's masked suffix,
type, and nickname, replace only that placeholder, and abort on any missing or
ambiguous match. The external OpenClaw runner performs the equivalent binding
inside its audited local process.

Indian direct execution adds mandatory boundaries: whole shares, cash-delivery
limit orders, a registered-static-IP attestation for live placement, and a
separate demat-authorization attestation for delivery sells. The broker API
remains the final authority and can still reject an order.
`artha_broker_search_instruments`
returns broker-verified identifiers so a host does not invent an Upstox token or
silently confuse an NSE and BSE listing.

## Workflows And Communication

Long operations return a persisted `mcpjob_*` handle. Clients poll
`artha_job_status` or read `artha://job/{job_id}`. The server runs only an
allowlist: scheduled scan, bounded scan/analyze, Supervisor, execution
readiness, broker-router preview, and sell review. At most eight jobs may be
queued or running by default (`ARTHA_MCP_MAX_PENDING_JOBS`, valid range 1-64),
and scan jobs are serialized. MCP research artifacts, execution receipts, job
state, and logs are stored in owner-only directories/files on POSIX systems.

The MCP layer exposes the existing Artha SQLite journal, dossiers, theses,
watches, execution queue, portfolio state, scheduler slots, Supervisor status,
Telegram adapter, HTTPS webhook adapter, and Robinhood/OpenClaw operation
contract. It does not install workstation cron or launchd jobs remotely.
Persistent scheduling remains an explicit deployment responsibility. Portable
commands use `python -m run ...`, so wheel and container installations do not
depend on a checkout-specific `run.py` path.

## Remote HTTP

Streamable HTTP is stateless and uses one MCP endpoint:

```bash
ARTHA_MCP_TRANSPORT=streamable-http \
ARTHA_MCP_HOST=0.0.0.0 \
ARTHA_MCP_ALLOWED_HOSTS=artha.example.com \
ARTHA_MCP_ALLOWED_ORIGINS=https://client.example.com \
ARTHA_MCP_OAUTH_ISSUER=https://issuer.example.com \
ARTHA_MCP_OAUTH_RESOURCE_URL=https://artha.example.com/mcp \
ARTHA_MCP_OAUTH_AUDIENCE=artha-api \
ARTHA_MCP_OAUTH_JWKS_URL=https://issuer.example.com/.well-known/jwks.json \
artha-mcp
```

Non-loopback HTTP refuses to start without OAuth metadata. JWT verification is
signature-, issuer-, audience-, expiry-, subject-, and scope-bound. Put a
trusted TLS reverse proxy in front of the process. DNS-rebinding protection,
allowed hosts, allowed origins, and a 1 MiB request limit are enabled. Host
entries without a port are expanded only to the configured bind port; no
wildcard hosts are introduced.

## Container And Updates

`Dockerfile.mcp` builds a non-root OCI image. GitHub Actions builds it on pull
requests and publishes these GHCR tags after pushes:

- `main`: latest tested commit on the default branch.
- commit SHA: immutable build from that commit.
- semantic version and `latest`: release-tag builds.

Following `main` provides automatic source updates but can change behavior.
Production installations should pin a semantic version or digest and update
after reviewing release notes. Every `v*` release tag publishes the matching
immutable image and updates MCP Registry metadata through GitHub OIDC; no
registry password is stored. MCP Registry metadata is in `server.json`; the
registry remains a metadata index, not a credential or hosting service. A
running container is never silently replaced: deployment operators choose when
to pull a new version, while the `main` channel follows tested default-branch
updates.

Custom broker plugins are trusted in-process code. A plugin factory must return
a subclass of `BrokerAdapter`, which includes the complete quote, portfolio,
preview, placement, duplicate-check, order-status, and shutdown contract.

## Sources

- [MCP specification](https://modelcontextprotocol.io/specification/2026-07-28)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [Official MCP Registry publishing](https://modelcontextprotocol.io/registry/quickstart)
- [Official MCP OCI package rules](https://modelcontextprotocol.io/registry/package-types)
