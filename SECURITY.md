# Security Policy

Artha handles market data and can be integrated with brokerage tooling. Treat any
money-moving deployment as security-sensitive infrastructure.

## Reporting a Vulnerability

Use [GitHub private vulnerability reporting](https://github.com/akira231097/artha-council/security/advisories/new).
Do not place credentials, account identifiers, exploit details, or private broker
output in a public issue.

Include:

- affected module and revision
- reproducible steps using synthetic data
- expected and observed behavior
- possible impact
- suggested mitigation, if known

The maintainer will acknowledge a complete report when it is reviewed. No fixed
response or remediation deadline is promised for this personal open-source project.

## Public Data Boundary

Never commit:

- `.env` or provider credentials
- OAuth state, access tokens, or refresh tokens
- broker snapshots, account numbers, positions, orders, or fills
- portfolio files, journals, databases, generated dossiers, or traces
- Telegram tokens, chat identifiers, or callback tokens
- local automation configuration, lock files, logs, or temporary handoffs

Use invented values in examples and tests. If a real secret reaches Git history,
revoke it immediately; deleting the current file is not sufficient.

## Safe Defaults

The checked-in configuration is intentionally incapable of live trading:

- review-only mode enabled
- dry-run mode enabled
- agentic broker access disabled
- auto-buy and auto-sell disabled
- runtime kill switch enabled
- fresh-snapshot and exact-review requirements enabled

Do not weaken fail-closed behavior. Missing, stale, malformed, or contradictory
broker evidence must block placement.

## MCP Deployments

- Keep stdio local unless a remote deployment is required.
- Terminate remote HTTP with TLS and configure issuer-, audience-, and
  resource-bound OAuth/JWKS validation.
- Treat `artha:trade` as a privileged money-moving scope.
- Keep local access mode, operations flag, trading flag, limits, and kill switch
  as the final permission ceiling; remote scopes must never widen it.
- Supply credentials through the process environment or a secret manager, never
  in MCP tool arguments, prompts, repository files, or container layers.
- Pin OCI releases by version or digest for live deployments.
- Audit custom broker and research plugins as trusted in-process code.
- Reconcile any `UNKNOWN` placement at the broker before retrying.

## Dependency and Supply-Chain Controls

The repository uses pinned GitHub Actions, Dependabot, dependency review, CodeQL,
and OpenSSF Scorecard. Release reviews should also verify dependency licenses and
inspect GitHub's SPDX-compatible software bill of materials.
