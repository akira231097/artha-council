# Changelog

All notable public releases of Artha Council are documented here.

## 1.4.0 - 2026-08-11

- Synchronized the public core through local production source commit
  `6a43dd1786e`, including sector-aware routing, non-blocking scan work,
  completed trade-learning/fill cleanup, guarded feedback-loop audits, and the
  rebuilt operations dashboard.
- Added an MCP-visible build/source fingerprint and release manifest so clients
  can prove that the loaded Artha core and MCP boundary are from one artifact.
- Rebuilt the rolling MCP image from every tested `main` commit, embedded OCI
  commit/version provenance, and added in-container startup verification.
- Added a source-only local promotion runner that excludes runtime/private data,
  requires clean committed code, Gitleaks, full regression tests, protected pull
  requests, and fail-closed conflict handling. The managed boundary includes
  the three broker runner modules while preserving the portable public overlay.
- Added release-integrity tests covering version parity, Docker/CI coupling,
  private-data boundaries, source drift, and secret-safe promotion.

## 1.3.0 - 2026-08-11

- Added an MCP server built with the Python SDK 2.0, with structured tools,
  resources, prompts, persisted workflow jobs, local stdio, and stateless
  Streamable HTTP.
- Added fail-closed capability policy, OAuth/JWKS validation, recursive secret
  redaction, DNS-rebinding controls, and safe read-only defaults.
- Added immutable exact-order preview receipts, atomic daily caps, final broker
  rechecks, duplicate-order detection, broker-order status tools, and
  no-retry ambiguous-outcome reconciliation.
- Added direct Upstox and Zerodha cash-equity adapters plus a broker and research
  plugin boundary for portable deployments.
- Added explicit US/India market profiles, whole-share Indian delivery rules,
  sell-holdings proof, and a hard block against reusing the US Council as an
  India-native analysis engine.
- Added broker-verified Indian instrument lookup, explicit static-IP readiness,
  limit-only India API execution, official Upstox/Kite response parsing, and
  netting of settled holdings with same-day positions.
- Added exact-order Upstox margin proof, explicit broker business-status
  validation, market-aware USD/INR trading caps, crossed/stale quote rejection,
  bounded broker/snapshot payloads, and serialized workflow startup.
- Removed workstation-specific Node and account fallbacks from the OpenClaw
  runners, packaged their locked assets with wheels, and made account selection
  explicit and fail-closed.
- Added OCI packaging, MCP Registry metadata, automated GHCR builds, deployment
  documentation, and MCP protocol/security/adapter regression tests.
- Made Upstox BSE equity-series handling exchange-native, separated live and
  sandbox credentials, and labeled sandbox placement as submission-only because
  the broker sandbox does not expose fill reconciliation.
- Required duplicate/status capability for every live direct broker adapter,
  made missing India live credentials a startup error, and restricted portable
  Robinhood runner handoffs and OAuth state to owner-only files.
- Bounded the MCP workflow queue, stored MCP-generated state in owner-only
  paths, and stopped Upstox sandbox reads from calling its unavailable order
  book while retaining explicit submission-only warnings.

## 1.2.0 - 2026-08-11

- Synchronized the public source with the current buy, sell, Council, scheduler,
  broker-bridge, and portfolio-risk implementation.
- Added alpha shadow evaluation, rank-coverage enforcement, position
  classification, execution learning, and idempotent broker-fill finalization.
- Added portable deterministic snapshot, auto-trade, and manual-review MCP
  runners without workstation-specific paths or account identifiers.
- Added sell-accounting and position-classification repair utilities.
- Expanded the regression suite with isolated portfolio fixtures so public tests
  do not depend on private runtime state.

## 1.1.1 - 2026-08-10

- Raised the minimum `requests` and `python-dotenv` versions above four published
  security advisories and verified the environment with `pip-audit`.
- Upgraded CodeQL and its Scorecard SARIF uploader to v4, setup-node to v6, and
  the Gitleaks action to its Node 24 v3 runtime.
- Added a direct private vulnerability-reporting link and enabled repository
  security controls and protected-branch checks.

## 1.1.0 - 2026-08-10

- Published the current buy-side funnel, broker-aware router, opportunity scout,
  multi-role Council, CIO synthesis, and execution officer implementation.
- Published portfolio-capacity, buy stand-down, sell-council, thesis-monitoring,
  trailing-stop, reconciliation, and fail-closed broker bridge logic.
- Added the current production-hardening regression suite and dashboard source.
- Added Apache License 2.0, author attribution, machine-readable citation,
  contribution guidance, security policy, and public-release documentation.
- Added pinned CI, CodeQL, dependency review, Dependabot, and OpenSSF Scorecard
  configuration.
