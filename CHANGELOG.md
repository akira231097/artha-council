# Changelog

All notable public releases of Artha Council are documented here.

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
