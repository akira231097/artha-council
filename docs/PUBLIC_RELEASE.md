# Public Release and Data Boundary

This repository is a source release, not a copy of a running Artha installation.

## Included

- current Python decision, monitoring, and broker-contract source
- synthetic regression tests
- local dashboard source
- safe environment template
- architecture, contribution, security, attribution, and license documents
- automated CI and security workflows

## Excluded

- provider and AI credentials
- Robinhood OAuth and account data
- Telegram credentials and identifiers
- portfolio, position, order, fill, and buying-power records
- SQLite databases and JSON runtime state
- generated reports, dossiers, traces, model output, and market-data payloads
- workstation paths, launch agents, OpenClaw authorization state, and temporary handoffs

These exclusions protect the operator and avoid redistributing third-party data.
They do not represent missing application logic.

## Creating a Public Release

1. Start from the sanitized public repository, never from a live runtime clone.
2. Synchronize only reviewed source, tests, and documentation. For the local
   production checkout, use the allowlisted source-promotion tool documented in
   `docs/MCP_UPDATES.md`; never push its Git branch or history.
3. replace personal deployment paths and identifiers with configuration.
4. Run secret and personal-data scans over files and Git history.
5. Run compile, regression, and static-analysis checks.
6. Review dependency licenses and the repository SBOM.
7. Commit through a review branch and verify GitHub checks before merging.
8. Tag the release and update `CHANGELOG.md` and `CITATION.cff`.

Every public pull request runs the source/version integrity check. Every merge
to `main` rebuilds the rolling MCP image, while stable MCP Registry releases
remain immutable and require a new semantic version.

## Third-Party Services

Artha integrates with external market-data, AI, search, messaging, and broker
services. Users must obtain their own accounts and comply with each service's
terms. Apache License 2.0 grants rights to Artha's code only.

## Live Trading

The public repository ships with live trading disabled. Deployment-specific
OpenClaw/launchd runners and broker authorization state are intentionally not
published. The checked-in broker bridge documents and tests the handoff contract,
but no clone can trade merely because it was downloaded.
