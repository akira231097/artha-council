# MCP Source Alignment and Updates

Artha and its MCP server are one monorepo and one Python distribution. The MCP
imports the `artha` package directly; there is no copied decision engine to
update separately.

The committed private operator checkout is authoritative for shared Artha
decision and execution logic. Promotion is intentionally one-way into the
sanitized public repository. Remote-only MCP, adapter, documentation, or
packaging changes rebuild the public artifact, but public commits are never
silently pulled into a live trading checkout.

## Update Channels

- `ghcr.io/akira231097/artha-council-mcp:main` is the rolling channel. Every
  tested commit merged to public `main` rebuilds this tag and an immutable
  commit-SHA tag.
- `ghcr.io/akira231097/artha-council-mcp:<version>` is a stable, versioned
  release. A new semantic version and Git tag publish a new MCP Registry entry.
- A source checkout or editable installation reads the Artha package in that
  checkout. Restart the MCP process after code changes so Python loads the new
  modules.

An image tag moving does not modify an already running container. Deployments
must pull and restart the rolling image, or deliberately upgrade to a newer
stable version. This is intentional: silently replacing code inside a process
that can reach a broker is not a safe update mechanism.

## Machine Proof

`artha_sync_status` and `artha://sync-status` return:

- package and build version
- immutable build commit and Git ref when supplied by CI
- current runtime-source SHA-256 fingerprint
- expected release fingerprint
- the sanitized local-source baseline
- rolling and stable update channels

`PASS` means the loaded Artha core and MCP files match the release manifest.
`WARN` identifies a development checkout or unavailable commit metadata.
`FAIL` means the source bytes or version surfaces disagree; packaged MCP
startup and CI stop rather than serving an unverified build.

Run the same proof locally:

```bash
python scripts/check_release_integrity.py
artha-mcp --check
```

## Local Production Source Promotion

The live operator checkout contains private portfolio and broker state and has
a separate Git history. It must never be mirrored or pushed to the public
repository. `scripts/artha_source_sync.py` instead compares a committed-source
fingerprint and transfers only:

- `artha/**/*.py`, except the public-only portable path module
- `dashboard/**/*.{py,js,html,css,md}`
- `run.py`
- the three allowlisted Robinhood snapshot/review/auto-trade runner modules

It excludes credentials, `.env` files, `data/`, databases, broker snapshots,
logs, reports, traces, OpenClaw state, dashboard state, and every other script.
The public runner files contain portability and privacy adaptations. Future
private changes are applied as patches over that public overlay; an overlap
that cannot be applied cleanly stops for review instead of replacing it.

Status check:

```bash
python scripts/artha_source_sync.py status \
  --source /path/to/live/artha \
  --public /path/to/artha-public-release
```

Automated promotion creates an isolated worktree, applies only the allowlisted
patch, requires a clean committed source tree, runs local pattern checks and
Gitleaks, runs the complete MCP and Artha regression suites, pushes a dedicated
branch, and opens a protected pull request:

```bash
python scripts/artha_source_sync.py publish \
  --source /path/to/live/artha \
  --public /path/to/artha-public-release \
  --auto-merge
```

Auto-merge still waits for required GitHub checks. Conflicts, missing tools,
test failures, secret findings, stale baselines, or uncommitted core edits stop
the promotion.

On macOS, install the guarded poller after authenticating `gh` and installing
Gitleaks:

```bash
python scripts/install_artha_source_sync.py install \
  --source /path/to/live/artha \
  --public /path/to/artha-public-release \
  --interval 300
```

It checks committed managed source every five minutes, writes atomic health
state under `~/Library/Application Support/artha-council/`, and opens an
auto-merge pull request only after local verification succeeds. Uncommitted
edits are reported as `BLOCKED`; they are never published.

## Remote Update Contract

GitHub Actions runs the MCP build on every pull request and every `main` push;
there is deliberately no path filter. The image embeds OCI revision/version
labels and the same values as runtime environment metadata. CI starts the
built image and verifies both labels and `artha-mcp --check` before publishing.

The MCP Registry requires a unique version for changed metadata, so stable
updates are released under a new semantic version. The mutable `main` channel
is for operators who explicitly choose rolling updates.

Neither a merged commit nor a moved container tag can replace code already
loaded in memory. A rolling deployment must pull the new `main` image and
restart; a source process must restart. This explicit activation boundary is
required for software that can reach a brokerage account.
