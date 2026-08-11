#!/usr/bin/env python3
"""Promote committed Artha core changes into the sanitized public monorepo.

The source and public repositories intentionally have different Git histories.
This tool transfers only an allowlisted source patch in an isolated worktree;
it never pushes a source branch or copies runtime state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_release_integrity import (
    PROMOTED_RUNNER_SCRIPTS,
    committed_source_fingerprint,
    dirty_managed_paths,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows is not an installed sync target
    fcntl = None

MANIFEST_NAME = "artha_release_manifest.json"
ALLOWED_PUBLIC_PATH = re.compile(
    r"^(?:artha/(?!paths\.py$).+\.py|dashboard/.+\.(?:py|js|html|css|md)|run\.py)$"
)
SECRET_PATTERNS = {
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "provider_token": re.compile(
        r"(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16})"
    ),
    "telegram_token": re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{30,}\b"),
    "personal_absolute_path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "private_network_address": re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.\d{1,3}){2})\b"
    ),
}


def _allowed_public_path(path: str) -> bool:
    return bool(ALLOWED_PUBLIC_PATH.fullmatch(path)) or path in PROMOTED_RUNNER_SCRIPTS


def _run(
    args: list[str],
    *,
    cwd: Path,
    input_bytes: bytes | None = None,
    check: bool = True,
    timeout: int = 900,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        args,
        cwd=cwd,
        input=input_bytes,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return _run(["git", "-C", str(repo), *args], cwd=repo, check=check)


def _decode(payload: bytes) -> str:
    return payload.decode("utf-8", errors="replace").strip()


def _manifest(repo: Path) -> dict[str, Any]:
    try:
        payload = json.loads((repo / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid public release manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError("public release manifest must be an object")
    return payload


def _manifest_from_ref(repo: Path, ref: str) -> dict[str, Any]:
    completed = _git(repo, "show", f"{ref}:{MANIFEST_NAME}", check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{ref} does not contain a readable {MANIFEST_NAME}")
    try:
        payload = json.loads(_decode(completed.stdout))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid {MANIFEST_NAME} at {ref}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{MANIFEST_NAME} at {ref} must be an object")
    return payload


def _state_path() -> Path:
    configured = os.getenv("ARTHA_SOURCE_SYNC_STATE_FILE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "artha-council"
        / "source-sync-state.json"
    )


def _write_state(payload: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as output:
        output.write(rendered)
        output.flush()
        os.fsync(output.fileno())
        temp_path = Path(output.name)
    os.chmod(temp_path, 0o600)
    temp_path.replace(path)


@contextmanager
def _lock() -> Iterator[None]:
    path = _state_path().with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _source_status(
    source: Path,
    public: Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest if manifest is not None else _manifest(public)
    recorded = manifest.get("source_provenance")
    recorded = recorded if isinstance(recorded, dict) else {}
    commit, fingerprint, count = committed_source_fingerprint(source)
    dirty = dirty_managed_paths(source)
    recorded_fingerprint = str(recorded.get("tree_sha256") or "")
    return {
        "status": "BLOCKED" if dirty else "CURRENT" if fingerprint == recorded_fingerprint else "DRIFT",
        "source_commit": commit,
        "source_tree_sha256": fingerprint,
        "source_file_count": count,
        "recorded_commit": recorded.get("commit"),
        "recorded_tree_sha256": recorded_fingerprint or None,
        "dirty_managed_paths": dirty,
    }


def _changed_allowed_paths(source: Path, baseline: str, head: str) -> list[str]:
    result = _git(
        source,
        "diff",
        "--name-only",
        "-z",
        baseline,
        head,
        "--",
        "artha",
        "dashboard",
        "scripts",
        "run.py",
    )
    paths = [
        value.decode("utf-8")
        for value in result.stdout.split(b"\0")
        if value and _allowed_public_path(value.decode("utf-8"))
    ]
    return sorted(set(paths))


def _source_patch(source: Path, baseline: str, head: str) -> tuple[bytes, list[str]]:
    ancestry = _git(source, "merge-base", "--is-ancestor", baseline, head, check=False)
    if ancestry.returncode != 0:
        raise RuntimeError("recorded source commit is not an ancestor of current source HEAD")
    paths = _changed_allowed_paths(source, baseline, head)
    if not paths:
        return b"", []
    patch = _git(
        source,
        "diff",
        "--binary",
        "--full-index",
        baseline,
        head,
        "--",
        *paths,
    ).stdout
    return patch, paths


def _scan_staged(worktree: Path) -> list[str]:
    names = _decode(_git(worktree, "diff", "--cached", "--name-only", "-z").stdout).split("\0")
    violations: list[str] = []
    for raw in names:
        path = raw.strip()
        if not path:
            continue
        if path != MANIFEST_NAME and not _allowed_public_path(path):
            violations.append(f"path outside source-promotion allowlist: {path}")
            continue
        target = worktree / path
        if not target.is_file():
            continue
        text = target.read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                violations.append(f"{label} pattern in {path}")
    return violations


def _verify(worktree: Path, source: Path, python: Path) -> None:
    commands = [
        [str(python), "-m", "compileall", "-q", "artha", "artha_mcp", "run.py"],
        [
            str(python),
            "scripts/check_release_integrity.py",
            "--source",
            str(source),
        ],
        [str(python), "-m", "unittest", "discover", "-s", "tests", "-t", "."],
        [str(python), "-m", "artha.test_enhancements"],
        [str(python), "-m", "artha.test_production_hardening"],
        [str(python), "-m", "artha.test_alpha_pipeline_hardening"],
        [str(python), "-m", "artha.test_feedback_loop_hardening"],
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(worktree)
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=worktree,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=1800,
        )
        if completed.returncode != 0:
            detail = (completed.stdout + "\n" + completed.stderr)[-6000:]
            raise RuntimeError(f"verification failed: {' '.join(command)}\n{detail}")


def _require_gitleaks(worktree: Path) -> None:
    executable = shutil.which("gitleaks")
    if not executable:
        raise RuntimeError("gitleaks is required before an automated public push")
    completed = subprocess.run(
        [executable, "detect", "--source", str(worktree), "--no-git", "--redact"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError("gitleaks rejected the staged public source tree")


def _existing_pr(public: Path, branch: str) -> str:
    completed = _run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            "akira231097/artha-council",
            "--head",
            branch,
            "--state",
            "open",
            "--json",
            "url",
            "--jq",
            ".[0].url // empty",
        ],
        cwd=public,
        check=False,
        timeout=30,
    )
    return _decode(completed.stdout) if completed.returncode == 0 else ""


def publish(source: Path, public: Path, *, auto_merge: bool) -> dict[str, Any]:
    source = source.expanduser().resolve()
    public = public.expanduser().resolve()
    _git(source, "rev-parse", "--is-inside-work-tree")
    _git(public, "rev-parse", "--is-inside-work-tree")
    _git(public, "fetch", "--quiet", "origin", "main")
    public_manifest = _manifest_from_ref(public, "origin/main")
    status = _source_status(source, public, manifest=public_manifest)
    if status["status"] == "BLOCKED":
        return {**status, "message": "Commit or discard managed source edits before promotion."}
    if status["status"] == "CURRENT":
        return {**status, "message": "Public source baseline already matches committed Artha."}

    baseline = str((public_manifest.get("source_provenance") or {}).get("commit") or "")
    head = str(status["source_commit"])
    if not re.fullmatch(r"[0-9a-f]{40}", baseline):
        raise RuntimeError("release manifest does not contain a valid source baseline commit")
    branch = f"sync/artha-source-{head[:12]}"
    existing = _existing_pr(public, branch)
    if existing:
        return {**status, "status": "PENDING", "branch": branch, "pull_request": existing}

    patch, paths = _source_patch(source, baseline, head)
    if not patch or not paths:
        raise RuntimeError("source fingerprint changed but no allowlisted source patch was produced")

    python = public / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError("public release virtual environment is missing")

    with tempfile.TemporaryDirectory(prefix="artha-public-sync-") as temp:
        worktree = Path(temp) / "repo"
        _git(public, "worktree", "add", "--detach", str(worktree), "origin/main")
        try:
            _git(worktree, "switch", "-c", branch)
            applied = _run(
                ["git", "apply", "--index", "--whitespace=error-all", "-"],
                cwd=worktree,
                input_bytes=patch,
                check=False,
            )
            if applied.returncode != 0:
                raise RuntimeError(
                    "source patch conflicts with the portable public overlay; manual review is required: "
                    + _decode(applied.stderr)[-2000:]
                )
            refresh = _run(
                [
                    str(python),
                    "scripts/check_release_integrity.py",
                    "--source",
                    str(source),
                    "--refresh",
                ],
                cwd=worktree,
                check=False,
                timeout=300,
            )
            _git(worktree, "add", MANIFEST_NAME)
            violations = _scan_staged(worktree)
            if violations:
                raise RuntimeError("public-source safety scan failed: " + "; ".join(violations))
            _require_gitleaks(worktree)
            if refresh.returncode != 0:
                raise RuntimeError("release manifest refresh failed: " + _decode(refresh.stdout + refresh.stderr))
            _verify(worktree, source, python)
            _git(worktree, "config", "user.name", "Artha Source Sync")
            _git(worktree, "config", "user.email", "akira231097@users.noreply.github.com")
            _git(worktree, "commit", "-m", f"sync: promote Artha source {head[:12]}")
            _git(worktree, "push", "--set-upstream", "origin", branch)
            body = (
                "Automated source-only promotion from the private runtime checkout.\n\n"
                f"- Source commit: `{head}`\n"
                f"- Previous baseline: `{baseline}`\n"
                f"- Managed files changed: {len(paths)}\n"
                "- Runtime data, credentials, broker state, and local deployment files were excluded.\n"
                "- Local compile, integrity, secret, MCP, and Artha regression checks passed.\n\n"
                "GitHub branch protection and CI remain authoritative before merge."
            )
            created = _run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--repo",
                    "akira231097/artha-council",
                    "--base",
                    "main",
                    "--head",
                    branch,
                    "--title",
                    f"Sync Artha source {head[:12]}",
                    "--body",
                    body,
                ],
                cwd=worktree,
                timeout=60,
            )
            pr_url = _decode(created.stdout)
            if auto_merge:
                merged = _run(
                    [
                        "gh",
                        "pr",
                        "merge",
                        pr_url,
                        "--auto",
                        "--squash",
                        "--delete-branch",
                    ],
                    cwd=worktree,
                    check=False,
                    timeout=60,
                )
                if merged.returncode != 0:
                    raise RuntimeError("pull request opened but auto-merge could not be enabled")
            return {
                **status,
                "status": "PENDING",
                "branch": branch,
                "pull_request": pr_url,
                "changed_paths": paths,
                "auto_merge": auto_merge,
            }
        finally:
            _git(public, "worktree", "remove", "--force", str(worktree), check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "publish"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--public", type=Path, default=ROOT)
    parser.add_argument("--auto-merge", action="store_true")
    args = parser.parse_args()
    generated = datetime.now(UTC).isoformat()
    try:
        with _lock():
            result = (
                _source_status(args.source.expanduser().resolve(), args.public.expanduser().resolve())
                if args.command == "status"
                else publish(args.source, args.public, auto_merge=args.auto_merge)
            )
        payload = {"generated_at": generated, **result}
        _write_state(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0
    except BlockingIOError:
        print(json.dumps({"status": "SKIPPED", "message": "source sync is already running"}))
        return 0
    except Exception as exc:  # noqa: BLE001 - deterministic runner must persist every failure
        payload = {
            "generated_at": generated,
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {str(exc)[:3000]}",
        }
        _write_state(payload)
        print(json.dumps(payload, indent=2, ensure_ascii=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
