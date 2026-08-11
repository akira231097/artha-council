"""Build and source provenance for the Artha MCP distribution."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from artha import __version__
from artha.paths import PROJECT_ROOT, runtime_asset_path

MANIFEST_NAME = "artha_release_manifest.json"
_PACKAGED_ASSETS = (
    "package.json",
    "package-lock.json",
    "scripts/robinhood_auto_trade_runner.mjs",
    "scripts/robinhood_manual_review_runner.mjs",
    "scripts/robinhood_snapshot_sync.mjs",
)


def _source_files(root: Path) -> list[Path]:
    """Return the executable source files shipped in the MCP artifact."""
    files: list[Path] = []
    for package in ("artha", "artha_mcp"):
        package_root = root / package
        if package_root.is_dir():
            files.extend(
                path
                for path in package_root.rglob("*.py")
                if "__pycache__" not in path.parts
            )
    run_module = root / "run.py"
    if run_module.is_file():
        files.append(run_module)
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def fingerprint_files(root: Path, files: Iterable[Path]) -> str:
    """Hash paths and bytes so renamed or modified files change the digest."""
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def runtime_tree_fingerprint(root: Path | None = None) -> str:
    resolved = (root or PROJECT_ROOT).resolve()
    entries = [
        (path.relative_to(resolved).as_posix(), path.read_bytes())
        for path in _source_files(resolved)
    ]
    for relative in _PACKAGED_ASSETS:
        source_path = resolved / relative
        asset = source_path if source_path.is_file() else runtime_asset_path(*relative.split("/"))
        if asset.is_file():
            entries.append((relative, asset.read_bytes()))
    digest = hashlib.sha256()
    for name, payload in sorted(entries, key=lambda item: item[0]):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _manifest_path() -> Path:
    source_path = PROJECT_ROOT / MANIFEST_NAME
    if source_path.is_file():
        return source_path
    return runtime_asset_path(MANIFEST_NAME)


def load_release_manifest() -> dict[str, Any]:
    path = _manifest_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout.strip()


def build_provenance() -> dict[str, Any]:
    """Return a credential-free alignment report for clients and health checks."""
    manifest = load_release_manifest()
    expected_version = str(manifest.get("release_version") or "")
    expected_fingerprint = str(manifest.get("runtime_tree_sha256") or "")
    source = manifest.get("source_provenance")
    source = source if isinstance(source, dict) else {}
    source_commit = str(source.get("commit") or "")
    source_fingerprint = str(source.get("tree_sha256") or "")
    actual_fingerprint = runtime_tree_fingerprint()
    build_commit = os.getenv("ARTHA_BUILD_COMMIT", "").strip()
    build_ref = os.getenv("ARTHA_BUILD_REF", "").strip()
    build_version = os.getenv("ARTHA_BUILD_VERSION", "").strip()
    runtime_commit = _git(["rev-parse", "HEAD"])
    dirty_rows = _git(
        [
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "artha",
            "artha_mcp",
            "scripts",
            "run.py",
            MANIFEST_NAME,
        ]
    )

    errors: list[str] = []
    warnings: list[str] = []
    if not manifest:
        errors.append("Release provenance manifest is missing or invalid.")
    if manifest and manifest.get("schema_version") != 1:
        errors.append("Release provenance manifest schema is unsupported.")
    if manifest and not expected_version:
        errors.append("Release provenance manifest has no release version.")
    if manifest and not re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint):
        errors.append("Release provenance manifest has an invalid runtime fingerprint.")
    if manifest and not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        errors.append("Release provenance manifest has an invalid source commit.")
    if manifest and not re.fullmatch(r"[0-9a-f]{64}", source_fingerprint):
        errors.append("Release provenance manifest has an invalid source fingerprint.")
    if expected_version and expected_version != __version__:
        errors.append(
            f"Manifest version {expected_version} does not match package version {__version__}."
        )
    if build_version and build_version != __version__:
        errors.append(
            f"Build version {build_version} does not match package version {__version__}."
        )
    if expected_fingerprint and expected_fingerprint != actual_fingerprint:
        errors.append("Runtime source fingerprint does not match the release manifest.")
    if build_commit and runtime_commit and build_commit != runtime_commit:
        errors.append("Container build commit does not match the checked-out runtime commit.")
    if build_commit and not re.fullmatch(r"[0-9a-f]{40}", build_commit):
        errors.append("Container build commit is not a full Git SHA.")
    if dirty_rows:
        warnings.append("Tracked runtime source has uncommitted changes.")
    if not build_commit and not runtime_commit and not source_commit:
        warnings.append("No build or Git commit identifier is available in this installation.")

    status = "FAIL" if errors else "WARN" if warnings else "PASS"
    return {
        "status": status,
        "package_version": __version__,
        "build": {
            "commit": build_commit or None,
            "ref": build_ref or None,
            "version": build_version or None,
        },
        "runtime": {
            "git_commit": runtime_commit or None,
            "tracked_source_dirty": bool(dirty_rows),
            "tree_sha256": actual_fingerprint,
        },
        "manifest": {
            "schema_version": manifest.get("schema_version"),
            "release_version": expected_version or None,
            "tree_sha256": expected_fingerprint or None,
            "source_commit": source.get("commit"),
            "source_tree_sha256": source.get("tree_sha256"),
        },
        "matches": {
            "package_version": bool(expected_version and expected_version == __version__),
            "runtime_tree": bool(
                expected_fingerprint and expected_fingerprint == actual_fingerprint
            ),
            "build_version": not build_version or build_version == __version__,
            "build_commit": not build_commit or not runtime_commit or build_commit == runtime_commit,
        },
        "update_channels": manifest.get("update_channels") or {},
        "errors": errors,
        "warnings": warnings,
        "guarantee": (
            "PASS proves the loaded Artha and MCP source bytes match this release manifest. "
            "A running process still requires restart or redeployment to load a newer build."
        ),
    }
