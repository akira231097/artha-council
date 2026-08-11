#!/usr/bin/env python3
"""Fail-closed integrity checks for the public Artha/MCP monorepo."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artha_mcp.provenance import MANIFEST_NAME, runtime_tree_fingerprint

SOURCE_PATHS = ("artha/",)
PROMOTED_RUNNER_SCRIPTS = {
    "scripts/robinhood_auto_trade_runner.mjs",
    "scripts/robinhood_manual_review_runner.mjs",
    "scripts/robinhood_snapshot_sync.mjs",
}
SOURCE_EXACT = {"run.py", *PROMOTED_RUNNER_SCRIPTS}
SOURCE_EXCLUDED = {"artha/paths.py"}
SOURCE_DASHBOARD_SUFFIXES = {".py", ".js", ".html", ".css", ".md"}
EXPECTED_MANAGED_PATHS = [
    "artha/**/*.py",
    "dashboard/**/*.{py,js,html,css,md}",
    "run.py",
    *sorted(PROMOTED_RUNNER_SCRIPTS),
]


def _run(repo: Path, args: list[str], *, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=not binary,
        timeout=30,
    )
    return completed.stdout


def _managed_source_path(value: str) -> bool:
    path = value.replace("\\", "/")
    if path in SOURCE_EXCLUDED:
        return False
    if path in SOURCE_EXACT:
        return True
    if path.startswith("dashboard/") and Path(path).suffix in SOURCE_DASHBOARD_SUFFIXES:
        return True
    return path.endswith(".py") and path.startswith(SOURCE_PATHS)


def _framed_fingerprint(entries: Iterable[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(entries, key=lambda item: item[0]):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def committed_source_fingerprint(source: Path, commit: str = "HEAD") -> tuple[str, str, int]:
    resolved = str(_run(source, ["rev-parse", commit])).strip()
    names_raw = _run(source, ["ls-tree", "-r", "--name-only", "-z", resolved], binary=True)
    names = [
        value.decode("utf-8")
        for value in bytes(names_raw).split(b"\0")
        if value and _managed_source_path(value.decode("utf-8"))
    ]
    entries = []
    for name in names:
        payload = bytes(_run(source, ["show", f"{resolved}:{name}"], binary=True))
        entries.append((name, payload))
    return resolved, _framed_fingerprint(entries), len(entries)


def dirty_managed_paths(source: Path) -> list[str]:
    raw = str(
        _run(
            source,
            [
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                "artha",
                "dashboard",
                "scripts",
                "run.py",
            ],
        )
    )
    paths: list[str] = []
    for line in raw.splitlines():
        candidate = line[3:].strip()
        if " -> " in candidate:
            candidate = candidate.split(" -> ", 1)[1]
        if _managed_source_path(candidate):
            paths.append(candidate)
    return sorted(set(paths))


def _version_surfaces(root: Path) -> dict[str, str]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    node = json.loads((root / "package.json").read_text(encoding="utf-8"))
    node_lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))
    init_text = (root / "artha" / "__init__.py").read_text(encoding="utf-8")
    init_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)', init_text, re.MULTILINE)
    server = json.loads((root / "server.json").read_text(encoding="utf-8"))
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    citation_match = re.search(r"^version:\s*([^\s]+)", citation, re.MULTILINE)
    image = str(server["packages"][0]["identifier"])
    return {
        "project": str(project["project"]["version"]),
        "package": init_match.group(1) if init_match else "",
        "registry": str(server.get("version") or ""),
        "citation": citation_match.group(1).strip('"\'') if citation_match else "",
        "image": image.rsplit(":", 1)[-1],
        "node": str(node.get("version") or ""),
        "node_lock": str(node_lock.get("version") or ""),
        "node_lock_root": str((node_lock.get("packages", {}).get("") or {}).get("version") or ""),
    }


def _tracked_boundary_errors(root: Path) -> list[str]:
    names = str(_run(root, ["ls-files", "-z"])).split("\0")
    blocked: list[str] = []
    for raw in names:
        path = raw.strip()
        lower = path.lower()
        if not path:
            continue
        blocked_env = lower.startswith("data/") or lower in {".env", ".env.local"}
        blocked_env_variant = lower.startswith(".env.") and lower != ".env.example"
        blocked_suffix = lower.endswith(
            (".bak", ".pem", ".key", ".p12", ".sqlite", ".db")
        )
        if blocked_env or blocked_env_variant or blocked_suffix:
            blocked.append(path)
    return sorted(blocked)


def _workflow_errors(root: Path) -> list[str]:
    workflow = (root / ".github" / "workflows" / "mcp-image.yml").read_text(
        encoding="utf-8"
    )
    dockerfile = (root / "Dockerfile.mcp").read_text(encoding="utf-8")
    errors: list[str] = []
    if re.search(r"(?m)^\s+paths:\s*$", workflow):
        errors.append("MCP image workflow has a path filter and can silently miss build inputs.")
    for token in ("ARTHA_BUILD_COMMIT", "ARTHA_BUILD_REF", "ARTHA_BUILD_VERSION"):
        if token not in workflow:
            errors.append(f"MCP image workflow does not propagate {token}.")
        if token not in dockerfile:
            errors.append(f"Dockerfile does not embed {token}.")
    if "artha_release_manifest.json" not in dockerfile:
        errors.append("Dockerfile does not embed artha_release_manifest.json.")
    return errors


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as output:
        json.dump(payload, output, indent=2, ensure_ascii=True)
        output.write("\n")
        output.flush()
        temp_path = Path(output.name)
    temp_path.replace(path)


def check(root: Path, *, source: Path | None = None, refresh: bool = False) -> dict:
    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    versions = _version_surfaces(root)
    release_version = versions["project"]
    runtime_fingerprint = runtime_tree_fingerprint(root)
    source_state = None
    source_dirty: list[str] = []
    if source is not None:
        source = source.expanduser().resolve()
        commit, source_fingerprint, source_count = committed_source_fingerprint(source)
        source_dirty = dirty_managed_paths(source)
        source_state = {
            "commit": commit,
            "tree_sha256": source_fingerprint,
            "file_count": source_count,
        }

    if refresh:
        manifest["schema_version"] = 1
        manifest["release_version"] = release_version
        manifest["runtime_tree_sha256"] = runtime_fingerprint
        manifest.setdefault("source_provenance", {})
        if source_state is not None:
            manifest["source_provenance"].update(source_state)
            manifest["source_provenance"]["managed_paths"] = EXPECTED_MANAGED_PATHS
        channels = manifest.setdefault("update_channels", {})
        channels["rolling"] = "ghcr.io/akira231097/artha-council-mcp:main"
        channels["stable"] = (
            f"ghcr.io/akira231097/artha-council-mcp:{release_version}"
        )
        _write_manifest(manifest_path, manifest)

    errors: list[str] = []
    warnings: list[str] = []
    if len(set(versions.values())) != 1:
        errors.append(f"Version surfaces disagree: {versions}")
    if str(manifest.get("release_version") or "") != release_version:
        errors.append("Release manifest version does not match the package version.")
    if str(manifest.get("runtime_tree_sha256") or "") != runtime_fingerprint:
        errors.append("Release manifest runtime fingerprint is stale.")
    recorded_source = manifest.get("source_provenance") or {}
    if recorded_source.get("managed_paths") != EXPECTED_MANAGED_PATHS:
        errors.append("Release manifest managed-source boundary is stale.")
    if source_state is not None:
        recorded = recorded_source
        if str(recorded.get("commit") or "") != source_state["commit"]:
            errors.append("Committed local Artha revision differs from the public release baseline.")
        if str(recorded.get("tree_sha256") or "") != source_state["tree_sha256"]:
            errors.append("Committed local Artha source differs from the public release baseline.")
        if int(recorded.get("file_count") or -1) != source_state["file_count"]:
            errors.append("Committed local Artha file count differs from the public release baseline.")
        if source_dirty:
            warnings.append(
                "Local Artha has uncommitted managed source changes: "
                + ", ".join(source_dirty[:20])
            )
    errors.extend(_workflow_errors(root))
    blocked = _tracked_boundary_errors(root)
    if blocked:
        errors.append("Public repository tracks blocked private/runtime paths: " + ", ".join(blocked))

    return {
        "status": "FAIL" if errors else "WARN" if warnings else "PASS",
        "release_version": release_version,
        "versions": versions,
        "runtime_tree_sha256": runtime_fingerprint,
        "source": source_state,
        "source_dirty_paths": source_dirty,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = check(args.root, source=args.source, refresh=args.refresh)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
