#!/usr/bin/env python3
"""Install the fail-closed Artha source-promotion runner as a macOS LaunchAgent."""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_LABEL = "com.artha.public-source-sync"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("install", "uninstall", "print"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    return parser


def _payload(args: argparse.Namespace) -> dict:
    source = args.source.expanduser().resolve()
    public = args.public.expanduser().resolve()
    python = public / ".venv" / "bin" / "python"
    runner = public / "scripts" / "artha_source_sync.py"
    if not python.is_file() or not runner.is_file():
        raise ValueError("public checkout must contain .venv/bin/python and the sync runner")
    if not (source / ".git").exists() or not (public / ".git").exists():
        raise ValueError("source and public paths must be Git checkouts")
    if args.interval < 120:
        raise ValueError("interval must be at least 120 seconds")
    logs = Path.home() / "Library" / "Logs" / "Artha"
    logs.mkdir(parents=True, exist_ok=True)
    path = ":".join(
        value
        for value in (
            str(Path(sys.executable).resolve().parent),
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        )
        if Path(value).exists()
    )
    return {
        "Label": args.label,
        "ProgramArguments": [
            str(python),
            str(runner),
            "publish",
            "--source",
            str(source),
            "--public",
            str(public),
            "--auto-merge",
        ],
        "WorkingDirectory": str(public),
        "RunAtLoad": True,
        "StartInterval": args.interval,
        "ProcessType": "Background",
        "LowPriorityIO": True,
        "EnvironmentVariables": {"PATH": path},
        "StandardOutPath": str(logs / f"{args.label}.out.log"),
        "StandardErrorPath": str(logs / f"{args.label}.err.log"),
    }


def main() -> int:
    if sys.platform != "darwin":
        print("This installer supports macOS launchd only.", file=sys.stderr)
        return 2
    args = _parser().parse_args()
    destination = Path.home() / "Library" / "LaunchAgents" / f"{args.label}.plist"
    if args.command == "uninstall":
        subprocess.run(
            ["launchctl", "bootout", f"gui/{os.getuid()}", str(destination)],
            check=False,
            capture_output=True,
        )
        destination.unlink(missing_ok=True)
        return 0
    payload = _payload(args)
    rendered = plistlib.dumps(payload, fmt=plistlib.FMT_XML, sort_keys=False)
    if args.command == "print":
        sys.stdout.buffer.write(rendered)
        return 0
    if not shutil.which("gh") or not shutil.which("gitleaks"):
        print("Both gh and gitleaks must be installed before enabling automatic sync.", file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(rendered)
    os.chmod(temporary, 0o600)
    temporary.replace(destination)
    subprocess.run(
        ["launchctl", "bootout", f"gui/{os.getuid()}", str(destination)],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["launchctl", "bootstrap", f"gui/{os.getuid()}", str(destination)],
        check=True,
    )
    subprocess.run(
        ["launchctl", "enable", f"gui/{os.getuid()}/{args.label}"],
        check=True,
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
