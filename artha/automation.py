"""Launchd automation helpers for running Artha on macOS."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "launchd"


def _python_path() -> str:
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _plist(label: str, args: list[str], extra: str) -> str:
    arg_xml = "\n".join(f"        <string>{escape(str(arg))}</string>" for arg in args)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{escape(label)}</string>
    <key>WorkingDirectory</key>
    <string>{escape(str(PROJECT_ROOT))}</string>
    <key>ProgramArguments</key>
    <array>
{arg_xml}
    </array>
    <key>StandardOutPath</key>
    <string>{escape(str(PROJECT_ROOT / "data" / "logs" / f"{label}.out.log"))}</string>
    <key>StandardErrorPath</key>
    <string>{escape(str(PROJECT_ROOT / "data" / "logs" / f"{label}.err.log"))}</string>
{extra}
</dict>
</plist>
"""


def build_launchd_plists(python_path: str | None = None) -> dict[str, str]:
    """Return launchd plist bodies for Artha monitoring and calibration."""
    py = python_path or _python_path()
    run_py = str(PROJECT_ROOT / "run.py")
    return {
        "com.artha.monitor.plist": _plist(
            "com.artha.monitor",
            [py, run_py, "monitor"],
            """    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
""",
        ),
        "com.artha.calibrate-nightly.plist": _plist(
            "com.artha.calibrate-nightly",
            [py, run_py, "calibrate"],
            """    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>21</integer>
        <key>Minute</key>
        <integer>30</integer>
    </dict>
""",
        ),
        "com.artha.diagnose-nightly.plist": _plist(
            "com.artha.diagnose-nightly",
            [py, run_py, "diagnose", "--telegram"],
            """    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>21</integer>
        <key>Minute</key>
        <integer>45</integer>
    </dict>
""",
        ),
        "com.artha.supervise-nightly.plist": _plist(
            "com.artha.supervise-nightly",
            [py, run_py, "supervise", "--telegram"],
            """    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>21</integer>
        <key>Minute</key>
        <integer>55</integer>
    </dict>
""",
        ),
        "com.artha.portfolio-check.plist": _plist(
            "com.artha.portfolio-check",
            [py, run_py, "check"],
            """    <key>StartInterval</key>
    <integer>1800</integer>
""",
        ),
    }


def write_launchd_plists(output_dir: Path | None = None, python_path: str | None = None) -> dict[str, Any]:
    """Write launchd plist templates under data/launchd and return paths."""
    target = output_dir or DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    written = {}
    for filename, body in build_launchd_plists(python_path=python_path).items():
        path = target / filename
        path.write_text(body, encoding="utf-8")
        written[filename] = str(path)
    return {
        "output_dir": str(target),
        "written": written,
        "load_hint": "Copy wanted plists to ~/Library/LaunchAgents and run launchctl bootstrap gui/$(id -u) <plist>.",
    }
