"""Portable runtime paths shared by Artha and its MCP boundary.

A source checkout keeps its historical ``data/`` layout. Installed wheels use
an OS-appropriate user data directory so runtime state is never written into
``site-packages``. Deployments may override every location explicitly.
"""

from __future__ import annotations

import os
import sysconfig
from pathlib import Path

from dotenv import load_dotenv
from platformdirs import user_config_path, user_data_path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _resolved_env_path(name: str) -> Path | None:
    raw = os.getenv(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _source_checkout_root() -> Path | None:
    if (PACKAGE_ROOT / ".git").exists() or (
        (PACKAGE_ROOT / "pyproject.toml").is_file()
        and (PACKAGE_ROOT / "run.py").is_file()
    ):
        return PACKAGE_ROOT
    return None


PROJECT_ROOT = (
    _resolved_env_path("ARTHA_ROOT") or _source_checkout_root() or PACKAGE_ROOT
)
_SOURCE_CHECKOUT = _source_checkout_root() is not None and PROJECT_ROOT == PACKAGE_ROOT

CONFIG_DIR = (
    PROJECT_ROOT
    if _SOURCE_CHECKOUT
    else Path(user_config_path("artha-council", "Sarath")).expanduser().resolve()
)
ENV_FILE = _resolved_env_path("ARTHA_ENV_FILE") or CONFIG_DIR / ".env"

# Explicit process variables remain authoritative in hosted deployments.
load_dotenv(ENV_FILE, override=False)

DATA_DIR = (
    _resolved_env_path("ARTHA_DATA_DIR")
    or _resolved_env_path("ARTHA_MCP_DATA_DIR")
    or (
        PROJECT_ROOT / "data"
        if _SOURCE_CHECKOUT
        else Path(user_data_path("artha-council", "Sarath")).expanduser().resolve()
    )
)


def runtime_asset_path(*parts: str) -> Path:
    """Locate a source-checkout or wheel-installed runtime asset."""

    source_candidate = PROJECT_ROOT.joinpath(*parts)
    if source_candidate.exists():
        return source_candidate
    installed_candidate = (
        Path(sysconfig.get_path("data")) / "share" / "artha-council"
    ).joinpath(*parts)
    return installed_candidate


def runtime_summary() -> dict[str, str | bool]:
    """Return a credential-free description suitable for diagnostics."""

    return {
        "source_checkout": _SOURCE_CHECKOUT,
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "env_file": str(ENV_FILE),
    }
