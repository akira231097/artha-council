"""Private, crash-safe storage helpers for MCP runtime artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TextIO
from uuid import uuid4


def ensure_private_dir(path: Path) -> Path:
    """Create an owner-only runtime directory when the platform supports modes."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            path.chmod(0o700)
        except OSError:
            pass
    return path


def ensure_private_file(path: Path) -> Path:
    """Restrict an existing runtime file to its owner on POSIX systems."""
    if os.name == "posix":
        try:
            path.chmod(0o600)
        except OSError:
            pass
    return path


def open_private_text(path: Path) -> TextIO:
    """Open a truncated owner-only text file for subprocess output."""
    ensure_private_dir(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    ensure_private_file(path)
    return os.fdopen(descriptor, "w", encoding="utf-8")


def atomic_write_private_text(path: Path, text: str) -> None:
    """Atomically replace a text artifact without a world-readable temp file."""
    ensure_private_dir(path.parent)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            descriptor = None
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        ensure_private_file(path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
