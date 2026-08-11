"""Bounded subprocess jobs for long-running Artha workflows."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .markets import normalize_instrument
from .models import WorkflowJob
from .security import redact
from .storage import atomic_write_private_text, ensure_private_dir, open_private_text


class WorkflowJobManager:
    """Run only allowlisted Artha workflows and persist inspectable handles."""

    def __init__(
        self,
        root: Path,
        data_dir: Path,
        *,
        timeout_seconds: int = 10800,
        max_pending_jobs: int = 8,
    ) -> None:
        if not 1 <= max_pending_jobs <= 64:
            raise ValueError("max_pending_jobs must be between 1 and 64")
        self.root = root
        self.state_dir = data_dir / "mcp" / "jobs"
        self.log_dir = data_dir / "mcp" / "job_logs"
        ensure_private_dir(data_dir / "mcp")
        ensure_private_dir(self.state_dir)
        ensure_private_dir(self.log_dir)
        self.timeout_seconds = timeout_seconds
        self.max_pending_jobs = max_pending_jobs
        self._executor = ThreadPoolExecutor(
            max_workers=min(2, max_pending_jobs), thread_name_prefix="artha-mcp-job"
        )
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._owned_job_ids: set[str] = set()
        self._lock = threading.RLock()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def _path(self, job_id: str) -> Path:
        if not job_id.startswith("mcpjob_") or not job_id[7:].isalnum():
            raise ValueError("Invalid MCP job id")
        return self.state_dir / f"{job_id}.json"

    def _write(self, job: WorkflowJob) -> None:
        target = self._path(job.job_id)
        atomic_write_private_text(target, job.model_dump_json(indent=2))

    def _read(self, job_id: str) -> WorkflowJob:
        try:
            return WorkflowJob.model_validate_json(
                self._path(job_id).read_text(encoding="utf-8")
            )
        except FileNotFoundError as exc:
            raise ValueError("MCP workflow job was not found") from exc

    def _command(
        self, workflow: str, *, symbols: list[str], limit: int, telegram: bool
    ) -> list[str]:
        run = [sys.executable, "-m", "run"]
        if workflow == "scheduled_scan":
            return [*run, "scheduled-scan"]
        if workflow == "analyze":
            if not symbols:
                raise ValueError("analyze requires at least one symbol")
            if len(symbols) > 8:
                raise ValueError("analyze accepts at most 8 symbols per job")
            clean = []
            for symbol in symbols:
                value = str(symbol).upper().strip()
                try:
                    normalized = normalize_instrument(value, market="US").symbol
                except ValueError:
                    raise ValueError(f"Invalid symbol: {symbol}")
                clean.append(normalized)
            return [*run, "analyze", *clean]
        if workflow == "scan":
            command = [*run, "scan", str(max(1, min(limit, 40)))]
            if telegram:
                command.append("--telegram")
            return command
        if workflow == "supervisor":
            return [*run, "supervise"]
        if workflow == "execution_readiness":
            return [*run, "execution-readiness"]
        if workflow == "broker_router_preview":
            return [*run, "broker-router-preview", "--no-persist"]
        if workflow == "sell_review":
            return [*run, "sell-review"]
        raise ValueError("Unknown or disallowed Artha workflow")

    def start(
        self,
        workflow: str,
        *,
        symbols: list[str] | None = None,
        limit: int = 8,
        telegram: bool = False,
    ) -> WorkflowJob:
        with self._lock:
            command = self._command(
                workflow, symbols=symbols or [], limit=limit, telegram=telegram
            )
            active_jobs = [
                row
                for row in self.list(limit=100)
                if row.status in {"queued", "running"}
            ]
            if len(active_jobs) >= self.max_pending_jobs:
                raise ValueError(
                    "MCP workflow queue is at its configured pending-job limit "
                    f"({self.max_pending_jobs}); wait for or cancel an existing job"
                )
            if workflow in {"scheduled_scan", "scan"}:
                for existing in active_jobs:
                    if existing.workflow in {
                        "scheduled_scan",
                        "scan",
                    } and existing.status in {"queued", "running"}:
                        raise ValueError(
                            f"A scan job is already {existing.status}: {existing.job_id}"
                        )
            job_id = f"mcpjob_{uuid4().hex}"
            log_path = self.log_dir / f"{job_id}.log"
            job = WorkflowJob(
                job_id=job_id,
                workflow=workflow,
                status="queued",
                command=[
                    Path(part).name if index in {0, 1} else part
                    for index, part in enumerate(command)
                ],
                created_at=self._now(),
                log_path=log_path.name,
                message="Queued by Artha MCP.",
            )
            self._write(job)
            self._owned_job_ids.add(job_id)
            self._executor.submit(self._run, job_id, command, log_path)
        return job

    def _run(self, job_id: str, command: list[str], log_path: Path) -> None:
        job = self._read(job_id)
        job.status = "running"
        job.started_at = self._now()
        job.message = "Workflow process started."
        self._write(job)
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["ARTHA_ROOT"] = str(self.root)
        env["ARTHA_DATA_DIR"] = str(self.state_dir.parent.parent)
        env["ARTHA_MCP_DATA_DIR"] = str(self.state_dir.parent.parent)
        try:
            with open_private_text(log_path) as output:
                process = subprocess.Popen(
                    command,
                    cwd=self.root,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                with self._lock:
                    self._processes[job_id] = process
                try:
                    return_code = process.wait(timeout=self.timeout_seconds)
                    status = "succeeded" if return_code == 0 else "failed"
                    message = f"Workflow exited with code {return_code}."
                except subprocess.TimeoutExpired:
                    self._terminate_process_tree(process)
                    return_code = -9
                    status = "failed"
                    message = f"Workflow exceeded {self.timeout_seconds}s timeout."
        except Exception as exc:  # noqa: BLE001 - subprocess startup can raise platform-specific errors
            return_code = -1
            status = "failed"
            message = f"Workflow failed to start: {type(exc).__name__}."
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
        job = self._read(job_id)
        if job.status != "cancelled":
            job.status = status
            job.return_code = return_code
            job.message = message
        job.finished_at = self._now()
        self._write(job)

    def get(self, job_id: str, *, tail_lines: int = 80) -> dict[str, Any]:
        job = self._read(job_id)
        log_tail = ""
        path = self.log_dir / Path(job.log_path).name
        if path.exists():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = redact("\n".join(lines[-max(1, min(tail_lines, 200)) :]))
        return {**job.model_dump(mode="json"), "log_tail": log_tail}

    def list(self, *, limit: int = 20) -> list[WorkflowJob]:
        rows: list[WorkflowJob] = []
        for path in sorted(
            self.state_dir.glob("mcpjob_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            try:
                row = WorkflowJob.model_validate_json(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if (
                row.status in {"queued", "running"}
                and row.job_id not in self._processes
                and row.job_id not in self._owned_job_ids
            ):
                row.status = "unknown"
                row.message = (
                    "Server restarted before this workflow reached a terminal state; "
                    "inspect the log and Artha journal."
                )
                self._write(row)
            rows.append(row)
            if len(rows) >= max(1, min(limit, 100)):
                break
        return rows

    def cancel(self, job_id: str) -> WorkflowJob:
        with self._lock:
            process = self._processes.get(job_id)
        job = self._read(job_id)
        if process is None or process.poll() is not None:
            raise ValueError("Workflow is not running in this MCP process")
        self._terminate_process_tree(process)
        job.status = "cancelled"
        job.finished_at = self._now()
        job.message = "Cancellation requested by MCP operator."
        self._write(job)
        return job

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        """Stop the isolated workflow process group without leaving children behind."""
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - exercised on Windows deployments
                process.terminate()
            process.wait(timeout=10)
        except (ProcessLookupError, PermissionError):
            return
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    return
            else:  # pragma: no cover - exercised on Windows deployments
                process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        """Terminate owned workflows and release executor threads on server shutdown."""
        with self._lock:
            processes = list(self._processes.values())
            owned_job_ids = list(self._owned_job_ids)
        for process in processes:
            self._terminate_process_tree(process)
        self._executor.shutdown(wait=True, cancel_futures=True)
        for job_id in owned_job_ids:
            try:
                job = self._read(job_id)
            except ValueError:
                continue
            if job.status not in {"queued", "running"}:
                continue
            job.status = "cancelled"
            job.finished_at = self._now()
            job.message = "MCP server shut down before this workflow completed."
            self._write(job)
