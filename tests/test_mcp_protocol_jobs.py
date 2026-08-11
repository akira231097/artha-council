from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import call, patch

from mcp import ClientSession
from mcp.client._memory import InMemoryTransport

import artha_mcp.server as server_module
from artha_mcp.jobs import WorkflowJobManager
from artha_mcp.models import AccessMode, BrokerName
from artha_mcp.server import create_server
from artha_mcp.service import ArthaMCPService

from .mcp_helpers import FakeBroker, test_settings


class TestMCPProtocol(unittest.IsolatedAsyncioTestCase):
    async def test_full_protocol_surface_and_structured_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = replace(
                test_settings(root),
                access_mode=AccessMode.READ_ONLY,
                operations_enabled=False,
                trading_enabled=False,
                kill_switch=True,
                broker=BrokerName.NONE,
                broker_plugin="",
            )
            service = ArthaMCPService(settings, broker=FakeBroker())
            server = create_server(settings, service=service)
            async with (
                InMemoryTransport(server) as streams,
                ClientSession(*streams) as session,
            ):
                initialized = await session.initialize()
                self.assertEqual(initialized.server_info.name, "artha-council")
                tools = await session.list_tools()
                names = {tool.name for tool in tools.tools}
                self.assertIn("artha_capabilities", names)
                self.assertIn("artha_place_previewed_order", names)
                self.assertIn("artha_reconcile_execution", names)
                self.assertIn("artha_broker_orders", names)
                self.assertIn("artha_broker_search_instruments", names)
                destructive = next(
                    tool
                    for tool in tools.tools
                    if tool.name == "artha_place_previewed_order"
                )
                self.assertTrue(destructive.annotations.destructive_hint)
                robinhood = next(
                    tool
                    for tool in tools.tools
                    if tool.name == "artha_robinhood_execution_operation"
                )
                self.assertTrue(robinhood.annotations.destructive_hint)
                result = await session.call_tool(
                    "artha_resolve_instrument",
                    {"symbol": "reliance.ns", "market": "IN"},
                )
                self.assertFalse(result.is_error)
                self.assertEqual(
                    result.structured_content["research_symbol"], "RELIANCE.NS"
                )
                resources = await session.list_resources()
                self.assertIn(
                    "artha://capabilities",
                    {str(item.uri) for item in resources.resources},
                )
                prompts = await session.list_prompts()
                self.assertIn(
                    "artha_india_onboarding",
                    {prompt.name for prompt in prompts.prompts},
                )
                prompt = await session.get_prompt(
                    "artha_research_symbol", {"symbol": "RELIANCE", "market": "IN"}
                )
                self.assertIn("India-native", prompt.messages[0].content.text)

    async def test_read_only_server_rejects_operator_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = replace(
                test_settings(root),
                access_mode=AccessMode.READ_ONLY,
                operations_enabled=False,
                trading_enabled=False,
                kill_switch=True,
                broker=BrokerName.NONE,
                broker_plugin="",
            )
            service = ArthaMCPService(settings, broker=FakeBroker())
            server = create_server(settings, service=service)
            async with (
                InMemoryTransport(server) as streams,
                ClientSession(*streams) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    "artha_start_workflow", {"workflow": "supervisor"}
                )
                self.assertTrue(result.is_error)
                self.assertIn("does not permit", result.content[0].text)

    async def test_read_tools_enforce_remote_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = replace(
                test_settings(root),
                access_mode=AccessMode.READ_ONLY,
                operations_enabled=False,
                trading_enabled=False,
                kill_switch=True,
                broker=BrokerName.NONE,
                broker_plugin="",
            )
            service = ArthaMCPService(settings, broker=FakeBroker())
            server = create_server(settings, service=service)
            with patch.object(server_module, "_scopes", return_value={"unrelated"}):
                async with (
                    InMemoryTransport(server) as streams,
                    ClientSession(*streams) as session,
                ):
                    await session.initialize()
                    result = await session.call_tool("artha_capabilities", {})
                    self.assertTrue(result.is_error)
                    self.assertIn("missing scope", result.content[0].text)


class TestWorkflowJobs(unittest.TestCase):
    def test_job_is_allowlisted_persisted_and_paths_are_not_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            run = root / "run.py"
            run.write_text(
                "import sys\n"
                "from pathlib import Path\n"
                "print('job-ok', sys.argv[1:])\n"
                "print('api_key=must-not-leak', Path.home() / 'private.log')\n",
                encoding="utf-8",
            )
            manager = WorkflowJobManager(root, data, timeout_seconds=30)
            job = manager.start("execution_readiness")
            deadline = time.time() + 5
            state = manager.get(job.job_id)
            while state["status"] in {"queued", "running"} and time.time() < deadline:
                time.sleep(0.05)
                state = manager.get(job.job_id)
            self.assertEqual(state["status"], "succeeded")
            self.assertIn("job-ok", state["log_tail"])
            self.assertNotIn("must-not-leak", state["log_tail"])
            self.assertNotIn(str(Path.home()), state["log_tail"])
            self.assertEqual(state["command"][1:3], ["-m", "run"])
            self.assertFalse(Path(state["log_path"]).is_absolute())
            if os.name == "posix":
                self.assertEqual((data / "mcp" / "jobs").stat().st_mode & 0o777, 0o700)
                self.assertEqual(
                    (data / "mcp" / "job_logs").stat().st_mode & 0o777, 0o700
                )
                self.assertEqual(
                    (data / "mcp" / "jobs" / f"{job.job_id}.json").stat().st_mode
                    & 0o777,
                    0o600,
                )
                self.assertEqual(
                    (data / "mcp" / "job_logs" / f"{job.job_id}.log").stat().st_mode
                    & 0o777,
                    0o600,
                )
            manager.close()

    def test_pending_job_queue_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = WorkflowJobManager(
                root,
                root / "data",
                timeout_seconds=30,
                max_pending_jobs=2,
            )
            with patch.object(manager._executor, "submit", return_value=None):
                manager.start("supervisor")
                manager.start("execution_readiness")
                with self.assertRaisesRegex(ValueError, "pending-job limit"):
                    manager.start("sell_review")
            manager.close()

    def test_concurrent_scan_starts_are_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = WorkflowJobManager(root, root / "data", timeout_seconds=30)
            with (
                patch.object(manager._executor, "submit", return_value=None),
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                futures = [pool.submit(manager.start, "scan") for _ in range(2)]
                outcomes = []
                for future in futures:
                    try:
                        outcomes.append(future.result())
                    except ValueError as exc:
                        outcomes.append(exc)
            self.assertEqual(
                sum(not isinstance(value, Exception) for value in outcomes), 1
            )
            self.assertEqual(
                sum(isinstance(value, ValueError) for value in outcomes), 1
            )
            manager.close()

    def test_owned_running_job_is_not_mislabeled_during_startup_race(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = WorkflowJobManager(root, root / "data", timeout_seconds=30)
            with patch.object(manager._executor, "submit", return_value=None):
                job = manager.start("supervisor")
            job.status = "running"
            job.started_at = manager._now()
            manager._write(job)
            listed = manager.list(limit=10)
            self.assertEqual(listed[0].status, "running")
            manager.close()

    def test_restart_marks_orphaned_queued_scan_unknown_and_unblocks_next_scan(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = WorkflowJobManager(root, root / "data", timeout_seconds=30)
            with patch.object(first._executor, "submit", return_value=None):
                orphaned = first.start("scan")

            restarted = WorkflowJobManager(root, root / "data", timeout_seconds=30)
            listed = restarted.list(limit=10)
            self.assertEqual(listed[0].job_id, orphaned.job_id)
            self.assertEqual(listed[0].status, "unknown")
            with patch.object(restarted._executor, "submit", return_value=None):
                replacement = restarted.start("scan")
            self.assertEqual(replacement.status, "queued")

            restarted.close()
            first.close()

    def test_symbol_injection_and_unknown_workflow_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = WorkflowJobManager(root, root / "data", timeout_seconds=30)
            with self.assertRaisesRegex(ValueError, "Invalid symbol"):
                manager.start("analyze", symbols=["AAPL;rm -rf /"])
            with self.assertRaisesRegex(ValueError, "Invalid symbol"):
                manager.start("analyze", symbols=["--telegram"])
            with self.assertRaisesRegex(ValueError, "Unknown"):
                manager.start("arbitrary-shell")
            manager.close()

    @unittest.skipUnless(os.name == "posix", "process-group assertion is POSIX-only")
    def test_process_tree_termination_escalates_from_term_to_kill(self) -> None:
        class FakeProcess:
            pid = 4321

            def __init__(self) -> None:
                self.wait_count = 0

            def poll(self):
                return None

            def wait(self, *, timeout):
                self.wait_count += 1
                if self.wait_count == 1:
                    raise subprocess.TimeoutExpired("fake", timeout)
                return -signal.SIGKILL

        process = FakeProcess()
        with patch("artha_mcp.jobs.os.killpg") as killpg:
            WorkflowJobManager._terminate_process_tree(process)
        self.assertEqual(
            killpg.call_args_list,
            [call(4321, signal.SIGTERM), call(4321, signal.SIGKILL)],
        )


if __name__ == "__main__":
    unittest.main()
