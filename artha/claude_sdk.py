"""
Claude Agent SDK Wrapper — Sync Interface for Artha

Provides a simple synchronous call_claude() function that uses the
Claude Agent SDK with OAuth authentication.

Compatible with both sync contexts (manual CLI) and async contexts
(daemon scheduler) via nest_asyncio.

Token resolution:
  1. CLAUDE_CODE_OAUTH_TOKEN env var
  2. Personal workspace .env (~/.openclaw/workspace-personal/.env)
"""

import asyncio
import logging
import os
import threading
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

logger = logging.getLogger(__name__)

# Personal workspace .env path for OAuth token fallback
_WORKSPACE_ENV = Path.home() / ".openclaw" / "workspace-personal" / ".env"


def _ensure_oauth_token() -> None:
    """Ensure CLAUDE_CODE_OAUTH_TOKEN is in the environment.

    Also removes ANTHROPIC_API_KEY if present — the bundled CLI will
    prefer it over the OAuth token, and stale/invalid keys cause
    silent auth failures on large prompts.
    """
    # Ensure no ANTHROPIC_API_KEY leaks into the CLI subprocess.
    os.environ.pop("ANTHROPIC_API_KEY", None)

    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return

    # Fallback: read from personal workspace .env
    if _WORKSPACE_ENV.exists():
        for line in _WORKSPACE_ENV.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == "CLAUDE_CODE_OAUTH_TOKEN":
                os.environ["CLAUDE_CODE_OAUTH_TOKEN"] = value.strip()
                logger.info("Loaded OAuth token from personal workspace .env")
                return

    raise EnvironmentError(
        "No OAuth token found. Set CLAUDE_CODE_OAUTH_TOKEN env var "
        "or add it to ~/.openclaw/workspace-personal/.env"
    )


def _run_in_new_loop(coro, timeout: float):
    """Run an async coroutine in a brand-new event loop on a separate thread.
    
    This avoids the 'asyncio.run() cannot be called from a running event loop'
    error when called from within the async daemon scheduler.
    """
    result = None
    error = None

    def _target():
        nonlocal result, error
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    asyncio.wait_for(coro, timeout=timeout)
                )
            finally:
                loop.close()
        except Exception as e:
            error = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout + 5)  # Extra 5s grace for thread cleanup

    if thread.is_alive():
        raise TimeoutError(f"Claude Agent SDK call timed out (thread still alive after {timeout}s)")
    if error is not None:
        if isinstance(error, asyncio.TimeoutError):
            raise TimeoutError(f"Claude Agent SDK call timed out after {timeout}s")
        raise error
    return result


def call_claude(
    prompt: str,
    model: str = "claude-opus-4-6",
    timeout_sec: float = 60.0,
) -> tuple[str, dict]:
    """
    Call Claude via the Agent SDK (sync wrapper).

    Works from both sync code (CLI) and async code (daemon scheduler)
    by running the Agent SDK query in a separate thread with its own event loop.

    Args:
        prompt: The user prompt to send.
        model: Model name (e.g. 'claude-opus-4-6', 'claude-sonnet-4-6').
        timeout_sec: Timeout in seconds.

    Returns:
        (response_text, usage_dict) where usage_dict has
        'input_tokens' and 'output_tokens' keys.
    """
    _ensure_oauth_token()

    async def _run() -> tuple[str, dict]:
        options = ClaudeAgentOptions(
            model=model,
            max_turns=1,
            permission_mode="bypassPermissions",
        )
        text = ""
        usage: dict = {}

        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text += block.text
            if isinstance(msg, ResultMessage) and hasattr(msg, "usage"):
                raw = msg.usage if isinstance(msg.usage, dict) else {}
                usage = {
                    "input_tokens": raw.get("input_tokens", 0),
                    "output_tokens": raw.get("output_tokens", 0),
                }

        return text.strip(), usage

    text, usage = _run_in_new_loop(_run(), timeout=timeout_sec)

    logger.info(
        f"Agent SDK call complete: model={model} "
        f"({usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out)"
    )
    return text, usage
