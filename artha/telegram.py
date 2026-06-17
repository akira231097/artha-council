"""Telegram Bot API delivery for Artha alerts and reports.

Uses the Telegram Bot API directly via requests. Handles:
- Message splitting (4096 char limit)
- Retry with exponential backoff
- Markdown/HTML formatting
- Rate limiting (30 msgs/sec bot limit)
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

from .config import Config

logger = logging.getLogger(__name__)

# Telegram max message length
MAX_MESSAGE_LENGTH = 4096


class TelegramSender:
    """Send messages to Telegram via Bot API."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or Config.TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._session = requests.Session()
        self._session.timeout = 30

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send_message(
        self,
        text: str,
        parse_mode: str = "Markdown",
        disable_preview: bool = True,
        silent: bool = False,
        reply_markup: Optional[dict] = None,
    ) -> bool:
        """Send a text message to Telegram. Auto-splits long messages.

        Returns True if all parts sent successfully.
        """
        if not self.enabled:
            logger.warning("Telegram not configured — skipping send")
            return False

        parts = self._split_message(text)
        success = True
        for i, part in enumerate(parts):
            ok = self._send_single(
                part,
                parse_mode=parse_mode,
                disable_preview=disable_preview,
                silent=silent,
                reply_markup=reply_markup if i == 0 else None,
            )
            if not ok:
                # Retry once with plain text (markdown can cause parse errors)
                logger.warning(f"Markdown send failed for part {i+1}, retrying as plain text")
                ok = self._send_single(
                    part,
                    parse_mode=None,
                    disable_preview=disable_preview,
                    silent=silent,
                    reply_markup=reply_markup if i == 0 else None,
                )
            if not ok:
                success = False
            # Brief pause between multi-part messages
            if len(parts) > 1 and i < len(parts) - 1:
                time.sleep(0.5)
        return success

    def _send_single(
        self,
        text: str,
        parse_mode: Optional[str] = "Markdown",
        disable_preview: bool = True,
        silent: bool = False,
        max_retries: int = 3,
        reply_markup: Optional[dict] = None,
    ) -> bool:
        """Send a single message with retry + backoff."""
        payload: dict = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": disable_preview,
            "disable_notification": silent,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        if reply_markup:
            payload["reply_markup"] = reply_markup

        for attempt in range(max_retries):
            try:
                resp = self._session.post(
                    f"{self.base_url}/sendMessage",
                    json=payload,
                    timeout=30,
                )
                data = resp.json()
                if data.get("ok"):
                    return True

                error_code = data.get("error_code", 0)
                description = data.get("description", "Unknown error")

                # Rate limited — respect retry_after
                if error_code == 429:
                    retry_after = data.get("parameters", {}).get("retry_after", 5)
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    time.sleep(retry_after)
                    continue

                # Markdown parse error — caller should retry without parse_mode
                if error_code == 400 and "parse" in description.lower():
                    logger.warning(f"Parse error: {description}")
                    return False

                logger.error(f"Telegram API error {error_code}: {description}")
                return False

            except requests.exceptions.Timeout:
                logger.warning(f"Telegram send timeout (attempt {attempt+1}/{max_retries})")
                time.sleep(2 ** attempt)
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Telegram connection error: {e} (attempt {attempt+1}/{max_retries})")
                time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Unexpected Telegram send error: {e}")
                return False

        logger.error(f"Failed to send Telegram message after {max_retries} retries")
        return False

    @staticmethod
    def _split_message(text: str) -> list[str]:
        """Split text into chunks respecting Telegram's 4096 char limit.

        Splits on newlines when possible, otherwise hard-splits.
        """
        if len(text) <= MAX_MESSAGE_LENGTH:
            return [text]

        parts: list[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= MAX_MESSAGE_LENGTH:
                parts.append(remaining)
                break

            # Find a good split point (newline near the limit)
            chunk = remaining[:MAX_MESSAGE_LENGTH]
            split_at = chunk.rfind("\n")
            if split_at < MAX_MESSAGE_LENGTH // 2:
                # No good newline found — hard split
                split_at = MAX_MESSAGE_LENGTH

            parts.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip("\n")

        return parts

    def send_alert(self, telegram_message: str) -> bool:
        """Send a pre-formatted alert message."""
        return self.send_message(telegram_message, parse_mode=None)

    def send_report(self, report: str) -> bool:
        """Send a council analysis report."""
        return self.send_message(report, parse_mode=None)

    def send_health_check(self, message: str) -> bool:
        """Send daily health check results (only if there are alerts)."""
        return self.send_message(message, parse_mode=None, silent=True)
