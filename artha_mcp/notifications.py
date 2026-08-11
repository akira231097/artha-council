"""Portable notifications without putting credentials in MCP arguments."""

from __future__ import annotations

import asyncio
import os
from typing import Any
from urllib.parse import urlparse

import httpx

from .security import redact


class NotificationHub:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    @staticmethod
    def _webhook_url() -> str:
        return str(
            os.getenv("ARTHA_MCP_NOTIFICATION_WEBHOOK_URL")
            or os.getenv("ARTHA_MCP_WEBHOOK_URL")
            or os.getenv("ARTHA_NOTIFICATION_WEBHOOK_URL")
            or ""
        ).strip()

    @staticmethod
    def _valid_webhook_url(value: str) -> bool:
        parsed = urlparse(value)
        return bool(
            parsed.scheme == "https"
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.fragment
        )

    def status(self) -> dict[str, Any]:
        if self.mode == "telegram":
            configured = bool(
                os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID")
            )
        elif self.mode == "webhook":
            configured = self._valid_webhook_url(self._webhook_url())
        elif self.mode in {"none", "off"}:
            configured = True
        else:
            configured = False
        return {"mode": self.mode, "configured": configured}

    async def send(self, message: str) -> dict[str, Any]:
        text = str(message or "").strip()
        if not text:
            raise ValueError("Notification message is empty")
        if len(text) > 4000:
            raise ValueError("Notification message exceeds 4000 characters")
        if self.mode == "telegram":
            from artha.telegram import TelegramSender

            sender = TelegramSender()
            if not sender.enabled:
                return {"status": "BLOCKED", "message": "Telegram is not configured."}
            sent = await asyncio.to_thread(sender.send_message, text, None)
            return {"status": "PASS" if sent else "FAIL", "channel": "telegram"}
        if self.mode == "webhook":
            url = self._webhook_url()
            if not self._valid_webhook_url(url):
                return {"status": "BLOCKED", "message": "Webhook must be an HTTPS URL."}
            headers = {"Content-Type": "application/json"}
            token = str(
                os.getenv("ARTHA_MCP_NOTIFICATION_WEBHOOK_TOKEN")
                or os.getenv("ARTHA_NOTIFICATION_WEBHOOK_TOKEN")
                or ""
            ).strip()
            if token:
                headers["Authorization"] = f"Bearer {token}"
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        url, headers=headers, json={"source": "artha", "message": text}
                    )
            except httpx.HTTPError as exc:
                return {
                    "status": "FAIL",
                    "message": f"Webhook delivery failed: {type(exc).__name__}.",
                }
            if response.status_code < 200 or response.status_code >= 300:
                return {
                    "status": "FAIL",
                    "message": f"Webhook returned HTTP {response.status_code}.",
                }
            return {"status": "PASS", "channel": "webhook"}
        return {
            "status": "BLOCKED",
            "message": f"Notification mode {redact(self.mode)} is disabled or unsupported.",
        }
