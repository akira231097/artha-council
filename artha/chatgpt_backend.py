"""ChatGPT backend API client using Codex OAuth tokens."""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import requests

from .config import Config

logger = logging.getLogger(__name__)


class ChatGPTBackendClient:
    """Minimal client for ChatGPT backend Responses endpoint."""

    RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
    OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"

    def __init__(
        self,
        auth_path: str | None = None,
        model: str | None = None,
        timeout: int = 90,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
    ) -> None:
        configured_path = auth_path or Config.CODEX_AUTH_PATH
        self.auth_path = Path(configured_path).expanduser()
        self.model = model or Config.GPT_MODEL
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort or Config.GPT_REASONING_EFFORT
        self.temperature = Config.GPT_TEMPERATURE if temperature is None else temperature

    def chat(self, prompt: str) -> str:
        """Send one user prompt and return model text output."""
        return self._chat_with_model(prompt, self.model)

    def _chat_with_model(self, prompt: str, model: str) -> str:
        access_token = self._get_valid_access_token()
        response = self._send_request(prompt, model, access_token)

        if response.status_code == 401:
            logger.warning("ChatGPT backend returned 401, refreshing token and retrying")
            access_token = self._refresh_tokens()
            response = self._send_request(prompt, model, access_token)

        if response.status_code == 404 and model == Config.GPT_MODEL and Config.GPT_FALLBACK_MODEL:
            logger.warning(
                "Model %s unavailable, retrying with %s",
                Config.GPT_MODEL,
                Config.GPT_FALLBACK_MODEL,
            )
            return self._chat_with_model(prompt, Config.GPT_FALLBACK_MODEL)

        if response.status_code == 400 and "Unsupported parameter: temperature" in response.text:
            logger.warning(
                "ChatGPT backend rejected temperature=%s; retrying with reasoning.effort=%s only",
                self.temperature,
                self.reasoning_effort,
            )
            response = self._send_request(prompt, model, access_token, include_temperature=False)

        if response.status_code == 503:
            for attempt in range(1, 3):
                logger.warning("ChatGPT backend returned 503, retrying in 5s (attempt %d/2)", attempt)
                time.sleep(5)
                response = self._send_request(prompt, model, access_token)
                if response.status_code != 503:
                    break
            if response.status_code == 503:
                snippet = response.text[:400]
                raise RuntimeError(f"ChatGPT backend request failed (503) after 2 retries: {snippet}")

        if response.status_code >= 400:
            snippet = response.text[:400]
            raise RuntimeError(
                f"ChatGPT backend request failed ({response.status_code}): {snippet}"
            )

        content_type = response.headers.get("Content-Type", "")
        raw_text = response.text
        looks_like_sse = (
            "text/event-stream" in content_type
            or "data:" in raw_text
            or "event:" in raw_text
        )
        if looks_like_sse:
            text = self._extract_output_text_from_sse(raw_text)
        else:
            try:
                payload = response.json()
            except ValueError as exc:
                text = self._extract_output_text_from_sse(raw_text)
                if not text:
                    snippet = raw_text[:400]
                    raise RuntimeError(
                        f"ChatGPT backend response was not valid JSON: {snippet}"
                    ) from exc
            else:
                text = self._extract_output_text(payload)

        if not text:
            raise RuntimeError("No text output found in ChatGPT backend response")
        return text

    def _send_request(
        self,
        prompt: str,
        model: str,
        access_token: str,
        *,
        include_temperature: bool = True,
    ) -> requests.Response:
        body = {
            "model": model,
            "store": False,
            "stream": True,
            "instructions": "You are a precise financial analysis assistant.",
            "reasoning": {"effort": self.reasoning_effort},
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
        }
        if include_temperature and self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        last_exc: requests.RequestException | None = None
        for attempt in range(1, 4):
            try:
                return requests.post(
                    self.RESPONSES_URL,
                    headers=headers,
                    json=body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= 3:
                    break
                delay = min(2 ** (attempt - 1), 4)
                logger.warning(
                    "ChatGPT backend transport error, retrying in %ss (attempt %d/3): %s",
                    delay,
                    attempt,
                    exc,
                )
                time.sleep(delay)
        raise RuntimeError(f"ChatGPT backend request error after 3 attempts: {last_exc}") from last_exc

    def _get_valid_access_token(self) -> str:
        auth_data = self._load_auth_data()
        tokens = auth_data.get("tokens", {})
        access_token = tokens.get("access_token", "")
        if not access_token:
            raise RuntimeError(f"Missing access token in {self.auth_path}")

        if self._is_jwt_expired(access_token):
            logger.info("Codex access token expired, refreshing")
            return self._refresh_tokens(auth_data)
        return access_token

    def _refresh_tokens(self, auth_data: dict[str, Any] | None = None) -> str:
        auth_data = auth_data or self._load_auth_data()
        tokens = auth_data.get("tokens", {})
        refresh_token = tokens.get("refresh_token", "")
        access_token = tokens.get("access_token", "")
        if not refresh_token:
            raise RuntimeError(f"Missing refresh token in {self.auth_path}")

        client_id = self._extract_client_id(access_token)
        if not client_id:
            client_id = "app_EMoamEEZ73f0CkXaXp7hrann"

        try:
            response = requests.post(
                self.OAUTH_TOKEN_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"OAuth refresh request failed: {exc}") from exc

        if response.status_code >= 400:
            snippet = response.text[:400]
            raise RuntimeError(
                f"OAuth refresh failed ({response.status_code}): {snippet}"
            )

        try:
            refreshed = response.json()
        except ValueError as exc:
            raise RuntimeError("OAuth refresh response was not valid JSON") from exc

        new_access_token = refreshed.get("access_token")
        if not new_access_token:
            raise RuntimeError("OAuth refresh response missing access_token")

        tokens["access_token"] = new_access_token
        if refreshed.get("refresh_token"):
            tokens["refresh_token"] = refreshed["refresh_token"]
        if refreshed.get("id_token"):
            tokens["id_token"] = refreshed["id_token"]

        auth_data["tokens"] = tokens
        auth_data["last_refresh"] = int(time.time())
        self._save_auth_data(auth_data)
        logger.info("Refreshed Codex OAuth tokens")

        return new_access_token

    def _load_auth_data(self) -> dict[str, Any]:
        if not self.auth_path.exists():
            raise RuntimeError(f"Auth file not found: {self.auth_path}")
        try:
            return json.loads(self.auth_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Failed to read auth file {self.auth_path}: {exc}") from exc

    def _save_auth_data(self, auth_data: dict[str, Any]) -> None:
        try:
            self.auth_path.parent.mkdir(parents=True, exist_ok=True)
            self.auth_path.write_text(
                json.dumps(auth_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to write auth file {self.auth_path}: {exc}") from exc

    @staticmethod
    def _extract_output_text(payload: dict[str, Any]) -> str:
        if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
            return payload["output_text"].strip()

        output = payload.get("output", [])
        texts: list[str] = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue
                for chunk in content:
                    if not isinstance(chunk, dict):
                        continue
                    text_value = chunk.get("text")
                    if isinstance(text_value, str) and text_value.strip():
                        texts.append(text_value.strip())
        return "\n".join(texts).strip()

    @staticmethod
    def _extract_output_text_from_sse(raw: str) -> str:
        deltas: list[str] = []
        final_payload: dict[str, Any] | None = None

        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("data:"):
                data = stripped[5:].strip()
            else:
                data = stripped
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue

            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    deltas.append(delta)
            elif event_type == "response.completed":
                response_payload = event.get("response")
                if isinstance(response_payload, dict):
                    final_payload = response_payload

        delta_text = "".join(deltas).strip()
        if delta_text:
            return delta_text
        if final_payload:
            return ChatGPTBackendClient._extract_output_text(final_payload)
        return ""

    @staticmethod
    def _extract_client_id(access_token: str) -> str | None:
        payload = ChatGPTBackendClient._decode_jwt_payload(access_token)
        client_id = payload.get("cid")
        if isinstance(client_id, str) and client_id:
            return client_id
        return None

    @staticmethod
    def _is_jwt_expired(token: str, skew_seconds: int = 30) -> bool:
        payload = ChatGPTBackendClient._decode_jwt_payload(token)
        exp = payload.get("exp")
        if not isinstance(exp, int):
            logger.warning("Access token missing exp claim, forcing refresh")
            return True
        return time.time() >= (exp - skew_seconds)

    @staticmethod
    def _decode_jwt_payload(token: str) -> dict[str, Any]:
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return {}
            payload_part = parts[1]
            padded = payload_part + "=" * (-len(payload_part) % 4)
            decoded = base64.urlsafe_b64decode(padded.encode("utf-8"))
            payload = json.loads(decoded.decode("utf-8"))
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {}
        return {}
