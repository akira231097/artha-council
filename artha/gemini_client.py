"""Small REST client for Gemini.

The project already depends on requests, and REST keeps Artha independent of
local SDK import/version quirks in the daemon environment.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from .config import Config

logger = logging.getLogger(__name__)

_SESSION = requests.Session()
_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def gemini_generate(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int = 60,
    google_search: bool = False,
    thinking_level: str | None = None,
    temperature: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate text with Gemini and optionally enable Google Search grounding.

    Returns:
        (text, grounding_metadata)
    """
    api_key = Config.GEMINI_API_KEY or Config.GOOGLE_API_KEY
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY is not configured")

    selected_model = model or Config.GEMINI_FLASH_MODEL
    selected_thinking_level = thinking_level or Config.GEMINI_THINKING_LEVEL
    selected_temperature = Config.GEMINI_TEMPERATURE if temperature is None else temperature
    payload: dict[str, Any] = {
        "contents": [
            {
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": selected_temperature,
        },
    }
    if selected_model.startswith("gemini-3") and selected_thinking_level:
        payload["generationConfig"]["thinkingConfig"] = {
            "thinkingLevel": selected_thinking_level,
        }
    if google_search:
        payload["tools"] = [{"google_search": {}}]

    response = _SESSION.post(
        _BASE_URL.format(model=selected_model),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Gemini request failed ({response.status_code}, model={selected_model}): "
            f"{response.text[:400]}"
        )

    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates (model={selected_model})")

    candidate = candidates[0]
    parts = ((candidate.get("content") or {}).get("parts") or [])
    text_parts = [str(part.get("text") or "") for part in parts if isinstance(part, dict)]
    text = "\n".join(p for p in text_parts if p).strip()
    if not text:
        raise RuntimeError(f"Gemini returned no text output (model={selected_model})")

    grounding = candidate.get("groundingMetadata") or {}
    return text, grounding
