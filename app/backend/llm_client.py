"""Async OpenRouter client wrapper used by profile generation and the API.

OpenRouter is OpenAI-compatible, so the AsyncOpenAI client from the `openai`
package works directly — just swap base_url + api_key.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    """Return a shared AsyncOpenAI client configured for OpenRouter."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set in .env")
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client
