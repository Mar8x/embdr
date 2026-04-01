"""
LLM answer generator using Anthropic Claude API.

Set `backend = "claude"` in config.toml and provide an API key via:
- config `claude_api_key`, or
- environment variable `ANTHROPIC_API_KEY` (recommended).
"""
from __future__ import annotations

import logging
from typing import Any

from .generator_mlx import SYSTEM_PROMPT_GROUNDED, _build_user_message
from .retriever import RetrievedChunk
from .token_usage import TokenUsage

logger = logging.getLogger(__name__)


class ClaudeGenerator:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt or SYSTEM_PROMPT_GROUNDED

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, TokenUsage | None]:
        """Generate a grounded answer via Claude Messages API."""
        from anthropic import Anthropic  # type: ignore[import-untyped]

        if not chunks:
            return "I don't know based on the provided documents.", None

        client = Anthropic(api_key=self._api_key)
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=self._system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": _build_user_message(question, chunks),
                    }
                ],
            )
        except Exception as exc:
            raise RuntimeError(
                f"Claude request failed.\nModel: {self._model}\nError: {exc}"
            ) from exc

        text_parts: list[str] = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", ""))

        answer = "\n".join(p for p in text_parts if p).strip()
        if not answer:
            answer = "I don't know based on the provided documents."

        usage_obj: TokenUsage | None = None
        raw_usage: Any = getattr(response, "usage", None)
        if raw_usage is not None:
            usage_obj = TokenUsage(
                prompt_tokens=getattr(raw_usage, "input_tokens", None),
                completion_tokens=getattr(raw_usage, "output_tokens", None),
            )

        return answer, usage_obj
