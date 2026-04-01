"""
LLM answer generator using Ollama as the backend (fallback option).

Set `backend = "ollama"` in config.toml to use this instead of MLX.
Requires:
    brew install ollama
    ollama serve
    ollama pull llama3.2   # or whatever model you set in config

Uses the same system prompt and context formatting as MLXGenerator
so answers are comparable when switching backends.

Returns token counts when the Ollama response includes ``prompt_eval_count``
and ``eval_count`` (common in current Ollama APIs).
"""
from __future__ import annotations

import logging

from .generator_mlx import SYSTEM_PROMPT_GROUNDED, _build_user_message
from .retriever import RetrievedChunk
from .token_usage import TokenUsage

logger = logging.getLogger(__name__)


class OllamaGenerator:
    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        temperature: float = 0.1,
        num_predict: int = 512,
        system_prompt: str | None = None,
    ) -> None:
        self._model = model
        self._host = host
        self._temperature = temperature
        self._num_predict = num_predict
        self._system_prompt = system_prompt or SYSTEM_PROMPT_GROUNDED

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, TokenUsage | None]:
        """Generate a grounded answer via the Ollama local API."""
        import ollama  # type: ignore[import-untyped]

        if not chunks:
            return "I don't know based on the provided documents.", None

        client = ollama.Client(host=self._host)
        try:
            response = client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user",   "content": _build_user_message(question, chunks)},
                ],
                options={
                    "temperature": self._temperature,
                    "num_predict": self._num_predict,
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ollama request failed. Is 'ollama serve' running?\n"
                f"Host: {self._host}  Model: {self._model}\n"
                f"Error: {exc}"
            ) from exc

        msg = response["message"]["content"].strip()
        pe = response.get("prompt_eval_count")
        ev = response.get("eval_count")
        usage: TokenUsage | None = None
        if pe is not None or ev is not None:
            usage = TokenUsage(
                prompt_tokens=int(pe) if pe is not None else None,
                completion_tokens=int(ev) if ev is not None else None,
            )
        return msg, usage
