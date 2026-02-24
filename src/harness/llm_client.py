"""Unified LLM client supporting OpenAI and local (OpenAI-compatible) backends."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Generator

import tiktoken
from openai import OpenAI


@dataclass
class LLMResponse:
    """Result of a single LLM call."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    ttft_s: float | None = None  # time-to-first-token (streaming only)


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""

    provider: str = "openai"  # "openai" or "local"
    model: str = "gpt-5-mini"
    base_url: str | None = None  # Override for local providers
    api_key: str | None = None  # Defaults to OPENAI_API_KEY env var
    temperature: float = 0.0
    max_tokens: int = 4096


class LLMClient:
    """Thin wrapper around the OpenAI SDK that supports both cloud and local backends.

    For local models (ollama, vLLM, llama.cpp), set ``provider="local"`` and
    ``base_url`` to the OpenAI-compatible server URL (e.g. ``http://localhost:11434/v1``).
    """

    def __init__(self, config: LLMConfig | dict | None = None):
        if config is None:
            config = LLMConfig()
        elif isinstance(config, dict):
            config = LLMConfig(**{k: v for k, v in config.items() if v is not None})
        self.config = config

        client_kwargs: dict[str, Any] = {}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        if config.api_key:
            client_kwargs["api_key"] = config.api_key
        elif config.provider == "local":
            # Local servers typically don't need a real key
            client_kwargs["api_key"] = "not-needed"

        self._client = OpenAI(**client_kwargs)

        # Tokenizer for counting when API doesn't return usage
        try:
            self._enc = tiktoken.encoding_for_model(config.model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Synchronous chat completion (non-streaming)."""
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kwargs.pop("temperature", self.config.temperature),
            max_tokens=kwargs.pop("max_tokens", self.config.max_tokens),
            stream=False,
            **kwargs,
        )
        elapsed = time.perf_counter() - t0

        choice = response.choices[0]
        content = choice.message.content or ""

        # Token counts — prefer API-reported, fall back to local counting
        usage = response.usage
        if usage:
            prompt_tok = usage.prompt_tokens
            completion_tok = usage.completion_tokens
            total_tok = usage.total_tokens
        else:
            prompt_tok = sum(len(self._enc.encode(m["content"])) for m in messages)
            completion_tok = len(self._enc.encode(content))
            total_tok = prompt_tok + completion_tok

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            total_tokens=total_tok,
            latency_s=elapsed,
            ttft_s=None,
        )

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        """Streaming chat completion — returns the full aggregated response
        but also measures time-to-first-token."""
        t0 = time.perf_counter()
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=kwargs.pop("temperature", self.config.temperature),
            max_tokens=kwargs.pop("max_tokens", self.config.max_tokens),
            stream=True,
            **kwargs,
        )

        chunks: list[str] = []
        ttft: float | None = None

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks.append(delta.content)

        elapsed = time.perf_counter() - t0
        content = "".join(chunks)

        # Local counting since streaming doesn't always return usage
        prompt_tok = sum(len(self._enc.encode(m["content"])) for m in messages)
        completion_tok = len(self._enc.encode(content))

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            total_tokens=prompt_tok + completion_tok,
            latency_s=elapsed,
            ttft_s=ttft,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in a string using the model's tokenizer."""
        return len(self._enc.encode(text))
