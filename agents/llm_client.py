"""
agents/llm_client.py — a provider-agnostic chat interface.

The rest of the codebase should never talk to Groq / Anthropic / OpenAI /
Gemini SDKs directly. Instead it calls :func:`get_llm` to obtain an
:class:`LLMClient` and uses two methods:

    complete(messages, *, temperature, max_tokens, json_mode) -> str
    complete_json(messages, ...) -> dict

The active provider is selected by ``LLM_PROVIDER`` (see ``config.py``).
Groq remains the default so the existing deployment keeps working unchanged;
Anthropic / OpenAI / Gemini are drop-in alternatives that only require the
relevant SDK and API key.

Design notes
------------
* Providers are imported lazily so that installing, say, only ``groq`` does
  not force ``anthropic`` and ``openai`` to be present.
* ``complete_json`` is resilient: it prefers native JSON modes where available
  and otherwise strips ``` fences and extracts the first balanced JSON object,
  because small/instruct models occasionally wrap JSON in prose.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

from config import settings
from agents.logging_config import get_logger

log = get_logger("llm_client")

Message = dict[str, str]  # {"role": "system"|"user"|"assistant", "content": str}


class LLMError(RuntimeError):
    """Raised when the underlying provider call fails irrecoverably."""


def _extract_json(text: str) -> dict[str, Any]:
    """Best-effort extraction of a JSON object from a model response.

    Handles the common failure modes: ```json fences, leading/trailing prose,
    and single quotes. Raises :class:`LLMError` if nothing parseable is found.
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced { ... } span.
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError(f"Model did not return valid JSON: {text[:200]!r}")


class LLMClient(ABC):
    """Abstract chat client. Concrete subclasses wrap a single provider."""

    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """Return the assistant's text completion for ``messages``."""

    def complete_json(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call :meth:`complete` in JSON mode and parse the result."""
        raw = self.complete(
            messages,
            temperature=temperature if temperature is not None else 0.0,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return _extract_json(raw)


class GroqClient(LLMClient):
    """Default provider — wraps the Groq OpenAI-compatible chat API."""

    def __init__(self, model: str, api_key: str | None) -> None:
        super().__init__(model, api_key)
        from groq import Groq  # lazy import

        self._client = Groq(api_key=api_key)

    def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — normalize all SDK errors
            raise LLMError(f"Groq request failed: {e}") from e
        return resp.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    """Anthropic Claude via the Messages API."""

    def __init__(self, model: str, api_key: str | None) -> None:
        super().__init__(model, api_key)
        import anthropic  # lazy import

        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        # Anthropic separates the system prompt from the turn list.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        if json_mode:
            system = (system + "\n\nRespond with a single valid JSON object and nothing else.").strip()
        try:
            resp = self._client.messages.create(
                model=self.model,
                system=system or None,
                messages=turns,
                temperature=settings.llm_temperature if temperature is None else temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Anthropic request failed: {e}") from e
        return "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")


class OpenAIClient(LLMClient):
    """OpenAI chat completions (also works for OpenAI-compatible endpoints)."""

    def __init__(self, model: str, api_key: str | None) -> None:
        super().__init__(model, api_key)
        from openai import OpenAI  # lazy import

        self._client = OpenAI(api_key=api_key)

    def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"OpenAI request failed: {e}") from e
        return resp.choices[0].message.content or ""


class GeminiClient(LLMClient):
    """Google Gemini via the google-generativeai SDK."""

    def __init__(self, model: str, api_key: str | None) -> None:
        super().__init__(model, api_key)
        import google.generativeai as genai  # lazy import

        genai.configure(api_key=api_key)
        self._genai = genai

    def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        # Gemini uses "user"/"model" roles and a `parts` payload.
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]}
            for m in messages
            if m["role"] != "system"
        ]
        gen_config: dict[str, Any] = {
            "temperature": settings.llm_temperature if temperature is None else temperature,
        }
        if max_tokens is not None:
            gen_config["max_output_tokens"] = max_tokens
        if json_mode:
            gen_config["response_mime_type"] = "application/json"
        try:
            model = self._genai.GenerativeModel(self.model, system_instruction=system or None)
            resp = model.generate_content(contents, generation_config=gen_config)
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Gemini request failed: {e}") from e
        return resp.text or ""


_PROVIDERS: dict[str, type[LLMClient]] = {
    "groq": GroqClient,
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """Return the configured :class:`LLMClient` singleton.

    Provider and model come from ``LLM_PROVIDER`` / ``LLM_MODEL``. Raises
    :class:`LLMError` for an unknown provider so misconfiguration fails loudly
    at first use rather than silently defaulting.
    """
    provider = settings.llm_provider
    cls = _PROVIDERS.get(provider)
    if cls is None:
        raise LLMError(
            f"Unknown LLM_PROVIDER {provider!r}. Supported: {', '.join(sorted(_PROVIDERS))}."
        )
    log.info("LLM provider=%s model=%s", provider, settings.llm_model)
    return cls(model=settings.llm_model, api_key=settings.llm_api_key)
