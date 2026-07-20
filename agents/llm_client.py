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


def _extract_sbg_text(data: Any) -> str:
    """Pull the assistant text out of the SBG gateway's JSON response.

    The gateway proxies Bedrock, and response shapes vary by deployment, so this
    checks the common layouts (Anthropic/Bedrock content blocks, OpenAI-style
    ``choices``, and simple ``answer``/``response``/``text`` keys). If none match
    it raises :class:`LLMError` (which triggers the configured-provider fallback)
    and logs the keys so the shape can be added.
    """
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        # OpenAI-style: {"choices":[{"message":{"content":"..."}}]}
        choices = data.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            msg = choices[0].get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(choices[0].get("text"), str):
                return choices[0]["text"]
        # Anthropic/Bedrock: {"content":"..."} or {"content":[{"type":"text","text":"..."}]}
        content = data.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict)]
            if any(parts):
                return "".join(parts)
        # {"message":"..."} or {"message":{"content":"..."}}
        message = data.get("message")
        if isinstance(message, str):
            return message
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        # Simple single-field wrappers
        for key in ("answer", "response", "text", "output", "completion", "result"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val
        log.warning("SBG: unrecognized response shape; top-level keys=%s", list(data.keys()))
    raise LLMError(f"SBG response shape not recognized: {str(data)[:200]!r}")


class SBGClient(LLMClient):
    """Managed SBG/Bedrock chat gateway (POST ``{base}{path}``).

    Mirrors Anthropic's split between a ``system_prompt`` and the ``messages``
    turn list. ``json_mode`` is a no-op (the gateway has no JSON switch); callers
    that need JSON go through :meth:`complete_json`, whose ``_extract_json`` copes
    with prose-wrapped output.
    """

    def __init__(self, base: str, path: str, api_key: str | None, model_id: str, timeout: int) -> None:
        super().__init__(model_id, api_key)
        self.base = base.rstrip("/")
        self.path = path if path.startswith("/") else f"/{path}"
        self.timeout = timeout

    def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        import httpx  # lazy; already a project dependency

        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        # Match the gateway's documented payload exactly (adding unknown fields
        # like temperature risks a 400), so only the fields it expects are sent.
        payload: dict[str, Any] = {
            "model_id": self.model,
            "messages": turns,
            "max_tokens": max_tokens or settings.llm_max_tokens,
        }
        if system:
            payload["system_prompt"] = system
        try:
            resp = httpx.post(
                f"{self.base}{self.path}",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001 — normalize network/HTTP errors
            raise LLMError(f"SBG request failed: {e}") from e
        return _extract_sbg_text(data)


class FallbackLLMClient(LLMClient):
    """Try ``primary`` first; on :class:`LLMError`, use ``fallback``.

    Used to make the SBG/Bedrock gateway the primary model while keeping the
    configured provider (e.g. Groq) as a safety net. After a primary failure it
    skips the primary for ``cooldown`` seconds so a persistently-down gateway
    doesn't add a failed round-trip to every request; it retries once the
    cooldown elapses, so recovery is automatic.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient, cooldown: int = 60) -> None:
        super().__init__(primary.model, primary.api_key)
        self.primary = primary
        self.fallback = fallback
        self.cooldown = cooldown
        self._skip_until = 0.0

    def complete(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        import time

        now = time.monotonic()
        if now >= self._skip_until:
            try:
                return self.primary.complete(
                    messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
                )
            except LLMError as e:
                self._skip_until = now + self.cooldown
                log.warning(
                    "primary LLM (%s) unavailable (%s) -> falling back to %s for ~%ds",
                    type(self.primary).__name__, e, type(self.fallback).__name__, self.cooldown,
                )
        return self.fallback.complete(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )


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
    base_client = cls(model=settings.llm_model, api_key=settings.llm_api_key)

    # Prefer the SBG/Bedrock gateway when configured, with the provider above as
    # an automatic fallback for any call the gateway can't serve.
    if settings.sbg_available:
        log.info(
            "LLM: primary=SBG/Bedrock model=%s (fallback=%s model=%s)",
            settings.sbg_model_id, provider, settings.llm_model,
        )
        sbg = SBGClient(
            settings.sbg_api_base, settings.sbg_chat_path,
            settings.sbg_api_key, settings.sbg_model_id, settings.sbg_timeout,
        )
        return FallbackLLMClient(sbg, base_client)

    log.info("LLM provider=%s model=%s", provider, settings.llm_model)
    return base_client
