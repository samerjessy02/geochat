"""
agents/memory.py — adaptive hybrid conversational memory (LangChain-backed).

Uses LangChain's **ConversationSummaryBufferMemory**: keep the most recent
messages verbatim in a token-bounded buffer, and as the buffer overflows the
token limit, fold the oldest messages into ONE running summary (LangChain
re-summarizes the existing summary together with the newly-pruned messages, so it
stays a single updated summary — never nested). The summary prompt is customized
(:data:`_SUMMARY_PROMPT`) to preserve goals, preferences/constraints, key facts &
entities, decisions, open questions, reasoning/conclusions, and any commitments/
plans/instructions.

The LangChain memory is driven by the app's own LLM (not an OpenAI key) via a
small :class:`_AppLLM` adapter, and token counting uses a lightweight estimator so
no tokenizer/transformers dependency is required.

Everything is wrapped by :class:`HybridConversationMemory`, which preserves the
public surface the rest of the app relies on (``add``, ``as_text``, ``messages``,
``has_context``, ``summary``, ``ensure_within_budget``, ``clear``), plus a
per-session registry (:func:`get_memory`) and the follow-up
:func:`condense_query`.

If LangChain isn't installed, a minimal built-in recent-window buffer is used as a
fallback (summarization disabled) so the app keeps working.
"""

from __future__ import annotations

import threading
from collections import deque

from config import settings
from agents.logging_config import get_logger, snippet

log = get_logger("memory")


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token, word-count floor)."""
    if not text:
        return 0
    return max(len(text) // 4, len(text.split()))


# --------------------------------------------------------------------------- #
# LangChain wiring: an LLM adapter + a customized structured-summary prompt
# --------------------------------------------------------------------------- #

_LANGCHAIN_OK = True
try:
    from langchain.memory import (
        ConversationSummaryBufferMemory,
        ConversationBufferWindowMemory as _LCWindowMemory,
    )
    from langchain_core.language_models.llms import LLM
    from langchain_core.prompts import PromptTemplate
except Exception as _e:  # noqa: BLE001 — degrade gracefully if LangChain is absent
    _LANGCHAIN_OK = False
    log.warning("LangChain not available (%s) — memory summarization disabled, using a plain window", _e)


# Structured progressive-summary prompt (LangChain calls this with the current
# ``summary`` and the ``new_lines`` being pruned, and expects the updated summary).
_SUMMARY_INSTRUCTIONS = (
    "Progressively update the running MEMORY SUMMARY of a conversation between a user and an "
    "assistant in a geospatial data app. Merge the new lines into the current summary and return "
    "a SINGLE consolidated summary that supersedes the old one — never nested or multiple summaries.\n"
    "Be concise and strictly factual: include only what was actually said; do not invent. Never omit "
    "anything that could affect future answers; on conflict prefer the most recent statement.\n"
    "Organize under these headings (omit a heading only when it has no content):\n"
    "Goals & objectives; Preferences & constraints; Key facts & entities; Decisions made; "
    "Open questions / unresolved tasks; Reasoning & conclusions; Commitments, plans & instructions.\n\n"
    "Current summary:\n{summary}\n\nNew lines of conversation:\n{new_lines}\n\nUpdated summary:"
)

if _LANGCHAIN_OK:
    _SUMMARY_PROMPT = PromptTemplate(
        input_variables=["summary", "new_lines"], template=_SUMMARY_INSTRUCTIONS
    )

    class _AppLLM(LLM):
        """Adapter so LangChain memory can summarize via the app's own LLM client.

        Only text completion is needed (LangChain formats the summary prompt to a
        string). Token counting is overridden with the local estimator so no
        tokenizer/transformers dependency is pulled in.
        """

        max_tokens: int = 512

        @property
        def _llm_type(self) -> str:
            return "geochat-app-llm"

        def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:  # noqa: ANN001
            from agents.llm_client import get_llm  # lazy: avoid import cycle
            try:
                return get_llm().complete(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                ) or ""
            except Exception as e:  # noqa: BLE001 — memory must never break a request
                log.warning("memory: summary LLM call failed (%s)", e)
                return ""

        def get_num_tokens(self, text: str) -> int:
            return estimate_tokens(text)


# --------------------------------------------------------------------------- #
# minimal fallback used only when LangChain is unavailable
# --------------------------------------------------------------------------- #
class _WindowFallback:
    """Recent-window buffer with the same surface (no summarization)."""

    def __init__(self, k: int) -> None:
        self._turns: deque[tuple[str, str]] = deque(maxlen=max(1, int(k)))

    def add(self, user: str, assistant: str) -> None:
        self._turns.append((user, assistant))

    @property
    def summary(self) -> str:
        return ""

    def as_text(self) -> str:
        lines = []
        for u, a in self._turns:
            if u:
                lines.append(f"Human: {u}")
            if a:
                lines.append(f"AI: {a}")
        return "\n".join(lines)

    def messages(self) -> list[dict]:
        out = []
        for u, a in self._turns:
            if u:
                out.append({"role": "user", "content": u})
            if a:
                out.append({"role": "assistant", "content": a})
        return out

    def clear(self) -> None:
        self._turns.clear()

    def prune(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self._turns)


# --------------------------------------------------------------------------- #
# public facade
# --------------------------------------------------------------------------- #
class HybridConversationMemory:
    """Recent buffer + running summary, backed by LangChain's
    ``ConversationSummaryBufferMemory`` (or a plain window if LangChain is absent).

    The token budget (``context_budget``) is LangChain's ``max_token_limit`` — the
    buffer is kept under it and the overflow is summarized. ``ensure_within_budget``
    additionally prunes before an LLM call, accounting for the incoming query.
    """

    def __init__(self, *, recent_k: int = 5, trigger_turns: int = 12,
                 trigger_tokens: int = 1500, context_budget: int = 2000,
                 summary_max_tokens: int = 512, summary_enabled: bool = True) -> None:
        self.recent_k = max(1, int(recent_k))
        self.context_budget = max(1, int(context_budget))
        self.summary_enabled = bool(summary_enabled) and _LANGCHAIN_OK
        self._lock = threading.RLock()

        if self.summary_enabled:
            self._mem = ConversationSummaryBufferMemory(
                llm=_AppLLM(max_tokens=int(summary_max_tokens)),
                max_token_limit=self.context_budget,
                prompt=_SUMMARY_PROMPT,
                memory_key="history",
                return_messages=False,
            )
        elif _LANGCHAIN_OK:
            self._mem = _LCWindowMemory(k=self.recent_k, memory_key="history", return_messages=False)
        else:
            self._mem = _WindowFallback(self.recent_k)

    # -- writes ------------------------------------------------------------- #
    def add(self, user: str, assistant: str) -> None:
        u, a = (user or "").strip(), (assistant or "").strip()
        if not u and not a:
            return
        with self._lock:
            try:
                if isinstance(self._mem, _WindowFallback):
                    self._mem.add(u, a)
                else:
                    # LangChain prunes to max_token_limit here, summarizing overflow.
                    self._mem.save_context({"input": u}, {"output": a})
            except Exception as e:  # noqa: BLE001 — never break the request on memory
                log.warning("memory: add failed (%s)", e)

    # -- token budgeting ---------------------------------------------------- #
    def accumulated_tokens(self) -> int:
        return estimate_tokens(self.as_text())

    def ensure_within_budget(self, extra_tokens: int = 0) -> None:
        """Prune/summarize before an LLM call so history (+ the incoming query) fits."""
        if not self.summary_enabled:
            return
        with self._lock:
            try:
                extra = max(0, int(extra_tokens))
                if extra:
                    orig = self._mem.max_token_limit
                    self._mem.max_token_limit = max(1, orig - extra)
                    try:
                        self._mem.prune()
                    finally:
                        self._mem.max_token_limit = orig
                else:
                    self._mem.prune()
            except Exception as e:  # noqa: BLE001
                log.warning("memory: prune failed (%s)", e)

    # -- reads -------------------------------------------------------------- #
    @property
    def summary(self) -> str:
        return getattr(self._mem, "moving_summary_buffer", "") or ""

    def as_text(self) -> str:
        with self._lock:
            try:
                if isinstance(self._mem, _WindowFallback):
                    return self._mem.as_text()
                return self._mem.load_memory_variables({}).get("history", "") or ""
            except Exception as e:  # noqa: BLE001
                log.warning("memory: as_text failed (%s)", e)
                return self.summary

    @property
    def has_context(self) -> bool:
        return bool(self.as_text().strip())

    def messages(self) -> list[dict]:
        if isinstance(self._mem, _WindowFallback):
            return self._mem.messages()
        out: list[dict] = []
        if self.summary:
            out.append({"role": "system", "content": "Summary of earlier conversation:\n" + self.summary})
        try:
            for m in self._mem.chat_memory.messages:
                role = {"human": "user", "ai": "assistant"}.get(getattr(m, "type", ""), "system")
                out.append({"role": role, "content": m.content})
        except Exception:  # noqa: BLE001
            pass
        return out

    def clear(self) -> None:
        with self._lock:
            try:
                self._mem.clear()
                # Some LangChain versions don't reset the summary on clear().
                if hasattr(self._mem, "moving_summary_buffer"):
                    self._mem.moving_summary_buffer = ""
            except Exception:  # noqa: BLE001
                pass

    def __len__(self) -> int:
        try:
            if isinstance(self._mem, _WindowFallback):
                return len(self._mem)
            return len(self._mem.chat_memory.messages) // 2
        except Exception:  # noqa: BLE001
            return 0


# --------------------------------------------------------------------------- #
# per-session registry
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_sessions: dict[str, HybridConversationMemory] = {}


def _norm(session_id: str | None) -> str:
    return (session_id or "default").strip() or "default"


def get_memory(session_id: str | None) -> HybridConversationMemory:
    """Return the (lazily created) hybrid memory for ``session_id``."""
    sid = _norm(session_id)
    with _lock:
        mem = _sessions.get(sid)
        if mem is None:
            mem = HybridConversationMemory(
                recent_k=settings.memory_recent_k,
                trigger_turns=settings.memory_summary_trigger_turns,
                trigger_tokens=settings.memory_summary_trigger_tokens,
                context_budget=settings.memory_context_budget_tokens,
                summary_max_tokens=settings.memory_summary_max_tokens,
                summary_enabled=settings.memory_summary_enabled,
            )
            _sessions[sid] = mem
        return mem


def reset_memory(session_id: str | None) -> None:
    """Forget a session's history (start a fresh conversation)."""
    sid = _norm(session_id)
    with _lock:
        _sessions.pop(sid, None)
    log.info("memory reset for session '%s'", sid)


# --------------------------------------------------------------------------- #
# follow-up condensing
# --------------------------------------------------------------------------- #

_CONDENSE_SYSTEM = (
    "You rewrite a user's follow-up question into a fully standalone question "
    "using the conversation history. Resolve every pronoun or reference "
    "(it, that, that one, they, there, this place) to the specific entity it "
    "refers to in the history. Preserve the original intent and wording as much "
    "as possible — only add what is needed to make it self-contained. If the "
    "question is ALREADY standalone, or the history is irrelevant to it, return "
    "it UNCHANGED. Return ONLY the rewritten question text, with no preamble, "
    "quotes, or explanation."
)


def condense_query(memory: HybridConversationMemory, query: str) -> str:
    """Rewrite ``query`` into a standalone question using ``memory``.

    Returns ``query`` unchanged when memory/condensing is disabled, there is no
    history, or the rewrite looks unreliable. Any LLM failure falls back to the
    original query. Before the call, memory is pruned to the token budget.
    """
    if not settings.memory_enabled or not settings.memory_condense:
        return query
    if not getattr(memory, "has_context", False):
        return query

    # Dynamically monitor token usage: make room for this query before the call.
    try:
        memory.ensure_within_budget(extra_tokens=estimate_tokens(query))
    except Exception:  # noqa: BLE001
        log.warning("memory: budget enforcement failed — proceeding with current history")

    from agents.llm_client import get_llm, LLMError  # lazy: avoid import cycle

    prompt = (
        f"Conversation history:\n{memory.as_text()}\n\n"
        f"Follow-up question: {query}\n\n"
        "Standalone question:"
    )
    try:
        out = get_llm().complete(
            [
                {"role": "system", "content": _CONDENSE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=160,
        )
    except (LLMError, Exception):  # noqa: BLE001 — never fail the request on condensing
        log.warning("condense failed — using the original query")
        return query

    rewritten = (out or "").strip().strip('"').strip()
    original = (query or "").strip()
    if not rewritten or len(rewritten) > 4 * len(original) + 200:
        return original
    if rewritten.lower() != original.lower():
        log.info("condensed follow-up: %s -> %s", snippet(original), snippet(rewritten))
    return rewritten
