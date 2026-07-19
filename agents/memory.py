"""
agents/memory.py — short-term conversational memory (windowed).

Implements **ConversationBufferWindowMemory**: keep only the last ``k``
interactions, where one interaction is a user turn plus the assistant's reply.
Older interactions are dropped, so the memory is bounded by *turn count* and
never grows without limit. That bound matters here because every RAG answer
already fills the model's context window with retrieved chunks (dense + BM25,
sometimes a whole document) — conversation history has to stay small so it
doesn't crowd out that context.

Two things are exposed:

* :class:`ConversationBufferWindowMemory` — the windowed buffer itself, plus a
  per-session registry (:func:`get_memory`) keyed by an opaque ``session_id``.
* :func:`condense_query` — rewrite a follow-up ("does it deliver?", "show that
  one on the map") into a standalone question using the window, so retrieval and
  intent classification receive a self-contained query with pronouns resolved.

The classic LangChain ``ConversationBufferWindowMemory`` class is deprecated, so
this is a small, dependency-free implementation of the same idea (last-``k``
trimming) wired to the app's own LLM client.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from config import settings
from agents.logging_config import get_logger, snippet

log = get_logger("memory")


@dataclass
class Turn:
    """One interaction: the user's message and the assistant's reply."""

    user: str
    assistant: str


class ConversationBufferWindowMemory:
    """Keep the last ``k`` interactions, dropping anything older.

    Backed by a ``deque(maxlen=k)`` so appending the (k+1)-th interaction evicts
    the oldest automatically — O(1) and inherently bounded.
    """

    def __init__(self, k: int = 5) -> None:
        self.k = max(1, int(k))
        self._turns: deque[Turn] = deque(maxlen=self.k)

    def add(self, user: str, assistant: str) -> None:
        """Record one interaction (no-op if both sides are empty)."""
        u, a = (user or "").strip(), (assistant or "").strip()
        if not u and not a:
            return
        self._turns.append(Turn(user=u, assistant=a))

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def messages(self) -> list[dict]:
        """History as chat messages (oldest first), for prompting."""
        msgs: list[dict] = []
        for t in self._turns:
            if t.user:
                msgs.append({"role": "user", "content": t.user})
            if t.assistant:
                msgs.append({"role": "assistant", "content": t.assistant})
        return msgs

    def as_text(self) -> str:
        """History as a plain transcript (oldest first)."""
        lines: list[str] = []
        for t in self._turns:
            if t.user:
                lines.append(f"User: {t.user}")
            if t.assistant:
                lines.append(f"Assistant: {t.assistant}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)


# --------------------------------------------------------------------------- #
# per-session registry
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_sessions: dict[str, ConversationBufferWindowMemory] = {}


def _norm(session_id: str | None) -> str:
    return (session_id or "default").strip() or "default"


def get_memory(session_id: str | None) -> ConversationBufferWindowMemory:
    """Return the (lazily created) window memory for ``session_id``.

    A single default session is used when the client doesn't supply one, which
    is the right behaviour for the local single-user app: consecutive queries
    share one window so follow-ups work out of the box.
    """
    sid = _norm(session_id)
    with _lock:
        mem = _sessions.get(sid)
        if mem is None:
            mem = ConversationBufferWindowMemory(k=settings.memory_window_k)
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


def condense_query(memory: ConversationBufferWindowMemory, query: str) -> str:
    """Rewrite ``query`` into a standalone question using ``memory``.

    Returns ``query`` unchanged when memory/condensing is disabled, there is no
    history, or the rewrite looks unreliable (empty or implausibly long). Any
    LLM failure falls back to the original query — condensing must never break a
    request.
    """
    if not settings.memory_enabled or not settings.memory_condense:
        return query
    if len(memory) == 0:
        return query

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
    # Reject junk: empty, or a runaway that ballooned far beyond the question.
    if not rewritten or len(rewritten) > 4 * len(original) + 200:
        return original
    if rewritten.lower() != original.lower():
        log.info("condensed follow-up: %s -> %s", snippet(original), snippet(rewritten))
    return rewritten
