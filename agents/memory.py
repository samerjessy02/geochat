"""
agents/memory.py — adaptive hybrid conversational memory.

Rather than retaining only the last *N* messages, this keeps context in two tiers
so long conversations stay useful without overflowing the model's context window:

* **Recent memory** — the newest ``recent_k`` interactions (a user turn + the
  assistant reply) are kept verbatim.
* **Summarized memory** — once the conversation grows past a turn count *X* OR a
  token budget *Y*, the older turns are folded into ONE running, structured
  summary. Further growth re-summarizes *the existing summary together with the
  newly-aged-out turns*, always producing a single updated summary (never nested
  summaries). The summary preserves goals, preferences/constraints, key facts and
  entities, decisions, open questions, reasoning/conclusions, and any
  commitments/plans/instructions.

Token usage is monitored before each LLM call that consumes the history
(:meth:`HybridConversationMemory.ensure_within_budget`): if the summary + recent
turns would exceed the configured budget, summarization is triggered first (and,
only as a last resort if the summarizer is unavailable, the oldest turns are
dropped) so the request always fits.

Exposed:

* :class:`HybridConversationMemory` — the two-tier buffer, plus a per-session
  registry (:func:`get_memory`) keyed by an opaque ``session_id``.
* :func:`condense_query` — rewrite a follow-up ("does it deliver?", "show that
  one on the map") into a standalone question using the history (summary + recent
  turns), so retrieval and intent classification get a self-contained query.

Dependency-free: the classic LangChain memory classes are deprecated, so this is
a small implementation wired to the app's own LLM client.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

from config import settings
from agents.logging_config import get_logger, snippet

log = get_logger("memory")


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate.

    Uses ~4 characters/token (a good English heuristic) with a word-count floor,
    so short-but-wordy text is never under-counted. Good enough to drive budget
    decisions without pulling in a tokenizer.
    """
    if not text:
        return 0
    return max(len(text) // 4, len(text.split()))


@dataclass
class Turn:
    """One interaction: the user's message and the assistant's reply."""

    user: str
    assistant: str

    def tokens(self) -> int:
        return estimate_tokens(self.user) + estimate_tokens(self.assistant)


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
# structured summarizer — folds old turns (+ any prior summary) into ONE summary
# --------------------------------------------------------------------------- #

_SUMMARY_SYSTEM = (
    "You maintain a single running MEMORY SUMMARY of a conversation between a user "
    "and an assistant in a geospatial data application. You are given the PRIOR "
    "SUMMARY (may be empty) and the OLDER MESSAGES that are about to age out of the "
    "recent window. Merge them into ONE updated summary. Do NOT create nested or "
    "multiple summaries — return a single consolidated summary that supersedes the "
    "prior one.\n\n"
    "Rules:\n"
    "- Be concise, factual, and strictly grounded in what was actually said. Never "
    "invent, guess, or add information not present in the messages.\n"
    "- Never omit anything that could affect future responses. When the prior "
    "summary and new messages conflict, prefer the most recent statement.\n"
    "- Use these headings; omit a heading only when it has no content:\n"
    "  Goals & objectives\n  Preferences & constraints\n  Key facts & entities\n"
    "  Decisions made\n  Open questions / unresolved tasks\n"
    "  Reasoning & conclusions\n  Commitments, plans & instructions\n"
    "- Use short bullet points under each heading. No preamble or closing remarks."
)


def _summarize(prior_summary: str, old_turns: list[Turn], max_tokens: int) -> str | None:
    """Produce one consolidated summary from ``prior_summary`` + ``old_turns``.

    Returns the new summary text, or ``None`` on any LLM failure (so the caller can
    keep the raw turns instead of losing information).
    """
    from agents.llm_client import get_llm, LLMError  # lazy: avoid import cycle

    transcript = "\n".join(
        line for t in old_turns for line in
        ((f"User: {t.user}",) if t.user else ()) + ((f"Assistant: {t.assistant}",) if t.assistant else ())
    )
    user_prompt = (
        f"PRIOR SUMMARY:\n{prior_summary or '(none)'}\n\n"
        f"OLDER MESSAGES (oldest first):\n{transcript}\n\n"
        "Return the single updated MEMORY SUMMARY:"
    )
    try:
        out = get_llm().complete(
            [{"role": "system", "content": _SUMMARY_SYSTEM},
             {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    except (LLMError, Exception):  # noqa: BLE001 — never crash a request on summarization
        log.warning("memory: summarization failed — keeping raw turns for now")
        return None
    text = (out or "").strip()
    return text or None


# --------------------------------------------------------------------------- #
# adaptive hybrid memory: recent window + one running structured summary
# --------------------------------------------------------------------------- #

class HybridConversationMemory:
    """Two-tier memory: the newest ``recent_k`` turns verbatim, older turns folded
    into a single running structured summary.

    Summarization triggers when the full turns exceed ``trigger_turns`` (X) OR the
    summary + turns exceed ``trigger_tokens`` (Y). :meth:`ensure_within_budget`
    additionally enforces a hard token budget before an LLM call.
    """

    def __init__(self, *, recent_k: int = 5, trigger_turns: int = 12,
                 trigger_tokens: int = 1500, context_budget: int = 2000,
                 summary_max_tokens: int = 512, summary_enabled: bool = True) -> None:
        self.recent_k = max(1, int(recent_k))
        self.trigger_turns = max(self.recent_k, int(trigger_turns))
        self.trigger_tokens = max(1, int(trigger_tokens))
        self.context_budget = max(1, int(context_budget))
        self.summary_max_tokens = max(64, int(summary_max_tokens))
        self.summary_enabled = bool(summary_enabled)
        self.summary: str = ""            # single running structured summary
        self._turns: list[Turn] = []      # recent turns kept in full
        self._lock = threading.RLock()

    # -- writes ------------------------------------------------------------- #
    def add(self, user: str, assistant: str) -> None:
        """Record one interaction, then summarize/trim if thresholds are crossed."""
        u, a = (user or "").strip(), (assistant or "").strip()
        if not u and not a:
            return
        with self._lock:
            self._turns.append(Turn(user=u, assistant=a))
            self._maybe_summarize()

    # -- token accounting --------------------------------------------------- #
    def _turns_tokens(self) -> int:
        return sum(t.tokens() for t in self._turns)

    def accumulated_tokens(self) -> int:
        """Estimated tokens of everything memory would contribute (summary + turns)."""
        with self._lock:
            return estimate_tokens(self.summary) + self._turns_tokens()

    def _over_threshold(self) -> bool:
        return (len(self._turns) > self.trigger_turns
                or estimate_tokens(self.summary) + self._turns_tokens() > self.trigger_tokens)

    # -- summarization ------------------------------------------------------ #
    def _maybe_summarize(self) -> None:
        """Trigger summarization of older turns when a threshold is crossed."""
        if not self.summary_enabled:
            # Degrade to a pure recent window when summarization is off.
            if len(self._turns) > self.trigger_turns:
                self._turns = self._turns[-self.trigger_turns:]
            return
        if self._over_threshold():
            self._summarize_older(keep=self.recent_k)

    def _summarize_older(self, keep: int) -> bool:
        """Fold every turn except the last ``keep`` into the running summary.

        Returns True if the summary was updated (turns were compacted). On LLM
        failure the raw turns are retained (no data loss) and False is returned.
        """
        keep = max(0, keep)
        if len(self._turns) <= keep:
            return False
        old = self._turns[:len(self._turns) - keep]
        recent = self._turns[len(self._turns) - keep:]
        new_summary = _summarize(self.summary, old, self.summary_max_tokens)
        if new_summary is None:
            return False
        self.summary = new_summary
        self._turns = recent
        log.info("memory: summarized %d old turn(s) -> summary now ~%d tokens, %d recent turn(s) kept",
                 len(old), estimate_tokens(self.summary), len(self._turns))
        return True

    def ensure_within_budget(self, extra_tokens: int = 0) -> None:
        """Guarantee summary + recent turns (+ ``extra_tokens``) fit the budget.

        Called before an LLM call that consumes the history. Prefers summarization;
        falls back to dropping the oldest turns only if the summarizer can't run, so
        the request never exceeds the budget.
        """
        with self._lock:
            if self.accumulated_tokens() + extra_tokens <= self.context_budget:
                return
            # 1) Summarize progressively down to fewer recent turns.
            if self.summary_enabled:
                keep = self.recent_k
                while (self.accumulated_tokens() + extra_tokens > self.context_budget
                       and len(self._turns) > 1):
                    keep = min(keep, len(self._turns) - 1)
                    if not self._summarize_older(keep=keep):
                        break                      # summarizer unavailable / no-op
                    keep = max(1, keep - 1)         # be more aggressive next pass
            # 2) Hard fallback: drop oldest full turns to force a fit.
            while self.accumulated_tokens() + extra_tokens > self.context_budget and len(self._turns) > 1:
                dropped = self._turns.pop(0)
                log.warning("memory: budget still exceeded — dropped oldest turn (%s)",
                            snippet(dropped.user or dropped.assistant))

    # -- reads -------------------------------------------------------------- #
    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    @property
    def has_context(self) -> bool:
        return bool(self.summary) or bool(self._turns)

    def messages(self) -> list[dict]:
        """History as chat messages: the summary (as a system note) then recent turns."""
        msgs: list[dict] = []
        if self.summary:
            msgs.append({"role": "system",
                         "content": "Summary of earlier conversation:\n" + self.summary})
        for t in self._turns:
            if t.user:
                msgs.append({"role": "user", "content": t.user})
            if t.assistant:
                msgs.append({"role": "assistant", "content": t.assistant})
        return msgs

    def as_text(self) -> str:
        """History as a plain transcript: summary first, then recent turns."""
        lines: list[str] = []
        if self.summary:
            lines.append("[Summary of earlier conversation]\n" + self.summary + "\n[Recent messages]")
        for t in self._turns:
            if t.user:
                lines.append(f"User: {t.user}")
            if t.assistant:
                lines.append(f"Assistant: {t.assistant}")
        return "\n".join(lines)

    def clear(self) -> None:
        with self._lock:
            self._turns.clear()
            self.summary = ""

    def __len__(self) -> int:
        return len(self._turns)


# --------------------------------------------------------------------------- #
# per-session registry
# --------------------------------------------------------------------------- #

_lock = threading.Lock()
_sessions: dict[str, HybridConversationMemory] = {}


def _norm(session_id: str | None) -> str:
    return (session_id or "default").strip() or "default"


def get_memory(session_id: str | None) -> HybridConversationMemory:
    """Return the (lazily created) hybrid memory for ``session_id``.

    A single default session is used when the client doesn't supply one, which
    is the right behaviour for the local single-user app: consecutive queries
    share one memory so follow-ups work out of the box.
    """
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
    history, or the rewrite looks unreliable (empty or implausibly long). Any
    LLM failure falls back to the original query — condensing must never break a
    request.

    Before building the prompt, memory is trimmed to the configured token budget
    (summarizing the older turns first) so the history + incoming query fit.
    """
    if not settings.memory_enabled or not settings.memory_condense:
        return query
    if not getattr(memory, "has_context", len(memory) > 0):
        return query

    # Dynamically monitor token usage: make room for this query before the call.
    try:
        memory.ensure_within_budget(extra_tokens=estimate_tokens(query))
    except Exception:  # noqa: BLE001 — budgeting must never break a request
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
    # Reject junk: empty, or a runaway that ballooned far beyond the question.
    if not rewritten or len(rewritten) > 4 * len(original) + 200:
        return original
    if rewritten.lower() != original.lower():
        log.info("condensed follow-up: %s -> %s", snippet(original), snippet(rewritten))
    return rewritten
