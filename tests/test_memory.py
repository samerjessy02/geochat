"""
Standalone tests for the LangChain-backed conversational memory
(agents/memory.py) — no pytest required.

Run from the repo root:

    python tests/test_memory.py

Memory is backed by LangChain's ConversationSummaryBufferMemory (recent buffer +
running summary). These tests exercise the public facade with a FAKE app LLM (no
network). Summarization-specific checks run only when LangChain is installed
(otherwise memory falls back to a plain recent-window buffer, and those checks are
reported as SKIP). Exits non-zero if anything fails.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.llm_client as llm_client
import agents.memory as memory
from agents.memory import HybridConversationMemory, estimate_tokens

_passed = _failed = _skipped = 0


def check(label, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"  PASS  {label}")
    else:
        _failed += 1; print(f"  FAIL  {label}")


def skip(label):
    global _skipped
    _skipped += 1; print(f"  SKIP  {label} (LangChain not installed)")


def section(t):
    print(f"\n== {t} ==")


# ---- fake app LLM (used both by the summary adapter and by condense) ---------
class FakeLLM:
    def __init__(self):
        self.calls = []
        self.reply = None
        self.fail = False

    def complete(self, messages, *, temperature=None, max_tokens=None):
        self.calls.append([m["content"] for m in messages])
        if self.fail:
            raise RuntimeError("simulated LLM outage")
        return self.reply if self.reply is not None else "MERGED SUMMARY: goals; facts; decisions."


_fake = FakeLLM()
llm_client.get_llm = lambda: _fake


def new_fake():
    _fake.calls.clear(); _fake.reply = None; _fake.fail = False
    return _fake


def mem(**kw):
    d = dict(recent_k=2, trigger_turns=12, trigger_tokens=1500,
             context_budget=120, summary_max_tokens=64, summary_enabled=True)
    d.update(kw)
    return HybridConversationMemory(**d)


LC = memory._LANGCHAIN_OK
BIG = "word " * 40


def test_estimate_tokens():
    section("estimate_tokens")
    check("empty -> 0", estimate_tokens("") == 0)
    check("None-safe -> 0", estimate_tokens(None) == 0)
    check("word floor", estimate_tokens(" ".join(["x"] * 40)) >= 40)


def test_langchain_backed():
    section("uses LangChain when available")
    if LC:
        check("summary memory active", mem().summary_enabled is True)
    else:
        skip("LangChain-backed summary memory")


def test_add_and_reads():
    section("add + as_text / messages / has_context")
    new_fake()
    m = mem()
    check("empty has no context", not m.has_context)
    m.add("show me cairo university", "(mapped 1 feature)")
    m.add("show me al azhar university", "(mapped 1 feature)")
    txt = m.as_text()
    check("as_text has the user turns", "cairo university" in txt and "al azhar" in txt)
    check("has_context true", m.has_context)
    check("len counts turns", len(m) >= 1)
    msgs = m.messages()
    check("messages() non-empty", len(msgs) > 0)


def test_summarization_on_overflow():
    section("older turns are summarized when the buffer overflows the budget")
    if not LC:
        skip("summarization"); skip("summary in as_text"); return
    new_fake()
    _fake.reply = "MERGED SUMMARY (single, consolidated)"
    m = mem(context_budget=120, summary_max_tokens=64)
    for i in range(6):                       # push well past the token budget
        m.add(f"{BIG} q{i}", f"{BIG} a{i}")
    check("a running summary was produced", bool(m.summary))
    check("summary surfaced in as_text", "SUMMARY" in m.as_text().upper())
    check("summary LLM adapter was invoked", len(_fake.calls) >= 1)


def test_budget_prune_no_crash():
    section("ensure_within_budget prunes before a call")
    new_fake()
    m = mem(context_budget=120)
    for i in range(5):
        m.add(f"{BIG} q{i}", f"{BIG} a{i}")
    m.ensure_within_budget(extra_tokens=500)   # must not raise
    check("still has recent context after prune", m.has_context)
    if LC:
        check("buffer kept under budget-ish", estimate_tokens(m.as_text()) <= 2000)
    else:
        skip("prune tightening (window fallback)")


def test_clear():
    section("clear resets memory")
    new_fake()
    m = mem()
    m.add("hi", "hello"); m.add("more", "ok")
    m.clear()
    check("no context after clear", not m.has_context)
    check("summary cleared", m.summary == "")


def test_summary_prompt_fields():
    section("structured summary prompt preserves required fields")
    tmpl = memory._SUMMARY_INSTRUCTIONS
    for h in ["Goals & objectives", "Preferences & constraints", "Key facts & entities",
              "Decisions made", "Open questions", "Reasoning & conclusions",
              "Commitments, plans & instructions"]:
        check(f"prompt mentions '{h}'", h in tmpl)
    check("single/non-nested instruction", "SINGLE" in tmpl.upper() and "nested" in tmpl.lower())
    check("factual / no-invent instruction", "factual" in tmpl.lower() and "invent" in tmpl.lower())
    check("has {summary} and {new_lines} slots", "{summary}" in tmpl and "{new_lines}" in tmpl)


def test_condense_query():
    section("condense_query uses history + respects settings")
    from config import settings

    def set_s(name, val):
        object.__setattr__(settings, name, val)

    orig_en, orig_cond = settings.memory_enabled, settings.memory_condense
    try:
        set_s("memory_enabled", False)
        check("disabled -> unchanged", memory.condense_query(mem(), "does it deliver?") == "does it deliver?")
        set_s("memory_enabled", True); set_s("memory_condense", True)

        f = new_fake()
        check("no history -> unchanged", memory.condense_query(mem(), "hello") == "hello")
        check("no LLM call w/ empty history", len(f.calls) == 0)

        f = new_fake(); f.reply = "Does Cilantro deliver?"
        m = mem(); m.add("Tell me about Cilantro", "Cilantro is a cafe.")
        check("returns the LLM rewrite", memory.condense_query(m, "does it deliver?") == "Does Cilantro deliver?")

        f = new_fake(); f.fail = True
        m2 = mem(); m2.add("Tell me about Cilantro", "Cilantro is a cafe.")
        check("LLM failure -> original", memory.condense_query(m2, "does it deliver?") == "does it deliver?")
    finally:
        set_s("memory_enabled", orig_en); set_s("memory_condense", orig_cond)


def test_registry():
    section("get_memory registry + reset_memory")
    a1 = memory.get_memory("A"); a2 = memory.get_memory("A"); b1 = memory.get_memory("B")
    check("same session -> same instance", a1 is a2)
    check("different session -> different instance", a1 is not b1)
    a1.add("hi", "hello")
    memory.reset_memory("A")
    check("reset -> fresh empty memory", not memory.get_memory("A").has_context)


def main():
    for t in [test_estimate_tokens, test_langchain_backed, test_add_and_reads,
              test_summarization_on_overflow, test_budget_prune_no_crash, test_clear,
              test_summary_prompt_fields, test_condense_query, test_registry]:
        t()
    print(f"\n{'='*52}\nRESULT: {_passed} passed, {_failed} failed, {_skipped} skipped"
          f"  (LangChain {'present' if LC else 'ABSENT — fallback window'})\n{'='*52}")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
