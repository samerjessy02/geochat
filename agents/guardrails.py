"""
agents/guardrails.py — layered input / retrieval / output protections.

Three independent layers, each a pure function (no I/O, no model calls) so they
are fast and unit-testable:

  Input      check_input(query)        prompt injection, jailbreak, SQL injection,
                                        prompt-leak probes, malicious URLs, length.
  Retrieval  scan_document_text(text)  strip embedded instructions from ingested
             redact_secrets(text)      docs; redact secrets/keys/passwords/env vars
                                        so they can never be retrieved or surfaced.
  Output     validate_output(...)      groundedness, citation presence, system-prompt
                                        leakage, sensitive-data leakage in answers.

Nothing here is a substitute for the SQL AST validator (``validator.py``) or the
allow-listed dataset tables — these are defense-in-depth on the *natural
language* boundary, which the SQL layer never sees.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import settings
from agents.logging_config import get_logger

log = get_logger("guardrails")

# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #

# Prompt-injection / jailbreak phrasings (case-insensitive).
_INJECTION_PATTERNS = [
    r"ignore (?:all|any|the)?\s*(?:previous|prior|above|earlier)\s+instructions",
    r"disregard (?:all|any|the)?\s*(?:previous|prior|above)\s+(?:instructions|prompts?)",
    r"forget (?:everything|all previous|your instructions)",
    r"you are now (?:a|an|in)\b",
    r"act as (?:if you are|a|an)\b",
    r"\bdeveloper mode\b",
    r"\bdo anything now\b|\bDAN\b",
    r"\bjailbreak\b",
    r"pretend (?:to be|you are)\b",
    r"reveal (?:your|the)\s+(?:system|hidden|initial)\s+prompt",
    r"(?:print|show|repeat|output)\s+(?:your|the)\s+(?:system|initial|hidden)\s+prompt",
    r"what (?:are|is) your (?:system )?instructions",
    r"override (?:your|the) (?:safety|guardrails|rules)",
    r"bypass (?:the|your) (?:safety|filter|guardrail)",
]

# SQL-injection style probes appearing inside natural-language input.
_SQL_INJECTION_PATTERNS = [
    r";\s*(?:drop|delete|truncate|update|insert|alter|create|grant)\b",
    r"\bunion\s+select\b",
    r"--\s*$",
    r"/\*.*\*/",
    r"\bor\s+1\s*=\s*1\b",
    r"'\s*or\s*'",
    r"\bxp_cmdshell\b",
]

# Secrets / credentials to redact from ingested or retrieved text.
_SECRET_PATTERNS = [
    (re.compile(r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*\S+"), "[REDACTED_KEY]"),
    (re.compile(r"(?i)\b(?:password|passwd|pwd)\b\s*[:=]\s*\S+"), "[REDACTED_PASSWORD]"),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}"), "[REDACTED_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[REDACTED_TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GOOGLE_KEY]"),
    (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"(?mi)^\s*[A-Z][A-Z0-9_]{3,}\s*=\s*\S+"), "[REDACTED_ENV_VAR]"),
    (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"), "[REDACTED_PRIVATE_KEY]"),
]

# Suspicious URL indicators (raw IPs, punycode, credential-in-URL, shorteners).
_MALICIOUS_URL_PATTERNS = [
    r"https?://\d{1,3}(?:\.\d{1,3}){3}",          # raw IP host
    r"https?://[^/\s]*@",                          # user:pass@host
    r"https?://xn--",                              # punycode
    r"https?://[^\s]*\.(?:zip|mov|exe|scr|bat)\b", # risky TLD/extension
]

# Phrases that indicate the model is leaking its own configuration.
_SYSTEM_LEAK_PATTERNS = [
    r"you are a postgresql \+ postgis expert",
    r"system prompt",
    r"my (?:system )?instructions are",
    r"as an ai (?:language )?model,? i (?:was|am) (?:instructed|configured)",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_SQLI_RE = [re.compile(p, re.IGNORECASE) for p in _SQL_INJECTION_PATTERNS]
_URL_RE = [re.compile(p, re.IGNORECASE) for p in _MALICIOUS_URL_PATTERNS]
_LEAK_RE = [re.compile(p, re.IGNORECASE) for p in _SYSTEM_LEAK_PATTERNS]


# --------------------------------------------------------------------------- #
# Input layer
# --------------------------------------------------------------------------- #

@dataclass
class GuardResult:
    """Outcome of an input guardrail check."""

    allowed: bool
    reasons: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"allowed": self.allowed, "reasons": self.reasons, "categories": self.categories}


def check_input(query: str) -> GuardResult:
    """Screen a user query before it reaches the router/LLM.

    Blocks prompt injection, jailbreak attempts, SQL-injection probes, malicious
    URLs and over-length input. Returns a :class:`GuardResult`; callers should
    reject when ``allowed`` is False.
    """
    if not settings.guardrails_enabled:
        return GuardResult(allowed=True)

    reasons: list[str] = []
    categories: list[str] = []
    text = query or ""

    if len(text) > settings.max_query_chars:
        reasons.append(f"Query exceeds maximum length of {settings.max_query_chars} characters.")
        categories.append("length")

    if any(r.search(text) for r in _INJECTION_RE):
        reasons.append("Query appears to contain a prompt-injection or jailbreak attempt.")
        categories.append("prompt_injection")

    if any(r.search(text) for r in _SQLI_RE):
        reasons.append("Query contains SQL-injection-like patterns.")
        categories.append("sql_injection")

    if any(r.search(text) for r in _URL_RE):
        reasons.append("Query contains a suspicious or potentially malicious URL.")
        categories.append("malicious_url")

    if reasons:
        log.warning("input guardrail BLOCKED query — categories=%s", categories)
    return GuardResult(allowed=not reasons, reasons=reasons, categories=categories)


# --------------------------------------------------------------------------- #
# Retrieval layer
# --------------------------------------------------------------------------- #

def redact_secrets(text: str) -> str:
    """Redact API keys, passwords, tokens, env vars and private keys from text."""
    if not text:
        return text
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def scan_document_text(text: str) -> tuple[str, list[str]]:
    """Neutralize an ingested document's text (retrieval guardrail).

    (1) Redacts secrets so credentials embedded in a file can never be indexed
        and later retrieved. (2) Detects and neutralizes injected instructions
        so a document cannot hijack the model at answer time — matched
        instruction spans are wrapped as inert quoted text.

    Returns ``(cleaned_text, flags)`` where ``flags`` names what was found.
    """
    flags: list[str] = []
    cleaned = redact_secrets(text)
    if cleaned != text:
        flags.append("secret_redacted")

    for regex in _INJECTION_RE:
        if regex.search(cleaned):
            flags.append("embedded_instruction")
            cleaned = regex.sub(lambda m: f"[quoted document text: {m.group(0)}]", cleaned)
    return cleaned, sorted(set(flags))


def filter_retrieved_chunks(hits: list[dict]) -> list[dict]:
    """Redact secrets from retrieved chunk text before it reaches the LLM."""
    for hit in hits:
        if "text" in hit:
            hit["text"] = redact_secrets(hit["text"])
    return hits


# --------------------------------------------------------------------------- #
# Output layer
# --------------------------------------------------------------------------- #

@dataclass
class OutputValidation:
    """Outcome of validating a generated answer."""

    valid: bool
    reasons: list[str] = field(default_factory=list)
    leaked_system_prompt: bool = False
    leaked_secret: bool = False
    has_citations: bool = False

    def as_dict(self) -> dict:
        return {
            "valid": self.valid,
            "reasons": self.reasons,
            "leaked_system_prompt": self.leaked_system_prompt,
            "leaked_secret": self.leaked_secret,
            "has_citations": self.has_citations,
        }


def validate_output(
    answer: str,
    *,
    sources: list[str] | None = None,
    require_citations: bool = True,
    found: bool = True,
) -> OutputValidation:
    """Validate a generated answer before returning it to the user.

    Checks for system-prompt leakage, leaked secrets, and (when the answer
    claims to have found information) the presence of at least one citation.
    Callers should reject or regenerate when ``valid`` is False.
    """
    reasons: list[str] = []
    text = answer or ""

    leaked_prompt = any(r.search(text) for r in _LEAK_RE)
    if leaked_prompt:
        reasons.append("Answer appears to leak the system prompt or internal instructions.")

    redacted = redact_secrets(text)
    leaked_secret = redacted != text
    if leaked_secret:
        reasons.append("Answer contains what looks like a secret or credential.")

    has_citations = bool(sources)
    if require_citations and found and not has_citations:
        reasons.append("Answer claims information but provides no citations.")

    if reasons:
        log.warning("output guardrail flagged answer: %s", reasons)
    return OutputValidation(
        valid=not reasons,
        reasons=reasons,
        leaked_system_prompt=leaked_prompt,
        leaked_secret=leaked_secret,
        has_citations=has_citations,
    )
