"""
agents/column_describer.py — LLM-generated column descriptions for datasets.

When a user uploads a GeoJSON/CSV, the columns are known but undocumented. This
module drafts a concise, human-readable description for each column using the
modular LLM, grounded in a few real sample values pulled from the freshly
ingested table (so the descriptions reflect the actual data, not just the column
name). The user reviews and edits these before they are saved via
``/datasets/describe`` — nothing is persisted here.
"""

from __future__ import annotations

from db import run_query
import registry
from agents.llm_client import get_llm, LLMError
from agents.logging_config import get_logger

log = get_logger("column_describer")

_SYSTEM = (
    "You document geospatial dataset columns. For each column, write ONE short, plain-English "
    "sentence describing what it contains, using the column name, its data type, and the example "
    "values as evidence. Be specific and factual; do not invent meaning that the examples don't "
    "support. Keep each description under ~15 words.\n"
    'Return ONLY JSON: {"descriptions": {"<column_name>": "<description>", ...}} covering every column.'
)


def _sample_values(table_name: str, columns: list[str], limit: int = 5) -> dict[str, list[str]]:
    """Fetch up to ``limit`` rows and collect non-empty example values per column.

    ``table_name`` is machine-generated (``user_data_<hex>``) so it is safe to
    interpolate; on any query error we simply return no samples.
    """
    try:
        rows = run_query(f'SELECT * FROM "{table_name}" LIMIT {int(limit)}')
    except Exception as e:  # noqa: BLE001
        log.warning("could not sample table '%s': %s", table_name, e)
        return {c: [] for c in columns}

    samples: dict[str, list[str]] = {c: [] for c in columns}
    for row in rows:
        for c in columns:
            val = row.get(c)
            if val is None:
                continue
            text = str(val).strip()
            if text and c != "wkb_geometry" and text not in samples[c]:
                samples[c].append(text[:60])
    return {c: v[:3] for c, v in samples.items()}


def _sample_from_features(features: list[dict], columns: list[str], limit: int = 20) -> dict[str, list[str]]:
    """Collect example values per column from in-memory GeoJSON features (used by
    the validation flow, before any table exists)."""
    samples: dict[str, list[str]] = {c: [] for c in columns}
    for f in features[:limit]:
        props = f.get("properties") or {}
        for c in columns:
            val = props.get(c)
            if val is None:
                continue
            text = str(val).strip()
            if text and text not in samples[c]:
                samples[c].append(text[:60])
    return {c: v[:3] for c, v in samples.items()}


def _fallback_desc(name: str, dtype: str, samples: list[str]) -> str:
    """Deterministic description for a column the LLM left blank / failed on, so
    EVERY column ends up with a non-empty description."""
    label = name.replace("_", " ").replace(":", " ").strip() or name
    if samples:
        return f"{label} (e.g. {', '.join(samples[:2])})"
    return f"{label} — {dtype} value for each feature"


def _generate_from_samples(display_name: str, geometry_type: str,
                           columns: list[dict], samples: dict[str, list[str]]) -> list[dict]:
    """Core LLM call shared by the DB-backed and feature-backed generators.

    Guarantees a non-empty description for every column: the LLM draft is used
    when present, otherwise a deterministic fallback built from the name, type
    and example values.
    """
    lines = []
    for c in columns:
        ex = samples.get(c["column_name"], [])
        ex_str = ", ".join(ex) if ex else "n/a"
        lines.append(f'- {c["column_name"]} (type: {c["data_type"]}; examples: {ex_str})')
    user_prompt = (
        f'Dataset: "{display_name}" (geometry: {geometry_type}).\n'
        f"Columns:\n" + "\n".join(lines)
    )
    try:
        out = get_llm().complete_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user_prompt}]
        )
        descs = out.get("descriptions", {}) or {}
    except (LLMError, Exception) as e:  # noqa: BLE001
        log.warning("auto-description generation failed: %s", e)
        descs = {}
    result = []
    for c in columns:
        name = c["column_name"]
        desc = str(descs.get(name, "")).strip()
        if not desc:
            desc = _fallback_desc(name, c["data_type"], samples.get(name, []))
        result.append({"column_name": name, "data_type": c["data_type"], "description": desc})
    return result


def describe_from_features(display_name: str, features: list[dict], columns: list[dict]) -> list[dict]:
    """Draft descriptions from in-memory features (validation flow — no table yet)."""
    col_names = [c["column_name"] for c in columns]
    samples = _sample_from_features(features, col_names)
    log.info("auto-generating descriptions for %d column(s) from %d feature(s)", len(columns), len(features))
    return _generate_from_samples(display_name, "?", columns, samples)


def generate_descriptions(dataset_id: str, columns: list[dict]) -> list[dict]:
    """Draft a description for each column of ``dataset_id``.

    Args:
        dataset_id: the registered dataset id.
        columns: ``[{"column_name": str, "data_type": str}, ...]``.

    Returns:
        The same columns with a generated ``description`` filled in (empty string
        if generation failed for that column). Raises ``ValueError`` if the
        dataset is unknown.
    """
    datasets = registry.get_datasets_by_ids([dataset_id])
    if not datasets:
        raise ValueError("Dataset not found.")
    ds = datasets[0]
    col_names = [c["column_name"] for c in columns]
    samples = _sample_values(ds["table_name"], col_names)

    lines = []
    for c in columns:
        ex = samples.get(c["column_name"], [])
        ex_str = ", ".join(ex) if ex else "n/a"
        lines.append(f'- {c["column_name"]} (type: {c["data_type"]}; examples: {ex_str})')
    user_prompt = (
        f'Dataset: "{ds["display_name"]}" (geometry: {ds.get("geometry_type", "?")}).\n'
        f"Columns:\n" + "\n".join(lines)
    )

    log.info("auto-generating descriptions for %d column(s) of '%s'", len(columns), ds["display_name"])
    try:
        out = get_llm().complete_json(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user_prompt}]
        )
        descs = out.get("descriptions", {}) or {}
    except (LLMError, Exception) as e:  # noqa: BLE001
        log.warning("auto-description generation failed: %s", e)
        descs = {}

    return [
        {
            "column_name": c["column_name"],
            "data_type": c["data_type"],
            "description": str(descs.get(c["column_name"], "")).strip(),
        }
        for c in columns
    ]
