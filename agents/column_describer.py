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
