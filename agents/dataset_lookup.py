"""
agents/dataset_lookup.py — answer knowledge questions from structured dataset columns.

For place-specific questions ("what are Bean House's opening hours?", "does Costa
have wifi?"), the answer is usually already a column value in the uploaded
GeoJSON/PostGIS feature — no documents or web search needed. This module finds
the feature(s) whose name matches the queried entity, formats their columns as a
plain-text record for grounded generation, and surfaces the feature's own
``website`` value so later tiers can scrape the official site if the structured
data doesn't answer the question.

Column names come from ``information_schema`` (not the description registry), so
lookup works even when the user skipped the describe step. The entity value is
always passed as a bound parameter; only the machine-generated table name and
sanitized column names are interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import text

from db import engine
import registry
from agents.logging_config import get_logger

log = get_logger("dataset_lookup")

_GEOM = "wkb_geometry"
# Column-name fragments that indicate a name/label field to match the entity on.
_NAME_HINTS = ("name", "title", "label", "brand", "operator", "display_name")
_WEBSITE_HINTS = ("website", "url", "homepage", "site")


@dataclass
class DatasetMatch:
    """Structured-data context assembled from matching features."""

    context: str
    website: str | None = None
    sources: list[str] = field(default_factory=list)
    num_records: int = 0


def _table_columns(table_name: str) -> list[str]:
    """Return the real column names of ``table_name`` via information_schema."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
                {"t": table_name},
            )
            return [r[0] for r in rows]
    except Exception as e:  # noqa: BLE001
        log.warning("could not read columns for '%s': %s", table_name, e)
        return []


def _format_record(display_name: str, row: dict) -> tuple[str, str | None]:
    """Render a feature row as 'col: value' text and extract its website, if any."""
    parts: list[str] = []
    website: str | None = None
    for key, value in row.items():
        if key == _GEOM or value is None:
            continue
        sval = str(value).strip()
        if not sval or sval.lower() == "nan":
            continue
        if website is None and any(h in key.lower() for h in _WEBSITE_HINTS) and sval.startswith("http"):
            website = sval
        parts.append(f"{key}: {sval}")
    if not parts:
        return "", website
    return f"[{display_name}] " + " | ".join(parts), website


def lookup(entity: str, dataset_ids: list[str], *, limit: int = 5) -> DatasetMatch | None:
    """Find features matching ``entity`` across the selected datasets.

    Returns a :class:`DatasetMatch` (record context + website) or ``None`` when
    nothing matches / no entity was given.
    """
    if not entity or not dataset_ids:
        return None

    datasets = registry.get_datasets_by_ids(dataset_ids)
    records: list[str] = []
    website: str | None = None
    sources: list[str] = []

    for ds in datasets:
        table = ds["table_name"]
        cols = _table_columns(table)
        name_cols = [c for c in cols if c != _GEOM and any(h in c.lower() for h in _NAME_HINTS)]
        if not name_cols:
            continue

        select_cols = [c for c in cols if c != _GEOM] or ["*"]
        col_list = ", ".join(f'"{c}"' for c in select_cols)
        # Bidirectional match: the column contains the entity ("Cilantro" in
        # "Cilantro Downtown"), OR the entity/question contains the feature name
        # ("Cilantro" inside "opening hours of cilantro cafe in egypt"). The
        # length>=3 guard on the reverse direction avoids matching tiny/generic
        # feature names against any query.
        conds = [
            f'("{c}"::text ILIKE :q OR (length("{c}"::text) >= 3 AND :ent ILIKE \'%\' || "{c}"::text || \'%\'))'
            for c in name_cols
        ]
        where = " OR ".join(conds)
        sql = f'SELECT {col_list} FROM "{table}" WHERE {where} LIMIT :lim'
        try:
            with engine.connect() as conn:
                rows = [
                    dict(r._mapping)
                    for r in conn.execute(text(sql), {"q": f"%{entity}%", "ent": entity, "lim": limit})
                ]
        except Exception as e:  # noqa: BLE001
            log.warning("dataset lookup on '%s' failed: %s", table, e)
            continue

        for row in rows:
            record, w = _format_record(ds["display_name"], row)
            if record:
                records.append(record)
                sources.append(ds["display_name"])
                if website is None and w:
                    website = w

    if not records:
        log.info("dataset lookup: no feature matched '%s'", entity)
        return None

    log.info("dataset lookup matched %d record(s) for '%s'%s",
             len(records), entity, f" (website: {website})" if website else "")
    return DatasetMatch(
        context="\n\n".join(records),
        website=website,
        sources=sorted(set(sources)),
        num_records=len(records),
    )
