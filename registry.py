"""
registry.py — metadata registry for user-uploaded geospatial datasets.

Two tables, created once at startup:
  datasets(id, table_name, display_name, geometry_type, srid, created_at)
  dataset_columns(id, dataset_id, column_name, data_type, description)

table_name is always machine-generated (user_data_<uuid hex>), never taken
directly from user input, so it is always safe to interpolate into DDL/SQL —
this is the thing that keeps dynamic tables from becoming a SQL-injection
vector.
"""

import uuid
from sqlalchemy import text
from db import engine

DDL = """
CREATE TABLE IF NOT EXISTS datasets (
    id UUID PRIMARY KEY,
    table_name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    geometry_type TEXT NOT NULL,
    srid INTEGER NOT NULL DEFAULT 4326,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dataset_columns (
    id SERIAL PRIMARY KEY,
    dataset_id UUID NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    column_name TEXT NOT NULL,
    data_type TEXT NOT NULL,
    description TEXT DEFAULT '',
    is_primary_key BOOLEAN NOT NULL DEFAULT FALSE,
    foreign_key TEXT,
    UNIQUE(dataset_id, column_name)
);
"""

# Migration for databases created before the key columns existed.
_MIGRATIONS = [
    "ALTER TABLE dataset_columns ADD COLUMN IF NOT EXISTS is_primary_key BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE dataset_columns ADD COLUMN IF NOT EXISTS foreign_key TEXT",
]


def init_registry():
    with engine.begin() as conn:
        conn.execute(text(DDL))
        for stmt in _MIGRATIONS:
            try:
                conn.execute(text(stmt))
            except Exception:
                pass


def ensure_spatial_indexes() -> int:
    """Backfill a GiST index on wkb_geometry for every registered table missing
    one (tables ingested before spatial indexing was added). Idempotent; returns
    the number of tables processed. Safe to call at startup."""
    processed = 0
    with engine.begin() as conn:
        rows = conn.execute(text("SELECT table_name FROM datasets")).fetchall()
        for (table_name,) in rows:
            # table_name is machine-generated (new_table_name) -> safe to interpolate.
            try:
                conn.execute(text(
                    f'CREATE INDEX IF NOT EXISTS "{table_name}_geom_gist" '
                    f'ON "{table_name}" USING GIST (wkb_geometry)'
                ))
                processed += 1
            except Exception:
                # e.g. table dropped out-of-band or lacks wkb_geometry; skip it.
                pass
    return processed


def new_table_name() -> str:
    return f"user_data_{uuid.uuid4().hex[:12]}"


def register_dataset(table_name: str, display_name: str, geometry_type: str, srid: int = 4326) -> str:
    dataset_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO datasets (id, table_name, display_name, geometry_type, srid)
                     VALUES (:id, :table_name, :display_name, :geometry_type, :srid)"""),
            {"id": dataset_id, "table_name": table_name, "display_name": display_name,
             "geometry_type": geometry_type, "srid": srid},
        )
    return dataset_id


def set_column_descriptions(dataset_id: str, columns: list[dict]):
    """columns: [{column_name, data_type, description, is_primary_key?, foreign_key?}, ...]"""
    with engine.begin() as conn:
        for col in columns:
            params = {
                "dataset_id": dataset_id,
                "column_name": col["column_name"],
                "data_type": col.get("data_type", ""),
                "description": col.get("description", ""),
                "is_primary_key": bool(col.get("is_primary_key", False)),
                "foreign_key": (col.get("foreign_key") or None),
            }
            conn.execute(
                text("""INSERT INTO dataset_columns
                          (dataset_id, column_name, data_type, description, is_primary_key, foreign_key)
                        VALUES (:dataset_id, :column_name, :data_type, :description, :is_primary_key, :foreign_key)
                        ON CONFLICT (dataset_id, column_name)
                        DO UPDATE SET description = EXCLUDED.description,
                                      data_type = EXCLUDED.data_type,
                                      is_primary_key = EXCLUDED.is_primary_key,
                                      foreign_key = EXCLUDED.foreign_key"""),
                params,
            )


def list_datasets() -> list[dict]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT d.id, d.table_name, d.display_name, d.geometry_type, d.srid, d.created_at,
                   COALESCE(json_agg(json_build_object(
                       'column_name', c.column_name,
                       'data_type', c.data_type,
                       'description', c.description,
                       'is_primary_key', c.is_primary_key,
                       'foreign_key', c.foreign_key
                   )) FILTER (WHERE c.id IS NOT NULL), '[]') AS columns
            FROM datasets d
            LEFT JOIN dataset_columns c ON c.dataset_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """))
        return [dict(r._mapping) for r in rows]


def get_datasets_by_ids(dataset_ids: list[str]) -> list[dict]:
    if not dataset_ids:
        return []
    all_ds = list_datasets()
    wanted = set(dataset_ids)
    return [d for d in all_ds if str(d["id"]) in wanted]


def get_table_names(dataset_ids: list[str]) -> set[str]:
    """Whitelist of real table names for the given dataset ids — used by the validator
    to make sure any table the LLM references was actually registered, not invented
    or smuggled in."""
    return {d["table_name"] for d in get_datasets_by_ids(dataset_ids)}


def delete_dataset(dataset_id: str):
    with engine.begin() as conn:
        row = conn.execute(text("SELECT table_name FROM datasets WHERE id = :id"), {"id": dataset_id}).fetchone()
        if not row:
            return False
        table_name = row[0]
        # table_name is machine-generated (see new_table_name), so safe to interpolate
        conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
        conn.execute(text("DELETE FROM datasets WHERE id = :id"), {"id": dataset_id})
    return True
