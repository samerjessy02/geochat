"""
agents/spatial_search.py — search features inside a user-drawn polygon.

Given a GeoJSON Polygon/MultiPolygon (drawn on the map) and a feature type
(schools, hospitals, …), find the features of that layer that fall inside the
area with a parameterized PostGIS query. Kept separate from the LLM NL->SQL path
so the geometry is never string-interpolated (the polygon is always passed as a
bound parameter to ``ST_GeomFromGeoJSON``) and the query is deterministic.

Validation (per requirements):
  * geometry type must be Polygon / MultiPolygon
  * coordinates must be present and in WGS84 (EPSG:4326) lon/lat range
  * empty geometries are rejected
  * self-intersecting / invalid rings are auto-repaired with ST_MakeValid
    (a ``repaired`` flag is returned) rather than failing the search

Performance:
  * relies on the GiST index on ``wkb_geometry`` (created at ingest) so the
    spatial predicate is index-assisted
  * selects only the layer's real attribute columns (never the raw geometry blob)
  * supports ``limit`` / ``offset`` pagination; ``count`` is the full match total
"""

from __future__ import annotations

import json

from sqlalchemy import text

from db import engine
from registry import get_datasets_by_ids, list_datasets
from agents.logging_config import get_logger

log = get_logger("spatial_search")

_ALLOWED_GEOM_TYPES = {"Polygon", "MultiPolygon"}
_PREDICATES = {"intersects": "ST_Intersects", "within": "ST_Within"}


class SpatialSearchError(Exception):
    """Raised for an invalid request (bad geometry, unknown feature type, …)."""


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #

def _iter_coords(coords):
    """Yield (lon, lat) pairs from arbitrarily-nested GeoJSON coordinate arrays."""
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)):
            yield coords[0], coords[1]
        else:
            for c in coords:
                yield from _iter_coords(c)


def _validate_geometry(geometry: dict) -> None:
    if not isinstance(geometry, dict):
        raise SpatialSearchError("geometry must be a GeoJSON object.")
    gtype = geometry.get("type")
    if gtype not in _ALLOWED_GEOM_TYPES:
        raise SpatialSearchError(
            f"geometry.type must be Polygon or MultiPolygon (got {gtype!r})."
        )
    coords = geometry.get("coordinates")
    if not coords:
        raise SpatialSearchError("geometry has no coordinates (empty area).")
    n = 0
    for lon, lat in _iter_coords(coords):
        n += 1
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise SpatialSearchError(
                "coordinates are outside WGS84 (EPSG:4326) lon/lat range — "
                "the drawn area must be in latitude/longitude."
            )
    if n < 3:
        raise SpatialSearchError("polygon needs at least 3 points.")


def _resolve_dataset(feature_type: str | None, dataset_id: str | None,
                     dataset_ids: list[str] | None) -> dict:
    """Map a requested feature type (or explicit id) to a registered dataset."""
    if dataset_id:
        match = [d for d in list_datasets() if str(d["id"]) == str(dataset_id)]
        if match:
            return match[0]

    pool = get_datasets_by_ids(dataset_ids) if dataset_ids else list_datasets()
    if feature_type:
        ft = feature_type.strip().lower()
        ft_sing = ft.rstrip("s")
        # 1) exact / plural-insensitive display-name match
        for d in pool:
            dn = (d.get("display_name") or "").strip().lower()
            if dn == ft or dn.rstrip("s") == ft_sing:
                return d
        # 2) substring either direction (e.g. "coffee shops" vs "cafes")
        for d in pool:
            dn = (d.get("display_name") or "").strip().lower()
            if dn and (ft_sing in dn or dn.rstrip("s") in ft):
                return d
    elif len(pool) == 1:
        return pool[0]

    avail = ", ".join(sorted({d.get("display_name", "") for d in pool})) or "none"
    raise SpatialSearchError(
        f"Could not match feature type {feature_type or '(unspecified)'!r} to a "
        f"selected dataset. Available layers: {avail}."
    )


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #

def search_by_polygon(
    geometry: dict,
    *,
    feature_type: str | None = None,
    dataset_id: str | None = None,
    dataset_ids: list[str] | None = None,
    mode: str = "intersects",
    limit: int = 2000,
    offset: int = 0,
) -> dict:
    """Return features of ``feature_type`` that fall inside ``geometry``.

    Result: ``{feature_type, table, predicate, repaired, count, bbox, features}``
    where ``features`` are row dicts (attributes + ``geometry`` GeoJSON string),
    matching the shape the frontend already renders.
    """
    _validate_geometry(geometry)
    predicate = _PREDICATES.get(mode, "ST_Intersects")
    dataset = _resolve_dataset(feature_type, dataset_id, dataset_ids)
    table = dataset["table_name"]  # machine name -> safe to interpolate
    cols = [c["column_name"] for c in (dataset.get("columns") or [])
            if c["column_name"] != "wkb_geometry"]
    select_cols = ", ".join(f't."{c}"' for c in cols) or "t.*"

    geom_json = json.dumps(geometry)
    limit = max(1, min(int(limit), 10000))
    offset = max(0, int(offset))

    poly_cte = "WITH poly AS (SELECT ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON(:g), 4326)) AS g)"

    try:
        with engine.connect() as conn:
            # Deep validation via PostGIS on the raw (pre-repair) geometry.
            chk = conn.execute(
                text("SELECT ST_IsValid(g0) AS valid, ST_IsEmpty(g0) AS empty FROM "
                     "(SELECT ST_SetSRID(ST_GeomFromGeoJSON(:g), 4326) AS g0) s"),
                {"g": geom_json},
            ).mappings().first()
            if chk is None:
                raise SpatialSearchError("could not parse the drawn geometry.")
            if chk["empty"]:
                raise SpatialSearchError("the drawn area is empty.")
            repaired = not chk["valid"]
            if repaired:
                log.info("search polygon was invalid (self-intersecting?) -> ST_MakeValid applied")

            count = conn.execute(
                text(f'{poly_cte} SELECT COUNT(*) FROM "{table}" t, poly '
                     f'WHERE {predicate}(t.wkb_geometry, poly.g)'),
                {"g": geom_json},
            ).scalar() or 0

            rows = conn.execute(
                text(f'{poly_cte} '
                     f'SELECT {select_cols}, ST_AsGeoJSON(t.wkb_geometry) AS geometry '
                     f'FROM "{table}" t, poly '
                     f'WHERE {predicate}(t.wkb_geometry, poly.g) '
                     f'LIMIT :lim OFFSET :off'),
                {"g": geom_json, "lim": limit, "off": offset},
            ).mappings().all()
    except SpatialSearchError:
        raise
    except Exception as e:  # noqa: BLE001 — normalize DB/parse errors
        log.warning("polygon search failed on '%s': %s", table, e)
        raise SpatialSearchError(f"spatial search failed: {e}") from e

    features = [dict(r) for r in rows]
    bbox = _bbox_from_features(features)
    log.info("polygon search: %s -> %d match(es) (showing %d), predicate=%s",
             dataset.get("display_name"), count, len(features), predicate)
    return {
        "feature_type": dataset.get("display_name"),
        "table": table,
        "predicate": predicate,
        "repaired": repaired,
        "count": int(count),
        "returned": len(features),
        "limit": limit,
        "offset": offset,
        "bbox": bbox,
        "features": features,
    }


def _bbox_from_features(features: list[dict]) -> list[float] | None:
    """[minLon, minLat, maxLon, maxLat] over the returned features' geometries."""
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    seen = False
    for f in features:
        try:
            geom = json.loads(f["geometry"])
        except (KeyError, TypeError, ValueError):
            continue
        for lon, lat in _iter_coords(geom.get("coordinates", [])):
            seen = True
            minx, miny = min(minx, lon), min(miny, lat)
            maxx, maxy = max(maxx, lon), max(maxy, lat)
    return [minx, miny, maxx, maxy] if seen else None
