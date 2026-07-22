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
import re

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
# manual polygon input: parse pasted coordinates into a validated GeoJSON polygon
# --------------------------------------------------------------------------- #

def _close_ring(ring: list) -> list:
    """Ensure a linear ring is closed (first point == last point)."""
    if len(ring) >= 3 and ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    return ring


def _coords_to_polygon(coords) -> dict:
    """Turn a raw coordinate array into a GeoJSON Polygon.

    Accepts a flat ring ``[[lon,lat], ...]`` or an already-nested polygon
    ``[[[lon,lat], ...]]``. Rings are auto-closed.
    """
    if not isinstance(coords, list) or not coords:
        raise SpatialSearchError("No coordinates found to build a polygon.")
    # nested: [[[lon,lat], ...], ...] (polygon with rings)
    if isinstance(coords[0], list) and coords[0] and isinstance(coords[0][0], list):
        rings = [_close_ring([list(p) for p in ring]) for ring in coords]
        return {"type": "Polygon", "coordinates": rings}
    # flat ring: [[lon,lat], ...]
    if isinstance(coords[0], list) and coords[0] and isinstance(coords[0][0], (int, float)):
        return {"type": "Polygon", "coordinates": [_close_ring([list(p) for p in coords])]}
    raise SpatialSearchError("Coordinates must be a list of [lon, lat] pairs.")


def _parse_wkt_via_postgis(wkt: str) -> dict:
    """Convert a WKT POLYGON/MULTIPOLYGON string to GeoJSON using PostGIS."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT ST_AsGeoJSON(ST_SetSRID(ST_GeomFromText(:w), 4326)) AS g"),
                {"w": wkt},
            ).mappings().first()
    except Exception as e:  # noqa: BLE001
        raise SpatialSearchError(f"Could not parse WKT: {e}") from e
    if not row or not row["g"]:
        raise SpatialSearchError("Could not parse the WKT geometry.")
    return json.loads(row["g"])


def _parse_text_pairs(raw: str) -> dict:
    """Parse newline/semicolon-separated ``lon, lat`` (or ``lon lat``) pairs."""
    pts = []
    for line in re.split(r"[;\n]+", raw.strip()):
        line = line.strip().strip("[]()")
        if not line:
            continue
        parts = re.split(r"[,\s]+", line)
        try:
            nums = [float(p) for p in parts if p != ""]
        except ValueError:
            raise SpatialSearchError(f"Could not read a coordinate pair from: {line!r}")
        if len(nums) < 2:
            raise SpatialSearchError(f"Each line needs a lon and a lat — got: {line!r}")
        pts.append([nums[0], nums[1]])
    if len(pts) < 3:
        raise SpatialSearchError("A polygon needs at least 3 coordinate pairs.")
    return _coords_to_polygon(pts)


def parse_polygon_text(raw: str) -> dict:
    """Parse pasted polygon input into a normalized, validated GeoJSON Polygon.

    Accepts three shapes:
      * a GeoJSON ``Polygon`` / ``MultiPolygon`` object (or a ``Feature`` /
        ``FeatureCollection`` wrapping one);
      * a raw coordinate array (``[[lon,lat], ...]`` or ``[[[lon,lat], ...]]``),
        as JSON;
      * WKT (``POLYGON((lon lat, ...))`` / ``MULTIPOLYGON(...)``);
      * or plain ``lon, lat`` pairs, one per line.

    The result is validated to be a non-empty polygon in WGS84 lon/lat range and
    repaired with ``ST_MakeValid`` if the ring self-intersects. Returns
    ``{geometry, bbox, repaired, num_points}``.
    """
    if not raw or not raw.strip():
        raise SpatialSearchError("Paste some polygon coordinates first.")
    s = raw.strip()

    geometry: dict | None = None
    # 1) JSON: GeoJSON object or a coordinate array
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        obj = None
    if obj is not None:
        if isinstance(obj, dict):
            t = obj.get("type")
            if t in ("Polygon", "MultiPolygon"):
                geometry = obj
            elif t == "Feature":
                geometry = obj.get("geometry")
            elif t == "FeatureCollection":
                feats = obj.get("features") or []
                for f in feats:
                    g = (f or {}).get("geometry") or {}
                    if g.get("type") in ("Polygon", "MultiPolygon"):
                        geometry = g
                        break
                if geometry is None:
                    raise SpatialSearchError("No Polygon feature found in the FeatureCollection.")
            elif "coordinates" in obj:
                geometry = _coords_to_polygon(obj["coordinates"])
            else:
                raise SpatialSearchError(
                    f"Unsupported GeoJSON type {t!r} — paste a Polygon or MultiPolygon.")
        elif isinstance(obj, list):
            geometry = _coords_to_polygon(obj)
    # 2) WKT
    if geometry is None and re.match(r"^\s*(MULTI)?POLYGON\s*\(", s, re.I):
        geometry = _parse_wkt_via_postgis(s)
    # 3) plain "lon, lat" lines
    if geometry is None:
        geometry = _parse_text_pairs(s)

    if not isinstance(geometry, dict) or geometry.get("type") not in _ALLOWED_GEOM_TYPES:
        raise SpatialSearchError("Could not read a polygon from the pasted text.")

    _validate_geometry(geometry)
    normalized, repaired = _normalize_geometry(geometry)
    num_points = sum(1 for _ in _iter_coords(normalized.get("coordinates", [])))
    return {
        "geometry": normalized,
        "bbox": _bbox_from_geometry(normalized),
        "repaired": repaired,
        "num_points": num_points,
    }


def _normalize_geometry(geometry: dict) -> tuple[dict, bool]:
    """Round-trip through PostGIS: validate/repair and return canonical GeoJSON."""
    geom_json = json.dumps(geometry)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT ST_IsValid(g0) AS valid, ST_IsEmpty(g0) AS empty, "
                     "ST_AsGeoJSON(ST_MakeValid(g0)) AS fixed "
                     "FROM (SELECT ST_SetSRID(ST_GeomFromGeoJSON(:g), 4326) AS g0) s"),
                {"g": geom_json},
            ).mappings().first()
    except Exception as e:  # noqa: BLE001
        raise SpatialSearchError(f"Could not parse the geometry: {e}") from e
    if row is None:
        raise SpatialSearchError("Could not parse the geometry.")
    if row["empty"]:
        raise SpatialSearchError("The polygon is empty.")
    repaired = not row["valid"]
    fixed = json.loads(row["fixed"]) if row["fixed"] else geometry
    # ST_MakeValid can turn a bad polygon into a collection/multipolygon; keep it
    # only if it's still an area type, else fall back to the original.
    if fixed.get("type") not in _ALLOWED_GEOM_TYPES:
        fixed = geometry
    return fixed, repaired


def _bbox_from_geometry(geometry: dict) -> list[float] | None:
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    seen = False
    for lon, lat in _iter_coords(geometry.get("coordinates", [])):
        seen = True
        minx, miny = min(minx, lon), min(miny, lat)
        maxx, maxy = max(maxx, lon), max(maxy, lat)
    return [minx, miny, maxx, maxy] if seen else None


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


def search_area(
    geometry: dict,
    *,
    dataset_ids: list[str],
    feature_types: list[str] | None = None,
    mode: str = "intersects",
    limit_per_layer: int = 2000,
) -> dict:
    """Search several layers inside one polygon — one entry per layer.

    ``feature_types`` optionally restricts to layers named in the query; when
    omitted, every dataset in ``dataset_ids`` is searched. Each layer is resolved
    to a distinct dataset (so "pharmacies and schools" returns BOTH, not one).
    Returns ``{layers: [...], bbox, total}``.
    """
    _validate_geometry(geometry)
    pool = get_datasets_by_ids(dataset_ids) if dataset_ids else list_datasets()
    if not pool:
        raise SpatialSearchError("Select at least one layer to search inside the area.")

    # Resolve which datasets to search.
    targets: list[dict] = []
    seen_ids: set = set()
    if feature_types:
        for ft in feature_types:
            try:
                ds = _resolve_dataset(ft, None, dataset_ids)
            except SpatialSearchError:
                continue
            if ds["id"] not in seen_ids:
                targets.append(ds)
                seen_ids.add(ds["id"])
        if not targets:
            avail = ", ".join(sorted({d.get("display_name", "") for d in pool})) or "none"
            raise SpatialSearchError(
                f"Could not match {feature_types!r} to a selected layer. Available: {avail}.")
    else:
        targets = pool

    layers = []
    all_features = []
    for ds in targets:
        try:
            res = search_by_polygon(
                geometry, dataset_id=str(ds["id"]),
                dataset_ids=dataset_ids, mode=mode, limit=limit_per_layer)
        except SpatialSearchError as e:
            layers.append({"dataset_id": str(ds["id"]),
                           "feature_type": ds.get("display_name"),
                           "count": 0, "features": [], "error": str(e)})
            continue
        layers.append({
            "dataset_id": str(ds["id"]),
            "feature_type": res["feature_type"],
            "count": res["count"],
            "returned": res["returned"],
            "repaired": res["repaired"],
            "bbox": res["bbox"],
            "features": res["features"],
        })
        all_features.extend(res["features"])

    return {
        "layers": layers,
        "total": sum(l["count"] for l in layers),
        "bbox": _bbox_from_features(all_features),
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
