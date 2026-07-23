"""
agents/geojson_table.py — flatten GeoJSON properties into analyzable rows.

The Analytics agent treats each feature's ``properties`` as one tabular row and
its ``geometry`` as spatial metadata. Two entry points:

* :func:`geojsonToTable` — flatten a raw GeoJSON FeatureCollection (in memory)
  into a list of row dicts, optionally adding derived geometry stats
  (``_geometry_type`` and a representative ``_lon`` / ``_lat``).
* :func:`fetch_dataset_tables` — pull the already-ingested rows for the selected
  datasets straight from PostGIS (properties columns + the geometry centroid), so
  analytics runs on the same data the map uses. Each row is tagged with
  ``_dataset`` (the layer's display name) so several layers can be compared.
"""

from __future__ import annotations

from db import run_query
from registry import get_datasets_by_ids
from agents.logging_config import get_logger

log = get_logger("geojson_table")

_GEOM_COL = "wkb_geometry"


def _iter_coords(coords):
    """Yield (lon, lat) pairs from arbitrarily-nested GeoJSON coordinates."""
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)):
            yield coords[0], coords[1]
        else:
            for c in coords:
                yield from _iter_coords(c)


def _representative_point(geom: dict):
    """A single (lon, lat) for a geometry — the point itself, or the mean of its
    vertices for lines/polygons (a cheap centroid, no GEOS needed)."""
    if not isinstance(geom, dict):
        return None, None
    gtype = geom.get("type")
    coords = geom.get("coordinates")
    try:
        if gtype == "Point" and coords:
            return float(coords[0]), float(coords[1])
        pts = list(_iter_coords(coords))
        if pts:
            return (sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts))
    except Exception:  # noqa: BLE001
        pass
    return None, None


def geojsonToTable(feature_collection: dict, *, include_geometry_stats: bool = True) -> list[dict]:
    """Flatten a GeoJSON FeatureCollection's ``properties`` into row objects.

    Each returned dict is a feature's properties, plus (when
    ``include_geometry_stats``) ``_geometry_type`` and a representative ``_lon`` /
    ``_lat``. A plain ``Feature`` or a bare list of features is also accepted.
    """
    if isinstance(feature_collection, dict):
        feats = feature_collection.get("features")
        if feats is None and feature_collection.get("type") == "Feature":
            feats = [feature_collection]
    elif isinstance(feature_collection, list):
        feats = feature_collection
    else:
        feats = None
    feats = feats or []

    rows: list[dict] = []
    for f in feats:
        if not isinstance(f, dict):
            continue
        props = dict(f.get("properties") or {})
        if include_geometry_stats:
            geom = f.get("geometry") or {}
            props["_geometry_type"] = geom.get("type")
            lon, lat = _representative_point(geom)
            if lon is not None:
                props["_lon"], props["_lat"] = lon, lat
        rows.append(props)
    return rows


def fetch_dataset_tables(dataset_ids: list[str]) -> list[dict]:
    """Return the ingested rows for each selected dataset, from PostGIS.

    Result: ``[{id, display_name, geometry_type, columns, rows}, ...]`` where each
    row is ``{<property columns>, _lon, _lat, _dataset}``. ``table_name`` is
    machine-generated so it is safe to interpolate.
    """
    out: list[dict] = []
    for d in get_datasets_by_ids(dataset_ids):
        cols = [c["column_name"] for c in (d.get("columns") or [])
                if c["column_name"] != _GEOM_COL]
        sel = ", ".join(f'"{c}"' for c in cols) if cols else "*"
        table = d["table_name"]
        try:
            rows = run_query(
                f'SELECT {sel}, '
                f'ST_X(ST_Centroid({_GEOM_COL})) AS _lon, '
                f'ST_Y(ST_Centroid({_GEOM_COL})) AS _lat '
                f'FROM "{table}"'
            )
        except Exception as e:  # noqa: BLE001
            log.warning("analytics: row fetch failed for '%s': %s", table, e)
            rows = []
        dn = d["display_name"]
        for r in rows:
            r["_dataset"] = dn
        out.append({
            "id": str(d["id"]),
            "display_name": dn,
            "geometry_type": d.get("geometry_type"),
            "columns": cols,
            "rows": rows,
        })
    return out
