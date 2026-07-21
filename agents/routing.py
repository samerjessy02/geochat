"""
agents/routing.py — walking/driving routes via a self-hosted Valhalla engine.

Given two place names (or coordinates), this:
  1. geocodes each place against the user's own datasets (name ILIKE match),
  2. asks Valhalla for a route, and
  3. returns a GeoJSON LineString plus distance / duration / mode.

Valhalla is expected at ``VALHALLA_URL`` (default http://localhost:8002 — the
port the gis-ops docker image serves on). Its ``/route`` returns each leg's
geometry as an encoded polyline with precision 6, which ``_decode_shape`` turns
into ``[lon, lat]`` coordinates.

Nothing here talks to the LLM; parsing is regex-based and geocoding is a
parameterized PostGIS query, so routing is deterministic and safe.
"""

from __future__ import annotations

import re

import httpx
from sqlalchemy import text

from config import settings
from db import engine
from registry import get_datasets_by_ids, list_datasets
from agents.logging_config import get_logger

log = get_logger("routing")

# user phrasing -> Valhalla costing model
_COSTING = {"pedestrian": "pedestrian", "bicycle": "bicycle", "auto": "auto"}


class RoutingError(Exception):
    """Raised when a route can't be produced (engine down, no path, …)."""


# --------------------------------------------------------------------------- #
# request parsing
# --------------------------------------------------------------------------- #

def _clean_place(s: str) -> str:
    s = re.sub(r"\s+(by car|by bike|by bicycle|on foot|by foot|walking|driving|cycling)\s*$", "", s, flags=re.I)
    return s.strip().strip("\"'?.! ")


def parse_route_request(query: str) -> tuple[str | None, str | None, str]:
    """Extract ``(from_place, to_place, mode)`` from a routing question.

    mode is a Valhalla costing: ``pedestrian`` | ``bicycle`` | ``auto`` (default).
    Returns ``(None, None, mode)`` when start/end can't be found.
    """
    q = query or ""
    if re.search(r"\b(walk|walking|on foot|pedestrian)\b", q, re.I):
        mode = "pedestrian"
    elif re.search(r"\b(bike|bicycle|cycl)\w*\b", q, re.I):
        mode = "bicycle"
    else:
        mode = "auto"

    m = re.search(r"\bfrom\s+(.+?)\s+to\s+(.+?)\s*[\?\.!]*$", q, re.I)
    if not m:
        m = re.search(r"\bbetween\s+(.+?)\s+and\s+(.+?)\s*[\?\.!]*$", q, re.I)
    if m:
        return _clean_place(m.group(1)), _clean_place(m.group(2)), mode
    return None, None, mode


# --------------------------------------------------------------------------- #
# geocoding against the user's datasets
# --------------------------------------------------------------------------- #

def geocode_place(name: str, dataset_ids: list[str] | None) -> dict | None:
    """Resolve a place name to ``{name, lat, lon}`` using the selected datasets.

    Searches each dataset's name-like columns with ILIKE and returns the centroid
    of the first match (works for point, line and polygon features).
    """
    if not name:
        return None
    datasets = get_datasets_by_ids(dataset_ids) if dataset_ids else list_datasets()
    like = f"%{name.strip()}%"
    for d in datasets:
        table = d["table_name"]
        cols = [c["column_name"] for c in (d.get("columns") or [])]
        name_cols = [c for c in cols if c == "name" or "name" in c.lower()
                     or c in ("title", "label", "display_name")]
        if not name_cols:
            continue
        conds = " OR ".join(f'"{c}" ILIKE :q' for c in name_cols)
        label = name_cols[0]
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(f'SELECT ST_X(ST_Centroid(wkb_geometry)) AS lon, '
                         f'ST_Y(ST_Centroid(wkb_geometry)) AS lat, "{label}" AS nm '
                         f'FROM "{table}" WHERE {conds} '
                         f'ORDER BY LENGTH("{label}") ASC LIMIT 1'),
                    {"q": like},
                ).mappings().first()
        except Exception as e:  # noqa: BLE001 — a table without that column etc.
            log.debug("geocode probe failed on %s: %s", table, e)
            continue
        if row and row["lat"] is not None:
            return {"name": row["nm"] or name, "lat": float(row["lat"]), "lon": float(row["lon"])}
    return None


# --------------------------------------------------------------------------- #
# Valhalla
# --------------------------------------------------------------------------- #

def is_available() -> bool:
    """Quick reachability check against Valhalla's /status."""
    try:
        r = httpx.get(f"{settings.valhalla_url}/status", timeout=settings.valhalla_timeout)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _decode_shape(encoded: str, precision: int = 6) -> list[list[float]]:
    """Decode a Valhalla encoded polyline into ``[[lon, lat], ...]``."""
    inv = 10 ** -precision
    coords: list[list[float]] = []
    lat = lon = i = 0
    length = len(encoded)
    while i < length:
        for is_lat in (True, False):
            shift = result = 0
            while True:
                b = ord(encoded[i]) - 63
                i += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lat:
                lat += delta
            else:
                lon += delta
        coords.append([lon * inv, lat * inv])   # GeoJSON order: [lon, lat]
    return coords


def route(points: list[tuple[float, float]], mode: str = "auto") -> dict:
    """Route through ``points`` (list of ``(lat, lon)``) with Valhalla.

    Returns ``{geometry (GeoJSON LineString), distance_m, duration_s, mode}``.
    """
    costing = _COSTING.get(mode, "auto")
    payload = {
        "locations": [{"lat": lat, "lon": lon} for lat, lon in points],
        "costing": costing,
        "directions_options": {"units": "kilometers"},
    }
    try:
        r = httpx.post(f"{settings.valhalla_url}/route", json=payload, timeout=settings.valhalla_timeout)
    except Exception as e:  # noqa: BLE001
        raise RoutingError(f"could not reach the routing engine at {settings.valhalla_url} ({e})") from e
    if r.status_code != 200:
        detail = ""
        try:
            detail = r.json().get("error", "")
        except Exception:  # noqa: BLE001
            detail = r.text[:200]
        raise RoutingError(f"routing engine returned {r.status_code}: {detail}")

    trip = (r.json() or {}).get("trip") or {}
    legs = trip.get("legs") or []
    if not legs:
        raise RoutingError("no route found between those points.")

    coords: list[list[float]] = []
    for leg in legs:
        shape = leg.get("shape")
        if shape:
            pts = _decode_shape(shape)
            if coords and pts and coords[-1] == pts[0]:
                pts = pts[1:]
            coords.extend(pts)

    summary = trip.get("summary") or {}
    length_km = float(summary.get("length") or 0.0)
    time_s = float(summary.get("time") or 0.0)
    return {
        "geometry": {"type": "LineString", "coordinates": coords},
        "distance_m": round(length_km * 1000.0, 1),
        "duration_s": round(time_s, 1),
        "mode": mode,
    }
