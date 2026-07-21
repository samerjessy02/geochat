"""
agents/geojson_validator.py — GeoJSON ingest validation & cleaning pipeline.

Runs the staged validation contract used before a dataset is committed:

  Stage 1  FILE-LEVEL   (critical — reject whole file)
  Stage 2  GEOMETRY     (per-feature: skip / auto-fix / flag HITL)
  Stage 3  ATTRIBUTES   (per-feature, once required fields are chosen)
  Stage 4  DATASET      (non-blocking warnings)

The engine is deterministic and side-effect free: it never touches the database.
It returns a structured result (accepted features, skipped features, auto-fix
log, HITL items awaiting a user decision, dataset warnings, and summary counts).
Commit happens elsewhere, only after every HITL item has a recorded decision.

Severity levels:
  CRITICAL  file -> reject; feature -> skip (collected in the report)
  IMPORTANT feature kept but flagged; usually raises a HITL item
  WARNING   logged only, never blocks

Heavy geometry ops (validity, make_valid, area, reprojection) use shapely /
pyproj, imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field, asdict

CRITICAL = "critical"
IMPORTANT = "important"
WARNING = "warning"

_GEOM_TYPES = {"Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon"}
_MAX_BYTES = 100 * 1024 * 1024          # 100 MB
_MAX_FEATURES = 100_000                 # Stage 4.6 soft cap
_COMPLEX_VERTS = 10_000                 # Stage 4.5 complexity threshold
_OUTLIER_FIELD_HINTS = ("capacity", "count", "population", "students", "beds",
                        "size", "area", "number", "num_", "qty", "quantity", "rooms")


class FileReject(Exception):
    """Stage-1 critical failure — the whole file is rejected."""


@dataclass
class Issue:
    stage: int
    severity: str
    check: str
    feature_index: int | None
    feature_id: str | None
    message: str
    action: str                    # what the system did (skipped/auto-fixed/flagged/…)
    resolver: str = "System"       # System | User
    field: str | None = None       # the column involved, when applicable

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class HITLItem:
    kind: str                      # missing_geometry | missing_attribute | latlong_swap |
                                   # crs | duplicate | category | outlier | feature_count
    feature_index: int | None
    feature_id: str | None
    field: str | None
    message: str
    options: list[str]             # allowed resolutions for the UI
    context: dict = field(default_factory=dict)   # partial data / suggestions
    resolved: bool = False
    resolution: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# coordinate helpers
# --------------------------------------------------------------------------- #

def _iter_positions(coords):
    """Yield [x, y(, …)] positions from any nested GeoJSON coordinate array."""
    if isinstance(coords, (list, tuple)):
        if coords and isinstance(coords[0], (int, float)):
            yield coords
        else:
            for c in coords:
                yield from _iter_positions(c)


def _in_range(pos) -> bool:
    lon, lat = pos[0], pos[1]
    return -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0


def _looks_swapped(coords) -> bool:
    """Heuristic: 2nd value (lat slot) exceeds 90 while the 1st value would be a
    valid latitude — i.e. axes are probably reversed."""
    for pos in _iter_positions(coords):
        lon, lat = pos[0], pos[1]
        if abs(lat) > 90 and abs(lat) <= 180 and abs(lon) <= 90:
            return True
    return False


def _swap_coords(coords):
    """Return a copy of the coordinate tree with x/y swapped in every position."""
    if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
        return [coords[1], coords[0]] + list(coords[2:])
    return [_swap_coords(c) for c in coords]


def _ring_closed(ring) -> bool:
    return len(ring) >= 1 and list(ring[0]) == list(ring[-1])


def _dedupe_consecutive(ring) -> tuple[list, int]:
    out, removed = [], 0
    for p in ring:
        if out and list(out[-1]) == list(p):
            removed += 1
            continue
        out.append(p)
    return out, removed


def _min_vertices_ok(gtype: str, coords) -> bool:
    if gtype == "LineString":
        return len(coords) >= 2
    if gtype == "Polygon":
        return all(len(r) >= 4 for r in coords)
    if gtype == "MultiLineString":
        return all(len(l) >= 2 for l in coords)
    if gtype == "MultiPolygon":
        return all(all(len(r) >= 4 for r in poly) for poly in coords)
    return True


def _geom_key(geom: dict) -> str:
    return json.dumps([geom.get("type"), geom.get("coordinates")], sort_keys=True)


def _feature_id(feat: dict, idx: int) -> str | None:
    if feat.get("id") is not None:
        return str(feat["id"])
    props = feat.get("properties") or {}
    for k in ("id", "ID", "fid", "gid", "objectid", "OBJECTID"):
        if props.get(k) is not None:
            return str(props[k])
    return None


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #

@dataclass
class ValidationResult:
    filename: str
    columns: list[dict]            # [{name, dtype, null_count}]
    accepted: list[dict]           # cleaned features ready to commit
    skipped: list[dict]            # {feature_index, feature_id, reason}
    issues: list[Issue]
    hitl: list[HITLItem]
    crs_epsg: int | None
    reprojected: bool

    def summary(self) -> dict:
        by_sev = {CRITICAL: 0, IMPORTANT: 0, WARNING: 0}
        for i in self.issues:
            by_sev[i.severity] = by_sev.get(i.severity, 0) + 1
        return {
            "total_features": len(self.accepted) + len(self.skipped),
            "accepted": len(self.accepted),
            "skipped": len(self.skipped),
            "auto_fixed": sum(1 for i in self.issues if i.action.startswith("auto")),
            "pending_hitl": sum(1 for h in self.hitl if not h.resolved),
            "issues_by_severity": by_sev,
        }

    def report(self) -> dict:
        return {
            "filename": self.filename,
            "summary": self.summary(),
            "columns": self.columns,
            "issues": [i.as_dict() for i in self.issues],
            "hitl": [h.as_dict() for h in self.hitl],
            "skipped": self.skipped,
            "crs_epsg": self.crs_epsg,
            "reprojected": self.reprojected,
        }


# --------------------------------------------------------------------------- #
# Stage 1 — file level
# --------------------------------------------------------------------------- #

def stage1_file(filename: str, raw: bytes) -> dict:
    """Validate file → return the parsed GeoJSON dict. Raises FileReject (critical)."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in ("geojson", "json"):
        raise FileReject(f"File extension .{ext} not allowed — upload a .geojson or .json file.")
    if len(raw) > _MAX_BYTES:
        raise FileReject(f"File is {len(raw)//(1024*1024)} MB, over the {_MAX_BYTES//(1024*1024)} MB limit "
                         "— split it or use chunked upload.")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise FileReject(f"File is not valid UTF-8 (unreadable) at byte {e.start}.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise FileReject(f"Invalid JSON syntax at line {e.lineno}, column {e.colno}: {e.msg}.")
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise FileReject('Not a GeoJSON FeatureCollection (top-level "type" must be "FeatureCollection").')
    feats = data.get("features")
    if not isinstance(feats, list):
        raise FileReject('GeoJSON is missing a "features" array.')
    if len(feats) == 0:
        raise FileReject("The FeatureCollection has no features (empty dataset).")
    return data


# --------------------------------------------------------------------------- #
# CRS detection / reprojection
# --------------------------------------------------------------------------- #

def _detect_crs(data: dict) -> tuple[int | None, bool]:
    """Return (epsg, undetermined). GeoJSON default is 4326 when no crs member."""
    crs = data.get("crs")
    if not crs:
        return 4326, False
    try:
        name = (crs.get("properties") or {}).get("name", "")
    except AttributeError:
        return None, True
    if not name:
        return None, True
    up = str(name).upper()
    if "CRS84" in up or "4326" in up:
        return 4326, False
    # e.g. "urn:ogc:def:crs:EPSG::32636" or "EPSG:32636"
    for tok in up.replace(":", " ").split():
        if tok.isdigit():
            return int(tok), False
    return None, True


def _reproject(features: list[dict], epsg: int) -> None:
    """Reproject every feature's coordinates from ``epsg`` to 4326 in place."""
    from pyproj import Transformer  # lazy
    tr = Transformer.from_crs(epsg, 4326, always_xy=True)

    def conv(coords):
        if isinstance(coords, (list, tuple)) and coords and isinstance(coords[0], (int, float)):
            x, y = tr.transform(coords[0], coords[1])
            return [x, y] + list(coords[2:])
        return [conv(c) for c in coords]

    for f in features:
        g = f.get("geometry")
        if g and g.get("coordinates") is not None:
            g["coordinates"] = conv(g["coordinates"])


# --------------------------------------------------------------------------- #
# Stage 2 — geometry (per feature)
# --------------------------------------------------------------------------- #

def _flag_invalid(feat: dict, idx: int, fid: str | None, reason: str, keepable: bool,
                  issues: list[Issue], hitl: list[HITLItem]) -> dict:
    """Flag a broken-geometry feature for a human decision instead of dropping it.

    ``keepable`` controls whether "keep as-is" is even offered (structurally sound
    but degenerate geometries can be kept; truly broken ones can only be dropped).
    The feature is returned (kept pending) so nothing leaves the dataset without a
    recorded user decision.
    """
    opts = ["keep", "drop"] if keepable else ["drop"]
    hitl.append(HITLItem("invalid_geometry", idx, fid, None,
                         f"Geometry problem: {reason}. Decide before it's dropped.",
                         opts, context={"reason": reason, "keepable": keepable}))
    issues.append(Issue(2, IMPORTANT, "invalid_geometry", idx, fid, reason, "flagged for HITL"))
    return feat


def _shapely_fix(geom: dict, idx: int, fid: str | None, issues: list[Issue]):
    """Validity/self-intersection/zero-area checks via shapely.

    Returns ``(geometry_dict_or_None, reason_or_None, keepable)``. A non-None
    reason means the feature can't be auto-cleaned and must go to HITL.
    """
    try:
        from shapely.geometry import shape, mapping
        from shapely.validation import make_valid
    except Exception:  # noqa: BLE001 — shapely missing: skip these checks gracefully
        return geom, None, True

    try:
        g = shape(geom)
    except Exception as e:  # noqa: BLE001
        return None, f"could not build geometry ({e})", False

    if g.is_empty:
        return None, "empty geometry", False

    if not g.is_valid:
        try:
            repaired = make_valid(g)
            if repaired.is_empty or not repaired.is_valid:
                raise ValueError("still invalid")
            issues.append(Issue(2, IMPORTANT, "self_intersection", idx, fid,
                                "invalid/self-intersecting geometry", "auto-fixed (make_valid)"))
            g = repaired
        except Exception:  # noqa: BLE001
            return None, "invalid geometry, repair failed", False

    if g.geom_type in ("Polygon", "MultiPolygon") and g.area == 0:
        return None, "zero-area polygon", True   # structurally fine, user may keep

    return mapping(g), None, None


def stage2_feature(feat: dict, idx: int, seen_geoms: dict, seen_ids: set,
                   issues: list[Issue], hitl: list[HITLItem]) -> dict | None:
    """Validate & clean one feature's geometry. Returns the cleaned feature, or
    None if it was skipped. May append a HITL item (missing geometry / swap)."""
    fid = _feature_id(feat, idx)
    geom = feat.get("geometry")

    # 2.2 missing geometry -> HITL (never auto-drop)
    if geom is None:
        hitl.append(HITLItem("missing_geometry", idx, fid, None,
                             "Feature has no geometry.",
                             ["fix", "autofill", "drop"],
                             context={"properties": feat.get("properties") or {}}))
        issues.append(Issue(2, IMPORTANT, "missing_geometry", idx, fid,
                            "geometry is null", "flagged for HITL"))
        return feat  # kept pending; commit is blocked until resolved

    gtype = geom.get("type")
    coords = geom.get("coordinates")

    # 2.1 geometry type
    if gtype not in _GEOM_TYPES:
        return _flag_invalid(feat, idx, fid, f"unsupported geometry type {gtype!r}", False, issues, hitl)
    # 2.3 empty coordinates
    if not coords:
        return _flag_invalid(feat, idx, fid, "empty coordinates", False, issues, hitl)

    # 2.6 lat/long swap detection -> HITL
    if _looks_swapped(coords):
        hitl.append(HITLItem("latlong_swap", idx, fid, None,
                             "Coordinates look like they have swapped lat/long.",
                             ["fix", "keep", "skip"],
                             context={"suggestion": "swap x/y"}))
        issues.append(Issue(2, IMPORTANT, "latlong_swap", idx, fid,
                            "possible reversed axes", "flagged for HITL"))
        return feat

    # 2.5 coordinate range
    if not all(_in_range(p) for p in _iter_positions(coords)):
        return _flag_invalid(feat, idx, fid,
                             "coordinate out of lon[-180,180]/lat[-90,90]", True, issues, hitl)

    # 2.9 polygon ring closure (auto-fix) + 2.14 dedupe consecutive vertices
    if gtype in ("Polygon", "MultiPolygon"):
        polys = coords if gtype == "MultiPolygon" else [coords]
        removed_total = 0
        for poly in polys:
            for r, ring in enumerate(poly):
                ring2, removed = _dedupe_consecutive(ring)
                removed_total += removed
                if not _ring_closed(ring2):
                    ring2 = ring2 + [ring2[0]]
                    issues.append(Issue(2, WARNING, "polygon_closure", idx, fid,
                                        "unclosed ring", "auto-fixed (closed ring)"))
                poly[r] = ring2
        if removed_total:
            issues.append(Issue(2, WARNING, "duplicate_vertices", idx, fid,
                                f"{removed_total} consecutive duplicate vertex(es)",
                                "auto-fixed (removed)"))

    # 2.10 minimum vertices
    if not _min_vertices_ok(gtype, coords):
        return _flag_invalid(feat, idx, fid, "too few vertices for geometry type", False, issues, hitl)

    # 2.11/2.12/2.13 validity, self-intersection, zero-area (shapely)
    fixed, reason, keepable = _shapely_fix({"type": gtype, "coordinates": coords}, idx, fid, issues)
    if reason:
        return _flag_invalid(feat, idx, fid, reason, bool(keepable), issues, hitl)
    feat = {**feat, "geometry": fixed}

    # 2.15 duplicate geometry (tracked; HITL decided at dataset level)
    key = _geom_key(feat["geometry"])
    seen_geoms.setdefault(key, []).append(idx)

    # 2.16 duplicate feature id -> regenerate
    if fid is not None:
        if fid in seen_ids:
            new_id = str(uuid.uuid4())
            feat = {**feat, "id": new_id}
            props = dict(feat.get("properties") or {})
            props["_orig_id"] = fid
            feat["properties"] = props
            issues.append(Issue(2, WARNING, "duplicate_id", idx, new_id,
                                f"duplicate id {fid}", f"auto-fixed (new id {new_id[:8]})"))
        else:
            seen_ids.add(fid)

    return feat


# --------------------------------------------------------------------------- #
# Stage 3 — attributes (per feature)
# --------------------------------------------------------------------------- #

def _coerce_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return None
    return None


def stage3_attributes(features: list[dict], required_fields: list[str],
                      numeric_fields: list[str], issues: list[Issue],
                      hitl: list[HITLItem]) -> None:
    """Attribute checks over the ACCEPTED features. ``required_fields`` are the
    columns the user marked not-null; ``numeric_fields`` are treated numerically."""
    req = set(required_fields or [])
    for pos, feat in enumerate(features):
        idx = feat.get("_src_index", pos)   # key HITL by original file index
        props = dict(feat.get("properties") or {})
        fid = _feature_id(feat, idx)

        # 3.3 empty strings -> NULL, EXCEPT on not-null (required) columns — those
        # are left empty so the required-field check below raises a HITL decision
        # instead of silently nulling a value the user said must be present.
        for k, v in list(props.items()):
            if isinstance(v, str) and v.strip() == "":
                if k in req:
                    continue
                props[k] = None
                issues.append(Issue(3, WARNING, "empty_string", idx, fid,
                                    f"empty string in '{k}'", "auto-fixed (set NULL)", field=k))

        # 3.2 numeric coercion + 3.5 outliers
        for k in numeric_fields or []:
            if k in props and props[k] is not None and not isinstance(props[k], (int, float)):
                num = _coerce_number(props[k])
                if num is None:
                    issues.append(Issue(3, IMPORTANT, "type_coercion", idx, fid,
                                        f"'{k}'={props[k]!r} not numeric", "auto-fixed (set NULL)", field=k))
                    props[k] = None
                else:
                    props[k] = num
            val = props.get(k)
            if isinstance(val, (int, float)) and val < 0 and _is_outlier_field(k):
                hitl.append(HITLItem("outlier", idx, fid, k,
                                     f"'{k}' = {val} looks invalid (negative).",
                                     ["fix", "keep", "null"], context={"value": val}))
                issues.append(Issue(3, IMPORTANT, "outlier", idx, fid,
                                    f"negative '{k}'={val}", "flagged for HITL", field=k))

        # 3.1 missing required attribute -> HITL
        for k in req:
            if props.get(k) in (None, ""):
                hitl.append(HITLItem("missing_attribute", idx, fid, k,
                                     f"Required field '{k}' is missing.",
                                     ["fix", "autofill", "drop"],
                                     context={"properties": props}))
                issues.append(Issue(3, IMPORTANT, "missing_required", idx, fid,
                                    f"required '{k}' is null", "flagged for HITL", field=k))

        feat["properties"] = props


def _is_outlier_field(name: str) -> bool:
    n = (name or "").lower()
    return any(h in n for h in _OUTLIER_FIELD_HINTS)


# --------------------------------------------------------------------------- #
# Stage 4 — dataset level (warnings)
# --------------------------------------------------------------------------- #

def stage4_dataset(features: list[dict], issues: list[Issue], hitl: list[HITLItem]) -> None:
    # 4.6 feature count
    if len(features) > _MAX_FEATURES:
        hitl.append(HITLItem("feature_count", None, None, None,
                             f"{len(features)} features exceeds the {_MAX_FEATURES} cap.",
                             ["proceed", "cancel"]))
        issues.append(Issue(4, IMPORTANT, "feature_count", None, None,
                            f"{len(features)} features > cap", "flagged for HITL"))

    # centroids for spatial-outlier + bbox
    pts = []
    for idx, f in enumerate(features):
        g = f.get("geometry")
        if not g or not g.get("coordinates"):
            continue
        xs = [p[0] for p in _iter_positions(g["coordinates"])]
        ys = [p[1] for p in _iter_positions(g["coordinates"])]
        if xs and ys:
            pts.append((idx, sum(xs) / len(xs), sum(ys) / len(ys), len(xs)))
            # 4.5 complexity
            if len(xs) > _COMPLEX_VERTS:
                issues.append(Issue(4, WARNING, "complexity", idx, _feature_id(f, idx),
                                    f"{len(xs)} vertices", "warned (simplification available)"))
    if not pts:
        return

    minx = min(p[1] for p in pts); maxx = max(p[1] for p in pts)
    miny = min(p[2] for p in pts); maxy = max(p[2] for p in pts)
    # 4.1 implausibly large extent (likely mixed CRS / errors)
    if (maxx - minx) > 60 or (maxy - miny) > 60:
        issues.append(Issue(4, WARNING, "bbox_extent", None, None,
                            f"dataset spans a very large area [{minx:.2f},{miny:.2f},{maxx:.2f},{maxy:.2f}]",
                            "warned"))

    # 4.2 spatial outliers: centroid far from the median center
    cx = sorted(p[1] for p in pts)[len(pts) // 2]
    cy = sorted(p[2] for p in pts)[len(pts) // 2]
    dists = [math.hypot(p[1] - cx, p[2] - cy) for p in pts]
    mean = sum(dists) / len(dists)
    var = sum((d - mean) ** 2 for d in dists) / len(dists)
    std = math.sqrt(var)
    if std > 0:
        for (idx, _, _, _), d in zip(pts, dists):
            if d > mean + 4 * std and d > 1.0:
                issues.append(Issue(4, WARNING, "spatial_outlier", idx, None,
                                    "feature lies far from the main cluster", "flagged for review"))


# --------------------------------------------------------------------------- #
# top-level entry (Stages 1, 2, 4). Stage 3 runs after the user picks fields.
# --------------------------------------------------------------------------- #

def validate_upload(filename: str, raw: bytes) -> ValidationResult:
    data = stage1_file(filename, raw)                 # raises FileReject on critical
    features = data["features"]

    crs_epsg, undetermined = _detect_crs(data)
    issues: list[Issue] = []
    hitl: list[HITLItem] = []
    reprojected = False

    if undetermined:
        # 2.7 CRS undetermined -> block reprojection until user supplies EPSG
        hitl.append(HITLItem("crs", None, None, None,
                             "The dataset's CRS could not be determined.",
                             ["set_epsg"], context={}))
        issues.append(Issue(2, CRITICAL, "crs_undetermined", None, None,
                            "CRS undetermined", "flagged for HITL"))
    elif crs_epsg and crs_epsg != 4326:
        try:
            _reproject(features, crs_epsg)             # 2.8 reproject to 4326
            reprojected = True
            issues.append(Issue(2, WARNING, "reprojection", None, None,
                                f"reprojected from EPSG:{crs_epsg} to EPSG:4326", "auto-fixed"))
        except Exception as e:  # noqa: BLE001
            issues.append(Issue(2, IMPORTANT, "reprojection", None, None,
                                f"reprojection failed: {e}", "flagged"))

    accepted: list[dict] = []
    skipped: list[dict] = []
    seen_geoms: dict = {}
    seen_ids: set = set()
    for idx, feat in enumerate(features):
        before = len(issues)
        cleaned = stage2_feature(feat, idx, seen_geoms, seen_ids, issues, hitl)
        if cleaned is None:
            reason = next((i.message for i in reversed(issues[before:])), "invalid geometry")
            skipped.append({"feature_index": idx, "feature_id": _feature_id(feat, idx), "reason": reason})
        else:
            accepted.append({**cleaned, "_src_index": idx})

    # 2.15 duplicate geometries -> HITL only past a threshold (>5%)
    dupe_groups = [v for v in seen_geoms.values() if len(v) > 1]
    dupe_count = sum(len(v) - 1 for v in dupe_groups)
    if dupe_count:
        sev = IMPORTANT if dupe_count > 0.05 * max(1, len(accepted)) else WARNING
        issues.append(Issue(2, sev, "duplicate_geometry", None, None,
                            f"{dupe_count} duplicate geometry(ies) across {len(dupe_groups)} group(s)",
                            "flagged for HITL" if sev == IMPORTANT else "warned"))
        if sev == IMPORTANT:
            hitl.append(HITLItem("duplicate", None, None, None,
                                 f"{dupe_count} duplicate geometries (> 5% of dataset).",
                                 ["keep", "drop_duplicates"], context={"count": dupe_count}))

    stage4_dataset(accepted, issues, hitl)

    columns = _infer_columns(features)
    return ValidationResult(
        filename=filename, columns=columns, accepted=accepted, skipped=skipped,
        issues=issues, hitl=hitl, crs_epsg=crs_epsg, reprojected=reprojected,
    )


def _infer_columns(features: list[dict]) -> list[dict]:
    """Infer the attribute columns (name, dominant dtype, null count) from properties."""
    names: list[str] = []
    dtypes: dict[str, set] = {}
    nulls: dict[str, int] = {}
    for f in features:
        props = f.get("properties") or {}
        for k, v in props.items():
            if k not in dtypes:
                names.append(k)
                dtypes[k] = set()
                nulls[k] = 0
            if v is None or (isinstance(v, str) and v.strip() == ""):
                nulls[k] += 1
            else:
                dtypes[k].add(_pytype(v))
    cols = []
    for k in names:
        ds = dtypes[k]
        dtype = "mixed" if len(ds) > 1 else (next(iter(ds)) if ds else "null")
        cols.append({"name": k, "dtype": dtype, "null_count": nulls[k]})
    return cols


def _pytype(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, str):
        return "string"
    return "object"


# --------------------------------------------------------------------------- #
# commit: apply the user's HITL decisions to produce the final feature list
# --------------------------------------------------------------------------- #

def all_hitl_resolved(result: ValidationResult) -> bool:
    return all(h.resolved for h in result.hitl)


def finalize(result: ValidationResult) -> list[dict]:
    """Apply every recorded HITL decision to the accepted features and return the
    final GeoJSON features (geometry + properties, internal fields stripped)."""
    by_idx: dict[int, list[HITLItem]] = {}
    globals_: list[HITLItem] = []
    for h in result.hitl:
        (globals_ if h.feature_index is None else by_idx.setdefault(h.feature_index, [])).append(h)

    out: list[dict] = []
    for feat in result.accepted:
        idx = feat.get("_src_index")
        geom = feat.get("geometry")
        props = dict(feat.get("properties") or {})
        drop = False
        for h in by_idx.get(idx, []):
            r = h.resolution or {}
            act = r.get("action")
            if h.kind == "missing_geometry":
                if act == "drop":
                    drop = True
                elif act in ("fix", "autofill") and r.get("geometry"):
                    geom = r["geometry"]
            elif h.kind == "missing_attribute":
                if act == "drop":
                    drop = True
                elif act in ("fix", "autofill"):
                    props[h.field] = r.get("value")
            elif h.kind == "latlong_swap":
                if act == "skip":
                    drop = True
                elif act == "fix" and geom:
                    geom = {**geom, "coordinates": _swap_coords(geom["coordinates"])}
            elif h.kind == "outlier":
                if act == "null":
                    props[h.field] = None
                elif act == "fix":
                    props[h.field] = r.get("value")
            elif h.kind == "invalid_geometry":
                if act == "drop":
                    drop = True
                # "keep" leaves the geometry as-is
        props.pop("_orig_id", None)
        if drop or geom is None:
            continue
        out.append({"type": "Feature", "geometry": geom, "properties": props})

    # global duplicate-geometry decision
    for h in globals_:
        if h.kind == "duplicate" and (h.resolution or {}).get("action") == "drop_duplicates":
            seen: set = set()
            deduped: list[dict] = []
            for f in out:
                k = _geom_key(f["geometry"])
                if k in seen:
                    continue
                seen.add(k)
                deduped.append(f)
            out = deduped
    return out
