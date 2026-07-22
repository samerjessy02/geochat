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
        # Unified FieldMetadata — the same shape whether the description later
        # comes from a schema file ("schema") or the LLM ("llm") or is edited.
        cols.append({
            "name": k, "dtype": dtype, "type": dtype, "null_count": nulls[k],
            "description": "", "required": False, "source": "",
            "is_primary_key": False, "foreign_key": None,
        })
    return cols


# --------------------------------------------------------------------------- #
# optional schema file: the source of truth for column metadata (no LLM)
# --------------------------------------------------------------------------- #

_SCHEMA_CONTAINERS = ("fields", "columns", "tables")


def validate_schema(data) -> None:
    """Validate that ``data`` (parsed JSON) is a well-formed dataset schema.

    A valid schema is a JSON object that declares its columns through one of the
    recognized containers — ``fields`` (list), ``columns`` (list or object), or a
    database-DDL ``tables`` object — or a flat ``{column: spec}`` mapping. Each
    column's ``type``/``description`` must be strings and ``required`` a boolean
    when present. Raises :class:`FileReject` listing every problem found.
    """
    errors: list[str] = []

    def _check_col(spec, where: str, name_required: bool) -> None:
        if isinstance(spec, str):
            return  # shorthand: value is a type or description string
        if not isinstance(spec, dict):
            errors.append(f"{where}: must be an object or string, got {type(spec).__name__}.")
            return
        if name_required:
            nm = spec.get("name") or spec.get("column_name") or spec.get("column")
            if not (isinstance(nm, str) and nm.strip()):
                errors.append(f"{where}: missing a non-empty string 'name'.")
        t = spec.get("type", spec.get("data_type"))
        if t is not None and not isinstance(t, str):
            errors.append(f"{where}: 'type' must be a string.")
        d = spec.get("description", spec.get("desc"))
        if d is not None and not isinstance(d, str):
            errors.append(f"{where}: 'description' must be a string.")
        r = spec.get("required", spec.get("not_null", spec.get("notNull")))
        if r is not None and not isinstance(r, bool):
            errors.append(f"{where}: 'required' must be true or false.")

    if not isinstance(data, dict):
        raise FileReject('Schema must be a JSON object, e.g. {"fields": [ {"name": "...", '
                         '"type": "...", "description": "..."} ]} or {"tables": {...}}.')

    if isinstance(data.get("tables"), dict) or "tables" in data:
        tables = data.get("tables")
        if not isinstance(tables, dict) or not tables:
            errors.append("'tables' must be a non-empty object of table definitions.")
        else:
            for tname, tdef in tables.items():
                if not isinstance(tdef, dict):
                    errors.append(f"tables.{tname}: must be an object.")
                    continue
                cols = tdef.get("columns")
                if cols is None:
                    errors.append(f"tables.{tname}: missing 'columns'.")
                elif isinstance(cols, dict):
                    if not cols:
                        errors.append(f"tables.{tname}.columns: is empty.")
                    for cn, cv in cols.items():
                        _check_col(cv, f"tables.{tname}.columns.{cn}", name_required=False)
                elif isinstance(cols, list):
                    for i, cv in enumerate(cols):
                        _check_col(cv, f"tables.{tname}.columns[{i}]", name_required=True)
                else:
                    errors.append(f"tables.{tname}.columns: must be an object or list.")
                pk = tdef.get("primary_key")
                if pk is not None and not isinstance(pk, str):
                    errors.append(f"tables.{tname}.primary_key: must be a column-name string.")
                fks = tdef.get("foreign_keys")
                if fks is not None and not isinstance(fks, dict):
                    errors.append(f"tables.{tname}.foreign_keys: must be an object of column → reference.")
    else:
        fields = data.get("fields")
        if fields is not None:
            if not isinstance(fields, list):
                errors.append("'fields' must be a list of column objects.")
            elif not fields:
                errors.append("'fields' is empty — declare at least one column.")
            else:
                for i, f in enumerate(fields):
                    _check_col(f, f"fields[{i}]", name_required=True)
        elif "columns" in data:
            cols = data["columns"]
            if isinstance(cols, list):
                if not cols:
                    errors.append("'columns' is empty — declare at least one column.")
                for i, cv in enumerate(cols):
                    _check_col(cv, f"columns[{i}]", name_required=True)
            elif isinstance(cols, dict):
                if not cols:
                    errors.append("'columns' is empty — declare at least one column.")
                for cn, cv in cols.items():
                    _check_col(cv, f"columns.{cn}", name_required=False)
            else:
                errors.append("'columns' must be a list or object.")
        else:
            # flat {column: spec} mapping fallback
            if not data:
                errors.append('Schema is empty — provide "fields", "columns", or "tables".')
            else:
                for cn, cv in data.items():
                    if not (isinstance(cn, str) and cn.strip()):
                        errors.append("Top-level column names must be non-empty strings.")
                    _check_col(cv, str(cn), name_required=False)

    if errors:
        shown = errors[:8]
        more = "" if len(errors) <= 8 else f"\n… and {len(errors) - 8} more issue(s)."
        raise FileReject("Schema file is not a valid dataset schema:\n- "
                         + "\n- ".join(shown) + more)


def parse_schema(raw: bytes) -> dict[str, dict]:
    """Parse an optional schema file into ``{lower_column_name: metadata}``.

    The file is validated first (see :func:`validate_schema`) and rejected with a
    clear message if it is not a well-formed dataset schema.

    Supports three shapes, all of which may carry ``description`` text:
      * the documented ``{"fields": [{name,type,description,required}, ...]}`` form;
      * a ``{"columns": [...]}`` / ``{"columns": {...}}`` form;
      * a database-DDL form ``{"tables": {"<t>": {"primary_key", "foreign_keys",
        "columns": {"<col>": "<TYPE>"}, "descriptions": {...}}}}`` — every table's
        columns are flattened into one map, with primary/foreign keys attached.

    Flexible key aliases are honoured throughout (pk/primary_key,
    references/foreign_key/fk, not_null/required, desc/description).
    """
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise FileReject(f"Schema file is not valid JSON: {e}")

    validate_schema(data)

    out: dict[str, dict] = {}

    def _add(name, spec) -> None:
        if not name:
            return
        if not isinstance(spec, dict):
            spec = {"description": str(spec)}
        fk = spec.get("foreign_key") or spec.get("references") or spec.get("fk")
        if isinstance(fk, dict):
            fk = (fk.get("table", "") + ("." + fk["column"] if fk.get("column") else "")) or None
        pk = bool(spec.get("primary_key") or spec.get("pk") or spec.get("primaryKey"))
        key = str(name).strip().lower()
        meta = {
            "type": spec.get("type") or spec.get("data_type"),
            "description": str(spec.get("description") or spec.get("desc") or "").strip(),
            "required": bool(spec.get("required") or spec.get("not_null") or spec.get("notNull") or pk),
            "is_primary_key": pk,
            "foreign_key": fk or None,
        }
        # If the same column name appears in several tables, keep the richest entry
        # (prefer one that carries a description / primary key).
        prev = out.get(key)
        if prev and not meta["description"] and not meta["is_primary_key"]:
            if prev.get("description") or prev.get("is_primary_key"):
                return
        out[key] = meta

    # --- database-DDL form: {"tables": {tname: {columns, primary_key, foreign_keys}}} ---
    tables = data.get("tables")
    if isinstance(tables, dict):
        for tdef in tables.values():
            if not isinstance(tdef, dict):
                continue
            pk = tdef.get("primary_key")
            fks = tdef.get("foreign_keys") or {}
            descs = tdef.get("descriptions") or {}
            tcols = tdef.get("columns") or {}
            if isinstance(tcols, dict):
                col_items = list(tcols.items())
            elif isinstance(tcols, list):
                col_items = [(c.get("name") or c.get("column"), c)
                             for c in tcols if isinstance(c, dict)]
            else:
                col_items = []
            for cname, cval in col_items:
                if not cname:
                    continue
                spec = dict(cval) if isinstance(cval, dict) else {"type": str(cval)}
                spec.setdefault("type", None)
                if cname == pk:
                    spec["primary_key"] = True
                if cname in fks:
                    spec["foreign_key"] = fks[cname]
                if not spec.get("description") and isinstance(descs, dict) and descs.get(cname):
                    spec["description"] = descs[cname]
                _add(cname, spec)
        if not out:
            raise FileReject("Schema defines no usable columns — check the 'tables' definitions.")
        return out

    # --- {"fields": [...]} / {"columns": [...]|{...}} / flat {col: spec} forms ---
    fields = data.get("fields")
    if fields is None:
        fields = data.get("columns", data)

    if isinstance(fields, list):
        for f in fields:
            if isinstance(f, dict):
                _add(f.get("name") or f.get("column_name") or f.get("column"), f)
    elif isinstance(fields, dict):
        for name, spec in fields.items():
            _add(name, spec)
    if not out:
        raise FileReject("Schema defines no usable columns — declare them under "
                         '"fields", "columns", or "tables".')
    return out


# Column-name aliases for the two axes of a Point geometry, so a schema that
# documents coordinates (which live in ``geometry``, not ``properties``) can still
# surface them as editable columns.
_LON_KEYS = {"longitude", "lon", "lng", "long", "x"}
_LAT_KEYS = {"latitude", "lat", "y"}


def coordinate_columns_from_schema(schema: dict, features: list[dict]) -> list[dict]:
    """Synthesize longitude/latitude columns that the schema declares but that live
    in the Point ``geometry`` rather than in ``properties``.

    Returns unified-shape column dicts (dtype ``float``) with a real ``null_count``
    computed from the features, so their schema descriptions can be displayed and
    edited like any attribute column. Fields the schema does not mention are not
    added — the no-schema flow is unaffected.
    """
    def _coord(f, axis):
        g = f.get("geometry") or {}
        if g.get("type") != "Point":
            return None
        c = g.get("coordinates")
        if isinstance(c, (list, tuple)) and len(c) > axis and isinstance(c[axis], (int, float)):
            return c[axis]
        return None

    out: list[dict] = []
    for key in schema:
        axis = 0 if key in _LON_KEYS else (1 if key in _LAT_KEYS else None)
        if axis is None:
            continue
        nulls = sum(1 for f in features if _coord(f, axis) is None)
        out.append({
            "name": key, "dtype": "float", "type": "float", "null_count": nulls,
            "description": "", "required": False, "source": "",
            "is_primary_key": False, "foreign_key": None,
        })
    return out


def enrich_columns_with_schema(columns: list[dict], schema: dict) -> None:
    """Overlay schema metadata onto the inferred columns (in place).

    Columns present in the schema inherit its type/required/keys. ``source`` is
    set to ``"schema"`` only when the schema actually supplies description text —
    so a DDL schema (types + keys, no prose) leaves ``description``/``source``
    empty and those columns can still be auto-described by the LLM.
    """
    for c in columns:
        s = schema.get(c["name"].strip().lower())
        if not s:
            continue
        c["required"] = s["required"]
        c["type"] = s["type"] or c["dtype"]
        c["is_primary_key"] = s["is_primary_key"]
        c["foreign_key"] = s["foreign_key"]
        if s["description"]:
            c["description"] = s["description"]
            c["source"] = "schema"


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
