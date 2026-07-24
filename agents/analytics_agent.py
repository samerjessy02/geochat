"""
agents/analytics_agent.py — natural-language analytics over dataset properties.

Flow (numbers are NEVER produced by the LLM):
  1. Load the selected datasets' rows from PostGIS (:mod:`agents.geojson_table`).
  2. Infer each field's type (numeric vs categorical) and gather sample values.
  3. Ask the LLM ONLY for an analysis *plan* — which field(s) to group by, which
     aggregation, and whether a chart and/or table is appropriate.
  4. Execute that plan deterministically in pure Python (group-by count/agg,
     two-field cross-tab comparison, numeric histogram).
  5. Return normalized output the UI renders directly:
       {summary, plan, chart:{type,x,series,...}|None, table:{columns,rows}|None,
        dataset_scope:[...]}

Chart shape is renderer-agnostic (the frontend builds Plotly traces from it):
  chart = {"type": "bar|pie|line|scatter|histogram",
           "title", "x_label", "y_label",
           "x": [...], "series": [{"name": str, "y": [...]}, ...]}
"""

from __future__ import annotations

import json
import re

from agents.geojson_table import fetch_dataset_tables
from agents.llm_client import get_llm, LLMError
from agents.logging_config import get_logger, log_step

log = get_logger("analytics_agent")

# Columns we add ourselves / never want to group on by default.
_INTERNAL_NUMERIC = {"_lon", "_lat"}


class AnalyticsError(Exception):
    """Invalid analytics request (no datasets, no rows, etc.)."""


# --------------------------------------------------------------------------- #
# typing + helpers
# --------------------------------------------------------------------------- #
def _is_number(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, str):
        s = v.strip().replace(",", "")
        if not s:
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False


def _num(v) -> float | None:
    try:
        return float(str(v).strip().replace(",", ""))
    except (ValueError, AttributeError, TypeError):
        return None


def _field_profile(rows: list[dict]) -> dict:
    """{field: {"type": "numeric|categorical", "distinct": [sample values]}}."""
    fields: dict[str, dict] = {}
    for r in rows:
        for k in r.keys():
            fields.setdefault(k, {"nonnull": 0, "num": 0, "distinct": []})
    for r in rows:
        for k, meta in fields.items():
            v = r.get(k)
            if v is None or v == "":
                continue
            meta["nonnull"] += 1
            if _is_number(v):
                meta["num"] += 1
            elif len(meta["distinct"]) < 12 and v not in meta["distinct"]:
                meta["distinct"].append(v)
    out = {}
    for k, meta in fields.items():
        is_num = meta["nonnull"] > 0 and meta["num"] == meta["nonnull"]
        out[k] = {"type": "numeric" if is_num else "categorical",
                  "distinct": meta["distinct"]}
    return out


def _categorical_fields(profile: dict) -> list[str]:
    return [k for k, m in profile.items()
            if m["type"] == "categorical" and k not in ("_geometry_type",)]


def _numeric_fields(profile: dict) -> list[str]:
    return [k for k, m in profile.items()
            if m["type"] == "numeric" and k not in _INTERNAL_NUMERIC]


# --------------------------------------------------------------------------- #
# LLM planning
# --------------------------------------------------------------------------- #
_PLAN_SYSTEM = (
    "You plan a data analysis over a tabular dataset. You DO NOT compute any numbers — "
    "you only choose fields and the operation. Given the user's question and the available "
    "fields (with types and sample values), return a JSON plan.\n"
    "Schema of the plan:\n"
    '{\n'
    '  "intent": "group_count" | "group_aggregate" | "comparison" | "distribution" | "table",\n'
    '  "group_by": "<categorical field>",\n'
    '  "second_group_by": "<categorical field or null>",   // for side-by-side comparison / cross-tab\n'
    '  "filter_field": "<categorical field to restrict rows by, or null>",\n'
    '  "filter_values": ["<value>", ...],   // ONLY rows whose filter_field is one of these are analyzed\n'
    '  "aggregate": "count" | "avg" | "min" | "max" | "sum" | null,\n'
    '  "aggregate_field": "<numeric field or null>",\n'
    '  "chart_type": "bar" | "pie" | "line" | "scatter" | "histogram" | "none",\n'
    '  "show_table": true | false,\n'
    '  "title": "<short chart/table title>"\n'
    '}\n'
    "Guidance: 'compare X per Y' or 'X by Y and Z' -> comparison with group_by=Y, second_group_by=Z. "
    "'distribution of <numeric>' -> distribution/histogram on that field. "
    "Counts per category -> group_count with chart_type=bar (or pie for a share-of-total question).\n"
    "IMPORTANT — when the user names SPECIFIC items/categories to compare or focus on "
    "(e.g. 'compare pharmacies and schools', 'only universities and hospitals'), you MUST set "
    "filter_field to the field holding those values (usually 'category' or '_dataset') and "
    "filter_values to EXACTLY those values, matching the sample values shown (map plural/lowercase "
    "to the real value, e.g. 'pharmacies' -> 'Pharmacy'). Use that same field as second_group_by so "
    "the comparison is between just those items. Example: 'compare pharmacies and schools per district' "
    "-> group_by='district', second_group_by='category', filter_field='category', "
    "filter_values=['Pharmacy','School'].\n"
    "Use ONLY field names from the provided list. Prefer a bar chart plus a table unless the question "
    "clearly wants only one. Return ONLY the JSON."
)


def _plan(question: str, profile: dict, multi_dataset: bool) -> dict:
    cats = _categorical_fields(profile)
    nums = _numeric_fields(profile)
    lines = []
    for k in cats:
        vals = ", ".join(map(str, profile[k]["distinct"][:8])) or "…"
        lines.append(f'- {k} (categorical; e.g. {vals})')
    for k in nums:
        lines.append(f'- {k} (numeric)')
    note = ("\nMultiple datasets are loaded; the field '_dataset' identifies which layer a row "
            "belongs to — use it as a group_by to compare datasets.\n") if multi_dataset else ""
    user = f"Question: {question}\n\nAvailable fields:\n" + "\n".join(lines) + note

    try:
        out = get_llm().complete_json(
            [{"role": "system", "content": _PLAN_SYSTEM}, {"role": "user", "content": user}]
        )
    except (LLMError, Exception) as e:  # noqa: BLE001
        log.warning("analytics: planning LLM failed (%s) — using heuristic plan", e)
        out = {}
    return _sanitize_plan(out, cats, nums, question, profile)


def _norm_forms(word: str) -> set[str]:
    """Case-folded singular/plural variants of a word, for lenient matching
    ('pharmacies' <-> 'Pharmacy', 'schools' <-> 'School')."""
    w = str(word).strip().lower()
    forms = {w}
    if w.endswith("ies"):
        forms.add(w[:-3] + "y")          # pharmacies -> pharmacy
    if w.endswith("es"):
        forms.add(w[:-2])                # boxes -> box
    if w.endswith("s"):
        forms.add(w[:-1])                # schools -> school
    forms.add(w + "s")
    if w.endswith("y"):
        forms.add(w[:-1] + "ies")        # pharmacy -> pharmacies
    return forms


def _match_values(requested: list, allowed: list[str]) -> list[str]:
    """Map requested filter values to the actual field values (case- and
    plural-insensitive), preserving the real casing of the matched value."""
    allowed_forms = [(a, _norm_forms(a)) for a in allowed]
    out: list[str] = []
    for v in requested or []:
        rf = _norm_forms(v)
        hit = next((a for a, af in allowed_forms if rf & af), None)
        if hit is not None and hit not in out:
            out.append(hit)
    return out


def _sanitize_plan(plan: dict, cats: list[str], nums: list[str], question: str, profile: dict) -> dict:
    """Repair the LLM plan against the real fields; provide a sensible fallback."""
    plan = plan if isinstance(plan, dict) else {}

    def valid_cat(f):
        return f if f in cats else None

    def valid_num(f):
        return f if f in nums else None

    # Filter: restrict rows to specific values of a categorical field. The raw
    # requested values are kept here; they are matched to the real field values in
    # analyze() (which sees every row, not just the sampled distinct values).
    filter_field = valid_cat(plan.get("filter_field"))
    fv = plan.get("filter_values")
    filter_values = [str(v) for v in fv] if isinstance(fv, list) else []
    if not filter_values:
        filter_field = None

    group_by = valid_cat(plan.get("group_by"))
    if not group_by:
        # fallback: first categorical field that isn't an id-like unique key
        group_by = next((c for c in cats if c not in ("id", "_dataset")), (cats[0] if cats else None))
    second = valid_cat(plan.get("second_group_by"))
    if second == group_by:
        second = None

    aggregate = plan.get("aggregate")
    aggregate = aggregate if aggregate in ("count", "avg", "min", "max", "sum") else "count"
    agg_field = valid_num(plan.get("aggregate_field"))
    if aggregate != "count" and not agg_field:
        aggregate = "count"  # no numeric field to aggregate

    intent = plan.get("intent")
    if intent not in ("group_count", "group_aggregate", "comparison", "distribution", "table"):
        intent = "comparison" if second else ("group_aggregate" if aggregate != "count" else "group_count")

    chart_type = plan.get("chart_type")
    if chart_type not in ("bar", "pie", "line", "scatter", "histogram", "none"):
        chart_type = "histogram" if intent == "distribution" else "bar"

    if intent == "distribution" and not agg_field:
        agg_field = nums[0] if nums else None
        if not agg_field:
            intent, chart_type = "group_count", "bar"

    return {
        "intent": intent,
        "group_by": group_by,
        "second_group_by": second,
        "filter_field": filter_field,
        "filter_values": filter_values,
        "aggregate": aggregate,
        "aggregate_field": agg_field,
        "chart_type": chart_type,
        "show_table": bool(plan.get("show_table", True)),
        "title": str(plan.get("title") or "").strip() or _default_title(intent, group_by, second, aggregate, agg_field),
    }


def _default_title(intent, group_by, second, aggregate, agg_field) -> str:
    if intent == "distribution" and agg_field:
        return f"Distribution of {agg_field}"
    if second:
        return f"{group_by} by {second}"
    if aggregate != "count" and agg_field:
        return f"{aggregate.title()} of {agg_field} per {group_by}"
    return f"Count per {group_by}"


# --------------------------------------------------------------------------- #
# deterministic execution
# --------------------------------------------------------------------------- #
def _agg(values: list[float], how: str) -> float:
    if not values:
        return 0.0
    if how == "avg":
        return round(sum(values) / len(values), 4)
    if how == "min":
        return min(values)
    if how == "max":
        return max(values)
    if how == "sum":
        return round(sum(values), 4)
    return float(len(values))


def _group(rows, group_by, aggregate, agg_field):
    """Ordered [(label, value)] grouped by ``group_by``."""
    buckets: dict[str, list] = {}
    order: list[str] = []
    for r in rows:
        key = r.get(group_by)
        key = "(blank)" if key is None or key == "" else str(key)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        if aggregate == "count":
            buckets[key].append(1)
        else:
            n = _num(r.get(agg_field))
            if n is not None:
                buckets[key].append(n)
    pairs = [(k, _agg(buckets[k], aggregate)) for k in order]
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    return pairs


def _crosstab(rows, gb1, gb2, aggregate, agg_field):
    """Cross-tabulate gb1 (rows) x gb2 (columns)."""
    col_order: list[str] = []
    cells: dict[str, dict[str, list]] = {}
    row_order: list[str] = []
    for r in rows:
        a = r.get(gb1); a = "(blank)" if a in (None, "") else str(a)
        b = r.get(gb2); b = "(blank)" if b in (None, "") else str(b)
        if a not in cells:
            cells[a] = {}; row_order.append(a)
        if b not in col_order:
            col_order.append(b)
        cells[a].setdefault(b, [])
        if aggregate == "count":
            cells[a][b].append(1)
        else:
            n = _num(r.get(agg_field))
            if n is not None:
                cells[a][b].append(n)
    matrix = {a: {b: _agg(cells[a].get(b, []), aggregate) for b in col_order} for a in row_order}
    # order rows by total desc
    row_order.sort(key=lambda a: sum(matrix[a].values()), reverse=True)
    return row_order, col_order, matrix


def _histogram(rows, field, bins=10):
    vals = [n for n in (_num(r.get(field)) for r in rows) if n is not None]
    if not vals:
        return [], []
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return [f"{lo:g}"], [len(vals)]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        idx = min(int((v - lo) / width), bins - 1)
        counts[idx] += 1
    labels = [f"{lo + i * width:.1f}–{lo + (i + 1) * width:.1f}" for i in range(bins)]
    return labels, counts


_VIZ_LABELS = {
    "bar": "Bar chart", "hbar": "Horizontal bar", "stacked_bar": "Stacked bar",
    "pie": "Pie chart", "line": "Line chart", "scatter": "Scatter", "histogram": "Histogram",
}


def _viz_options(base_type: str, num_series: int, num_cats: int) -> list[dict]:
    """All suitable visualization types for this result's shape, recommended first.

    The dataset is already computed — these are just alternate renderings the UI
    can switch between without re-querying."""
    if base_type == "histogram":
        vals = ["histogram", "line"]
    elif num_series > 1:                       # comparison / cross-tab
        vals = ["bar", "stacked_bar", "hbar", "line"]
    else:                                       # single-series group
        vals = ["bar", "hbar", "line"] + (["pie"] if 2 <= num_cats <= 12 else [])
    vals = [base_type] + [v for v in vals if v != base_type]
    out, seen = [], set()
    for v in vals:
        if v not in seen:
            seen.add(v)
            out.append({"value": v, "label": _VIZ_LABELS.get(v, v)})
    return out


# Explicit chart type named in the question (checked most-specific first).
_CHART_REQUEST = [
    (re.compile(r"\b(pie|donut|doughnut)\s*(chart|graph)?\b", re.I), "pie"),
    (re.compile(r"\bhorizontal\s+bar\b|\bh-?bar\b", re.I), "hbar"),
    (re.compile(r"\bstacked(\s+bar)?\b", re.I), "stacked_bar"),
    (re.compile(r"\bhistogram\b", re.I), "histogram"),
    (re.compile(r"\bscatter(\s*plot)?\b", re.I), "scatter"),
    (re.compile(r"\bline\s+(chart|graph|plot)\b|\bas\s+a\s+line\b|\btrend\b", re.I), "line"),
    (re.compile(r"\b(bar|column)\s+(chart|graph|plot)\b|\bas\s+a\s+bar\b", re.I), "bar"),
]


def _requested_chart_type(question: str) -> str | None:
    """Return a chart type the user explicitly asked for, else None."""
    for rx, t in _CHART_REQUEST:
        if rx.search(question or ""):
            return t
    return None


def _apply_requested_chart(chart: dict, requested: str) -> None:
    """Make ``requested`` the displayed-first type when it fits the data shape.

    The dataset is unchanged — this only reorders which visualization is default
    and guarantees it appears in the switcher."""
    if not chart or not requested:
        return
    num_series = len(chart.get("series") or [])
    # Pie can only show a single series; skip it for multi-series comparisons.
    if requested == "pie" and num_series > 1:
        return
    chart["type"] = requested
    opts = chart.get("viz_options") or []
    if requested not in [o["value"] for o in opts]:
        opts.insert(0, {"value": requested, "label": _VIZ_LABELS.get(requested, requested)})
    else:
        opts.sort(key=lambda o: 0 if o["value"] == requested else 1)   # stable: move to front
    chart["viz_options"] = opts


def _execute(plan: dict, rows: list[dict]) -> dict:
    intent = plan["intent"]
    gb, gb2 = plan["group_by"], plan["second_group_by"]
    agg, agg_field = plan["aggregate"], plan["aggregate_field"]
    chart_type = plan["chart_type"]
    title = plan["title"]
    ylabel = "count" if agg == "count" else f"{agg} of {agg_field}"

    chart = None
    table = None

    if intent == "distribution" and agg_field:
        labels, counts = _histogram(rows, agg_field)
        chart = {"type": "histogram", "title": title, "x_label": agg_field, "y_label": "count",
                 "x": labels, "series": [{"name": "count", "y": counts}],
                 "viz_options": _viz_options("histogram", 1, len(labels))}
        table = {"columns": [f"{agg_field} range", "count"],
                 "rows": [[l, c] for l, c in zip(labels, counts)]}
        summary = f"Distribution of {agg_field} across {len(rows)} row(s), in {len(labels)} bin(s)."
        return {"summary": summary, "chart": chart, "table": table}

    if (intent == "comparison" or gb2) and gb and gb2:
        row_order, col_order, matrix = _crosstab(rows, gb, gb2, agg, agg_field)
        # grouped bar: one series per gb2 value
        series = [{"name": b, "y": [matrix[a][b] for a in row_order]} for b in col_order]
        base_type = chart_type if chart_type in ("bar", "line") else "bar"
        chart = {"type": base_type,
                 "title": title, "x_label": gb, "y_label": ylabel,
                 "x": row_order, "series": series,
                 "viz_options": _viz_options(base_type, len(col_order), len(row_order))}
        cols = [gb] + col_order + ["Total"]
        trows = []
        for a in row_order:
            vals = [matrix[a][b] for b in col_order]
            trows.append([a] + vals + [round(sum(vals), 4)])
        table = {"columns": cols, "rows": trows}
        summary = (f"{ylabel.title()} of '{gb}' broken down by '{gb2}' "
                   f"({len(row_order)}×{len(col_order)} groups).")
        return {"summary": summary, "chart": chart, "table": table}

    # group_count / group_aggregate
    pairs = _group(rows, gb, agg, agg_field)
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    ctype = chart_type if chart_type in ("bar", "pie", "line", "scatter") else "bar"
    chart = {"type": ctype, "title": title, "x_label": gb, "y_label": ylabel,
             "x": labels, "series": [{"name": ylabel, "y": values}],
             "viz_options": _viz_options(ctype, 1, len(labels))}
    table = {"columns": [gb, ylabel], "rows": [[l, v] for l, v in zip(labels, values)]}
    top = f"{labels[0]} ({values[0]:g})" if labels else "—"
    summary = f"{ylabel.title()} across {len(labels)} '{gb}' group(s); highest: {top}."
    return {"summary": summary, "chart": chart, "table": table}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def analyze(question: str, dataset_ids: list[str]) -> dict:
    """Answer an analytical question over the selected datasets' properties."""
    if not dataset_ids:
        raise AnalyticsError("Select at least one dataset to analyze.")
    with log_step(log, "analytics: load rows"):
        tables = fetch_dataset_tables(dataset_ids)
    rows = [r for t in tables for r in t["rows"]]
    if not rows:
        raise AnalyticsError("The selected dataset(s) have no rows to analyze.")

    profile = _field_profile(rows)
    plan = _plan(question, profile, multi_dataset=len(tables) > 1)
    log.info("analytics plan: %s", plan)
    if not plan.get("group_by"):
        raise AnalyticsError("No categorical field found to group by in the selected data.")

    # Apply the row filter (e.g. "compare pharmacies and schools" -> only those).
    # Match the requested values against the FULL set of values for that field.
    if plan.get("filter_field") and plan.get("filter_values"):
        ff = plan["filter_field"]
        allowed = sorted({str(r.get(ff)) for r in rows if r.get(ff) not in (None, "")})
        matched = _match_values(plan["filter_values"], allowed)
        if matched:
            keep = set(matched)
            rows = [r for r in rows if str(r.get(ff)) in keep]
            plan["filter_values"] = matched          # record the normalized values
            if not rows:
                raise AnalyticsError(
                    f"No rows match {matched} in '{ff}' for the selected dataset(s).")
        else:
            plan["filter_field"] = None              # couldn't match — analyze everything
            plan["filter_values"] = []

    result = _execute(plan, rows)
    # If the user explicitly named a chart type, show it first (and in the switcher).
    requested = _requested_chart_type(question)
    if requested and result.get("chart"):
        _apply_requested_chart(result["chart"], requested)
        plan["requested_chart_type"] = requested
    if plan.get("filter_values"):
        result["summary"] = (f"Filtered to {', '.join(plan['filter_values'])}. "
                             + result.get("summary", ""))
    result["plan"] = plan
    result["dataset_scope"] = [t["display_name"] for t in tables]
    result["row_count"] = len(rows)
    return result
