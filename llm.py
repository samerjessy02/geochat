import os
import re
from dotenv import load_dotenv
from groq import Groq

from registry import get_datasets_by_ids

load_dotenv()

client = Groq(api_key=os.getenv("API_KEY"))

MODEL = "openai/gpt-oss-20b"

# Added a dedicated section for Semantic Mapping to guide the LLM's query generation
RULES = """
Rules:
- ONLY SELECT queries allowed
- DYNAMIC SELECT CLAUSE:
    For each table referenced in your query, inspect its "SUGGESTED SELECT COLUMNS" in the schema below.
    ALWAYS include those suggested columns in your SELECT clause as a baseline, PLUS any specific columns requested by the user's query (if they exist in the table schema).
    ALWAYS include the geometry transformation:
    ST_AsGeoJSON(wkb_geometry) AS geometry

- SEMANTIC MAPPING & VALUE TRANSLATION (CRITICAL):
    Users write queries in casual, everyday language, but the database columns and values often use technical abbreviations or OpenStreetMap standards.
    You MUST translate user concepts into their schema equivalents:
      * Column Mapping: If the user asks for a feature (e.g., "wifi"), and there is no direct "wifi" column, inspect the schema for relevant columns like "wlan", "internet_access", "network", or "access".
      * Value Mapping: When filtering values in a WHERE clause, expand the search using ILIKE with OR statements to capture common synonyms and technical abbreviations:
        - "wifi" / "wireless internet" -> Search for 'wlan', 'wifi', 'internet', or 'yes' (e.g., `WHERE "network:access" ILIKE '%wlan%' OR "network:access" ILIKE '%wifi%'`)
        - "restroom" / "bathroom" -> Search for 'toilets', 'restroom', 'washroom'
        - "metro" / "subway" -> Search for 'subway', 'light_rail', 'underground', 'railway'
        - "cafe" / "coffee" -> Search for 'cafe', 'coffee_shop'
        - "restaurant" / "food" -> Search for 'restaurant', 'fast_food', 'food_court', 'cafe'

- FILTERING DISCIPLINE (CRITICAL):
    Convert EVERY constraint in the user's question into a WHERE filter. Do not return
    all rows unless the user explicitly asks for "all"/"every"/"the whole dataset".
      * a category/type (cafe, hospital, museum, restaurant) -> filter the type/amenity/category
        column with ILIKE '%value%' (expand synonyms with OR per the mapping above).
      * a street/district/area/city/landmark -> filter the matching address or name column with
        ILIKE '%value%' (e.g. "addr:street" ILIKE '%tahrir%').
      * an attribute (wifi, wheelchair, rating, delivery) -> filter the matching column.
    Combine multiple constraints with AND. Prefer broad ILIKE '%term%' over exact equality.

- DESCRIPTIONS ARE HINTS ONLY:
    Any "— description" text next to a column is only a hint to help you choose columns.
    Rely on the actual column NAMES and real data values, never on the wording of a
    description. If a description seems wrong or vague, ignore it and use the column name.

- BOOLEAN PRECEDENCE (CRITICAL):
    SQL evaluates AND before OR. Whenever you combine an AND filter with a group of OR
    synonyms, you MUST wrap the OR group in parentheses, or the AND filter is ignored.
      WRONG: WHERE amenity ILIKE '%cafe%' AND description ILIKE '%a%' OR description ILIKE '%b%'
      RIGHT: WHERE amenity ILIKE '%cafe%' AND (description ILIKE '%a%' OR description ILIKE '%b%')

- COLUMN QUOTING:
    Always wrap column names containing special characters (such as colons ":", e.g., "name:en", "name:ar") in double quotes in the SQL query.
- Only reference the tables listed below. Never invent a table or column.
- When a user asks for items relative to a named landmark/place that is itself one of
  the listed tables (e.g. a park, a station), find that landmark's geometry dynamically
  using a subquery from the matching table, filtering by the most appropriate name/localized name column with ILIKE.
- No INSERT / DELETE / UPDATE / DROP / ALTER / TRUNCATE
- Use ILIKE for text search on name/text columns

- SPATIAL OPERATIONS (PostGIS) — CRITICAL:
    When the question is about a spatial relationship between features, use the
    patterns below. wkb_geometry is SRID 4326 (degrees), so ALWAYS cast to
    ::geography when you need distances or buffers in METERS. Resolve a named
    reference place (a landmark/area/road/river) to its geometry with a subquery
    against the table that holds it, filtered by its name column with ILIKE.
    Keep ST_AsGeoJSON(t.wkb_geometry) AS geometry for the RESULT features, and
    add the extra columns noted so the map can draw the overlay.
    Alias the main/result table as t.

    TABLE NAMES (#1 cause of failure — read carefully): a spatial query touches
    MORE THAN ONE table (the result table AND one or more reference tables in
    JOINs/subqueries). EVERY one of them — in FROM, JOIN, and every subquery —
    MUST be the exact machine table_name from the schema above (they look like
    user_data_xxxxxxxxxxxx). NEVER write a display name such as "hospitals",
    "schools", "universities" or "flood_zones" as a table name. Map the user's
    everyday word (hospital, school, university, park, flood zone) to the table
    whose *display name* matches, and use that table's quoted "user_data_..."
    name. The placeholders below (e.g. <PHARMACIES_TABLE>) mean "substitute the
    real user_data_ table name" — never emit the placeholder or the display name.

    * RADIUS — "within 500 meters of X", "near X", "around X", "close to X":
        WHERE ST_DWithin(
                  t.wkb_geometry::geography,
                  (SELECT wkb_geometry FROM <ref_table> WHERE <name> ILIKE '%X%' LIMIT 1)::geography,
                  <meters>)
        ALSO SELECT (so the circle + distance can be drawn):
            ROUND(ST_Distance(t.wkb_geometry::geography,
                  (SELECT wkb_geometry FROM <ref_table> WHERE <name> ILIKE '%X%' LIMIT 1)::geography)::numeric, 1) AS distance_m,
            ST_AsGeoJSON((SELECT wkb_geometry FROM <ref_table> WHERE <name> ILIKE '%X%' LIMIT 1)) AS reference_geometry,
            <meters> AS search_radius_m

    * NEAREST — "nearest / closest X to Y" (optionally "N nearest"):
        ORDER BY ST_Distance(t.wkb_geometry::geography,
                 (SELECT wkb_geometry FROM <ref_table> WHERE <name> ILIKE '%Y%' LIMIT 1)::geography)
        LIMIT <n or 1>
        ALSO SELECT (so a connecting line + distance can be drawn):
            ROUND(ST_Distance(t.wkb_geometry::geography,
                  (SELECT wkb_geometry FROM <ref_table> WHERE <name> ILIKE '%Y%' LIMIT 1)::geography)::numeric, 1) AS distance_m,
            ST_AsGeoJSON((SELECT wkb_geometry FROM <ref_table> WHERE <name> ILIKE '%Y%' LIMIT 1)) AS reference_geometry

    * INSIDE POLYGON — "X inside/within <area>":
        WHERE ST_Contains(
                  (SELECT wkb_geometry FROM <area_table> WHERE <name> ILIKE '%area%' LIMIT 1),
                  t.wkb_geometry)
        ALSO SELECT (so the containing polygon is highlighted):
            ST_AsGeoJSON((SELECT wkb_geometry FROM <area_table> WHERE <name> ILIKE '%area%' LIMIT 1)) AS reference_geometry

    * INTERSECTS — "X that intersect/overlap/touch Y":
        FROM <t_table> t JOIN <y_table> r
             ON ST_Intersects(t.wkb_geometry, r.wkb_geometry)
        (add WHERE r.<name> ILIKE '%Y%' if Y is a specific feature)
        ALSO SELECT ST_AsGeoJSON(r.wkb_geometry) AS reference_geometry

    * BUFFER — "within 1 km of the Nile / a road / a district":
        Same as RADIUS but the reference is a line/area feature. ST_DWithin on
        ::geography is index-friendly and equivalent to buffering, so prefer it:
        WHERE ST_DWithin(t.wkb_geometry::geography,
                  (SELECT wkb_geometry FROM <ref_table> WHERE <name> ILIKE '%X%' LIMIT 1)::geography, <meters>)
        ALSO SELECT reference_geometry (as above) AND <meters> AS search_radius_m

    Notes: keep it ONE SELECT statement. Only reference the listed tables. If the
    reference place cannot be matched to a listed table, fall back to a normal
    attribute filter instead of inventing geometry.

    REFERENCE LABEL (optional but preferred): whenever you output
    reference_geometry AND the reference table has a name/category column, also
    output the reference feature's name and category so the map can label it:
        ... AS reference_name, ... AS reference_category
    e.g. for INTERSECTS add: r.name AS reference_name, r.category AS reference_category;
    for a subquery reference add:
        (SELECT name FROM <ref_table> WHERE <name> ILIKE '%X%' LIMIT 1) AS reference_name,
        (SELECT category FROM <ref_table> WHERE <name> ILIKE '%X%' LIMIT 1) AS reference_category

- For simple distance filters without a named landmark, still use
    ST_DWithin(wkb_geometry::geography, ref::geography, meters).

- DENSITY / HEATMAP / CLUSTERING requests ("density", "heatmap", "where are X
    concentrated", "hotspots", "clustered X"): do NOT aggregate, GROUP BY, or
    count. Return ALL matching rows with their point geometry
    (ST_AsGeoJSON(wkb_geometry) AS geometry), filtered by category as usual — the
    MAP renders the density/heatmap/clusters from the raw points. This must stay a
    plain SELECT of individual features, never an aggregate.

EXAMPLES (patterns — adapt table/column names to the actual schema above):
- "show cafes on Tahrir Street"
    -> WHERE amenity ILIKE '%cafe%' AND "addr:street" ILIKE '%tahrir%'
- "hospitals in Downtown Cairo"
    -> WHERE amenity ILIKE '%hospital%' AND ("addr:district" ILIKE '%downtown%' OR "addr:city" ILIKE '%cairo%')
  (In the examples below, <..._TABLE> is a placeholder for the real
   "user_data_..." table_name from the schema — substitute it; do not emit it.)
- "pharmacies within 500 meters of Cairo University"
    -> SELECT <suggested cols>, ST_AsGeoJSON(t.wkb_geometry) AS geometry,
              ROUND(ST_Distance(t.wkb_geometry::geography, (SELECT wkb_geometry FROM <UNIVERSITIES_TABLE> WHERE name ILIKE '%cairo university%' LIMIT 1)::geography)::numeric,1) AS distance_m,
              ST_AsGeoJSON((SELECT wkb_geometry FROM <UNIVERSITIES_TABLE> WHERE name ILIKE '%cairo university%' LIMIT 1)) AS reference_geometry,
              500 AS search_radius_m
       FROM <PHARMACIES_TABLE> t
       WHERE ST_DWithin(t.wkb_geometry::geography, (SELECT wkb_geometry FROM <UNIVERSITIES_TABLE> WHERE name ILIKE '%cairo university%' LIMIT 1)::geography, 500)
- "nearest hospital to Downtown School"
    -> SELECT <suggested cols>, ST_AsGeoJSON(t.wkb_geometry) AS geometry,
              ROUND(ST_Distance(t.wkb_geometry::geography, (SELECT wkb_geometry FROM <SCHOOLS_TABLE> WHERE name ILIKE '%downtown school%' LIMIT 1)::geography)::numeric,1) AS distance_m,
              ST_AsGeoJSON((SELECT wkb_geometry FROM <SCHOOLS_TABLE> WHERE name ILIKE '%downtown school%' LIMIT 1)) AS reference_geometry
       FROM <HOSPITALS_TABLE> t
       ORDER BY ST_Distance(t.wkb_geometry::geography, (SELECT wkb_geometry FROM <SCHOOLS_TABLE> WHERE name ILIKE '%downtown school%' LIMIT 1)::geography)
       LIMIT 1
- "schools inside Nasr City"
    -> SELECT <suggested cols>, ST_AsGeoJSON(t.wkb_geometry) AS geometry,
              ST_AsGeoJSON((SELECT wkb_geometry FROM <NEIGHBORHOODS_TABLE> WHERE name ILIKE '%nasr city%' LIMIT 1)) AS reference_geometry,
              (SELECT name FROM <NEIGHBORHOODS_TABLE> WHERE name ILIKE '%nasr city%' LIMIT 1) AS reference_name
       FROM <SCHOOLS_TABLE> t
       WHERE ST_Contains((SELECT wkb_geometry FROM <NEIGHBORHOODS_TABLE> WHERE name ILIKE '%nasr city%' LIMIT 1), t.wkb_geometry)
- "parks that intersect flood zones"
    -> SELECT <suggested cols>, ST_AsGeoJSON(t.wkb_geometry) AS geometry,
              ST_AsGeoJSON(r.wkb_geometry) AS reference_geometry,
              r.name AS reference_name, r.category AS reference_category
       FROM <PARKS_TABLE> t JOIN <FLOOD_ZONES_TABLE> r ON ST_Intersects(t.wkb_geometry, r.wkb_geometry)
- "show all museums"  (explicit "all" -> no WHERE filter)
    -> (select suggested columns + geometry, no WHERE)
"""

SYSTEM_PROMPT_HEADER = """
You are a PostgreSQL + PostGIS expert working over a set of user-uploaded geospatial datasets.

You convert natural language into SAFE, READ-ONLY SQL queries.

STRICT RULES:
- Return ONLY SQL (no explanation, no markdown)
- Only use the tables and columns given in the schema below
- Never invent tables or columns
- Always ensure the query is valid PostGIS
"""


def get_suggested_select_columns(columns: list[dict]) -> list[str]:
    """
    Analyzes the available columns for a dynamic dataset and returns a list of 
    the best descriptive and identifying columns (names, localized names, titles, tags).
    """
    suggested = []
    has_primary_name = False
    
    # 1. Look for a primary "name" column
    for col in columns:
        name = col["column_name"]
        if name == "name":
            has_primary_name = True
            suggested.append("name")
            break
            
    # 2. Look for localized names (e.g., name:en, name:ar) or common title fallbacks
    for col in columns:
        name = col["column_name"]
        if name != "name" and ("name:" in name or name in ["name_en", "name_ar", "title", "label", "display_name"]):
            suggested.append(name)
            
    # 3. If absolutely no standard name column exists, look for any column containing 'name'
    if not has_primary_name and not suggested:
        for col in columns:
            name = col["column_name"]
            if "name" in name.lower() and name not in suggested:
                suggested.append(name)

    # 4. Include structural descriptors/tags if present
    for col in columns:
        name = col["column_name"]
        if name in ["tags", "type", "category", "amenity", "classification", "description"] and name not in suggested:
            suggested.append(name)
            
    return suggested


def build_schema_text(dataset_ids: list[str]) -> str:
    """Builds the natural-language schema block the LLM sees. Dynamically analyzes
    the schema of each selected dataset to tell the LLM exactly what columns to select."""
    datasets = get_datasets_by_ids(dataset_ids)
    if not datasets:
        return ""

    lines = ["Tables (each also has an id column and a wkb_geometry column, SRID 4326):"]
    for ds in datasets:
        cols = ds.get("columns") or []
        col_desc = ", ".join(
            f"{c['column_name']} ({c['data_type']}"
            + (f" — {c['description']}" if c.get("description") else "")
            + (" [PRIMARY KEY]" if c.get("is_primary_key") else "")
            + (f" [references {c['foreign_key']}]" if c.get("foreign_key") else "")
            + ")"
            for c in cols
        ) or "(no additional columns)"
        
        # Programmatically analyze the columns of the uploaded dataset
        suggested_cols = get_suggested_select_columns(cols)
        
        # Quote columns containing special characters (like "name:en") so the LLM writes valid SQL
        formatted_suggested = [f'"{c}"' if ":" in c or " " in c else c for c in suggested_cols]
        suggested_str = ", ".join(formatted_suggested) if formatted_suggested else "None"
        
        lines.append(
            f'- "{ds["table_name"]}"  [display name: "{ds["display_name"]}", geometry type: {ds["geometry_type"]}]\n'
            f"    columns: {col_desc}\n"
            f"    -> SUGGESTED SELECT COLUMNS: {suggested_str}, ST_AsGeoJSON(wkb_geometry) AS geometry"
        )
    return "\n".join(lines)


# Table references only appear right after FROM or JOIN, so matching there lets
# us rewrite table names without ever touching columns, functions, or string
# literals (a plain word-replace would corrupt e.g. ILIKE '%school%').
_TABLE_REF_RE = re.compile(r'\b(FROM|JOIN)\s+("?)([A-Za-z_][A-Za-z0-9_]*)\2', re.IGNORECASE)


def remap_table_names(sql: str, dataset_ids: list[str]) -> str:
    """Deterministic safety net for multi-table spatial SQL.

    The model reliably uses a dataset's real machine ``table_name``
    (``user_data_...``) for single-table queries, but in spatial JOIN/subquery
    patterns it sometimes writes the dataset's *display name* (e.g. ``schools``,
    ``hospitals``) as the table instead — which the validator then rejects as
    "not in your available datasets". Here we rewrite any FROM/JOIN table token
    that matches a selected dataset's display name to that dataset's real
    ``table_name``, so the query is valid regardless of what the model emitted.
    """
    datasets = get_datasets_by_ids(dataset_ids)
    if not datasets:
        return sql
    real = {d["table_name"] for d in datasets}
    by_display: dict[str, str] = {}
    for d in datasets:
        dn = (d.get("display_name") or "").strip().lower()
        if dn and dn not in by_display:
            by_display[dn] = d["table_name"]

    def _sub(m: "re.Match") -> str:
        kw, name = m.group(1), m.group(3)
        if name in real:                       # already a real table name
            return m.group(0)
        repl = by_display.get(name.lower())     # display name -> real table name
        return f'{kw} "{repl}"' if repl else m.group(0)

    return _TABLE_REF_RE.sub(_sub, sql)


def generate_sql(user_query: str, dataset_ids: list[str]) -> str:
    schema_text = build_schema_text(dataset_ids)
    if not schema_text:
        raise ValueError("No datasets selected. Upload or select at least one dataset before chatting.")

    system_prompt = SYSTEM_PROMPT_HEADER + "\n" + schema_text + "\n" + RULES

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.0,
    )

    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = remap_table_names(sql, dataset_ids)   # display-name -> real table_name safety net
    return sql