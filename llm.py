import os
from dotenv import load_dotenv
from groq import Groq

from registry import get_datasets_by_ids

load_dotenv()

client = Groq(api_key=os.getenv("API_KEY"))

MODEL = "llama-3.3-70b-versatile"

RULES = """
Rules:
- ONLY SELECT queries allowed
- ALWAYS include ALL of these in SELECT:
    name,
    tags,
    ST_AsGeoJSON(wkb_geometry) AS geometry
  (if the table has no "name" or "tags" column, omit it, but ALWAYS include the geometry line)
- Only reference the tables listed below. Never invent a table or column.
- When a user asks for items relative to a named landmark/place that is itself one of
  the listed tables (e.g. a park, a station), find that landmark's geometry dynamically
  using a subquery from the matching table, filtering by name with ILIKE.
- No INSERT / DELETE / UPDATE / DROP / ALTER / TRUNCATE
- Use ILIKE for text search on name/text columns
- For distance queries:
    ST_DWithin(wkb_geometry::geography, ref::geography, meters)
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


def build_schema_text(dataset_ids: list[str]) -> str:
    """Builds the natural-language schema block the LLM sees, from whichever
    datasets the user has selected for this chat. Replaces the old hardcoded
    Egypt-specific SCHEMA string."""
    datasets = get_datasets_by_ids(dataset_ids)
    if not datasets:
        return ""

    lines = ["Tables (each also has an id column and a wkb_geometry column, SRID 4326):"]
    for ds in datasets:
        cols = ds.get("columns") or []
        col_desc = ", ".join(
            f"{c['column_name']} ({c['data_type']}" + (f" — {c['description']}" if c.get("description") else "") + ")"
            for c in cols
        ) or "(no additional columns)"
        lines.append(
            f'- "{ds["table_name"]}"  [display name: "{ds["display_name"]}", geometry type: {ds["geometry_type"]}]\n'
            f"    columns: {col_desc}"
        )
    return "\n".join(lines)


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
    return sql
