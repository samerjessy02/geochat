import os
from dotenv import load_dotenv
from groq import Groq

from registry import get_datasets_by_ids

load_dotenv()

client = Groq(api_key=os.getenv("API_KEY"))

MODEL = "llama-3.3-70b-versatile"

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

- COLUMN QUOTING:
    Always wrap column names containing special characters (such as colons ":", e.g., "name:en", "name:ar") in double quotes in the SQL query.
- Only reference the tables listed below. Never invent a table or column.
- When a user asks for items relative to a named landmark/place that is itself one of
  the listed tables (e.g. a park, a station), find that landmark's geometry dynamically
  using a subquery from the matching table, filtering by the most appropriate name/localized name column with ILIKE.
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
            f"{c['column_name']} ({c['data_type']}" + (f" — {c['description']}" if c.get("description") else "") + ")"
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