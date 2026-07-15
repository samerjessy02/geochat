"""
ingest.py — turn a user-uploaded GeoJSON or lat/lon CSV into a new PostGIS
table, and register it in the dataset registry.

Supported inputs:
  - .geojson / .json  (any geometry type)
  - .csv               (must contain recognizable lat/lon columns)
"""

import io
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from db import engine
from registry import new_table_name, register_dataset

LAT_NAMES = {"lat", "latitude", "y"}
LON_NAMES = {"lon", "lng", "long", "longitude", "x"}


class IngestError(Exception):
    pass


def _detect_lat_lon(columns: list[str]) -> tuple[str, str]:
    lower = {c.lower(): c for c in columns}
    lat_col = next((lower[c] for c in LAT_NAMES if c in lower), None)
    lon_col = next((lower[c] for c in LON_NAMES if c in lower), None)
    if not lat_col or not lon_col:
        raise IngestError(
            "Could not find latitude/longitude columns in CSV. "
            "Expected columns named like 'lat'/'lon' (or 'latitude'/'longitude')."
        )
    return lat_col, lon_col


def load_geojson(raw_bytes: bytes) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(io.BytesIO(raw_bytes))
    except Exception as e:
        raise IngestError(f"Could not parse GeoJSON: {e}")
    if gdf.empty:
        raise IngestError("GeoJSON contains no features.")
    return gdf


def load_csv(raw_bytes: bytes) -> gpd.GeoDataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise IngestError(f"Could not parse CSV: {e}")
    if df.empty:
        raise IngestError("CSV contains no rows.")
    lat_col, lon_col = _detect_lat_lon(list(df.columns))
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf

def ingest_file(filename: str, raw_bytes: bytes, display_name: str) -> dict:
    """Parses the file, writes it to a new Postgres table, registers it.
    Returns {dataset_id, table_name, geometry_type, row_count, columns: [{column_name, data_type}]}
    for the caller to hand back to the user so they can fill in descriptions.
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext in ("geojson", "json"):
        gdf = load_geojson(raw_bytes)
    elif ext == "csv":
        gdf = load_csv(raw_bytes)
    else:
        raise IngestError(
            f"Unsupported file type: .{ext}. Upload a .geojson or .csv file."
        )

    # Ensure CRS is WGS84
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    # Normalize geometry column name for PostGIS
    gdf = gdf.rename_geometry("wkb_geometry")

    # ==========================================
    # STEP 1: Sanitize all column names first
    # ==========================================
    sanitized_columns = []
    for col in gdf.columns:
        if col == "wkb_geometry":
            sanitized_columns.append(col)
            continue

        clean = "".join(
            c.lower() if c.isalnum() else "_"
            for c in col
        ).strip("_")

        clean = clean or "col"
        sanitized_columns.append(clean)

    # ==========================================
    # STEP 2: Deduplicate and Normalize IDs
    # ==========================================
    final_columns = []
    id_seen = False
    source_id_counter = 0
    seen_counts = {}  # Tracks non-ID duplicates (e.g., duplicate 'name' cols)

    def is_id_variant(col_name: str) -> bool:
        # Matches 'id' or any standard numeric suffix variation like 'id_1'
        if col_name == "id":
            return True
        if col_name.startswith("id_") and col_name[3:].isdigit():
            return True
        return False

    for col in sanitized_columns:
        if col == "wkb_geometry":
            final_columns.append(col)
            continue

        if is_id_variant(col):
            if not id_seen:
                final_columns.append("id")
                id_seen = True
            else:
                if source_id_counter == 0:
                    final_columns.append("source_id")
                else:
                    final_columns.append(f"source_id_{source_id_counter}")
                source_id_counter += 1
        else:
            # Handle generic duplicate columns (e.g., 'addr_city', 'addr_city' -> 'addr_city_1')
            if col in seen_counts:
                seen_counts[col] += 1
                final_columns.append(f"{col}_{seen_counts[col]}")
            else:
                seen_counts[col] = 0
                final_columns.append(col)

    # Apply the final, perfectly unique and clean column list
    gdf.columns = final_columns

    # ==========================================
    # Database Write & Registration
    # ==========================================
    geometry_types = gdf.geometry.geom_type.unique().tolist()
    geometry_type = (
        geometry_types[0]
        if len(geometry_types) == 1
        else "GEOMETRY"
    )

    table_name = new_table_name()

    print("Table:", table_name)
    print("Columns:", gdf.columns.tolist())
    print("Index name:", gdf.index.name)
    print(gdf.head())

    try:
        gdf.to_postgis(
            table_name,
            engine,
            if_exists="replace",
            index=False
        )
    except Exception as e:
        raise IngestError(
            f"Failed to write dataset to database: {e}"
        )

    dataset_id = register_dataset(
        table_name=table_name,
        display_name=display_name,
        geometry_type=geometry_type,
        srid=4326,
    )

    columns = [
        {
            "column_name": c,
            "data_type": str(gdf[c].dtype)
        }
        for c in gdf.columns
        if c != "wkb_geometry"
    ]

    return {
        "dataset_id": dataset_id,
        "table_name": table_name,
        "geometry_type": geometry_type,
        "row_count": len(gdf),
        "columns": columns,
    }