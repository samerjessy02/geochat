from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry
import os, json
from dotenv import load_dotenv
load_dotenv()

engine = create_engine(os.getenv("DB_URL"))

def run_query(sql: str) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row._mapping) for row in result]  #returns list of dicts


def ensure_postgis():
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis")) 