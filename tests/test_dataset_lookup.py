"""Unit tests for dataset-record formatting (structured knowledge tier).

The module imports the DB stack at load time, so the suite skips cleanly when
sqlalchemy/geoalchemy2 aren't installed.
"""

import pytest

pytest.importorskip("geoalchemy2", reason="DB stack not installed")

from agents.dataset_lookup import _format_record, _GEOM


def test_format_record_extracts_website_and_drops_geometry():
    row = {
        "name": "Bean House",
        "opening_hours": "Mo-Su 08:00-23:00",
        "website": "https://www.beanhouse.eg",
        "rating": "4.5",
        _GEOM: "0101000020E6...",
        "email": None,        # nulls dropped
    }
    record, website = _format_record("Cairo Cafes", row)
    assert website == "https://www.beanhouse.eg"
    assert "wkb_geometry" not in record
    assert "opening_hours: Mo-Su 08:00-23:00" in record
    assert "email" not in record          # null skipped
    assert record.startswith("[Cairo Cafes]")


def test_format_record_empty_when_all_null():
    record, website = _format_record("X", {"a": None, _GEOM: "..."})
    assert record == ""
    assert website is None
