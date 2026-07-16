"""Unit tests for the SQL AST validator (defense-in-depth on generated SQL)."""

from validator import validate_sql


def test_allows_whitelisted_select():
    ok, _ = validate_sql(
        'SELECT name, ST_AsGeoJSON(wkb_geometry) FROM user_data_abc',
        allowed_tables={"user_data_abc"},
    )
    assert ok


def test_rejects_non_select():
    ok, reason = validate_sql("DROP TABLE user_data_abc", allowed_tables={"user_data_abc"})
    assert not ok


def test_rejects_unknown_table():
    ok, reason = validate_sql("SELECT * FROM secret_table", allowed_tables={"user_data_abc"})
    assert not ok
    assert "not in your available datasets" in reason


def test_rejects_multiple_statements():
    ok, _ = validate_sql(
        "SELECT * FROM user_data_abc; DROP TABLE user_data_abc",
        allowed_tables={"user_data_abc"},
    )
    assert not ok


def test_rejects_empty():
    ok, _ = validate_sql("", allowed_tables=set())
    assert not ok
