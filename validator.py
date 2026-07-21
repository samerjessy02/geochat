"""
validator.py — defense-in-depth check on LLM-generated SQL before execution.

Two checks:
  1. Statement-level: must be a single read-only SELECT (via sqlglot AST,
     not string matching — string matching on keywords is trivially bypassed).
  2. Table whitelist: every table referenced must be one of the tables the
     user actually has access to (the registered dataset table_names, plus
     any fixed system tables like layer_metadata). This matters a lot now
     that table names are dynamic/user-driven — never trust the LLM's SQL
     to only touch what it was told about in the prompt.
"""

import sqlglot
from sqlglot import exp

ALWAYS_ALLOWED_TABLES = {"layer_metadata"}

DISALLOWED_STATEMENTS = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter,
    exp.Create, exp.TruncateTable, exp.Grant,
)


def validate_sql(sql: str, allowed_tables: set[str] | None = None) -> tuple[bool, str]:
    if not sql or not sql.strip():
        return False, "Empty query."

    try:
        statements = sqlglot.parse(sql, read="postgres")
    except Exception as e:
        return False, f"Could not parse SQL: {e}"

    if len(statements) != 1:
        return False, "Only a single SQL statement is allowed."

    stmt = statements[0]
    if stmt is None:
        return False, "Could not parse SQL."

    # Accept any read-only root: a plain SELECT, a CTE (WITH ... SELECT parses as
    # a Select carrying a `with` arg), or set operations (UNION/INTERSECT/EXCEPT).
    # Write/DDL statements are still rejected by the DISALLOWED_STATEMENTS walk
    # below, so broadening the root stays safe.
    READ_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)
    if not isinstance(stmt, READ_ROOTS):
        return False, "Only read-only SELECT queries are allowed."

    for node in stmt.walk():
        n = node[0] if isinstance(node, tuple) else node
        if isinstance(n, DISALLOWED_STATEMENTS):
            return False, "Query contains a disallowed statement type."

    whitelist = ALWAYS_ALLOWED_TABLES | (allowed_tables or set())
    referenced = {t.name for t in stmt.find_all(exp.Table)}
    unknown = referenced - whitelist
    if unknown:
        return False, f"Query references table(s) not in your available datasets: {', '.join(sorted(unknown))}"

    return True, ""
