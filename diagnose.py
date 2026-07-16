"""
diagnose.py — trace a single query through every stage of the pipeline.

Run against your live databases/LLM (the server does NOT need to be running):

    python diagnose.py "which cafes have handcrafted beverages"

It prints, for that one query: the guardrail result, the classified intent, the
structured dataset lookup (+ generated SQL and the rows it returns), and the
document RAG retrieval with scores. Paste the whole output back and it pinpoints
exactly which stage is producing wrong results.
"""

import sys


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or "which cafes have handcrafted beverages"
    print(f"QUERY: {query!r}")

    import registry

    # --- datasets present -------------------------------------------------
    section("DATASETS")
    try:
        datasets = registry.list_datasets()
        for d in datasets:
            print(f"  id={d['id']}  name={d['display_name']!r}  "
                  f"geom={d['geometry_type']}  cols={len(d.get('columns') or [])}")
        dataset_ids = [d["id"] for d in datasets]
        if not datasets:
            print("  (none — upload/select a dataset)")
    except Exception as e:  # noqa: BLE001
        print("  ERROR listing datasets:", repr(e))
        dataset_ids = []

    # --- input guardrail --------------------------------------------------
    section("INPUT GUARDRAIL")
    try:
        from agents import guardrails
        print(" ", guardrails.check_input(query).as_dict())
    except Exception as e:  # noqa: BLE001
        print("  ERROR:", repr(e))

    # --- intent -----------------------------------------------------------
    section("INTENT CLASSIFICATION")
    entity = None
    try:
        from agents.intent_router import classify_intent
        intent = classify_intent(query)
        print(" ", intent.as_dict())
        entity = intent.entity_focus
    except Exception as e:  # noqa: BLE001
        print("  ERROR:", repr(e))

    # --- structured dataset lookup ---------------------------------------
    section("DATASET LOOKUP (structured columns)")
    try:
        from agents import dataset_lookup
        match = dataset_lookup.lookup(entity or query, dataset_ids)
        if not match:
            print("  no matching feature")
        else:
            print(f"  matched {match.num_records} record(s); website={match.website}")
            print("  context (first 600 chars):")
            print("   ", match.context[:600].replace("\n", "\n    "))
    except Exception as e:  # noqa: BLE001
        print("  ERROR:", repr(e))

    # --- spatial SQL (map path) ------------------------------------------
    section("MAP PATH — generated SQL + rows")
    try:
        from llm import generate_sql
        from validator import validate_sql
        from db import run_query

        sql = generate_sql(query, dataset_ids)
        print("  SQL:\n   ", sql.replace("\n", "\n    "))
        ok, reason = validate_sql(sql, allowed_tables=registry.get_table_names(dataset_ids))
        print("  valid:", ok, "" if ok else f"({reason})")
        if ok:
            rows = run_query(sql)
            print("  rows returned:", len(rows))
            for r in rows[:3]:
                print("   ", {k: v for k, v in r.items() if k != "geometry"})
    except Exception as e:  # noqa: BLE001
        print("  ERROR:", repr(e))

    # --- document RAG -----------------------------------------------------
    section("DOCUMENT RAG — hybrid retrieval + validation")
    try:
        from agents.hybrid_retriever import hybrid_search
        from agents.retrieval_validator import validate

        hits = hybrid_search(query)
        v = validate(query, hits)
        print(f"  {len(hits)} hit(s); sufficient={v.is_sufficient}; "
              f"context_score={v.context_score:.2f}; keyword_overlap={v.keyword_overlap:.2f}")
        for h in hits[:4]:
            md = h.get("metadata", {})
            print(f"   [{md.get('source')} p{md.get('page')}] score={h.get('score'):.4f} "
                  f":: {h.get('text', '')[:110]}")
    except Exception as e:  # noqa: BLE001
        print("  ERROR:", repr(e))

    print("\nDone. Paste everything above back to continue.")


if __name__ == "__main__":
    main()
