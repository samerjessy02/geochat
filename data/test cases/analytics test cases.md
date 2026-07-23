# Analytics tab — test cases & expected answers

Numbers below are computed directly from the synthetic datasets in
`data/Synthetic map data/`. The **counts are deterministic** (the backend computes
them). The exact chart *type*, *title*, and summary wording are chosen by the LLM
plan, so treat those as "expected shape," not exact strings.

Dataset sizes: Universities **5**, Schools **120**, Hospitals **35**,
Pharmacies **150**, Restaurants **300**.

---

## A. Single dataset — group & count

**Select only `Universities`.**

1. `count universities per district`
   - Chart: bar · Table sortable + CSV
   - Expected: **University District = 4**, **Zamalek Isle = 1** (total 5). Top = University District.

2. `which category has the most entries`
   - Expected: **University = 5** (it's the only category).

3. `show the opening hours distribution`
   - Expected: a single group **08:00-18:00 = 5** (all universities share it).

**Select only `Schools` (120).**

4. `count schools per district`
   - Expected (highest→lowest): **Nasr City 16**, New Cairo Heights 15, Helwan Gardens 15,
     Maadi Gardens 15, Nile Riverside 12, Zamalek Isle 12, Downtown 11, Old Town 10,
     University District 7, Industrial Zone 7. Sum = 120. Top = **Nasr City (16)**.

5. `how many schools are there in total`
   - Expected: **120** (single category "School").

**Select only `Restaurants` (300).**

6. `which district has the most restaurants`
   - Expected: **Downtown = 178** (dominant), then Nasr City 28, Zamalek Isle 25, Nile Riverside 13,
     and the rest ≤ 11. Top = **Downtown (178)**.

**Select only `Hospitals` (35).**

7. `count hospitals per district`
   - Expected: **Downtown 9**, **Zamalek Isle 9**, Nile Riverside 7, Old Town 3,
     University District 2, then 1 each for Industrial Zone, Nasr City, New Cairo Heights,
     Helwan Gardens, Maadi Gardens. Sum = 35.

---

## B. Comparison (cross-tab) — two categorical fields

**Select `Pharmacies` + `Schools`.**

8. `compare number of pharmacies and schools per district`
   - Chart: grouped bar (one color per dataset) · Table: district × dataset with a **Total** column · CSV export.
   - Expected per district (Pharmacies / Schools):
     Downtown 30 / 11 · Zamalek Isle 21 / 12 · Nasr City 20 / 16 · University District 14 / 7 ·
     Nile Riverside 13 / 12 · New Cairo Heights 13 / 15 · Maadi Gardens 12 / 15 ·
     Helwan Gardens 11 / 15 · Industrial Zone 8 / 7 · Old Town 8 / 10.
   - Column totals: **Pharmacies 150**, **Schools 120**.

**Select ALL datasets.**

9. `compare total entries per dataset`
   - Expected (count per layer): **Restaurants 300**, **Pharmacies 150**, **Schools 120**,
     **Hospitals 35**, **Universities 5**.

10. `compare category counts across all datasets`
    - Expected: each dataset is one category → Restaurant 300, Pharmacy 150, School 120, Hospital 35, University 5.

---

## C. Aggregates over geometry (these datasets have no numeric business fields)

The only numeric fields are the derived coordinates `_lon` / `_lat`, so avg/min/max
apply to location. Use these to exercise the aggregate path:

11. `average latitude per district for pharmacies` (select Pharmacies)
    - Expected: one averaged latitude value per district (≈ 30.0x). Exact value is computed from
      the point coordinates; verify it's a bar/table of ~10 districts, not a hallucinated number.

12. `northernmost restaurant per district` → `max latitude per district` (select Restaurants)
    - Expected: max `_lat` per district; 10 rows, Downtown present.

---

## D. UI behaviors to verify (not questions)

- **Sortable table**: click a column header → rows re-sort; click again → reverse.
- **CSV export**: "⭳ Export CSV" downloads a file matching the on-screen table.
- **Separate state**: switch to GeoChat and back — Analytics history/charts persist and are
  independent of the map chat.
- **Dataset picker**: unchecking a dataset excludes it from the next question.
- **Empty/edge**: ask with no dataset selected → "Select at least one dataset to analyze."

---

## E. GeoChat (map / RAG) sanity — for regression

13. `show me all universities` → map plots **5** points.
14. `show cairo university` → **1** point.
15. `show pharmacies in Downtown` → **30** pharmacies.
16. `show hospitals in Zamalek Isle` → **9** hospitals.
17. Paste a polygon covering central Cairo, then `show all pharmacies and schools in this area`
    → **both** layers plotted in different colors with a legend.

---

## F. Memory / follow-ups — for regression

18. `show cairo university` → then `tell me more about it`
    - Expected: resolves "it" → **Cairo University**; knowledge answer with address
      (164 El Nil Street), phone, website. Should NOT ask "specify what you mean."
19. `summarize the conversation`
    - Expected: a factual recap of what was asked (universities, Cairo University). Should **not**
      invent unrelated content (e.g. "coffee shops").
