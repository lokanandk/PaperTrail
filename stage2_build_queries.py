"""
Stage 2: Build PubMed queries from predictions.

All query content comes from the prediction dict — no hardcoded lookup tables.

Synonym fields (all optional, add to predictions.yaml for best recall):
  disease_synonyms:   ["full name 1", "alt name 2"]
  cell_type_synonyms: ["synonym 1", "synonym 2"]
  tissue_synonyms:    ["synonym 1"]
  organism_terms:     ["human", "patient", "biopsy"]

Query strategy per prediction (in descending specificity):
  Q1  specific       — entity AND each disease term / synonym (up to 3)
  Q2  cell_type      — entity AND disease AND each cell-type term (up to 2)
  Q3  cell_broad     — entity AND cell-type (no disease constraint)
  Q4  broad_context  — entity AND tissue/context terms (high-recall safety net)
  Q5  organism       — entity AND disease AND organism study-type terms
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_STOP_TOKENS = {
    "the", "and", "for", "with", "from", "its", "has", "are", "was",
    "not", "any", "all", "but", "can", "may", "per",
}

def _is_queryable(alias: str) -> bool:
    a = alias.strip().lower()
    return len(a) >= 3 and a not in _STOP_TOKENS


def _quote(term: str) -> str:
    """Quote multi-word or hyphenated terms for PubMed; leave single tokens bare."""
    t = term.strip()
    if not t:
        return ""
    if " " in t or "-" in t:
        return f'"{t}"'
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Term derivation — from prediction fields only
# ─────────────────────────────────────────────────────────────────────────────

def _disease_terms(prediction: dict) -> List[str]:
    """
    Build disease search terms from disease_context + disease_synonyms.
    Add disease_synonyms to predictions.yaml for the best query recall:
      disease_context: DKD
      disease_synonyms: ["diabetic nephropathy", "diabetic kidney disease"]
    """
    dc = (prediction.get("disease_context") or "any").strip()
    if dc.lower() == "any":
        return []
    terms = [_quote(dc)]
    for syn in (prediction.get("disease_synonyms") or [])[:4]:
        q = _quote(str(syn).strip())
        if q and q not in terms:
            terms.append(q)
    return terms


def _cell_type_terms(prediction: dict) -> List[str]:
    """
    Build cell-type search terms from cell_type + cell_type_synonyms.
      cell_type: pDC
      cell_type_synonyms: ["plasmacytoid dendritic cell", "plasmacytoid DC"]
    """
    ct = (prediction.get("cell_type") or "any").strip()
    if ct.lower() == "any":
        return []
    terms = [_quote(ct)]
    for syn in (prediction.get("cell_type_synonyms") or [])[:3]:
        q = _quote(str(syn).strip())
        if q and q not in terms:
            terms.append(q)
    return terms


def _exclusion_not_clause(prediction: dict) -> str:
    """
    Build a PubMed NOT clause from prediction["disease_exclusions"].

    Returns a string like ' NOT ("renal carcinoma" OR "RCC")' to append to
    any query, or an empty string if no exclusions are specified.

    Add to predictions.yaml:
      disease_exclusions: ["renal carcinoma", "renal cell carcinoma", "RCC"]

    The NOT clause narrows retrieval at source, reducing stage3 noise.
    It complements the stage4 gate (which filters after retrieval).
    """
    excl = prediction.get("disease_exclusions") or []
    if not excl:
        return ""
    quoted = []
    for ex in excl[:8]:   # cap at 8 to keep query length reasonable
        t = str(ex).strip()
        if t:
            quoted.append(f'"{t}"' if " " in t or "-" in t else t)
    if not quoted:
        return ""
    if len(quoted) == 1:
        return f" NOT {quoted[0]}"
    return " NOT (" + " OR ".join(quoted) + ")"


def _tissue_terms(prediction: dict) -> List[str]:
    """
    Build tissue/organ terms from tissue + tissue_synonyms.
    These are used in the broad_context query for high-recall retrieval.
    """
    tissue = (prediction.get("tissue") or "any").strip().lower()
    if tissue in ("any", "systemic", ""):
        return []
    terms = [_quote(tissue)]
    for syn in (prediction.get("tissue_synonyms") or [])[:3]:
        q = _quote(str(syn).strip())
        if q and q not in terms:
            terms.append(q)
    return terms


def _organism_terms(prediction: dict) -> List[str]:
    """Build organism/study-type terms. Uses organism_terms if provided."""
    explicit = prediction.get("organism_terms") or []
    if explicit:
        return [str(t).strip() for t in explicit[:4]]
    org = (prediction.get("organism") or "any").strip().lower()
    if org in ("any", ""):
        return []
    _defaults = {
        "human":     ['"human"', '"patient"', '"clinical"', '"biopsy"'],
        "mouse":     ['"mouse"', '"murine"', '"mice"'],
        "rat":       ['"rat"', '"rodent"'],
        "zebrafish": ['"zebrafish"', '"danio rerio"'],
        "in vitro":  ['"in vitro"', '"cell line"'],
    }
    return _defaults.get(org, [_quote(org)])


def _broad_context_terms(prediction: dict) -> List[str]:
    """
    Build broad context terms for the high-recall safety-net query.

    Priority:
    1. tissue_synonyms (explicit, most reliable)
    2. disease_synonyms words (already have disease terms; tissue gives breadth)
    3. Individual words from disease_context (catches papers using the full name)

    This replaces the old hardcoded broad_kidney query and generalises it to
    any disease/tissue context.
    """
    terms = []

    # Tissue terms — most direct for broad context
    tissue_t = _tissue_terms(prediction)
    if tissue_t:
        terms.extend(tissue_t[:2])

    # If no tissue, try individual words from disease_context / disease_synonyms
    if not terms:
        dc = (prediction.get("disease_context") or "").strip()
        if dc and dc.lower() != "any":
            # tokenise: "diabetic kidney disease" → "kidney" is useful for broad query
            for w in dc.lower().split():
                if len(w) >= 4 and w not in _STOP_TOKENS:
                    terms.append(_quote(w))
                    break   # one word is enough for a broad query

        for syn in (prediction.get("disease_synonyms") or [])[:2]:
            for w in str(syn).lower().split():
                if len(w) >= 4 and w not in _STOP_TOKENS:
                    terms.append(_quote(w))
                    break

    return list(dict.fromkeys(terms))[:3]   # deduplicate


# ─────────────────────────────────────────────────────────────────────────────
# Core query builder
# ─────────────────────────────────────────────────────────────────────────────

def build_queries(prediction: dict) -> List[Dict]:
    """
    Build PubMed queries for one prediction.
    All content comes from the prediction dict.
    """
    aliases  = list(prediction.get("aliases", []) or [])
    entity   = prediction.get("entity", "")
    if not aliases:
        aliases = [entity]

    queryable = [a for a in aliases if _is_queryable(a)]
    if not queryable:
        queryable = [entity] if entity else []
    if not queryable:
        return []

    entity_clause = "(" + " OR ".join(_quote(a) for a in queryable[:8]) + ")"

    disease  = _disease_terms(prediction)
    cell     = _cell_type_terms(prediction)
    tissue   = _tissue_terms(prediction)
    organism = _organism_terms(prediction)
    broad    = _broad_context_terms(prediction)

    queries  = []

    not_clause = _exclusion_not_clause(prediction)

    # Q1: entity + disease — primary, fires once per disease term / synonym
    if disease:
        for d in disease[:3]:
            queries.append({"name": "specific",
                            "query": f"{entity_clause} AND {d}{not_clause}",
                            "weight": 1.0})
    else:
        anchor = tissue[0] if tissue else None
        if anchor:
            queries.append({"name": "entity_tissue",
                            "query": f"{entity_clause} AND {anchor}{not_clause}",
                            "weight": 1.0})
        else:
            queries.append({"name": "entity_bare",
                            "query": f"{entity_clause}{not_clause}", "weight": 1.0})

    # Q2: entity + disease + cell-type (high specificity, good for scRNA-seq lit)
    if disease and cell:
        for ct in cell[:2]:
            queries.append({"name": "cell_type",
                            "query": f"{entity_clause} AND {disease[0]} AND {ct}{not_clause}",
                            "weight": 0.9})

    # Q3: entity + cell-type alone (catches papers that don't name the disease)
    if cell:
        queries.append({"name": "cell_broad",
                        "query": f"{entity_clause} AND {cell[0]}{not_clause}",
                        "weight": 0.85})

    # Q4: entity + broad context — high-recall safety net
    if broad:
        broad_clause = " OR ".join(broad) if len(broad) == 1 else "(" + " OR ".join(broad) + ")"
        queries.append({"name": "broad_context",
                        "query": f"{entity_clause} AND {broad_clause}{not_clause}",
                        "weight": 0.75})

    # Q5: entity + disease + organism / study type
    if disease and organism:
        queries.append({"name": "organism",
                        "query": f"{entity_clause} AND {disease[0]} AND {organism[0]}{not_clause}",
                        "weight": 0.80})

    return queries


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def build_all(predictions_path: Path, output_path: Path) -> None:
    with open(predictions_path) as f:
        data = yaml.safe_load(f)
    rows = []
    for p in data["predictions"]:
        qs = build_queries(p)
        for q in qs:
            rows.append({"prediction_id": p["id"], "query_name": q["name"],
                         "query": q["query"], "weight": q["weight"]})
    with open(output_path, "w") as f:
        yaml.safe_dump({"queries": rows}, f, sort_keys=False, allow_unicode=True)
    print(f"Built {len(rows)} queries across {len(data['predictions'])} predictions")


if __name__ == "__main__":
    base = Path(__file__).parent
    build_all(base / "predictions_expanded.yaml", base / "queries.yaml")
