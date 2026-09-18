"""
Stage 1: Expand gene aliases via authoritative sources.

For each gene/transporter prediction, query NCBI Gene (via mygene.info) to obtain
the full set of approved symbols, aliases, and Ensembl/MeSH cross-references.

This expands fuzzy-match recall without sacrificing precision: we keep prediction-
specific aliases AND officially documented synonyms.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import yaml
import mygene
from rapidfuzz import fuzz

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def expand_gene(symbol: str, cache: dict) -> dict:
    """Query mygene for a single symbol, return aliases and cross-refs.

    Cached to disk to avoid repeat API calls.
    """
    if symbol in cache:
        return cache[symbol]
    mg = mygene.MyGeneInfo()
    try:
        result = mg.query(
            symbol,
            species="human",
            fields="symbol,name,alias,entrezgene,ensembl.gene,MeSH",
            size=1,
        )
        hits = result.get("hits", [])
        if not hits:
            cache[symbol] = {"resolved": None, "aliases": []}
            return cache[symbol]
        h = hits[0]
        aliases = [h.get("symbol")]
        if h.get("alias"):
            if isinstance(h["alias"], list):
                aliases.extend(h["alias"])
            else:
                aliases.append(h["alias"])
        if h.get("name"):
            aliases.append(h["name"])
        # Deduplicate, preserve order
        seen = set()
        clean = []
        for a in aliases:
            if a and a.lower() not in seen:
                clean.append(a)
                seen.add(a.lower())
        cache[symbol] = {
            "resolved": h.get("symbol"),
            "entrez": h.get("entrezgene"),
            "ensembl": (h.get("ensembl") or {}).get("gene") if isinstance(h.get("ensembl"), dict) else None,
            "name": h.get("name"),
            "aliases": clean,
        }
    except Exception as e:
        cache[symbol] = {"resolved": None, "aliases": [], "error": str(e)}
    time.sleep(0.1)  # be polite to mygene
    return cache[symbol]


# Fields every later stage assumes are lists. A prediction written by hand (or
# returned by an LLM) sometimes has one of these as a bare string instead —
# `disease_synonyms: "lupus"` instead of `["lupus"]`. Most call sites don't
# crash on that; they silently iterate the string's characters instead of its
# words, which is worse than a crash because nothing looks wrong until the
# PubMed queries come back empty. Normalizing here, once, means every later
# stage can trust the type without checking it itself.
_LIST_FIELDS = ("aliases", "disease_synonyms", "disease_exclusions", "cell_type_synonyms")


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # A single value typed without brackets, or a comma-separated list.
        return [v.strip() for v in value.split(",") if v.strip()]
    return [value]


def expand_predictions(predictions_path: Path, output_path: Path) -> None:
    with open(predictions_path) as f:
        data = yaml.safe_load(f)

    cache_path = CACHE_DIR / "gene_aliases.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    expanded = []
    for p in data["predictions"]:
        for field in _LIST_FIELDS:
            if field in p:
                p[field] = _ensure_list(p[field])
        primary = p["entity"]
        if p["entity_type"] in ("gene", "transporter"):
            info = expand_gene(primary, cache)
            p.setdefault("aliases", [])
            existing = set(a.lower() for a in p["aliases"])
            auto = []
            for alias in info.get("aliases", []):
                if alias.lower() not in existing:
                    p["aliases"].append(alias)
                    existing.add(alias.lower())
                    auto.append(alias)
            # Record which aliases we added ourselves. They are good for search
            # recall but must not be trusted as proof that a paper is about this
            # gene: mygene lists "PSSA" for PTDSS1, and in the literature "PSSa"
            # nearly always means poly(styrene sulfonic acid), while "PSSA" means
            # penicillin-susceptible S. aureus. Stage 4 weighs them accordingly.
            if auto:
                p["auto_aliases"] = auto
            p["resolved_symbol"] = info.get("resolved")
            p["entrez_id"] = info.get("entrez")
            p["ensembl_id"] = info.get("ensembl")
        expanded.append(p)

    cache_path.write_text(json.dumps(cache, indent=2))

    # ── Synonym enrichment (optional) ────────────────────────────────────────
    # Adds disease_synonyms via NCBI MeSH and cell_type_synonyms via Cell Ontology
    # for predictions that don't already have them in predictions.yaml.
    # Graceful degradation: if synonym_enrichment.py is absent or any lookup
    # fails, predictions are written as-is with no error.
    try:
        import synonym_enrichment as _se
        ncbi_email   = _se.Entrez.email   if hasattr(_se, 'Entrez') else ""
        ncbi_api_key = getattr(_se.Entrez, 'api_key', "")  if hasattr(_se, 'Entrez') else ""
        # Use NCBI credentials from environment if available
        import os
        email   = os.environ.get("NCBI_EMAIL",   ncbi_email)
        api_key = os.environ.get("NCBI_API_KEY", ncbi_api_key)
        # Only call MeSH enrichment if we have an email
        expanded = _se.enrich_all_predictions(
            expanded,
            ncbi_email=email,
            ncbi_api_key=api_key,
            use_mesh=bool(email),
        )
    except ImportError:
        pass   # synonym_enrichment.py not found — continue without it
    except Exception as _e:
        print(f"  [stage1] Synonym enrichment failed: {_e} — continuing without it")

    with open(output_path, "w") as f:
        yaml.safe_dump({"predictions": expanded}, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {len(expanded)} expanded predictions to {output_path}")


def fuzzy_match_to_predictions(text: str, predictions: list[dict], threshold: int = 85) -> list[dict]:
    """Return predictions whose aliases fuzzy-match anywhere in text.

    Used in Stage 4 to filter literature to relevant predictions.
    """
    text_lower = text.lower()
    matches = []
    for p in predictions:
        best_score = 0
        best_alias = None
        for alias in p.get("aliases", []):
            if alias.lower() in text_lower:
                # exact substring match — give max score
                best_score = 100
                best_alias = alias
                break
            score = fuzz.partial_ratio(alias.lower(), text_lower)
            if score > best_score:
                best_score = score
                best_alias = alias
        if best_score >= threshold:
            matches.append({
                "prediction_id": p["id"],
                "match_score": best_score,
                "matched_alias": best_alias,
            })
    return matches


if __name__ == "__main__":
    import sys
    base = Path(__file__).parent
    expand_predictions(
        base / "predictions.yaml",
        base / "predictions_expanded.yaml",
    )
