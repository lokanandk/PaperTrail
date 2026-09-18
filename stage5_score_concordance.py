"""
Stage 5 (v3): Per-prediction concordance scoring with sensitivity analysis.

Inherits from v2:
  - Trichotomous classification (concordant / opposite / neutral)
  - Neutral records excluded from inferential statistics
  - Optional direction_scope filter for baseline metabolite predictions

Adds in v3:
  - Quality-weighted concordance — records weighted by relevance × confidence
    × (1 - hedge_discount) × study_quality_weight. High-quality, unhedged,
    confident extractions dominate the score.
  - Wilson 95% confidence intervals on each concordance estimate.
  - Leave-one-out sensitivity analysis — flags predictions where the
    concordance is driven by a single paper. Outputs the LOO range so
    reviewers can audit.
  - Cross-prediction meta-pooling within each mechanistic category.
  - Discordance investigation report — for opposite records, surface the
    abstract excerpt and direction snippet so reviewers can investigate.
  - Three-tier reporting: simple, weighted, quality-weighted concordance —
    all reported. STRONG tier requires both simple ≥ 0.75 AND quality-
    weighted ≥ 0.70 to avoid concordance dominated by single high-weight outliers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

DIRECTION_MAP = {
    "up":           {"up"},
    "down":         {"down", "absent"},
    "preserved":    {"preserved"},
    "absent":       {"absent", "down"},
    "bidirectional": {"up", "down", "preserved", "absent"},  # kept for YAML backward compat
    "associated":    {"up", "down", "preserved", "absent"},  # preferred synonym of bidirectional
    # "absent" is also routed to _descriptive (volume-based scoring) above the map lookup
    "binary_present_absent": {"absent", "down"},
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple:
    """Wilson 95% CI for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify_relation(expected_dir: str, extracted_dir: str,
                      extraction_confidence: float = 1.0) -> str:
    """concordant / opposite / neutral.

    Low-confidence extractions (conf < 0.25) are classified as neutral rather
    than opposite, because when direction words nearly tie in count the top
    signal is unreliable. This prevents weak ties from being counted as
    genuine literature disagreement.
    """
    if extracted_dir is None:
        return "neutral"
    # Low-confidence signals are uninformative — treat as neutral not opposite
    if extraction_confidence < 0.20:
        return "neutral"
    compat = DIRECTION_MAP.get(expected_dir, set())
    if expected_dir in {"up", "down"}:
        if extracted_dir in compat:
            return "concordant"
        if expected_dir == "up" and extracted_dir in {"down", "absent"}:
            return "opposite"
        if expected_dir == "down" and extracted_dir == "up":
            return "opposite"
        return "neutral"
    if expected_dir == "preserved":
        if extracted_dir == "preserved":
            return "concordant"
        if extracted_dir in {"up", "down", "absent"}:
            return "opposite"
        return "neutral"
    if expected_dir in {"absent", "binary_present_absent"}:
        if extracted_dir in compat:
            return "concordant"
        if extracted_dir == "up":
            return "opposite"
        return "neutral"
    return "neutral"


def record_weight(r: dict) -> float:
    """Combined weight: relevance × confidence × (1-hedge_discount) × quality
    × context_anchor_factor.

    context_anchor_factor:
      1.0 — direction signal co-occurred with disease/cell-type context (reliable)
      0.5 — direction signal found with no disease context in the window (unreliable)
            set by extract_direction when context_anchored=False
    """
    relevance  = r.get("relevance", 0.5)
    direction  = r.get("direction_extraction", {}) or {}
    conf       = direction.get("confidence", 0)
    hedge_disc = direction.get("hedge_discount", 0)
    quality    = r.get("quality_weight", 1.0)
    # Papers where direction was never co-located with disease context
    # are down-weighted further — their directional signal is less trustworthy.
    anchor     = 1.0 if direction.get("context_anchored", True) else 0.6
    return float(relevance * (0.5 + 0.5 * conf) * (1 - 0.5 * hedge_disc) * quality * anchor)


# ─────────────────────────────────────────────────────────────────────────────
# Per-prediction scoring
# ─────────────────────────────────────────────────────────────────────────────
def score_one(prediction_data: dict) -> dict:
    pred = prediction_data["prediction"]
    evidence = prediction_data["evidence"]
    n_relevant = len(evidence)
    expected_dir = pred["direction"]
    direction_scope = pred.get("direction_scope", "any")

    if n_relevant == 0:
        return _empty_result(pred, expected_dir, "No relevant literature retrieved")

    # Records with directional signal
    directional = []
    for r in evidence:
        d = r.get("direction_extraction", {}) or {}
        if d.get("signal_direction") in (None, ""):
            continue
        directional.append({
            **r,
            "extracted_direction": d["signal_direction"],
            "extraction_confidence": d.get("confidence", 0),
            "hedge_discount": d.get("hedge_discount", 0),
        })

    # Filter to observational studies for baseline_disease_vs_control predictions.
    # Originally only applied to metabolites — BUG: gene predictions like NPHS1
    # and GCLC are equally contaminated by treatment papers that report the gene
    # going UP after treatment (opposite to baseline DKD direction).
    # Fix: apply to ALL entity types that have this scope flag.
    if direction_scope == "baseline_disease_vs_control":
        non_interventional = [
            r for r in directional if r.get("study_context") != "interventional"
        ]
        if non_interventional:
            directional = non_interventional
        # Also filter by study_context for all entity types

    n_directional = len(directional)
    if n_directional == 0:
        return _no_directional(pred, expected_dir, n_relevant, evidence)

    if expected_dir in {"bidirectional", "associated", "absent"}:
        return _descriptive(pred, expected_dir, n_relevant, n_directional,
                            directional, evidence)

    # Classify each record
    informative = []
    n_conc = n_opp = n_neu = 0
    for r in directional:
        rel = classify_relation(
            expected_dir, r["extracted_direction"],
            r.get("extraction_confidence", 1.0)
        )
        r["relation"] = rel
        if rel == "concordant":
            n_conc += 1
            informative.append(r)
        elif rel == "opposite":
            n_opp += 1
            informative.append(r)
        else:
            n_neu += 1

    n_informative = len(informative)
    if n_informative == 0:
        return _no_informative(pred, expected_dir, n_relevant, n_directional,
                                directional, evidence)

    # Three concordance metrics
    conc_simple = n_conc / n_informative

    # Relevance × confidence weighted (no quality)
    w_rel = np.array([
        r.get("relevance", 0.5) * (0.5 + 0.5 * r.get("extraction_confidence", 0))
        * (1 - 0.5 * r.get("hedge_discount", 0))
        for r in informative
    ])
    conc_arr = np.array([
        1.0 if r["relation"] == "concordant" else 0.0
        for r in informative
    ])
    conc_weighted = (float(np.average(conc_arr, weights=w_rel))
                      if w_rel.sum() > 0 else conc_simple)

    # Quality-weighted (relevance × confidence × hedge × study_quality)
    w_qual = np.array([record_weight(r) for r in informative])
    conc_quality_weighted = (float(np.average(conc_arr, weights=w_qual))
                              if w_qual.sum() > 0 else conc_simple)

    # 95% Wilson CI on simple concordance
    ci_low, ci_high = wilson_ci(n_conc, n_informative)

    # Binomial sign test
    try:
        p_val = float(binomtest(n_conc, n_informative,
                                 p=0.5, alternative="greater").pvalue)
    except Exception:
        p_val = None

    # Leave-one-out sensitivity
    loo = leave_one_out_concordance(informative)

    # Tier
    tier = _assign_tier(n_informative, conc_simple, p_val, conc_quality_weighted, loo)
    tier_reason = (f"n_informative={n_informative}, simple={conc_simple:.0%}, "
                    f"quality-weighted={conc_quality_weighted:.0%}, p={p_val}, "
                    f"LOO range={loo['range']:.2f}")

    # Discordance investigation: list of opposite records
    discordance_reports = [
        {
            "pmid": r["pmid"], "title": r.get("title"),
            "journal": r.get("journal"), "year": r.get("year"),
            "extracted_direction": r["extracted_direction"],
            "abstract_excerpt": r.get("abstract_excerpt", ""),
            "evidence_phrases": (r.get("direction_extraction", {}) or {}).get("evidence_phrases", []),
        }
        for r in informative if r["relation"] == "opposite"
    ]

    return {
        "prediction_id": pred["id"],
        "prediction": pred,
        "n_relevant": n_relevant,
        "n_directional": n_directional,
        "n_informative": n_informative,
        "n_concordant": n_conc,
        "n_opposite": n_opp,
        "n_neutral": n_neu,
        "concordance_simple": round(conc_simple, 3),
        "concordance_weighted": round(conc_weighted, 3),
        "concordance_quality_weighted": round(conc_quality_weighted, 3),
        "ci95_low": round(ci_low, 3),
        "ci95_high": round(ci_high, 3),
        "binomial_p": round(p_val, 4) if p_val is not None else None,
        "leave_one_out": loo,
        "tier": tier,
        "tier_reason": tier_reason,
        "scoring_mode": "directional",
        "summary_evidence": [_summary_row(r) for r in informative],
        "discordance_reports": discordance_reports,
        "magnitude_pool": _pool_magnitude(evidence),
        "novelty": pred.get("novelty"),
        "expected_direction": expected_dir,
        "direction_distribution": _distribution(directional),
    }


def leave_one_out_concordance(informative: list) -> dict:
    """LOO concordance — flag if removing any single paper changes result by ≥ 0.10."""
    n = len(informative)
    if n < 2:
        return {"min": None, "max": None, "range": 0.0,
                "fragile": bool(n == 1), "values": [], "loo_papers": []}
    n_conc_total = sum(1 for r in informative if r["relation"] == "concordant")
    base_conc    = n_conc_total / n
    values, papers_delta = [], []
    for i, rec in enumerate(informative):
        k = n_conc_total - 1 if rec["relation"] == "concordant" else n_conc_total
        conc_without = k / (n - 1)
        values.append(conc_without)
        delta = abs(conc_without - base_conc)
        papers_delta.append({
            "pmid":        rec.get("pmid", ""),
            "title":       rec.get("title", ""),
            "year":        rec.get("year"),
            "relation":    rec.get("relation", ""),
            "delta":       round(delta, 3),
            "conc_without": round(conc_without, 3),
        })
    arr       = np.array(values)
    loo_range = float(arr.max() - arr.min())
    # Top 3 most impactful papers (those causing ≥ 0.10 pp shift)
    papers_delta.sort(key=lambda x: -x["delta"])
    loo_papers = [p for p in papers_delta[:3] if p["delta"] >= 0.10]
    return {
        "min":       float(arr.min()),
        "max":       float(arr.max()),
        "range":     loo_range,
        "fragile":   bool(loo_range >= 0.10),   # ≥ 0.10 pp shift = fragile
        "values":    [round(v, 3) for v in values],
        "loo_papers": loo_papers,
    }


def _empty_result(pred, expected, reason):
    return {
        "prediction_id": pred["id"], "prediction": pred,
        "n_relevant": 0, "n_directional": 0, "n_informative": 0,
        "n_concordant": 0, "n_opposite": 0, "n_neutral": 0,
        "concordance_simple": None, "concordance_weighted": None,
        "concordance_quality_weighted": None,
        "ci95_low": None, "ci95_high": None,
        "binomial_p": None, "leave_one_out": None,
        "tier": "NONE", "tier_reason": reason,
        "summary_evidence": [], "discordance_reports": [], "magnitude_pool": None,
        "novelty": pred.get("novelty"), "expected_direction": expected,
    }


def _no_directional(pred, expected, n_rel, evidence):
    return {
        "prediction_id": pred["id"], "prediction": pred,
        "n_relevant": n_rel, "n_directional": 0, "n_informative": 0,
        "n_concordant": 0, "n_opposite": 0, "n_neutral": 0,
        "concordance_simple": None, "concordance_weighted": None,
        "concordance_quality_weighted": None,
        "ci95_low": None, "ci95_high": None,
        "binomial_p": None, "leave_one_out": None,
        "tier": "NO_DIRECTIONAL",
        "tier_reason": f"{n_rel} relevant records but none yielded a directional signal",
        "summary_evidence": [_summary_row(r) for r in evidence[:20]],
        "discordance_reports": [],
        "magnitude_pool": _pool_magnitude(evidence),
        "novelty": pred.get("novelty"), "expected_direction": expected,
    }


def _no_informative(pred, expected, n_rel, n_dir, directional, evidence):
    return {
        "prediction_id": pred["id"], "prediction": pred,
        "n_relevant": n_rel, "n_directional": n_dir, "n_informative": 0,
        "n_concordant": 0, "n_opposite": 0, "n_neutral": n_dir,
        "concordance_simple": None, "concordance_weighted": None,
        "concordance_quality_weighted": None,
        "ci95_low": None, "ci95_high": None,
        "binomial_p": None, "leave_one_out": None,
        "tier": "NO_INFORMATIVE",
        "tier_reason": f"{n_dir} directional but all neutral wrt prediction",
        "summary_evidence": [_summary_row(r) for r in directional],
        "discordance_reports": [],
        "magnitude_pool": _pool_magnitude(evidence),
        "novelty": pred.get("novelty"), "expected_direction": expected,
        "direction_distribution": _distribution(directional),
    }


def _descriptive(pred, expected, n_rel, n_dir, directional, evidence):
    """
    Score a prediction with direction: associated (or bidirectional).
    Tier reflects literature volume confirming the entity is biologically active
    in this context — direction is not tested.
    """
    # Volume alone is a poor signal: a pile of loosely-related papers should not
    # read as strong evidence. Require the retained papers to be squarely on
    # topic before calling an association strong.
    rel_scores = [r.get("relevance", 0.0) for r in directional]
    mean_relevance = (sum(rel_scores) / len(rel_scores)) if rel_scores else 0.0
    well_targeted = mean_relevance >= 0.50

    if n_rel == 0:
        tier, reason = "NONE", "No relevant papers found"
    elif (n_dir >= 8 or n_rel >= 12) and well_targeted:
        tier   = "STRONG"
        reason = (f"Strong association: {n_rel} relevant papers "
                  f"({n_dir} with directional signals, "
                  f"mean relevance {mean_relevance:.2f}). "
                  f"Direction not tested (associated prediction).")
    elif n_dir >= 8 or n_rel >= 12:
        tier   = "MODERATE"
        reason = (f"Association by volume only: {n_rel} relevant papers, but "
                  f"mean relevance {mean_relevance:.2f} is low — the literature "
                  f"mentions this entity without focusing on this context.")
    elif n_dir >= 4 or n_rel >= 5:
        tier   = "MODERATE"
        reason = (f"Moderate association: {n_rel} relevant, {n_dir} directional")
    elif n_rel >= 2:
        tier   = "WEAK_SUPPORT"
        reason = f"Weak association: {n_rel} relevant paper(s) found"
    else:
        tier   = "NO_INFORMATIVE"
        reason = f"Limited evidence: {n_rel} relevant, {n_dir} directional"

    return {
        "prediction_id": pred["id"], "prediction": pred,
        "n_relevant": n_rel, "n_directional": n_dir, "n_informative": None,
        "n_concordant": None, "n_opposite": None, "n_neutral": None,
        "concordance_simple": None, "concordance_weighted": None,
        "concordance_quality_weighted": None,
        "ci95_low": None, "ci95_high": None,
        "binomial_p": None, "leave_one_out": None,
        "tier": tier,
        "tier_reason": reason,
        "scoring_mode": "association",    # volume-based, not directional concordance
        "summary_evidence": [_summary_row(r) for r in directional],
        "discordance_reports": [],
        "magnitude_pool": _pool_magnitude(evidence),
        "novelty": pred.get("novelty"), "expected_direction": expected,
        "direction_distribution": _distribution(directional),
    }


def _summary_row(r):
    direction = (r.get("direction_extraction") or {})
    return {
        "pmid": r["pmid"], "title": r.get("title"),
        "journal": r.get("journal"), "year": r.get("year"),
        "extracted_direction": r.get("extracted_direction") or direction.get("signal_direction"),
        "extraction_confidence": r.get("extraction_confidence", direction.get("confidence", 0)),
        "hedge_discount": r.get("hedge_discount", direction.get("hedge_discount", 0)),
        "context_anchored": direction.get("context_anchored", True),
        "relevance": r.get("relevance"),
        "semantic_score": r.get("semantic_score"),
        "quality_weight": r.get("quality_weight"),
        "study_context": r.get("study_context"),
        "relation": r.get("relation"),
        "best_excerpt": r.get("best_excerpt") or r.get("abstract_excerpt") or "",
        "abstract_excerpt": r.get("abstract_excerpt") or r.get("best_excerpt") or "",
        "has_pmc": bool(r.get("pmc_sentences")),
        "excerpt_source": r.get("excerpt_source", "abstract"),
    }


def _distribution(records):
    counts = {"up": 0, "down": 0, "preserved": 0, "absent": 0}
    for r in records:
        d = r["extracted_direction"]
        if d in counts:
            counts[d] += 1
    return counts


def _pool_magnitude(evidence):
    folds = []
    for r in evidence:
        for f in (r.get("quantitative_extraction", {}) or {}).get("fold_changes", []):
            if 0.01 <= f <= 100:
                folds.append(f)
    if not folds:
        return None
    return {
        "n_records_with_fold": len(folds),
        "median_fold": float(np.median(folds)),
        "iqr_fold": [float(np.percentile(folds, 25)), float(np.percentile(folds, 75))],
    }


def _coerce_bool(v) -> bool:
    """Coerce JSON-deserialised bool-ish value to Python bool.
    Handles True/False, "True"/"False" strings (from default=str), numpy bool_, None.
    """
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _assign_tier(n: int, conc: float, p, conc_q: float, loo: dict) -> str:
    """Tier assignment considers LOO fragility; coerces fragile safely."""
    fragile = _coerce_bool(loo.get("fragile") if loo else False)
    if n >= 5 and conc >= 0.75 and conc_q >= 0.70 and (p is None or p < 0.05):
        return "STRONG"
    if n >= 3 and conc >= 0.6 and conc_q >= 0.55 and not fragile:
        return "MODERATE"
    if n >= 3 and 0.4 <= conc < 0.6:
        return "MIXED"
    if n >= 1 and conc >= 0.5:
        return "WEAK_SUPPORT"
    if n >= 1 and conc < 0.5:
        return "WEAK_DISCORDANT"
    return "NONE"


# ─────────────────────────────────────────────────────────────────────────────
# Cross-prediction meta-pooling
# ─────────────────────────────────────────────────────────────────────────────
def category_pooled_summary(scored: dict) -> dict:
    """Pool concordance within each mechanistic category."""
    categories = {}
    for s in scored.values():
        cat = s["prediction"].get("category", "other")
        categories.setdefault(cat, []).append(s)

    pooled = {}
    for cat, ss in categories.items():
        n_inf_total = sum((s.get("n_informative") or 0) for s in ss)
        n_conc_total = sum((s.get("n_concordant") or 0) for s in ss
                            if s.get("n_concordant") is not None)
        if n_inf_total > 0:
            ci_low, ci_high = wilson_ci(n_conc_total, n_inf_total)
            try:
                p = float(binomtest(n_conc_total, n_inf_total,
                                     p=0.5, alternative="greater").pvalue)
            except Exception:
                p = None
        else:
            ci_low = ci_high = p = None
        pooled[cat] = {
            "n_predictions": len(ss),
            "n_informative_total": n_inf_total,
            "n_concordant_total": n_conc_total,
            "pooled_concordance": (round(n_conc_total / n_inf_total, 3)
                                    if n_inf_total else None),
            "ci95": [round(ci_low, 3) if ci_low is not None else None,
                     round(ci_high, 3) if ci_high is not None else None],
            "binomial_p": round(p, 4) if p is not None else None,
            "tier_distribution": {t: sum(1 for s in ss if s["tier"] == t)
                                   for t in {s["tier"] for s in ss}},
        }
    return pooled


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def score_all(extracted_path: Path, output_path: Path) -> None:
    with open(extracted_path) as f:
        extracted = json.load(f)
    scored = {pid: score_one(data) for pid, data in extracted.items()}
    pooled = category_pooled_summary(scored)
    with open(output_path, "w") as f:
        json.dump({"predictions": scored, "category_pooled": pooled},
                   f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else (float(o) if hasattr(o, "__float__") else str(o)))
    print(f"Wrote {output_path}")
    print()
    print("Tier summary:")
    tiers = {}
    for s in scored.values():
        tiers[s["tier"]] = tiers.get(s["tier"], 0) + 1
    for t, c in sorted(tiers.items()):
        print(f"  {t:18s}: {c}")
    print()
    print("Category pooling:")
    for cat, p in pooled.items():
        if p["pooled_concordance"] is not None:
            ci = p["ci95"]
            print(f"  {cat:18s}: {p['n_concordant_total']}/{p['n_informative_total']} "
                  f"= {p['pooled_concordance']:.0%}  "
                  f"[{ci[0]:.2f}-{ci[1]:.2f}]  p={p['binomial_p']}")
    n_fragile = sum(1 for s in scored.values()
                     if _coerce_bool((s.get("leave_one_out") or {}).get("fragile")))
    if n_fragile:
        print(f"\n  Fragile (LOO range > 0.25): {n_fragile} predictions")


if __name__ == "__main__":
    base = Path(__file__).parent
    score_all(base / "extracted_evidence.json",
               base / "scored_predictions.json")
