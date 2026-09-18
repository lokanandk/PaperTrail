"""
Stage 4: Filter retrieved literature for relevance and extract structured claims.

All context is derived from the prediction dict at runtime.
No hardcoded disease, tissue, or organism lookup tables.

To improve precision and recall for a prediction, add these optional fields
to predictions.yaml:
  disease_synonyms:       ["full name 1", "alt name 2"]
  cell_type_synonyms:     ["synonym 1", "synonym 2"]
  tissue_synonyms:        ["synonym 1"]
  tier1_journals:         ["journal name 1", "journal name 2"]   # extra journal tier

Relevance formula:
  relevance = 0.40 * alias_match
            + 0.25 * context_match
            + 0.20 * semantic
            + 0.15 * vocab_overlap

All four components are computed from prediction fields only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from rapidfuzz import fuzz

try:
    from semantic_relevance import compute_semantic_scores_for_records
    SEMANTIC_AVAILABLE = True
except Exception as e:
    print(f"[stage4] semantic_relevance unavailable: {e}")
    SEMANTIC_AVAILABLE = False
    def compute_semantic_scores_for_records(*args, **kwargs):
        return [0.0] * len(args[1]) if len(args) > 1 else []

# direction_verifier: optional NLI-based direction claim verification.
# Safe default defined first so NameError can never occur even if the import
# raises something other than ImportError (e.g. a missing transitive dep).
DIRECTION_VERIFY_AVAILABLE = False
def _batch_verify_directions(pairs, use_nli=False): return [1.0] * len(pairs)
try:
    from direction_verifier import batch_verify as _batch_verify_directions
    DIRECTION_VERIFY_AVAILABLE = True
except Exception:
    pass  # direction_verifier.py absent or broken — harmless fallback above


# ─────────────────────────────────────────────────────────────────────────────
# Direction lexicon
# ─────────────────────────────────────────────────────────────────────────────
UP_PATTERNS = [
    r"\b(?:up[- ]?regulat\w+|upregulat\w+|over[- ]?expressed?|over[- ]?expression|"
    r"increased|elevated|enhanced|"
    r"higher|greater|induced|augmented|raised|gained?|hyperactivat\w+|over[- ]?activat\w+|"
    r"amplif\w+|re-express\w+|re-expressed?)\b",
]
DOWN_PATTERNS = [
    r"\b(?:down[- ]?regulat\w+|downregulat\w+|under[- ]?expressed|decreased|reduced|"
    r"diminished|lower|lesser|suppressed|attenuated|impaired|deficient|loss of|loss-of|"
    r"depleted|abolished|silenc\w+|knocked? ?out)\b",
]
PRESERVED_PATTERNS = [
    r"\b(?:preserved|maintained|unchanged|comparable|similar levels?|"
    r"no significant difference|no difference|stable|unaltered|"
    r"not significantly (?:different|changed|altered|reduced|elevated)|"
    r"no significant (?:change|difference|reduction|increase|alteration)|"
    r"similar to (?:controls?|baseline|normal)|"
    r"did not (?:change|differ|decrease|increase)|"
    r"was not (?:significantly|appreciably)|"
    r"consistent (?:expression|levels?|activity)|"
    r"comparable to (?:control|normal))\b",
]
ABSENT_PATTERNS = [
    r"\b(?:absent|undetectable|not detected|complete loss|abolished|silenced)\b",
]
NEGATION_PATTERNS = [
    r"\bnot\s+(?:up\w*|down\w*|elevated|decreased|reduced|increased|altered|changed|significant)\b",
    r"\bno\s+(?:significant\s+)?(?:up\w*|down\w*|change|alteration|difference)\b",
    r"\b(?:fail\w+ to|did not|does not)\b",
]
HEDGE_PATTERNS = [
    r"\b(?:may|might|could|possibly|potentially|likely|suggest\w*|propose\w*|implicat\w*|"
    r"hypothes\w*|speculat\w*|presumed|putative|aims? to|attempt\w*|"
    r"it is (?:possible|likely|known|established|well-known)|"
    r"previous(?:ly)?|earlier (?:studies|work|reports?)|has been (?:shown|reported|demonstrated)|"
    r"\bknown to\b|\bshown to\b|\breported to\b)\b",
]

# Rescue / intervention patterns — when "overexpression" is THERAPEUTIC
# (i.e., used to reverse the disease phenotype), the entity is actually
# DEPLETED in disease.  An UP-word in this context implies disease-state = DOWN.
_RESCUE_REVERSAL_RE = re.compile(
    r"(?:overexpression|overexpress|re-expressed?|forced expression|ectopic expression)"
    r".{0,150}"
    r"(?:reversed?|rescued?|restored?|ameliorated?|attenuated?|protected? against|"
    r"prevented?|suppressed?|abrogated?|alleviated?|mitigated?)"
    , re.IGNORECASE | re.DOTALL)
_REVERSAL_BEFORE_RE = re.compile(
    r"(?:reversed?|rescued?|restored?|ameliorated?|attenuated?|protected? against|"
    r"prevented?|suppressed?|abrogated?|alleviated?|mitigated?)"
    r".{0,150}"
    r"(?:overexpression|overexpress|re-expressed?|forced expression|ectopic expression)"
    , re.IGNORECASE | re.DOTALL)
# "Loss of [entity]", "[entity] deficiency" → entity is DOWN in disease
_LOSS_OF_RE = re.compile(
    r"(?:loss of|deficiency|null mutation|depleted|depletes|lacking|"
    r"knockout|knock-out|silenced|ablated)"
    , re.IGNORECASE)

FOLD_PATTERN     = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:fold|x|times)\s*(?:up|down|increase|decrease|higher|lower)?", re.IGNORECASE)
P_VALUE_PATTERN  = re.compile(r"\bp\s*[<=>]?\s*(?:0?\.\d+|1\.0+|10\^?-?\d+|1[eE]-\d+)", re.IGNORECASE)
N_PATTERN        = re.compile(r"\bn\s*=\s*(\d+)", re.IGNORECASE)
COHENS_D_PATTERN = re.compile(r"(?:Cohen'?s\s+d|effect\s+size)\s*[=:]?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Study quality weighting
# ─────────────────────────────────────────────────────────────────────────────
PUB_TYPE_WEIGHTS = {
    "Meta-Analysis": 1.5, "Systematic Review": 1.4,
    "Randomized Controlled Trial": 1.3, "Multicenter Study": 1.2,
    "Clinical Trial": 1.15, "Comparative Study": 1.1,
    "Review": 0.9, "Journal Article": 1.0,
    "Case Reports": 0.5, "Editorial": 0.4, "Comment": 0.3, "Letter": 0.4,
}

# High-impact general and broad-scope biomedical journals.
# Domain-specific journals can be added per-prediction via:
#   tier1_journals: ["kidney international", "jasn"]
_BASE_TIER1_JOURNALS = {
    "nature", "science", "cell", "lancet", "new england journal of medicine",
    "nejm", "jama", "bmj", "annals of internal medicine",
    "nature medicine", "nature genetics", "nature immunology",
    "nature communications", "cell metabolism", "immunity",
    "journal of experimental medicine", "journal of clinical investigation",
    "proceedings of the national academy of sciences",
}


def _get_tier1_journals(prediction: dict) -> set:
    """
    Return the journal tier set for a prediction.
    Merges the base set with any prediction-specific journals.
    Add prediction-level journals in predictions.yaml:
      tier1_journals: ["kidney international", "jasn", "diabetologia"]
    """
    extra = prediction.get("tier1_journals") or []
    if extra:
        return _BASE_TIER1_JOURNALS | {j.strip().lower() for j in extra}
    return _BASE_TIER1_JOURNALS


# ─────────────────────────────────────────────────────────────────────────────
# Section parsing
# ─────────────────────────────────────────────────────────────────────────────
SECTION_HEADER_PATTERN = re.compile(
    r"\b(BACKGROUND|INTRODUCTION|OBJECTIVE|AIM|AIMS|METHODS|"
    r"MATERIALS AND METHODS|DESIGN|RESULTS|FINDINGS|CONCLUSION|CONCLUSIONS|"
    r"DISCUSSION|INTERPRETATION)\s*[:.\-]",
    re.IGNORECASE,
)
RESULTS_LIKE_SECTIONS = {"results", "findings", "conclusion", "conclusions",
                          "discussion", "interpretation"}


def split_abstract_sections(abstract: str) -> List[tuple]:
    if not abstract:
        return [("all", "")]
    matches = list(SECTION_HEADER_PATTERN.finditer(abstract))
    if not matches:
        return [("all", abstract)]
    sections = []
    for i, m in enumerate(matches):
        name  = m.group(1).lower().strip()
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(abstract)
        sections.append((name, abstract[start:end].strip()))
    return sections


def get_relevant_text_for_extraction(record: dict) -> str:
    title    = record.get("title", "") or ""
    abstract = record.get("abstract", "") or ""
    sections = split_abstract_sections(abstract)
    if len(sections) > 1:
        results_text = " ".join(t for name, t in sections if name in RESULTS_LIKE_SECTIONS)
        if results_text.strip():
            return f"{title}. {results_text}"
    return f"{title}. {abstract}"


# ─────────────────────────────────────────────────────────────────────────────
# Direction extraction
# ─────────────────────────────────────────────────────────────────────────────
def detect_negation_near(text: str, position: int, window: int = 30) -> bool:
    snippet = text[max(0, position - window):position]
    return any(re.search(p, snippet, re.IGNORECASE) for p in NEGATION_PATTERNS)


def detect_hedging_near(text: str, position: int, window: int = 60) -> bool:
    snippet = text[max(0, position - window):position + window]
    return any(re.search(p, snippet, re.IGNORECASE) for p in HEDGE_PATTERNS)


def _get_direction_context_tokens(prediction: dict) -> List[str]:
    """
    Tokens used to weight direction signals by disease co-location in
    extract_direction and _score_sentence_for_direction.

    Only HIGH-SPECIFICITY tokens are used:
    - Multi-word phrases (e.g. "diabetic nephropathy") — specific by definition
    - Single words that are ≥9 chars AND not generic/tissue-only
      (e.g. "nephropathy" ✓, "glomerular" ✓; "diabetic" ✗ — modifies many diseases)

    The rationale: "diabetic" alone matches diabetic retinopathy, neuropathy,
    cardiomyopathy — any diabetic complication. It must NOT grant full context
    weight to papers that are diabetic but not about the predicted disease.
    Multi-word phrases like "diabetic nephropathy" are specific.
    """
    ctx = []

    # From gate tokens: only multi-word phrases or long specific single words
    for t in _get_gate_tokens(prediction):
        if t in _TISSUE_ONLY_TOKENS:
            continue
        if " " in t:
            ctx.append(t)            # multi-word phrase — always specific
        elif len(t) >= 9:
            ctx.append(t)            # long single word — likely specific (e.g. nephropathy)

    # Cell-type synonyms — often very specific (e.g. "proximal tubule",
    # "thick ascending limb", "cortical thick ascending limb")
    for val in _get_all_synonyms(prediction, "cell_type", "cell_type_synonyms"):
        val_l = val.strip().lower()
        if " " in val_l:
            ctx.append(val_l)        # multi-word always specific
        else:
            for w in re.split(r"[\s/,_-]+", val_l):
                if len(w) >= 6 and w not in _TISSUE_ONLY_TOKENS:
                    ctx.append(w)

    # Tissue synonyms (e.g. "renal cortex", "proximal tubule")
    for val in _get_all_synonyms(prediction, "tissue", "tissue_synonyms"):
        val_l = val.strip().lower()
        if " " in val_l and val_l.lower() not in ("any", "systemic"):
            ctx.append(val_l)

    return list(dict.fromkeys(ctx))  # deduplicate, preserve order


def extract_direction(text: str, prediction: dict) -> dict:
    aliases    = [a.lower() for a in prediction.get("aliases", []) if len(a) >= 2]
    aliases_set = set(aliases)
    text_l       = text.lower()
    alias_positions = []

    def _is_different_family_member(pos: int, end: int, alias: str) -> bool:
        """
        Returns True when this match is a family member we don't want.
        e.g., alias="syndecan" matching "syndecan-4" when we want syndecan-1.
        Only fires when:
          - the match is followed by -N or <space>N (isoform number)
          - that full variant (e.g. "syndecan-4") is NOT in our alias list
          - we DO have an alias for a DIFFERENT numbered variant (e.g. "syndecan-1")
        """
        trailer = text_l[end:end + 4]
        # Match both "syndecan-4" (hyphen/space separator) and "SDC4" (direct digit)
        m = re.match(r'([-\s])(\d+)', trailer)   # e.g. "syndecan-4"
        m2 = re.match(r'(\d+)', trailer)           # e.g. "SDC4"
        if not m and not m2:
            return False
        if m:
            variant_num = m.group(2)
            full_variant = alias + m.group(1) + variant_num   # e.g. "syndecan-4"
        else:
            variant_num = m2.group(1)
            full_variant = alias + variant_num                 # e.g. "SDC4"
        if full_variant in aliases_set:
            return False   # this IS our protein variant
        # Check if we have an alias for a different number → different family member
        for a2 in aliases_set:
            if a2.startswith(alias) and len(a2) > len(alias):
                tail = a2[len(alias):]
                m2 = re.match(r'[-\s]?(\d+)$', tail)  # [-\s]? allows no separator
                if m2 and m2.group(1) != variant_num:
                    return True   # our alias ends in different number → skip
        return False

    for a in aliases:
        idx = 0
        while True:
            pos = text_l.find(a, idx)
            if pos < 0: break
            end = pos + len(a)
            if not _is_different_family_member(pos, end, a):
                alias_positions.append((pos, end, a))
            idx = pos + 1
    if not alias_positions:
        return {"signal_direction": None, "confidence": 0,
                "evidence_phrases": [], "counts": {}, "hedge_discount": 0.0, "scores": {"up": 0.0, "down": 0.0, "preserved": 0.0, "absent": 0.0}}

    # Context tokens for co-location weighting.
    # A direction signal found in a window that also contains the disease
    # context is much stronger evidence than one found without that context.
    ctx_toks = _get_direction_context_tokens(prediction)
    has_ctx_constraint = bool(ctx_toks)

    WINDOW = 130
    up_score = down_score = preserved_score = absent_score = 0.0
    up_hits, down_hits, preserved_hits, absent_hits = [], [], [], []
    n_hedged = 0
    _context_anchored_flag = not has_ctx_constraint  # True if no constraint; set True on first ctx hit
    for start, end, alias in alias_positions:
        win = text_l[max(0, start - WINDOW): min(len(text_l), end + WINDOW)]

        # Context weight: 1.0 if at least one disease/cell-type context token
        # is present in this window; 0.5 if the prediction has context constraints
        # but none are co-located with this alias occurrence.
        # If prediction has no context constraint (disease_context="any"), always 1.0.
        if has_ctx_constraint:
            ctx_weight = 1.0 if any(ct in win for ct in ctx_toks) else 0.5
            if ctx_weight == 1.0:
                _context_anchored_flag = True  # at least one anchored signal found
        else:
            ctx_weight = 1.0

        # ── Rescue context detection ────────────────────────────────────────
        # When "overexpression reverses disease", the UP signal is a therapeutic
        # intervention — the actual disease-state direction is DOWN.
        # Detect this in the FULL abstract window (wider than the 130-char window)
        full_context = text_l[max(0, start - 400): min(len(text_l), end + 400)]
        is_rescue_context = bool(
            _RESCUE_REVERSAL_RE.search(full_context) or
            _REVERSAL_BEFORE_RE.search(full_context)
        )
        # "Loss of entity" → reinforces DOWN signal
        has_loss_signal = bool(_LOSS_OF_RE.search(win))

        for ptype, patterns, hits in [
            ("up",        UP_PATTERNS,        up_hits),
            ("down",      DOWN_PATTERNS,      down_hits),
            ("preserved", PRESERVED_PATTERNS, preserved_hits),
            ("absent",    ABSENT_PATTERNS,    absent_hits),
        ]:
            for p in patterns:
                for m in re.finditer(p, win, re.IGNORECASE):
                    gpos = max(0, start - WINDOW) + m.start()
                    if ptype in ("up", "down") and detect_negation_near(text_l, gpos):
                        continue

                    is_hedged = detect_hedging_near(text_l, gpos)
                    base_contribution = (0.4 if is_hedged else 1.0) * ctx_weight

                    # Rescue context: UP signal in a rescue experiment
                    # → this implies disease-state is DOWN, not UP
                    # Reclassify: reduce UP contribution, add to DOWN instead
                    if ptype == "up" and is_rescue_context:
                        # The entity is being overexpressed therapeutically.
                        # Disease-state is DOWN — count as a weak DOWN signal.
                        if is_hedged: n_hedged += 1
                        hits.append((alias, f"[rescue↑→↓] {m.group(0)}", is_hedged))
                        down_score += base_contribution * 0.6   # weaker than direct
                        continue   # do NOT also add to up_score

                    # "Loss of entity" near DOWN word strengthens the signal
                    if ptype == "down" and has_loss_signal:
                        base_contribution *= 1.3

                    contribution = base_contribution
                    if is_hedged: n_hedged += 1
                    hits.append((alias, m.group(0), is_hedged))
                    if ptype == "up":          up_score        += contribution
                    elif ptype == "down":      down_score      += contribution
                    elif ptype == "preserved": preserved_score += contribution
                    elif ptype == "absent":    absent_score    += contribution
    counts = {"up": len(up_hits), "down": len(down_hits),
              "preserved": len(preserved_hits), "absent": len(absent_hits)}
    scores = {"up": up_score, "down": down_score,
              "preserved": preserved_score, "absent": absent_score}
    total_score = sum(scores.values())
    if total_score == 0:
        return {"signal_direction": None, "confidence": 0,
                "evidence_phrases": [], "counts": counts, "hedge_discount": 0.0,
                "scores": {"up": 0.0, "down": 0.0, "preserved": 0.0, "absent": 0.0},
                "context_anchored": False}
    sorted_dirs = sorted(scores.items(), key=lambda kv: -kv[1])
    top_dir, top_score = sorted_dirs[0]
    second      = sorted_dirs[1][1] if len(sorted_dirs) > 1 else 0
    conf        = (top_score - second) / max(top_score, 1e-9)
    total_hits  = sum(counts.values())
    hedge_disc  = (n_hedged / total_hits) if total_hits else 0.0
    evidence    = (up_hits + down_hits + preserved_hits + absent_hits)[:8]

    # context_anchored: True if at least one direction signal co-occurred
    # with a disease/cell-type context token in the extraction window.
    # When False and the prediction has context constraints, confidence is
    # halved — direction found with no disease co-location is weak evidence.
    # This is the key discriminator between DKD-specific and off-context papers.
    context_anchored = (not has_ctx_constraint) or (top_score > total_score * 0.5)
    # Note: if ctx_weight=1.0 signals dominate (>50% of total score), anchored=True.
    # If all signals were ctx_weight=0.5 (no context co-location), score is all-halved
    # and this still equals total_score * 0.5 exactly. Use a different check:
    # Compare unweighted vs weighted: if top_score is close to total_score/2, not anchored.
    # Simpler: track whether any window had ctx_weight=1.0
    context_anchored = _context_anchored_flag  # set during the loop above
    if has_ctx_constraint and not context_anchored:
        conf = round(conf * 0.5, 3)   # halve confidence for unanchored signals

    return {
        "signal_direction": top_dir,
        "confidence": round(conf, 3),
        "counts": counts,
        "scores": {k: round(v, 2) for k, v in scores.items()},
        "evidence_phrases": [{"alias": a, "phrase": p, "hedged": h} for a, p, h in evidence],
        "hedge_discount": round(hedge_disc, 3),
        "context_anchored": context_anchored,
    }


def extract_quantitative(text: str) -> dict:
    return {
        "fold_changes":  [float(m.group(1)) for m in FOLD_PATTERN.finditer(text)][:5],
        "p_values":      [m.group(0) for m in P_VALUE_PATTERN.finditer(text)][:5],
        "sample_sizes":  [int(m.group(1)) for m in N_PATTERN.finditer(text)][:5],
        "cohens_d":      [float(m.group(1)) for m in COHENS_D_PATTERN.finditer(text)][:3],
    }


# ─────────────────────────────────────────────────────────────────────────────
# PMC full-text support
# ─────────────────────────────────────────────────────────────────────────────
_ABBREV_RE = re.compile(
    r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|Fig|et al|e\.g|i\.e|approx|'
    r'Eq|No|vol|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.'
)

def _split_sentences(text: str) -> List[str]:
    text  = _ABBREV_RE.sub(lambda m: m.group().replace('.', '\x00'), text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\(\[])', text)
    return [p.replace('\x00', '.').strip() for p in parts if len(p.strip()) > 20]


def extract_direction_from_pmc(pmc_sentences: list, prediction: dict) -> dict:
    if not pmc_sentences:
        return {"signal_direction": None, "confidence": 0,
                "evidence_phrases": [], "counts": {}, "hedge_discount": 0.0, "scores": {"up": 0.0, "down": 0.0, "preserved": 0.0, "absent": 0.0}}
    return extract_direction("  ".join(pmc_sentences), prediction)


def _score_sentence_for_direction(sent: str, expected_dir: str, aliases: list,
                                   context_toks: list = None) -> float:
    """
    Score a sentence as evidence for a directional prediction.

    Scoring components:
    - Alias present in sentence (required, else 0)
    - Direction words matching expected direction (+0.5 per hit)
    - Opposing direction words (-0.3 per hit)
    - Quantitative evidence (fold change, p-value, n=)
    - Disease/cell-type context co-location: +0.4 bonus if the sentence
      also contains a context token — this ensures excerpts shown in the
      dashboard are from disease-relevant sentences, not generic ones.
    """
    s = sent.lower()
    if not any(a.lower() in s for a in aliases if len(a) >= 2):
        return 0.0
    score    = 0.1
    up_hits  = sum(bool(re.search(p, sent, re.I)) for p in UP_PATTERNS)
    dn_hits  = sum(bool(re.search(p, sent, re.I)) for p in DOWN_PATTERNS)
    pr_hits  = sum(bool(re.search(p, sent, re.I)) for p in PRESERVED_PATTERNS)
    ab_hits  = sum(bool(re.search(p, sent, re.I)) for p in ABSENT_PATTERNS)
    expected_hits = {"up": up_hits, "down": dn_hits, "preserved": pr_hits,
                     "absent": ab_hits}.get(expected_dir, 0)
    score += 0.5 * expected_hits
    opposite_hits = (dn_hits if expected_dir == "up" else
                     up_hits if expected_dir in ("down", "absent") else 0)
    score -= 0.3 * opposite_hits
    if FOLD_PATTERN.search(sent):    score += 0.2
    if P_VALUE_PATTERN.search(sent): score += 0.15
    if N_PATTERN.search(sent):       score += 0.1
    # Significant bonus for sentences where direction co-occurs with disease context.
    # This is the key fix: a sentence saying "SHMT2 was reduced in diabetic nephropathy"
    # scores much higher than "SHMT2 was reduced" with no disease context.
    if context_toks and any(ct in s for ct in context_toks):
        score += 0.4
    return score


def build_best_excerpt(record: dict, prediction: dict, max_chars: int = 500) -> Tuple[str, str]:
    aliases      = prediction.get("aliases", [prediction.get("entity", "")])
    expected_dir = prediction.get("direction", "up")
    # Disease/cell-type context tokens for sentence preference
    ctx_toks = _get_direction_context_tokens(prediction)
    def _top(sents: list, n: int) -> str:
        scored = [(_score_sentence_for_direction(s, expected_dir, aliases, ctx_toks), s)
                  for s in sents if len(s) > 25]
        scored.sort(key=lambda x: -x[0])
        return "  ".join(s for sc, s in scored[:n] if sc > 0)
    pmc_sents = record.get("pmc_sentences", [])
    if pmc_sents:
        exc = _top(pmc_sents, n=3)
        if exc: return exc[:max_chars], "full_text"
    abstract = record.get("abstract", "") or ""
    if abstract:
        exc = _top(_split_sentences(abstract), n=2)
        if exc: return exc[:max_chars], "abstract"
        return abstract[:400], "abstract"
    return "", "abstract"


# ─────────────────────────────────────────────────────────────────────────────
# Disease gate — derived from prediction fields, no hardcoded tables
# ─────────────────────────────────────────────────────────────────────────────

# Generic medical/scientific words that must NOT be individual gate tokens.
# These appear in virtually every biomedical abstract and would pass anything.
# They remain usable as PARTS of multi-word phrases (e.g. "kidney disease" as
# a full phrase is fine; the individual word "disease" alone is not).
_GENERIC_GATE_WORDS = frozenset({
    "disease", "diseases", "disorder", "disorders", "syndrome", "syndromes",
    "condition", "conditions", "related", "associated", "induced", "dependent",
    "type", "types", "stage", "stages", "form", "forms", "factor", "factors",
    "patient", "patients", "subject", "subjects", "study", "studies",
    "model", "models", "effect", "effects", "level", "levels",
    "activity", "function", "expression", "pathway", "signaling",
})


def _get_gate_tokens(prediction: dict) -> List[str]:
    """
    Derive disease gate tokens from the prediction dict.

    Rules:
    - Entity aliases are EXCLUDED (alias_match_score handles entity presence).
    - Generic medical words (disease, disorder, syndrome, condition…) are
      EXCLUDED as individual tokens — they match everything.
    - Multi-word phrases from disease_synonyms are always included in full.
    - Individual word tokens from synonyms are only added if they are
      disease-specific (not generic, not tissue-only, not alias words).
    """
    dc = (prediction.get("disease_context") or "any").strip()
    if dc.lower() == "any":
        return []

    # Build alias set to exclude from gate tokens
    alias_words = set()
    for a in (prediction.get("aliases") or []):
        for w in re.split(r"[\s/,_-]+", a.lower()):
            if len(w) >= 2:
                alias_words.add(w)
    entity = (prediction.get("entity") or "").lower()
    if entity:
        alias_words.add(entity)

    # Combined exclusion set for individual word tokens
    skip_words = alias_words | _GENERIC_GATE_WORDS | _TISSUE_ONLY_TOKENS

    tokens = set()

    # 1. disease_synonyms — full phrases are always added.
    #    Individual words only if disease-specific (not in skip_words).
    for syn in (prediction.get("disease_synonyms") or []):
        s = str(syn).strip().lower()
        if not s:
            continue
        tokens.add(s)          # always add the full phrase
        for w in re.split(r"[\s/,]+", s):
            if len(w) >= 4 and w not in skip_words:
                tokens.add(w)  # e.g. "nephropathy", "erythematosus"

    # 2. disease_context itself.
    #    Short abbreviations (DKD, SLE, RA) are kept as-is — they are specific.
    #    Longer strings are tokenised; individual words filtered by skip_words.
    dc_lower = dc.lower()
    if dc_lower not in alias_words:
        tokens.add(dc_lower)   # the full string, e.g. "dkd"
    words = [w for w in re.split(r"[\s/,_]+", dc_lower)
             if len(w) >= 3 and w not in skip_words]
    for w in words:
        tokens.add(w)
    # Bigrams from disease_context (multi-word contexts like "kidney injury")
    for i in range(len(words) - 1):
        tokens.add(f"{words[i]} {words[i+1]}")

    tokens.discard("")
    return list(tokens)



def _get_exclusion_tokens(prediction: dict) -> List[str]:
    """
    Derive exclusion tokens from prediction["disease_exclusions"].

    Records that match any of these tokens are candidates for rejection.
    Add to predictions.yaml:
      disease_exclusions: ["renal carcinoma", "RCC", "clear cell carcinoma"]

    IMPORTANT: unlike the inclusion gate which tokenises aggressively,
    exclusion tokens are kept as full phrases to avoid false matches.
    A multi-word entry like "renal cell carcinoma" is used as-is (requiring
    the full phrase), not split into "renal", "cell", "carcinoma" separately.
    Only single-word entries (e.g. "RCC") are used verbatim as single tokens.

    This prevents innocuous words like "cell" or "renal" from causing papers
    to be incorrectly penalised.
    """
    tokens = []
    for ex in (prediction.get("disease_exclusions") or []):
        s = str(ex).strip().lower()
        if not s or len(s) < 3:
            continue
        # Always add the full phrase — this is the primary matching term
        tokens.append(s)
        # For single-word entries that may appear as abbreviations in text,
        # also try lowercase (already done above) — no further splitting.
        # For multi-word entries: do NOT split into individual words.
    # Deduplicate preserving order
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t); out.append(t)
    return out


# Tokens that are tissue/organ words — they match many papers including cancer.
# A record that passes the gate ONLY on these words, with no disease-specific
# token, is not reliably on-topic and needs stricter checking.
_TISSUE_ONLY_TOKENS = frozenset({
    "kidney", "renal", "liver", "hepatic", "lung", "pulmonary",
    "brain", "neural", "cardiac", "heart", "muscle", "bone",
    "tissue", "cell", "cells", "organ",
})

# Signals that a paper is about a neoplastic/tumour context.
_TUMOUR_SIGNALS = frozenset({
    "cancer", "tumor", "tumour", "neoplasm", "neoplasia", "malignant",
    "malignancy", "carcinoma", "sarcoma", "oncology", "oncogenic",
    "metastasis", "metastatic", "xenograft", "cell line", "786-o", "caki",
    "vhl", "ccrcc", "ccRCC", "kidney neoplasm", "renal neoplasm",
    "renal tumor", "renal tumour", "kidney tumor", "kidney tumour",
    "kidney cancer", "renal cancer", "tumorigenic", "tumorigenesis",
})

# Signals that a paper studies ONLY healthy/normal tissue with no disease comparator.
# A paper studying healthy kidney only is not relevant to a DKD prediction.
_HEALTHY_ONLY_SIGNALS = frozenset({
    "healthy donor", "healthy donors", "healthy volunteer", "healthy volunteers",
    "healthy subject", "healthy subjects", "healthy individual", "healthy individuals",
    "normal subject", "normal subjects", "normal volunteer", "normal volunteers",
    "normal tissue only", "non-diseased", "disease-free", "pathology-free",
})


def passes_disease_gate(record: dict, prediction: dict) -> bool:
    """
    Hard filter combining inclusion and exclusion checks.

    Inclusion rule:
      The record must mention at least one disease/context token.
      HOWEVER: if the only matching tokens are generic tissue words
      (kidney, renal, liver…) AND the record also contains tumour signals
      (cancer, tumor, carcinoma, cell line…), the record is blocked —
      a kidney cancer paper must not pass just because it mentions "kidney".

    Exclusion rule:
      If any disease_exclusions phrase matches AND the correct disease has
      fewer than 2 specific token hits, the record is blocked.

    disease_context="any" passes unconditionally.
    tissue="systemic" skips the inclusion check.
    """
    title    = record.get("title", "") or ""
    abstract = record.get("abstract", "") or ""
    mesh     = " ".join(record.get("mesh_terms", []) or [])
    combo    = f"{title} {abstract} {mesh}".lower()

    gate_tokens = _get_gate_tokens(prediction)

    # ── Inclusion gate ────────────────────────────────────────────────────────
    if gate_tokens and prediction.get("tissue", "").lower() != "systemic":
        matching = [tok for tok in gate_tokens if tok in combo]
        if not matching:
            return False   # no disease context at all

        # Check if ALL matching tokens are tissue-only words
        specific_matches = [t for t in matching if t not in _TISSUE_ONLY_TOKENS]
        if not specific_matches:
            # Only tissue words matched (e.g. "kidney" alone).
            # Block if the record has a tumour/cancer signal.
            if any(sig in combo for sig in _TUMOUR_SIGNALS):
                return False
            # Also block papers studying ONLY healthy/normal tissue
            # (no disease comparator) — not relevant to disease predictions.
            if any(sig in combo for sig in _HEALTHY_ONLY_SIGNALS):
                return False

    # ── Exclusion gate ────────────────────────────────────────────────────────
    excl_tokens = _get_exclusion_tokens(prediction)
    if excl_tokens:
        excl_hits = [tok for tok in excl_tokens if tok in combo]
        if excl_hits:
            correct_hits = sum(1 for tok in gate_tokens if tok in combo
                               and tok not in _TISSUE_ONLY_TOKENS)
            if correct_hits < 2:
                return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Relevance scoring — all from prediction fields
# ─────────────────────────────────────────────────────────────────────────────

def _get_all_synonyms(prediction: dict, field: str, syn_field: str) -> List[str]:
    """Return primary value + synonyms for a prediction field."""
    primary = (prediction.get(field) or "any").strip()
    values  = [] if primary.lower() == "any" else [primary]
    for syn in (prediction.get(syn_field) or []):
        s = str(syn).strip()
        if s and s not in values:
            values.append(s)
    return values


def _build_vocab(prediction: dict) -> List[str]:
    """
    Build a relevance vocabulary from all prediction fields.
    Used by vocab_overlap_score.
    """
    vocab = set()
    for tok in _get_gate_tokens(prediction):
        vocab.add(tok)
    for val in _get_all_synonyms(prediction, "cell_type", "cell_type_synonyms"):
        for w in re.split(r"[\s/,_-]+", val.lower()):
            if len(w) >= 4:
                vocab.add(w)
    for val in _get_all_synonyms(prediction, "tissue", "tissue_synonyms"):
        for w in re.split(r"[\s/,_-]+", val.lower()):
            if len(w) >= 4:
                vocab.add(w)
    for a in prediction.get("aliases", []):
        al = a.strip().lower()
        if 3 <= len(al) <= 20:
            vocab.add(al)
    vocab.discard("")
    return list(vocab)


def contrastive_penalty(record: dict, prediction: dict) -> float:
    """
    Score penalty (0.0–0.35) applied when a record mentions excluded contexts.

    Unlike passes_disease_gate (binary), this provides a soft down-ranking:
    papers that mention the wrong disease lose relevance score continuously
    proportional to how dominant the excluded context is.

    Returns a penalty value to subtract from the relevance score.
    No penalty if disease_exclusions is absent.
    """
    excl_tokens = _get_exclusion_tokens(prediction)
    if not excl_tokens:
        return 0.0

    combo = (record.get("title", "") + " " + record.get("abstract", "") + " " +
             " ".join(record.get("mesh_terms", []) or [])).lower()

    excl_hits    = sum(1 for tok in excl_tokens    if tok in combo)
    correct_hits = sum(1 for tok in _get_gate_tokens(prediction) if tok in combo)

    if excl_hits == 0:
        return 0.0

    # Penalty scales with how many exclusion tokens match relative to correct tokens
    # Full penalty (0.35) only when exclusively about excluded context
    ratio = excl_hits / max(excl_hits + correct_hits, 1)
    return round(min(ratio * 0.35, 0.35), 3)


def cell_type_specificity_penalty(record: dict, prediction: dict) -> float:
    """
    Small penalty (0.0–0.15) when a cell-type-specific prediction has no
    cell-type evidence in the paper at all.

    A paper about SHMT2 in whole kidney without mentioning tubular cells,
    proximal tubule, or C_TAL is less relevant than one that does.
    Only applied when the prediction specifies a non-generic cell type.
    """
    ct_vals = _get_all_synonyms(prediction, "cell_type", "cell_type_synonyms")
    if not ct_vals or ct_vals[0].lower() in ("any", ""):
        return 0.0

    text     = (record.get("title", "") + " " + record.get("abstract", "")).lower()
    ct_words = [w for v in ct_vals for w in re.split(r"[\s/,_-]+", v.lower()) if len(w) >= 3]
    if any(w in text for w in ct_words[:6]):
        return 0.0   # cell type mentioned — no penalty
    return 0.10      # cell type not mentioned at all


def context_match_score(record: dict, prediction: dict) -> float:
    """
    Score how well a record matches the prediction context.
    Derived entirely from prediction fields.
    """
    score = 0.0
    text  = (record.get("title", "") + " " + record.get("abstract", "")).lower()
    mesh  = " ".join(record.get("mesh_terms", [])).lower()
    full  = text + " " + mesh

    # Disease match (0.4)
    gate_tokens = _get_gate_tokens(prediction)
    if gate_tokens:
        if any(tok in full for tok in gate_tokens[:8]):
            score += 0.4
    else:
        score += 0.2

    # Cell type match (0.3)
    ct_vals = _get_all_synonyms(prediction, "cell_type", "cell_type_synonyms")
    if ct_vals:
        ct_words = [w for v in ct_vals for w in re.split(r"[\s/,_-]+", v.lower()) if len(w) >= 3]
        if any(w in text for w in ct_words[:6]):
            score += 0.3
    else:
        score += 0.15

    # Tissue match (0.2)
    # This replaces the old if tissue == "kidney" branch with a generic version
    tis_vals = _get_all_synonyms(prediction, "tissue", "tissue_synonyms")
    if tis_vals and tis_vals[0].lower() not in ("any", "systemic"):
        tis_words = [w for v in tis_vals for w in re.split(r"[\s/,_-]+", v.lower()) if len(w) >= 3]
        # Also check gate tokens as a fallback (disease tokens often contain tissue words)
        all_tis = tis_words[:4] + [t for t in gate_tokens[:4] if len(t) >= 4]
        if any(w in full for w in all_tis):
            score += 0.2
    else:
        score += 0.1

    # Organism match (0.1)
    org = (prediction.get("organism") or "any").strip().lower()
    if org not in ("any", ""):
        if org in full or "human" in full or "patient" in full:
            score += 0.1
    else:
        score += 0.1

    return min(score, 1.0)


def alias_match_score(record: dict, prediction: dict) -> tuple:
    text = (record.get("title", "") + " " + record.get("abstract", "")).lower()
    best, best_a = 0, None
    for a in prediction.get("aliases", []):
        al = a.lower()
        if al in text:
            return 100.0, a
        sc = fuzz.partial_ratio(al, text)
        if sc > best:
            best, best_a = sc, a
    return float(best), best_a


def vocab_overlap_score(record: dict, prediction: dict = None) -> float:
    """
    Vocab overlap between record and prediction context vocabulary.

    Uses a fixed denominator of 10 (matching the old KIDNEY_DISEASE_VOCAB
    normalisation) to keep scores stable and comparable across predictions,
    regardless of how many vocab terms are derived.
    """
    if not prediction:
        return 0.0
    text  = (record.get("title", "") + " " + record.get("abstract", "") + " " +
             " ".join(record.get("mesh_terms", []))).lower()
    vocab = _build_vocab(prediction)
    if not vocab:
        return 0.0
    hits  = sum(1 for term in vocab if term in text)
    # Fixed denominator 10 — matches old KIDNEY_DISEASE_VOCAB normalisation
    # and keeps scores stable regardless of vocabulary size
    return min(hits / 10.0, 1.0)


def study_quality_weight(record: dict, prediction: dict = None) -> float:
    """
    Compute study quality weight.
    prediction is optional — if provided, checks prediction-specific tier1_journals.
    """
    pub_types = record.get("publication_types", [])
    base = 1.0
    for pt in pub_types:
        if pt in PUB_TYPE_WEIGHTS:
            base = max(base, PUB_TYPE_WEIGHTS[pt])
    journal  = (record.get("journal", "") or "").lower()
    tier1    = _get_tier1_journals(prediction) if prediction else _BASE_TIER1_JOURNALS
    if any(j in journal for j in tier1):
        base *= 1.15
    abstract_text = record.get("abstract", "") or ""
    ns = [n for n in extract_quantitative(abstract_text).get("sample_sizes", [])
          if 5 <= n <= 100000]
    sample_max = max(ns) if ns else 0
    if sample_max >= 500:   base *= 1.15
    elif sample_max >= 100: base *= 1.05
    elif sample_max == 0:   base *= 0.95
    return round(min(base, 2.0), 3)


def detect_study_context(record: dict) -> str:
    pub_types = record.get("publication_types", []) or []
    interv_pt = {"Randomized Controlled Trial", "Clinical Trial",
                  "Clinical Trial, Phase I", "Clinical Trial, Phase II",
                  "Clinical Trial, Phase III"}
    if any(pt in interv_pt for pt in pub_types):
        return "interventional"
    text = (record.get("title", "") + " " + record.get("abstract", "")).lower()
    if any(tok in text for tok in [
        "randomized", "randomised", "placebo-controlled", "double-blind",
        "treatment with", "supplementation", "supplemented", "administered",
        "dosing", "intervention",
    ]):
        return "interventional"
    return "observational"


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def extract_all(literature_path: Path, predictions_path: Path,
                output_path: Path, semantic_model: Optional[str] = None) -> None:
    with open(literature_path)  as f: lit   = json.load(f)
    with open(predictions_path) as f: preds = yaml.safe_load(f)["predictions"]
    pred_by_id = {p["id"]: p for p in preds}

    extracted = {}
    for pred_id, payload in lit.items():
        prediction = pred_by_id.get(pred_id)
        if not prediction:
            continue

        gated_records = [r for r in payload.get("records", [])
                         if passes_disease_gate(r, prediction)]

        if SEMANTIC_AVAILABLE and gated_records:
            semantic_scores = compute_semantic_scores_for_records(
                prediction, gated_records, model_name=semantic_model)
        else:
            semantic_scores = [0.0] * len(gated_records)

        rows = []
        for rec, sem_score in zip(gated_records, semantic_scores):
            text_full = (rec.get("title", "") + ". " + (rec.get("abstract", "") or ""))
            if not text_full.strip() or text_full.strip() == ".":
                continue

            am_score, am_alias = alias_match_score(rec, prediction)
            ctx     = context_match_score(rec, prediction)
            vocab   = vocab_overlap_score(rec, prediction)
            quality = study_quality_weight(rec, prediction)

            # Base relevance
            relevance = (0.40 * (am_score / 100.0) + 0.25 * ctx
                         + 0.20 * sem_score + 0.15 * vocab)

            # Contrastive penalties: subtract for wrong-context papers
            penalty = (contrastive_penalty(rec, prediction)
                       + cell_type_specificity_penalty(rec, prediction))
            relevance = max(0.0, round(relevance - penalty, 3))

            if sem_score > 0.7:          min_relevance = 0.28
            elif not SEMANTIC_AVAILABLE: min_relevance = 0.30
            else:                        min_relevance = 0.38
            if relevance < min_relevance:
                continue

            direction_text     = get_relevant_text_for_extraction(rec)
            direction          = extract_direction(direction_text, prediction)
            quant              = extract_quantitative(text_full)
            study_ctx          = detect_study_context(rec)
            pmc_sents          = rec.get("pmc_sentences", [])
            pmc_direction      = None
            direction_conflict = False

            if pmc_sents:
                pmc_direction = extract_direction_from_pmc(pmc_sents, prediction)
                if (direction["signal_direction"] is None and
                        pmc_direction["signal_direction"] is not None):
                    direction = pmc_direction
                elif (direction["signal_direction"] is not None and
                      pmc_direction["signal_direction"] is not None and
                      direction["signal_direction"] != pmc_direction["signal_direction"]):
                    direction_conflict = True
                quant = extract_quantitative(text_full + "  " + "  ".join(pmc_sents))

            # ── NLI direction verification (optional) ─────────────────────────
            # Verifies the extracted direction is a disease-state claim rather than
            # a therapeutic intervention or off-context signal.
            # Needs direction_verifier.py; model (~84 MB) downloads on first use.
            # Graceful fallback: if unavailable, multiplier = 1.0 (no change).
            if direction.get("signal_direction") and DIRECTION_VERIFY_AVAILABLE:
                entity_name  = prediction.get("entity", "")
                disease_name = ((prediction.get("disease_synonyms") or []) + [
                    prediction.get("disease_context", "")])[0]
                verify_pairs = [(direction_text[:600], entity_name,
                                 direction["signal_direction"], disease_name)]
                nli_mult = _batch_verify_directions(verify_pairs, use_nli=True)[0]
                direction = dict(direction)
                direction["nli_multiplier"] = round(nli_mult, 3)
                if nli_mult < 0.5:
                    direction["confidence"] = round(direction.get("confidence", 0) * nli_mult, 3)
                    direction["nli_verified"] = False
                else:
                    direction["nli_verified"] = True

            best_excerpt, excerpt_source = build_best_excerpt(rec, prediction)

            rows.append({
                "pmid": rec["pmid"], "title": rec.get("title"),
                "journal": rec.get("journal"), "year": rec.get("year"),
                "publication_types": rec.get("publication_types", []),
                "alias_match_score": am_score, "alias_matched": am_alias,
                "context_score": round(ctx, 3), "semantic_score": round(sem_score, 3),
                "vocab_score": round(vocab, 3), "quality_weight": quality,
                "relevance": round(relevance, 3), "context_penalty": round(penalty, 3),
                "study_context": study_ctx,
                "direction_extraction": direction, "pmc_direction": pmc_direction,
                "direction_conflict": direction_conflict,
                "quantitative_extraction": quant,
                "abstract_excerpt": rec.get("abstract", "")[:400],
                "best_excerpt": best_excerpt, "excerpt_source": excerpt_source,
                "has_pmc": bool(pmc_sents),
            })

        rows.sort(key=lambda r: -r["relevance"])
        extracted[pred_id] = {
            "prediction": prediction,
            "n_relevant": len(rows),
            "n_after_gate": len(gated_records),
            "n_before_gate": len(payload.get("records", [])),
            "evidence": rows,
        }

    with open(output_path, "w") as f:
        json.dump(extracted, f, indent=2, default=str)
    print(f"Wrote {output_path}")
    n_with      = sum(1 for v in extracted.values() if v["n_relevant"] > 0)
    n_gated_out = sum(v["n_before_gate"] - v["n_after_gate"] for v in extracted.values())
    print(f"  predictions with ≥1 relevant record: {n_with}/{len(extracted)}")
    print(f"  total records rejected by disease gate: {n_gated_out}")


if __name__ == "__main__":
    base = Path(__file__).parent
    extract_all(base / "literature_raw.json",
                base / "predictions_expanded.yaml",
                base / "extracted_evidence.json")
