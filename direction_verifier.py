"""
direction_verifier.py — NLI-based direction claim verification.

Addresses the fundamental limitation of regex-based direction extraction:
regex finds direction WORDS but cannot tell whether they describe:
  (a) the disease-state direction  ← what we want
  (b) a therapeutic intervention   ← false concordant/discordant
  (c) a different disease/context  ← off-context noise

Solution: Natural Language Inference (NLI). For each extracted direction
signal, form a hypothesis:
  "In [disease], [entity] expression is [increased/decreased]"
and check whether the source sentence ENTAILS this hypothesis.

Two backends (used in priority order, both free):

  1. cross-encoder/nli-deberta-v3-small (~84 MB, from HuggingFace)
     Zero-shot NLI, CPU-runnable, ~0.05s per pair.
     Same sentence-transformers package already used for SapBERT.

  2. Local heuristic fallback (always available, no download)
     Regex-based intervention/rescue detection.

Usage in stage4 (called once per record after extract_direction):
  from direction_verifier import verify_direction
  multiplier = verify_direction(sentence, entity, direction, disease)
  # multiplier in [0.0, 0.5, 1.0] — multiply into direction score weight
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# NLI model (optional — graceful fallback if not installed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from sentence_transformers import CrossEncoder
    _NLI_AVAILABLE = True
except ImportError:
    _NLI_AVAILABLE = False


_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"  # 84 MB, Apache 2.0
_NLI_LABELS = ["contradiction", "entailment", "neutral"]


@lru_cache(maxsize=1)
def _load_nli_model():
    """Load the NLI cross-encoder. Cached after first call."""
    if not _NLI_AVAILABLE:
        return None
    try:
        print(f"[direction_verifier] Loading NLI model: {_NLI_MODEL_NAME}")
        return CrossEncoder(_NLI_MODEL_NAME)
    except Exception as e:
        print(f"[direction_verifier] NLI model load failed: {e} — using heuristic fallback")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic fallback (always available)
# ─────────────────────────────────────────────────────────────────────────────

# Patterns indicating this sentence describes an INTERVENTION, not a disease state.
# When these fire near the entity + direction word, the concordance is unreliable.
_INTERVENTION_PATTERNS = [
    # Overexpression/gene therapy used as treatment
    re.compile(
        r"(?:overexpression|re-expression|forced expression|ectopic expression|"
        r"adenoviral|lentiviral|vector-mediated|exogenous)"
        r".{0,100}"
        r"(?:reversed?|rescued?|restored?|ameliorat|attenuated?|protected?|prevented?|"
        r"suppressed?|abrogat|alleviat|mitigat)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"(?:reversed?|rescued?|restored?|ameliorat|attenuated?|protected?|prevented?)"
        r".{0,100}"
        r"(?:overexpression|re-expression|forced expression|ectopic expression)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Recombinant protein / drug treatment
    re.compile(
        r"(?:administration of|treatment with|recombinant|supplementation with)"
        r".{0,60}"
        r"(?:reduced?|decreased?|attenuated?|improved?|restored?)",
        re.IGNORECASE | re.DOTALL,
    ),
    # Knockdown/knockout CAUSING disease = entity IS DOWN in disease
    # (This one is correct direction, not an intervention confound — we skip it)
]

_DIRECTION_PHRASES = {
    "up":        "expression is increased, elevated, or upregulated",
    "down":      "expression is decreased, reduced, or downregulated",
    "preserved": "expression is unchanged or preserved",
    "absent":    "expression is absent or undetectable",
}


def _heuristic_multiplier(sentence: str, direction: str) -> float:
    """
    Rule-based check. Returns:
      1.0 — no intervention signal detected
      0.4 — intervention context detected (direction word from therapeutic context)
    """
    if not sentence:
        return 1.0
    for pat in _INTERVENTION_PATTERNS:
        if pat.search(sentence):
            # Intervention detected. If direction is "up" (overexpression therapeutic),
            # this is suspicious — it means entity is likely DOWN in disease.
            if direction == "up":
                return 0.2   # strong penalty: this UP is from rescue experiment
            else:
                return 0.6   # mild penalty: direction may still be informative
    return 1.0


# ─────────────────────────────────────────────────────────────────────────────
# NLI-based verification
# ─────────────────────────────────────────────────────────────────────────────

def _nli_multiplier(sentence: str, entity: str, direction: str,
                    disease: str, model) -> float:
    """
    Use zero-shot NLI to verify: does this sentence entail that
    [entity] is [up/down] in [disease]?

    Returns a multiplier [0.0, 1.0]:
      entailment     → 1.0 (sentence supports the directional claim)
      neutral        → 0.7 (sentence doesn't clearly support or contradict)
      contradiction  → 0.1 (sentence contradicts the claim)
    """
    direction_phrase = _DIRECTION_PHRASES.get(direction, f"shows {direction} expression")
    hypothesis = f"In {disease}, {entity} {direction_phrase}."

    try:
        # CrossEncoder NLI returns logits in order [contradiction, entailment, neutral]
        import numpy as np
        scores = model.predict([[sentence, hypothesis]])
        scores = scores[0] if len(scores.shape) > 1 else scores
        # Apply softmax
        exp_s = np.exp(scores - np.max(scores))
        probs = exp_s / exp_s.sum()
        # Order: contradiction=0, entailment=1, neutral=2
        contradiction_p = float(probs[0])
        entailment_p    = float(probs[1])
        neutral_p       = float(probs[2])

        if entailment_p > 0.50:
            return 1.0
        elif contradiction_p > 0.50:
            return 0.1   # strong contradiction
        elif contradiction_p > 0.30:
            return 0.3   # moderate contradiction
        else:
            return 0.7   # neutral — neither supports nor contradicts

    except Exception:
        return 0.7   # fallback to neutral on any error


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def verify_direction(
    sentence: str,
    entity: str,
    direction: str,
    disease: str = "",
    use_nli: bool = True,
) -> float:
    """
    Verify a directional claim extracted from text.

    Args:
        sentence:  The sentence containing the direction signal.
        entity:    The entity name (gene, protein, metabolite etc.).
        direction: Extracted direction ("up", "down", "preserved", "absent").
        disease:   Disease context (e.g. "DKD", "diabetic nephropathy").
        use_nli:   If True, attempt NLI verification (requires model download).

    Returns:
        A float multiplier [0.0, 1.0]:
          1.0 → claim is well-supported
          0.5 → uncertain / neutral
          0.0 → claim is contradicted or likely wrong

    The multiplier is used to scale the direction score contribution in stage4.
    Papers where the direction signal comes from a rescue experiment receive
    lower weight; papers with direct disease-state claims receive full weight.
    """
    if not sentence or not entity:
        return 1.0

    # Step 1: Heuristic check (always runs, fast)
    heuristic = _heuristic_multiplier(sentence, direction)

    # If heuristic is already very low, no need for NLI
    if heuristic < 0.3:
        return heuristic

    # Step 2: NLI check (optional, slower)
    if use_nli and disease:
        model = _load_nli_model()
        if model is not None:
            nli_score = _nli_multiplier(sentence, entity, direction, disease, model)
            # Combine: take the more conservative (lower) of the two
            return min(heuristic, nli_score) if nli_score < 0.5 else heuristic * nli_score

    return heuristic


def batch_verify(
    pairs: list,   # list of (sentence, entity, direction, disease) tuples
    use_nli: bool = True,
) -> list:
    """
    Batch verification for multiple direction claims.
    More efficient than calling verify_direction() in a loop when NLI is used,
    because the CrossEncoder can process pairs in batches.

    Returns a list of multipliers in the same order as input pairs.
    """
    if not pairs:
        return []

    # Heuristic pass (always)
    heuristic_scores = [
        _heuristic_multiplier(sent, direction)
        for sent, entity, direction, disease in pairs
    ]

    if not use_nli:
        return heuristic_scores

    # Filter pairs that need NLI (heuristic not already very low, disease known)
    model = _load_nli_model()
    if model is None:
        return heuristic_scores

    nli_indices = [
        i for i, (sent, entity, direction, disease) in enumerate(pairs)
        if heuristic_scores[i] >= 0.3 and disease
    ]
    if not nli_indices:
        return heuristic_scores

    # Batch NLI
    import numpy as np
    nli_pairs = []
    for i in nli_indices:
        sent, entity, direction, disease = pairs[i]
        dp = _DIRECTION_PHRASES.get(direction, f"shows {direction} expression")
        hypothesis = f"In {disease}, {entity} {dp}."
        nli_pairs.append([sent, hypothesis])

    try:
        scores_batch = model.predict(nli_pairs)
        results = list(heuristic_scores)
        for j, i in enumerate(nli_indices):
            scores = scores_batch[j]
            exp_s = np.exp(scores - np.max(scores))
            probs = exp_s / exp_s.sum()
            contradiction_p = float(probs[0])
            entailment_p    = float(probs[1])
            if entailment_p > 0.50:
                nli_m = 1.0
            elif contradiction_p > 0.50:
                nli_m = 0.1
            elif contradiction_p > 0.30:
                nli_m = 0.3
            else:
                nli_m = 0.7
            results[i] = min(heuristic_scores[i], nli_m) if nli_m < 0.5 else heuristic_scores[i] * nli_m
        return results
    except Exception:
        return heuristic_scores
