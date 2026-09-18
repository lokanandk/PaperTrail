"""
semantic_relevance.py — Semantic relevance scoring for literature validation.

Builds a biomedical sentence from the prediction dict and scores each record
against it using SapBERT (or MiniLM as CPU-friendly fallback).

All prediction-query text is derived dynamically from the prediction dict.
No hardcoded disease or tissue phrases exist here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Dict
import os
import re
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

DEFAULT_MODELS = [
    #"ncbi/MedCPT-Article-Encoder",                         # for encoding abstracts
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",       # fallback
    "sentence-transformers/all-MiniLM-L6-v2",              # last resort
]

CACHE_DIR = Path(os.environ.get("LITREV_EMBED_CACHE",
                                 Path(__file__).parent / "cache" / "embeddings"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RECORD_EMB_CACHE: Dict[str, np.ndarray] = {}
_PRED_EMB_CACHE:   Dict[str, np.ndarray] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=2)
def _load_model(model_name: Optional[str] = None):
    if SentenceTransformer is None:
        print("[semantic_relevance] sentence-transformers not installed; disabled.")
        return None
    names = [model_name] if model_name else DEFAULT_MODELS
    for name in names:
        try:
            print(f"[semantic_relevance] Loading model: {name} on CPU...")
            return SentenceTransformer(name, device="cpu")
        except Exception as e:
            print(f"[semantic_relevance] Failed to load {name}: {e}")
    print("[semantic_relevance] No usable model; semantic scoring disabled.")
    return None


def _model_tag(model_name: Optional[str]) -> str:
    name = model_name or DEFAULT_MODELS[0]
    return name.split("/")[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Disk-backed embedding cache
# ─────────────────────────────────────────────────────────────────────────────
def _disk_cache_path(model_tag: str, pmid: str) -> Path:
    if not pmid:
        return CACHE_DIR / model_tag / "_empty.npz"
    fan = pmid[:2] if len(pmid) >= 2 else "_x"
    d   = CACHE_DIR / model_tag / fan
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{pmid}.npy"


def _load_record_emb_from_disk(model_tag: str, pmid: str) -> Optional[np.ndarray]:
    p = _disk_cache_path(model_tag, pmid)
    if p.exists():
        try: return np.load(p)
        except Exception: return None
    return None


def _save_record_emb_to_disk(model_tag: str, pmid: str, emb: np.ndarray) -> None:
    try: np.save(_disk_cache_path(model_tag, pmid), emb)
    except Exception: pass


def _encode_texts(texts: List[str], model_name: Optional[str] = None,
                  batch_size: int = 32) -> Optional[np.ndarray]:
    model = _load_model(model_name)
    if model is None:
        return None
    return model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction query construction — fully derived from prediction dict
# ─────────────────────────────────────────────────────────────────────────────

DIRECTION_PHRASES = {
    "up":                    "is upregulated, increased, or elevated",
    "down":                  "is downregulated, decreased, or reduced",
    "preserved":             "is preserved, maintained, or unchanged",
    "absent":                "is absent, undetectable, or lost",
    "bidirectional":         "shows bidirectional or variable expression",
    "binary_present_absent": "is present or absent depending on condition",
    "associated":            "is associated with the disease or condition",
}


def build_prediction_query(prediction: dict) -> str:
    """
    Build a biomedical sentence from a prediction dict.

    Uses: entity, aliases, direction, cell_type, disease_context, prediction_note.
    Nothing is hardcoded — the sentence reads like a PubMed abstract sentence
    that SapBERT will match well against.
    """
    entity    = prediction.get("entity", "")
    aliases   = prediction.get("aliases", []) or []
    direction = prediction.get("direction", "bidirectional")
    cell_type = (prediction.get("cell_type") or "").strip()
    disease   = (prediction.get("disease_context") or "").strip()

    # Alias parenthetical — take up to 3 aliases that differ from entity name
    extra = [a for a in aliases if a.lower() != entity.lower()][:3]
    alias_str = f" ({', '.join(extra)})" if extra else ""

    direction_phrase = DIRECTION_PHRASES.get(direction,
                       f"shows {direction} expression or activity")

    parts = [f"{entity}{alias_str} {direction_phrase}"]

    # Cell type phrase — use raw value, not a lookup table
    if cell_type and cell_type.lower() not in ("any", ""):
        parts.append(f"in {cell_type}")

    # Disease phrase — use raw value directly
    if disease and disease.lower() not in ("any", ""):
        parts.append(f"in {disease}")

    sentence = " ".join(parts).strip().rstrip(".") + "."

    # Append prediction_note / note for additional anchoring
    sv = (prediction.get("prediction_note") or prediction.get("note") or "")
    if sv:
        sentence += f" {str(sv)[:200]}"

    return sentence


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_record_text(record: dict) -> str:
    title    = record.get("title", "") or ""
    abstract = record.get("abstract", "") or ""
    mesh     = " ".join((record.get("mesh_terms", []) or [])[:8])
    return f"{title}. {abstract} {mesh}".strip()


def compute_semantic_scores_for_records(
    prediction_or_text,
    records: Iterable[dict],
    model_name: Optional[str] = None,
    batch_size: int = 32,
    use_disk_cache: bool = True,
) -> List[float]:
    """Cosine similarity in [0,1] between prediction query and each record."""
    records = list(records)
    if not records:
        return []

    if isinstance(prediction_or_text, dict):
        pred_text = build_prediction_query(prediction_or_text)
        pred_key  = prediction_or_text.get("id", pred_text[:50])
    else:
        pred_text = str(prediction_or_text)
        pred_key  = pred_text[:50]

    model_tag     = _model_tag(model_name)
    pred_cache_key = f"{model_tag}::{pred_key}"

    emb_pred = _PRED_EMB_CACHE.get(pred_cache_key)
    if emb_pred is None:
        embs = _encode_texts([pred_text], model_name=model_name, batch_size=batch_size)
        if embs is None:
            return [0.0] * len(records)
        emb_pred = embs[0]
        _PRED_EMB_CACHE[pred_cache_key] = emb_pred

    pmids    = [str(r.get("pmid", "")) for r in records]
    emb_recs: List[Optional[np.ndarray]] = [None] * len(records)
    to_encode_texts: List[str] = []
    to_encode_idx:   List[int] = []

    for i, (pmid, rec) in enumerate(zip(pmids, records)):
        cache_key = f"{model_tag}::{pmid}" if pmid else None
        if cache_key and cache_key in _RECORD_EMB_CACHE:
            emb_recs[i] = _RECORD_EMB_CACHE[cache_key]; continue
        if use_disk_cache and pmid:
            disk_emb = _load_record_emb_from_disk(model_tag, pmid)
            if disk_emb is not None:
                emb_recs[i] = disk_emb
                _RECORD_EMB_CACHE[cache_key] = disk_emb; continue
        text = get_record_text(rec)
        if not text or text == ".":
            emb_recs[i] = None; continue
        to_encode_texts.append(text)
        to_encode_idx.append(i)

    if to_encode_texts:
        new_embs = _encode_texts(to_encode_texts, model_name=model_name, batch_size=batch_size)
        if new_embs is None:
            for i in to_encode_idx: emb_recs[i] = None
        else:
            for j, i in enumerate(to_encode_idx):
                emb = new_embs[j]
                emb_recs[i] = emb
                pmid = pmids[i]
                if pmid:
                    ck = f"{model_tag}::{pmid}"
                    _RECORD_EMB_CACHE[ck] = emb
                    if use_disk_cache:
                        _save_record_emb_to_disk(model_tag, pmid, emb)

    sims: List[float] = []
    for emb in emb_recs:
        if emb is None: sims.append(0.0); continue
        s = float(np.dot(emb, emb_pred))
        sims.append((s + 1.0) / 2.0)
    return sims


def compute_semantic_diagnostic(prediction: dict, records: List[dict],
                                  model_name: Optional[str] = None) -> dict:
    query  = build_prediction_query(prediction)
    scores = compute_semantic_scores_for_records(prediction, records, model_name=model_name)
    return {"prediction_query": query, "scores": scores,
            "n_records": len(records), "model": model_name or DEFAULT_MODELS[0]}


def clear_caches():
    _RECORD_EMB_CACHE.clear()
    _PRED_EMB_CACHE.clear()
