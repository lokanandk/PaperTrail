#!/usr/bin/env python3
"""
PaperTrail — web front end for the literature concordance pipeline.

Two ways in:

  Demo mode  shows four worked examples from pre-computed results, so you can
             see what the output looks like without waiting for PubMed.
  Full mode  runs stages 1-5 for real against PubMed and streams the log back
             to the browser as it goes.

Predictions can be typed as YAML directly, or described in plain English and
converted — by an LLM if one is configured, by the regex parser otherwise.
See llm_gateway.py; no LLM is required for any part of the app to work.

Pipeline runs are serialised. The stage modules configure themselves through
module-level globals and PubMed enforces a per-IP rate limit, so running two
at once would corrupt both and earn HTTP 429s rather than finish sooner. Extra
submissions queue and report their position.

Usage:
  pip install -r requirements.txt
  python3 papertrail_app.py --port 5050
  → http://localhost:5050

Expects alongside it:
  papertrail_dashboard.py    dashboard HTML template and paper enrichment
  demo_data.json             the four worked examples
  papertrail_vocab.json      disease/cell-type/tissue vocabulary
"""
import os, sys, json, re, threading, queue, time, traceback
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, Response

import llm_gateway

app = Flask(__name__)
app.secret_key = os.urandom(24)

BASE      = Path(__file__).parent

# ── Shared PubMed cache ───────────────────────────────────────────────────────
# All pipeline runs share one cache directory so PubMed records are never
# downloaded twice, even across users. Set LITREV_CACHE env var to override.
# You are responsible for periodic cleanup (e.g. monthly cron job).
SHARED_CACHE = Path(os.environ.get("LITREV_CACHE", str(BASE / "shared_cache")))
SHARED_CACHE.mkdir(parents=True, exist_ok=True)
(SHARED_CACHE / "pubmed_records").mkdir(exist_ok=True)

# ── Demo data ────────────────────────────────────────────────────────────────
# The four worked examples shown on the landing page. Kept in demo_data.json
# rather than inline so the file stays readable and the demos can be
# regenerated from a real pipeline run without touching this module.
_demo_file = BASE / "demo_data.json"
try:
    DEMO_DATA = json.loads(_demo_file.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    # Losing the demos should not stop the app: the real pipeline still works.
    print(f"[demo] could not load {_demo_file.name}: {exc} — demos disabled")
    DEMO_DATA = {}
RUNS_DIR  = BASE / "runs"; RUNS_DIR.mkdir(exist_ok=True)

# ── Run bookkeeping ──────────────────────────────────────────────────────────
# Both dicts are reachable from request threads and pipeline threads, so every
# mutation goes through _RUNS_LOCK. They are also capped: an app left running
# for weeks would otherwise accumulate one entry per run forever.
_RUNS_LOCK = threading.Lock()
LOG_QUEUES: dict = {}       # run_id → queue of log lines, drained by the SSE route
RUN_COMPLETION: dict = {}   # run_id → final "DONE:…" or "ERROR:…" message
                            # persists so reconnecting browsers can see the result
MAX_TRACKED_RUNS = 200      # oldest entries are evicted beyond this

# Stages 1-5 configure the pipeline modules by assigning to their globals
# (stage1.CACHE_DIR, stage3.RECORDS_DIR, Entrez.email, …). Those are process
# wide, so two pipelines running at once would overwrite each other's settings
# mid-run. NCBI is the harder limit: stage3 paces itself at ~3 requests/second,
# which is the whole per-IP budget, so a second concurrent run does not go
# faster — it earns HTTP 429s and eventually a block.
#
# Runs are therefore serialised. Extra submissions queue up and report their
# position instead of failing. Raise PAPERTRAIL_MAX_CONCURRENT_RUNS only if you
# have an NCBI API key and have confirmed the higher rate limit applies.
try:
    MAX_CONCURRENT_RUNS = max(1, int(os.environ.get("PAPERTRAIL_MAX_CONCURRENT_RUNS", "1")))
except ValueError:
    MAX_CONCURRENT_RUNS = 1
_PIPELINE_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)
_QUEUE_DEPTH = 0            # runs submitted but not yet finished; guarded by _RUNS_LOCK

# A single YAML with hundreds of predictions is not an error, but it is a long
# wait, so warn rather than let the user assume the app has hung.
LARGE_BATCH_WARN = 40
SECONDS_PER_PREDICTION = 12   # rough, dominated by the NCBI pacing in stage 3


def _register_run(run_id):
    """Create the log queue for a run and evict old bookkeeping entries."""
    with _RUNS_LOCK:
        LOG_QUEUES[run_id] = queue.Queue()
        # dicts keep insertion order, so the front of the list is the oldest
        for stale in list(LOG_QUEUES)[:-MAX_TRACKED_RUNS]:
            LOG_QUEUES.pop(stale, None)
        for stale in list(RUN_COMPLETION)[:-MAX_TRACKED_RUNS]:
            RUN_COMPLETION.pop(stale, None)
        return LOG_QUEUES[run_id]


def _finish_run(run_id, message):
    """Record a run's final message so a reconnecting browser can still see it."""
    with _RUNS_LOCK:
        RUN_COMPLETION[run_id] = message

# ── Dashboard template ─────────────────────────────────────────────────────
_DASH_PY = BASE / "papertrail_dashboard.py"
if _DASH_PY.exists():
    import importlib.util as _ilu
    _s = _ilu.spec_from_file_location("_dash", _DASH_PY)
    _m = _ilu.module_from_spec(_s); _s.loader.exec_module(_m)
    DASH_TPL = _m.HTML
else:
    DASH_TPL = ("<html><body style='font-family:sans-serif;padding:2rem'>"
                "<h2>⚠ Dashboard template not found</h2>"
                "<p>Place <code>papertrail_dashboard.py</code> next to this app.</p>"
                "</body></html>")

def _inject(data: dict, meta: dict) -> str:
    """Inject DATA and META into the dashboard template."""
    html = DASH_TPL
    html = html.replace('__DATA__', json.dumps(data,  ensure_ascii=False))
    html = html.replace('__META__', json.dumps(meta,  ensure_ascii=False))
    return html


# ══════════════════════════════════════════════════════════════════
# DEMO CASE DEFINITIONS
# Shown as editable row-based tables. demo_key → demo_data.json key.
# The "type_label" appears as a badge explaining the search paradigm.
# ══════════════════════════════════════════════════════════════════

CASES = [
    {
        "id": "lung_adenocarcinoma", "demo_key": "lung_adenocarcinoma",
        "label": "Lung Adenocarcinoma", "subtitle": "scRNA-seq · LUAD tumor microenvironment",
        "emoji": "🫁", "accent": "#0F766E", "accent_l": "#CCFBF1",
        "type_label": "Gene expression (scRNA-seq)",
        "type_desc": "Typical use: checking whether genes identified as up/down in a single-cell atlas agree with independent published literature.",
        "blurb": "Key oncogenes, tumor suppressors, and immune checkpoint markers from a spatial scRNA-seq atlas of LUAD. Each row tests whether the expected expression direction is concordant with published studies.",
        "rows": [
            {"entity":"KRAS",   "aliases":"KRAS, Kirsten ras, KRAS oncogene",           "disease":"Lung adenocarcinoma","cell_type":"Tumor cells",   "direction":"up",   "note":"Driver mutation ~30% LUAD; MAPK activation"},
            {"entity":"TP53",   "aliases":"TP53, p53, tumor protein p53",                "disease":"Lung adenocarcinoma","cell_type":"Tumor cells",   "direction":"down", "note":"Tumor suppressor; inactivating mutations"},
            {"entity":"MYC",    "aliases":"MYC, c-Myc, MYC oncogene",                   "disease":"Lung adenocarcinoma","cell_type":"Tumor cells",   "direction":"up",   "note":"Amplified ~15% LUAD; transcription factor"},
            {"entity":"EGFR",   "aliases":"EGFR, ErbB1, epidermal growth factor receptor","disease":"Lung adenocarcinoma","cell_type":"Tumor cells",  "direction":"up",   "note":"Activating mutations drive TKI sensitivity"},
            {"entity":"FOXP3",  "aliases":"FOXP3, regulatory T cell marker, scurfin",    "disease":"Lung adenocarcinoma","cell_type":"Treg cells",    "direction":"up",   "note":"Tregs enriched in tumor → immunosuppression"},
            {"entity":"SLC2A1", "aliases":"SLC2A1, GLUT1, glucose transporter 1",        "disease":"Lung adenocarcinoma","cell_type":"Tumor cells",   "direction":"up",   "note":"Warburg effect — aerobic glycolysis"},
        ],
    },
    {
        "id": "sle", "demo_key": "sle",
        "label": "Systemic Lupus Erythematosus", "subtitle": "Innate immune · type I IFN axis",
        "emoji": "🔬", "accent": "#1E3A8A", "accent_l": "#DBEAFE",
        "type_label": "Pathway & immune activation",
        "type_desc": "Typical use: testing whether predicted pathway activation states (e.g. IFN signature, complement loss) are supported by published immunology literature.",
        "blurb": "Type I interferon pathway and B cell activation predictions in SLE. Tests concordance of both gene-level (TREX1↓, IRF5↑) and pathway-level (IFN signature↑) predictions against published immunology literature.",
        "rows": [
            {"entity":"IRF5",         "aliases":"IRF5, interferon regulatory factor 5",          "disease":"SLE","cell_type":"Plasmacytoid DCs","direction":"up",  "note":"IFN-α/β driver; gain-of-function variants"},
            {"entity":"TREX1",        "aliases":"TREX1, DNase III, three prime repair exonuclease","disease":"SLE","cell_type":"pDC / immune",    "direction":"down","note":"DNA exonuclease; loss activates cGAS-STING"},
            {"entity":"IFN signature","aliases":"IFNA1, IFNB1, type I interferon, ISG15",         "disease":"SLE","cell_type":"Monocytes",        "direction":"up",  "note":"IFN score elevated >60% SLE — pathway, not single gene"},
            {"entity":"C1Q",          "aliases":"C1Q, complement C1q, complement component 1q",  "disease":"SLE","cell_type":"Serum / immune",   "direction":"down","note":"Complement deficiency → impaired apoptotic clearance"},
            {"entity":"BLK",          "aliases":"BLK, B lymphocyte kinase, B cell kinase",       "disease":"SLE","cell_type":"B cells",          "direction":"up",  "note":"B cell hyperactivation; SLE susceptibility variants"},
        ],
    },
    {
        "id": "rheumatoid_arthritis", "demo_key": "rheumatoid_arthritis",
        "label": "Rheumatoid Arthritis", "subtitle": "Drug-target mechanism · IL-6/JAK-STAT axis",
        "emoji": "🧪", "accent": "#7C3AED", "accent_l": "#EDE9FE",
        "type_label": "Drug-target & therapeutic mechanism",
        "type_desc": "Typical use: checking whether the molecular target of a drug is upregulated/active in the disease of interest, and whether downstream signalling is concordant with the therapeutic rationale.",
        "blurb": "IL-6/JAK-STAT pathway in RA — a different kind of literature search. Rather than asking 'is gene X up or down?', this tests whether the mechanistic rationale for tocilizumab (IL-6R blockade) and baricitinib (JAK1 inhibition) is supported by independent studies. Includes cytokines, kinases, transcription factors, and proteases.",
        "rows": [
            {"entity":"IL6",   "aliases":"IL6, interleukin-6, IL-6",                          "disease":"Rheumatoid arthritis","cell_type":"Synovial fibroblast","direction":"up","note":"Tocilizumab target; drives acute-phase response"},
            {"entity":"TNF",   "aliases":"TNF, TNF-alpha, tumor necrosis factor, TNFA",        "disease":"Rheumatoid arthritis","cell_type":"Macrophage",        "direction":"up","note":"Anti-TNF therapy (adalimumab, etanercept) target"},
            {"entity":"JAK1",  "aliases":"JAK1, janus kinase 1, JAK-1",                       "disease":"Rheumatoid arthritis","cell_type":"Immune cells",       "direction":"up","note":"Baricitinib/upadacitinib target; downstream of IL-6R"},
            {"entity":"STAT3", "aliases":"STAT3, pSTAT3, signal transducer activator transcription 3","disease":"Rheumatoid arthritis","cell_type":"Immune cells","direction":"up","note":"Downstream of JAK1; drives pro-inflammatory programme"},
            {"entity":"MMP3",  "aliases":"MMP3, stromelysin-1, matrix metalloproteinase 3",   "disease":"Rheumatoid arthritis","cell_type":"Synovial fibroblast","direction":"up","note":"Cartilage ECM degradation; serum biomarker of joint damage"},
        ],
    },
    {
        "id": "ibd_gwas_eqtl", "demo_key": "ibd_gwas_eqtl",
        "label": "IBD GWAS \u2014 eQTL & Regulatory Variants",
        "subtitle": "Post-GWAS functional concordance \u00b7 fine-mapped variants",
        "emoji": "\U0001f9ec", "accent": "#6D28D9", "accent_l": "#EDE9FE",
        "type_label": "GWAS / eQTL / enhancer evidence",
        "type_desc": "Typical use: after GWAS or fine-mapping, checking whether top hits have cell-type-specific eQTL, enhancer, or chromatin evidence in published functional genomics studies. Direction can be up/down or 'associated' \u2014 no strict directionality required.",
        "blurb": "Post-GWAS functional concordance for IBD fine-mapped variants. Each row is a GWAS locus with a predicted cell-type-specific mechanism: eQTL effect, enhancer variant, or cis-regulatory logic. Tests whether published scATAC-seq, scRNA-seq, eQTL, and CRISPRa studies support the predicted functional impact. Unlike gene-expression cases, direction here is often 'associated' \u2014 the claim is about variant-function, not a fold-change.",
        "rows": [
            {"entity":"IL23R",  "aliases":"IL23R, interleukin-23 receptor, IL-23R",                    "disease":"IBD / Crohn's disease",   "cell_type":"Th17 / CD4+ T cells",               "direction":"up",         "note":"rs11209026 eQTL: increases IL23R in Th17; open chromatin scATAC; risankizumab target"},
            {"entity":"NOD2",   "aliases":"NOD2, CARD15, nucleotide-binding oligomerization domain 2", "disease":"IBD / Crohn's disease",   "cell_type":"Paneth cells / intestinal epithelium","direction":"associated", "note":"R702W/G908R/L1007fs loss-of-function; cell-type-specific eQTL in colonoids not blood"},
            {"entity":"PTPN22", "aliases":"PTPN22, LYP, protein tyrosine phosphatase non-receptor 22","disease":"IBD / autoimmune",        "cell_type":"T cells / Treg",                    "direction":"associated", "note":"R620W (rs2476601): shared autoimmune variant; colocalises with T cell eQTL"},
            {"entity":"SMAD3",  "aliases":"SMAD3, MAD homolog 3, JV15-2",                            "disease":"IBD / Crohn's fibrosis",  "cell_type":"Intestinal fibroblasts",             "direction":"up",         "note":"Enhancer variant 14kb upstream: fibroblast-specific H3K27ac; CRISPRa confirmed"},
            {"entity":"EOMES",  "aliases":"EOMES, eomesodermin, TBR2",                               "disease":"IBD",                     "cell_type":"CD8+ T cells / IEL",                "direction":"associated", "note":"rs4900384: CD8+ T cell-specific ATAC peak; eQTL in intraepithelial lymphocytes"},
        ],
    },
]


# ══════════════════════════════════════════════════════════════════
# BUILD DASHBOARD DATA
# ══════════════════════════════════════════════════════════════════

def _flip_relation(rel):
    if rel=="concordant": return "opposite"
    if rel=="opposite": return "concordant"
    return rel

def _reclassify_tier(nc, ni, qw):
    if ni==0: return "NO_DIRECTIONAL"
    if qw>=0.80 and ni>=5: return "STRONG"
    if qw>=0.65 and ni>=3: return "MODERATE"
    if qw>=0.50 and ni>=2: return "WEAK_SUPPORT"
    if qw<0.35: return "WEAK_DISCORDANT"
    return "MIXED"

_ABBREV_RE2 = re.compile(
    r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|Fig|et al|e\.g|i\.e|approx|'
    r'Eq|No|vol|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.')

def _split_excerpt(text: str) -> list:
    """Split an abstract excerpt into individual sentences for evidence display."""
    if not text: return []
    text = text.strip()
    # Mask common abbreviations to avoid false splits
    masked = _ABBREV_RE2.sub(lambda m: m.group().replace('.', '\x00'), text)
    parts  = re.split(r'(?<=[.!?])\s+(?=[A-Z\(\[])', masked)
    out = []
    for p in parts:
        p = p.replace('\x00', '.').strip()
        if len(p) > 20:
            out.append(p)
    return out or [text]

_STOP_WORDS = {
    'the','and','for','with','from','that','this','are','was','were','has',
    'have','been','its','our','their','than','into','via','against','using',
    'which','while','these','those','show','shows','showed','study','studies',
    'analysis','results','findings','suggest','suggests','demonstrate'
}

def _title_keyphrases(title: str, n: int = 5) -> list:
    """Extract key 2-3 word phrases from a paper title."""
    if not title: return []
    words = re.sub(r'[^\w\s\-]', ' ', title).strip().lower().split()
    words = [w for w in words if len(w) > 2 and w not in _STOP_WORDS]
    phrases = []
    # Prefer 2-word phrases
    for size in [3, 2, 1]:
        for i in range(len(words) - size + 1):
            phrase = ' '.join(words[i:i+size])
            if len(phrase) < 4: continue
            if any(phrase in ex or ex in phrase for ex in phrases): continue
            phrases.append(phrase)
            if len(phrases) >= n: return phrases
    return phrases[:n]

def build_demo_data(demo_key: str, edited_directions: dict = None) -> dict:
    demo   = DEMO_DATA[demo_key]
    scored = demo["predictions"]
    cats   = demo["categories"]
    tc = {}
    for s in scored.values(): tc[s["tier"]] = tc.get(s["tier"], 0) + 1
    summary = {
        "total":len(scored),"strong":tc.get("STRONG",0),"moderate":tc.get("MODERATE",0),
        "mixed":tc.get("MIXED",0),"weak_support":tc.get("WEAK_SUPPORT",0),
        "weak_discordant":tc.get("WEAK_DISCORDANT",0),"descriptive":tc.get("DESCRIPTIVE",0),
        "no_directional":tc.get("NO_DIRECTIONAL",0),"none":tc.get("NONE",0),"loo_fragile":0,
    }
    per_pred = []
    for pid, s in scored.items():
        p, loo = s["prediction"], s.get("leave_one_out") or {}
        # Apply user-edited direction if any
        entity_key  = p.get("entity","").strip().lower()
        orig_dir    = s.get("expected_direction", p.get("direction",""))
        user_dir    = (edited_directions or {}).get(entity_key, orig_dir)
        dir_changed = (user_dir != orig_dir) and (orig_dir not in ("associated","bidirectional"))
        recs = []
        for r in s.get("summary_evidence",[]):
            rel = r.get("relation","")
            if dir_changed and rel in ("concordant","opposite"):
                rel = _flip_relation(rel)
            sym = "✓" if rel=="concordant" else ("✗" if rel=="opposite" else "—")
            ex  = r.get("best_excerpt","") or r.get("abstract_excerpt","")
            title = r.get("title","")
            # Split excerpt into individual sentences for the paper card
            ev_sents = _split_excerpt(ex) if ex else (["See title."] if title else [])
            # Build key-phrase tags from title words
            kw = _title_keyphrases(title)
            # Relevance reason shown as a human-readable line
            if rel == "concordant":
                rel_reason = (f"Concordant: reports {r.get('extracted_direction','?')} direction, "
                              f"matching expected {s.get('expected_direction', p.get('direction',''))}.")
            elif rel == "opposite":
                rel_reason = (f"Discordant: reports {r.get('extracted_direction','?')} direction, "
                              f"opposing expected {s.get('expected_direction', p.get('direction',''))}.")
            else:
                rel_reason = "Neutral/descriptive."
            recs.append({
                "pmid":r.get("pmid",""),"year":r.get("year",0),"journal":r.get("journal",""),
                "direction":r.get("extracted_direction","?"),"concordant":sym,
                "quality":round(r.get("quality_weight",0),2),"title":title,
                "key_phrases":kw,"evidence_sentences":ev_sents,
                "mechanism":ev_sents[0] if ev_sents else "",
                "relevance_reason":rel_reason,
                "text_source":r.get("excerpt_source","abstract"),
            })
        # Recompute concordance metrics if direction was flipped
        if dir_changed:
            n_c = sum(1 for r in recs if r["concordant"] == "✓")
            n_o = sum(1 for r in recs if r["concordant"] == "✗")
            n_i = n_c + n_o
            qw  = round(n_c / n_i, 3) if n_i > 0 else 0.0
            new_tier = _reclassify_tier(n_c, n_i, qw)
        else:
            n_c  = s.get("n_concordant", 0)
            n_o  = s.get("n_opposite",   0)
            n_i  = s.get("n_informative", 0)
            qw   = s.get("concordance_quality_weighted")
            new_tier = s["tier"]
        per_pred.append({
            "id":pid,"label":f"{p['entity']} ({p['disease_context']}, {p['cell_type']})",
            "novelty":p.get("novelty"),"direction":user_dir,
            "tier":new_tier,"reason":s.get("tier_reason",""),"prediction_note":p.get("prediction_note",""),
            "loo_low":loo.get("min"),"loo_high":loo.get("max"),"loo_delta":loo.get("range"),
            "qw_conc":qw,"n_informative":n_i,
            "n_concordant":n_c,"n_opposite":n_o,
            "n_relevant":s.get("n_relevant",0),
            "scoring_mode": s.get("scoring_mode") or (
                "association" if (p.get("direction","") in ("bidirectional","associated","absent"))
                else "directional"),
            "loo_papers": (s.get("leave_one_out") or {}).get("loo_papers", []),
            "p_val":s.get("binomial_p"),"simple_conc":qw,
            "n_opposite":n_o,"records":recs,
        })
    return {
        "summary":summary,"categories":cats,
        "strong_preds":[{"id":p["id"],"label":p["label"],"qw_conc":p["qw_conc"] or 0,
            "n_inform":p["n_informative"] or 0,"p":p["p_val"] or 1}
            for p in per_pred if p["tier"]=="STRONG"],
        "exp_preds":[{"id":p["id"],"label":p["label"],"tier":p["tier"],
            "n_relevant":p["n_informative"] or 0}
            for p in per_pred if p["tier"] in ("NONE","NO_DIRECTIONAL","WEAK_SUPPORT")],
        "per_pred":per_pred,"demo_mode":True,"demo_title":demo["title"],
    }


# ══════════════════════════════════════════════════════════════════
# BIOLOGICAL VOCABULARY — loaded from papertrail_vocab.json
# ══════════════════════════════════════════════════════════════════
# All disease/cell-type/tissue knowledge lives in papertrail_vocab.json,
# not in this file.  Generate or extend that file with build_papertrail_vocab.py.
# If the file is absent the app still runs — canonicalisation is simply skipped.
import json as _json
_VOCAB_PATH = BASE / "papertrail_vocab.json"
try:
    _vocab_raw = _json.loads(_VOCAB_PATH.read_text("utf-8"))
    def _drop_comments(d):
        return {k: v for k, v in d.items() if k != "_comment"} if isinstance(d, dict) else d
    _DISEASE_CONTEXT_MAP    = _drop_comments(_vocab_raw.get("disease_context_map", {}))
    _DISEASE_TABLE          = {k: tuple(v) for k, v in
                                _drop_comments(_vocab_raw.get("disease_table", {})).items()}
    _KIDNEY_EXCLUSIONS_LIST = _vocab_raw.get("disease_exclusions", {}).get("kidney", [])
    _CELL_TYPE_TABLE        = {k: tuple(v) for k, v in
                                _drop_comments(_vocab_raw.get("cell_type_table", {})).items()}
    _TISSUE_KEYWORDS        = {k: tuple(v) for k, v in
                                _drop_comments(_vocab_raw.get("tissue_keywords", {})).items()}
except Exception as _ve:
    import warnings as _w
    _w.warn(f"[PaperTrail] papertrail_vocab.json not loaded ({_ve}). "
            f"Run build_papertrail_vocab.py to generate it. Expected: {_VOCAB_PATH}")
    _DISEASE_CONTEXT_MAP = {}; _DISEASE_TABLE = {}; _KIDNEY_EXCLUSIONS_LIST = []
    _CELL_TYPE_TABLE = {};     _TISSUE_KEYWORDS = {}


# ══════════════════════════════════════════════════════════════════
# ROWS → YAML
# ══════════════════════════════════════════════════════════════════

def _llm_normalise_disease(raw: str) -> str:
    """
    Ask the LLM for the standard abbreviation of a disease name.

    Returns '' on any failure — no gateway, a timeout, a surprising answer —
    so the caller falls through to the static lookup and the app keeps working.
    """
    answer, _err = llm_gateway.chat(
        [{'role': 'user', 'content':
          f'What is the standard medical abbreviation for this disease: "{raw}"?\n'
          f'Reply with ONLY the abbreviation (2–6 characters, e.g. DKD, SLE, NSCLC).\n'
          f'If no standard abbreviation exists reply with the original name unchanged.'}],
        max_tokens=20, temperature=0, timeout=10,
    )
    if not answer:
        return ''
    abbrev = answer.strip().strip('"').strip("'")
    return abbrev if len(abbrev) <= 20 else ''


def _normalise_disease(raw: str) -> str:
    """
    Map a free-text disease label to the stage2/stage4 disease_context key.

    Priority:
      1. Static lookup in _DISEASE_CONTEXT_MAP (papertrail_vocab.json) — instant.
      2. LLM normalisation — accurate for novel/rare diseases not in the JSON.
      3. Return as-is — the pipeline gate will still match on full text.
    """
    if not raw or not raw.strip():
        return raw
    key = raw.strip().lower()

    # 1. Exact static match
    if key in _DISEASE_CONTEXT_MAP:
        return _DISEASE_CONTEXT_MAP[key]

    # 2. Partial static match (substring in either direction)
    for k, v in _DISEASE_CONTEXT_MAP.items():
        if k in key or key in k:
            return v

    # 3. LLM normalisation (fast: max_tokens=20, dedicated short call)
    abbrev = _llm_normalise_disease(raw)
    if abbrev:
        # Cache so subsequent rows with the same label skip the LLM call
        _DISEASE_CONTEXT_MAP[key] = abbrev
        return abbrev

    # 4. Return original if all else fails
    return raw.strip()


def rows_to_yaml(rows: list, label: str) -> str:
    lines = [f"# {label}", "predictions:"]
    for i, r in enumerate(rows):
        entity  = (r.get("entity") or "").strip()
        aliases = (r.get("aliases") or "").strip()
        disease_raw = (r.get("disease") or "").strip()
        disease = _normalise_disease(disease_raw)
        ct      = (r.get("cell_type") or "").strip()
        dirn    = (r.get("direction") or "up").strip().lower()
        note    = (r.get("note") or "").strip()
        pid     = re.sub(r'[^a-zA-Z0-9]','_',f"{entity}_{dirn}").strip('_')+f"_{i+1}"
        al = ", ".join(f'"{a.strip()}"' for a in aliases.split(',') if a.strip())
        lines += ["",f"  - id: {pid}",f"    entity: {entity}",f"    entity_type: gene",
            f"    aliases: [{al}]",f"    disease_context: {disease}",
            f"    cell_type: {ct}",f"    direction: {dirn}",f"    novelty: extending",
            f'    note: "{note or entity+" "+dirn}"']
    return '\n'.join(lines)


# ══════════════════════════════════════════════════════════════════
# FULL PIPELINE RUNNER
# ══════════════════════════════════════════════════════════════════

STAGE_NAMES = [
    "Expanding gene aliases (mygene.info)",
    "Building PubMed queries",
    "Retrieving PubMed records",
    "Extracting directional evidence",
    "Scoring concordance",
]

def _count_predictions(pred_yaml):
    """Number of predictions in a YAML batch, or 0 if it will not parse."""
    try:
        import yaml as _yaml
        parsed = _yaml.safe_load(pred_yaml)
        return len(parsed.get("predictions") or []) if isinstance(parsed, dict) else 0
    except Exception:
        return 0


def run_pipeline_async(run_id, run_dir, pipeline_dir,
                       pred_yaml, skip_pmc, ncbi_email, ncbi_key, project_name):
    global _QUEUE_DEPTH
    q = _register_run(run_id)
    def emit(m): q.put(m); print(f"[{run_id}] {m}")

    n_preds = _count_predictions(pred_yaml)
    if n_preds >= LARGE_BATCH_WARN:
        mins = max(1, round(n_preds * SECONDS_PER_PREDICTION / 60))
        emit(f"  ⓘ Large batch: {n_preds} predictions, roughly {mins} min. "
             f"PubMed retrieval is paced to stay inside NCBI's rate limit.")

    # Wait for a pipeline slot rather than trampling a run already in progress.
    if not _PIPELINE_SLOTS.acquire(blocking=False):
        with _RUNS_LOCK:
            ahead = max(0, _QUEUE_DEPTH - 1)
        emit(f"  ⓘ Queued behind {ahead} run(s) — starting automatically when they finish.")
        _PIPELINE_SLOTS.acquire()

    try:
        import yaml as _yaml
        if str(pipeline_dir) not in sys.path:
            sys.path.insert(0, str(pipeline_dir))
        os.environ["NCBI_EMAIL"] = ncbi_email
        if ncbi_key: os.environ["NCBI_API_KEY"] = ncbi_key
        pred_path = run_dir/"predictions.yaml"
        expanded  = run_dir/"predictions_expanded.yaml"
        q_yaml    = run_dir/"queries.yaml"
        lit_raw   = run_dir/"literature_raw.json"
        extracted = run_dir/"extracted_evidence.json"
        scored    = run_dir/"scored_predictions.json"
        dash_out  = run_dir/"dashboard.html"
        # Use the shared cache so records fetched by one user
        # are available to all subsequent runs — no redundant downloads.
        cache_dir = SHARED_CACHE / "pubmed_records"
        cache_dir.mkdir(parents=True, exist_ok=True)
        pred_path.write_text(pred_yaml)

        emit("STAGE:1")
        import stage1_expand_aliases as s1
        s1.CACHE_DIR = run_dir/"cache"; (run_dir/"cache").mkdir(exist_ok=True)
        s1.expand_predictions(pred_path, expanded); emit("  ✓ aliases expanded")

        emit("STAGE:2")
        # Previously reloaded on every run; that rebinds the module for any
        # other thread mid-call, and it buys nothing now that runs are serialised.
        import stage2_build_queries as s2
        s2.build_all(expanded, q_yaml)
        with open(q_yaml) as f: qd = _yaml.safe_load(f)
        emit(f"  ✓ {len(qd['queries'])} queries built")

        emit("STAGE:3")
        import stage3_retrieve_pubmed as s3; s3.RECORDS_DIR = cache_dir
        from Bio import Entrez; Entrez.email = ncbi_email
        if ncbi_key: Entrez.api_key = ncbi_key
        s3.retrieve_for_all_queries(q_yaml, lit_raw,
            predictions_path=expanded, skip_pmc=skip_pmc)
        with open(lit_raw) as f: lit = json.load(f)
        n = len({p for v in lit.values() for p in v.get("all_pmids",[])})
        emit(f"  ✓ {n} unique PMIDs retrieved")

        emit("STAGE:4")
        import stage4_extract_evidence as s4
        s4.extract_all(lit_raw, expanded, extracted, semantic_model=None)
        emit("  ✓ evidence extracted")

        emit("STAGE:5")
        import stage5_score_concordance as s5
        s5.score_all(extracted, scored); emit("  ✓ concordance scored")

        emit("STAGE:6")
        import papertrail_dashboard as dash
        papers   = dash.load_from_pipeline(scored, cache_dir)
        enriched = dash.enrich_papers(papers)
        try:
            import stage6_visualise_report as s6; s6.generate(scored, run_dir)
            data = dash.parse_report((run_dir/"report.md").read_text())
        except Exception:
            with open(scored) as f: sd = json.load(f)
            data = _min_data(sd)
        data = dash.merge_evidence(data, enriched)
        meta = {"project_name": project_name, "is_demo": False}
        dash_out.write_text(_inject(data, meta), encoding="utf-8")
        emit("  ✓ dashboard written")
        emit(f"DONE:{run_id}")
        _finish_run(run_id, f"DONE:{run_id}")
    except Exception as e:
        emit(f"ERROR: {e}\n{traceback.format_exc()}")
        _finish_run(run_id, f"ERROR: {e}")
    finally:
        # Record the outcome before releasing the slot, so the next run in the
        # queue never observes this one as still in flight.
        q.put(None)
        _PIPELINE_SLOTS.release()
        with _RUNS_LOCK:
            _QUEUE_DEPTH = max(0, _QUEUE_DEPTH - 1)

def _min_data(sd):
    sc = sd.get("predictions",sd); cats = sd.get("category_pooled",{})
    tc = {}
    for s in sc.values(): tc[s["tier"]] = tc.get(s["tier"],0)+1
    sm = {k:tc.get(k.upper().replace(" ","_"),0) for k in [
        "total","strong","moderate","mixed","weak_support","weak_discordant",
        "descriptive","no_directional","none","loo_fragile"]}
    sm["total"]=len(sc)
    pp = []
    for pid,s in sc.items():
        p=s.get("prediction",{}); lo=s.get("leave_one_out") or {}
        pp.append({"id":pid,"label":f"{p.get('entity','?')} ({p.get('disease_context','?')}, {p.get('cell_type','?')})",
            "novelty":p.get("novelty"),"direction":p.get("direction"),"tier":s.get("tier","NONE"),
            "reason":s.get("tier_reason",""),"prediction_note":p.get("prediction_note",""),
            "loo_low":lo.get("min"),"loo_high":lo.get("max"),"loo_delta":lo.get("range"),
            "qw_conc":s.get("concordance_quality_weighted"),"n_informative":s.get("n_informative"),
            "n_concordant":s.get("n_concordant",0),"n_opposite":s.get("n_opposite",0),
            "n_relevant":s.get("n_relevant",0),
            "p_val":s.get("binomial_p"),"simple_conc":s.get("concordance_simple"),
            "scoring_mode": s.get("scoring_mode") or (
                "association" if p.get("direction","") in ("bidirectional","associated","absent")
                else "directional"),
            "loo_papers": (s.get("leave_one_out") or {}).get("loo_papers", []),
            "records":[]})
    return {"summary":sm,"categories":[{"category":k,"n_pred":v.get("n_predictions",0),
        "n_inform":v.get("n_informative_total",0),"n_concord":v.get("n_concordant_total",0),
        "pooled":round((v.get("pooled_concordance") or 0)*100),"ci":"—","p":v.get("binomial_p") or 1}
        for k,v in cats.items() if v.get("pooled_concordance")],
        "strong_preds":[{"id":p["id"],"label":p["label"],"qw_conc":p["qw_conc"] or 0,
            "n_inform":p["n_informative"] or 0,"p":p["p_val"] or 1} for p in pp if p["tier"]=="STRONG"],
        "exp_preds":[],"per_pred":pp}


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(UI, cases=CASES)

@app.route("/api/demo", methods=["POST"])
def demo_run():
    payload = request.get_json(force=True) or {}
    dk      = payload.get("demo_key","")
    if dk not in DEMO_DATA: return jsonify({"error":"unknown demo key"}), 404
    edited  = {r["entity"].strip().lower(): r["direction"]
               for r in payload.get("edited_rows",[]) if r.get("entity")}
    data = build_demo_data(dk, edited_directions=edited)
    meta = {"project_name": "", "is_demo": True}
    return _inject(data, meta)

@app.route("/api/demo_json", methods=["POST"])
def demo_json():
    """Return prediction data as JSON for the visualization panel (demo mode)."""
    payload = request.get_json(force=True) or {}
    dk      = payload.get("demo_key","")
    if dk not in DEMO_DATA: return jsonify({"error":"unknown demo key"}), 404
    edited  = {r["entity"].strip().lower(): r["direction"]
               for r in payload.get("edited_rows",[]) if r.get("entity")}
    data = build_demo_data(dk, edited_directions=edited)
    return jsonify(data)


@app.route("/api/summary/<run_id>")
def api_summary(run_id):
    """Return prediction data as JSON for the visualization panel (pipeline runs)."""
    scored_path = RUNS_DIR / run_id / "scored_predictions.json"
    if not scored_path.exists():
        return jsonify({"error": "Run not ready or not found"}), 404
    try:
        import json as _j2
        sd = _j2.loads(scored_path.read_text())
        return jsonify(_min_data(sd))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/yaml", methods=["POST"])
def get_yaml():
    p = request.get_json(force=True) or {}
    return jsonify({"yaml": rows_to_yaml(p.get("rows",[]), p.get("label","Predictions"))})

@app.route("/api/run", methods=["POST"])
def start_run():
    global _QUEUE_DEPTH
    import uuid
    p = request.get_json(force=True) or {}
    yt = p.get("predictions_yaml","")
    if not yt: return jsonify({"error":"predictions_yaml required"}), 400
    rid = uuid.uuid4().hex[:10]
    rd  = RUNS_DIR/rid; rd.mkdir(parents=True, exist_ok=True)

    with _RUNS_LOCK:
        _QUEUE_DEPTH += 1
        queued_ahead = max(0, _QUEUE_DEPTH - MAX_CONCURRENT_RUNS)

    t = threading.Thread(target=run_pipeline_async, args=(
        rid, rd, Path(p.get("pipeline_dir","") or str(BASE)),
        yt, p.get("skip_pmc",True),
        p.get("ncbi_email","litrev@example.com"),
        p.get("ncbi_api_key",""),
        p.get("project_name","My Project"),
    ), daemon=True)
    t.start()

    n_preds = _count_predictions(yt)
    return jsonify({
        "run_id": rid,
        "n_predictions": n_preds,
        "queued_ahead": queued_ahead,
        "estimated_seconds": n_preds * SECONDS_PER_PREDICTION,
    })

@app.route("/api/logs/<run_id>")
def stream_logs(run_id):
    def gen():
        # ── Fast path: run already finished before client connected ───────────
        # This handles the reconnect-after-disconnect case: if Stage 4 was slow,
        # the SSE connection may have dropped and the client reconnected after
        # DONE: was emitted.  Send the stored result immediately.
        if run_id in RUN_COMPLETION:
            yield f"data: {RUN_COMPLETION[run_id]}\n\n"
            yield "data: [DONE]\n\n"
            return

        q = None
        for _ in range(60):
            if run_id in LOG_QUEUES: q = LOG_QUEUES[run_id]; break
            time.sleep(0.1)
        if not q:
            yield "data: Run not found\n\n"; return

        while True:
            try:
                # Timeout of 15s so keepalives fire before any proxy/browser
                # idle-timeout (nginx default = 60s, browsers ≈ 30s).
                m = q.get(timeout=15)
                if m is None:
                    yield "data: [DONE]\n\n"; break
                yield f"data: {m.replace(chr(10),'↵')}\n\n"
                if m.startswith("DONE:") or m.startswith("ERROR:"):
                    _finish_run(run_id, m)   # store for late/reconnecting clients
                    time.sleep(0.3)
                    yield "data: [DONE]\n\n"; break
            except queue.Empty:
                # Send an SSE comment — keeps TCP alive without triggering onmessage
                yield ": keepalive\n\n"
            except Exception:
                break
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/run_status/<run_id>")
def run_status(run_id):
    """Lightweight polling endpoint for clients that lost the SSE connection."""
    with _RUNS_LOCK:
        msg = RUN_COMPLETION.get(run_id)
        known = run_id in LOG_QUEUES
        depth = _QUEUE_DEPTH
    if msg:
        return jsonify({
            "status": "done" if msg.startswith("DONE:") else "error",
            "message": msg,
        })
    if known:
        return jsonify({"status": "running", "runs_in_flight": depth})
    return jsonify({"status": "unknown"})

@app.route("/api/dashboard/<run_id>")
def get_dashboard(run_id):
    f = RUNS_DIR/run_id/"dashboard.html"
    return f.read_text("utf-8") if f.exists() else ("Not ready", 404)


# ── Prompt → predictions.yaml conversion ─────────────────────────────────────
# A free-text description of a finding is turned into predictions.yaml by an
# LLM where one is available, and by the regex parser below where one is not.
# Order of preference:
#
#   1. Any OpenAI-compatible gateway found in the environment (see llm_gateway)
#   2. A local GGUF model via llama-cpp-python
#   3. The rule-based parser, which needs nothing and always works
#
# Optional overrides:
#   PAPERTRAIL_LLM_BACKEND = rules | llamacpp   (skip gateway detection)
#   PAPERTRAIL_GGUF_PATH   = /path/to/model.gguf
#
# Run `python llm_gateway.py` to see which gateway would be picked and why.

_CONVERSION_SYSTEM_PROMPT = """Convert the scientific description into predictions.yaml.

OUTPUT: Valid YAML only, starting with "predictions:". No markdown fences. No explanation.

SCHEMA (one block per entity):
predictions:
  - id:                ENTITY_DISEASE_CELLTYPE_direction
    entity:            gene symbol, protein, metabolite, or other entity name
    entity_type:       gene | protein | metabolite | lipid | ncRNA | small_molecule | drug | other
    aliases:           [SYMBOL, "full official name", "common synonym"]
    disease_context:   DKD | HKD | CKD | AKI | SLE | RA | IBD | any
    disease_synonyms:  ["Full disease name 1", "Full disease name 2"]
    disease_exclusions: ["off-topic context to exclude"]
    cell_type:         C_TAL | iPT | pDC | Podo | tubular | any
    cell_type_synonyms: ["full cell type name"]
    tissue:            kidney | immune | liver | lung | brain | any
    organism:          human | mouse | any
    direction:         up | down | preserved | absent | associated
    note:              "brief rationale"
    confidence:        high | medium | low
    novelty:           novel | extending | known

RULES:
1. One prediction block per entity — never skip or merge entities
2. direction: up=elevated/increased/overexpressed, down=reduced/decreased/downregulated
3. Disease synonyms: DKD=["diabetic nephropathy","diabetic kidney disease"], HKD=["hypertensive nephrosclerosis","hypertensive kidney disease"], CKD=["chronic kidney disease","chronic renal failure"], SLE=["systemic lupus erythematosus","lupus"]
4. Kidney predictions: always add disease_exclusions: ["renal carcinoma","renal cell carcinoma","RCC","kidney cancer","kidney tumor","ccRCC"]
5. Add realistic aliases from biochemistry (full gene name, protein name, gene ID synonyms)
6. DIRECTION RULE — CRITICAL. Use direction: associated by default.
   Only use direction: up or down when the input contains EXPLICIT expression-level
   or activity-level language for that SPECIFIC entity — nothing else qualifies.
     Allowed → up:   elevated, increased, high, overexpressed, upregulated, amplified,
                     activates/activated (entity becomes active/expressed),
                     promotes/promoted (entity's level or activity is increased)
     Allowed → down: reduced, decreased, low, suppressed, downregulated, depleted,
                     inactivated, inhibited, loss-of-expression
     Use absent:     only for literal absence ("not detected", "completely absent",
                     "knock-out", "deletion"). For GWAS/eQTL/variant predictions
                     where the gene is mutated but still expressed, prefer associated.

   FORBIDDEN direction inference (must use associated instead):
   - Pathway/mechanism: drives, mediates, regulates, controls, governs, enables
   - Role descriptions: is a key factor, is involved in, is responsible for
   - Pathway membership: is downstream of
   Examples:
   ✗ "PKM2 drives glycolytic switch"         → direction: associated  (not up)
   ✗ "HIF1A regulates metabolic adaptation"  → direction: associated  (not up)
   ✓ "PKM2 activates glycolytic enzymes"     → direction: up
   ✓ "IL6 promotes inflammation"             → direction: up
   ✓ "PKM2 is elevated in DKD tubular cells" → direction: up
   ✓ "MTHFD2 reduced in DKD"                 → direction: down
   ✓ "NOD2 R702W variant in IBD cohort"      → direction: associated  (variant, not absent)
   The isoform-switch exception is REMOVED — even "PKM1 is replaced by PKM2"
   does not justify direction: up/down without an explicit expression-level word.

7. ENTITY SCOPE — Only generate predictions for entities EXPLICITLY named in the
   input. Do NOT add downstream pathway members, co-regulators, or inferred
   partners. If the user says "validate PKM2 drives glycolytic switch", predict
   ONLY PKM2. Do NOT add PKM1, LDHA, HIF1A, or any other inferred gene.

EXAMPLE:
Input: "Our scRNA-seq shows MTHFD2 and ALDH1L2 are reduced in DKD C_TAL cells, MTHFS is elevated"
Output:
predictions:
  - id: MTHFD2_DKD_CTAL_reduced
    entity: MTHFD2
    entity_type: gene
    aliases: [MTHFD2, "methylenetetrahydrofolate dehydrogenase 2", "mitochondrial MTHFD"]
    disease_context: DKD
    disease_synonyms: ["diabetic nephropathy", "diabetic kidney disease"]
    disease_exclusions: ["renal carcinoma", "renal cell carcinoma", "RCC", "kidney cancer", "ccRCC"]
    cell_type: C_TAL
    cell_type_synonyms: ["cortical thick ascending limb", "thick ascending limb", "TAL cells"]
    tissue: kidney
    organism: human
    direction: down
    note: "MTHFD2 reduced in DKD C_TAL — mitochondrial folate cycle impaired"
    confidence: medium
    novelty: extending

  - id: ALDH1L2_DKD_CTAL_reduced
    entity: ALDH1L2
    entity_type: gene
    aliases: [ALDH1L2, "aldehyde dehydrogenase 1 family member L2", "mitochondrial 10-formylTHF dehydrogenase"]
    disease_context: DKD
    disease_synonyms: ["diabetic nephropathy", "diabetic kidney disease"]
    disease_exclusions: ["renal carcinoma", "renal cell carcinoma", "RCC", "kidney cancer", "ccRCC"]
    cell_type: C_TAL
    cell_type_synonyms: ["cortical thick ascending limb", "thick ascending limb", "TAL cells"]
    tissue: kidney
    organism: human
    direction: down
    note: "ALDH1L2 reduced in DKD C_TAL — THF regeneration impaired"
    confidence: medium
    novelty: extending

  - id: MTHFS_DKD_CTAL_elevated
    entity: MTHFS
    entity_type: gene
    aliases: [MTHFS, "5,10-methenyltetrahydrofolate synthetase", "5-formyltetrahydrofolate cyclo-ligase"]
    disease_context: DKD
    disease_synonyms: ["diabetic nephropathy", "diabetic kidney disease"]
    disease_exclusions: ["renal carcinoma", "renal cell carcinoma", "RCC", "kidney cancer", "ccRCC"]
    cell_type: C_TAL
    cell_type_synonyms: ["cortical thick ascending limb", "thick ascending limb", "TAL cells"]
    tissue: kidney
    organism: human
    direction: up
    note: "MTHFS elevated in DKD C_TAL — compensatory 5-formyl-THF recycling"
    confidence: medium
    novelty: novel"""

# ── LLM gateway ──────────────────────────────────────────────────────────────
# One code path for every hosted model. llm_gateway works out which endpoint
# the user has configured — a company proxy, a public provider, a local Ollama
# — from whatever they happen to have in their environment, so there is nothing
# provider-specific left to maintain here. See llm_gateway.setup_help().


def _strip_code_fence(text: str) -> str:
    """Models like wrapping YAML in ```yaml fences; the parser does not want them."""
    text = re.sub(r'^```(?:yaml)?\s*', '', text, flags=re.MULTILINE)
    return re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE).strip()


def _gateway_convert(prompt: str) -> tuple:
    """Turn a free-text description into predictions.yaml. Returns (yaml, error)."""
    text, err = llm_gateway.chat(
        [{'role': 'system', 'content': _CONVERSION_SYSTEM_PROMPT},
         {'role': 'user',   'content': prompt}],
        max_tokens=800, temperature=0.05, timeout=60,
    )
    return (_strip_code_fence(text), None) if text else (None, err)


# ── llama-cpp-python backend ───────────────────────────────────────────────────
# FIX: cache the model so it loads once per server start, not once per request
# This was the root cause of the 5-minute hang (model reloaded on every call)
_LLAMACPP_MODEL_CACHE: dict = {}
_LLAMACPP_LOCK = None

def _get_llamacpp_model(gguf_path: str):
    global _LLAMACPP_LOCK
    if _LLAMACPP_LOCK is None:
        import threading
        _LLAMACPP_LOCK = threading.Lock()
    with _LLAMACPP_LOCK:
        if gguf_path not in _LLAMACPP_MODEL_CACHE:
            from llama_cpp import Llama
            print(f'[LLM] Loading (first time): {os.path.basename(gguf_path)}')
            _LLAMACPP_MODEL_CACHE[gguf_path] = Llama(
                model_path=gguf_path,
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                n_batch=512,
                verbose=False,
            )
            print('[LLM] Model cached in memory for future requests.')
    return _LLAMACPP_MODEL_CACHE[gguf_path]

def _llamacpp_convert(prompt: str) -> tuple:
    import re as _re, threading
    gguf_path = os.environ.get('PAPERTRAIL_GGUF_PATH', '')
    if not gguf_path:
        return None, 'PAPERTRAIL_GGUF_PATH not set'
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return None, 'llama-cpp-python not installed'
    timeout_s = int(os.environ.get('PAPERTRAIL_LLM_TIMEOUT', '90'))
    result, err_box = [None], [None]
    def _run():
        try:
            llm = _get_llamacpp_model(gguf_path)
            out = llm.create_chat_completion(
                messages=[
                    {'role': 'system', 'content': _CONVERSION_SYSTEM_PROMPT},
                    {'role': 'user',   'content': prompt},
                ],
                temperature=0.05, max_tokens=400,
            )
            text = out['choices'][0]['message']['content'].strip()
            text = _re.sub(r'^```(?:yaml)?\s*', '', text, flags=_re.MULTILINE)
            text = _re.sub(r'\s*```\s*$',       '', text, flags=_re.MULTILINE)
            result[0] = text.strip()
        except Exception as e:
            err_box[0] = str(e)
    t = threading.Thread(target=_run, daemon=True)
    t.start(); t.join(timeout_s)
    if t.is_alive():
        return None, (f'llama-cpp timed out after {timeout_s}s. '
                      'A hosted gateway returns in seconds — see llm_gateway.py.')
    if err_box[0]:
        return None, f'llama-cpp: {err_box[0]}'
    return result[0], None


# ── Which backend will actually be used ──────────────────────────────────────
def _detect_best_backend() -> tuple:
    """
    Return (backend, model, human_label).

    backend is 'gateway' for anything reachable over HTTP, 'llamacpp' for a
    local GGUF file, or 'rules' when there is no LLM at all. Callers treat
    'rules' as "fall back to the regex parser", never as an error.
    """
    forced = os.environ.get('PAPERTRAIL_LLM_BACKEND', '').strip().lower()
    if forced == 'rules':
        return 'rules', '', 'Rule-based (forced)'

    if forced != 'llamacpp':
        gateway = llm_gateway.resolve()
        if gateway is not None:
            return 'gateway', gateway.model, gateway.label()

    # A local GGUF is the last resort: it works offline but is slow on CPU.
    gguf = os.environ.get('PAPERTRAIL_GGUF_PATH', '')
    if gguf and os.path.exists(gguf):
        try:
            import llama_cpp  # noqa: F401
            return 'llamacpp', gguf, f'llama-cpp-python: {os.path.basename(gguf)}'
        except ImportError:
            pass

    return 'rules', '', 'Rule-based fallback (no LLM configured)'


@app.route("/api/llm_status")
def api_llm_status():
    backend, model, label = _detect_best_backend()
    info = llm_gateway.describe()
    speed = {'gateway': '~1-5s', 'llamacpp': '~60-120s on CPU', 'rules': '<1ms'}

    if backend == 'rules':
        hint = llm_gateway.setup_help(short=True)
    elif backend == 'llamacpp':
        hint = ('Local CPU generation is slow. Any OpenAI-compatible gateway '
                'will be far quicker — run: python llm_gateway.py')
    else:
        # Show which variables were picked up; useful when someone has several
        # gateways configured and wonders why this one won.
        picked = ', '.join(f'{role}={var}'
                           for role, var in sorted(info['detected_from'].items()))
        hint = f'Detected from {picked}' if picked else ''

    return jsonify({
        'backend': backend,
        'model': model,
        'message': label,
        'expected_time': speed.get(backend, '?'),
        'gateway': info['name'],
        'gateway_configured': info['configured'],
        'base_url': info['base_url'],
        'detected_from': info['detected_from'],
        'setup_hint': hint,
    })



# ── Comprehensive rule-based fallback ─────────────────────────────────────────
_DIRECTION_TOKENS = {
    "up":
        r"\b(?:upregulat\w*|overexpress\w*|over-express\w*|elevated?|increas\w*|"
        r"higher|greater|enhanc\w*|induc\w*|augment\w*|amplif\w*|gain\w*|"
        r"activat\w*|hyperactivat\w*|accumula\w*|abundant\w*)\b",
    "down":
        r"\b(?:downregulat\w*|underexpress\w*|reduc\w*|decreas\w*|lower\w*|"
        r"suppress\w*|attenuate\w*|impair\w*|deficien\w*|deplet\w*|diminish\w*|"
        r"abrogate\w*|ablat\w*|loss\w*|silenc\w*|knock\w*out\w*|knock-out\w*)\b",
    "preserved":
        r"\b(?:preserv\w*|maintain\w*|unchang\w*|stable|unalter\w*|"
        r"no (?:significant )?(?:change|difference|alteration)|"
        r"similar to (?:controls?|normal)|comparable)\b",
    "absent":
        r"\b(?:absent|undetectable|not detected|complete loss|abolish\w*)\b",
    "bidirectional":
        r"\b(?:bidirectional|variable|heterogeneous|mixed|context-dependent)\b",
}
_CONFIDENCE_BOOST = {
    "high": r"\b(?:significantly|markedly|strongly|consistently|robustly|confirmed|validated|p\s*<\s*0\.0[15]|\d+\.\d+x|fold change)\b",
    "low":  r"\b(?:may|might|possibly|potentially|suggest\w+|speculate|putative|hypothes\w+|trend toward|borderline)\b",
}
_NOVELTY_TOKENS = {
    "novel":    r"\b(?:novel|first report|unprecedented|previously unknown|not (?:previously|yet) (?:reported|described|shown))\b",
    "extending":r"\b(?:extend\w+|further|additionally|also|in addition|support\w+|confirm\w+|consistent with)\b",
    "known":    r"\b(?:well[- ]known|established|reported|previously (?:shown|described|demonstrated)|literature|review)\b",
}
_GENE_SKIP = {"IN","IS","ARE","WAS","WERE","THE","AND","OR","FOR","OF","UP",
              "TO","BY","AT","IT","NO","NOT","BUT","WE","OUR","THIS","THAT",
              "WITH","FROM","AS","ON","AN","BE","CAN","MAY","HAS","HAD",
              "DID","DO","DOES","IF","SO","ALL","ITS","WHICH","BOTH","EACH",
              "THESE","THOSE","BETWEEN","AMONG","THAN","WHEN","WHERE","HOW",
              "WHAT","WHO","ANY","ONE","TWO","THREE","FOUR","FIVE","SOME",
              "AFTER","BEFORE","DURING","WHILE","ALSO","EITHER","NEITHER"}

def _vocab_lookup(table, text: str):
    """
    Most specific vocabulary key that appears in text as a standalone word.

    Plain `key in text` is unsafe here. The vocabulary is full of short
    abbreviations — "as" (ankylosing spondylitis), "b" (B cell), "all" (acute
    lymphoblastic leukaemia), "mi", "ad" — and those match inside ordinary
    words: "disease" contains "as", "ascending" contains "as", and almost every
    sentence contains a "b". That is how a prediction about the thick ascending
    limb ended up labelled as ankylosing spondylitis in a B cell.

    Longest key wins, so a multi-word term is preferred over a short
    abbreviation that happens to also appear. A trailing plural "s" is
    tolerated, so "pDCs" still matches the key "pdc".
    """
    for key in sorted(table, key=len, reverse=True):
        k = str(key).strip().lower()
        if k and re.search(rf"(?<![0-9a-z]){re.escape(k)}s?(?![0-9a-z])", text):
            return key
    return None


def _rule_convert(prompt: str) -> str:
    import re as _re
    p, p_l = prompt.strip(), prompt.strip().lower()
    _clause_separators = _re.compile(
        r"\s+(?:while|whereas|but|although|however|though|yet|on the other hand|in contrast|conversely)\s+",
        _re.IGNORECASE)
    clauses = _clause_separators.split(p)
    _exclude_as_entities = (
        _GENE_SKIP
        | set(_DISEASE_TABLE.keys())
        | {v[0] for v in _DISEASE_TABLE.values()}
        | set(_CELL_TYPE_TABLE.keys())
        | {v[0] for v in _CELL_TYPE_TABLE.values()}
        | {"DKD","HKD","CKD","AKI","SLE","RA","IBD","CRC","NAFLD","AD","PD","T2D","T1D",
           "C_TAL","CTAL","IPT","PDC","TREG","PODO"}
    )
    _generic_biochem = {"serine","glycine","alanine","leucine","isoleucine","valine",
                        "threonine","methionine","cysteine","proline","tyrosine",
                        "phenylalanine","tryptophan","histidine","arginine","lysine",
                        "aspartate","glutamate","asparagine","glutamine","folate",
                        "folic","acid","oxidase","reductase","synthetase","kinase",
                        "ligase","dehydrogenase"}
    all_entities, seen_e = [], set()
    for match in _re.finditer(r"\b([A-Z][A-Z0-9]{1,11}[0-9]?)\b", p):
        e = match.group(1)
        if (e.upper() not in (s.upper() for s in _exclude_as_entities)
                and e.lower() not in _generic_biochem and e not in seen_e):
            all_entities.append(e); seen_e.add(e)
    # Aliases for non-gene entities (metabolites, lipids, drugs, etc.)
    # that use lowercase/hyphenated names. Extend in papertrail_vocab.json.
    _known_entity_aliases = {
        "d-serine":"D-serine","d-ser":"D-serine","homocysteine":"homocysteine",
        "hcy":"homocysteine","glutathione":"glutathione","gsh":"glutathione",
        "d-alanine":"D-alanine","putrescine":"putrescine","spermidine":"spermidine",
        "norepinephrine":"norepinephrine","thf":"THF","pge2":"PGE2","dag":"DAG",
    }
    for alias, canonical in _known_entity_aliases.items():
        # Use word-boundary check: 'thf' must not match inside 'mthfs'
        import re as _re2
        if _re2.search(r'(?<![a-z])' + _re2.escape(alias) + r'(?![a-z0-9])', p_l) \
                and canonical not in seen_e:
            all_entities.append(canonical); seen_e.add(canonical)
    all_entities = list(dict.fromkeys(all_entities))
    disease_ctx, disease_syns = "any", []
    _dis_key = _vocab_lookup(_DISEASE_TABLE, p_l)
    if _dis_key:
        disease_ctx, disease_syns = _DISEASE_TABLE[_dis_key]
    tissue = "any"
    for tname, (keywords, tval) in _TISSUE_KEYWORDS.items():
        if _vocab_lookup(keywords, p_l): tissue = tval; break
    if disease_ctx in ("DKD","HKD","CKD","AKI") and tissue == "any": tissue = "kidney"
    cell_code, cell_syns = "any", []
    _ct_key = _vocab_lookup(_CELL_TYPE_TABLE, p_l)
    if _ct_key:
        cell_code, cell_syns = _CELL_TYPE_TABLE[_ct_key]
    global_direction = "bidirectional"
    for dname, dpat in _DIRECTION_TOKENS.items():
        if _re.search(dpat, p, _re.IGNORECASE): global_direction = dname; break
    confidence = "medium"
    if _re.search(_CONFIDENCE_BOOST["high"], p, _re.IGNORECASE): confidence = "high"
    elif _re.search(_CONFIDENCE_BOOST["low"], p, _re.IGNORECASE): confidence = "low"
    novelty = "extending"
    if _re.search(_NOVELTY_TOKENS["novel"], p, _re.IGNORECASE): novelty = "novel"
    elif _re.search(_NOVELTY_TOKENS["known"], p, _re.IGNORECASE): novelty = "known"
    excl = _KIDNEY_EXCLUSIONS_LIST if tissue == "kidney" else []
    entities_to_use = all_entities if all_entities else ["GENE"]
    predictions = []
    for entity in entities_to_use[:8]:
        local_dir = global_direction
        pat_e = _re.escape(entity)
        entity_clause = next(
            (c for c in clauses if _re.search(pat_e, c, _re.IGNORECASE)), None)
        if entity_clause:
            best_dist, best_dir = float("inf"), None
            for m_entity in _re.finditer(pat_e, entity_clause, _re.IGNORECASE):
                for dname, dpat in _DIRECTION_TOKENS.items():
                    for m_dir in _re.finditer(dpat, entity_clause, _re.IGNORECASE):
                        dist = abs(m_entity.start() - m_dir.start())
                        if dist < best_dist: best_dist, best_dir = dist, dname
            if best_dir: local_dir = best_dir
        else:
            for m_e in _re.finditer(pat_e, p, _re.IGNORECASE):
                win = p[max(0,m_e.start()-100):m_e.end()+100].lower()
                for dname, dpat in _DIRECTION_TOKENS.items():
                    if _re.search(dpat, win, _re.IGNORECASE): local_dir = dname; break
                break
        dir_word = {"up":"elevated","down":"reduced","preserved":"preserved",
                    "absent":"absent","bidirectional":"bidirectional"}.get(local_dir, local_dir)
        pred_id = (f"{entity}_{disease_ctx}_{cell_code}_{dir_word}"
                   .replace("_any","").replace("__","_"))
        predictions.append({
            "id":pred_id,"entity":entity,"entity_type":"gene","aliases":[entity],
            "disease_context":disease_ctx,"disease_synonyms":disease_syns,
            "disease_exclusions":excl,"cell_type":cell_code,
            "cell_type_synonyms":cell_syns,"tissue":tissue,"organism":"human",
            "direction":local_dir,
            "note":f"{entity} {dir_word} in {disease_ctx}" + (f" {cell_code}" if cell_code != "any" else ""),
            "confidence":confidence,"novelty":novelty,
        })
    import yaml as _y
    return _y.dump({"predictions":predictions},default_flow_style=False,
                   allow_unicode=True,sort_keys=False,width=120)

def _expand_facts_to_yaml(facts: list) -> str:
    """
    Convert a list of compact fact dicts (from LLM JSON extraction) to
    full predictions.yaml with all synonym/exclusion fields populated.
    Each fact: {entity, direction, disease, cell_type, confidence, novelty}
    """
    import yaml as _y
    predictions = []
    for fact in facts:
        entity    = str(fact.get("entity",   "GENE")).strip()
        direction = str(fact.get("direction", "bidirectional")).strip().lower()
        disease   = str(fact.get("disease",   "any")).strip()
        cell_type = str(fact.get("cell_type", "any")).strip()
        confidence= str(fact.get("confidence","medium")).strip().lower()
        novelty   = str(fact.get("novelty",   "extending")).strip().lower()

        # Validate direction
        if direction not in ("up","down","preserved","absent","bidirectional"):
            direction = "bidirectional"

        # Look up disease synonyms and exclusions
        dis_entry  = _DISEASE_TABLE.get(disease.lower())
        if dis_entry:
            dis_abbrev, dis_syns = dis_entry
        else:
            dis_abbrev, dis_syns = disease.upper(), []
        # Infer tissue from disease
        _kidney_diseases = {"DKD","HKD","CKD","AKI"}
        tissue  = "kidney" if dis_abbrev in _kidney_diseases else "any"
        excl    = _KIDNEY_EXCLUSIONS_LIST if tissue == "kidney" else []

        # Look up cell type synonyms
        ct_key   = cell_type.lower()
        ct_entry = _CELL_TYPE_TABLE.get(ct_key)
        if ct_entry:
            ct_code, ct_syns = ct_entry
        else:
            ct_code, ct_syns = cell_type, []

        dir_word = {"up":"elevated","down":"reduced","preserved":"preserved",
                    "absent":"absent","bidirectional":"bidirectional"}.get(direction,direction)
        pred_id  = (f"{entity}_{dis_abbrev}_{ct_code}_{dir_word}"
                    .replace("_any","").replace("__","_"))

        predictions.append({
            "id":               pred_id,
            "entity":           entity,
            "entity_type":      "gene",
            "aliases":          [entity],
            "disease_context":  dis_abbrev,
            "disease_synonyms": dis_syns,
            "disease_exclusions": excl,
            "cell_type":        ct_code,
            "cell_type_synonyms": ct_syns,
            "tissue":           tissue,
            "organism":         "human",
            "direction":        direction,
            "note":             (f"{entity} {dir_word} in {dis_abbrev}"
                                 + (f" {ct_code}" if ct_code != "any" else "")),
            "confidence":       confidence,
            "novelty":          novelty,
        })
    return _y.dump({"predictions": predictions}, default_flow_style=False,
                   allow_unicode=True, sort_keys=False, width=120)


# ══════════════════════════════════════════════════════════════════════════
# _strip_yaml_anchors()
# PyYAML uses &id001/*id001 anchor notation when the SAME Python list object
# is assigned to multiple prediction dicts (e.g. _KIDNEY_EXCLUSIONS_LIST
# referenced by every kidney prediction in _canonicalise_yaml).
# Stripping anchors: round-trip through safe_load to create fresh objects,
# then dump with a custom Dumper that overrides ignore_aliases.
# Called immediately after _canonicalise_yaml — that function is untouched.
# ══════════════════════════════════════════════════════════════════════════
import yaml as _yaml_mod

class _NoAliasDumper(_yaml_mod.Dumper):
    def ignore_aliases(self, data):
        return True

def _strip_yaml_anchors(yaml_str: str) -> str:
    """Re-serialise YAML without any anchor/alias notation."""
    try:
        return _yaml_mod.dump(
            _yaml_mod.safe_load(yaml_str),
            Dumper=_NoAliasDumper,
            default_flow_style=False, allow_unicode=True,
            sort_keys=False, width=120)
    except Exception:
        return yaml_str


# ══════════════════════════════════════════════════════════════════════════
# _canonicalise_yaml() — ADDED
# Post-processes LLM YAML to replace disease_synonyms, disease_exclusions,
# and cell_type_synonyms with deterministic values from _DISEASE_TABLE /
# _CELL_TYPE_TABLE. Ensures PubMed queries are stable across LLM runs.
# ══════════════════════════════════════════════════════════════════════════
def _canonicalise_yaml(yaml_str: str) -> str:
    """Replace LLM-generated synonym/exclusion fields with canonical rule-based values."""
    try:
        import yaml as _y
        parsed = _y.safe_load(yaml_str)
        if not isinstance(parsed, dict):
            return yaml_str
        preds = parsed.get("predictions", [])
        if not isinstance(preds, list):
            return yaml_str

        _kidney_diseases = {"DKD", "HKD", "CKD", "AKI"}

        for pred in preds:
            if not isinstance(pred, dict):
                continue

            # ── 1. Canonicalise disease_synonyms & disease_exclusions ─────
            dc = str(pred.get("disease_context", "") or "").strip()
            dc_lower = dc.lower()

            canon_entry = None
            if dc_lower in _DISEASE_TABLE:
                canon_entry = _DISEASE_TABLE[dc_lower]
            else:
                for key, val in _DISEASE_TABLE.items():
                    if val[0].lower() == dc_lower:
                        canon_entry = val
                        break

            if canon_entry:
                pred["disease_context"]  = canon_entry[0]   # normalise abbreviation
                pred["disease_synonyms"] = canon_entry[1]   # always canonical

            # Kidney exclusions: always override with full canonical list
            tissue_str = str(pred.get("tissue", "")).lower()
            if (pred.get("disease_context", "") in _kidney_diseases
                    or any(kw in tissue_str for kw in
                           ("kidney", "renal", "nephro", "tubular", "glomerular"))):
                pred["disease_exclusions"] = _KIDNEY_EXCLUSIONS_LIST

            # ── 2. Canonicalise cell_type_synonyms ────────────────────────
            ct = str(pred.get("cell_type", "") or "").strip()
            ct_lower = ct.lower()
            ct_entry = None
            if ct_lower in _CELL_TYPE_TABLE:
                ct_entry = _CELL_TYPE_TABLE[ct_lower]
            else:
                for key, val in _CELL_TYPE_TABLE.items():
                    if key in ct_lower or ct_lower in key:
                        ct_entry = val
                        break

            if ct_entry:
                pred["cell_type"]          = ct_entry[0]    # normalise code
                pred["cell_type_synonyms"] = ct_entry[1]    # always canonical

            # ── 3. Hard-cap alias list to 3 items ─────────────────────────
            aliases = pred.get("aliases", [])
            if isinstance(aliases, list) and len(aliases) > 3:
                pred["aliases"] = aliases[:3]

        return _y.dump({"predictions": preds},
                       default_flow_style=False, allow_unicode=True,
                       sort_keys=False, width=120)
    except Exception:
        # If YAML parsing fails, return unchanged — let _validate_and_supplement handle it
        return yaml_str


def _validate_and_supplement(prompt: str, llm_yaml: str) -> tuple:
    """
    Check that every entity found in the prompt has a prediction in the LLM YAML.
    If the LLM truncated (e.g. only generated 1 of 3 predictions), run the
    rule-based converter and merge in the missing entries.
    Returns (final_yaml, was_supplemented: bool).
    """
    import re as _re
    try:
        import yaml as _y
        parsed = _y.safe_load(llm_yaml)
        llm_preds = parsed.get("predictions", []) if isinstance(parsed, dict) else []
    except Exception:
        # LLM returned invalid YAML — fall back entirely to rules
        return _rule_convert(prompt), True

    # Extract expected entities from the prompt using the rule-based extractor
    rule_yaml = _rule_convert(prompt)
    try:
        rule_parsed = _y.safe_load(rule_yaml)
        rule_preds  = rule_parsed.get("predictions", []) if isinstance(rule_parsed, dict) else []
    except Exception:
        rule_preds = []

    expected_entities = {p["entity"].upper() for p in rule_preds}
    llm_entities      = {p.get("entity", "").upper() for p in llm_preds}
    missing           = expected_entities - llm_entities

    if not missing:
        return llm_yaml, False   # LLM covered everything

    # Merge missing predictions from rule-based into the LLM output
    rule_map = {p["entity"].upper(): p for p in rule_preds}
    for entity_upper in sorted(missing):
        filler = rule_map.get(entity_upper)
        if filler:
            llm_preds.append(filler)

    merged_yaml = _y.dump({"predictions": llm_preds},
                           default_flow_style=False, allow_unicode=True,
                           sort_keys=False, width=120)
    return merged_yaml, True



@app.route("/api/convert_prompt", methods=["POST"])
def api_convert_prompt():
    data = request.get_json(force=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is empty"}), 400

    backend, model, _msg = _detect_best_backend()

    yaml_str, err = None, None
    if backend == "gateway":
        yaml_str, err = _gateway_convert(prompt)
    elif backend == "llamacpp":
        yaml_str, err = _llamacpp_convert(prompt)

    if yaml_str:
        # Step 1: canonicalise synonyms/exclusions (deterministic override)
        yaml_str = _canonicalise_yaml(yaml_str)
        # Step 2: remove YAML anchor/alias notation (&id001/*id001 artefacts)
        yaml_str = _strip_yaml_anchors(yaml_str)
        # Step 3: supplement any entities the LLM missed
        yaml_str, was_supplemented = _validate_and_supplement(prompt, yaml_str)
        return jsonify({
            "yaml": yaml_str, "source": backend, "model": model,
            "supplemented": was_supplemented,
        })

    # No LLM, or the gateway call failed — the regex parser still produces
    # usable YAML, so this is a degraded result rather than an error.
    return jsonify({
        "yaml": _rule_convert(prompt), "source": "rules",
        "warning": err or "No LLM configured — rule-based extraction used",
        "setup_hint": llm_gateway.setup_help(short=True),
    })


# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════

UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PaperTrail — Literature Concordance</title>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Instrument+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@300;400;500&display=swap" rel="stylesheet"/>
<style>
*,::before,::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --ink:#0d1117;--ink2:#1f2937;--ink3:#4b5563;--ink4:#9ca3af;
  --paper:#ffffff;--p2:#f9fafb;--p3:#f3f4f6;--p4:#e5e7eb;
  --accent:#1d4ed8;--al:#eff6ff;--am:#3b82f6;
  --green:#065f46;--gl:#ecfdf5;--gm:#10b981;
  --amber:#78350f;--aml:#fffbeb;--amm:#f59e0b;
  --red:#7f1d1d;--rl:#fef2f2;--rm:#ef4444;
  --purple:#5b21b6;--pl:#f5f3ff;--pm:#8b5cf6;
  --ff-h:'Instrument Serif',Georgia,serif;
  --ff-b:'Instrument Sans',system-ui,sans-serif;
  --ff-m:'JetBrains Mono',monospace;
  --r:8px;--r2:14px;
  --sh:0 1px 3px rgba(0,0,0,.07),0 1px 2px rgba(0,0,0,.04);
  --sh2:0 4px 16px rgba(0,0,0,.09),0 2px 6px rgba(0,0,0,.05);
  --sh3:0 20px 60px rgba(0,0,0,.13),0 4px 12px rgba(0,0,0,.07);
  --dur:.22s;--ease:cubic-bezier(.4,0,.2,1);
}
html{font-size:15px;scroll-behavior:smooth}
body{background:var(--p2);color:var(--ink);font-family:var(--ff-b);
  line-height:1.6;min-height:100vh;overflow-x:hidden}
button,input,select,textarea{font-family:inherit}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-thumb{background:var(--p4);border-radius:3px}

/* ── TOPBAR ────────────────────────────────────────────────────── */
.topbar{background:var(--ink);height:52px;display:flex;align-items:center;
  padding:0 2rem;position:sticky;top:0;z-index:200;
  border-bottom:1px solid rgba(255,255,255,.05)}
.tb-logo{font-family:var(--ff-h);font-size:1.3rem;font-style:italic;color:#fff;
  letter-spacing:-.01em;display:flex;align-items:baseline;gap:.35rem}
.tb-logo .dot{color:var(--am);font-size:1rem;font-style:normal}
.tb-sub{font-family:var(--ff-m);font-size:.58rem;color:rgba(255,255,255,.28);
  letter-spacing:.14em;text-transform:uppercase;margin-left:.2rem}
.tb-right{margin-left:auto;display:flex;align-items:center;gap:.75rem}

/* Mode toggle */
.mode-toggle{display:flex;background:rgba(255,255,255,.07);border-radius:8px;
  padding:3px;gap:2px;border:1px solid rgba(255,255,255,.08)}
.mt-btn{font-family:var(--ff-m);font-size:.62rem;letter-spacing:.02em;
  padding:.3rem .95rem;border-radius:6px;border:none;cursor:pointer;
  background:transparent;color:rgba(255,255,255,.4);transition:all var(--dur) var(--ease);
  display:flex;align-items:center;gap:.35rem}
.mt-btn.on{background:#fff;color:var(--ink2);font-weight:500;
  box-shadow:0 1px 3px rgba(0,0,0,.15)}
.mt-btn:not(.on):hover{color:rgba(255,255,255,.75);background:rgba(255,255,255,.08)}

/* ── PANELS ─────────────────────────────────────────────────────── */
.panel{display:none;max-width:1080px;margin:0 auto;padding:2.5rem 2rem 5rem;
  animation:panelIn var(--dur) var(--ease) both}
.panel.on{display:block}
@keyframes panelIn{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}

/* ── DEMO HERO ─────────────────────────────────────────────────── */
.demo-hero{background:var(--ink);margin:-2.5rem -2rem 2.5rem;
  padding:3rem 2rem 2.5rem;position:relative;overflow:hidden}
.demo-hero::after{content:'';position:absolute;bottom:0;left:0;right:0;
  height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.07),transparent)}
.dh-eye{font-family:var(--ff-m);font-size:.6rem;letter-spacing:.2em;
  text-transform:uppercase;color:rgba(255,255,255,.25);margin-bottom:.65rem;
  display:flex;align-items:center;gap:.6rem}
.dh-eye::before{content:'';width:16px;height:1px;background:rgba(255,255,255,.2)}
.dh-h{font-family:var(--ff-h);font-size:2.2rem;font-style:italic;color:#fff;
  line-height:1.18;max-width:640px;margin-bottom:.65rem}
.dh-h em{color:var(--am);font-style:inherit}
.dh-p{font-size:.93rem;color:rgba(255,255,255,.48);max-width:540px;line-height:1.72}

/* ── STEP SYSTEM ─────────────────────────────────────────────────── */
.step-row{display:flex;align-items:center;gap:.55rem;margin-bottom:1.1rem}
.step-num{width:22px;height:22px;background:var(--ink);color:#fff;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-family:var(--ff-m);font-size:.62rem;font-weight:500;flex-shrink:0;
  border:1.5px solid var(--p4)}
.step-num.done{background:var(--gm);border-color:var(--gm)}
.step-num.active{background:var(--accent);border-color:var(--accent)}
.step-lbl{font-family:var(--ff-m);font-size:.62rem;letter-spacing:.12em;
  text-transform:uppercase;color:var(--ink3)}
.step-div{flex:1;height:1px;background:var(--p4)}
.step-hint{font-family:var(--ff-b);font-size:.82rem;font-style:italic;
  color:var(--ink4);font-weight:400;letter-spacing:0;text-transform:none}

/* ── CASE TILES ─────────────────────────────────────────────────── */
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:.9rem;margin-bottom:2.5rem}
@media(max-width:900px){.tiles{grid-template-columns:repeat(2,1fr)}}
@media(max-width:500px){.tiles{grid-template-columns:1fr}}
.tile{background:var(--paper);border:1.5px solid var(--p4);border-radius:var(--r2);
  padding:1.3rem 1.15rem 1.1rem;cursor:pointer;transition:all var(--dur) var(--ease);
  position:relative;overflow:hidden;user-select:none}
.tile-accent{position:absolute;top:0;left:0;right:0;height:3px;
  background:var(--tc);transform:scaleX(0);transform-origin:left;
  transition:transform var(--dur) var(--ease)}
.tile:hover .tile-accent,.tile.active .tile-accent{transform:scaleX(1)}
.tile:hover{border-color:var(--tc);box-shadow:var(--sh2)}
.tile.active{border-color:var(--tc);background:var(--tl);box-shadow:var(--sh2)}
.tile-type{font-family:var(--ff-m);font-size:.57rem;letter-spacing:.08em;
  text-transform:uppercase;color:var(--tc);margin-bottom:.6rem;
  background:rgba(var(--tc-rgb),.1);padding:.15rem .45rem;border-radius:3px;
  display:inline-block;opacity:.85}
.tile-emoji{font-size:1.7rem;display:block;margin-bottom:.5rem}
.tile-name{font-family:var(--ff-h);font-size:.97rem;font-style:italic;color:var(--ink);
  line-height:1.3;margin-bottom:.15rem}
.tile-sub{font-family:var(--ff-m);font-size:.6rem;color:var(--ink4);
  letter-spacing:.02em;line-height:1.4}
.tile-count{position:absolute;top:.75rem;right:.85rem;font-family:var(--ff-m);
  font-size:.58rem;color:var(--ink4);background:var(--p3);
  border:1px solid var(--p4);border-radius:3px;padding:.1rem .42rem}

/* ── CASE DETAIL ─────────────────────────────────────────────────── */
.case-detail{display:none;animation:panelIn var(--dur) var(--ease) both}
.case-detail.on{display:block}
.cd-nav{display:flex;align-items:center;gap:.85rem;margin-bottom:1.5rem;flex-wrap:wrap}
.cd-back{font-family:var(--ff-m);font-size:.62rem;padding:.35rem .8rem;
  background:var(--paper);border:1px solid var(--p4);border-radius:6px;
  color:var(--ink3);cursor:pointer;display:inline-flex;align-items:center;gap:.35rem;
  transition:all var(--dur) var(--ease)}
.cd-back:hover{background:var(--p3);color:var(--ink);border-color:var(--ink3)}
.cd-crumb{font-family:var(--ff-m);font-size:.65rem;color:var(--ink4);
  display:flex;align-items:center;gap:.35rem}
.cd-crumb span{color:var(--ink)}
.cd-nav-r{margin-left:auto;display:flex;gap:.5rem}
.cd-nav-arrow{width:30px;height:30px;background:var(--paper);border:1px solid var(--p4);
  border-radius:6px;display:flex;align-items:center;justify-content:center;
  cursor:pointer;color:var(--ink3);transition:all var(--dur) var(--ease);flex-shrink:0}
.cd-nav-arrow:hover{background:var(--p3);color:var(--ink);border-color:var(--ink3)}
.cd-nav-arrow:disabled{opacity:.3;cursor:not-allowed}

.cd-header{display:flex;align-items:flex-start;gap:1.25rem;margin-bottom:.9rem;flex-wrap:wrap}
.cd-emoji{font-size:2rem;flex-shrink:0;margin-top:.1rem}
.cd-info{flex:1;min-width:0}
.cd-name{font-family:var(--ff-h);font-size:1.65rem;font-style:italic;color:var(--ink);
  line-height:1.2;margin-bottom:.15rem}
.cd-sub{font-family:var(--ff-m);font-size:.63rem;color:var(--ink4);letter-spacing:.03em}
.type-pill{display:inline-flex;align-items:center;gap:.35rem;margin-top:.45rem;
  font-family:var(--ff-m);font-size:.6rem;padding:.22rem .65rem;
  border-radius:20px;border:1px solid currentColor;color:var(--tc);opacity:.85}
.type-pill::before{content:'';width:5px;height:5px;border-radius:50%;
  background:currentColor;flex-shrink:0}

.blurb{font-size:.92rem;color:var(--ink3);line-height:1.7;
  padding:.75rem 1rem .75rem 1.1rem;background:var(--p3);
  border-left:3px solid var(--tc);border-radius:0 var(--r) var(--r) 0;
  margin-bottom:1.5rem}
.type-desc{font-size:.85rem;color:var(--ink4);font-style:italic;
  padding:.4rem .65rem;background:var(--paper);border:1px solid var(--p4);
  border-radius:var(--r);margin-top:.55rem;display:flex;align-items:flex-start;gap:.4rem}
.type-desc::before{content:'💡';font-style:normal;flex-shrink:0}

/* Edit hint */
.edit-hint{font-family:var(--ff-m);font-size:.6rem;color:var(--ink4);
  margin-bottom:.55rem;display:flex;align-items:center;gap:.4rem}
.eh-dot{width:5px;height:5px;border-radius:50%;background:var(--amm);flex-shrink:0;
  animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:.4;transform:scale(.8)}50%{opacity:1;transform:scale(1.2)}}

/* Prediction table */
.pred-wrap{background:var(--paper);border:1px solid var(--p4);border-radius:var(--r2);
  overflow:hidden;box-shadow:var(--sh);margin-bottom:1.1rem}
.ptbl{width:100%;border-collapse:collapse;font-size:.83rem}
.ptbl thead th{font-family:var(--ff-m);font-size:.57rem;letter-spacing:.09em;
  text-transform:uppercase;color:var(--ink3);padding:.6rem .85rem .5rem;
  text-align:left;border-bottom:1px solid var(--p4);background:var(--p3);
  white-space:nowrap;position:sticky;top:0;z-index:2}
.ptbl tbody tr{border-bottom:1px solid var(--p4);transition:background var(--dur)}
.ptbl tbody tr:last-child{border-bottom:none}
.ptbl tbody tr:hover{background:var(--p2)}
.ptbl td{padding:.45rem .85rem;vertical-align:middle}
.rn{font-family:var(--ff-m);font-size:.6rem;color:var(--ink4);
  background:var(--p3);border:1px solid var(--p4);border-radius:4px;
  padding:.08rem .38rem;user-select:none}

/* Editable cells */
.ec{border:1.5px solid transparent;border-radius:5px;padding:.18rem .38rem;
  outline:none;transition:all var(--dur) var(--ease);cursor:text;
  min-width:50px;display:inline-block;line-height:1.45;
  border-bottom:1px dashed var(--p4)}
.ec:hover{background:var(--p3);border-bottom-color:var(--ink4)}
.ec:focus{border:1.5px solid var(--am);background:var(--al);
  box-shadow:0 0 0 3px rgba(59,130,246,.1);border-radius:5px}
.ec.entity{font-family:var(--ff-m);font-size:.77rem;font-weight:500;color:var(--ink)}
.ec.small{font-size:.76rem;color:var(--ink3)}
.ec.note{font-size:.74rem;color:var(--ink4);font-style:italic;max-width:190px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* Direction select */
.dir-sel{font-family:var(--ff-m);font-size:.64rem;padding:.22rem .55rem;
  border:1.5px solid var(--p4);border-radius:5px;background:var(--p3);
  color:var(--ink);cursor:pointer;outline:none;
  transition:all var(--dur) var(--ease)}
.dir-sel:focus{border-color:var(--am);background:var(--al)}
.dir-sel[data-val="up"]   {color:var(--green);background:var(--gl);border-color:var(--gm)}
.dir-sel[data-val="down"] {color:var(--red);background:var(--rl);border-color:var(--rm)}

/* Row actions */
.row-del{width:22px;height:22px;border:1px solid var(--p4);border-radius:4px;
  background:transparent;color:var(--ink4);cursor:pointer;font-size:.8rem;
  display:flex;align-items:center;justify-content:center;
  transition:all var(--dur) var(--ease);flex-shrink:0}
.row-del:hover{background:var(--rl);color:var(--rm);border-color:var(--rm)}

/* Action bar */
.action-bar{display:flex;align-items:center;gap:.7rem;flex-wrap:wrap;padding:.1rem 0}
.btn-run{font-family:var(--ff-m);font-size:.74rem;letter-spacing:.02em;
  padding:.55rem 1.5rem;background:var(--ink);color:#fff;border:none;
  border-radius:8px;cursor:pointer;display:inline-flex;align-items:center;gap:.4rem;
  transition:all var(--dur) var(--ease);box-shadow:var(--sh)}
.btn-run:hover:not(:disabled){background:var(--accent);box-shadow:var(--sh2);
  transform:translateY(-1px)}
.btn-run:active{transform:none}
@keyframes spin{to{transform:rotate(360deg)}}
.btn-run:disabled{background:var(--ink3);cursor:not-allowed}
.btn-out{font-family:var(--ff-m);font-size:.64rem;padding:.48rem .95rem;
  background:transparent;border:1px solid var(--p4);border-radius:7px;
  color:var(--ink3);cursor:pointer;transition:all var(--dur) var(--ease)}
.btn-out:hover{background:var(--p3);color:var(--ink);border-color:var(--ink4)}
.btn-out.active{background:var(--ink);color:#fff;border-color:var(--ink)}
.btn-add-row{font-family:var(--ff-m);font-size:.62rem;padding:.35rem .75rem;
  background:transparent;border:1px dashed var(--p4);border-radius:6px;
  color:var(--ink4);cursor:pointer;transition:all var(--dur) var(--ease);
  display:flex;align-items:center;gap:.3rem}
.btn-add-row:hover{border-color:var(--ink3);color:var(--ink);background:var(--p3)}

/* Spinner dots */
.ld{display:none;align-items:center;gap:.4rem;font-family:var(--ff-m);
  font-size:.64rem;color:var(--ink4);margin-left:.2rem}
.ld-d{width:5px;height:5px;border-radius:50%;background:var(--amm);
  animation:ld-pulse 1s ease-in-out infinite}
.ld-d:nth-child(2){animation-delay:.14s}.ld-d:nth-child(3){animation-delay:.28s}
@keyframes ld-pulse{0%,100%{opacity:.2;transform:scale(.7)}50%{opacity:1;transform:scale(1.3)}}

/* NL prompt input */
.input-mode-toggle{display:flex;gap:6px;margin-bottom:12px}
.imt-btn{font-family:var(--ff-m);font-size:.72rem;padding:5px 14px;border-radius:16px;
  border:1.5px solid var(--ink3);background:transparent;cursor:pointer;
  color:var(--ink2);transition:all .15s}
.imt-btn.active{border-color:var(--accent);background:var(--accent);
  color:#fff;font-weight:600}
.nl-panel{display:none}
.nl-panel.visible{display:block}
.yaml-panel-wrap{display:block}
.yaml-panel-wrap.hidden{display:none}
.nl-prompt-area{width:100%;min-height:110px;box-sizing:border-box;
  font-family:var(--ff-s);font-size:.82rem;padding:10px 12px;
  border:1.5px solid var(--ink3);border-radius:8px;resize:vertical;
  background:var(--bg1);color:var(--ink1);transition:border-color .15s}
.nl-prompt-area:focus{outline:none;border-color:var(--accent)}
.nl-prompt-area::placeholder{color:var(--ink3)}
.nl-actions{display:flex;gap:8px;align-items:center;margin-top:8px}
.nl-source-badge{font-size:.68rem;padding:2px 8px;border-radius:10px;
  background:var(--bg2);color:var(--ink2)}
.nl-source-badge.llm{background:#e6f4ea;color:#1e6b35}
.nl-source-badge.rules{background:#fff8e1;color:#7a5c00}
.nl-warn{font-size:.71rem;color:var(--rm);margin-top:4px}
/* YAML panel */
.yaml-panel{display:none;border-radius:var(--r2);overflow:hidden;
  border:1px solid #1e293b;box-shadow:var(--sh2);margin-top:.85rem}
.yaml-panel.on{display:block;animation:panelIn .2s ease both}
.yaml-bar{background:#0f172a;padding:.5rem 1rem;display:flex;
  align-items:center;justify-content:space-between}
.yaml-bar-l{font-family:var(--ff-m);font-size:.6rem;color:rgba(255,255,255,.28);
  letter-spacing:.1em;text-transform:uppercase}
.yaml-bar-r{display:flex;gap:.5rem}
.ybtn{font-family:var(--ff-m);font-size:.58rem;padding:.2rem .6rem;
  background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);
  border-radius:4px;color:rgba(255,255,255,.4);cursor:pointer;transition:all var(--dur)}
.ybtn:hover{background:rgba(255,255,255,.14);color:#fff}
.yaml-pre{background:#0f172a;color:#94a3b8;font-family:var(--ff-m);
  font-size:.67rem;line-height:1.75;padding:1rem 1.2rem;
  overflow-x:auto;white-space:pre;max-height:360px;overflow-y:auto;margin:0}
.yk{color:#7dd3fc}.yv{color:#86efac}.ys{color:#fde68a}.yc{color:#4b5563;font-style:italic}

/* ── FULL PANEL ─────────────────────────────────────────────────── */
.full-hero{background:var(--ink);margin:-2.5rem -2rem 2.5rem;
  padding:2.75rem 2rem 2.25rem}
.fh-h{font-family:var(--ff-h);font-size:1.6rem;font-style:italic;color:#fff;margin-bottom:.3rem}
.fh-p{font-size:.9rem;color:rgba(255,255,255,.42);max-width:520px;line-height:1.65}
.full-layout{display:grid;grid-template-columns:1fr 320px;gap:1.5rem;align-items:start}
@media(max-width:760px){.full-layout{grid-template-columns:1fr}}
.form-card{background:var(--paper);border:1px solid var(--p4);border-radius:var(--r2);
  padding:1.75rem;box-shadow:var(--sh)}
.proj-row{display:flex;align-items:center;gap:.75rem;margin-bottom:1.25rem;
  padding:.75rem 1rem;background:var(--p2);border:1px solid var(--p4);
  border-radius:var(--r);border-left:3px solid var(--am)}
.proj-row label{font-family:var(--ff-m);font-size:.62rem;text-transform:uppercase;
  letter-spacing:.1em;color:var(--amber);white-space:nowrap;flex-shrink:0}
.proj-input{flex:1;background:transparent;border:none;outline:none;
  font-family:var(--ff-m);font-size:.82rem;color:var(--ink);font-weight:500}
.proj-input::placeholder{color:var(--ink4);font-weight:400}
.fl{font-family:var(--ff-m);font-size:.6rem;letter-spacing:.1em;text-transform:uppercase;
  color:var(--ink3);margin-bottom:.4rem;display:flex;align-items:center;gap:.4rem}
.fl em{font-family:var(--ff-b);font-size:.8rem;font-style:italic;
  color:var(--ink4);text-transform:none;letter-spacing:0}
.yaml-upload-bar{display:flex;align-items:center;gap:.65rem;margin-bottom:.45rem}
.upload-btn{font-family:var(--ff-m);font-size:.62rem;padding:.3rem .75rem;
  background:var(--p3);border:1px solid var(--p4);border-radius:6px;
  color:var(--ink3);cursor:pointer;display:inline-flex;align-items:center;gap:.35rem;
  transition:all var(--dur) var(--ease)}
.upload-btn:hover{background:var(--p2);color:var(--ink);border-color:var(--ink4)}
.yaml-upload-hint{font-family:var(--ff-m);font-size:.62rem;color:var(--ink4);font-style:italic}
.fi{width:100%;background:var(--p3);border:1px solid var(--p4);border-radius:7px;
  padding:.55rem .85rem;font-size:.8rem;color:var(--ink);outline:none;
  transition:border-color var(--dur),background var(--dur)}
.fi:focus{border-color:var(--am);background:var(--paper)}
textarea.fi{min-height:220px;resize:vertical;font-family:var(--ff-m);
  font-size:.67rem;line-height:1.65}
.fsec{margin-bottom:.95rem}
.frow2{display:grid;grid-template-columns:1fr 1fr;gap:.85rem;margin-bottom:.95rem}
.chk{display:flex;align-items:center;gap:.45rem;font-family:var(--ff-m);
  font-size:.7rem;color:var(--ink3);cursor:pointer}

/* Progress bar */
.progress-wrap{margin-top:1.25rem;display:none}
.prog-bar-bg{background:var(--p3);border-radius:4px;height:6px;
  overflow:hidden;margin-bottom:.6rem;border:1px solid var(--p4)}
.prog-bar-fill{height:100%;background:linear-gradient(90deg,var(--am),var(--gm));
  border-radius:4px;transition:width .5s var(--ease);width:0%}
.prog-stage{font-family:var(--ff-m);font-size:.62rem;color:var(--ink3);
  margin-bottom:.35rem;display:flex;align-items:center;gap:.4rem}
.prog-stage .ps-dot{width:6px;height:6px;border-radius:50%;
  background:var(--am);animation:ld-pulse 1s ease-in-out infinite}

/* Log console */
.log-con{background:#0f172a;border-radius:var(--r2);padding:.9rem 1.1rem;
  margin-top:.75rem;font-family:var(--ff-m);font-size:.66rem;color:#94a3b8;
  line-height:1.75;max-height:260px;overflow-y:auto;display:none}
.li{color:#7dd3fc}.lok{color:#86efac}.lwn{color:#fde68a}
.ler{color:#f87171}.ldn{color:#4ade80;font-weight:500}

/* Info card */
.info-card{background:var(--p2);border:1px solid var(--p4);border-radius:var(--r2);
  padding:1.3rem 1.4rem}
.ic-h{font-family:var(--ff-h);font-size:.98rem;font-style:italic;color:var(--ink);
  margin-bottom:.65rem;display:flex;align-items:center;gap:.5rem}
.steps-list{display:flex;flex-direction:column;gap:.5rem}
.step-it{display:flex;align-items:flex-start;gap:.6rem;font-size:.83rem;
  color:var(--ink3);line-height:1.5}
.step-it strong{color:var(--ink)}
.sn{font-family:var(--ff-m);font-size:.58rem;background:var(--ink);color:#fff;
  border-radius:3px;padding:.1rem .38rem;flex-shrink:0;margin-top:.18rem}
.also-note{margin-top:1rem;padding:.7rem .85rem;background:var(--al);
  border-radius:var(--r);border:1px solid var(--am);font-size:.82rem;
  color:var(--accent);line-height:1.55}
.also-note strong{font-weight:600}

/* ── OVERLAY (dashboard) ────────────────────────────────────────── */
.overlay{display:none;position:fixed;inset:0;z-index:300;flex-direction:column;
  background:var(--paper)}
.overlay.on{display:flex}
.ov-enter{animation:ovIn .28s var(--ease) both}
.ov-exit{animation:ovOut .2s var(--ease) both}
@keyframes ovIn{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:none}}
@keyframes ovOut{from{opacity:1;transform:none}to{opacity:0;transform:translateY(8px)}}
.ov-bar{background:var(--ink);height:48px;display:flex;align-items:center;
  padding:0 1.5rem;gap:1rem;flex-shrink:0;
  border-bottom:1px solid rgba(255,255,255,.06)}
.ov-back{font-family:var(--ff-m);font-size:.62rem;padding:.28rem .75rem;
  background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);
  border-radius:5px;color:rgba(255,255,255,.55);cursor:pointer;
  transition:all var(--dur);display:flex;align-items:center;gap:.35rem}
.ov-back:hover{background:rgba(255,255,255,.15);color:#fff}
.ov-title{font-family:var(--ff-h);font-size:1rem;font-style:italic;color:#fff;
  flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ov-dl{font-family:var(--ff-m);font-size:.62rem;padding:.28rem .75rem;
  background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.1);
  border-radius:5px;color:rgba(255,255,255,.55);cursor:pointer;
  transition:all var(--dur);display:flex;align-items:center;gap:.35rem}
.ov-dl:hover{background:rgba(255,255,255,.15);color:#fff}
.ov-iframe{flex:1;border:none;width:100%}

/* Run button row */
.run-row{display:flex;gap:.65rem;align-items:center;flex-wrap:wrap;padding-top:.5rem}
.fst{font-family:var(--ff-m);font-size:.64rem;color:var(--ink3)}

/* Kbd hint */
.kbd{font-family:var(--ff-m);font-size:.6rem;padding:.1rem .35rem;
  background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);
  border-radius:3px;color:rgba(255,255,255,.35)}
/* ── Concordance Visualisation Overlay ─────────────────────────────────────── */
#viz-ov{position:fixed;inset:0;z-index:1200;background:rgba(15,23,42,.72);
  backdrop-filter:blur(6px);display:none;align-items:center;justify-content:center;
  padding:16px}
#viz-ov.on{display:flex;animation:fadeIn .2s ease both}
.viz-panel{background:var(--bg0);border-radius:16px;box-shadow:0 24px 64px rgba(0,0,0,.28);
  width:100%;max-width:880px;max-height:90vh;overflow-y:auto;
  display:flex;flex-direction:column}
.viz-panel-head{display:flex;align-items:center;justify-content:space-between;
  padding:16px 20px 12px;border-bottom:1px solid var(--ink3);flex-shrink:0}
.viz-panel-title{font-weight:700;font-size:1rem;color:var(--ink1)}
.viz-panel-actions{display:flex;gap:8px}
.viz-btn-full{font-family:var(--ff-s);font-size:.76rem;font-weight:600;
  padding:7px 16px;border-radius:8px;background:var(--accent);color:#fff;
  border:none;cursor:pointer;transition:opacity .15s}
.viz-btn-full:hover{opacity:.85}
.viz-btn-close{font-family:var(--ff-s);font-size:.76rem;padding:7px 14px;
  border-radius:8px;border:1px solid var(--ink3);background:transparent;
  color:var(--ink2);cursor:pointer;transition:all .15s}
.viz-btn-close:hover{background:var(--bg1);color:var(--ink1)}
.viz-panel-body{padding:20px;flex:1}
/* Summary row */
.viz-summ{display:grid;grid-template-columns:auto 1fr;gap:20px;
  background:var(--bg1);border-radius:12px;padding:16px;margin-bottom:20px}
.viz-summ-kpis{display:flex;flex-direction:column;gap:10px;min-width:140px}
.viz-kpi{display:flex;flex-direction:column;align-items:center;
  background:var(--bg0);border-radius:8px;padding:10px 16px}
.viz-kpi-val{font-size:1.5rem;font-weight:800;line-height:1;color:var(--ink1)}
.viz-kpi-lbl{font-size:.64rem;color:var(--ink2);margin-top:3px;text-align:center}
.viz-summ-bars{flex:1;display:flex;flex-direction:column;justify-content:center;gap:6px}
.viz-bar-row{display:flex;align-items:center;gap:8px}
.viz-bar-lbl{font-size:.72rem;color:var(--ink2);width:130px;flex-shrink:0;text-align:right}
.viz-bar-track{flex:1;height:10px;background:var(--bg2);border-radius:5px;overflow:hidden}
.viz-bar-fill{height:100%;border-radius:5px;transition:width .6s cubic-bezier(.25,.46,.45,.94)}
.viz-bar-n{font-size:.68rem;color:var(--ink2);min-width:28px;text-align:right}
/* Single-prediction banner */
.viz-single{border-radius:12px;padding:18px 20px;margin-bottom:20px;
  border-left-width:5px;border-left-style:solid;background:var(--bg1)}
.viz-single-tier{font-size:.95rem;font-weight:700;margin-bottom:6px}
.viz-single-msg{font-size:.83rem;line-height:1.55;color:var(--ink1);margin:0 0 12px}
.viz-single-nums{display:flex;gap:16px;flex-wrap:wrap;font-size:.78rem;font-weight:600}
/* Prediction cards grid */
.viz-cards-head{font-size:.72rem;font-weight:600;letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink2);margin-bottom:10px}
.viz-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.viz-card{background:var(--bg1);border:1px solid var(--ink3);border-radius:10px;
  padding:14px;transition:box-shadow .15s}
.viz-card:hover{box-shadow:0 4px 18px rgba(0,0,0,.1)}
.viz-card-top{display:flex;justify-content:space-between;align-items:flex-start;
  margin-bottom:10px;gap:6px}
.viz-card-info .viz-entity{font-size:.9rem;font-weight:700;color:var(--ink1);display:block}
.viz-card-info .viz-ctx{font-size:.7rem;color:var(--ink2);display:block;margin-top:1px}
.viz-card-info .viz-dir{font-size:.66rem;color:var(--ink3);display:block;margin-top:1px}
.viz-tbadge{font-size:.64rem;padding:3px 8px;border-radius:10px;font-weight:700;
  white-space:nowrap;flex-shrink:0}
.viz-card-body{display:grid;grid-template-columns:auto 1fr;gap:10px;align-items:start}
.viz-donut-wrap{display:flex;flex-direction:column;align-items:center;gap:4px}
.viz-legend{display:flex;flex-direction:column;gap:3px}
.viz-leg-row{display:flex;align-items:center;gap:5px;font-size:.64rem;color:var(--ink2)}
.viz-leg-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.viz-card-interp{font-size:.7rem;color:var(--ink2);line-height:1.45;
  margin:8px 0 0;grid-column:1/-1;padding-top:8px;border-top:1px solid var(--ink3)}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="topbar">
  <div class="tb-logo">PaperTrail<span class="dot">·</span><span class="tb-sub">Literature Concordance</span></div>
  <div class="tb-right">
    <span class="kbd">Esc</span><span style="font-family:var(--ff-m);font-size:.58rem;color:rgba(255,255,255,.22)">close dashboard</span>
    <div class="mode-toggle">
      <button class="mt-btn on" id="tab-demo" onclick="setTab('demo')">
        <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor"><path d="M1.5 1l7 4-7 4V1z"/></svg>
        Demo
      </button>
      <button class="mt-btn" id="tab-full" onclick="setTab('full')">
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="5" cy="5" r="3.5"/><path d="M5 1.5v7M1.5 5h7" opacity=".5"/></svg>
        Full pipeline
      </button>
    </div>
  </div>
</div>

<!-- ═════ DEMO PANEL ════════════════════════════════════════════ -->
<div class="panel on" id="panel-demo">

  <div class="demo-hero">
    <div class="dh-eye">PaperTrail · literature concordance · automated</div>
    <h1 class="dh-h">Does the literature agree<br/>with your <em>predictions?</em></h1>
    <p class="dh-p">Choose an example case, review or edit the predictions in the table, then hit Run. The full concordance dashboard opens instantly — with sentence-level evidence for every paper.</p>
  </div>

  <!-- Step 1: picker -->
  <div id="picker-sec">
    <div class="step-row">
      <div class="step-num active">1</div>
      <div class="step-lbl">Choose a case</div>
      <div class="step-div"></div>
      <div class="step-hint">4 worked examples across different search paradigms</div>
    </div>
    <div class="tiles">
      {% for c in cases %}
      <div class="tile" id="tile-{{ c.id }}"
           style="--tc:{{ c.accent }};--tl:{{ c.accent_l }}"
           onclick="selectCase('{{ c.id }}')" tabindex="0"
           onkeydown="if(event.key==='Enter')selectCase('{{ c.id }}')">
        <div class="tile-accent"></div>
        <span class="tile-count">{{ c.rows|length }} predictions</span>
        <div class="tile-type" style="color:{{ c.accent }}">{{ c.type_label }}</div>
        <span class="tile-emoji">{{ c.emoji }}</span>
        <div class="tile-name">{{ c.label }}</div>
        <div class="tile-sub">{{ c.subtitle }}</div>
      </div>
      {% endfor %}
    </div>
  </div>

  <!-- Step 2: case detail (one at a time) -->
  {% for c in cases %}
  <div class="case-detail" id="detail-{{ c.id }}">

    <div class="cd-nav">
      <button class="cd-back" onclick="backToPicker()">
        <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M7 2L3.5 5.5 7 9"/></svg>
        All cases
      </button>
      <div class="cd-crumb">
        <span style="color:var(--ink4)">Cases</span>
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M4 2.5l3 3-3 3"/></svg>
        <span>{{ c.label }}</span>
      </div>
      <div class="cd-nav-r">
        <button class="cd-nav-arrow" id="prev-{{ c.id }}" onclick="prevCase('{{ c.id }}')" title="Previous case (←)">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M8 2.5L4 6l4 3.5"/></svg>
        </button>
        <button class="cd-nav-arrow" id="next-{{ c.id }}" onclick="nextCase('{{ c.id }}')" title="Next case (→)">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 2.5L8 6 4 9.5"/></svg>
        </button>
      </div>
    </div>

    <div class="step-row" style="margin-bottom:1rem">
      <div class="step-num done">1</div>
      <div class="step-lbl" style="color:var(--ink4)">Case selected</div>
      <div class="step-div"></div>
      <div class="step-num active">2</div>
      <div class="step-lbl">Review &amp; edit</div>
      <div class="step-div"></div>
      <div class="step-num">3</div>
      <div class="step-lbl">Run</div>
    </div>

    <div class="cd-header">
      <div class="cd-emoji">{{ c.emoji }}</div>
      <div class="cd-info">
        <div class="cd-name">{{ c.label }}</div>
        <div class="cd-sub">{{ c.subtitle }}</div>
        <span class="type-pill" style="--tc:{{ c.accent }}">{{ c.type_label }}</span>
      </div>
    </div>

    <div class="blurb" style="--tc:{{ c.accent }}">
      {{ c.blurb }}
      <div class="type-desc">{{ c.type_desc }}</div>
    </div>

    <div class="step-row" style="margin-bottom:.55rem">
      <div class="step-num active">2</div>
      <div class="step-lbl">Prediction table</div>
      <div class="step-div"></div>
      <div class="step-hint">click any cell to edit</div>
    </div>
    <div class="edit-hint">
      <div class="eh-dot"></div>
      Every cell is editable — change gene names, disease, or direction. Add rows with +. Changes reflect live in the YAML view.
    </div>

    <div class="pred-wrap">
      <table class="ptbl" id="ptbl-{{ c.id }}">
        <thead><tr>
          <th>#</th><th>Gene / Entity</th><th>Aliases (comma-separated)</th>
          <th>Disease / Context</th><th>Cell type</th>
          <th>Direction</th><th>Note</th><th></th>
        </tr></thead>
        <tbody id="tbody-{{ c.id }}">
          {% for row in c.rows %}
          <tr data-case="{{ c.id }}" data-row="{{ loop.index0 }}">
            <td><span class="rn">{{ loop.index }}</span></td>
            <td><span class="ec entity" contenteditable="true"
              data-f="entity" data-c="{{ c.id }}" data-r="{{ loop.index0 }}">{{ row.entity }}</span></td>
            <td><span class="ec small" contenteditable="true" title="{{ row.aliases }}"
              data-f="aliases" data-c="{{ c.id }}" data-r="{{ loop.index0 }}">{{ row.aliases }}</span></td>
            <td><span class="ec small" contenteditable="true"
              data-f="disease" data-c="{{ c.id }}" data-r="{{ loop.index0 }}">{{ row.disease }}</span></td>
            <td><span class="ec small" contenteditable="true"
              data-f="cell_type" data-c="{{ c.id }}" data-r="{{ loop.index0 }}">{{ row.cell_type }}</span></td>
            <td>
              <select class="dir-sel" data-val="{{ row.direction }}"
                data-f="direction" data-c="{{ c.id }}" data-r="{{ loop.index0 }}"
                onchange="onDir(this)">
                <option value="up"           {% if row.direction=='up' %}selected{% endif %}>↑ up</option>
                <option value="down"         {% if row.direction=='down' %}selected{% endif %}>↓ down</option>
                <option value="preserved"    {% if row.direction=='preserved' %}selected{% endif %}>→ preserved</option>
                <option value="bidirectional"{% if row.direction=='bidirectional' %}selected{% endif %}>↕ bidirectional</option>
                <option value="absent"       {% if row.direction=='absent' %}selected{% endif %}>✕ absent</option>
              </select>
            </td>
            <td><span class="ec note" contenteditable="true" title="{{ row.note }}"
              data-f="note" data-c="{{ c.id }}" data-r="{{ loop.index0 }}">{{ row.note }}</span></td>
            <td><button class="row-del" onclick="delRow(this,'{{ c.id }}')" title="Remove row">×</button></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="action-bar">
      <button class="btn-run" id="runbtn-{{ c.id }}" onclick="runDemo('{{ c.id }}')">
        <svg width="11" height="11" viewBox="0 0 11 11" fill="currentColor"><path d="M2 1.5l7.5 4L2 9.5V1.5z"/></svg>
        Run demo
      </button>
      <button class="btn-add-row" onclick="addRow('{{ c.id }}')">
        <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M5.5 1.5v8M1.5 5.5h8"/></svg>
        Add row
      </button>
      <button class="btn-out" id="yamlbtn-{{ c.id }}" onclick="toggleYaml('{{ c.id }}')">View YAML</button>
      <button class="btn-out" onclick="resetCase('{{ c.id }}')">Reset</button>
      <div class="ld" id="ld-{{ c.id }}">
        <div class="ld-d"></div><div class="ld-d"></div><div class="ld-d"></div>
        <span>Building dashboard…</span>
      </div>
    </div>

    <div class="yaml-panel" id="yamlpanel-{{ c.id }}">
      <div class="yaml-bar">
        <span class="yaml-bar-l">predictions.yaml</span>
        <div class="yaml-bar-r">
          <button class="ybtn" onclick="copyYaml('{{ c.id }}')">Copy</button>
          <button class="ybtn" onclick="useInFull('{{ c.id }}')">Use in full pipeline →</button>
          <button class="ybtn" onclick="toggleYaml('{{ c.id }}')">✕</button>
        </div>
      </div>
      <pre class="yaml-pre" id="yamlpre-{{ c.id }}"></pre>
    </div>

  </div>
  {% endfor %}

</div><!-- /panel-demo -->


<!-- ═════ FULL PANEL ════════════════════════════════════════════ -->
<div class="panel" id="panel-full">

  <div class="full-hero">
    <div class="fh-h">Run the full PaperTrail pipeline</div>
    <div class="fh-p">Paste your <code>predictions.yaml</code>, enter your project name and NCBI credentials, and run. All 5 stages execute in the background with live progress tracking.</div>
  </div>

  <div class="full-layout">
    <div class="form-card">

      <!-- Project name -->
      <div class="proj-row">
        <label>Project name</label>
        <input class="proj-input" id="proj-name" type="text"
          placeholder="e.g. My scRNA-seq Study, RA Drug Targets, IBD GWAS…"
          title="Shown in the dashboard sidebar and title. Not displayed in demo mode."/>
      </div>

      <div class="fsec">
        <label class="fl">Predictions input <span style="color:var(--rm)">required</span></label>

        <!-- Mode toggle: YAML or Natural Language -->
        <div class="input-mode-toggle">
          <button class="imt-btn active" id="mode-yaml-btn" onclick="setInputMode('yaml')">
            <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" stroke-width="1.6" style="vertical-align:-1px;margin-right:4px"><rect x="1" y="1" width="9" height="9" rx="1.5"/><path d="M3 4h5M3 6h3"/></svg>
            predictions.yaml
          </button>
          <button class="imt-btn" id="mode-nl-btn" onclick="setInputMode('nl')">
            <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" stroke-width="1.6" style="vertical-align:-1px;margin-right:4px"><circle cx="5.5" cy="5.5" r="4.5"/><path d="M3.5 4.5c0-.8.7-1.5 2-1.5s2 .6 2 1.3c0 .8-.9 1.3-2 1.7v.5"/><circle cx="5.5" cy="8" r=".4" fill="currentColor"/></svg>
            Just Ask
          </button>
        </div>

        <!-- YAML input (default mode) -->
        <div class="yaml-panel-wrap" id="yaml-panel-wrap">
          <div class="yaml-upload-bar">
            <label class="upload-btn" title="Upload a predictions.yaml file">
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M6.5 9V3M3.5 5.5L6.5 3l3 2.5"/><rect x="1.5" y="9.5" width="10" height="2" rx=".5"/></svg>
              Upload .yaml
              <input type="file" id="yaml-file-input" accept=".yaml,.yml,.txt" style="display:none" onchange="handleYamlUpload(event)"/>
            </label>
            <span class="yaml-upload-hint" id="yaml-upload-hint"></span>
          </div>
          <textarea class="fi" id="full-yaml"
            placeholder="predictions:&#10;  - id: IRF5_SLE_up&#10;    entity: IRF5&#10;    entity_type: gene&#10;    aliases: [IRF5, &quot;interferon regulatory factor 5&quot;]&#10;    disease_context: SLE&#10;    disease_synonyms: [&quot;systemic lupus erythematosus&quot;, &quot;lupus&quot;]&#10;    cell_type: pDC&#10;    cell_type_synonyms: [&quot;plasmacytoid dendritic cell&quot;]&#10;    direction: up&#10;    novelty: known&#10;    note: &quot;IRF5 gain-of-function drives type I IFN in pDCs&quot;"></textarea>
        </div>

        <!-- Just Ask input -->
        <div class="nl-panel" id="nl-panel">
          <textarea class="nl-prompt-area" id="nl-prompt"
            placeholder="Describe your prediction in plain language. Examples:&#10;&#10;• SHMT2 is reduced in diabetic kidney disease in C_TAL cells (novel finding from our scRNA-seq)&#10;&#10;• IRF5 is elevated in SLE plasmacytoid dendritic cells driving type I IFN production&#10;&#10;• MTHFD2 and ALDH1L2 are both reduced in DKD cortical thick ascending limb cells&#10;&#10;• Check literature for PKM2 upregulation in diabetic nephropathy tubular cells and its role in glycolytic switch"
          ></textarea>
          <div id="nl-llm-status" style="font-size:.71rem;color:var(--ink2);margin-bottom:8px;display:none">
            <span id="nl-llm-icon">⬤</span>
            <span id="nl-llm-label"></span>
            <span id="nl-setup-hint" style="color:var(--rm)"></span>
          </div>
          <div class="nl-actions">
            <button class="btn-run" id="nl-gen-btn" onclick="generateYamlFromPrompt()" style="padding:6px 16px;font-size:.74rem">
              <svg width="11" height="11" viewBox="0 0 11 11" fill="currentColor" style="margin-right:4px"><path d="M2 1.5l7.5 4L2 9.5V1.5z"/></svg>
              Generate YAML
            </button>
            <span id="nl-source-badge"></span>
          </div>
          <div id="nl-warn" class="nl-warn" style="display:none"></div>
          <div id="nl-preview-wrap" style="display:none;margin-top:12px">
            <label class="fl" style="margin-bottom:4px">
              Generated predictions.yaml — review and edit before running:
            </label>
            <textarea class="fi" id="nl-preview" style="min-height:180px;font-size:.78rem"
              onchange="syncNlPreviewToMain()"></textarea>
          </div>
        </div>
      </div>
      <div class="frow2">
        <div>
          <label class="fl">NCBI e-mail <span style="color:var(--rm)">required</span></label>
          <input class="fi" id="full-email" type="email" placeholder="your@email.com"/>
        </div>
        <div>
          <label class="fl">NCBI API key <em>— optional</em></label>
          <input class="fi" id="full-key" type="text" placeholder="optional — 10 req/s vs 3"/>
        </div>
      </div>

      <div class="fsec">
        <label class="chk"><input type="checkbox" id="full-skippmc" checked/>
          Skip PMC full-text enrichment (faster; abstract-only mode)</label>
      </div>

      <div class="run-row">
        <button class="btn-run" id="full-run-btn" onclick="startFull()">
          <svg width="11" height="11" viewBox="0 0 11 11" fill="currentColor"><path d="M2 1.5l7.5 4L2 9.5V1.5z"/></svg>
          Run pipeline
        </button>
        <button class="btn-out" id="load-example-btn" onclick="loadExample()">Load example YAML</button>
        <span class="fst" id="full-status"></span>
      </div>

      <div class="progress-wrap" id="prog-wrap">
        <div class="prog-bar-bg"><div class="prog-bar-fill" id="prog-fill"></div></div>
        <div class="prog-stage" id="prog-stage">
          <div class="ps-dot"></div><span id="prog-txt">Starting…</span>
        </div>
      </div>
      <div class="log-con" id="full-log"></div>
    </div>

    <div>
      <div class="info-card">
        <div class="ic-h">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="8" cy="8" r="6.5"/><path d="M8 7v4M8 5v.5"/></svg>
          How it works
        </div>
        <div class="steps-list">
          <div class="step-it"><span class="sn">1</span><div><strong>Alias expansion</strong> — mygene.info enriches gene synonyms from NCBI Gene</div></div>
          <div class="step-it"><span class="sn">2</span><div><strong>Query builder</strong> — multi-strategy PubMed queries: specific, broad, cell-type, human biopsy</div></div>
          <div class="step-it"><span class="sn">3</span><div><strong>PubMed retrieval</strong> — ESearch + EFetch + optional PMC full-text stream-extract</div></div>
          <div class="step-it"><span class="sn">4</span><div><strong>Evidence extraction</strong> — direction with negation + hedge detection, quality weighting</div></div>
          <div class="step-it"><span class="sn">5</span><div><strong>Concordance scoring</strong> — Wilson CI, LOO sensitivity, tier classification</div></div>
        </div>
        <div class="also-note">
          <strong>Tip:</strong> Click <em>View YAML</em> then <em>Use in full pipeline →</em> in any demo case to pre-fill this form with that case's YAML.
        </div>
      </div>
    </div>
  </div>

</div><!-- /panel-full -->


<!-- VISUALISATION OVERLAY (quick concordance summary + charts) -->
<div id="viz-ov">
  <div class="viz-panel">
    <div class="viz-panel-head">
      <span class="viz-panel-title" id="viz-ov-title">Concordance Results</span>
      <div class="viz-panel-actions">
        <button class="viz-btn-full" id="viz-btn-full" onclick="vizOpenFull()">
          View Full Dashboard →
        </button>
        <button class="viz-btn-close" onclick="closeVizOverlay()">Close</button>
      </div>
    </div>
    <div class="viz-panel-body" id="viz-panel-body">
      <!-- Filled dynamically by _showVizOverlay() -->
    </div>
  </div>
</div>

<!-- DASHBOARD OVERLAY -->
<div class="overlay" id="overlay">
  <div class="ov-bar">
    <button class="ov-back" onclick="closeOverlay()">
      <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M7 2L3.5 5.5 7 9"/></svg>
      Back
    </button>
    <div class="ov-title" id="ov-title">Concordance Dashboard</div>
    <button class="ov-dl" id="ov-dl" onclick="downloadDash()">
      <svg width="11" height="11" viewBox="0 0 11 11" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5.5 1.5v6M2.5 8.5l3 1.5 3-1.5M1.5 10h8"/></svg>
      Download HTML
    </button>
  </div>
  <iframe class="ov-iframe" id="ov-iframe" src="about:blank" title="Dashboard"></iframe>
</div>


<script>
// ══ State ════════════════════════════════════════════════════════
const CASE_ORDER = [{% for c in cases %}'{{ c.id }}'{% if not loop.last %},{% endif %}{% endfor %}];
const DEFAULTS   = {};
const STATE      = {};
{% for c in cases %}
DEFAULTS["{{ c.id }}"] = {{ c.rows | tojson }};
STATE["{{ c.id }}"]    = { label: {{ c.label|tojson }}, demo_key: {{ c.demo_key|tojson }},
                            accent: {{ c.accent|tojson }}, rows: JSON.parse(JSON.stringify({{ c.rows|tojson }})) };
{% endfor %}

// Keep STATE in sync
document.querySelectorAll('[contenteditable]').forEach(el => {
  el.addEventListener('input', () => {
    STATE[el.dataset.c].rows[+el.dataset.r][el.dataset.f] = el.textContent.trim();
    if (_yamlOpen === el.dataset.c) _refreshYaml(el.dataset.c);
  });
});
function onDir(sel){
  STATE[sel.dataset.c].rows[+sel.dataset.r][sel.dataset.f] = sel.value;
  sel.dataset.val = sel.value;
  if(_yamlOpen === sel.dataset.c) _refreshYaml(sel.dataset.c);
}

// ══ Tab / mode switch ════════════════════════════════════════════
function setTab(t){
  ['demo','full'].forEach(x=>{
    document.getElementById('panel-'+x).classList.toggle('on',x===t);
    document.getElementById('tab-'+x).classList.toggle('on',x===t);
  });
}

// ══ Case picker ══════════════════════════════════════════════════
let _active = null;
function selectCase(id){
  document.getElementById('picker-sec').style.display='none';
  document.querySelectorAll('.case-detail').forEach(d=>d.classList.remove('on'));
  document.getElementById('detail-'+id).classList.add('on');
  _active=id;
  // arrow buttons
  const idx=CASE_ORDER.indexOf(id);
  document.getElementById('prev-'+id).disabled = idx===0;
  document.getElementById('next-'+id).disabled = idx===CASE_ORDER.length-1;
}
function backToPicker(){
  document.querySelectorAll('.case-detail').forEach(d=>d.classList.remove('on'));
  document.getElementById('picker-sec').style.display='block';
  _active=null; if(_yamlOpen){_hideYaml(_yamlOpen); _yamlOpen=null;}
}
function prevCase(id){
  const i=CASE_ORDER.indexOf(id); if(i>0) selectCase(CASE_ORDER[i-1]);
}
function nextCase(id){
  const i=CASE_ORDER.indexOf(id); if(i<CASE_ORDER.length-1) selectCase(CASE_ORDER[i+1]);
}

// Keyboard: left/right between cases, Esc closes overlay
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){ closeOverlay(); return; }
  if(e.key==='ArrowLeft'  && _active && !document.querySelector('[contenteditable]:focus')) prevCase(_active);
  if(e.key==='ArrowRight' && _active && !document.querySelector('[contenteditable]:focus')) nextCase(_active);
});

// ══ Add / delete rows ════════════════════════════════════════════
function addRow(cid){
  const rows = STATE[cid].rows;
  const newRow={entity:'',aliases:'',disease:rows[0]?.disease||'',
                cell_type:'',direction:'up',note:''};
  rows.push(newRow);
  _rebuildTable(cid);
  if(_yamlOpen===cid) _refreshYaml(cid);
}
function delRow(btn, cid){
  const tr = btn.closest('tr');
  const idx = +tr.dataset.row;
  STATE[cid].rows.splice(idx,1);
  _rebuildTable(cid);
  if(_yamlOpen===cid) _refreshYaml(cid);
}
function resetCase(cid){
  STATE[cid].rows = JSON.parse(JSON.stringify(DEFAULTS[cid]));
  _rebuildTable(cid);
  if(_yamlOpen===cid) _refreshYaml(cid);
}
const _DLBL={up:'↑ up',down:'↓ down',preserved:'→ preserved',bidirectional:'↕ bidirectional',absent:'✕ absent'};
function _rebuildTable(cid){
  const tbody = document.getElementById('tbody-'+cid);
  tbody.innerHTML = STATE[cid].rows.map((row,i)=>`
    <tr data-case="${cid}" data-row="${i}">
      <td><span class="rn">${i+1}</span></td>
      <td><span class="ec entity" contenteditable="true" data-f="entity" data-c="${cid}" data-r="${i}">${esc(row.entity||'')}</span></td>
      <td><span class="ec small"  contenteditable="true" data-f="aliases" data-c="${cid}" data-r="${i}" title="${esc(row.aliases||'')}">${esc(row.aliases||'')}</span></td>
      <td><span class="ec small"  contenteditable="true" data-f="disease" data-c="${cid}" data-r="${i}">${esc(row.disease||'')}</span></td>
      <td><span class="ec small"  contenteditable="true" data-f="cell_type" data-c="${cid}" data-r="${i}">${esc(row.cell_type||'')}</span></td>
      <td><select class="dir-sel" data-val="${row.direction||'up'}" data-f="direction" data-c="${cid}" data-r="${i}" onchange="onDir(this)">
        ${['up','down','preserved','bidirectional','absent'].map(d=>
          `<option value="${d}"${d===row.direction?' selected':''}>${_DLBL[d]||d}</option>`
        ).join('')}
      </select></td>
      <td><span class="ec note" contenteditable="true" data-f="note" data-c="${cid}" data-r="${i}" title="${esc(row.note||'')}">${esc(row.note||'')}</span></td>
      <td><button class="row-del" onclick="delRow(this,'${cid}')" title="Remove">×</button></td>
    </tr>`).join('');
  // re-attach listeners
  tbody.querySelectorAll('[contenteditable]').forEach(el=>{
    el.addEventListener('input',()=>{
      STATE[el.dataset.c].rows[+el.dataset.r][el.dataset.f]=el.textContent.trim();
      if(_yamlOpen===el.dataset.c) _refreshYaml(el.dataset.c);
    });
  });
}

// ══ CONCORDANCE VISUALISATION ══════════════════════════════════════

// Tier metadata: colour, short label, symbol
const TIER_CFG = {
  STRONG:          {c:'#16a34a', l:'Strong support',      s:'✓✓'},
  MODERATE:        {c:'#2563eb', l:'Moderate support',    s:'✓' },
  WEAK_SUPPORT:    {c:'#d97706', l:'Weak support',        s:'~' },
  WEAK_DISCORDANT: {c:'#ea580c', l:'Weak against',        s:'±' },
  DISCORDANT:      {c:'#dc2626', l:'Discordant',          s:'✗' },
  NO_DIRECTIONAL:  {c:'#7c3aed', l:'Association found',   s:'◎' },
  NO_INFORMATIVE:  {c:'#64748b', l:'Minimal evidence',    s:'○' },
  NONE:            {c:'#374151', l:'No evidence',         s:'—' },
};

// SVG arc from start angle, sweeping by `sweep` radians (clockwise from top)
function _svgArc(cx,cy,r,start,sweep){
  if(sweep>=Math.PI*2-0.001)
    return `M ${cx} ${cy-r} A ${r} ${r} 0 1 1 ${cx-0.001} ${cy-r}`;
  const x1=cx+r*Math.cos(start-Math.PI/2),y1=cy+r*Math.sin(start-Math.PI/2);
  const x2=cx+r*Math.cos(start+sweep-Math.PI/2),y2=cy+r*Math.sin(start+sweep-Math.PI/2);
  return `M ${x1} ${y1} A ${r} ${r} 0 ${sweep>Math.PI?1:0} 1 ${x2} ${y2}`;
}

// Build a 4-segment SVG donut chart
// nc=concordant(green), nd=discordant(red), nn=neutral/descriptive(gray), size=px
function _buildDonut(nc,nd,nn,size){
  size=size||80;
  const sw=Math.max(7,size*0.14), r=(size-sw)/2-1;
  const cx=size/2, cy=size/2, total=nc+nd+nn;
  let svg=`<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">`;
  svg+=`<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="#f1f5f9" stroke-width="${sw}"/>`;
  if(total>0){
    const segs=[{n:nc,c:'#16a34a'},{n:nd,c:'#ef4444'},{n:nn,c:'#e2e8f0'}];
    let a=0;
    segs.forEach(seg=>{
      if(seg.n<=0)return;
      const sw2=Math.max(seg.n/total*Math.PI*2,0.001);
      svg+=`<path d="${_svgArc(cx,cy,r,a,sw2)}" fill="none" stroke="${seg.c}" stroke-width="${sw}"/>`;
      a+=sw2;
    });
    // Centre label
    const hasDir=nc+nd>0;
    const pct=hasDir?Math.round(nc/(nc+nd)*100):null;
    const topLabel=pct!==null?`${pct}%`:String(total);
    const botLabel=pct!==null?'agree':'papers';
    svg+=`<text x="${cx}" y="${cy-size*0.04}" text-anchor="middle"
      dominant-baseline="middle" font-size="${size*0.2}" font-weight="700" fill="#0f172a">${topLabel}</text>`;
    svg+=`<text x="${cx}" y="${cy+size*0.18}" text-anchor="middle"
      dominant-baseline="middle" font-size="${size*0.13}" fill="#64748b">${botLabel}</text>`;
  }else{
    svg+=`<text x="${cx}" y="${cy}" text-anchor="middle" dominant-baseline="middle"
      font-size="${size*0.16}" fill="#9ca3af">no data</text>`;
  }
  return svg+'</svg>';
}

// Generate human-readable interpretation for a prediction
function _interp(pred){
  const {tier,n_concordant:nc=0,n_opposite:nd=0,n_relevant:nr=0,n_informative:ni=0,direction}=pred;
  const cfg=TIER_CFG[tier]||TIER_CFG.NONE;
  if(tier==='STRONG')
    return `${nc} of ${ni} papers with directional signals agree with this prediction. Strong literature support.`;
  if(tier==='MODERATE')
    return `${nc} of ${ni} directional papers agree. Moderate concordance across ${nr} relevant papers.`;
  if(tier==='WEAK_SUPPORT')
    return `Weak but positive signal: ${nc} concordant vs ${nd} discordant across ${ni} directional papers.`;
  if(tier==='DISCORDANT'||tier==='WEAK_DISCORDANT')
    return `${nd} papers oppose this prediction vs ${nc} concordant. Review the discordant evidence carefully.`;
  if(tier==='NO_DIRECTIONAL'&&nr>0)
    return `${nr} relevant paper${nr!==1?'s':''} found. The literature confirms this entity is active in this context `+
           `but reports it descriptively (no quantified up/down signal). `+
           `This is typical for metabolic reprogramming, GWAS associations, and eQTLs — `+
           `the association is validated even if the direction cannot be scored automatically.`;
  if(nr===0)
    return 'No papers retrieved for this prediction. Try broader disease synonyms or aliases.';
  return `${nr} relevant papers found. Insufficient directional signals to score.`;
}

// Build the per-prediction card HTML
function _buildVizCard(pred){
  const {tier,n_concordant:nc=0,n_opposite:nd=0,n_relevant:nr=0,n_informative:ni=0}=pred;
  const nn=Math.max(0,nr-nc-nd);
  const cfg=TIER_CFG[tier]||TIER_CFG.NONE;
  const parts=pred.label.split('(');
  const entity=parts[0].trim();
  const ctx=parts[1]?parts[1].replace(/\)$/,'').trim():'';
  const donut=_buildDonut(nc,nd,nn,72);
  return `<div class="viz-card" data-tier="${tier}">
    <div class="viz-card-top">
      <div class="viz-card-info">
        <span class="viz-entity">${entity}</span>
        ${ctx?`<span class="viz-ctx">${ctx}</span>`:''}
        <span class="viz-dir">Expected: ${pred.direction||'?'}</span>
      </div>
      <span class="viz-tbadge"
        style="background:${cfg.c}1a;color:${cfg.c};border:1px solid ${cfg.c}33">
        ${cfg.s}&nbsp;${cfg.l}
      </span>
    </div>
    <div class="viz-card-body">
      <div class="viz-donut-wrap">
        ${donut}
        <div class="viz-legend">
          <div class="viz-leg-row"><span class="viz-leg-dot" style="background:#16a34a"></span>${nc} agree</div>
          <div class="viz-leg-row"><span class="viz-leg-dot" style="background:#ef4444"></span>${nd} oppose</div>
          <div class="viz-leg-row"><span class="viz-leg-dot" style="background:#e2e8f0;border:1px solid #cbd5e1"></span>${nn} descriptive</div>
        </div>
      </div>
      <p class="viz-card-interp">${_interp(pred)}</p>
    </div>
  </div>`;
}

// Build the summary section (tier distribution bar chart + KPIs)
function _buildSummarySection(preds){
  const tierOrder=['STRONG','MODERATE','WEAK_SUPPORT','WEAK_DISCORDANT','DISCORDANT',
                   'NO_DIRECTIONAL','NO_INFORMATIVE','NONE'];
  const counts={}; preds.forEach(p=>counts[p.tier]=(counts[p.tier]||0)+1);
  const total=preds.length;
  const nConc=(counts.STRONG||0)+(counts.MODERATE||0)+(counts.WEAK_SUPPORT||0);
  const nDisc=(counts.DISCORDANT||0)+(counts.WEAK_DISCORDANT||0);
  const nAssoc=total-nConc-nDisc;

  let bars='';
  tierOrder.forEach(tier=>{
    const n=counts[tier]||0; if(!n)return;
    const cfg=TIER_CFG[tier]||TIER_CFG.NONE;
    const pct=Math.round(n/total*100);
    bars+=`<div class="viz-bar-row">
      <span class="viz-bar-lbl">${cfg.s} ${cfg.l}</span>
      <div class="viz-bar-track"><div class="viz-bar-fill"
        style="width:${Math.max(4,pct)}%;background:${cfg.c}"></div></div>
      <span class="viz-bar-n">${n}</span>
    </div>`;
  });

  return `<div class="viz-summ">
    <div class="viz-summ-kpis">
      <div class="viz-kpi">
        <span class="viz-kpi-val" style="color:#16a34a">${Math.round(nConc/total*100)}%</span>
        <span class="viz-kpi-lbl">concordant</span>
      </div>
      <div class="viz-kpi">
        <span class="viz-kpi-val" style="color:#ef4444">${Math.round(nDisc/total*100)}%</span>
        <span class="viz-kpi-lbl">discordant</span>
      </div>
      <div class="viz-kpi">
        <span class="viz-kpi-val" style="color:#7c3aed">${Math.round(nAssoc/total*100)}%</span>
        <span class="viz-kpi-lbl">association<br>only</span>
      </div>
      <div class="viz-kpi">
        <span class="viz-kpi-val">${total}</span>
        <span class="viz-kpi-lbl">predictions</span>
      </div>
    </div>
    <div class="viz-summ-bars">${bars}</div>
  </div>`;
}

// Special single-prediction banner
function _buildSingleBanner(pred){
  const {tier,n_concordant:nc=0,n_opposite:nd=0,n_relevant:nr=0,n_informative:ni=0}=pred;
  const nn=Math.max(0,nr-nc-nd);
  const cfg=TIER_CFG[tier]||TIER_CFG.NONE;
  const donut=_buildDonut(nc,nd,nn,100);
  return `<div class="viz-single" style="border-color:${cfg.c}">
    <div class="viz-single-tier" style="color:${cfg.c}">${cfg.s} ${cfg.l}</div>
    <div style="display:flex;gap:20px;align-items:flex-start">
      ${donut}
      <div>
        <p class="viz-single-msg">${_interp(pred)}</p>
        <div class="viz-single-nums">
          <span style="color:#16a34a">↑ ${nc} concordant</span>
          <span style="color:#ef4444">↓ ${nd} discordant</span>
          <span style="color:#64748b">◯ ${nn} descriptive</span>
          <span style="font-weight:700">∑ ${nr} relevant</span>
        </div>
      </div>
    </div>
  </div>`;
}

// Store the callback to open the full dashboard (set per-invocation)
let _vizFullCallback=null;

// Show the visualisation overlay
function _showVizOverlay(data, title, onViewFull){
  _vizFullCallback=onViewFull||null;
  document.getElementById('viz-ov-title').textContent=title||'Results';
  const preds=(data.per_pred||[]);
  let body='';
  if(preds.length===1){
    body+=_buildSingleBanner(preds[0]);
  } else if(preds.length>1){
    body+=_buildSummarySection(preds);
    body+=`<div class="viz-cards-head">Per-prediction breakdown</div>`;
    body+=`<div class="viz-grid">`;
    preds.forEach(p=>{ body+=_buildVizCard(p); });
    body+=`</div>`;
  } else {
    body='<p style="color:var(--ink2);font-size:.85rem">No prediction data available.</p>';
  }
  document.getElementById('viz-panel-body').innerHTML=body;
  document.getElementById('viz-ov').classList.add('on');
  document.body.style.overflow='hidden';
}
function closeVizOverlay(){
  document.getElementById('viz-ov').classList.remove('on');
  document.body.style.overflow='';
}
function vizOpenFull(){
  closeVizOverlay();
  if(_vizFullCallback) _vizFullCallback();
}

// ══ Run demo ═════════════════════════════════════════════════════
function runDemo(id){
  const btn=document.getElementById('runbtn-'+id);
  const ld =document.getElementById('ld-'+id);
  btn.disabled=true; ld.style.display='flex';
  // Detect if every flippable direction has been reversed from the default
  const defs=DEFAULTS[id]||[]; const rows=STATE[id].rows;
  const FLIPMAP={up:'down',down:'up',preserved:'absent',absent:'preserved'};
  const allFlipped=rows.length>0 && rows.every((r,i)=>{
    const orig=(defs[i]||{}).direction||'up';
    if(orig==='associated'||orig==='bidirectional') return true; // not flippable
    return r.direction===FLIPMAP[orig];
  });
  const demoKey = allFlipped ? STATE[id].demo_key+'_reversed' : STATE[id].demo_key;
  const editedRows = allFlipped ? [] : rows;
  const titleSuffix = allFlipped ? ' — Reversed' : '';
  fetch('/api/demo',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({demo_key:demoKey, edited_rows:editedRows})})
  .then(r=>r.text())
  .then(html=>{btn.disabled=false;ld.style.display='none';openOverlay(html,STATE[id].label+titleSuffix);})
  .catch(e=>{btn.disabled=false;ld.style.display='none';alert('Error: '+e);});
}

// ══ Dashboard overlay ════════════════════════════════════════════
let _ovHtml='';
function openOverlay(html, title){
  _ovHtml=html;
  document.getElementById('ov-title').textContent=title+' · Concordance Dashboard';
  const ov=document.getElementById('overlay');
  const iframe=document.getElementById('ov-iframe');
  iframe.src=URL.createObjectURL(new Blob([html],{type:'text/html'}));
  ov.classList.add('on');
  ov.querySelector('.ov-iframe').classList.remove('ov-exit');
  ov.querySelector('.ov-iframe').classList.add('ov-enter');
  document.body.style.overflow='hidden';
}
function closeOverlay(){
  const ov=document.getElementById('overlay');
  if(!ov.classList.contains('on')) return;
  ov.classList.remove('on');
  document.getElementById('ov-iframe').src='about:blank';
  document.body.style.overflow='';
}
function downloadDash(){
  const a=document.createElement('a');
  const blob=new Blob([_ovHtml],{type:'text/html'});
  a.href=URL.createObjectURL(blob);
  a.download=`litrev-dashboard-${Date.now()}.html`;
  a.click();
}

// ══ YAML panel ═══════════════════════════════════════════════════
let _yamlOpen=null;
function toggleYaml(id){
  const panel=document.getElementById('yamlpanel-'+id);
  const btn  =document.getElementById('yamlbtn-'+id);
  if(_yamlOpen&&_yamlOpen!==id) _hideYaml(_yamlOpen);
  if(panel.classList.contains('on')){_hideYaml(id);}
  else{
    _refreshYaml(id);
    panel.classList.add('on');
    btn.classList.add('active'); btn.textContent='Hide YAML';
    _yamlOpen=id;
  }
}
function _hideYaml(id){
  document.getElementById('yamlpanel-'+id).classList.remove('on');
  const btn=document.getElementById('yamlbtn-'+id);
  btn.classList.remove('active'); btn.textContent='View YAML';
  if(_yamlOpen===id) _yamlOpen=null;
}
function _refreshYaml(id){
  fetch('/api/yaml',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({rows:STATE[id].rows,label:STATE[id].label})})
  .then(r=>r.json()).then(d=>{document.getElementById('yamlpre-'+id).innerHTML=_hl(d.yaml);});
}
function copyYaml(id){
  const t=document.getElementById('yamlpre-'+id).textContent;
  navigator.clipboard.writeText(t).then(()=>{
    const b=document.querySelector(`#yamlpanel-${id} .ybtn`);
    const o=b.textContent; b.textContent='Copied!';
    setTimeout(()=>b.textContent=o,1400);
  });
}
function useInFull(id){
  const t=document.getElementById('yamlpre-'+id).textContent;
  document.getElementById('full-yaml').value=t;
  setTab('full');
  window.scrollTo({top:0,behavior:'smooth'});
}

// ══ YAML highlight ═══════════════════════════════════════════════
function _hl(yaml){
  return yaml.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .split('\n').map(line=>{
      if(/^\s*#/.test(line)) return`<span class="yc">${line}</span>`;
      const m=line.match(/^(\s*)(- )?(\w[\w\d_]*)(:\s*)(.*)$/);
      if(!m) return line;
      const[,ind,dash,key,colon,val]=m;
      const kk=`${ind}${dash||''}<span class="yk">${key}</span>${colon}`;
      if(!val) return kk;
      if(/^".*"$/.test(val.trim())||/^\[/.test(val.trim())) return kk+`<span class="ys">${val}</span>`;
      return kk+`<span class="yv">${val}</span>`;
    }).join('\n');
}

// ══ Full pipeline ════════════════════════════════════════════════
const STAGE_LABELS=['Expanding aliases','Building queries','Retrieving PubMed records','Extracting evidence','Scoring concordance','Building dashboard'];
let _es=null;
function startFull(){
  const yaml =document.getElementById('full-yaml').value.trim();
  const email=document.getElementById('full-email').value.trim();
  const proj =document.getElementById('proj-name').value.trim()||'My Project';
  if(!yaml) {alert('Paste your predictions.yaml first');return;}
  if(!email){alert('NCBI e-mail is required');return;}
  const btn=document.getElementById('full-run-btn');
  const log=document.getElementById('full-log');
  btn.disabled=true;
  document.getElementById('full-status').textContent='Starting…';
  document.getElementById('prog-wrap').style.display='block';
  log.style.display='block'; log.innerHTML='';
  _setStage(0);
  fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      predictions_yaml:yaml,
      ncbi_email:email,ncbi_api_key:document.getElementById('full-key').value.trim(),
      skip_pmc:document.getElementById('full-skippmc').checked,
      project_name:proj,
    })
  }).then(r=>r.json()).then(({run_id,error})=>{
    if(error){alert(error);btn.disabled=false;return;}
    document.getElementById('full-status').textContent='Run: '+run_id;
    _streamLogs(run_id,log,btn,proj);
  });
}
function handleYamlUpload(evt){
  const file=evt.target.files[0];
  if(!file) return;
  const reader=new FileReader();
  reader.onload=e=>{
    document.getElementById('full-yaml').value=e.target.result;
    document.getElementById('yaml-upload-hint').textContent='✓ '+file.name+' loaded';
    setTimeout(()=>document.getElementById('yaml-upload-hint').textContent='',4000);
  };
  reader.readAsText(file);
  // Reset so same file can be re-selected
  evt.target.value='';
}

function _setStage(n){
  const pct=Math.round((n/6)*100);
  document.getElementById('prog-fill').style.width=pct+'%';
  const lbl=n<STAGE_LABELS.length?`Stage ${n+1}/6 — ${STAGE_LABELS[n]}`:'Done';
  document.getElementById('prog-txt').textContent=lbl;
}
function _streamLogs(rid,log,btn,proj){
  if(_es) _es.close();
  _es=new EventSource('/api/logs/'+rid);
  _es.onmessage=e=>{
    const raw=e.data;
    if(raw==='[DONE]'){_es.close();return;}
    const m=raw.replace(/↵/g,'\n');
    // parse stage markers
    if(m.startsWith('STAGE:')){const n=+m.split(':')[1]; _setStage(n-1); return;}
    const div=document.createElement('div');
    div.className=m.startsWith('ERROR:')?'ler':m.startsWith('  ✓')?'lok':
      m.startsWith('▶')||m.startsWith('STAGE:')?'li':m.startsWith('DONE:')?'ldn':
      m.includes('⚠')?'lwn':'';
    div.textContent=m;
    log.appendChild(div); log.scrollTop=log.scrollHeight;
    if(m.startsWith('DONE:')){
      _setStage(6);
      btn.disabled=false;
      document.getElementById('full-status').textContent='✓ Complete';
      document.getElementById('prog-stage').querySelector('.ps-dot').style.animation='none';
      setTimeout(()=>_loadFullDash(rid,proj),600);
    }
    if(m.startsWith('ERROR:')){btn.disabled=false;}
  };
  _es.onerror=()=>{
    _es.close();
    // Don't give up — the pipeline may still be running (Stage 4 can take minutes).
    // Poll run_status: if done, show dashboard; if still running, reconnect SSE.
    function pollStatus(attempts){
      if(attempts<=0){btn.disabled=false;return;}
      fetch('/api/run_status/'+rid).then(r=>r.json()).then(d=>{
        if(d.status==='done'){
          btn.disabled=false;
          _setStage(6);
          document.getElementById('full-status').textContent='✓ Complete';
          document.getElementById('prog-stage').querySelector('.ps-dot').style.animation='none';
          const div=document.createElement('div');
          div.className='ldn'; div.textContent='DONE:'+rid+' (reconnected)';
          log.appendChild(div); log.scrollTop=log.scrollHeight;
          setTimeout(()=>_loadFullDash(rid,proj),600);
        } else if(d.status==='error'){
          btn.disabled=false;
          const div=document.createElement('div');
          div.className='ler'; div.textContent=d.message||'Pipeline error';
          log.appendChild(div);
        } else if(d.status==='running'){
          // Reconnect SSE after a short backoff
          setTimeout(()=>_streamLogs(rid,log,btn,proj), 2000);
        } else {
          // Unknown — keep polling briefly then give up
          setTimeout(()=>pollStatus(attempts-1), 3000);
        }
      }).catch(()=>{ setTimeout(()=>pollStatus(attempts-1), 3000); });
    }
    pollStatus(20);  // poll up to 20 times (≈ 60s) before giving up
  };
}
function _loadFullDash(rid,proj){
  fetch('/api/dashboard/'+rid).then(r=>r.text())
    .then(html=>openOverlay(html,proj||'Full pipeline'));
}
function loadExample(){
  document.getElementById('full-yaml').value=
`predictions:\n  - id: KRAS_up_1\n    entity: KRAS\n    entity_type: gene\n    aliases: ["KRAS", "Kirsten ras", "KRAS oncogene"]\n    disease_context: OC\n    cell_type: Tumor cells\n    tissue: lung\n    organism: human\n    direction: up\n    novelty: extending\n    prediction_note: "Driver mutation ~30% LUAD; MAPK activation"\n\n  - id: IRF5_SLE_up\n    entity: IRF5\n    entity_type: gene\n    aliases: [IRF5, "interferon regulatory factor 5"]\n    disease_context: SLE\n    cell_type: pDC\n    organism: human\n    direction: up\n    novelty: known\n    prediction_note: "IRF5 elevated in SLE pDCs"`;
}

// ══ Input mode toggle ════════════════════════════════════════════════
let _llmStatusFetched = false;
function setInputMode(mode) {
  const isNl = mode === 'nl';
  document.getElementById('yaml-panel-wrap').classList.toggle('hidden', isNl);
  document.getElementById('nl-panel').classList.toggle('visible', isNl);
  document.getElementById('mode-yaml-btn').classList.toggle('active', !isNl);
  document.getElementById('mode-nl-btn').classList.toggle('active', isNl);
  document.getElementById('load-example-btn').style.display = isNl ? 'none' : '';
  if (isNl && !_llmStatusFetched) { _llmStatusFetched=true; checkLlmStatus(); }
}

async function checkLlmStatus() {
  const statusDiv   = document.getElementById('nl-llm-status');
  const iconEl      = document.getElementById('nl-llm-icon');
  const labelEl     = document.getElementById('nl-llm-label');
  const hintEl      = document.getElementById('nl-setup-hint');
  statusDiv.style.display = 'block';
  labelEl.textContent = 'Checking LLM…';
  try {
    const resp = await fetch('/api/llm_status');
    const d = await resp.json();
    if (d.backend === 'rules') {
      iconEl.style.color = '#f59e0b';
      labelEl.textContent = '⚠ No LLM found — rule-based extraction will be used. ';
      hintEl.textContent  = d.setup_hint || '';
    } else {
      iconEl.style.color  = '#16a34a';
      iconEl.textContent  = '✓';
      const timeStr = d.expected_time ? ` (${d.expected_time})` : '';
    labelEl.textContent = `LLM ready: ${d.message}${timeStr}`;
      hintEl.textContent  = d.setup_hint || '';
    }
  } catch(e) {
    labelEl.textContent = 'Status check failed — will use rule-based fallback';
  }
}

// ══ NL prompt → YAML ═════════════════════════════════════════════════
async function generateYamlFromPrompt() {
  const prompt = (document.getElementById('nl-prompt').value || '').trim();
  if (!prompt) { showNlWarn('Please enter a description first.'); return; }

  const btn = document.getElementById('nl-gen-btn');
  btn.disabled = true;
  btn.innerHTML = '<svg width="11" height="11" viewBox="0 0 11 11" fill="currentColor" style="margin-right:4px;animation:spin .8s linear infinite"><path d="M5.5 1A4.5 4.5 0 1 1 1 5.5"/></svg> Generating…';
  hideNlWarn();

  try {
    const resp = await fetch('/api/convert_prompt', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({prompt}),
    });
    const data = await resp.json();

    if (data.error) { showNlWarn(data.error); return; }

    // Show preview
    const preview = document.getElementById('nl-preview');
    preview.value = data.yaml;
    document.getElementById('nl-preview-wrap').style.display = 'block';

    // Sync to main YAML textarea (used by startFull())
    document.getElementById('full-yaml').value = data.yaml;

    // Source badge
    const badge = document.getElementById('nl-source-badge');
    const isLlm = data.source !== 'rules';
    badge.className = 'nl-source-badge ' + (isLlm ? 'llm' : 'rules');
    badge.textContent = isLlm
      ? `✓ ${data.source}${data.model ? ': '+data.model : ''}`
      : '⚠ Rule-based fallback';

    if (data.warning) {
      const hint = data.setup_hint ? '  Tip: ' + data.setup_hint : '';
      showNlWarn('Note: ' + data.warning + hint);
    }

  } catch(e) {
    showNlWarn('Request failed: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<svg width="11" height="11" viewBox="0 0 11 11" fill="currentColor" style="margin-right:4px"><path d="M2 1.5l7.5 4L2 9.5V1.5z"/></svg> Generate YAML';
  }
}

function syncNlPreviewToMain() {
  // Keep main textarea in sync when user edits the preview
  document.getElementById('full-yaml').value =
    document.getElementById('nl-preview').value;
}

function showNlWarn(msg) {
  const el = document.getElementById('nl-warn');
  el.textContent = msg; el.style.display = 'block';
}
function hideNlWarn() {
  document.getElementById('nl-warn').style.display = 'none';
}

// ══ Utility ══════════════════════════════════════════════════════
function esc(s){const d=document.createElement('div');d.textContent=String(s||'');return d.innerHTML;}
</script>
</body>
</html>"""

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="LitRev Concordance App")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--host", default="0.0.0.0")
    a = ap.parse_args()
    print(f"\nLitRev Concordance App")
    print(f"{'─'*36}")
    print(f"  Demo data:   {'✓ (embedded)' if not _demo_file.exists() else '✓ ' + str(_demo_file)}")
    print(f"  Dashboard:   {'✓' if _DASH_PY.exists() else '✗  missing papertrail_dashboard.py'}")
    print(f"  Runs dir:    {RUNS_DIR}")
    print(f"  Shared cache:{SHARED_CACHE}")
    print(f"  URL:         http://localhost:{a.port}")
    print()
    app.run(host=a.host, port=a.port, debug=False, threaded=True)
