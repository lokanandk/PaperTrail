#!/usr/bin/env python3
"""
PaperTrail — Literature Concordance Dashboard — concordance analysis + sentence-level evidence
=============================================================================

Reads:
  --report   report.md                (from litRev pipeline stage6, always required)
  --scored   scored_predictions.json  (from stage5, strongly recommended)
  --cache    cache/pubmed_records/    (from stage3, with optional pmc_sentences)
  --out      dashboard.html           (output path, default: next to report.md)

Usage:
  # Full mode — real abstracts + PMC full text for all papers:
  python3 papertrail_dashboard.py \\
      --report  output/report.md \\
      --scored  scored_predictions.json \\
      --cache   cache/pubmed_records/ \\
      --out     output/dashboard.html

  # Minimal mode — report.md only (evidence only for ~44 discordant papers):
  python3 papertrail_dashboard.py --report output/report.md
"""

import re, sys, json, subprocess, importlib
from pathlib import Path
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════
# DEPENDENCIES
# ══════════════════════════════════════════════════════════════════

def _require(pkg, pip_name=None):
    try:
        return importlib.import_module(pkg)
    except ImportError:
        print(f"Installing {pip_name or pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               pip_name or pkg, "--break-system-packages", "-q"])
        return importlib.import_module(pkg)

_require("yake")
_require("sklearn.feature_extraction.text", "scikit-learn")

import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ══════════════════════════════════════════════════════════════════
# PARSE REPORT.MD — summary/category/tier metadata only
# ══════════════════════════════════════════════════════════════════

def parse_report(md):
    """
    Extract dashboard summary data from report.md.

    Returns per_pred WITHOUT records — those come from load_from_pipeline.
    In minimal mode (no scored_predictions.json), records are filled from
    the report's per-prediction tables (max 6/pred, no abstracts for concordant).
    """
    data = {}

    # ── Summary ───────────────────────────────────────────────────
    sb = re.search(r'## Overall Summary(.*?)##', md, re.DOTALL)
    if sb:
        s = sb.group(1)
        def gv(label):
            m = re.search(rf'\*\*{re.escape(label)}:\*\*\s*(\d+)', s)
            return int(m.group(1)) if m else 0
        data['summary'] = {
            'total':           gv('Total predictions assessed'),
            'strong':          gv('Strong'),
            'moderate':        gv('Moderate'),
            'mixed':           gv('Mixed'),
            'weak_support':    gv('Weak Support'),
            'weak_discordant': gv('Weak Discordant'),
            'descriptive':     gv('Descriptive'),
            'no_directional':  gv('No Directional'),
            'none':            gv('None'),
            'loo_fragile':     gv('LOO-fragile'),
        }
    else:
        data['summary'] = {k: 0 for k in [
            'total','strong','moderate','mixed','weak_support',
            'weak_discordant','descriptive','no_directional','none','loo_fragile']}

    # ── Categories ────────────────────────────────────────────────
    cb = re.search(r'## Cross-prediction meta-pooling by category(.*?)##', md, re.DOTALL)
    categories = []
    if cb:
        for row in re.finditer(
            r'\|\s*(\w+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)%\s*\|\s*([\d.–\-]+)\s*\|\s*([\d.e\-]+)\s*\|',
            cb.group(1)):
            try:
                categories.append({
                    'category': row.group(1),
                    'n_pred':   int(row.group(2)),
                    'n_inform': int(row.group(3)),
                    'n_concord': int(row.group(4)),
                    'pooled':   int(row.group(5)),
                    'ci':       row.group(6),
                    'p':        float(row.group(7)),
                })
            except Exception:
                pass
    data['categories'] = categories

    # ── Strong preds ──────────────────────────────────────────────
    strong_block = re.search(r'## Strong Literature Support.*?(?=##)', md, re.DOTALL)
    strong_preds = []
    if strong_block:
        for m in re.finditer(
            r'\*\*(\S+)\*\*\s*\(([^)]+)\)\s*—\s*qw-conc=([\d.]+),\s*n_informative=([\d.]+),\s*p=([\d.e\-]+)',
            strong_block.group(0)):
            try:
                strong_preds.append({
                    'id': m.group(1), 'label': m.group(2),
                    'qw_conc':  float(m.group(3)),
                    'n_inform': float(m.group(4)),
                    'p':        float(m.group(5)),
                })
            except Exception:
                pass
    data['strong_preds'] = strong_preds

    # ── Experimental priority ─────────────────────────────────────
    ep = re.search(r'## (?:Highest Experimental Priority|Novel Claims)(.*?)(?=##|\Z)', md, re.DOTALL)
    exp_preds = []
    if ep:
        for m in re.finditer(
            r'\*\*(\S+)\*\*\s*\(([^)]+)\)\s*—\s*tier\s*`([^`]+)`,\s*n_relevant=(\d+)',
            ep.group(1)):
            try:
                exp_preds.append({
                    'id': m.group(1), 'label': m.group(2),
                    'tier': m.group(3), 'n_relevant': int(m.group(4)),
                })
            except Exception:
                pass
    data['exp_preds'] = exp_preds

    # ── n_opposite + simple_conc from Discordance section ─────────
    disc_meta = {}   # pred_id -> {n_opposite, simple_conc}
    disc_sec = re.search(r'## Discordance Investigations(.+?)(?=\n## |\Z)', md, re.DOTALL)
    if disc_sec:
        for m in re.finditer(
            r'### (\S+)[^\n]*\n.*?'
            r'Expected:.*?Concordance \(simple\):\s*([\d.]+).*?n_opposite=(\d+)',
            disc_sec.group(1), re.DOTALL):
            disc_meta[m.group(1)] = {
                'simple_conc': float(m.group(2)),
                'n_opposite':  int(m.group(3)),
            }

    # ── Per-prediction details — metadata + fallback records ──────
    # KEY FIX: only split within Per-prediction details section.
    # Use the correct table block regex that captures all data rows.
    TABLE_ROW   = re.compile(
        r'\|\s*\[(\d+)\]\([^)]+\)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*'
        r'\|\s*(up|down|preserved|None|bidirectional|associated)\s*'
        r'\|\s*([✓✗—])\s*\|\s*([^|]*?)\s*\|\s*([\d.]+)\s*\|'
    )
    TABLE_BLOCK = re.compile(
        r'(\| PMID[^\n]+\n\|[-|]+\n'
        r'(?:(?:\|[^\n]+|\s{2,}-[^\n]*)\n?)+)'
    )
    TITLE_RE    = re.compile(r'\s{2,}-\s+Title:\s+_?([^_\n]+)_?')
    EXCERPT_RE  = re.compile(
        r'\s{2,}-\s+Abstract excerpt[^:]*:\s+(.+?)(?=\n\s{2,}-|\n\|[^\n]|\Z)',
        re.DOTALL)

    details_m = re.search(r'## Per-prediction details(.+?)(?=\n## |\Z)', md, re.DOTALL)
    per_pred  = []
    if details_m:
        for block in re.split(r'\n### ', details_m.group(1))[1:]:
            lines = block.strip().split('\n')
            hm = re.match(r'(\S+)\s*—\s*(.+)', lines[0])
            if not hm:
                continue
            pred_id    = hm.group(1)
            pred_label = hm.group(2).strip()

            def get(pattern):
                m = re.search(pattern, block)
                return m.group(1).strip() if m else None

            # Parse fallback records from report table (used when no scored JSON)
            records = []
            tm = TABLE_BLOCK.search(block)
            if tm:
                for row in TABLE_ROW.finditer(tm.group(1)):
                    row_end   = row.end()
                    nxt       = TABLE_ROW.search(tm.group(1), row_end)
                    after     = tm.group(1)[row_end: nxt.start() if nxt else len(tm.group(1))]
                    title_m   = TITLE_RE.search(after)
                    exc_m     = EXCERPT_RE.search(after)
                    rec_title = title_m.group(1).strip().rstrip('_') if title_m else ''
                    rec_exc   = exc_m.group(1).strip().rstrip('…') if exc_m else ''
                    ev_sents  = ([s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z\(\[])', rec_exc)
                                  if len(s.strip()) > 20] if rec_exc else [])
                    records.append({
                        'pmid':       row.group(1),
                        'year':       int(row.group(2)),
                        'journal':    row.group(3).strip(),
                        'direction':  row.group(4),
                        'concordant': row.group(5),
                        'hedged':     row.group(6).strip(),
                        'quality':    float(row.group(7)),
                        'title':             rec_title,
                        'key_phrases':       [w for w in rec_title.lower().split() if len(w) > 4][:5],
                        'evidence_sentences': ev_sents or ([rec_exc[:300]] if rec_exc else []),
                        'mechanism':         ev_sents[0] if ev_sents else rec_exc[:200],
                        'relevance_reason':  '',
                        'text_source':       'abstract' if rec_exc else 'title_only',
                    })

            loo_m = re.search(r'\*\*LOO range:\*\*\s*([\d.]+)–([\d.]+)\s*\(Δ=([\d.]+)\)', block)
            fc_m  = re.search(r'\*\*Pooled fold-changes \(n=(\d+)\):\*\*\s*median=([\d.]+)', block)
            dm    = disc_meta.get(pred_id, {})

            _dir = get(r'\*\*Expected direction:\*\*\s*(\S+)') or ''
            per_pred.append({
                'id':            pred_id,
                'label':         pred_label,
                'novelty':       get(r'\*\*Novelty:\*\*\s*(\w+)'),
                'direction':     _dir,
                'scoring_mode':  ('association'
                                  if _dir.lower() in ('bidirectional','associated','absent')
                                  else 'directional'),
                'tier':          get(r'\*\*Tier:\*\*\s*`([^`]+)`') or 'NONE',
                'reason':        get(r'\*\*Reason:\*\*\s*([^\n]+)'),
                'prediction_note':        get(r'\*\*(?:prediction_note|Finding|Prediction note):\*\*\s*([^\n]+)'),
                'loo_low':       float(loo_m.group(1)) if loo_m else None,
                'loo_high':      float(loo_m.group(2)) if loo_m else None,
                'loo_delta':     float(loo_m.group(3)) if loo_m else None,
                'fc_n':          int(fc_m.group(1))    if fc_m  else None,
                'fc_median':     float(fc_m.group(2))  if fc_m  else None,
                'qw_conc':       float(get(r'quality-weighted=([\d.]+)%') or 0) / 100 or None,
                'n_informative': int(get(r'n_informative=(\d+)') or 0) or None,
                'n_concordant':  int(get(r'n_concordant=(\d+)') or 0) or None,
                'n_relevant':    int(get(r'n_relevant=(\d+)') or 0) or None,
                'p_val':         float(get(r',\s*p=([\d.e\-]+)') or 0) or None,
                'simple_conc':   dm.get('simple_conc'),
                'n_opposite':    dm.get('n_opposite'),
                'loo_papers':    [],
                'records':       records,  # fallback; replaced by merge_evidence if --scored given
            })

    data['per_pred'] = per_pred
    return data


# ══════════════════════════════════════════════════════════════════
# LOAD FROM PIPELINE JSON
# ══════════════════════════════════════════════════════════════════

def load_from_pipeline(scored_path, cache_dir):
    """
    Load ALL papers with abstracts + PMC sentences from pipeline JSON.
    Returns flat list of paper dicts with same schema as parse_papers().
    """
    with open(scored_path, encoding='utf-8') as f:
        scored = json.load(f)
    if isinstance(scored, dict) and 'predictions' in scored and isinstance(scored['predictions'], dict):
        scored = scored['predictions']

    _cache = {}
    def _get_cache(pmid):
        if pmid in _cache:
            return _cache[pmid]
        p = cache_dir / f"{pmid}.json"
        _cache[pmid] = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
        return _cache[pmid]

    results = []
    for pred_id, s in scored.items():
        pred  = s.get('prediction', {})
        label = (f"{pred.get('entity','?')} "
                 f"({pred.get('disease_context','?')}, {pred.get('cell_type','?')})")
        exp_dir = s.get('expected_direction', pred.get('direction', 'unknown'))
        tier    = s.get('tier', '')
        n_conc  = s.get('n_concordant') or 0
        n_opp   = s.get('n_opposite')   or 0
        # scoring_mode from stage5; fall back to direction inference
        _smode = s.get('scoring_mode') or (
            'association' if exp_dir.lower() in ('bidirectional', 'associated', 'absent')
            else 'directional')
        _loo_papers = (s.get('leave_one_out') or {}).get('loo_papers', [])
        try:
            pred_concordance = float(s.get('concordance_unweighted') or 0) or None
        except (TypeError, ValueError):
            pred_concordance = None

        for r in s.get('summary_evidence', []):
            pmid = str(r.get('pmid', ''))
            if not pmid:
                continue
            rel_raw   = r.get('relation', '')
            relation  = '✓' if rel_raw == 'concordant' else ('✗' if rel_raw == 'opposite' else '—')
            direction = r.get('extracted_direction') or exp_dir
            quality   = float(r.get('quality_weight') or r.get('relevance') or 0)
            cache_rec      = _get_cache(pmid)
            title          = (r.get('title')   or cache_rec.get('title')   or '').strip()
            journal        = (r.get('journal') or cache_rec.get('journal') or '').strip()
            year           = r.get('year')      or cache_rec.get('year')   or 0
            cache_abstract = (cache_rec.get('abstract') or '').strip()
            pmc_sents      = cache_rec.get('pmc_sentences', [])
            scored_excerpt = (r.get('best_excerpt') or r.get('abstract_excerpt') or '').strip()

            if pmc_sents:
                text = cache_abstract + "  " + "  ".join(pmc_sents)
                has_abstract = True; text_source = 'full_text'
            elif cache_abstract:
                text = cache_abstract
                has_abstract = True; text_source = 'abstract'
            elif scored_excerpt:
                text = scored_excerpt
                has_abstract = True; text_source = r.get('excerpt_source', 'abstract')
            else:
                text = ''; has_abstract = False; text_source = 'title_only'

            pred_ctx = ' '.join(p for p in [
                pred.get('disease_context',''), pred.get('cell_type',''), pred.get('tissue','')
            ] if p and p.lower() != 'any')
            results.append({
                'pmid': pmid, 'year': int(year) if year else 0,
                'title': title, 'journal': journal,
                'direction': direction, 'relation': relation,
                'concordant': relation,
                'quality': quality,
                'abstract': text, 'has_abstract': has_abstract,
                'text_source': text_source,
                'pred_id': pred_id, 'pred_label': label,
                'pred_expected': exp_dir, 'pred_concordance': pred_concordance,
                'pred_tier': tier, 'is_concordant': (relation == '✓'),
                'pred_n_concordant': n_conc, 'pred_n_opposite': n_opp,
                'pred_scoring_mode': _smode,
                'pred_loo_papers':   _loo_papers,
                'pred_context': pred_ctx,
            })

    seen, deduped = set(), []
    for p in results:
        k = (p['pmid'], p['pred_id'])
        if k not in seen:
            seen.add(k); deduped.append(p)
    return deduped


# ══════════════════════════════════════════════════════════════════
# EVIDENCE EXTRACTION — YAKE + TF-IDF (from extract_evidence_keywords.py)
# ══════════════════════════════════════════════════════════════════

BIOMEDICAL_TERMS = {
    "pkm","pkm2","ldha","slc3a2","nfat5","tonebp","mtor","rptor","eif4ebp1",
    "shmt1","shmt2","mthfs","hpgd","ngal","lcn2","kim1","havcr1","vimentin",
    "vim","sdc1","syndecan","acta2","col1a1","fn1","fibronectin","vegf","hif",
    "hif-1","pgc-1","akt","p70s6k","pten","ampk","sglt2","ace","ace2","enos",
    "nos","cdk5","smad","tgfb","tgf-b","wnt","nfkb","erk","mapk","pi3k",
    "homocysteine","hcy","d-serine","d-alanine","daao","glutathione","gsh",
    "gssg","lactate","pyruvate","acetyl-coa","norepinephrine","noradrenaline",
    "dopamine","prostaglandin","methylglyoxal","nad","bcaa","creatinine","egfr",
    "proteinuria","albuminuria","copeptin","vasopressin","glycolysis",
    "gluconeogenesis","tca cycle","mtorc1","mtorc2","ferroptosis","autophagy",
    "apoptosis","senescence","epithelial-mesenchymal transition","emt","fibrosis",
    "inflammation","oxidative stress","reactive oxygen species","ros",
    "renin-angiotensin","sympathetic nervous","renal sympathetic",
    "diabetic nephropathy","diabetic kidney disease","dkd","ckd",
    "chronic kidney disease","renal fibrosis","glomerular","tubular",
    "podocyte","proximal tubule","tubulointerstitial","peritubular",
    "mesangial","interstitial","hypertensive kidney",
}
CAUSAL_RE = re.compile(
    r'\b(activat|inhibit|suppress|induc|promot|mediat|regulat|trigger|driv|'
    r'enhanc|reduc|increas|decreas|upregulat|downregulat|contribut|associat|'
    r'lead to|result in|caus|prevent|attenuate|ameliorat|protect|exacerbat|'
    r'worsen|impair)\w*\b', re.IGNORECASE)
UP_RE   = re.compile(r'\b(upregulat|elevat|increas|higher|overexpress|activat|accumul)\w*\b', re.I)
DOWN_RE = re.compile(r'\b(downregulat|reduc|decreas|lower|suppress|inhibit|deplet)\w*\b', re.I)
VIA_RE  = re.compile(r'\b(via|through|by|pathway|signaling|cascade|axis)\b', re.I)
STRUCT_LABELS = re.compile(
    r'^(BACKGROUND(\s*AND\s*OBJECTIVE)?|OBJECTIVE|METHODS?|RESULTS?|'
    r'CONCLUSIONS?|AIMS?|INTRODUCTION|PURPOSE|SIGNIFICANCE|RATIONALE|'
    r'UNLABELLED|SUMMARY|CONTEXT|DESIGN|SETTING|PATIENTS?|'
    r'INTERVENTIONS?|MEASUREMENTS?|FINDINGS?)[:\s]+', re.I)
_ABBREV_RE = re.compile(
    r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|Fig|et al|e\.g|i\.e|approx|'
    r'Eq|No|vol|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.')

def _split_sentences(text):
    text = re.sub(r'<[^>]+>', '', text).strip()
    text = _ABBREV_RE.sub(lambda m: m.group().replace('.', '\x00'), text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\(\[])', text)
    out = []
    for p in parts:
        p = p.replace('\x00', '.').strip()
        p = STRUCT_LABELS.sub('', p).strip()
        if len(p) > 20:
            out.append(p)
    return out

def _extract_key_phrases(title, abstract, pred_id, n=6):
    if not abstract:
        return _phrases_from_title(title, pred_id, n)
    text = (title + ". " + abstract).strip()
    extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.75,
                                       dedupFunc='seqm', windowsSize=2, top=30)
    raw  = extractor.extract_keywords(text)
    pred_tokens = {t.lower() for t in re.split(r'[_\-]', pred_id) if len(t) > 3}
    STOP = {'background','objective','methods','results','conclusion',
            'introduction','this study','we found','we show','however',
            'therefore','furthermore','moreover','study','aim','purpose'}
    def score(ph, sc):
        p = ph.lower(); w = set(p.split())
        lex  = sum(1 for t in BIOMEDICAL_TERMS if t in p or any(x in t for x in w if len(x) > 3))
        pred = len(w & pred_tokens)
        caus = 1 if CAUSAL_RE.search(ph) else 0
        return (1.0 / (sc + 1e-9)) * (1 + 0.5*lex + 0.3*pred + 0.2*caus)
    ranked = sorted(raw, key=lambda x: -score(x[0], x[1]))
    out = []
    for ph, _ in ranked:
        pl = ph.strip().lower().rstrip('.')
        if pl in STOP or len(pl) < 5: continue
        if any(pl in ex or ex in pl for ex in out): continue
        out.append(pl)
        if len(out) >= n: break
    return out

def _phrases_from_title(title, pred_id, n):
    words = [w for w in re.sub(r'[^\w\s\-]', ' ', title).strip().lower().split() if len(w) > 2]
    pred_tokens = {t.lower() for t in re.split(r'[_\-]', pred_id) if len(t) > 3}
    STOP = {'the','and','for','with','from','that','this','are','was','were',
            'has','have','been','its','our','their','than','into','via'}
    out = []
    for size in [3, 2, 1]:
        for i in range(len(words) - size + 1):
            chunk = words[i:i+size]
            if any(w in STOP for w in chunk): continue
            phrase = ' '.join(chunk)
            if len(phrase) < 4: continue
            has_bio  = any(t in phrase or any(w in t for w in chunk if len(w) > 3)
                          for t in BIOMEDICAL_TERMS)
            has_pred = any(w in pred_tokens for w in chunk)
            if not (has_bio or has_pred): continue
            if any(phrase in ex or ex in phrase for ex in out): continue
            out.append(phrase)
            if len(out) >= n: return out
    return out

def _extract_evidence_sentences(title, abstract, pred_id, pred_expected, direction, n=3, pred_context=""):
    sents = _split_sentences(abstract)
    if not sents:
        return [title] if title and len(title) > 20 else []
    if len(sents) == 1:
        return sents
    pred_tokens = re.sub(r'[_\-]', ' ', pred_id)
    query = f"{pred_tokens} {pred_expected} {pred_context}"
    try:
        docs = [query] + sents
        vec  = TfidfVectorizer(ngram_range=(1, 2), stop_words='english',
                               min_df=1, sublinear_tf=True).fit_transform(docs)
        sims = cosine_similarity(vec[0:1], vec[1:]).flatten()
    except Exception:
        return sents[:min(n, len(sents))]
    dir_re = UP_RE if direction == 'up' else DOWN_RE
    scored = []
    for i, (sim, sent) in enumerate(zip(sims, sents)):
        boost  = 0.25 if dir_re.search(sent) else 0.0
        boost += 0.15 if CAUSAL_RE.search(sent) else 0.0
        boost += 0.10 if VIA_RE.search(sent) else 0.0
        scored.append((sim + boost - 0.005 * i, i, sent))
    scored.sort(key=lambda x: -x[0])
    # Take top-n but preserve reading order
    top_idx = sorted(idx for _, idx, _ in scored[:n])
    return [sents[i] for i in top_idx]

def _extract_mechanism(abstract, pred_id):
    sents = _split_sentences(abstract)
    if not sents: return ""
    pred_tokens = {t.lower() for t in re.split(r'[_\-]', pred_id) if len(t) > 3}
    def mscore(s):
        sc = 0.0
        if CAUSAL_RE.search(s): sc += 1.0
        if VIA_RE.search(s):    sc += 0.5
        if any(t in s.lower() for t in pred_tokens): sc += 0.4
        if 50 <= len(s) <= 250: sc += 0.2
        return sc
    return max(sents, key=mscore)

def _build_relevance_reason(p):
    conc = f"{p['pred_concordance']:.0%}" if p.get('pred_concordance') is not None else "N/A"
    nc, no = p.get('pred_n_concordant','?'), p.get('pred_n_opposite','?')
    rel = p.get('relation', p.get('concordant', '—'))
    if rel == '✓':
        return (f"Concordant: reports {p['direction']} direction, matching expected "
                f"{p['pred_expected']} ({nc} concordant vs {no} opposing; concordance = {conc}).")
    elif rel == '✗':
        return (f"Discordant: reports {p['direction']} direction, opposing expected "
                f"{p['pred_expected']} ({nc} concordant vs {no} opposing; concordance = {conc}).")
    else:
        return f"Neutral/descriptive for {p['pred_id']} (concordance = {conc})."

def enrich_papers(papers):
    """Run YAKE + TF-IDF evidence extraction on all papers."""
    total = len(papers)
    n_ft  = sum(1 for p in papers if p.get('text_source') == 'full_text')
    n_ab  = sum(1 for p in papers if p.get('text_source') == 'abstract')
    n_ti  = sum(1 for p in papers if p.get('text_source') == 'title_only')
    print(f"  {total} papers: 🔓{n_ft} full text  📄{n_ab} abstract  {n_ti} title only")
    enriched = []
    for i, p in enumerate(papers):
        sym = '✓' if p.get('relation','')=='✓' else ('✗' if p.get('relation','')=='✗' else '—')
        src = {'full_text':'🔓','abstract':'📄'}.get(p.get('text_source',''), '  ')
        print(f"  [{i+1:3d}/{total}] {src} {p['pmid']:>8s} {sym} {p['pred_id']}", end='', flush=True)
        text = p.get('abstract','') if p.get('has_abstract') else p.get('title','')
        kw   = _extract_key_phrases(p.get('title',''), text, p['pred_id'], n=6)
        evid = _extract_evidence_sentences(
                   p.get('title',''), text,
                   p['pred_id'], p['pred_expected'], p['direction'], n=3,
                   pred_context=p.get('pred_context',''))
        mech = _extract_mechanism(text, p['pred_id']) if p.get('has_abstract') else ''
        rel  = _build_relevance_reason(p)
        rec  = dict(p)
        rec.update(key_phrases=kw, evidence_sentences=evid,
                   mechanism=mech, relevance_reason=rel)
        enriched.append(rec)
        print(f"  → {len(kw)} phrases, {len(evid)} sentences")
    return enriched


# ══════════════════════════════════════════════════════════════════
# MERGE EVIDENCE INTO DASHBOARD
# ══════════════════════════════════════════════════════════════════

def merge_evidence(dash_data, enriched_papers):
    """
    Replace per_pred[i]['records'] with the full enriched paper list.
    Each record gets: title, key_phrases, evidence_sentences, mechanism,
    relevance_reason, text_source, concordant symbol.
    """
    # Group enriched papers by pred_id
    by_pred = defaultdict(list)
    for p in enriched_papers:
        by_pred[p['pred_id']].append(p)

    # Build a pred_id → pred metadata lookup from dash_data
    pred_meta = {p['id']: p for p in dash_data['per_pred']}

    for pred in dash_data['per_pred']:
        pred_id   = pred['id']
        enriched  = by_pred.get(pred_id, [])
        if not enriched:
            # No pipeline data for this pred — keep fallback records from parse_report
            continue

        # Sort: concordant first, then discordant, then neutral; within each by quality desc
        def sort_key(p):
            rel_order = {'✓': 0, '✗': 1, '—': 2}
            return (rel_order.get(p.get('relation','—'), 2), -p.get('quality', 0))

        enriched.sort(key=sort_key)

        pred['records'] = [{
            'pmid':              p['pmid'],
            'year':              p.get('year', 0),
            'journal':           p.get('journal', ''),
            'direction':         p.get('direction', '?'),
            'concordant':        p.get('relation', '—'),
            'quality':           p.get('quality', 0),
            'title':             p.get('title', ''),
            'key_phrases':       p.get('key_phrases', []),
            'evidence_sentences': p.get('evidence_sentences', []),
            'mechanism':         p.get('mechanism', ''),
            'relevance_reason':  p.get('relevance_reason', ''),
            'text_source':       p.get('text_source', 'title_only'),
        } for p in enriched]

        # Update n_opposite from actual enriched data if not already set
        if pred.get('n_opposite') is None:
            pred['n_opposite'] = sum(1 for p in enriched if p.get('relation') == '✗')

        # Propagate scoring_mode and loo_papers from the scored data
        if enriched:
            first = enriched[0]
            if first.get('pred_scoring_mode'):
                pred['scoring_mode'] = first['pred_scoring_mode']
            if first.get('pred_loo_papers') is not None:
                pred['loo_papers']   = first['pred_loo_papers']

    return dash_data


# ══════════════════════════════════════════════════════════════════
# HTML TEMPLATE
# ══════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PaperTrail · Literature Concordance</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,700;0,9..144,900;1,9..144,700&family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet"/>
<style>
:root{
  /* Clean white + slate palette */
  --s50:#FFFFFF;--s100:#F8FAFC;--s200:#E2E8F0;--s300:#CBD5E1;--s400:#94A3B8;
  /* Dark navy text / sidebar  */
  --mah:#0F172A;--mah2:#1E293B;--mah3:#334155;--mah4:#475569;--mah5:#94A3B8;
  /* Primary accent — blue */
  --cop:#3B82F6;--copl:#EFF6FF;--copm:#60A5FA;
  /* Success — emerald (concordant) */
  --ver:#10B981;--verl:#D1FAE5;--verm:#34D399;
  /* Association — violet */
  --ind:#7C3AED;--indl:#EDE9FE;--indm:#A78BFA;
  /* Error — red (discordant) */
  --cri:#EF4444;--cril:#FEE2E2;
  /* Weak support — emerald light */
  --sag:#10B981;--sagl:#D1FAE5;
  /* Warning — amber */
  --gol:#F59E0B;--goll:#FEF3C7;
  --ff-d:'Fraunces',Georgia,serif;
  --ff-b:'Inter','Helvetica Neue',Arial,sans-serif;
  --ff-m:'Inter',sans-serif;
  --sh-sm:0 1px 3px rgba(15,23,42,.06),0 1px 2px rgba(15,23,42,.04);
  --sh-md:0 4px 16px rgba(15,23,42,.08),0 2px 8px rgba(15,23,42,.05);
  --sh-lg:0 8px 32px rgba(15,23,42,.10),0 4px 12px rgba(15,23,42,.06);
  --r:8px;--rl:14px;
}
*,::before,::after{box-sizing:border-box;margin:0;padding:0}
html{font-size:15px;scroll-behavior:smooth;-webkit-font-smoothing:antialiased}
body{background:#FFFFFF;color:var(--mah2);font-family:var(--ff-b);line-height:1.65;min-height:100vh;font-size:15px}
::selection{background:var(--copl)}
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:var(--s100)}
::-webkit-scrollbar-thumb{background:var(--s300);border-radius:3px}
a{color:var(--ver);text-decoration:none}
a:hover{text-decoration:underline}

/* Layout */
.shell{display:flex;min-height:100vh}
.sidebar{width:248px;flex-shrink:0;background:#0F172A;display:flex;flex-direction:column;position:sticky;top:0;height:100vh;overflow-y:auto;z-index:200}
.sb-brand{padding:2rem 1.75rem 1.5rem;border-bottom:1px solid rgba(255,255,255,.08)}
.sb-eye{font-family:var(--ff-m);font-size:.65rem;letter-spacing:.2em;text-transform:uppercase;color:var(--copm);margin-bottom:.5rem}
.sb-title{font-family:var(--ff-d);font-size:1.35rem;font-weight:800;color:#fff;line-height:1.15;font-style:italic}
.sb-sub{font-family:var(--ff-m);font-size:.6rem;color:rgba(255,255,255,.3);margin-top:.4rem;line-height:1.6}
.sb-deco{padding:1.25rem 1.75rem .5rem;opacity:.18}
.sb-nav{flex:1;padding:.5rem 0}
.sb-grp{font-family:var(--ff-m);font-size:.58rem;letter-spacing:.2em;text-transform:uppercase;color:rgba(255,255,255,.2);padding:.75rem 1.75rem .25rem}
.sb-item{display:flex;align-items:center;gap:.75rem;padding:.6rem 1.75rem;cursor:pointer;font-family:var(--ff-m);font-size:.72rem;color:rgba(255,255,255,.45);border-left:2px solid transparent;transition:all .18s}
.sb-item svg{width:14px;height:14px;flex-shrink:0;opacity:.65}
.sb-item:hover{color:rgba(255,255,255,.85);background:rgba(255,255,255,.04)}
.sb-item.active{color:#fff;border-left-color:var(--copm);background:rgba(255,255,255,.07)}
.sb-badge{margin-left:auto;font-size:.6rem;background:rgba(255,255,255,.1);padding:.1rem .42rem;border-radius:10px;color:rgba(255,255,255,.45)}
.sb-item.active .sb-badge{background:var(--cop);color:#fff}
.sb-foot{padding:1rem 1.75rem;border-top:1px solid rgba(255,255,255,.07);font-family:var(--ff-m);font-size:.58rem;color:rgba(255,255,255,.18);line-height:1.7}
.main{flex:1;min-width:0;display:flex;flex-direction:column}
.pg-head{background:#FFFFFF;border-bottom:1px solid var(--s200);padding:2rem 2.75rem 1.5rem;position:sticky;top:0;z-index:100}
.pg-eye{font-family:var(--ff-m);font-size:.65rem;letter-spacing:.18em;text-transform:uppercase;color:var(--cop);margin-bottom:.4rem}
.pg-title{font-family:var(--ff-d);font-size:2rem;font-weight:800;color:var(--mah);line-height:1.1;font-style:italic}
.pg-desc{font-size:.95rem;color:var(--mah3);margin-top:.3rem;font-style:italic}
.content{flex:1;overflow-y:auto;padding:2.5rem 2.75rem 5rem}
.section{display:none}
.section.active{display:block;animation:fadeUp .35s ease both}
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}

/* Hero */
.hero{position:relative;overflow:hidden;background:linear-gradient(135deg,#0F172A 0%,#1E3A5F 100%);border-radius:var(--rl);padding:2.5rem 3rem;margin-bottom:2.5rem;display:flex;align-items:center;gap:3rem;color:#fff}
.hero-txt{flex:1;min-width:0}
.hero-lbl{font-family:var(--ff-m);font-size:.65rem;letter-spacing:.18em;text-transform:uppercase;color:var(--copm);margin-bottom:.6rem}
.hero-h{font-family:var(--ff-d);font-size:1.7rem;font-weight:800;font-style:italic;color:#fff;line-height:1.15;margin-bottom:.5rem}
.hero-p{font-size:1rem;color:rgba(255,255,255,.6);line-height:1.6}
.hero-stats{display:flex;gap:2rem;margin-top:1.25rem;flex-wrap:wrap}
.hs-val{font-family:var(--ff-d);font-size:2.4rem;font-weight:800;color:var(--copm);line-height:1}
.hs-lbl{font-family:var(--ff-m);font-size:.62rem;color:rgba(255,255,255,.4);text-transform:uppercase;letter-spacing:.1em;margin-top:.2rem}
.hero-illo{flex-shrink:0;opacity:.85}

/* KPIs */
.kpi-row{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:1rem;margin-bottom:2rem}
.kpi{background:#FFFFFF;border:1px solid var(--s200);border-radius:var(--rl);padding:1.4rem 1.25rem 1.1rem;position:relative;overflow:hidden;box-shadow:var(--sh-sm);transition:box-shadow .2s,transform .2s;cursor:default}
.kpi:hover{box-shadow:var(--sh-md);transform:translateY(-2px)}
.kpi-stripe{position:absolute;top:0;left:0;right:0;height:3px;background:var(--kc,var(--cop))}
.kpi-num{font-family:var(--ff-d);font-size:2.6rem;font-weight:800;color:var(--mah);line-height:1}
.kpi-lbl{font-family:var(--ff-m);font-size:.63rem;text-transform:uppercase;letter-spacing:.1em;color:var(--mah3);margin-top:.5rem}
.kpi-sub{font-size:.8rem;color:var(--mah4);margin-top:.1rem;font-style:italic}

/* Charts */
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.25rem;margin-bottom:1.5rem}
.chart-grid.tri{grid-template-columns:1fr 1fr 1fr}
.chart-card{background:#FFFFFF;border:1px solid var(--s200);border-radius:var(--rl);padding:1.5rem;box-shadow:var(--sh-sm);overflow:hidden;transition:box-shadow .2s}
.chart-card.span2{grid-column:span 2}
.chart-lbl{font-family:var(--ff-m);font-size:.63rem;letter-spacing:.1em;text-transform:uppercase;color:var(--mah4);margin-bottom:1.1rem;display:flex;align-items:center;gap:.5rem}
.chart-lbl::after{content:'';flex:1;height:1px;background:var(--s200)}
canvas{display:block;width:100%;height:auto}

/* Tiers */
.tier{display:inline-flex;align-items:center;gap:.3rem;font-family:var(--ff-m);font-size:.6rem;font-weight:500;padding:.22rem .55rem;border-radius:20px;white-space:nowrap;border:1px solid currentColor}
.tier::before{content:'';width:5px;height:5px;border-radius:50%;background:currentColor;flex-shrink:0}
.STRONG{color:var(--ver);background:var(--verl)}.MODERATE{color:var(--ind);background:var(--indl)}
.MIXED{color:var(--cop);background:var(--copl)}.WEAK_SUPPORT{color:var(--sag);background:var(--sagl)}
.WEAK_DISCORDANT{color:var(--cri);background:var(--cril)}.DESCRIPTIVE{color:var(--mah3);background:var(--s100)}
.NO_DIRECTIONAL{color:var(--indm);background:var(--indl)}.NONE{color:var(--mah4);background:var(--s200)}

/* Tables */
.tbl-wrap{overflow-x:auto;border-radius:var(--rl);border:1px solid var(--s200);background:#fff;box-shadow:var(--sh-sm)}
.tbl-scroll{max-height:540px;overflow-y:auto}
table{width:100%;border-collapse:collapse}
thead{background:var(--s100);position:sticky;top:0;z-index:2}
th{font-family:var(--ff-m);font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;color:var(--mah3);padding:.8rem 1.1rem;text-align:left;white-space:nowrap;border-bottom:1px solid var(--s200)}
td{padding:.8rem 1.1rem;border-bottom:1px solid var(--s100);vertical-align:middle;font-size:.95rem}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--s100)}
.mono{font-family:var(--ff-m);font-size:.75rem}

/* Heatmap */
.heatmap-wrap{background:#fff;border:1px solid var(--s200);border-radius:var(--rl);padding:1.6rem;box-shadow:var(--sh-sm);margin-bottom:1.5rem}
.heatmap-strip{display:flex;gap:2px;margin-top:.75rem;flex-wrap:nowrap;overflow-x:auto;padding-bottom:.5rem}
.heatmap-cell{flex-shrink:0;width:22px;height:48px;border-radius:4px;cursor:pointer;transition:transform .15s,box-shadow .15s}
.heatmap-cell:hover{transform:scaleY(1.15);box-shadow:0 4px 12px rgba(0,0,0,.15);z-index:10}
.hm-tip{position:fixed;background:var(--mah);color:#fff;font-family:var(--ff-m);font-size:.65rem;padding:.4rem .7rem;border-radius:6px;pointer-events:none;z-index:500;display:none;white-space:nowrap;box-shadow:var(--sh-md)}
.heatmap-legend{display:flex;gap:1.5rem;margin-top:.75rem;flex-wrap:wrap;align-items:center}
.leg-item{display:flex;align-items:center;gap:.4rem;font-size:.8rem;color:var(--mah3)}
.leg-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}

/* Filter bar */
.filter-bar{display:flex;gap:.75rem;margin-bottom:1.25rem;flex-wrap:wrap;align-items:center}
.srch-wrap{position:relative;flex:1;min-width:220px}
.srch-ico{position:absolute;left:.8rem;top:50%;transform:translateY(-50%);color:var(--mah4);pointer-events:none}
.srch{width:100%;background:#fff;border:1px solid var(--s200);border-radius:var(--r);padding:.6rem .9rem .6rem 2.3rem;font-family:var(--ff-b);font-size:1rem;color:var(--mah);outline:none;box-shadow:var(--sh-sm);transition:border-color .2s,box-shadow .2s}
.srch:focus{border-color:var(--cop);box-shadow:0 0 0 3px var(--copl)}
.srch::placeholder{color:var(--mah5);font-style:italic}
.tier-pills{display:flex;gap:.4rem;flex-wrap:wrap}
.tpill{font-family:var(--ff-m);font-size:.62rem;padding:.3rem .7rem;border-radius:20px;border:1px solid currentColor;background:transparent;cursor:pointer;transition:all .15s;opacity:.55}
.tpill:hover{opacity:.85}
.tpill.active{opacity:1;background:var(--mah);color:#fff!important;border-color:var(--mah)!important}
.pred-count-bar{font-family:var(--ff-m);font-size:.65rem;color:var(--mah4);margin-bottom:.75rem;display:flex;align-items:center;gap:.75rem}
.view-toggle{display:flex;gap:.3rem}
.vbtn{font-family:var(--ff-m);font-size:.62rem;padding:.3rem .6rem;border-radius:6px;border:1px solid var(--s200);background:#fff;color:var(--mah3);cursor:pointer;transition:all .15s}
.vbtn.active{background:var(--mah);color:#fff;border-color:var(--mah)}

/* Prediction cards */
.pred-list{display:flex;flex-direction:column;gap:.75rem}
.pc{background:#fff;border:1px solid var(--s200);border-radius:var(--rl);overflow:hidden;box-shadow:var(--sh-sm);transition:box-shadow .2s;border-left:4px solid var(--plc,var(--s200))}
.pc:hover{box-shadow:var(--sh-md)}
.pc-head{display:grid;grid-template-columns:56px 1fr auto auto;align-items:center;gap:1rem;padding:1rem 1.25rem;cursor:pointer}
.pc-head:hover{background:var(--s50)}
.ring-wrap{width:52px;height:52px;position:relative;flex-shrink:0}
.ring-wrap svg{transform:rotate(-90deg)}
.ring-lbl{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--ff-m);font-size:.6rem;font-weight:500;color:var(--mah)}
.pc-info{min-width:0}
.pc-id{font-family:var(--ff-m);font-size:.75rem;color:var(--ver);font-weight:500}
.pc-ttl{font-size:1rem;color:var(--mah2);margin-top:.06rem;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pc-chips{display:flex;gap:.3rem;margin-top:.3rem;flex-wrap:wrap}
.chip{font-family:var(--ff-m);font-size:.58rem;padding:.12rem .42rem;border-radius:3px;background:var(--s100);color:var(--mah3)}
.chip.sig{color:var(--ver);background:var(--verl)}.chip.warn{color:var(--cri);background:var(--cril)}
.pc-chev{width:16px;height:16px;color:var(--mah4);flex-shrink:0;transition:transform .22s}
.pc.open .pc-chev{transform:rotate(180deg)}
.pc-note{display:none;padding:.6rem 1.25rem .7rem;background:var(--s100);font-size:.9rem;color:var(--mah3);font-style:italic;border-top:1px solid var(--s200);border-bottom:1px solid var(--s200)}
.pc.open .pc-note{display:block}
.pc-body{max-height:0;overflow:hidden;transition:max-height .32s cubic-bezier(.4,0,.2,1)}
.pc.open .pc-body{max-height:9999px}
.pc-inner{padding:1.25rem}
.kv-row{display:flex;flex-wrap:wrap;gap:.6rem;margin-bottom:1rem}
.kv{background:var(--s50);border:1px solid var(--s200);border-radius:var(--r);padding:.5rem .85rem}
.kv-k{font-family:var(--ff-m);font-size:.58rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mah4)}
.kv-v{font-family:var(--ff-m);font-size:.85rem;color:var(--mah);margin-top:.08rem}

/* ── Paper evidence cards (THE KEY COMPONENT) ── */
.papers-section{margin-top:1rem}
.papers-head{font-family:var(--ff-m);font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:var(--mah4);margin-bottom:.75rem;padding-bottom:.4rem;border-bottom:1px solid var(--s200);display:flex;align-items:center;gap:.5rem}
.papers-head::after{content:'';flex:1;height:1px;background:var(--s200)}
.paper-groups{display:flex;flex-direction:column;gap:1rem}
.pgrp-head{font-family:var(--ff-m);font-size:.6rem;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.4rem;display:flex;align-items:center;gap:.5rem}
.pgrp-dot{width:7px;height:7px;border-radius:2px;flex-shrink:0}
.paper-cards{display:flex;flex-direction:column;gap:.55rem}

/* Individual paper card */
.pcard{border-radius:var(--r);border:1px solid var(--s200);overflow:hidden;background:#fff;transition:box-shadow .15s}
.pcard:hover{box-shadow:var(--sh-md)}
.pcard.conc{border-left:3px solid var(--ver)}
.pcard.disc{border-left:3px solid var(--cri)}
.pcard.neut{border-left:3px solid var(--s300)}
.pcard-head{display:flex;align-items:flex-start;gap:.8rem;padding:.8rem 1rem;cursor:pointer;user-select:none}
.pcard-head:hover{background:var(--s50)}
.conc-icon{flex-shrink:0;margin-top:.05rem}
.pcard-info{flex:1;min-width:0}
.pcard-title{font-size:.9rem;font-weight:500;color:var(--mah);line-height:1.4;margin-bottom:.2rem}
.pcard-title a{color:var(--mah);text-decoration:none}
.pcard-title a:hover{color:var(--ver);text-decoration:underline}
.pcard-meta{display:flex;gap:.4rem;flex-wrap:wrap;align-items:center;font-family:var(--ff-m);font-size:.6rem;color:var(--mah4)}
.pcard-chev{flex-shrink:0;color:var(--mah4);transition:transform .2s;width:15px;height:15px}
.pcard.open .pcard-chev{transform:rotate(180deg)}
.pcard-body{max-height:0;overflow:hidden;transition:max-height .3s cubic-bezier(.4,0,.2,1)}
.pcard.open .pcard-body{max-height:800px}
.pcard-inner{padding:.8rem 1rem .9rem;border-top:1px solid var(--s100)}

/* Source badge */
.src-badge{font-family:var(--ff-m);font-size:.58rem;padding:.1rem .38rem;border-radius:3px;border:1px solid currentColor}
.src-ft {color:var(--ver);background:var(--verl)}
.src-abs{color:var(--ind);background:var(--indl)}
.src-ti {color:var(--mah4);background:var(--s100)}

/* Key phrases */
.kw-wrap{display:flex;flex-wrap:wrap;gap:.3rem;margin-bottom:.65rem}
.kw-tag{font-family:var(--ff-m);font-size:.6rem;background:var(--indl);color:var(--ind);border:1px solid #c8cdd6;border-radius:3px;padding:.15rem .48rem}

/* Evidence sentences */
.ev-section{margin-bottom:.6rem}
.ev-label{font-family:var(--ff-m);font-size:.58rem;text-transform:uppercase;letter-spacing:.1em;color:var(--copm);margin-bottom:.3rem;display:flex;align-items:center;gap:.35rem}
.ev-sent{font-size:.88rem;line-height:1.7;color:var(--mah2);padding:.55rem .8rem;background:var(--s50);border-left:2px solid var(--ind);border-radius:0 5px 5px 0;margin-bottom:.35rem}
.ev-sent:last-child{margin-bottom:0}
.ev-none{font-size:.82rem;color:var(--mah4);font-style:italic;padding:.4rem .6rem;background:var(--s100);border-radius:4px}

/* Mechanism */
.mech-block{font-size:.87rem;color:var(--mah3);font-style:italic;padding:.55rem .8rem;background:var(--s100);border-radius:5px;border:1px solid var(--s200);line-height:1.65;margin-bottom:.5rem}
.mech-block strong{font-style:normal;color:var(--mah2)}

/* Relevance reason */
.rel-reason{font-family:var(--ff-m);font-size:.6rem;color:var(--mah4);line-height:1.5;padding:.35rem .55rem;background:var(--s100);border-radius:4px}

/* Category cards */
.cat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:1.1rem;margin-bottom:2rem}
.cat-card{background:#fff;border:1px solid var(--s200);border-radius:var(--rl);padding:1.4rem 1.5rem;box-shadow:var(--sh-sm);display:flex;align-items:center;gap:1.25rem;transition:box-shadow .2s,transform .2s}
.cat-card:hover{box-shadow:var(--sh-md);transform:translateY(-2px)}
.cat-ring{flex-shrink:0;position:relative;width:72px;height:72px}
.cat-ring-lbl{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.cat-pct{font-family:var(--ff-d);font-size:1.1rem;font-weight:800;line-height:1}
.cat-name-text{font-family:var(--ff-m);font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;color:var(--mah3)}
.cat-meta{font-size:.82rem;color:var(--mah3);margin-top:.2rem;line-height:1.5}
.cat-p{font-family:var(--ff-m);font-size:.62rem;margin-top:.3rem}

/* Strong support */
.strong-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1.1rem}
.sc{background:#fff;border:1px solid var(--s200);border-radius:var(--rl);padding:1.5rem;box-shadow:var(--sh-sm);display:flex;gap:1.25rem;align-items:flex-start;transition:box-shadow .2s,transform .2s}
.sc:hover{box-shadow:var(--sh-md);transform:translateY(-2px)}
.sc-info{flex:1;min-width:0}
.sc-id{font-family:var(--ff-m);font-size:.78rem;color:var(--ver);font-weight:500;cursor:pointer}
.sc-label{font-size:.95rem;color:var(--mah3);font-style:italic;margin-top:.1rem}
.sc-bar-wrap{margin-top:.85rem}
.sc-bar-head{display:flex;justify-content:space-between;font-family:var(--ff-m);font-size:.62rem;color:var(--mah3);margin-bottom:.3rem}
.sc-track{height:8px;background:var(--s200);border-radius:4px;overflow:hidden}
.sc-fill{height:100%;border-radius:4px;background:var(--ver)}
.sc-stats{display:flex;gap:1.1rem;margin-top:.55rem;flex-wrap:wrap}
.sc-stat{font-family:var(--ff-m);font-size:.65rem;color:var(--mah3)}
.sc-stat b{color:var(--mah)}

/* Priority */
.pri-section{margin-bottom:2rem}
.pri-sec-head{font-family:var(--ff-m);font-size:.65rem;text-transform:uppercase;letter-spacing:.14em;color:var(--mah4);margin-bottom:.75rem;padding-bottom:.5rem;border-bottom:1px solid var(--s200);display:flex;align-items:center;gap:.75rem}
.pri-sec-head::before{content:'';width:10px;height:10px;border-radius:2px;background:var(--plc,var(--s300));flex-shrink:0}
.pri-items{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:.7rem}
.pi{background:#fff;border:1px solid var(--s200);border-radius:var(--r);padding:.9rem 1.1rem;box-shadow:var(--sh-sm);border-left:4px solid var(--plc,var(--s300));display:flex;align-items:center;gap:.9rem;transition:box-shadow .15s;cursor:pointer}
.pi:hover{box-shadow:var(--sh-md)}
.pi-info{flex:1;min-width:0}
.pi-id{font-family:var(--ff-m);font-size:.72rem;color:var(--mah);font-weight:500}
.pi-lbl{font-size:.88rem;color:var(--mah3);font-style:italic;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* Discordance */
.dc-list{display:flex;flex-direction:column;gap:1.25rem}
.dc{background:#fff;border:1px solid rgba(122,28,40,.18);border-radius:var(--rl);overflow:hidden;box-shadow:var(--sh-sm)}
.dc-head{padding:1rem 1.4rem;background:var(--cril);display:flex;align-items:center;gap:1rem;flex-wrap:wrap}
.dc-id{font-family:var(--ff-m);font-size:.82rem;color:var(--cri);font-weight:500;cursor:pointer}
.dc-body{padding:1rem 1.4rem}
.dc-finding{font-size:.92rem;color:var(--mah3);font-style:italic;margin-bottom:.9rem;padding:.6rem .9rem;background:rgba(122,28,40,.04);border-radius:var(--r);border-left:3px solid rgba(122,28,40,.2)}

/* LOO */
/* LOO papers (shown under fragile predictions) */
.loo-papers{padding:.25rem .5rem .4rem 1.5rem;display:flex;flex-direction:column;
  gap:4px;border-left:2px solid var(--cril);margin-left:.5rem}
.loo-paper{display:grid;grid-template-columns:72px 1fr auto auto;gap:6px;
  align-items:center;font-size:.67rem;color:var(--mah4)}
.loo-pmid{font-family:var(--ff-m);color:var(--cop);font-size:.64rem}
.loo-ptitle{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:.67rem}
.loo-prel{font-size:.6rem;padding:1px 5px;border-radius:3px;font-weight:600;white-space:nowrap}
.loo-prel.concordant{background:var(--verl);color:var(--ver)}
.loo-prel.opposite{background:var(--cril);color:var(--cri)}
.loo-pdelta{font-family:var(--ff-m);font-size:.64rem;color:var(--cri);white-space:nowrap}
.loo-papers-lbl{font-size:.62rem;color:var(--mah4);font-style:italic;margin-bottom:1px}
.loo-row{display:grid;grid-template-columns:200px 1fr 80px 80px;align-items:center;gap:1rem;background:#fff;border:1px solid var(--s200);border-radius:var(--r);padding:.7rem 1.1rem;box-shadow:var(--sh-sm);transition:background .15s;cursor:pointer}
.loo-row:hover{background:var(--s50)}
.loo-id{font-family:var(--ff-m);font-size:.67rem;color:var(--mah2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.loo-track{position:relative;height:10px;background:var(--s200);border-radius:5px}
.loo-fill{position:absolute;top:0;height:100%;border-radius:5px;opacity:.75}
.loo-delta{font-family:var(--ff-m);font-size:.67rem;text-align:right}
.loo-flag{font-family:var(--ff-m);font-size:.62rem;color:var(--cri)}

/* Bubble tooltip */
#bubble-tip{position:fixed;background:var(--mah);color:#fff;font-family:var(--ff-m);font-size:.67rem;padding:.55rem .85rem;border-radius:8px;pointer-events:none;z-index:500;display:none;box-shadow:var(--sh-lg);max-width:220px}
#bubble-tip .bt-id{color:var(--copm);margin-bottom:.2rem;font-weight:500}
#bubble-tip .bt-row{display:flex;justify-content:space-between;gap:.75rem;opacity:.8}

/* Animations */
.reveal{opacity:0;animation:fadeUp .4s ease both}
.reveal:nth-child(1){animation-delay:.04s}.reveal:nth-child(2){animation-delay:.08s}
.reveal:nth-child(3){animation-delay:.12s}.reveal:nth-child(4){animation-delay:.16s}
.reveal:nth-child(n+5){animation-delay:.20s}

@media(max-width:800px){
  .sidebar{display:none}.pg-head,.content{padding:1.25rem}
  .chart-grid,.chart-grid.tri{grid-template-columns:1fr}
  .chart-card.span2{grid-column:span 1}
  .hero{flex-direction:column}.hero-illo{display:none}
  .loo-row{grid-template-columns:120px 1fr 60px}
}
/* ── Concordance summary bar ────────────────────────────────────────────── */
.conc-summary-bar{display:flex;flex-wrap:wrap;align-items:center;gap:1.25rem;
  background:#FFFFFF;border:1px solid var(--s200);border-radius:var(--rl);
  padding:1.2rem 1.5rem;margin-bottom:1.5rem;box-shadow:var(--sh-sm)}
.csb-kpis{display:flex;gap:1.5rem;flex-shrink:0}
.csb-kpi{display:flex;flex-direction:column;align-items:center;min-width:64px}
.csb-val{font-family:var(--ff-d);font-size:1.6rem;font-weight:800;line-height:1}
.csb-lbl{font-family:var(--ff-m);font-size:.58rem;text-transform:uppercase;
  letter-spacing:.08em;color:var(--mah4);margin-top:2px}
.csb-stack-wrap{flex:1;min-width:200px}
.csb-stack{height:14px;border-radius:7px;overflow:hidden;display:flex;
  background:var(--s200);margin-bottom:6px}
.csb-seg{height:100%;transition:width .6s ease}
.csb-legend{display:flex;flex-wrap:wrap;gap:8px}
.csb-leg{display:flex;align-items:center;gap:4px;font-size:.68rem;color:var(--mah4)}
.csb-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.csb-dl{display:flex;gap:6px;flex-wrap:wrap}
.dl-btn{display:flex;align-items:center;gap:5px;font-family:var(--ff-m);
  font-size:.65rem;padding:5px 10px;border-radius:6px;border:1px solid var(--s200);
  background:#FFFFFF;color:var(--mah3);cursor:pointer;transition:all .15s;white-space:nowrap}
.dl-btn:hover{background:var(--cop);color:#fff;border-color:var(--cop)}
.dl-btn svg{flex-shrink:0}
</style>
</head>
<body>
<div class="shell">

<aside class="sidebar">
  <div class="sb-brand">
    <div class="sb-eye">PaperTrail</div>
    <div class="sb-title">Literature<br/>Concordance</div>
    <div class="sb-sub">Quality-weighted<br/>Hedge-discounted<br/>LOO-sensitivity-tested</div>
  </div>
  <div class="sb-deco">
    <svg width="180" height="60" viewBox="0 0 180 60" fill="none">
      <circle cx="20" cy="30" r="12" stroke="#d4823c" stroke-width="1.5"/>
      <circle cx="60" cy="20" r="8" stroke="#d4823c" stroke-width="1.5"/>
      <circle cx="100" cy="35" r="15" stroke="#d4823c" stroke-width="1.5"/>
      <circle cx="145" cy="22" r="6" stroke="#d4823c" stroke-width="1.5"/>
      <circle cx="170" cy="38" r="9" stroke="#d4823c" stroke-width="1.5"/>
      <line x1="32" y1="30" x2="52" y2="20" stroke="#d4823c" stroke-width="1" stroke-dasharray="2,3"/>
      <line x1="68" y1="20" x2="85" y2="32" stroke="#d4823c" stroke-width="1" stroke-dasharray="2,3"/>
      <line x1="115" y1="30" x2="139" y2="24" stroke="#d4823c" stroke-width="1" stroke-dasharray="2,3"/>
      <line x1="151" y1="24" x2="161" y2="34" stroke="#d4823c" stroke-width="1" stroke-dasharray="2,3"/>
    </svg>
  </div>
  <nav class="sb-nav">
    <div class="sb-grp">Analysis</div>
    <div class="sb-item active" onclick="nav('overview')">
      <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="1" y="1" width="5" height="5" rx="1"/><rect x="8" y="1" width="5" height="5" rx="1"/><rect x="1" y="8" width="5" height="5" rx="1"/><rect x="8" y="8" width="5" height="5" rx="1"/></svg>
      Overview <span class="sb-badge" id="b-ov">—</span>
    </div>
    <div class="sb-item" onclick="nav('categories')">
      <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M1 11L4 4l3 4 3-3.5 3 6.5"/></svg>
      Categories <span class="sb-badge" id="b-cat">—</span>
    </div>
    <div class="sb-grp">Predictions</div>
    <div class="sb-item" onclick="nav('predictions')">
      <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M2 3.5h10M2 7h7M2 10.5h9"/></svg>
      All Predictions <span class="sb-badge" id="b-pred">—</span>
    </div>
    <div class="sb-item" onclick="nav('strong')">
      <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M7 1l1.7 4.2H13l-3.5 2.6 1.3 4.2L7 9.5l-3.8 2.5 1.3-4.2L1 5.2h4.3z"/></svg>
      Strong Support <span class="sb-badge" id="b-str">—</span>
    </div>
    <div class="sb-item" onclick="nav('priority')">
      <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M7 2v10M2.5 6.5L7 2l4.5 4.5"/></svg>
      Exp. Priority <span class="sb-badge" id="b-pri">—</span>
    </div>
    <div class="sb-grp">Quality</div>
    <div class="sb-item" onclick="nav('discordance')">
      <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="7" cy="7" r="5.5"/><path d="M7 4.5v3M7 9.5v.5"/></svg>
      Discordance <span class="sb-badge" id="b-disc">—</span>
    </div>
    <div class="sb-item" onclick="nav('loo')">
      <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="7" cy="7" r="3"/><path d="M1 7h3M10 7h3M7 1v3M7 10v3"/></svg>
      LOO Sensitivity <span class="sb-badge" id="b-loo">—</span>
    </div>
  </nav>
  <div class="sb-foot">Concordance is quality-weighted<br/>and hedge-discounted. LOO<br/>tests single-paper fragility.</div>
</aside>

<div class="main">
<div class="pg-head">
  <div class="pg-eye" id="pg-eye">Overview</div>
  <div class="pg-title" id="pg-title">Concordance Summary</div>
  <div class="pg-desc"  id="pg-desc"></div>
</div>
<div class="content">

<!-- OVERVIEW -->
<div class="section active" id="s-overview">
  <div class="hero">
    <div class="hero-txt">
      <div class="hero-lbl">PaperTrail · Literature Concordance</div>
      <h2 class="hero-h">Systematic concordance<br/>across all predictions</h2>
      <p class="hero-p">Quality-weighted, hedge-discounted concordance. Each prediction scored against the full literature. Click any paper to see sentence-level evidence.</p>
      <div class="hero-stats">
        <div><div class="hs-val" id="hs-total">0</div><div class="hs-lbl">Total predictions</div></div>
        <div><div class="hs-val" id="hs-strong">0</div><div class="hs-lbl">Strong concordance</div></div>
        <div><div class="hs-val" id="hs-fragile">0</div><div class="hs-lbl">LOO-fragile</div></div>
      </div>
    </div>
    <div class="hero-illo">
      <svg width="220" height="180" viewBox="0 0 220 180" fill="none">
        <circle cx="110" cy="90" r="75" stroke="rgba(255,255,255,.06)" stroke-width="1.5"/>
        <circle cx="110" cy="90" r="55" stroke="rgba(255,255,255,.06)" stroke-width="1.5"/>
        <circle cx="110" cy="90" r="35" stroke="rgba(255,255,255,.06)" stroke-width="1.5"/>
        <circle cx="110" cy="90" r="10" fill="rgba(212,130,60,.9)"/>
        <circle cx="50" cy="55" r="6" fill="rgba(42,155,128,.7)"/>
        <circle cx="170" cy="55" r="8" fill="rgba(42,155,128,.7)"/>
        <circle cx="40" cy="130" r="5" fill="rgba(74,85,144,.7)"/>
        <circle cx="180" cy="130" r="7" fill="rgba(122,28,40,.7)"/>
        <circle cx="110" cy="20" r="5" fill="rgba(212,130,60,.5)"/>
        <line x1="110" y1="90" x2="50" y2="55" stroke="rgba(255,255,255,.18)" stroke-width="1"/>
        <line x1="110" y1="90" x2="170" y2="55" stroke="rgba(255,255,255,.18)" stroke-width="1"/>
        <line x1="110" y1="90" x2="40" y2="130" stroke="rgba(255,255,255,.12)" stroke-width="1"/>
        <line x1="110" y1="90" x2="180" y2="130" stroke="rgba(255,255,255,.12)" stroke-width="1"/>
        <line x1="110" y1="90" x2="110" y2="20" stroke="rgba(255,255,255,.15)" stroke-width="1"/>
      </svg>
    </div>
  </div>
  <div class="conc-summary-bar" id="conc-bar">
    <div class="csb-kpis" id="csb-kpis">
      <div class="csb-kpi"><span class="csb-val" id="csb-concordant" style="color:#10B981">—</span><span class="csb-lbl">directional</span></div>
      <div class="csb-kpi"><span class="csb-val" id="csb-assoc" style="color:#7C3AED">—</span><span class="csb-lbl">association</span></div>
      <div class="csb-kpi"><span class="csb-val" id="csb-discordant" style="color:#EF4444">—</span><span class="csb-lbl">discordant</span></div>
      <div class="csb-kpi"><span class="csb-val" id="csb-none" style="color:#94A3B8">—</span><span class="csb-lbl">no evidence</span></div>
    </div>
    <div class="csb-stack-wrap">
      <div class="csb-stack" id="csb-stack"></div>
      <div class="csb-legend" id="csb-legend"></div>
    </div>
    <div class="csb-dl">
      <button class="dl-btn" onclick="dlCSV()" title="Export paper list as CSV">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M6 1v7M3 5.5l3 2.5 3-2.5M1 10h10"/></svg> Papers CSV
      </button>
      <button class="dl-btn" onclick="dlJSON()" title="Export prediction summary as JSON">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M6 1v7M3 5.5l3 2.5 3-2.5M1 10h10"/></svg> Summary JSON
      </button>
      <button class="dl-btn" onclick="dlCharts()" title="Download charts as PNG">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 1h10v10H1zM3 6h6M3 8l2-2 2 2 2-3"/></svg> Charts PNG
      </button>
      <button class="dl-btn" onclick="dlHTML()" title="Download dashboard as HTML">
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 1h10v10H1zM3 4l3 3-3 3M7 10h3"/></svg> Dashboard HTML
      </button>
    </div>
  </div>

  <div class="kpi-row" id="kpi-row"></div>
  <div class="heatmap-wrap">
    <div class="chart-lbl">All predictions — concordance heatmap (hover · click to open)</div>
    <div class="heatmap-strip" id="heatmap-strip"></div>
    <div class="heatmap-legend" id="heatmap-legend"></div>
  </div>
  <div class="chart-grid">
    <div class="chart-card"><div class="chart-lbl">Tier distribution</div><canvas id="cv-donut" height="240"></canvas></div>
    <div class="chart-card"><div class="chart-lbl">Category pooled concordance</div><canvas id="cv-catbar" height="240"></canvas></div>
    <div class="chart-card span2" style="position:relative">
      <div class="chart-lbl">QW concordance vs evidence volume — click bubble to open</div>
      <canvas id="cv-bubble" height="280" style="cursor:crosshair"></canvas>
      <div id="bubble-tip">
        <div class="bt-id" id="bt-id"></div>
        <div class="bt-row"><span>QW Conc.</span><span id="bt-qw"></span></div>
        <div class="bt-row"><span>N Informative</span><span id="bt-n"></span></div>
        <div class="bt-row"><span>Tier</span><span id="bt-tier"></span></div>
        <div class="bt-row"><span>p-value</span><span id="bt-p"></span></div>
      </div>
    </div>
  </div>
  <div class="chart-grid tri">
    <div class="chart-card"><div class="chart-lbl">Prediction novelty</div><canvas id="cv-novelty" height="180"></canvas></div>
    <div class="chart-card"><div class="chart-lbl">Expected direction</div><canvas id="cv-dir" height="180"></canvas></div>
    <div class="chart-card"><div class="chart-lbl">LOO fragility by tier</div><canvas id="cv-loo-tier" height="180"></canvas></div>
  </div>
</div>

<!-- CATEGORIES -->
<div class="section" id="s-categories">
  <div class="hero">
    <div class="hero-txt">
      <div class="hero-lbl">Meta-Pooling Analysis</div>
      <h2 class="hero-h">Biological pathways,<br/>systematically pooled</h2>
      <p class="hero-p">Quality-weighted concordance pooled across all predictions within each category.</p>
    </div>
    <div class="hero-illo">
      <svg width="200" height="160" viewBox="0 0 200 160" fill="none">
        <polygon points="100,20 175,65 155,148 45,148 25,65" stroke="rgba(255,255,255,.12)" stroke-width="1.5" fill="none"/>
        <polygon points="100,45 148,73 133,127 67,127 52,73" stroke="rgba(255,255,255,.1)" stroke-width="1.5" fill="none"/>
        <polygon points="100,70 121,80 111,108 89,108 79,80" fill="rgba(212,130,60,.2)" stroke="rgba(212,130,60,.5)" stroke-width="1.5"/>
      </svg>
    </div>
  </div>
  <div class="cat-grid" id="cat-grid"></div>
  <div class="tbl-wrap tbl-scroll">
    <table><thead><tr>
      <th>Category</th><th>Predictions</th><th>N Informative</th><th>N Concordant</th><th>Pooled %</th><th>95% CI</th><th>p-value</th><th style="min-width:120px">Bar</th>
    </tr></thead><tbody id="cat-tbody"></tbody></table>
  </div>
</div>

<!-- PREDICTIONS -->
<div class="section" id="s-predictions">
  <div class="filter-bar">
    <div class="srch-wrap">
      <svg class="srch-ico" width="15" height="15" viewBox="0 0 15 15" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="6.5" cy="6.5" r="4.5"/><path d="M10.5 10.5l3 3"/></svg>
      <input class="srch" id="psearch" placeholder="Search ID, gene, finding…" oninput="filterPreds()"/>
    </div>
    <div class="tier-pills" id="tier-pills"></div>
  </div>
  <div class="pred-count-bar">
    <span id="pred-count"></span>
    <div style="flex:1"></div>
    <div class="view-toggle">
      <button class="vbtn active" onclick="setView('card',this)">Cards</button>
      <button class="vbtn" onclick="setView('table',this)">Table</button>
    </div>
  </div>
  <div class="pred-list" id="pred-list"></div>
  <div class="tbl-wrap tbl-scroll" id="pred-tbl-wrap" style="display:none">
    <table><thead><tr>
      <th>ID</th><th>Gene / Analyte</th><th>Tier</th><th>QW Conc.</th><th>N Inf.</th><th>p-value</th><th>Novelty</th><th>Direction</th>
    </tr></thead><tbody id="pred-tbl-body"></tbody></table>
  </div>
</div>

<!-- STRONG -->
<div class="section" id="s-strong">
  <div class="hero">
    <div class="hero-txt">
      <div class="hero-lbl">Literature Support</div>
      <h2 class="hero-h">Robust concordance —<br/>cite, don't re-validate</h2>
      <p class="hero-p">These predictions have strong quality-weighted literature agreement.</p>
    </div>
    <div class="hero-illo">
      <svg width="180" height="160" viewBox="0 0 180 160" fill="none">
        <path d="M90 140 L20 60 Q90 10 160 60 Z" fill="none" stroke="rgba(42,155,128,.3)" stroke-width="1.5"/>
        <path d="M90 120 L40 70 Q90 30 140 70 Z" fill="rgba(42,155,128,.15)" stroke="rgba(42,155,128,.5)" stroke-width="1.5"/>
        <circle cx="90" cy="75" r="18" fill="rgba(42,155,128,.25)" stroke="rgba(42,155,128,.7)" stroke-width="2"/>
        <path d="M83 75 L88 81 L98 68" stroke="rgba(42,155,128,1)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
  </div>
  <div class="strong-grid" id="strong-grid"></div>
</div>

<!-- PRIORITY -->
<div class="section" id="s-priority">
  <div class="hero">
    <div class="hero-txt">
      <div class="hero-lbl">Experimental Design</div>
      <h2 class="hero-h">Novel predictions awaiting<br/>experimental validation</h2>
      <p class="hero-p">These predictions lack sufficient literature and represent the highest-value experimental targets.</p>
    </div>
    <div class="hero-illo">
      <svg width="180" height="160" viewBox="0 0 180 160" fill="none">
        <path d="M70 30 L70 85 L30 145 Q30 155 40 155 L140 155 Q150 155 150 145 L110 85 L110 30 Z" stroke="rgba(255,255,255,.2)" stroke-width="1.5" fill="none"/>
        <path d="M70 100 L35 148 Q35 153 40 153 L140 153 Q145 153 145 148 L110 100 Z" fill="rgba(212,130,60,.2)"/>
        <line x1="65" y1="30" x2="115" y2="30" stroke="rgba(255,255,255,.3)" stroke-width="2" stroke-linecap="round"/>
      </svg>
    </div>
  </div>
  <div id="pri-container"></div>
</div>

<!-- DISCORDANCE -->
<div class="section" id="s-discordance">
  <div class="hero">
    <div class="hero-txt">
      <div class="hero-lbl">Quality Control</div>
      <h2 class="hero-h">Predictions with opposing<br/>literature evidence</h2>
      <p class="hero-p">These predictions have at least one paper reporting the opposite direction. Click any paper to see its evidence sentences.</p>
    </div>
    <div class="hero-illo">
      <svg width="180" height="160" viewBox="0 0 180 160" fill="none">
        <path d="M30 60 L90 60 L90 40 L145 80 L90 120 L90 100 L30 100 Z" fill="rgba(42,155,128,.15)" stroke="rgba(42,155,128,.5)" stroke-width="1.5"/>
        <path d="M150 60 L90 60 L90 40 L35 80 L90 120 L90 100 L150 100 Z" fill="rgba(122,28,40,.12)" stroke="rgba(122,28,40,.4)" stroke-width="1.5" transform="translate(180,0) scale(-1,1)"/>
      </svg>
    </div>
  </div>
  <div class="dc-list" id="dc-list"></div>
</div>

<!-- LOO -->
<div class="section" id="s-loo">
  <div class="hero">
    <div class="hero-txt">
      <div class="hero-lbl">Robustness Analysis</div>
      <h2 class="hero-h">Leave-one-out sensitivity —<br/>how fragile is each finding?</h2>
      <p class="hero-p">LOO analysis removes one paper at a time and recomputes concordance. Δ ≥ 0.10 flags single-study-driven findings.</p>
    </div>
    <div class="hero-illo">
      <svg width="180" height="160" viewBox="0 0 180 160" fill="none">
        <line x1="90" y1="30" x2="90" y2="130" stroke="rgba(255,255,255,.2)" stroke-width="2"/>
        <line x1="40" y1="50" x2="140" y2="50" stroke="rgba(255,255,255,.25)" stroke-width="2"/>
        <circle cx="40" cy="50" r="8" fill="rgba(42,155,128,.4)" stroke="rgba(42,155,128,.8)" stroke-width="1.5"/>
        <circle cx="140" cy="50" r="8" fill="rgba(122,28,40,.4)" stroke="rgba(122,28,40,.8)" stroke-width="1.5"/>
      </svg>
    </div>
  </div>
  <div class="chart-grid" style="margin-bottom:1.5rem">
    <div class="chart-card span2"><div class="chart-lbl">LOO Δ — directional predictions</div><canvas id="cv-loo-bar" height="300"></canvas></div>
  </div>
  <div class="loo-grid" id="loo-grid"></div>
</div>

</div><!-- /content -->
</div><!-- /main -->
</div><!-- /shell -->

<div class="hm-tip" id="hm-tip"></div>

<script>
const DATA = __DATA__;

// ── Colours ────────────────────────────────────────────────
const TC={STRONG:'#10B981',MODERATE:'#3B82F6',MIXED:'#F97316',WEAK_SUPPORT:'#F59E0B',WEAK_DISCORDANT:'#F97316',DESCRIPTIVE:'#7C3AED',NO_DIRECTIONAL:'#7C3AED',NO_INFORMATIVE:'#94A3B8',NONE:'#64748B',LOO_FRAGILE:'#EF4444'};
const TL={STRONG:'Strong',MODERATE:'Moderate',MIXED:'Mixed',WEAK_SUPPORT:'Weak Support',WEAK_DISCORDANT:'Weak Discordant',DESCRIPTIVE:'Descriptive',NO_DIRECTIONAL:'No Directional',NONE:'None'};
const TIERS=['STRONG','MODERATE','MIXED','WEAK_SUPPORT','WEAK_DISCORDANT','DESCRIPTIVE','NO_DIRECTIONAL','NONE'];
const PM={
  overview:    {eye:'Overview',        title:'Concordance Summary',         desc:'Quality-weighted, hedge-discounted concordance · click any paper to see sentence evidence'},
  categories:  {eye:'Meta-Pooling',    title:'Category Analysis',           desc:'Cross-prediction pooled concordance by biological pathway'},
  predictions: {eye:'All Predictions', title:'Prediction Browser',          desc:'Search, filter, and drill into every prediction with full sentence-level evidence'},
  strong:      {eye:'Literature',      title:'Strong Concordance',          desc:'Predictions with robust, citable literature agreement'},
  priority:    {eye:'Experimental',    title:'Priority Targets',            desc:'Novel predictions requiring experimental validation'},
  discordance: {eye:'Quality Control', title:'Discordance Review',          desc:'Predictions where literature reports the opposite direction — with sentence evidence for each paper'},
  loo:         {eye:'Robustness',      title:'LOO Sensitivity Analysis',    desc:'How concordance shifts when any single paper is removed'},
};

function nav(id){
  document.querySelectorAll('.section').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.sb-item').forEach(s=>s.classList.remove('active'));
  document.getElementById('s-'+id).classList.add('active');
  const idx={overview:0,categories:1,predictions:2,strong:3,priority:4,discordance:5,loo:6};
  const items=[...document.querySelectorAll('.sb-item')];
  if(items[idx[id]])items[idx[id]].classList.add('active');
  const m=PM[id];
  document.getElementById('pg-eye').textContent=m.eye;
  document.getElementById('pg-title').textContent=m.title;
  document.getElementById('pg-desc').textContent=m.desc;
}

// ── Canvas helpers ─────────────────────────────────────────
function makeCtx(id,h){
  const el=document.getElementById(id);if(!el)return null;
  const dpr=Math.min(window.devicePixelRatio||1,2);
  const W=Math.max(el.parentElement.clientWidth-32,300)||600;
  const H=h||240;
  el.width=W*dpr;el.height=H*dpr;
  el.style.width=W+'px';el.style.height=H+'px';
  const ctx=el.getContext('2d');ctx.scale(dpr,dpr);
  return{ctx,W,H};
}
// Cross-browser rounded rectangle (ctx.roundRect not available in all browsers)
function fillRR(ctx,x,y,w,h,r){
  if(w<1||h<1)return;
  r=Math.min(r,w/2,h/2);
  ctx.beginPath();
  ctx.moveTo(x+r,y);ctx.lineTo(x+w-r,y);ctx.quadraticCurveTo(x+w,y,x+w,y+r);
  ctx.lineTo(x+w,y+h-r);ctx.quadraticCurveTo(x+w,y+h,x+w-r,y+h);
  ctx.lineTo(x+r,y+h);ctx.quadraticCurveTo(x,y+h,x,y+h-r);
  ctx.lineTo(x,y+r);ctx.quadraticCurveTo(x,y,x+r,y);
  ctx.closePath();ctx.fill();
}
function animRun(dur,cb){
  const s=performance.now();
  (function loop(now){const p=Math.min(1,(now-s)/dur);cb(1-Math.pow(1-p,3));if(p<1)requestAnimationFrame(loop);})(performance.now());
}
function countUp(id,target,delay=0){setTimeout(()=>animRun(900,p=>{const el=document.getElementById(id);if(el)el.textContent=Math.round(target*p);}),delay);}
function hexRgb(h){return`${parseInt(h.slice(1,3),16)},${parseInt(h.slice(3,5),16)},${parseInt(h.slice(5,7),16)}`;}

// ── Overview ───────────────────────────────────────────────
function buildOverview(){
  const s=DATA.summary;
  countUp('hs-total',s.total,0);countUp('hs-strong',s.strong,200);countUp('hs-fragile',s.loo_fragile,400);
  const kpis=[
    {l:'Total',v:s.total,sub:'predictions assessed',c:'#b5601a'},
    {l:'Strong',v:s.strong,sub:'robust concordance',c:TC.STRONG},
    {l:'Moderate',v:s.moderate,sub:'good support',c:TC.MODERATE},
    {l:'Mixed',v:s.mixed,sub:'conflicting signals',c:TC.MIXED},
    {l:'Weak Support',v:s.weak_support,sub:'thin evidence',c:TC.WEAK_SUPPORT},
    {l:'Weak Discordant',v:s.weak_discordant,sub:'opposing literature',c:TC.WEAK_DISCORDANT},
    {l:'No Literature',v:s.none,sub:'no records found',c:TC.NONE},
    {l:'LOO-Fragile',v:s.loo_fragile,sub:'single-paper driven',c:TC.WEAK_DISCORDANT},
  ];
  const row=document.getElementById('kpi-row');
  kpis.forEach((k,i)=>{
    const d=document.createElement('div');d.className='kpi reveal';d.style.setProperty('--kc',k.c);
    d.innerHTML=`<div class="kpi-stripe"></div><div class="kpi-num" id="kn${i}">0</div><div class="kpi-lbl">${k.l}</div><div class="kpi-sub">${k.sub}</div>`;
    row.appendChild(d);countUp('kn'+i,k.v,i*60);
  });
  buildHeatmap();drawDonut();drawCatBar();drawBubble();drawNovelty();drawDirection();drawLOObyTier();
}

function buildHeatmap(){
  const sorted=[...DATA.per_pred].sort((a,b)=>{const ti=TIERS.indexOf(a.tier)-TIERS.indexOf(b.tier);return ti!==0?ti:(b.qw_conc||0)-(a.qw_conc||0);});
  const strip=document.getElementById('heatmap-strip');
  const tip=document.getElementById('hm-tip');
  sorted.forEach(p=>{
    const pct=p.qw_conc!=null?Math.round(p.qw_conc*100):null;
    const col=TC[p.tier]||'#9a6450';
    const cell=document.createElement('div');cell.className='heatmap-cell';
    cell.style.background=pct!=null?`rgba(${hexRgb(col)},${0.2+(pct/100)*0.8})`:'rgba(154,100,80,0.15)';
    cell.addEventListener('mousemove',e=>{tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-30)+'px';tip.innerHTML=`<b style="color:var(--copm)">${p.id}</b><br/>${TL[p.tier]||p.tier} · ${pct!=null?pct+'% conc.':'no data'}`;});
    cell.addEventListener('mouseleave',()=>tip.style.display='none');
    cell.addEventListener('click',()=>{nav('predictions');setTimeout(()=>{document.getElementById('psearch').value=p.id;filterPreds();},100);});
    strip.appendChild(cell);
  });
  const leg=document.getElementById('heatmap-legend');
  TIERS.forEach(t=>{const n=DATA.per_pred.filter(p=>p.tier===t).length;if(!n)return;leg.innerHTML+=`<div class="leg-item"><div class="leg-dot" style="background:${TC[t]}"></div>${TL[t]} (${n})</div>`;});
}

function drawDonut(){
  const r=makeCtx('cv-donut',240);if(!r)return;
  const{ctx,W,H}=r;const s=DATA.summary;
  const sl=[{l:'Strong',v:s.strong,c:TC.STRONG},{l:'Moderate',v:s.moderate,c:TC.MODERATE},{l:'Mixed',v:s.mixed,c:TC.MIXED},{l:'Weak Support',v:s.weak_support,c:TC.WEAK_SUPPORT},{l:'Weak Discordant',v:s.weak_discordant,c:TC.WEAK_DISCORDANT},{l:'Descriptive',v:s.descriptive,c:TC.DESCRIPTIVE},{l:'No Directional',v:s.no_directional,c:TC.NO_DIRECTIONAL},{l:'None',v:s.none,c:TC.NONE}].filter(x=>x.v>0);
  const tot=sl.reduce((a,b)=>a+b.v,0);
  const cx=W*.38,cy=H/2,R=Math.min(cx,cy)-14,inn=R*.52;
  animRun(900,prog=>{
    ctx.clearRect(0,0,W,H);let a=-Math.PI/2;
    sl.forEach(s=>{const sw=(s.v/tot)*2*Math.PI*prog;ctx.beginPath();ctx.moveTo(cx+inn*Math.cos(a),cy+inn*Math.sin(a));ctx.arc(cx,cy,R,a,a+sw);ctx.arc(cx,cy,inn,a+sw,a,true);ctx.closePath();ctx.fillStyle=s.c;ctx.fill();ctx.strokeStyle='#faf8f5';ctx.lineWidth=2.5;ctx.stroke();a+=sw;});
    ctx.font=`800 22px 'Fraunces',serif`;ctx.fillStyle='#2c1810';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(tot,cx,cy-5);
    ctx.font=`10px 'Inter',sans-serif`;ctx.fillStyle='#64748B';ctx.fillText('predictions',cx,cy+12);
    const lx=W*.62,ly=H/2-(sl.length*20)/2;
    sl.forEach((s,i)=>{const y=ly+i*21;ctx.fillStyle=s.c;fillRR(ctx,lx,y+1,10,10,2);ctx.font=`11px 'Inter',sans-serif`;ctx.fillStyle='#334155';ctx.textAlign='left';ctx.textBaseline='top';ctx.fillText(s.l,lx+14,y+1);ctx.fillStyle='#64748B';ctx.textAlign='right';ctx.fillText(s.v,W-4,y+1);});
  });
}

function drawCatBar(){
  const r=makeCtx('cv-catbar',240);if(!r)return;
  const{ctx,W,H}=r;const cats=[...DATA.categories].sort((a,b)=>b.pooled-a.pooled);
  const pad={l:85,r:40,t:8,b:8};const bH=Math.floor((H-pad.t-pad.b)/cats.length)-5;
  animRun(900,prog=>{
    ctx.clearRect(0,0,W,H);
    cats.forEach((c,i)=>{
      const y=pad.t+i*(bH+5);const bW=(c.pooled/100)*(W-pad.l-pad.r)*prog;
      const col=c.pooled>=75?TC.STRONG:c.pooled>=60?TC.MODERATE:TC.MIXED;
      ctx.fillStyle='#E2E8F0';fillRR(ctx,pad.l,y,W-pad.l-pad.r,bH,4);
      if(bW>0){ctx.fillStyle=col;fillRR(ctx,pad.l,y,bW,bH,4);}
      if(c.p<0.05&&bW>10){ctx.fillStyle='#fff';ctx.font=`bold 9px 'Inter',sans-serif`;ctx.textAlign='left';ctx.textBaseline='middle';ctx.fillText('✓',pad.l+bW-16,y+bH/2);}
      ctx.font=`10.5px 'Inter',sans-serif`;ctx.fillStyle='#334155';ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(c.category,pad.l-7,y+bH/2);
      ctx.fillStyle='#64748B';ctx.textAlign='left';ctx.fillText(c.pooled+'%',pad.l+bW+5,y+bH/2);
    });
  });
}

function drawBubble(){
  const r=makeCtx('cv-bubble',280);if(!r)return;
  const{ctx,W,H}=r;const c=document.getElementById('cv-bubble');
  const tip=document.getElementById('bubble-tip');
  const preds=DATA.per_pred.filter(p=>p.n_informative&&p.qw_conc!=null);
  const maxN=Math.max(...preds.map(p=>p.n_informative));
  const pad={l:46,r:20,t:16,b:38};
  function draw(prog){
    ctx.clearRect(0,0,W,H);
    [0,25,50,75,100].forEach(v=>{const y=pad.t+(1-v/100)*(H-pad.t-pad.b);ctx.strokeStyle='#e8e0d0';ctx.lineWidth=1;ctx.setLineDash([3,5]);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.setLineDash([]);ctx.font=`9px 'Inter',sans-serif`;ctx.fillStyle='#b8a890';ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(v+'%',pad.l-4,y);});
    ctx.save();ctx.translate(10,H/2);ctx.rotate(-Math.PI/2);ctx.font=`9px 'Inter',sans-serif`;ctx.fillStyle='#b8a890';ctx.textAlign='center';ctx.fillText('QW Concordance',0,0);ctx.restore();
    ctx.font=`9px 'Inter',sans-serif`;ctx.fillStyle='#b8a890';ctx.textAlign='center';ctx.fillText('Evidence Volume (N informative)',pad.l+(W-pad.l-pad.r)/2,H-4);
    preds.forEach(p=>{const x=pad.l+(p.n_informative/maxN)*(W-pad.l-pad.r);const y=pad.t+(1-p.qw_conc)*(H-pad.t-pad.b);const rad=Math.sqrt(p.n_informative/maxN)*14*prog+3;ctx.beginPath();ctx.arc(x,y,rad,0,Math.PI*2);ctx.fillStyle=(TC[p.tier]||'#9a6450')+'b0';ctx.fill();ctx.strokeStyle=TC[p.tier]||'#9a6450';ctx.lineWidth=1.2;ctx.stroke();});
  }
  animRun(1000,draw);
  c.addEventListener('mousemove',e=>{
    const rect=c.getBoundingClientRect();const mx=(e.clientX-rect.left)*(c.width/rect.width/(window.devicePixelRatio||1));const my=(e.clientY-rect.top)*(c.height/rect.height/(window.devicePixelRatio||1));
    let nearest=null,nd=999;
    preds.forEach(p=>{const x=pad.l+(p.n_informative/maxN)*(W-pad.l-pad.r);const y=pad.t+(1-p.qw_conc)*(H-pad.t-pad.b);const d=Math.hypot(mx-x,my-y);if(d<nd){nd=d;nearest=p;}});
    if(nearest&&nd<25){tip.style.display='block';tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY-70)+'px';document.getElementById('bt-id').textContent=nearest.id;document.getElementById('bt-qw').textContent=nearest.qw_conc!=null?(nearest.qw_conc*100).toFixed(0)+'%':'—';document.getElementById('bt-n').textContent=nearest.n_informative;document.getElementById('bt-tier').textContent=TL[nearest.tier]||nearest.tier;document.getElementById('bt-p').textContent=nearest.p_val!=null?(nearest.p_val<0.001?nearest.p_val.toExponential(1):nearest.p_val.toFixed(3)):'—';c.style.cursor='pointer';}
    else{tip.style.display='none';c.style.cursor='crosshair';}
  });
  c.addEventListener('mouseleave',()=>tip.style.display='none');
  c.addEventListener('click',e=>{
    const rect=c.getBoundingClientRect();const mx=(e.clientX-rect.left)*(c.width/rect.width/(window.devicePixelRatio||1));const my=(e.clientY-rect.top)*(c.height/rect.height/(window.devicePixelRatio||1));
    let nearest=null,nd=999;
    preds.forEach(p=>{const x=pad.l+(p.n_informative/maxN)*(W-pad.l-pad.r);const y=pad.t+(1-p.qw_conc)*(H-pad.t-pad.b);const d=Math.hypot(mx-x,my-y);if(d<nd){nd=d;nearest=p;}});
    if(nearest&&nd<25){nav('predictions');setTimeout(()=>{document.getElementById('psearch').value=nearest.id;filterPreds();},80);}
  });
}

function drawHBar(ctx,W,H,items,cols,suffix=''){
  const pad={l:95,r:35,t:8,b:8};const bH=Math.floor((H-pad.t-pad.b)/items.length)-5;const max=Math.max(...items.map(([,v])=>v));
  animRun(850,prog=>{ctx.clearRect(0,0,W,H);items.forEach(([k,v],i)=>{const y=pad.t+i*(bH+5);const bW=(v/max)*(W-pad.l-pad.r)*prog;ctx.fillStyle='#E2E8F0';fillRR(ctx,pad.l,y,W-pad.l-pad.r,bH,4);if(bW>0){ctx.fillStyle=cols[i]||'#9a6450';fillRR(ctx,pad.l,y,bW,bH,4);}ctx.font=`10px 'Inter',sans-serif`;ctx.fillStyle='#334155';ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(k,pad.l-6,y+bH/2);ctx.fillStyle='#64748B';ctx.textAlign='left';ctx.fillText(v+suffix,pad.l+bW+5,y+bH/2);});});
}
function drawNovelty(){const r=makeCtx('cv-novelty',180);if(!r)return;const{ctx,W,H}=r;const c={};DATA.per_pred.forEach(p=>{const k=p.novelty||'unknown';c[k]=(c[k]||0)+1;});drawHBar(ctx,W,H,Object.entries(c).sort((a,b)=>b[1]-a[1]),['#b5601a','#1a6b5a','#2a3060','#9a6450']);}
function drawDirection(){const r=makeCtx('cv-dir',180);if(!r)return;const{ctx,W,H}=r;const c={};DATA.per_pred.forEach(p=>{const k=p.direction||'unspecified';c[k]=(c[k]||0)+1;});drawHBar(ctx,W,H,Object.entries(c).sort((a,b)=>b[1]-a[1]),[TC.STRONG,TC.WEAK_DISCORDANT,TC.DESCRIPTIVE,TC.MODERATE,TC.NONE]);}
function drawLOObyTier(){
  const r=makeCtx('cv-loo-tier',180);if(!r)return;const{ctx,W,H}=r;
  const data=[];TIERS.forEach(t=>{const pp=DATA.per_pred.filter(p=>p.tier===t&&p.loo_delta!=null);if(!pp.length)return;const frag=pp.filter(p=>p.loo_delta>=0.1).length;data.push([t,Math.round((frag/pp.length)*100)]);});
  data.sort((a,b)=>b[1]-a[1]);drawHBar(ctx,W,H,data.map(([t,v])=>[TL[t],v]),data.map(([t])=>TC[t]),'%');
}

// ── Categories ─────────────────────────────────────────────
function buildCategories(){
  const grid=document.getElementById('cat-grid');
  [...DATA.categories].sort((a,b)=>b.pooled-a.pooled).forEach(cat=>{
    const pct=cat.pooled;const col=pct>=75?TC.STRONG:pct>=60?TC.MODERATE:TC.MIXED;
    const circ=2*Math.PI*28,dash=(pct/100)*circ;
    const d=document.createElement('div');d.className='cat-card reveal';
    d.innerHTML=`<div class="cat-ring"><svg width="72" height="72" viewBox="0 0 72 72"><circle cx="36" cy="36" r="28" fill="none" stroke="#e8e0d0" stroke-width="8"/><circle cx="36" cy="36" r="28" fill="none" stroke="${col}" stroke-width="8" stroke-dasharray="${dash} ${circ}" stroke-dashoffset="${circ/4}" stroke-linecap="round"/></svg><div class="cat-ring-lbl"><span class="cat-pct" style="color:${col}">${pct}%</span></div></div><div><div class="cat-name-text">${cat.category}</div><div class="cat-meta">${cat.n_pred} predictions<br/>${cat.n_inform} informative records</div><div class="cat-p" style="color:${cat.p<0.05?TC.STRONG:'#b5601a'}">p = ${cat.p<0.001?cat.p.toExponential(2):cat.p.toFixed(4)}${cat.p<0.05?' ✓':''}</div></div>`;
    grid.appendChild(d);
  });
  const tbody=document.getElementById('cat-tbody');
  [...DATA.categories].sort((a,b)=>b.pooled-a.pooled).forEach(c=>{
    const col=c.pooled>=75?TC.STRONG:c.pooled>=60?TC.MODERATE:TC.MIXED;
    tbody.innerHTML+=`<tr><td class="mono" style="color:var(--ver);font-weight:500">${c.category}</td><td class="mono">${c.n_pred}</td><td class="mono">${c.n_inform}</td><td class="mono">${c.n_concord}</td><td class="mono" style="color:${col};font-weight:600">${c.pooled}%</td><td class="mono" style="color:var(--mah3)">${c.ci}</td><td class="mono" style="color:${c.p<0.05?TC.STRONG:'#b5601a'}">${c.p<0.001?c.p.toExponential(2):c.p.toFixed(4)}${c.p<0.05?' ✓':''}</td><td><div style="background:var(--s200);border-radius:3px;height:8px;overflow:hidden"><div style="width:${c.pooled}%;height:100%;background:${col};border-radius:3px"></div></div></td></tr>`;
  });
}

// ── Paper evidence card (THE KEY COMPONENT) ────────────────

function srcBadge(src){
  if(src==='full_text')return`<span class="src-badge src-ft">🔓 full text</span>`;
  if(src==='abstract') return`<span class="src-badge src-abs">📄 abstract</span>`;
  return`<span class="src-badge src-ti">title only</span>`;
}

function concIcon(rel){
  if(rel==='✓')return`<svg class="conc-icon" width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="8" fill="#d4ede8" stroke="#1a6b5a" stroke-width="1.5"/><path d="M6 9.5l2.2 2.2L12.5 7" stroke="#1a6b5a" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  if(rel==='✗')return`<svg class="conc-icon" width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="8" fill="#f0d4d8" stroke="#7a1c28" stroke-width="1.5"/><path d="M6.5 6.5l5 5M11.5 6.5l-5 5" stroke="#7a1c28" stroke-width="2" stroke-linecap="round"/></svg>`;
  return`<svg class="conc-icon" width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="8" fill="#e8e0d0" stroke="#9a6450" stroke-width="1.5"/><path d="M6 9h6" stroke="#9a6450" stroke-width="2" stroke-linecap="round"/></svg>`;
}

function buildPaperCard(rec, predId, predExpected){
  const rel   = rec.concordant || rec.relation || '—';
  const cls   = rel==='✓'?'conc':rel==='✗'?'disc':'neut';
  const src   = rec.text_source||'title_only';
  const title = rec.title||`PMID ${rec.pmid}`;
  const kws   = (rec.key_phrases||[]).slice(0,6);
  const evs   = rec.evidence_sentences||[];
  const mech  = rec.mechanism||'';
  const relR  = rec.relevance_reason||'';
  const dir   = rec.direction||'?';
  const dirCol= dir==='up'?TC.STRONG:dir==='down'?TC.WEAK_DISCORDANT:'#9a6450';
  const hasDetail = kws.length||evs.length||mech;

  const kwHtml = kws.length
    ? `<div class="kw-wrap">${kws.map(k=>`<span class="kw-tag">${esc(k)}</span>`).join('')}</div>`
    : '';

  let evHtml='';
  if(evs.length){
    evHtml=`<div class="ev-section"><div class="ev-label">Evidence sentences ${srcBadge(src)}</div>
      ${evs.map(s=>`<div class="ev-sent">${esc(s)}</div>`).join('')}</div>`;
  } else if(hasDetail===0 && src==='title_only'){
    evHtml=`<div class="ev-section"><div class="ev-none">No abstract available. Run stage3 with PMC enrichment for full-text evidence.</div></div>`;
  }

  const mechHtml = mech
    ? `<div class="mech-block"><strong>Mechanism:</strong> ${esc(mech)}</div>`
    : '';
  const relHtml = relR
    ? `<div class="rel-reason">${esc(relR)}</div>`
    : '';

  const chevron = hasDetail
    ? `<svg class="pcard-chev" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6l4 4 4-4"/></svg>`
    : '';

  return `<div class="pcard ${cls}">
    <div class="pcard-head" onclick="togglePcard(this.parentElement,${!!hasDetail})">
      ${concIcon(rel)}
      <div class="pcard-info">
        <div class="pcard-title">
          <a href="https://pubmed.ncbi.nlm.nih.gov/${esc(rec.pmid)}/" target="_blank" onclick="event.stopPropagation()">${esc(title)}</a>
        </div>
        <div class="pcard-meta">
          <span>PMID ${esc(rec.pmid)}</span>
          <span>·</span><span>${rec.year||''}</span>
          <span>·</span><span style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(rec.journal||'')}">${esc(rec.journal||'')}</span>
          <span>·</span><span style="color:${dirCol};font-weight:500">${esc(dir)}</span>
          <span>·</span><span>q=${rec.quality}</span>
          <span>·</span>${srcBadge(src)}
        </div>
      </div>
      ${chevron}
    </div>
    ${hasDetail?`<div class="pcard-body"><div class="pcard-inner">${kwHtml}${evHtml}${mechHtml}${relHtml}</div></div>`:''}
  </div>`;
}

function togglePcard(card,hasDetail){if(!hasDetail)return;card.classList.toggle('open');}

function buildPapersSection(records, predId, predExpected){
  if(!records||!records.length) return '';
  const conc = records.filter(r=>r.concordant==='✓');
  const disc = records.filter(r=>r.concordant==='✗');
  const neut = records.filter(r=>r.concordant!=='✓'&&r.concordant!=='✗');
  let html='<div class="papers-section"><div class="papers-head">Supporting literature</div><div class="paper-groups">';
  if(conc.length) html+=`<div><div class="pgrp-head"><div class="pgrp-dot" style="background:${TC.STRONG}"></div>Concordant (${conc.length})</div><div class="paper-cards">${conc.map(r=>buildPaperCard(r,predId,predExpected)).join('')}</div></div>`;
  if(disc.length) html+=`<div><div class="pgrp-head"><div class="pgrp-dot" style="background:${TC.WEAK_DISCORDANT}"></div>Discordant (${disc.length})</div><div class="paper-cards">${disc.map(r=>buildPaperCard(r,predId,predExpected)).join('')}</div></div>`;
  if(neut.length) html+=`<div><div class="pgrp-head"><div class="pgrp-dot" style="background:${TC.NONE}"></div>Neutral / Descriptive (${neut.length})</div><div class="paper-cards">${neut.map(r=>buildPaperCard(r,predId,predExpected)).join('')}</div></div>`;
  return html+'</div></div>';
}

// ── Prediction browser ─────────────────────────────────────
let activeTier='',viewMode='card';

function buildTierPills(){
  const container=document.getElementById('tier-pills');
  const allBtn=document.createElement('button');
  allBtn.className='tpill active';allBtn.textContent='All';allBtn.style.color='var(--mah)';allBtn.style.borderColor='var(--s300)';
  allBtn.onclick=()=>{activeTier='';container.querySelectorAll('.tpill').forEach(b=>b.classList.remove('active'));allBtn.classList.add('active');filterPreds();};
  container.appendChild(allBtn);
  TIERS.forEach(t=>{
    const n=DATA.per_pred.filter(p=>p.tier===t).length;if(!n)return;
    const btn=document.createElement('button');btn.className='tpill';btn.textContent=`${TL[t]} (${n})`;btn.style.color=TC[t];btn.style.borderColor=TC[t]+'66';
    btn.onclick=()=>{activeTier=activeTier===t?'':t;container.querySelectorAll('.tpill').forEach(b=>b.classList.remove('active'));if(activeTier)btn.classList.add('active');else allBtn.classList.add('active');filterPreds();};
    container.appendChild(btn);
  });
}

function setView(mode,btn){viewMode=mode;document.querySelectorAll('.vbtn').forEach(b=>b.classList.remove('active'));btn.classList.add('active');document.getElementById('pred-list').style.display=mode==='card'?'flex':'none';document.getElementById('pred-tbl-wrap').style.display=mode==='table'?'block':'none';filterPreds();}

function filterPreds(){
  const q=(document.getElementById('psearch').value||'').toLowerCase();
  const filtered=DATA.per_pred.filter(p=>{if(activeTier&&p.tier!==activeTier)return false;if(q&&!`${p.id} ${p.label} ${p.prediction_note||''} ${p.novelty||''}`.toLowerCase().includes(q))return false;return true;});
  document.getElementById('pred-count').textContent=`Showing ${filtered.length} of ${DATA.per_pred.length}`;
  if(viewMode==='card')renderCards(filtered);else renderTable(filtered);
}

function renderCards(preds){
  const list=document.getElementById('pred-list');list.innerHTML='';
  preds.forEach(p=>{
    const qwPct=p.qw_conc!=null?Math.round(p.qw_conc*100):null;
    const concCol=!qwPct?'#9a6450':qwPct>=75?TC.STRONG:qwPct>=60?TC.MODERATE:TC.MIXED;
    const circ=2*Math.PI*20,dash=qwPct?(qwPct/100)*circ:0;
    let chips='';
    if(p.novelty)chips+=`<span class="chip">${p.novelty}</span>`;
    if(p.n_informative)chips+=`<span class="chip">n = ${p.n_informative}</span>`;
    if(p.p_val!=null)chips+=`<span class="chip${p.p_val<0.05?' sig':''}">${p.p_val<0.001?p.p_val.toExponential(1):p.p_val.toFixed(3)}</span>`;
    if(p.fc_median)chips+=`<span class="chip">FC × ${p.fc_median}</span>`;
    if(p.loo_delta!=null&&p.loo_delta>=0.1)chips+=`<span class="chip warn">LOO fragile</span>`;
    const kvs=[];
    if(p.loo_delta!=null)kvs.push(['LOO Δ',p.loo_delta.toFixed(3)+(p.loo_delta>=0.1?' ⚠':'')]);
    if(p.fc_median)kvs.push(['Fold-change',p.fc_median+'×']);
    if(p.simple_conc!=null)kvs.push(['Simple conc.',(p.simple_conc*100).toFixed(0)+'%']);
    if(p.n_opposite)kvs.push(['N opposing',p.n_opposite]);
    if(p.direction)kvs.push(['Expected dir.',p.direction]);
    const papersHtml=buildPapersSection(p.records,p.id,p.direction);
    const card=document.createElement('div');card.className='pc';card.style.setProperty('--plc',TC[p.tier]||'#9a6450');
    card.innerHTML=`
      <div class="pc-head" onclick="this.parentElement.classList.toggle('open')">
        <div class="ring-wrap">
          <svg width="52" height="52" viewBox="0 0 52 52">
            <circle cx="26" cy="26" r="20" fill="none" stroke="#e8e0d0" stroke-width="5"/>
            ${qwPct?`<circle cx="26" cy="26" r="20" fill="none" stroke="${concCol}" stroke-width="5" stroke-dasharray="${dash} ${circ}" stroke-dashoffset="${circ/4}" stroke-linecap="round"/>`:''}
          </svg>
          <div class="ring-lbl" style="color:${concCol}">${qwPct!=null?qwPct+'%':'—'}</div>
        </div>
        <div class="pc-info">
          <div class="pc-id">${esc(p.id)}</div>
          <div class="pc-ttl">${esc(p.label)}</div>
          <div class="pc-chips">${chips}</div>
        </div>
        <span class="tier ${p.tier}">${TL[p.tier]||p.tier}</span>
        <svg class="pc-chev" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 6l4 4 4-4"/></svg>
      </div>
      <div class="pc-note">${p.prediction_note?`<em>"${esc(p.prediction_note)}"</em>`:'<em>No prediction note.</em>'}</div>
      <div class="pc-body"><div class="pc-inner">
        ${kvs.length?`<div class="kv-row">${kvs.map(([k,v])=>`<div class="kv"><div class="kv-k">${k}</div><div class="kv-v">${v}</div></div>`).join('')}</div>`:''}
        ${p.reason?`<div style="font-size:.9rem;color:var(--mah3);padding:.65rem .9rem;background:var(--s100);border-radius:var(--r);border-left:3px solid var(--s300);margin-bottom:1rem"><span style="font-family:var(--ff-m);font-size:.6rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mah4)">Tier reason</span><br/>${esc(p.reason)}</div>`:''}
        ${papersHtml}
      </div></div>`;
    list.appendChild(card);
  });
}

function renderTable(preds){
  const tbody=document.getElementById('pred-tbl-body');tbody.innerHTML='';
  preds.forEach(p=>{const qwPct=p.qw_conc!=null?(p.qw_conc*100).toFixed(0)+'%':'—';const col=TC[p.tier]||'#9a6450';tbody.innerHTML+=`<tr style="cursor:pointer" onclick="openPred('${esc(p.id)}')"><td class="mono" style="color:var(--ver)">${esc(p.id)}</td><td style="font-size:.9rem;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(p.label)}</td><td><span class="tier ${p.tier}">${TL[p.tier]}</span></td><td><div style="display:flex;align-items:center;gap:.5rem"><div style="width:70px;height:6px;background:var(--s200);border-radius:3px;overflow:hidden"><div style="width:${p.qw_conc!=null?(p.qw_conc*100):0}%;height:100%;background:${col};border-radius:3px"></div></div><span class="mono">${qwPct}</span></div></td><td class="mono">${p.n_informative||'—'}</td><td class="mono" style="color:${p.p_val!=null&&p.p_val<0.05?TC.STRONG:'inherit'}">${p.p_val!=null?(p.p_val<0.001?p.p_val.toExponential(1):p.p_val.toFixed(3)):'—'}</td><td class="mono">${p.novelty||'—'}</td><td class="mono">${p.direction||'—'}</td></tr>`;});
}

function openPred(id){setView('card',document.querySelectorAll('.vbtn')[0]);nav('predictions');setTimeout(()=>{document.getElementById('psearch').value=id;filterPreds();const c=document.querySelector('.pc');if(c)c.classList.add('open');},80);}

// ── Strong ─────────────────────────────────────────────────
function buildStrong(){
  const grid=document.getElementById('strong-grid');
  [...DATA.strong_preds].sort((a,b)=>b.qw_conc-a.qw_conc).forEach((s,i)=>{
    const pct=(s.qw_conc*100).toFixed(0);const circ=2*Math.PI*24,dash=s.qw_conc*circ;
    const d=document.createElement('div');d.className='sc reveal';
    d.innerHTML=`<div><svg width="60" height="60" viewBox="0 0 60 60"><circle cx="30" cy="30" r="24" fill="none" stroke="#e8e0d0" stroke-width="7"/><circle cx="30" cy="30" r="24" fill="none" stroke="${TC.STRONG}" stroke-width="7" stroke-dasharray="${dash} ${circ}" stroke-dashoffset="${circ/4}" stroke-linecap="round" style="transition:stroke-dasharray 1s ease"/><text x="30" y="34" text-anchor="middle" font-family="'Fira Code'" font-size="10" fill="${TC.STRONG}" font-weight="500">${pct}%</text></svg></div>
    <div class="sc-info"><div class="sc-id" onclick="openPred('${esc(s.id)}')">${esc(s.id)}</div><div class="sc-label">${esc(s.label)}</div>
    <div class="sc-bar-wrap"><div class="sc-bar-head"><span>QW Concordance</span><span style="color:${TC.STRONG};font-weight:600">${pct}%</span></div><div class="sc-track"><div class="sc-fill" style="width:${pct}%;transition:width 1.${i}s cubic-bezier(.4,0,.2,1)"></div></div></div>
    <div class="sc-stats"><span class="sc-stat">N informative: <b>${s.n_inform}</b></span><span class="sc-stat">p = <b style="color:${s.p<0.05?TC.STRONG:'#b5601a'}">${s.p<0.001?s.p.toExponential(2):s.p.toFixed(4)}</b></span></div></div>`;
    grid.appendChild(d);
  });
}

// ── Priority ───────────────────────────────────────────────
function buildPriority(){
  const c=document.getElementById('pri-container');
  const LEFT={NONE:'#9a6450',NO_DIRECTIONAL:TC.NO_DIRECTIONAL,WEAK_DISCORDANT:TC.WEAK_DISCORDANT,WEAK_SUPPORT:TC.WEAK_SUPPORT};
  ['NONE','NO_DIRECTIONAL','WEAK_DISCORDANT','WEAK_SUPPORT'].forEach(tier=>{
    const items=DATA.exp_preds.filter(p=>p.tier===tier);if(!items.length)return;
    const sec=document.createElement('div');sec.className='pri-section';
    sec.innerHTML=`<div class="pri-sec-head" style="--plc:${LEFT[tier]}">${TL[tier]} — ${items.length} target${items.length>1?'s':''}</div>
      <div class="pri-items">${items.map(p=>`<div class="pi" style="--plc:${LEFT[tier]}" onclick="openPred('${esc(p.id)}')"><div class="pi-info"><div class="pi-id">${esc(p.id)}</div><div class="pi-lbl">${esc(p.label)}</div></div><div style="text-align:right;flex-shrink:0"><span class="tier ${p.tier}">${TL[p.tier]}</span><div style="font-family:var(--ff-m);font-size:.6rem;color:var(--mah4);margin-top:.3rem">n_relevant = ${p.n_relevant}</div></div></div>`).join('')}</div>`;
    c.appendChild(sec);
  });
}

// ── Discordance ────────────────────────────────────────────
function buildDiscordance(){
  const preds=DATA.per_pred.filter(p=>p.n_opposite&&p.n_opposite>0).sort((a,b)=>b.n_opposite-a.n_opposite);
  document.getElementById('b-disc').textContent=preds.length;
  const list=document.getElementById('dc-list');
  if(!preds.length){list.innerHTML='<p style="color:var(--mah3)">No discordant predictions found.</p>';return;}
  preds.forEach(p=>{
    const conc=p.simple_conc!=null?(p.simple_conc*100).toFixed(0)+'%':'—';
    const opp=p.records.filter(r=>r.concordant==='✗');
    const div=document.createElement('div');div.className='dc';
    div.innerHTML=`
      <div class="dc-head">
        <span class="dc-id" onclick="openPred('${esc(p.id)}')">${esc(p.id)}</span>
        <span class="tier WEAK_DISCORDANT">${TL[p.tier]||p.tier}</span>
        <span class="mono" style="color:var(--mah2)">concordance: <b>${conc}</b></span>
        <span class="mono" style="color:var(--cri)">opposing: <b>${p.n_opposite}</b></span>
      </div>
      <div class="dc-body">
        ${p.prediction_note?`<div class="dc-finding">"${esc(p.prediction_note)}"</div>`:''}
        ${opp.length
          ? `<div class="paper-cards">${opp.map(rec=>buildPaperCard(rec,p.id,p.direction)).join('')}</div>`
          : '<p style="font-size:.85rem;color:var(--mah4)">No opposing records in display set — run with --scored for full data.</p>'}
      </div>`;
    list.appendChild(div);
  });
}

// ── LOO ────────────────────────────────────────────────────
function buildLOO(){
  const all=DATA.per_pred.filter(p=>p.loo_delta!=null).sort((a,b)=>b.loo_delta-a.loo_delta);
  document.getElementById('b-loo').textContent=all.length;
  const top=all.slice(0,25);
  const r=makeCtx('cv-loo-bar',300);if(r&&top.length>0){
    const{ctx,W,H}=r;const pad={l:200,r:75,t:8,b:8};
    const bH=Math.max(4,Math.floor((H-pad.t-pad.b)/top.length)-3);
    const maxD=Math.max(...top.map(p=>p.loo_delta),0.001);
    animRun(1000,prog=>{ctx.clearRect(0,0,W,H);top.forEach((p,i)=>{
      const y=pad.t+i*(bH+3);const fragile=p.loo_delta>=0.1;
      const col=fragile?TC.LOO_FRAGILE:(TC[p.tier]||TC.NONE);
      const bW=(p.loo_delta/maxD)*(W-pad.l-pad.r)*prog;
      ctx.fillStyle='#E2E8F0';fillRR(ctx,pad.l,y,W-pad.l-pad.r,bH,3);
      if(bW>0){ctx.fillStyle=col+'cc';fillRR(ctx,pad.l,y,bW,bH,3);}
      const id=p.id.length>30?p.id.slice(0,28)+'…':p.id;
      ctx.font=`500 10px 'Inter',sans-serif`;ctx.fillStyle='#334155';
      ctx.textAlign='right';ctx.textBaseline='middle';ctx.fillText(id,pad.l-6,y+bH/2);
      ctx.fillStyle=fragile?TC.LOO_FRAGILE:'#94A3B8';ctx.textAlign='left';
      ctx.fillText('Δ'+p.loo_delta.toFixed(3)+(fragile?' ⚠':''),pad.l+bW+5,y+bH/2);
    });});
  }
  const grid=document.getElementById('loo-grid');
  all.forEach(p=>{
    const fragile=p.loo_delta>=0.1;
    const col=TC[p.tier]||TC.NONE;
    const low=p.loo_low!=null?(p.loo_low*100).toFixed(1):'0';
    const wid=p.loo_high!=null&&p.loo_low!=null?((p.loo_high-p.loo_low)*100).toFixed(1):'0';
    const row=document.createElement('div');
    row.className='loo-row';row.onclick=()=>openPred(p.id);
    row.innerHTML=`<div class="loo-id" title="${esc(p.id)}">${esc(p.id)}</div>`+
      `<div class="loo-track"><div class="loo-fill" style="left:${low}%;width:${wid}%;background:${col}"></div></div>`+
      `<div class="loo-delta" style="color:${fragile?TC.LOO_FRAGILE:'var(--mah3)'}">Δ${(p.loo_delta||0).toFixed(3)}</div>`+
      `<div class="loo-flag">${fragile?'⚠ fragile':''}</div>`;
    grid.appendChild(row);
    // Show up to 3 impactful papers under fragile predictions
    const papers = p.loo_papers || [];
    if(fragile && papers.length > 0){
      const wrap=document.createElement('div');
      wrap.className='loo-papers';
      wrap.innerHTML=`<span class="loo-papers-lbl">Papers driving sensitivity (Δ ≥ 0.10):</span>`+
        papers.map(pa=>{
          const rel=pa.relation||'';
          const title=(pa.title||'No title').slice(0,70)+(((pa.title||'').length>70)?'…':'');
          const pmidLink=pa.pmid?
            `<a class="loo-pmid" href="https://pubmed.ncbi.nlm.nih.gov/${pa.pmid}/" target="_blank">PMID ${pa.pmid}</a>`:
            `<span class="loo-pmid">—</span>`;
          return `<div class="loo-paper">`+
            pmidLink+
            `<span class="loo-ptitle" title="${esc(pa.title||'')}">${esc(title)}</span>`+
            `<span class="loo-prel ${rel}">${rel||'—'}</span>`+
            `<span class="loo-pdelta">Δ${pa.delta.toFixed(3)}</span>`+
            `</div>`;
        }).join('');
      grid.appendChild(wrap);
    }
  });
}

// ── Utility ────────────────────────────────────────────────
function esc(s){const d=document.createElement('div');d.textContent=String(s||'');return d.innerHTML;}

// ── Init ───────────────────────────────────────────────────
document.getElementById('b-ov').textContent=DATA.per_pred.length;
document.getElementById('b-cat').textContent=DATA.categories.length;
document.getElementById('b-pred').textContent=DATA.per_pred.length;
document.getElementById('b-str').textContent=DATA.strong_preds.length;
document.getElementById('b-pri').textContent=DATA.exp_preds.length;
document.getElementById('pg-desc').textContent=`${DATA.per_pred.length} predictions · ${DATA.categories.length} categories · click any paper to view evidence`;

// ── Concordance summary bar ─────────────────────────────────────────────
function buildConcSummary(){
  const CONC_TIERS  = new Set(['STRONG','MODERATE','WEAK_SUPPORT']);
  const DISC_TIERS  = new Set(['DISCORDANT','WEAK_DISCORDANT']);
  const NO_TIERS    = new Set(['NONE']);
  const preds = DATA.per_pred || [];
  const total = preds.length || 1;
  // Separate: directional concordant vs association (volume-based) vs discordant vs no evidence
  const nConc  = preds.filter(p=>CONC_TIERS.has(p.tier) && p.scoring_mode !== 'association').length;
  const nAssoc = preds.filter(p=>
    p.scoring_mode === 'association' ||
    new Set(['DESCRIPTIVE','NO_DIRECTIONAL','NO_INFORMATIVE']).has(p.tier)
  ).length;
  const nDisc  = preds.filter(p=>DISC_TIERS.has(p.tier)).length;
  const nNone  = total - nConc - nAssoc - nDisc;
  const pct = n => Math.round(n/total*100)+'%';
  document.getElementById('csb-concordant').textContent = pct(nConc);
  document.getElementById('csb-discordant').textContent = pct(nDisc);
  document.getElementById('csb-assoc').textContent      = pct(nAssoc);
  document.getElementById('csb-none').textContent       = pct(nNone);
  // Stacked bar
  const stack = document.getElementById('csb-stack');
  const segs = [
    {n:nConc,  c:'#10B981', l:'Directional concordance'},
    {n:nAssoc, c:'#7C3AED', l:'Association'},
    {n:nDisc,  c:'#EF4444', l:'Discordant'},
    {n:nNone,  c:'#CBD5E1', l:'No evidence'},
  ];
  stack.innerHTML = segs.map(s=>`<div class="csb-seg" style="width:${Math.max(0,s.n/total*100)}%;background:${s.c}"></div>`).join('');
  // Legend
  const legend = document.getElementById('csb-legend');
  legend.innerHTML = segs.filter(s=>s.n>0).map(s=>
    `<span class="csb-leg"><span class="csb-dot" style="background:${s.c}"></span>${s.n} ${s.l}</span>`
  ).join('');
}

// ── Download helpers ─────────────────────────────────────────────────────
function _dlBlob(content, filename, type){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([content],{type}));
  a.download=filename; document.body.appendChild(a); a.click();
  setTimeout(()=>document.body.removeChild(a),200);
}
function dlCSV(){
  const rows=[['prediction_id','entity','disease','cell_type','direction','tier',
                'pmid','title','year','extracted_direction','concordant','relevance']];
  (DATA.per_pred||[]).forEach(p=>{
    const en=(p.label||'').split('(')[0].trim();
    const ctx=(p.label||'').match(/\(([^)]+)\)/);
    const [dis,ct]=(ctx?ctx[1]:'').split(',').map(s=>s.trim());
    (p.records||[]).forEach(r=>{
      rows.push([p.id,en,dis||'',ct||'',p.direction||'',p.tier||'',
        r.pmid||'',String(r.title||'').replace(/"/g,"'"),r.year||'',
        r.extracted_direction||'',r.concordant||'',
        (r.relevance||0).toFixed?Number(r.relevance||0).toFixed(3):''
      ]);
    });
    if(!(p.records||[]).length)
      rows.push([p.id,en,dis||'',ct||'',p.direction||'',p.tier||'','','','','','','']);
  });
  _dlBlob(rows.map(r=>r.map(v=>`"${v}"`).join(',')).join('\n'),
          `papertrail-papers-${Date.now()}.csv`,'text/csv');
}
function dlJSON(){
  const out={per_pred:(DATA.per_pred||[]).map(p=>({
    id:p.id,tier:p.tier,direction:p.direction,novelty:p.novelty,
    n_concordant:p.n_concordant,n_opposite:p.n_opposite,
    n_informative:p.n_informative,n_relevant:p.n_relevant,
    concordance_score:p.qw_conc,loo_delta:p.loo_delta,
    reason:p.reason
  }))};
  _dlBlob(JSON.stringify(out,null,2),`papertrail-summary-${Date.now()}.json`,'application/json');
}
function dlCharts(){
  const names={
    'cv-donut':'tier-distribution','cv-bubble':'concordance-scatter',
    'cv-catbar':'category-concordance','cv-loo-bar':'loo-sensitivity','cv-loo-tier':'loo-by-tier',
  };
  Object.entries(names).forEach(([id,name])=>{
    const cv=document.getElementById(id);
    if(!cv||!cv.width)return;
    // Composite onto white background at the canvas's native (DPR-scaled) resolution
    const out=document.createElement('canvas');
    out.width=cv.width; out.height=cv.height;
    const ctx=out.getContext('2d');
    ctx.fillStyle='#FFFFFF'; ctx.fillRect(0,0,out.width,out.height);
    ctx.drawImage(cv,0,0);
    const a=document.createElement('a');
    a.href=out.toDataURL('image/png',1.0);
    a.download=`papertrail-${name}-${Date.now()}.png`;
    document.body.appendChild(a); a.click();
    setTimeout(()=>document.body.removeChild(a),300);
  });
}
function dlHTML(){
  _dlBlob(document.documentElement.outerHTML,
          `papertrail-dashboard-${Date.now()}.html`,'text/html');
}

buildConcSummary();
buildOverview();buildCategories();buildTierPills();
renderCards(DATA.per_pred);document.getElementById('pred-count').textContent=`Showing ${DATA.per_pred.length} of ${DATA.per_pred.length}`;
buildStrong();buildPriority();buildDiscordance();buildLOO();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(description='PaperTrail — Literature Concordance Dashboard')
    ap.add_argument('--report',       required=True,  help='Path to report.md (always required)')
    ap.add_argument('--scored',       default=None,   help='Path to scored_predictions.json (strongly recommended)')
    ap.add_argument('--cache',        default=None,   help='Path to cache/pubmed_records/ directory')
    ap.add_argument('--out',          default=None,   help='Output HTML path (default: dashboard.html next to report)')
    ap.add_argument('--project-name', default='',     help='Project name shown in the dashboard header')
    args = ap.parse_args()

    report_path = Path(args.report)
    out_path    = Path(args.out) if args.out else report_path.parent / 'dashboard.html'

    print(f"\nPaperTrail Dashboard Builder")
    print(f"{'─'*44}")

    # ── Step 1: Parse report.md for summary/tier/LOO metadata ──
    print(f"\n1. Parsing {report_path}…")
    md   = report_path.read_text(encoding='utf-8')
    data = parse_report(md)
    pp   = data['per_pred']
    total_fallback_recs = sum(len(p['records']) for p in pp)
    print(f"   {len(pp)} predictions · {len(data['categories'])} categories · "
          f"{len(data['strong_preds'])} strong · {len(data['exp_preds'])} priority")
    print(f"   {total_fallback_recs} fallback records from report tables")

    # ── Step 2: Load & enrich from pipeline JSON (full evidence) ─
    if args.scored:
        scored_path = Path(args.scored)
        cache_dir   = Path(args.cache) if args.cache else scored_path.parent / 'cache' / 'pubmed_records'
        print(f"\n2. Loading pipeline data…")
        print(f"   scored_predictions: {scored_path}")
        print(f"   pubmed cache:       {cache_dir}")
        papers = load_from_pipeline(scored_path, cache_dir)
        n_ft   = sum(1 for p in papers if p.get('text_source') == 'full_text')
        n_ab   = sum(1 for p in papers if p.get('text_source') == 'abstract')
        n_ti   = sum(1 for p in papers if p.get('text_source') == 'title_only')
        print(f"   {len(papers)} papers: 🔓{n_ft} full text  📄{n_ab} abstract  {n_ti} title only")
        if n_ti:
            print(f"   ℹ  {n_ti} title-only: run updated stage3 for PMC full text")

        print(f"\n3. Extracting evidence sentences (YAKE + TF-IDF)…")
        enriched = enrich_papers(papers)

        print(f"\n4. Merging evidence into dashboard…")
        data = merge_evidence(data, enriched)
        total_merged = sum(len(p['records']) for p in data['per_pred'])
        n_with_ev    = sum(1 for p in data['per_pred']
                          for r in p['records'] if r.get('evidence_sentences'))
        print(f"   {total_merged} total records · {n_with_ev} with evidence sentences")
    else:
        print(f"\n2. No --scored provided — using report.md tables only.")
        print(f"   ⚠  Most concordant papers will have title-only (no evidence sentences).")
        print(f"   For full evidence: add --scored scored_predictions.json --cache cache/pubmed_records/")

    # ── Step 3: Write HTML ──────────────────────────────────────
    print(f"\n5. Writing dashboard…")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    project_name = getattr(args, 'project_name', '') or ''
    data['project_name'] = project_name
    meta = json.dumps({'project_name': project_name, 'is_demo': False}, ensure_ascii=False)
    html = (HTML
            .replace('__DATA__', json.dumps(data, ensure_ascii=False))
            .replace('__META__', meta))
    out_path.write_text(html, encoding='utf-8')
    size_kb = out_path.stat().st_size // 1024
    print(f"\n{'─'*44}")
    print(f"✓  {out_path}  ({size_kb} KB)")
    print(f"\nRun:")
    print(f"  python3 {Path(__file__).name} \\")
    print(f"      --report  {report_path} \\")
    print(f"      --scored  scored_predictions.json \\")
    print(f"      --cache   cache/pubmed_records/")


if __name__ == '__main__':
    main()
