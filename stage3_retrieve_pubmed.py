"""
Stage 3 (v2): Retrieve PubMed records via E-utilities.

Workflow:
1. ESearch  — get PMIDs matching each query (capped at per-query retmax or
              MAX_PMIDS_PER_QUERY)
2. EFetch   — get titles, abstracts, MeSH terms, journals, dates, authors
3. Cache everything to disk (one JSON per PMID) so re-runs are free
4. PMC full-text enrichment [NEW] — stream-and-extract, zero storage overhead:
   For each PMID that has an open-access PMC counterpart:
     a. Convert PMID → PMCID via ELink
     b. Fetch full-text XML from PMC (streamed, never written to disk)
     c. Parse in-memory: walk Results, Discussion, Conclusion, Abstract
        sections ONLY (Methods/References/Supplementary are skipped)
     d. Extract sentences that contain any alias mention, plus up to
        PMC_CONTEXT_WINDOW surrounding sentences for context
     e. Store only those sentences (~3–15 per paper) in the cache JSON
        under "pmc_sentences" — typically <2 KB vs ~500 KB for full text
   Papers without open-access PMC records get pmc_sentences=[].

New fields added to each cache/pubmed_records/<pmid>.json:
  "pmc_id"       : str | null   — e.g. "PMC7654321", null if not in PMC
  "pmc_sentences": [str, ...]   — alias-relevant sentences from full text
  "pmc_sections" : [str, ...]   — section names sentences came from

Run with --skip-pmc to skip step 4 (faster, abstract-only, original behaviour).

Uses Biopython's Entrez with NCBI rate-limit-friendly settings.
"""

from __future__ import annotations

import json
import os
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

import yaml
from Bio import Entrez
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────
CACHE_DIR   = Path(__file__).parent / "cache"
RECORDS_DIR = CACHE_DIR / "pubmed_records"
RECORDS_DIR.mkdir(exist_ok=True, parents=True)

# ── NCBI credentials ───────────────────────────────────────────────────────
Entrez.email = os.environ.get("NCBI_EMAIL", "litrev-pipeline@example.com")
Entrez.tool  = "litrev-pipeline/1.0"
# Defensive reset: if a previous module set api_key to "" (empty string),
# Biopython sends api_key= in every URL → NCBI HTTP 400.
# Only set api_key when it has a real value.
_api_key = os.environ.get("NCBI_API_KEY", "")
if _api_key:
    Entrez.api_key = _api_key
elif Entrez.api_key == "":
    Entrez.api_key = None   # clear any empty string left by earlier modules

# ── Rate limits ────────────────────────────────────────────────────────────
MAX_PMIDS_PER_QUERY = 80        # default cap per query; lowered to control PMC enrichment volume
SLEEP_BETWEEN_CALLS = 0.34      # 3 req/s without API key; 10/s with key
RETRY_LIMIT         = 3

# ── PMC enrichment settings ────────────────────────────────────────────────
PMC_CONTEXT_WINDOW = 1          # sentences before/after alias hit to include
PMC_MAX_SENTENCES  = 30         # hard cap on stored sentences per paper

# Sections to mine for directional evidence
PMC_SECTIONS_WANTED = {
    "results", "result", "discussion", "conclusions", "conclusion",
    "findings", "abstract",
}
# Sections to skip entirely — too noisy, no directional signal
PMC_SECTIONS_SKIP = {
    "methods", "method", "materials and methods", "materials & methods",
    "statistical analysis", "statistics", "statistical methods",
    "supplementary", "supplemental", "supporting information",
    "acknowledgements", "acknowledgments", "acknowledgement",
    "references", "bibliography",
    "author contributions", "authors contributions",
    "competing interests", "conflict of interest", "conflicts of interest",
    "funding", "financial support", "grant",
    "ethics", "ethical approval", "ethics statement",
    "availability", "data availability", "code availability",
    "abbreviations",
}


# ══════════════════════════════════════════════════════════════════
# RETRY HELPER
# ══════════════════════════════════════════════════════════════════

def _retry(fn, *args, **kwargs):
    """Exponential-backoff retry wrapper."""
    last_err = None
    for attempt in range(RETRY_LIMIT):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


# ══════════════════════════════════════════════════════════════════
# PUBMED SEARCH + FETCH  (original stage3 logic, unchanged)
# ══════════════════════════════════════════════════════════════════

def search_pmids(query: str, retmax: int = MAX_PMIDS_PER_QUERY) -> List[str]:
    """Run ESearch, return PMIDs."""
    def _do():
        h   = Entrez.esearch(db="pubmed", term=query, retmax=retmax, sort="relevance")
        rec = Entrez.read(h)
        h.close()
        return rec["IdList"]
    pmids = _retry(_do)
    time.sleep(SLEEP_BETWEEN_CALLS)
    return pmids


def fetch_records(pmids: Iterable[str]) -> List[Dict[str, Any]]:
    """Fetch full PubMed records (XML), return parsed dicts.
    Cached: any PMID with an existing cache file is loaded from disk.
    """
    pmids = list(pmids)
    cached: Dict[str, Dict[str, Any]] = {}
    to_fetch: List[str] = []
    for pid in pmids:
        path = RECORDS_DIR / f"{pid}.json"
        if path.exists():
            try:
                cached[pid] = json.loads(path.read_text())
            except Exception:
                to_fetch.append(pid)
        elif path.with_suffix(".tmp").exists():
            # Another process is writing this PMID right now — re-fetch
            to_fetch.append(pid)
        else:
            to_fetch.append(pid)

    fetched: List[Dict[str, Any]] = []
    BATCH = 25
    for i in range(0, len(to_fetch), BATCH):
        batch = to_fetch[i:i + BATCH]
        def _do():
            h   = Entrez.efetch(db="pubmed", id=",".join(batch),
                                rettype="xml", retmode="xml")
            rec = Entrez.read(h)
            h.close()
            return rec
        try:
            rec = _retry(_do)
        except Exception as e:
            print(f"  fetch batch failed: {e}")
            continue
        for art in rec.get("PubmedArticle", []):
            parsed = _parse_article(art)
            if parsed.get("pmid"):
                # NOTE: do NOT pre-populate pmc_sentences here.
                # enrich_with_pmc() uses absence of the key to decide
                # which records still need PMC enrichment.
                fetched.append(parsed)
                path = RECORDS_DIR / f"{parsed['pmid']}.json"
                # Atomic write: avoids corruption when two users fetch
                # the same PMID simultaneously (shared cache).
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(parsed, indent=2, default=str))
                tmp.replace(path)
        time.sleep(SLEEP_BETWEEN_CALLS)

    return list(cached.values()) + fetched


def _parse_article(art) -> Dict[str, Any]:
    """Extract fields from a Bio.Entrez PubmedArticle dict."""
    try:
        med     = art["MedlineCitation"]
        pmid    = str(med["PMID"])
        article = med["Article"]
        title   = str(article.get("ArticleTitle", "") or "")

        abstract_parts = []
        ab = article.get("Abstract", {}).get("AbstractText", [])
        if isinstance(ab, list):
            for part in ab:
                if hasattr(part, "attributes"):
                    label = part.attributes.get("Label", "")
                    abstract_parts.append(f"{label}: {str(part)}" if label else str(part))
                else:
                    abstract_parts.append(str(part))
        else:
            abstract_parts.append(str(ab))
        abstract = " ".join(abstract_parts)

        journal = str((article.get("Journal") or {}).get("Title", ""))
        year = None
        try:
            d    = (article.get("Journal", {})
                           .get("JournalIssue", {})
                           .get("PubDate", {}))
            year = int(str(d.get("Year"))) if d.get("Year") else None
        except Exception:
            pass

        mesh_terms = []
        for m in med.get("MeshHeadingList", []):
            try:
                mesh_terms.append(str(m["DescriptorName"]))
            except Exception:
                pass

        pub_types = [str(pt) for pt in (article.get("PublicationTypeList") or [])]
        authors   = []
        for a in article.get("AuthorList", [])[:6]:
            ln   = str(a.get("LastName", ""))
            init = str(a.get("Initials", ""))
            if ln:
                authors.append(f"{ln} {init}".strip())

        return {
            "pmid":              pmid,
            "title":             title,
            "abstract":          abstract,
            "journal":           journal,
            "year":              year,
            "mesh_terms":        mesh_terms,
            "publication_types": pub_types,
            "authors":           authors,
        }
    except Exception as e:
        return {"pmid": None, "error": str(e)}


# ══════════════════════════════════════════════════════════════════
# PMC FULL-TEXT: STREAM-AND-EXTRACT
# ══════════════════════════════════════════════════════════════════

def _pmid_to_pmcid(pmid: str) -> Optional[str]:
    """Single-PMID wrapper — used only as a fallback. Prefer _batch_pmids_to_pmcids."""
    result = _batch_pmids_to_pmcids([pmid])
    return result.get(pmid)


def _batch_pmids_to_pmcids(pmids: List[str],
                            batch_size: int = 100) -> Dict[str, Optional[str]]:
    """
    Convert a list of PMIDs to PMCIDs in batches via ELink.

    Batching is the key performance fix: instead of 1 ELink call per paper
    (N papers = N API round-trips), we send 100 PMIDs per call
    (N papers = N/100 round-trips). For 600 papers this is 6 calls vs 600.

    Returns {pmid: "PMCxxxxxxx" | None}
    """
    result: Dict[str, Optional[str]] = {p: None for p in pmids}
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        id_str = ",".join(batch)
        def _do(ids=id_str):
            h   = Entrez.elink(dbfrom="pubmed", db="pmc", id=ids, retmode="xml")
            rec = Entrez.read(h)
            h.close()
            return rec
        try:
            rec = _retry(_do)
            time.sleep(SLEEP_BETWEEN_CALLS)
            # ELink returns one LinkSet per input PMID when id= is a comma list
            for link_set in rec:
                # The source PMID is in IdList[0]
                src_ids = link_set.get("IdList", [])
                if not src_ids:
                    continue
                src_pmid = str(src_ids[0])
                for db_link in link_set.get("LinkSetDb", []):
                    if db_link.get("DbTo") == "pmc":
                        ids_out = db_link.get("Link", [])
                        if ids_out:
                            result[src_pmid] = "PMC" + str(ids_out[0]["Id"])
                            break
        except Exception as e:
            print(f"  ELink batch failed (PMIDs {batch[0]}–{batch[-1]}): {e}")
    return result


def _split_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter — no NLTK needed."""
    _abbrev = re.compile(
        r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|Fig|et al|e\.g|i\.e|approx|'
        r'Eq|No|vol|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.'
    )
    text  = _abbrev.sub(lambda m: m.group().replace('.', '\x00'), text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\(\[])', text)
    return [p.replace('\x00', '.').strip() for p in parts if len(p.strip()) > 20]


def _get_section_label(elem: ET.Element) -> str:
    """Return normalised section title text from an NXML <sec> element."""
    title_elem = elem.find("title")
    if title_elem is not None:
        return "".join(title_elem.itertext()).strip().lower()
    return ""


def _iter_section_text(root: ET.Element) -> List[Tuple[str, str]]:
    """
    Walk an NXML article tree and return (section_label, paragraph_text)
    tuples for sections we want to mine.

    Skips Methods, References, Supplementary, Acknowledgements etc.
    Descends into Results, Discussion, Conclusion, Abstract (and any
    unlabelled body sections).

    IMPORTANT: The wanted/skip decision is made at each <sec> boundary
    so that nested sub-sections inherit their parent's decision.
    """
    collected: List[Tuple[str, str]] = []

    def _should_skip(label: str) -> bool:
        return any(s in label for s in PMC_SECTIONS_SKIP)

    def _should_include(label: str) -> bool:
        # Include if explicitly wanted, or if unlabelled (might be body text)
        if not label:
            return True
        return any(s in label for s in PMC_SECTIONS_WANTED)

    def _walk(elem: ET.Element, current_label: str, include: bool):
        # Strip XML namespace from tag name
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag

        if tag == "sec":
            label = _get_section_label(elem) or current_label
            # Skip this section and all its children if it's in the skip list
            if _should_skip(label):
                return
            # Decide whether to include paragraphs in this section
            new_include = _should_include(label)
            for child in elem:
                _walk(child, label, new_include)

        elif tag == "p":
            if include:
                text = "".join(elem.itertext()).strip()
                if len(text) > 30:
                    collected.append((current_label, text))
            # Always recurse in case there are nested elements
            for child in elem:
                _walk(child, current_label, include)

        elif tag in ("abstract",):
            # Abstract is always included
            for child in elem:
                _walk(child, "abstract", True)

        else:
            for child in elem:
                _walk(child, current_label, include)

    # Start from root — include by default for top-level unlabelled content
    _walk(root, "", True)
    return collected


def _extract_pmc_sentences(
        pmcid: str,
        aliases: List[str],
        context_window: int = PMC_CONTEXT_WINDOW,
        max_sentences: int  = PMC_MAX_SENTENCES,
) -> Tuple[List[str], List[str]]:
    """
    Fetch PMC full-text XML for pmcid, extract alias-relevant sentences.

    The XML is fetched into memory, parsed, and then discarded — it is
    NEVER written to disk.  Only the extracted sentences are kept.

    Returns:
        sentences      — deduplicated list of relevant sentences with context
        sections_found — which sections the sentences came from
    """
    if not pmcid:
        return [], []

    # Fetch XML into memory
    try:
        def _do():
            numeric_id = pmcid.lstrip("PMCpmc")
            h   = Entrez.efetch(db="pmc", id=numeric_id,
                                rettype="full", retmode="xml")
            raw = h.read()
            h.close()
            return raw
        xml_bytes = _retry(_do)
        time.sleep(SLEEP_BETWEEN_CALLS)
    except Exception:
        return [], []

    # Parse XML — discarded after this function returns
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return [], []

    # Collect (section, paragraph) pairs from wanted sections
    para_pairs = _iter_section_text(root)

    # Build alias regex patterns
    alias_pats: List[re.Pattern] = []
    for a in aliases:
        if not a or len(a) < 2:
            continue
        esc = re.escape(a)
        pat = (re.compile(r'\b' + esc + r'\b', re.IGNORECASE)
               if len(a) >= 4
               else re.compile(esc, re.IGNORECASE))
        alias_pats.append(pat)

    if not alias_pats:
        return [], []

    def _has_alias(text: str) -> bool:
        return any(p.search(text) for p in alias_pats)

    # Find alias-containing sentences + context window
    seen:               set        = set()
    selected_sents:    List[str]  = []
    selected_sections: List[str]  = []

    for sec_label, para_text in para_pairs:
        sents = _split_sentences(para_text)
        for i, sent in enumerate(sents):
            if not _has_alias(sent):
                continue
            # Include sentence + PMC_CONTEXT_WINDOW neighbours
            start = max(0, i - context_window)
            end   = min(len(sents), i + context_window + 1)
            for s in sents[start:end]:
                s_norm = s.strip()
                if s_norm not in seen and len(s_norm) > 25:
                    seen.add(s_norm)
                    selected_sents.append(s_norm)
                    selected_sections.append(sec_label or "body")
                    if len(selected_sents) >= max_sentences:
                        return selected_sents, list(dict.fromkeys(selected_sections))

    return selected_sents, list(dict.fromkeys(selected_sections))


def enrich_with_pmc(
        records: List[Dict[str, Any]],
        aliases_by_pmid: Dict[str, List[str]],
) -> None:
    """
    For each record that has not yet been PMC-enriched, attempt a
    PMID→PMCID lookup + full-text sentence extraction.

    A record is considered unenriched if it does NOT have the key
    "pmc_id" in its dict (regardless of whether pmc_sentences is present).
    Records that already have "pmc_id" are skipped — this means the PMC
    step has already run for them (even if it found nothing).

    Modifies records IN PLACE and updates the on-disk cache file.

    aliases_by_pmid: {pmid: [alias, ...]}
    """
    # Select records that have never been through PMC enrichment
    # (identified by absence of the "pmc_id" key, which we write even
    #  for papers where PMC has no record, to mark them as attempted)
    to_enrich = [
        r for r in records
        if r.get("pmid") and "pmc_id" not in r
    ]

    # Deduplicate by PMID (same paper may appear for multiple predictions)
    seen_pmids: set = set()
    unique_to_enrich: List[Dict[str, Any]] = []
    for r in to_enrich:
        if r["pmid"] not in seen_pmids:
            seen_pmids.add(r["pmid"])
            unique_to_enrich.append(r)

    if not unique_to_enrich:
        print("  PMC enrichment: all records already processed (cache up to date)")
        return

    # Cap to avoid runaway enrichment from large synonym-expanded retrievals.
    # Papers are ordered by relevance (sorted by record dict order from fetch_records).
    # Raising this limit only increases PMC fetch time proportionally.
    PMC_MAX_PAPERS = int(os.environ.get("PMC_MAX_PAPERS", "10000"))
    if len(unique_to_enrich) > PMC_MAX_PAPERS:
        print(f"  PMC enrichment: capping at {PMC_MAX_PAPERS} of {len(unique_to_enrich)} records")
        print(f"  (set PMC_MAX_PAPERS env var to change this limit)")
        unique_to_enrich = unique_to_enrich[:PMC_MAX_PAPERS]

    print(f"\nPMC full-text enrichment: {len(unique_to_enrich)} new records to process...")
    print(f"  Step 1: Batch ELink lookup ({(len(unique_to_enrich)+99)//100} API calls)...")

        # One API call per 100 papers instead of 1 call per paper.
    all_pmids_to_enrich = [r["pmid"] for r in unique_to_enrich]
    pmcid_map = _batch_pmids_to_pmcids(all_pmids_to_enrich)

    n_has_pmc = sum(1 for v in pmcid_map.values() if v)
    print(f"  {n_has_pmc}/{len(all_pmids_to_enrich)} PMIDs have open-access PMC records")
    print(f"  Step 2: Fetching + extracting {n_has_pmc} full-text articles...")

    n_success = n_no_pmc = n_error = 0

    for r in tqdm(unique_to_enrich, desc="PMC enrich"):
        pmid    = r["pmid"]
        aliases = aliases_by_pmid.get(pmid, [])
        pmcid   = pmcid_map.get(pmid)

        r["pmc_id"] = pmcid  # write even if None — marks as attempted

        if not pmcid:
            r["pmc_sentences"] = []
            r["pmc_sections"]  = []
            n_no_pmc += 1
        else:
            # Fetch + extract full text (one call per paper — unavoidable)
            try:
                sents, sections = _extract_pmc_sentences(pmcid, aliases)
                r["pmc_sentences"] = sents
                r["pmc_sections"]  = sections
                if sents:
                    n_success += 1
                else:
                    n_no_pmc += 1
            except Exception:
                r["pmc_sentences"] = []
                r["pmc_sections"]  = []
                n_error += 1

        # Write updated record back to disk cache
        cache_path = RECORDS_DIR / f"{pmid}.json"
        try:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(r, indent=2, default=str))
            tmp.replace(cache_path)
        except Exception:
            pass

    total = len(unique_to_enrich)
    pct   = 100 * n_success / total if total else 0
    print(f"  ✓ {n_success}/{total} papers got PMC full-text sentences ({pct:.0f}% hit rate)")
    print(f"  — {n_no_pmc} no open-access record or no alias hits")
    if n_error:
        print(f"  ✗ {n_error} errors during fetch/parse")


# ══════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════

def retrieve_for_all_queries(
        queries_path: Path,
        output_path:  Path,
        predictions_path: Optional[Path] = None,
        skip_pmc: bool = False,
) -> None:
    """
    Main entry point.  Signature is backward-compatible with original stage3:
    the two new keyword arguments have safe defaults so existing run_pipeline.py
    callers that pass only (queries_path, output_path) will work unchanged.
    """
    with open(queries_path) as f:
        queries_data = yaml.safe_load(f)

    # ── Load alias map (pred_id → aliases) for PMC extraction ──────────────
    # Falls back to an empty dict if no predictions file is given — in that
    # case PMC enrichment will still run but will use an empty alias list,
    # meaning no sentences will be extracted.  Always pass predictions_path.
    alias_by_pred: Dict[str, List[str]] = {}
    if predictions_path and predictions_path.exists():
        with open(predictions_path) as f:
            preds = yaml.safe_load(f)["predictions"]
        for p in preds:
            alias_by_pred[p["id"]] = p.get("aliases", [p.get("entity", p["id"])])

    # ── Group queries by prediction_id ─────────────────────────────────────
    by_pred: Dict[str, Dict[str, Any]] = {}
    for q in queries_data["queries"]:
        by_pred.setdefault(q["prediction_id"], {"queries": [], "pmids_by_query": {}})
        by_pred[q["prediction_id"]]["queries"].append(q)

    print(f"Retrieving for {len(by_pred)} predictions...")
    all_results: Dict[str, Dict[str, Any]] = {}

    for pred_id, qd in tqdm(by_pred.items(), desc="ESearch"):
        pmid_set: set                           = set()
        per_query_pmids: Dict[str, List[str]]   = {}
        for q in qd["queries"]:
            retmax = q.get("retmax") or MAX_PMIDS_PER_QUERY
            try:
                pmids = search_pmids(q["query"], retmax=retmax)
            except Exception as e:
                pmids = []
                print(f"  search failed for {pred_id} / {q['query_name']}: {e}")
            per_query_pmids[q["query_name"]] = pmids
            pmid_set.update(pmids)

        all_results[pred_id] = {
            "queries":          qd["queries"],
            "per_query_pmids":  per_query_pmids,
            "all_pmids":        sorted(pmid_set),
        }

    # ── Fetch PubMed abstracts ──────────────────────────────────────────────
    print("\nFetching full PubMed records...")
    all_pmids = sorted({pmid for v in all_results.values() for pmid in v["all_pmids"]})
    print(f"  total unique PMIDs: {len(all_pmids)}")
    records = fetch_records(all_pmids)
    print(f"  successfully parsed: {len(records)}")

    # Index by PMID for fast lookup
    record_index: Dict[str, Dict[str, Any]] = {
        r["pmid"]: r for r in records if r.get("pmid")
    }

    # ── Build PMID → aliases union across all predictions ──────────────────
    # A PMID may appear in multiple predictions; union their alias lists so
    # PMC extraction can find alias mentions from any of those predictions.
    pmid_to_aliases: Dict[str, List[str]] = {}
    for pred_id, v in all_results.items():
        pred_aliases = alias_by_pred.get(pred_id, [])
        for pmid in v["all_pmids"]:
            existing = set(pmid_to_aliases.get(pmid, []))
            existing.update(pred_aliases)
            pmid_to_aliases[pmid] = list(existing)

    # ── PMC full-text enrichment ────────────────────────────────────────────
    if not skip_pmc:
        enrich_with_pmc(list(record_index.values()), pmid_to_aliases)
    else:
        print("\nSkipping PMC full-text enrichment (--skip-pmc flag set)")

    # ── Attach records to each prediction and write output ─────────────────
    for pred_id, v in all_results.items():
        v["records"] = [
            record_index[p] for p in v["all_pmids"] if p in record_index
        ]

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    import argparse
    base = Path(__file__).parent
    ap   = argparse.ArgumentParser(description="Stage 3: PubMed retrieval + PMC enrichment")
    ap.add_argument("--skip-pmc", action="store_true",
                    help="Skip PMC full-text enrichment (faster, abstract-only mode)")
    ap.add_argument("--queries",      default=None,
                    help="Path to queries.yaml (default: <script_dir>/queries.yaml)")
    ap.add_argument("--output",       default=None,
                    help="Path for literature_raw.json output")
    ap.add_argument("--predictions",  default=None,
                    help="Path to predictions_expanded.yaml (needed for alias lookup)")
    args = ap.parse_args()

    retrieve_for_all_queries(
        queries_path     = Path(args.queries)     if args.queries     else base / "queries.yaml",
        output_path      = Path(args.output)      if args.output      else base / "literature_raw.json",
        predictions_path = Path(args.predictions) if args.predictions else base / "predictions_expanded.yaml",
        skip_pmc         = args.skip_pmc,
    )
