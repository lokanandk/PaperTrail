"""
synonym_enrichment.py — Automatic synonym expansion for predictions.

Expands disease_synonyms (via NCBI MeSH) and cell_type_synonyms /
tissue context (via Cell Ontology OBO) for each prediction.

All results are cached to disk so subsequent runs are instant.
Both sources are free and use credentials already required by the pipeline:
  - NCBI MeSH: uses the same Entrez email/API key as stage3
  - Cell Ontology: freely downloadable OBO file (no auth required)

Graceful degradation: if any lookup fails, manual synonyms from
predictions.yaml are kept unchanged and a warning is printed.
The pipeline never breaks due to enrichment failure.

Called by stage1_expand_aliases.expand_predictions() at the end of
gene alias expansion, writing enriched synonyms into predictions_expanded.yaml.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    from Bio import Entrez
    ENTREZ_AVAILABLE = True
except ImportError:
    ENTREZ_AVAILABLE = False

try:
    import obonet
    import networkx as nx
    OBONET_AVAILABLE = True
except ImportError:
    OBONET_AVAILABLE = False

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_MESH_CACHE_FILE  = CACHE_DIR / "mesh_synonyms.json"
_CL_CACHE_FILE    = CACHE_DIR / "cl_synonyms.json"
_CL_OBO_FILE      = CACHE_DIR / "cl.obo"

# Cell Ontology OBO source — always publicly available, no auth
_CL_OBO_URL = "https://purl.obolibrary.org/obo/cl.obo"


# ─────────────────────────────────────────────────────────────────────────────
# NCBI MeSH synonym expansion
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_mesh_synonyms(term: str) -> List[str]:
    """
    Query NCBI MeSH for entry terms (synonyms) for a disease/tissue term.

    Uses Entrez esearch + efetch. Returns a list of synonym strings,
    empty list on failure or if no MeSH term found.
    """
    if not ENTREZ_AVAILABLE:
        return []
    try:
        # Step 1: find the MeSH UID
        h = Entrez.esearch(db="mesh", term=term, retmax=1)
        rec = Entrez.read(h); h.close()
        ids = rec.get("IdList", [])
        if not ids:
            return []

        # Step 2: fetch the full MeSH record
        time.sleep(0.35)   # NCBI rate limit
        h = Entrez.efetch(db="mesh", id=ids[0], rettype="full", retmode="text")
        text = h.read(); h.close()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")

        # Parse ENTRY = lines (each is a synonym / entry term)
        synonyms = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("ENTRY = "):
                entry = line[8:].strip()
                # Strip trailing qualifier like [Disease/Syndrome]
                entry = re.sub(r'\s*\[[^\]]+\]\s*$', '', entry).strip()
                if entry and entry.lower() != term.lower():
                    synonyms.append(entry)
            elif line.startswith("MH = "):
                mh = line[5:].strip()
                if mh and mh.lower() != term.lower():
                    synonyms.insert(0, mh)   # preferred MeSH heading goes first

        return synonyms[:12]   # cap at 12 to keep queries manageable

    except Exception as e:
        print(f"  [synonym_enrichment] MeSH lookup failed for {term!r}: {e}")
        return []


def get_mesh_synonyms(term: str, force_refresh: bool = False) -> List[str]:
    """
    Cached MeSH synonym lookup. Cache is persistent across runs.
    Results are keyed by the lowercase term.
    """
    cache = json.loads(_MESH_CACHE_FILE.read_text()) if _MESH_CACHE_FILE.exists() else {}
    key   = term.strip().lower()

    if key in cache and not force_refresh:
        return cache[key]

    syns = _fetch_mesh_synonyms(term)
    cache[key] = syns
    _MESH_CACHE_FILE.write_text(json.dumps(cache, indent=2))
    return syns


# ─────────────────────────────────────────────────────────────────────────────
# Cell Ontology synonym and parent-compartment expansion
# ─────────────────────────────────────────────────────────────────────────────

# Curated fallback mapping for common cell type codes in the pipeline.
# Used when Cell Ontology OBO is not available (no obonet, no internet).
# Format: cell_type_key → (cell_type_synonyms, tissue_synonyms)
_CURATED_CELL_MAP: Dict[str, Tuple[List[str], List[str]]] = {
    "iPT": (
        ["injured proximal tubule", "proximal tubular cell",
         "proximal tubule epithelial cell", "proximal tubule S1",
         "proximal tubule S2", "proximal tubule S3"],
        ["proximal tubule", "renal cortex", "kidney cortex",
         "tubular epithelium", "renal tubule"],
    ),
    "C_TAL": (
        ["cortical thick ascending limb cell", "thick ascending limb cell",
         "TAL cell", "thick ascending limb epithelial cell",
         "loop of Henle cell", "distal tubule"],
        ["cortical thick ascending limb", "thick ascending limb",
         "loop of Henle", "renal medulla", "renal cortex", "kidney tubule"],
    ),
    "PT_S1": (
        ["proximal tubule S1 cell", "proximal convoluted tubule cell"],
        ["proximal tubule", "renal cortex"],
    ),
    "PT_S2": (
        ["proximal tubule S2 cell", "proximal straight tubule cell"],
        ["proximal tubule", "renal cortex"],
    ),
    "PT_S3": (
        ["proximal tubule S3 cell", "proximal straight tubule", "pars recta"],
        ["proximal tubule", "outer medulla"],
    ),
    "Podo": (
        ["podocyte", "glomerular visceral epithelial cell",
         "glomerular podocyte", "glomerular epithelial cell"],
        ["glomerulus", "renal glomerulus", "glomerular basement membrane"],
    ),
    "MyoFib": (
        ["renal myofibroblast", "interstitial myofibroblast",
         "kidney fibroblast", "activated fibroblast"],
        ["renal interstitium", "kidney interstitium", "tubulointerstitium"],
    ),
    "PC": (
        ["peritubular capillary endothelial cell", "peritubular endothelium",
         "renal microvascular endothelial cell"],
        ["peritubular capillary", "renal microvasculature", "kidney vasculature"],
    ),
    "tubular": (
        ["renal tubular epithelial cell", "tubular epithelial cell",
         "renal tubule cell"],
        ["renal tubule", "tubular epithelium", "kidney tubule"],
    ),
    # Immune cells
    "pDC": (
        ["plasmacytoid dendritic cell", "plasmacytoid DC",
         "IPC", "interferon-producing cell", "type I IFN-producing cell"],
        ["peripheral blood mononuclear cell", "PBMC", "blood", "immune compartment"],
    ),
    "mDC": (
        ["myeloid dendritic cell", "conventional dendritic cell", "cDC"],
        ["PBMC", "peripheral blood", "immune compartment"],
    ),
    "Monocyte": (
        ["monocyte", "classical monocyte", "CD14+ monocyte",
         "peripheral blood monocyte"],
        ["peripheral blood", "PBMC"],
    ),
    "Macrophage": (
        ["macrophage", "tissue macrophage", "inflammatory macrophage",
         "M1 macrophage", "M2 macrophage"],
        ["macrophage compartment"],
    ),
    "Treg": (
        ["regulatory T cell", "FOXP3+ T cell", "CD4+CD25+ T cell",
         "T regulatory cell", "suppressor T cell"],
        ["T cell compartment", "lymph node", "peripheral blood"],
    ),
    "Th17": (
        ["Th17 cell", "T helper 17 cell", "IL-17-producing T cell",
         "RORγt+ T cell"],
        ["T cell compartment", "mucosal immune compartment"],
    ),
    "B cell": (
        ["B lymphocyte", "B cell", "naive B cell",
         "memory B cell", "marginal zone B cell"],
        ["B cell compartment", "lymph node", "peripheral blood"],
    ),
    "NK": (
        ["natural killer cell", "NK cell", "CD56+ lymphocyte"],
        ["peripheral blood", "lymph node"],
    ),
    # Oncology
    "Tumor cells": (
        ["cancer cell", "malignant cell", "tumour cell", "carcinoma cell"],
        ["tumour microenvironment", "tumour stroma"],
    ),
    "Treg cells": (
        ["regulatory T cell", "tumour-infiltrating Treg", "FOXP3+ T cell"],
        ["tumour microenvironment", "tumour-infiltrating lymphocyte"],
    ),
    # Intestinal
    "Paneth cells / intestinal epithelium": (
        ["Paneth cell", "intestinal epithelial cell", "crypt cell",
         "intestinal secretory cell"],
        ["intestinal crypt", "small intestinal epithelium", "ileal epithelium"],
    ),
    "Intestinal fibroblasts": (
        ["intestinal fibroblast", "intestinal stromal cell",
         "subepithelial myofibroblast", "intestinal myofibroblast"],
        ["intestinal stroma", "lamina propria"],
    ),
    "T cells / Treg": (
        ["T cell", "regulatory T cell", "effector T cell", "memory T cell"],
        ["intestinal mucosa", "lamina propria", "Peyer's patch"],
    ),
    "CD8+ T cells / IEL": (
        ["CD8+ T cell", "intraepithelial lymphocyte", "IEL",
         "cytotoxic T lymphocyte", "tissue-resident memory T cell"],
        ["intestinal epithelium", "intraepithelial compartment"],
    ),
    "Th17 / CD4+ T cells": (
        ["Th17 cell", "CD4+ T cell", "helper T cell",
         "IL-17-producing T cell"],
        ["intestinal mucosa", "lamina propria"],
    ),
}


def _download_cl_obo() -> bool:
    """Download Cell Ontology OBO file to cache. Returns True on success."""
    if _CL_OBO_FILE.exists():
        return True
    try:
        import urllib.request
        print(f"  [synonym_enrichment] Downloading Cell Ontology OBO (~50 MB)…")
        urllib.request.urlretrieve(_CL_OBO_URL, _CL_OBO_FILE)
        print(f"  [synonym_enrichment] CL OBO downloaded to {_CL_OBO_FILE}")
        return True
    except Exception as e:
        print(f"  [synonym_enrichment] CL OBO download failed: {e}")
        return False


def _parse_cl_graph():
    """Load and return the Cell Ontology graph. Returns None on failure."""
    if not OBONET_AVAILABLE:
        return None
    if not _CL_OBO_FILE.exists():
        if not _download_cl_obo():
            return None
    try:
        g = obonet.read_obo(str(_CL_OBO_FILE))
        return g
    except Exception as e:
        print(f"  [synonym_enrichment] CL OBO parse failed: {e}")
        return None


def _cl_lookup(cell_type_str: str, graph) -> Tuple[List[str], List[str]]:
    """
    Search the Cell Ontology graph for a cell type string.
    Returns (cell_synonyms, tissue_context).
    """
    if graph is None:
        return [], []

    target = cell_type_str.strip().lower()
    matched_node = None

    for node_id, data in graph.nodes(data=True):
        name = data.get("name", "").lower()
        if name == target:
            matched_node = node_id
            break
        # Check synonyms
        for syn in data.get("synonym", []):
            syn_text = re.match(r'"([^"]+)"', str(syn))
            if syn_text and syn_text.group(1).lower() == target:
                matched_node = node_id
                break
        if matched_node:
            break

    if not matched_node:
        return [], []

    data = graph.nodes[matched_node]
    cell_syns = [data.get("name", "")]

    # Collect all synonyms
    for syn in data.get("synonym", []):
        m = re.match(r'"([^"]+)"', str(syn))
        if m:
            s = m.group(1).strip()
            if s and s not in cell_syns:
                cell_syns.append(s)

    # Walk parent classes (is_a) for broader cell context
    tissue_context = []
    try:
        parents = list(graph.successors(matched_node))  # obonet: successors = is_a
        for p in parents[:4]:
            pdata = graph.nodes.get(p, {})
            pname = pdata.get("name", "")
            if pname and "cell" not in pname.lower() and len(pname) >= 4:
                tissue_context.append(pname)
            elif pname:
                cell_syns.append(pname)
    except Exception:
        pass

    return [s for s in cell_syns if s][:8], tissue_context[:4]


# Persistent cache for CL lookups
def get_cl_synonyms(cell_type: str,
                    force_refresh: bool = False) -> Tuple[List[str], List[str]]:
    """
    Return (cell_type_synonyms, tissue_context) for a cell type string.

    Priority:
    1. Disk cache (instant after first run)
    2. Curated fallback map (works offline, no deps)
    3. Cell Ontology OBO lookup (requires obonet + internet on first use)
    """
    cache = json.loads(_CL_CACHE_FILE.read_text()) if _CL_CACHE_FILE.exists() else {}
    key   = cell_type.strip().lower()

    if key in cache and not force_refresh:
        entry = cache[key]
        return entry.get("cell_syns", []), entry.get("tissue_ctx", [])

    # Curated map first (fast, no download)
    if cell_type in _CURATED_CELL_MAP:
        cell_syns, tissue_ctx = _CURATED_CELL_MAP[cell_type]
    else:
        # Try OBO lookup
        graph = _parse_cl_graph()
        cell_syns, tissue_ctx = _cl_lookup(cell_type, graph)

    cache[key] = {"cell_syns": cell_syns, "tissue_ctx": tissue_ctx}
    _CL_CACHE_FILE.write_text(json.dumps(cache, indent=2))
    return cell_syns, tissue_ctx


# ─────────────────────────────────────────────────────────────────────────────
# Main enrichment entry point
# ─────────────────────────────────────────────────────────────────────────────

def enrich_prediction(p: dict, ncbi_email: str = "",
                      ncbi_api_key: str = "") -> dict:
    """
    Enrich a single prediction dict with automatically derived synonyms.

    disease_synonyms:   augmented with NCBI MeSH entry terms
    cell_type_synonyms: augmented with Cell Ontology synonyms
    tissue_synonyms:    augmented with Cell Ontology parent compartments
    cell_type_ontology_context: new field with broader anatomical context

    Manual synonyms from predictions.yaml always take priority and are
    preserved. This function only ADDS, never removes.
    """
    # ── Disease synonyms via NCBI MeSH ────────────────────────────────────────
    dc = (p.get("disease_context") or "any").strip()
    if dc.lower() not in ("any", ""):
        if ENTREZ_AVAILABLE and ncbi_email:
            Entrez.email = ncbi_email
            # Only set api_key when non-empty.
            # Setting Entrez.api_key = "" sends api_key= in every URL,
            # which NCBI rejects with HTTP 400 for ALL subsequent requests —
            # corrupting the entire pipeline session. Leave None if not provided.
            if ncbi_api_key:
                Entrez.api_key = ncbi_api_key
            # Look up each existing disease synonym to find MeSH entry terms
            terms_to_look_up = [dc] + list(p.get("disease_synonyms") or [])
            existing_syns = set(s.lower() for s in (p.get("disease_synonyms") or []))
            new_syns = []
            for term in terms_to_look_up[:3]:   # cap to avoid too many API calls
                mesh_syns = get_mesh_synonyms(term)
                for s in mesh_syns:
                    if s.lower() not in existing_syns and s.lower() != dc.lower():
                        new_syns.append(s)
                        existing_syns.add(s.lower())
            if new_syns:
                p.setdefault("disease_synonyms", [])
                p["disease_synonyms"] = list(p["disease_synonyms"]) + new_syns[:6]

    # ── Cell type synonyms via Cell Ontology ──────────────────────────────────
    ct = (p.get("cell_type") or "any").strip()
    if ct.lower() not in ("any", ""):
        cl_cell_syns, cl_tissue_ctx = get_cl_synonyms(ct)

        existing_ct = set(s.lower() for s in (p.get("cell_type_synonyms") or []))
        new_ct = [s for s in cl_cell_syns if s.lower() not in existing_ct
                  and s.lower() != ct.lower()]
        if new_ct:
            p.setdefault("cell_type_synonyms", [])
            p["cell_type_synonyms"] = list(p["cell_type_synonyms"]) + new_ct[:4]

        # Tissue context from Cell Ontology parent compartments
        # This is the key feature: maps C_TAL → "thick ascending limb",
        # "loop of Henle", "renal cortex" so papers about the tissue region
        # can match even without naming the specific cell type.
        existing_tis = set(s.lower() for s in (p.get("tissue_synonyms") or []))
        new_tis = [s for s in cl_tissue_ctx if s.lower() not in existing_tis]
        if new_tis:
            p.setdefault("tissue_synonyms", [])
            p["tissue_synonyms"] = list(p["tissue_synonyms"]) + new_tis[:4]

        # Store the broader ontology context for stage4 direction weighting
        if cl_tissue_ctx:
            p["cell_type_ontology_context"] = cl_tissue_ctx

    return p


def enrich_all_predictions(predictions: list, ncbi_email: str = "",
                           ncbi_api_key: str = "",
                           use_mesh: bool = True) -> list:
    """
    Enrich all predictions in-place. Called from stage1.

    use_mesh=False skips NCBI MeSH lookups (faster, offline mode).
    Cell Ontology expansion runs regardless (uses curated map as fallback).
    """
    n_disease = n_cell = 0
    email = ncbi_email if use_mesh else ""

    print(f"  Synonym enrichment: "
          f"{'MeSH' if email else 'no-MeSH'} + "
          f"Cell Ontology ({'obonet' if OBONET_AVAILABLE else 'curated map'})")

    for p in predictions:
        before_ds = len(p.get("disease_synonyms") or [])
        before_ct = len(p.get("cell_type_synonyms") or [])

        enrich_prediction(p, ncbi_email=email, ncbi_api_key=ncbi_api_key)

        n_disease += len(p.get("disease_synonyms") or []) - before_ds
        n_cell    += len(p.get("cell_type_synonyms") or []) - before_ct

    print(f"  Added {n_disease} disease synonyms, {n_cell} cell-type synonyms")
    return predictions
