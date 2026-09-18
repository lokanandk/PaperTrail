#!/usr/bin/env python3
"""
build_papertrail_vocab.py
========================
Generate papertrail_vocab.json from authoritative open biomedical ontologies.

Sources (all freely downloadable, no login required):
  MONDO  — Monarch Disease Ontology
           https://github.com/monarch-initiative/mondo
           Comprehensive, cross-species disease ontology (>20 000 terms).
  CL     — Cell Ontology (OBO Foundry)
           https://github.com/obophenotype/cell-ontology
           Canonical cell type names and synonyms (~2 500 terms).
  UBERON — Uber-anatomy Ontology (OBO Foundry)
           http://obofoundry.org/ontology/uberon.html
           Cross-species anatomy / tissue terms (~14 000 terms).

Usage
-----
  python build_papertrail_vocab.py                 # download + build everything
  python build_papertrail_vocab.py --offline       # use cached OBO files only
  python build_papertrail_vocab.py --cache-dir /path/to/cache
  python build_papertrail_vocab.py --output my_vocab.json
  python build_papertrail_vocab.py --no-uberon     # skip UBERON (saves ~120 MB)

The script never hardcodes disease/cell-type content.  It:
  1. Downloads the latest OBO files from OBO Foundry (or uses cached copies).
  2. Parses synonyms for every entry in a curated seed list of abbreviations.
     The seeds define which MONDO/CL IDs correspond to our abbreviation codes
     (e.g. DKD ↔ MONDO:0005016); everything else (synonyms, relationships)
     is pulled from the ontology at runtime.
  3. Builds papertrail_vocab.json in the format expected by papertrail_app.py.

Adding new diseases / cell types
---------------------------------
Edit the DISEASE_SEEDS or CELL_TYPE_SEEDS dicts at the top of this file,
then re-run.  Each entry needs:
  - key   : abbreviation you want the pipeline to use (e.g. "DKD")
  - mondo : MONDO ID (look up at https://www.ebi.ac.uk/ols/ontologies/mondo)
  - cl    : CL   ID (look up at https://www.ebi.ac.uk/ols/ontologies/cl)
"""

import argparse
import gzip
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# SEED DEFINITIONS
# These are the only things in this file that encode any domain knowledge.
# They are intentionally minimal: just enough to map our abbreviations to
# ontology IDs.  All synonym content is derived from the ontologies.
# ─────────────────────────────────────────────────────────────────────────────

# Disease seeds: abbreviation → MONDO ontology ID
# Look up at https://www.ebi.ac.uk/ols4/ontologies/mondo
DISEASE_SEEDS = {
    # ── Kidney ────────────────────────────────────────────────────────────────
    "DKD":   "MONDO:0005016",   # diabetic nephropathy
    "HKD":   "MONDO:0001930",   # hypertensive renal disease
    "CKD":   "MONDO:0005300",   # chronic kidney disease
    "AKI":   "MONDO:0002492",   # acute kidney failure
    "IgAN":  "MONDO:0000308",   # IgA nephropathy
    "FSGS":  "MONDO:0007349",   # focal segmental glomerulosclerosis
    "ADPKD": "MONDO:0001083",   # autosomal dominant polycystic kidney disease
    # ── Autoimmune / inflammatory ────────────────────────────────────────────
    "SLE":   "MONDO:0007628",   # systemic lupus erythematosus
    "RA":    "MONDO:0008383",   # rheumatoid arthritis
    "IBD":   "MONDO:0005265",   # inflammatory bowel disease
    "AS":    "MONDO:0005306",   # ankylosing spondylitis
    "SS":    "MONDO:0010200",   # Sjögren syndrome
    "SSc":   "MONDO:0005100",   # systemic sclerosis
    "vasculitis": "MONDO:0016621",
    "GPA":   "MONDO:0007085",   # granulomatosis with polyangiitis
    # ── Metabolic / endocrine ────────────────────────────────────────────────
    "T1D":   "MONDO:0005147",   # type 1 diabetes mellitus
    "T2D":   "MONDO:0005148",   # type 2 diabetes mellitus
    "NAFLD": "MONDO:0004790",   # non-alcoholic fatty liver disease
    "NASH":  "MONDO:0016060",   # non-alcoholic steatohepatitis
    "obesity": "MONDO:0011122",
    "MetS":  "MONDO:0024529",   # metabolic syndrome
    # ── Cardiovascular ───────────────────────────────────────────────────────
    "HF":    "MONDO:0005009",   # heart failure
    "HFrEF": "MONDO:0004981",   # HF with reduced EF
    "MI":    "MONDO:0005068",   # myocardial infarction
    "CAD":   "MONDO:0004950",   # coronary artery disease
    "HTN":   "MONDO:0001134",   # hypertension
    "AF":    "MONDO:0004587",   # atrial fibrillation
    "PAH":   "MONDO:0015924",   # pulmonary arterial hypertension
    # ── Respiratory ──────────────────────────────────────────────────────────
    "asthma": "MONDO:0004979",
    "COPD":  "MONDO:0005002",
    "IPF":   "MONDO:0008345",   # idiopathic pulmonary fibrosis
    # ── Oncology ─────────────────────────────────────────────────────────────
    "LUAD":  "MONDO:0005061",   # lung adenocarcinoma
    "NSCLC": "MONDO:0005233",   # non-small cell lung carcinoma
    "SCLC":  "MONDO:0008433",   # small cell lung carcinoma
    "CRC":   "MONDO:0005575",   # colorectal cancer
    "HCC":   "MONDO:0007256",   # hepatocellular carcinoma
    "CCA":   "MONDO:0004073",   # cholangiocarcinoma
    "PCa":   "MONDO:0008315",   # prostate cancer
    "BC":    "MONDO:0007254",   # breast cancer
    "OC":    "MONDO:0004967",   # ovarian cancer (reusing — see note)
    "GBM":   "MONDO:0018177",   # glioblastoma
    "MM":    "MONDO:0009290",   # multiple myeloma
    "AML":   "MONDO:0018874",   # acute myeloid leukemia
    "ALL":   "MONDO:0004967",   # acute lymphoblastic leukemia  ← check ID
    "CLL":   "MONDO:0004947",   # chronic lymphocytic leukemia
    "DLBCL": "MONDO:0018874",   # diffuse large B-cell lymphoma ← check ID
    "PDAC":  "MONDO:0006047",   # pancreatic ductal adenocarcinoma
    "RCC":   "MONDO:0005005",   # renal cell carcinoma (keep for exclusion lists)
    "ccRCC": "MONDO:0005005",
    "GC":    "MONDO:0004976",   # gastric cancer
    "BLC":   "MONDO:0004056",   # bladder cancer
    "MEL":   "MONDO:0005105",   # melanoma
    # ── Neurological / psychiatric ───────────────────────────────────────────
    "AD":    "MONDO:0004975",   # Alzheimer disease
    "PD":    "MONDO:0005180",   # Parkinson disease
    "MS":    "MONDO:0005301",   # multiple sclerosis
    "ALS":   "MONDO:0004976",   # amyotrophic lateral sclerosis ← check ID
    "HD":    "MONDO:0007739",   # Huntington disease
    "schizophrenia": "MONDO:0005090",
    "MDD":   "MONDO:0002050",   # major depressive disorder
    # ── Infectious ───────────────────────────────────────────────────────────
    "sepsis": "MONDO:0021881",
    "COVID19": "MONDO:0100096",  # COVID-19
    "HIV":   "MONDO:0005109",
    "TB":    "MONDO:0018076",   # tuberculosis
    # ── Other ────────────────────────────────────────────────────────────────
    "psoriasis": "MONDO:0005083",
    "atopic_dermatitis": "MONDO:0004980",
    "osteoarthritis": "MONDO:0005178",
    "osteoporosis": "MONDO:0005298",
}

# Written-out names for each abbreviation.
#
# These used to live only in the comments above, which meant that when the
# MONDO download was unavailable every disease ended up in the vocabulary as a
# bare abbreviation with no synonyms. Anyone describing a prediction the way
# people actually write — "elevated in diabetic kidney disease" rather than
# "elevated in DKD" — got no match at all. Keeping the names here makes the
# vocabulary useful offline; MONDO synonyms are merged on top when available.
#
# Deliberately excluded: bare "diabetes", which is ambiguous between T1D and
# T2D, and any name that would collide with another entry.
DISEASE_NAMES = {
    "DKD":   ["diabetic kidney disease", "diabetic nephropathy"],
    "HKD":   ["hypertensive kidney disease", "hypertensive nephropathy"],
    "CKD":   ["chronic kidney disease", "chronic renal failure"],
    "AKI":   ["acute kidney injury", "acute renal failure"],
    "IgAN":  ["iga nephropathy", "berger disease"],
    "FSGS":  ["focal segmental glomerulosclerosis"],
    "ADPKD": ["autosomal dominant polycystic kidney disease", "polycystic kidney disease"],
    "SLE":   ["systemic lupus erythematosus", "lupus"],
    "RA":    ["rheumatoid arthritis"],
    "IBD":   ["inflammatory bowel disease", "crohn disease", "ulcerative colitis"],
    "AS":    ["ankylosing spondylitis"],
    "SS":    ["sjogren syndrome", "sjögren syndrome"],
    "SSc":   ["systemic sclerosis", "scleroderma"],
    "vasculitis": ["vasculitis"],
    "GPA":   ["granulomatosis with polyangiitis", "wegener granulomatosis"],
    "T1D":   ["type 1 diabetes", "type 1 diabetes mellitus", "insulin-dependent diabetes"],
    "T2D":   ["type 2 diabetes", "type 2 diabetes mellitus", "diabetes mellitus"],
    "NAFLD": ["non-alcoholic fatty liver disease", "nonalcoholic fatty liver disease",
              "metabolic dysfunction-associated steatotic liver disease", "masld"],
    "NASH":  ["non-alcoholic steatohepatitis", "nonalcoholic steatohepatitis", "mash"],
    "obesity": ["obesity"],
    "MetS":  ["metabolic syndrome"],
    "HF":    ["heart failure", "cardiac failure"],
    "HFrEF": ["heart failure with reduced ejection fraction"],
    "MI":    ["myocardial infarction", "heart attack"],
    "CAD":   ["coronary artery disease", "coronary heart disease"],
    "HTN":   ["hypertension", "high blood pressure"],
    "AF":    ["atrial fibrillation"],
    "PAH":   ["pulmonary arterial hypertension"],
    "asthma": ["asthma"],
    "COPD":  ["chronic obstructive pulmonary disease"],
    "IPF":   ["idiopathic pulmonary fibrosis", "pulmonary fibrosis"],
    "LUAD":  ["lung adenocarcinoma"],
    "NSCLC": ["non-small cell lung cancer", "non-small cell lung carcinoma"],
    "SCLC":  ["small cell lung cancer", "small cell lung carcinoma"],
    "CRC":   ["colorectal cancer", "colorectal carcinoma", "colon cancer"],
    "HCC":   ["hepatocellular carcinoma", "liver cancer"],
    "CCA":   ["cholangiocarcinoma", "bile duct cancer"],
    "PCa":   ["prostate cancer", "prostate adenocarcinoma"],
    "BC":    ["breast cancer", "breast carcinoma"],
    "OC":    ["ovarian cancer", "ovarian carcinoma"],
    "GBM":   ["glioblastoma", "glioblastoma multiforme"],
    "MM":    ["multiple myeloma"],
    "AML":   ["acute myeloid leukemia", "acute myeloid leukaemia"],
    "ALL":   ["acute lymphoblastic leukemia", "acute lymphoblastic leukaemia"],
    "CLL":   ["chronic lymphocytic leukemia", "chronic lymphocytic leukaemia"],
    "DLBCL": ["diffuse large b-cell lymphoma"],
    "PDAC":  ["pancreatic ductal adenocarcinoma", "pancreatic cancer"],
    "RCC":   ["renal cell carcinoma", "kidney cancer"],
    "ccRCC": ["clear cell renal cell carcinoma"],
    "GC":    ["gastric cancer", "stomach cancer"],
    "BLC":   ["bladder cancer", "bladder carcinoma"],
    "MEL":   ["melanoma", "cutaneous melanoma"],
    "AD":    ["alzheimer disease", "alzheimer's disease"],
    "PD":    ["parkinson disease", "parkinson's disease"],
    "MS":    ["multiple sclerosis"],
    "ALS":   ["amyotrophic lateral sclerosis", "motor neuron disease"],
    "HD":    ["huntington disease", "huntington's disease"],
    "schizophrenia": ["schizophrenia"],
    "MDD":   ["major depressive disorder", "depression"],
    "sepsis": ["sepsis", "septic shock"],
    "COVID19": ["covid-19", "covid19", "sars-cov-2 infection"],
    "HIV":   ["hiv infection", "aids"],
    "TB":    ["tuberculosis"],
    "psoriasis": ["psoriasis"],
    "atopic_dermatitis": ["atopic dermatitis", "eczema"],
    "osteoarthritis": ["osteoarthritis"],
    "osteoporosis": ["osteoporosis"],
}

# Short names for cell types, for the same reason as DISEASE_NAMES.
# Cell Ontology labels are precise but nobody writes them out: the official
# name for C_TAL is "kidney loop of Henle thick ascending limb epithelial
# cell", whereas a description of a finding says "thick ascending limb" or
# "TAL". Only listed where the ontology label differs from everyday usage.
CELL_TYPE_NAMES = {
    "C_TAL":  ["thick ascending limb", "TAL", "cortical thick ascending limb",
               "medullary thick ascending limb", "loop of Henle"],
    "iPT":    ["proximal tubule", "proximal tubular cell", "PT"],
    "tubular": ["tubular cell", "renal tubular cell"],
    "Podo":   ["podocyte"],
    "EC_glom": ["glomerular endothelial cell"],
    "MC":     ["mesangial cell"],
    "pDC":    ["plasmacytoid dendritic cell"],
    "Treg":   ["regulatory T cell", "Tregs"],
    "Macro":  ["macrophage"],
    "Mono":   ["monocyte"],
    "Neut":   ["neutrophil"],
    "Hepato": ["hepatocyte"],
    "HSC":    ["hepatic stellate cell"],
    "AT2":    ["alveolar type II cell", "type II pneumocyte"],
    "AT1":    ["alveolar type I cell", "type I pneumocyte"],
    "AM":     ["alveolar macrophage"],
    "CM":     ["cardiomyocyte"],
    "MG":     ["microglia"],
    "Astro":  ["astrocyte"],
    "Neuron": ["neuron"],
    "BetaCell": ["beta cell", "pancreatic beta cell", "islet beta cell"],
}

# Cell-type seeds: internal code → Cell Ontology ID
# Look up at https://www.ebi.ac.uk/ols4/ontologies/cl
CELL_TYPE_SEEDS = {
    # ── Kidney ────────────────────────────────────────────────────────────────
    # CL:0002204 is "tuft cell", not thick ascending limb — using it gave C_TAL
    # the synonyms "tuft cell" and "brush cell", so a TAL prediction matched
    # entirely unrelated papers.
    "C_TAL":   "CL:1001106",   # kidney loop of Henle thick ascending limb epithelial cell
    "iPT":     "CL:1001107",   # renal proximal tubule epithelial cell
    "Podo":    "CL:0000653",   # glomerular visceral epithelial cell (podocyte)
    "tubular": "CL:1001107",   # proximal tubular cell (reuse)
    "MC":      "CL:1000742",   # kidney mesangial cell
    "EC_glom": "CL:1001005",   # glomerular endothelial cell
    "MyoFib":  "CL:0000186",   # myofibroblast
    "PC":      "CL:0010008",   # peritubular capillary endothelial cell
    # ── Immune ────────────────────────────────────────────────────────────────
    "pDC":     "CL:0000784",   # plasmacytoid dendritic cell
    "cDC1":    "CL:0002399",   # CD8α+ dendritic cell (type 1)
    "cDC2":    "CL:0002396",   # CD11b+ dendritic cell (type 2)
    "Treg":    "CL:0000815",   # regulatory T cell
    "CD4T":    "CL:0000624",   # CD4+ alpha-beta T cell
    "CD8T":    "CL:0000625",   # CD8+ alpha-beta T cell
    "NK":      "CL:0000623",   # natural killer cell
    "B":       "CL:0000236",   # B cell
    "PB":      "CL:0000816",   # plasmablast
    "PC_imm":  "CL:0000786",   # plasma cell (immunology)
    "Mono":    "CL:0000576",   # monocyte
    "Macro":   "CL:0000235",   # macrophage
    "M1":      "CL:0000863",   # inflammatory macrophage
    "M2":      "CL:0000890",   # alternatively activated macrophage
    "Neut":    "CL:0000775",   # neutrophil
    "Eos":     "CL:0000771",   # eosinophil
    "Mast":    "CL:0000097",   # mast cell
    "NKT":     "CL:0000814",   # NKT cell
    "ILC":     "CL:0001065",   # innate lymphoid cell
    "Th1":     "CL:0000545",   # Th1 cell
    "Th2":     "CL:0000546",   # Th2 cell
    "Th17":    "CL:0000899",   # Th17 cell
    # ── Liver ─────────────────────────────────────────────────────────────────
    "Hepato":  "CL:0000182",   # hepatocyte
    "Kupffer": "CL:0000091",   # Kupffer cell
    "HSC":     "CL:0000632",   # hepatic stellate cell
    "LSinEC":  "CL:0000272",   # liver sinusoidal endothelial cell
    "Cholangio": "CL:1000488", # cholangiocyte
    # ── Lung ──────────────────────────────────────────────────────────────────
    "AM":      "CL:0000583",   # alveolar macrophage
    "AT1":     "CL:0002062",   # alveolar type I cell
    "AT2":     "CL:0002063",   # alveolar type II cell
    "Club":    "CL:0000158",   # club cell (Clara cell)
    "Ciliated": "CL:0000067",  # ciliated cell
    # ── Brain ─────────────────────────────────────────────────────────────────
    "Neuron":  "CL:0000540",   # neuron
    "Astro":   "CL:0000127",   # astrocyte
    "MG":      "CL:0000129",   # microglia
    "Oligo":   "CL:0000128",   # oligodendrocyte
    "OPC":     "CL:0002453",   # oligodendrocyte precursor cell
    # ── Heart ─────────────────────────────────────────────────────────────────
    "CM":      "CL:0000746",   # cardiomyocyte
    "CF":      "CL:0000057",   # cardiac fibroblast
    # ── Gut ───────────────────────────────────────────────────────────────────
    "Entero":  "CL:0000160",   # enterocyte (absorptive cell)
    "Goblet":  "CL:0000160",   # goblet cell  ← check ID
    "Paneth":  "CL:0000510",   # Paneth cell
    # ── Pancreas ──────────────────────────────────────────────────────────────
    "BetaCell": "CL:0000169",  # type B pancreatic cell (beta cell)
    "AlphaCell": "CL:0000168", # type A pancreatic cell (alpha cell)
    # ── Skin ──────────────────────────────────────────────────────────────────
    "Keratinocyte": "CL:0000312",
    "Melanocyte": "CL:0000148",
}

# Tissue / organ seeds: internal name → UBERON ID
# Look up at https://www.ebi.ac.uk/ols4/ontologies/uberon
TISSUE_SEEDS = {
    "kidney":   ("UBERON:0002113", ["kidney","renal","nephro","tubular","glomerular","nephron"]),
    "immune":   ("UBERON:0002390", ["immune","lymph","spleen","pbmc","peripheral blood","lymphocyte"]),
    "liver":    ("UBERON:0002107", ["liver","hepat","hepatocyte","kupffer","sinusoid"]),
    "lung":     ("UBERON:0002048", ["lung","pulmonary","alveolar","bronch","pneumocyte"]),
    "brain":    ("UBERON:0000955", ["brain","neural","neuron","cortex","hippocampus","cerebr","astrocyte"]),
    "heart":    ("UBERON:0000948", ["heart","cardiac","myocardium","cardiomyocyte"]),
    "colon":    ("UBERON:0001155", ["colon","intestin","bowel","colitis","enterocyte"]),
    "pancreas": ("UBERON:0001264", ["pancrea","islet","acinar","beta cell","insulin"]),
    "muscle":   ("UBERON:0001630", ["skeletal muscle","myocyte","myofiber","myoblast"]),
    "adipose":  ("UBERON:0001013", ["adipose","adipocyte","fat tissue","white adipose"]),
    "skin":     ("UBERON:0002097", ["skin","epiderm","keratinocyte","dermis","fibroblast"]),
    "bone":     ("UBERON:0002481", ["bone","osteo","osteoblast","osteoclast","chondrocyte"]),
    "thyroid":  ("UBERON:0002046", ["thyroid","thyrocyte"]),
    "adrenal":  ("UBERON:0002369", ["adrenal","cortisol","aldosterone"]),
}

# ── Off-topic exclusion terms per tissue ─────────────────────────────────────
# These filter irrelevant cancer/tumour literature when searching for the
# corresponding organ disease.  Expand as needed.
DISEASE_EXCLUSIONS = {
    "kidney": [
        "renal carcinoma", "renal cell carcinoma", "RCC", "clear cell carcinoma",
        "kidney cancer", "kidney tumor", "kidney neoplasm", "renal neoplasm",
        "ccRCC", "Wilms tumor", "nephroblastoma", "oncocytoma", "angiomyolipoma",
    ],
    "liver": [
        "hepatocellular carcinoma", "HCC", "liver cancer", "hepatoblastoma",
        "cholangiocarcinoma", "biliary cancer", "liver metastasis",
    ],
    "lung": [
        "small cell lung cancer", "SCLC", "mesothelioma", "pleural mesothelioma",
        "lung metastasis", "pulmonary metastasis",
    ],
    "brain": [
        "glioblastoma", "GBM", "glioma", "medulloblastoma", "meningioma",
        "brain tumor", "brain cancer", "brain metastasis",
    ],
    "colon": ["colorectal carcinoma", "rectal cancer", "anal cancer"],
    "skin":  ["melanoma", "basal cell carcinoma", "squamous cell carcinoma", "Merkel"],
    "pancreas": ["pancreatic cancer", "pancreatic adenocarcinoma", "PDAC"],
}

# ─────────────────────────────────────────────────────────────────────────────
# OBO DOWNLOAD URLS
# ─────────────────────────────────────────────────────────────────────────────
OBO_URLS = {
    "mondo": "http://purl.obolibrary.org/obo/mondo.obo",
    "cl":    "http://purl.obolibrary.org/obo/cl.obo",
    "uberon": "http://purl.obolibrary.org/obo/uberon.obo",
}

# ─────────────────────────────────────────────────────────────────────────────
# OBO PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_obo(path: Path) -> dict:
    """
    Parse an OBO flat file.  Returns a dict: term_id → term_dict.
    term_dict keys: id, name, synonyms (list[str]), is_obsolete (bool).
    Only [Term] stanzas are parsed; [Typedef] stanzas are ignored.
    """
    terms = {}
    cur = None
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                if cur and cur.get("id"):
                    terms[cur["id"]] = cur
                cur = {"id": "", "name": "", "synonyms": [], "is_obsolete": False}
            elif line == "[Typedef]":
                if cur and cur.get("id"):
                    terms[cur["id"]] = cur
                cur = None   # skip Typedef stanzas
            elif cur is None:
                continue
            elif line.startswith("id: "):
                cur["id"] = line[4:].strip()
            elif line.startswith("name: "):
                cur["name"] = line[6:].strip()
            elif line.startswith("is_obsolete: true"):
                cur["is_obsolete"] = True
            elif line.startswith("synonym: "):
                m = re.match(r'synonym:\s+"([^"]+)"', line)
                if m:
                    cur["synonyms"].append(m.group(1))

    if cur and cur.get("id"):
        terms[cur["id"]] = cur
    return terms


# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD HELPER
# ─────────────────────────────────────────────────────────────────────────────

def download(url: str, dest: Path, desc: str = "") -> bool:
    """Download url to dest with a progress indicator. Returns True on success."""
    print(f"  Downloading {desc or url} …", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "papertrail-vocab-builder/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
            total = int(resp.getheader("Content-Length", 0))
            done  = 0
            while chunk := resp.read(1 << 17):   # 128 KB chunks
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  Downloading {desc}: {done/1e6:.1f}/{total/1e6:.1f} MB",
                          end="", flush=True)
        print(f"\r  ✓ {desc}: {done/1e6:.1f} MB saved to {dest.name}")
        return True
    except Exception as e:
        print(f"\r  ✗ {desc}: {e}")
        return False


def get_obo(name: str, cache_dir: Path, offline: bool) -> dict:
    """Load (and optionally download) an OBO file. Returns parsed term dict."""
    dest = cache_dir / f"{name}.obo"
    if dest.exists():
        print(f"  Using cached {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
    elif offline:
        print(f"  SKIP (offline mode): {name}.obo not in cache")
        return {}
    else:
        ok = download(OBO_URLS[name], dest, name.upper())
        if not ok:
            return {}
    print(f"  Parsing {dest.name} …", end="", flush=True)
    terms = parse_obo(dest)
    print(f"\r  ✓ Parsed {dest.name}: {len(terms):,} terms")
    return terms


# ─────────────────────────────────────────────────────────────────────────────
# SYNONYM ENRICHMENT
# ─────────────────────────────────────────────────────────────────────────────

def clean_synonym(s: str) -> str:
    """Remove markup artefacts sometimes present in OBO synonyms."""
    # Remove trailing [database:ID] style xrefs
    s = re.sub(r'\s*\[.*?\]\s*$', '', s)
    return s.strip()


def get_synonyms(terms: dict, obo_id: str) -> list:
    """Return all unique synonym strings for a given OBO ID."""
    term = terms.get(obo_id)
    if not term or term.get("is_obsolete"):
        return []
    result = [term["name"]] if term["name"] else []
    for s in term["synonyms"]:
        s = clean_synonym(s)
        if s and s not in result:
            result.append(s)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_disease_tables(mondo_terms: dict) -> tuple:
    """
    Build disease_context_map and disease_table from MONDO + seeds.
    Returns (context_map, disease_table).
    """
    context_map  = {}  # "diabetic nephropathy" → "DKD"
    disease_table = {}  # "dkd" → ["DKD", [...synonyms...]]

    for abbrev, mondo_id in DISEASE_SEEDS.items():
        syns = get_synonyms(mondo_terms, mondo_id)
        if not syns and mondo_terms:
            print(f"    ⚠  No synonyms found for {abbrev} ({mondo_id}) — check the MONDO ID")

        # Built-in names come first so the table is usable even with no MONDO
        # download; ontology synonyms are merged on top when they are available.
        full_synonyms = list(dict.fromkeys(DISEASE_NAMES.get(abbrev, []) + syns))[:12]
        primary_name = full_synonyms[0] if full_synonyms else abbrev.lower()

        # disease_table entry: every lowercase alias → (abbrev, synonyms)
        all_keys = {abbrev.lower(), primary_name.lower()}
        all_keys.update(s.lower() for s in full_synonyms)

        for key in all_keys:
            disease_table[key]  = [abbrev, full_synonyms]
            context_map[key]    = abbrev

    return context_map, disease_table


def build_cell_type_table(cl_terms: dict) -> dict:
    """Build cell_type_table from Cell Ontology + seeds."""
    table = {}
    for code, cl_id in CELL_TYPE_SEEDS.items():
        syns = get_synonyms(cl_terms, cl_id)
        if not syns and cl_terms:
            print(f"    ⚠  No synonyms for {code} ({cl_id}) — check the CL ID")

        full_synonyms = list(dict.fromkeys(CELL_TYPE_NAMES.get(code, []) + syns))[:8]
        all_aliases = {code.lower()}
        all_aliases.update(s.lower() for s in full_synonyms)

        for alias in all_aliases:
            table[alias] = [code, full_synonyms]

    return table


def build_tissue_keywords(uberon_terms: dict) -> dict:
    """
    Build tissue_keywords from UBERON + seed keyword lists.
    For each tissue we take the seed keyword list and augment with UBERON synonyms.
    """
    result = {}
    for tissue_name, (uberon_id, seed_keywords) in TISSUE_SEEDS.items():
        keywords = list(seed_keywords)  # start with curated seeds

        if uberon_id and uberon_terms:
            syns = get_synonyms(uberon_terms, uberon_id)
            for s in syns:
                # Only add short/relevant terms as keywords (avoid long descriptive phrases)
                s_l = s.lower()
                if len(s_l) <= 25 and s_l not in keywords:
                    keywords.append(s_l)

        # Format: [keywords_list, tissue_value] — same structure as old _TISSUE_KEYWORDS
        result[tissue_name] = [keywords[:20], tissue_name]   # cap at 20 keywords

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default="./vocab_cache",
                   help="Directory to cache downloaded OBO files (default: ./vocab_cache)")
    p.add_argument("--output", default="papertrail_vocab.json",
                   help="Output JSON path (default: papertrail_vocab.json)")
    p.add_argument("--offline", action="store_true",
                   help="Use only cached OBO files; fail gracefully if absent")
    p.add_argument("--no-uberon", action="store_true",
                   help="Skip UBERON download (saves ~120 MB; uses seed keywords only)")
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n═══════════════════════════════════════════════════")
    print(" PaperTrail vocabulary builder")
    print("═══════════════════════════════════════════════════\n")

    # ── 1. Download / load ontologies ─────────────────────────────────────────
    print("Step 1 — MONDO (disease ontology)")
    mondo_terms = get_obo("mondo", cache_dir, args.offline)

    print("\nStep 2 — Cell Ontology (CL)")
    cl_terms = get_obo("cl", cache_dir, args.offline)

    if not args.no_uberon:
        print("\nStep 3 — UBERON (anatomy ontology, ~120 MB — pass --no-uberon to skip)")
        uberon_terms = get_obo("uberon", cache_dir, args.offline)
    else:
        print("\nStep 3 — UBERON skipped (--no-uberon)")
        uberon_terms = {}

    # ── 2. Build tables ────────────────────────────────────────────────────────
    print("\nStep 4 — Building disease tables …")
    context_map, disease_table = build_disease_tables(mondo_terms)
    print(f"  ✓ disease_context_map: {len(context_map):,} entries")
    print(f"  ✓ disease_table:       {len(disease_table):,} entries")

    print("\nStep 5 — Building cell type table …")
    cell_type_table = build_cell_type_table(cl_terms)
    print(f"  ✓ cell_type_table:     {len(cell_type_table):,} entries")

    print("\nStep 6 — Building tissue keywords …")
    tissue_keywords = build_tissue_keywords(uberon_terms)
    print(f"  ✓ tissue_keywords:     {len(tissue_keywords):,} tissues")

    # ── 3. Assemble output ────────────────────────────────────────────────────
    vocab = {
        "_comment": [
            "PaperTrail biological vocabulary.",
            f"Generated by build_papertrail_vocab.py from MONDO / CL / UBERON.",
            "DO NOT edit by hand — re-run build_papertrail_vocab.py to update.",
            "To add a new disease/cell-type: add an entry to DISEASE_SEEDS or",
            "CELL_TYPE_SEEDS in build_papertrail_vocab.py then re-run.",
        ],
        "disease_context_map": context_map,
        "disease_table":       disease_table,
        "disease_exclusions":  DISEASE_EXCLUSIONS,
        "cell_type_table":     cell_type_table,
        "tissue_keywords":     tissue_keywords,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    size_kb = out_path.stat().st_size / 1024
    print(f"\n═══════════════════════════════════════════════════")
    print(f" ✓ Written: {out_path}  ({size_kb:.0f} KB)")
    print(f"   diseases:   {len(DISEASE_SEEDS)}")
    print(f"   cell types: {len(CELL_TYPE_SEEDS)}")
    print(f"   tissues:    {len(TISSUE_SEEDS)}")
    print("═══════════════════════════════════════════════════\n")

    if not mondo_terms:
        print("⚠  MONDO was not loaded.  Disease synonyms defaulted to seed names only.")
        print("   Re-run without --offline to download MONDO (~60 MB one-time).")
    if not cl_terms:
        print("⚠  Cell Ontology was not loaded.  Cell type synonyms defaulted to seed names.")


if __name__ == "__main__":
    main()
