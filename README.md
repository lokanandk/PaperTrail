<p align="center">
  <img src="assets/logo.svg" alt="PaperTrail" width="320">
</p>

<p align="center">
  <em>Check what you found against what the literature already says.</em>
</p>

---

You have a list of findings — a gene goes up in a disease, a cell type expands,
a transporter is lost. Before building on any of it you need to know which
findings the published literature already supports, which it contradicts, and
which nobody has looked at.

PaperTrail does that check systematically. You describe your predictions, it
searches PubMed, reads what it finds, and scores each prediction by how well the
literature agrees with it.

The output is an interactive dashboard where every score traces back to the
sentences and papers it came from — so you can disagree with it.

## What you get

Each prediction is placed in an evidence tier:

| Tier | Meaning |
|---|---|
| `STRONG` | Consistently supported, and the result survives dropping any single paper |
| `MODERATE` | Supported, with the agreement weaker or the paper count lower |
| `MIXED` | Real evidence on both sides |
| `WEAK_SUPPORT` | Leans your way, but thin |
| `WEAK_DISCORDANT` | Leans against you |
| `DESCRIPTIVE` | Papers discuss the entity without a clear direction |
| `NO_DIRECTIONAL` | Nothing directional found |

Tiers come from the concordant/opposite split across informative papers, a
binomial test, quality weighting by study type, and a leave-one-out check that
flags any tier resting on a single paper.

## Quick start

```bash
git clone https://github.com/<your-username>/PaperTrail.git
cd PaperTrail
pip install -r requirements.txt
python papertrail_app.py --port 5050
```

Open <http://localhost:5050>. Start with **Demo mode** — four worked examples
(lung adenocarcinoma, rheumatoid arthritis, SLE, IBD GWAS/eQTL) load instantly
from pre-computed results, so you can see the output before running anything.

For a real run you will want to set your NCBI email, which PubMed asks for:

```bash
export NCBI_EMAIL='you@example.com'
```

An `NCBI_API_KEY` is optional and raises the rate limit from 3 to 10 requests
per second. See `.env.example` for everything that can be configured.

## Describing predictions

Write YAML directly:

```yaml
predictions:
  - id: MTHFS_DKD_up
    entity: MTHFS
    entity_type: gene
    aliases: ["MTHFS"]
    disease_context: DKD
    cell_type: C_TAL
    tissue: kidney
    organism: human
    direction: up
    note: "MTHFS elevated in DKD thick ascending limb"
    confidence: medium
    novelty: novel
```

Or type it in plain English and let PaperTrail convert it. That conversion is
the only place an LLM is used, and it is optional — without one, a rule-based
parser handles it.

## Connecting an LLM (optional)

PaperTrail works with any OpenAI-compatible endpoint and finds your
configuration on its own. If you already have a gateway set up under any
reasonable variable name — `OPENAI_API_KEY`, `LITELLM_BASE_URL`, a company
proxy called something else entirely — it will be picked up automatically.

To check what PaperTrail sees:

```bash
python llm_gateway.py
```

To set one explicitly:

```bash
export PAPERTRAIL_LLM_BASE_URL='https://your-gateway.example.com'
export PAPERTRAIL_LLM_API_KEY='your-key'
export PAPERTRAIL_LLM_MODEL='your-model'     # optional; discovered if omitted
```

Discovery pairs any `*_API_KEY`-style variable with a matching `*_BASE_URL`, so
unusual naming is fine. A variable it does not recognise is simply skipped —
PaperTrail falls back to the rule-based parser and tells you what to export
rather than failing. A local Ollama daemon is detected with no configuration at
all.

## Running the pipeline directly

The web app is a front end over a six-stage pipeline you can run headless:

```bash
python run_pipeline.py --predictions predictions.yaml --output-dir results/
```

| Stage | Does |
|---|---|
| 1 | Expands gene aliases via NCBI Gene / mygene.info |
| 2 | Builds PubMed queries from each prediction |
| 3 | Retrieves records through NCBI E-utilities, with PMC full text where available |
| 4 | Filters for relevance and extracts directional claims |
| 5 | Scores concordance, with binomial tests and leave-one-out sensitivity |
| 6 | Writes figures, `report.md`, and the interactive dashboard |

Useful flags: `--stage 4 5 6` to re-run part of it, `--skip-pmc` to stay with
abstracts, `--shared-cache DIR` to point several projects at one record cache.

## Batch size and concurrency

Large batches are fine. A few hundred predictions in one YAML works; it is
just slow, because stage 3 paces itself to stay inside NCBI's per-IP rate
limit. The app estimates the wait and streams progress as it goes.

Runs are executed **one at a time** by design. The stage modules configure
themselves through process-wide globals, and PubMed's rate limit is per IP, so
parallel runs would corrupt each other's state and collect HTTP 429s rather
than finish sooner. Submit as many as you like — they queue and report their
position. `PAPERTRAIL_MAX_CONCURRENT_RUNS` raises the limit if you have an NCBI
API key and know what you are doing.

## Caches

Downloaded PubMed records, embeddings, and ontology files are cached in
`cache/`, `shared_cache/`, and `vocab_cache/`. They make re-runs much faster and
are safe to delete — they will be re-downloaded. They are not tracked in git.

## Layout

```
papertrail_app.py          Flask app: demo mode, full runs, log streaming
llm_gateway.py             LLM endpoint discovery (run it to see what it finds)
run_pipeline.py            Headless pipeline orchestrator
stage1..stage6_*.py        The pipeline stages
papertrail_dashboard.py    Dashboard HTML and paper enrichment
semantic_relevance.py      Optional embedding-based relevance scoring
direction_verifier.py      Direction agreement checks
synonym_enrichment.py      MeSH / Cell Ontology synonym expansion
build_papertrail_vocab.py  Rebuilds papertrail_vocab.json from ontologies
demo_data.json             Pre-computed results for the four demos
papertrail_vocab.json      Disease, cell type, and tissue vocabulary
```

## Requirements

Python 3.9+. Install with `pip install -r requirements.txt`.

`sentence-transformers` is optional and enables embedding-based relevance
scoring in stage 4; without it, stage 4 uses keyword matching.
