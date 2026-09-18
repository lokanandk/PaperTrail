"""
Literature Validation Pipeline — orchestrator.

All biological context comes from predictions.yaml.
Nothing is hardcoded in the pipeline scripts.

Usage:
  python run_pipeline.py                         # run all stages
  python run_pipeline.py --skip-pmc              # skip PMC full-text fetch
  python run_pipeline.py --stage 4 5 6           # run specific stages only
  python run_pipeline.py --semantic-model NAME   # override SapBERT model
  python run_pipeline.py --predictions FILE      # specify predictions.yaml path
  python run_pipeline.py --output-dir DIR        # write outputs to custom directory
  python run_pipeline.py --shared-cache DIR      # shared PubMed cache directory
  python run_pipeline.py --no-dashboard          # skip interactive dashboard generation
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

import stage1_expand_aliases
import stage2_build_queries
import stage3_retrieve_pubmed
import stage4_extract_evidence
import stage5_score_concordance
import stage6_visualise_report

BASE = Path(__file__).parent


def _resolve_dirs(args) -> tuple[Path, Path]:
    """Return (pipeline_dir, output_dir)."""
    out_dir = Path(args.output_dir) if args.output_dir else BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    return BASE, out_dir


def run_stage1(out_dir: Path) -> None:
    print("\n=== Stage 1: Expand aliases ===")
    stage1_expand_aliases.expand_predictions(
        out_dir / "predictions.yaml", out_dir / "predictions_expanded.yaml")


def run_stage2(out_dir: Path) -> None:
    print("\n=== Stage 2: Build PubMed queries ===")
    stage2_build_queries.build_all(
        out_dir / "predictions_expanded.yaml", out_dir / "queries.yaml")


def run_stage3(out_dir: Path, skip_pmc: bool = False) -> None:
    print("\n=== Stage 3: Retrieve PubMed records ===")
    stage3_retrieve_pubmed.retrieve_for_all_queries(
        out_dir / "queries.yaml",
        out_dir / "literature_raw.json",
        predictions_path=out_dir / "predictions_expanded.yaml",
        skip_pmc=skip_pmc,
    )


def run_stage4(out_dir: Path, semantic_model: str = None) -> None:
    print("\n=== Stage 4: Extract directional evidence ===")
    stage4_extract_evidence.extract_all(
        out_dir / "literature_raw.json",
        out_dir / "predictions_expanded.yaml",
        out_dir / "extracted_evidence.json",
        semantic_model=semantic_model,
    )


def run_stage5(out_dir: Path) -> None:
    print("\n=== Stage 5: Score concordance ===")
    stage5_score_concordance.score_all(
        out_dir / "extracted_evidence.json",
        out_dir / "scored_predictions.json")


def run_stage6(out_dir: Path) -> None:
    print("\n=== Stage 6: Generate report and figures ===")
    stage6_visualise_report.generate(
        out_dir / "scored_predictions.json",
        out_dir / "output")


def run_dashboard(out_dir: Path, project_name: str = "") -> None:
    """
    Generate dashboard.html by delegating entirely to papertrail_dashboard.main().

    papertrail_dashboard.py is self-contained: it reads the files, enriches
    the papers with YAKE+TF-IDF, and writes the HTML.  We just call it with
    the right arguments rather than reimplementing its logic here.
    """
    print("\n=== Dashboard: Generating interactive HTML ===")

    scored_path = out_dir / "scored_predictions.json"
    report_path = out_dir / "output" / "report.md"
    output_html = out_dir / "output" / "dashboard.html"
    cache_dir   = stage3_retrieve_pubmed.RECORDS_DIR

    if not report_path.exists():
        print(f"  ⚠  report.md not found at {report_path} — skipping dashboard.")
        return

    # Make sure papertrail_dashboard is importable from the script directory
    import sys
    if str(BASE) not in sys.path:
        sys.path.insert(0, str(BASE))

    try:
        import papertrail_dashboard as dg
    except ImportError as e:
        print(f"  ⚠  papertrail_dashboard.py not found: {e}")
        print(f"     Place papertrail_dashboard.py in the same directory as run_pipeline.py")
        return

    # Build argv that papertrail_dashboard.main() understands, then call it.
    # This is the same as running:
    #   python papertrail_dashboard.py --report ... --scored ... --cache ... --out ...
    argv_orig = sys.argv[:]
    sys.argv = [
        "papertrail_dashboard.py",
        "--report", str(report_path),
        "--out",    str(output_html),
    ]
    if scored_path.exists():
        sys.argv += ["--scored", str(scored_path)]
    if cache_dir.exists():
        sys.argv += ["--cache", str(cache_dir)]
    if project_name:
        sys.argv += ["--project-name", project_name]

    try:
        dg.main()
    except SystemExit:
        pass   # argparse calls sys.exit(0) on success — swallow it
    except Exception as e:
        print(f"  ⚠  Dashboard generation failed: {e}")
        import traceback; traceback.print_exc()
    finally:
        sys.argv = argv_orig   # always restore original argv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the literature validation pipeline.")
    parser.add_argument("--stage", nargs="+", type=int,
        help="Run only specific stages (1-6). Default: all.")
    parser.add_argument("--skip-pmc", action="store_true",
        help="Skip PMC full-text enrichment in stage 3.")
    parser.add_argument("--semantic-model", type=str, default=None,
        help="SentenceTransformer model for stage 4 semantic scoring.")
    parser.add_argument("--predictions", type=str, default=None,
        help="Path to predictions.yaml.")
    parser.add_argument("--output-dir", type=str, default=None,
        help="Directory for all output files.")
    parser.add_argument("--shared-cache", type=str, default=None,
        help="Shared PubMed cache directory.")
    parser.add_argument("--project-name", type=str, default="",
        help="Project name shown in the dashboard.")
    parser.add_argument("--no-dashboard", action="store_true",
        help="Skip interactive dashboard generation after stage 6.")
    args = parser.parse_args()

    _, out_dir = _resolve_dirs(args)

    # Copy predictions.yaml if a custom path was given
    if args.predictions:
        src = Path(args.predictions)
        dst = out_dir / "predictions.yaml"
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)
            print(f"Copied {src} → {dst}")

    # Point stage3 at the shared cache
    if args.shared_cache:
        os.environ["LITREV_CACHE"] = args.shared_cache
        cache_rec = Path(args.shared_cache) / "pubmed_records"
        cache_rec.mkdir(parents=True, exist_ok=True)
        stage3_retrieve_pubmed.RECORDS_DIR = cache_rec
    elif os.environ.get("LITREV_CACHE"):
        cache_rec = Path(os.environ["LITREV_CACHE"]) / "pubmed_records"
        cache_rec.mkdir(parents=True, exist_ok=True)
        stage3_retrieve_pubmed.RECORDS_DIR = cache_rec

    stages = args.stage or [1, 2, 3, 4, 5, 6]

    runners = {
        1: lambda: run_stage1(out_dir),
        2: lambda: run_stage2(out_dir),
        3: lambda: run_stage3(out_dir, skip_pmc=args.skip_pmc),
        4: lambda: run_stage4(out_dir, semantic_model=args.semantic_model),
        5: lambda: run_stage5(out_dir),
        6: lambda: run_stage6(out_dir),
    }

    start = time.time()
    for s in sorted(stages):
        runners[s]()

    # Generate interactive dashboard after stage 6 (unless suppressed)
    if 6 in stages and not args.no_dashboard:
        run_dashboard(out_dir, project_name=args.project_name)

    elapsed = time.time() - start
    print(f"\n=== Pipeline completed in {elapsed:.1f}s ===")
    print(f"Outputs: {out_dir}/output/")


if __name__ == "__main__":
    main()
