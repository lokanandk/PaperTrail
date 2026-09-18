"""
Stage 6 (v3): Generate publication-quality figures and a structured report.

Outputs:
  figures/tier_summary.png        — bar chart of evidence tiers
  figures/forest_plot.png         — per-prediction concordance with 95% CI,
                                    three markers per row (simple ○, weighted □,
                                    quality-weighted ◇), LOO range as error bars
  figures/evidence_volume.png     — n_relevant / n_informative per prediction
  figures/novelty_gap.png         — priority map for experimental validation
  figures/concordance_heatmap.png — heatmap by category
  figures/category_pooled.png     — meta-pooled concordance with CI95
  figures/sensitivity_loo.png     — leave-one-out sensitivity ranges
  figures/discordance_counts.png  — concordant vs opposite vs neutral per pred
  report.md                       — structured markdown report (incl. discordance)
  summary.csv                     — tabular per-prediction summary
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ─── Palette ────────────────────────────────────────────────────────────────
NAVY    = "#1B2A4A"
TEAL    = "#0D7377"
TEAL_LT = "#14A8AD"
GOLD    = "#B8860B"
SLATE   = "#4A5568"
STEEL   = "#718096"
RED     = "#C0392B"
GREEN   = "#2E7D32"
PURPLE  = "#5B6DAE"
FOG     = "#E8EDF4"
WHITE   = "#FFFFFF"

TIER_COLORS = {
    "STRONG":          GREEN,  "MODERATE":      TEAL,
    "MIXED":           GOLD,   "WEAK_SUPPORT":  "#9090A8",
    "WEAK_DISCORDANT": RED,    "DESCRIPTIVE":   PURPLE,
    "NO_DIRECTIONAL":  "#B0B0B0", "NO_INFORMATIVE": "#C8C8C8",
    "NONE":            "#D0D0D0",
}

CATEGORY_COLORS = {
    "folate":         TEAL,   "dserine":        PURPLE,  "gthrd":   GREEN,
    "tubular_injury": RED,    "mtor":           GOLD,    "stress":  "#8E5572",
    "polyamine":      "#3E8B8E", "noradrenergic": "#B0723F", "other": STEEL,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "axes.labelweight": "bold",
    "axes.edgecolor": SLATE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE,
})


def _to_bool(v) -> bool:
    """Safely coerce any JSON-deserialised bool-ish value to Python bool."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _to_float_or_none(v):
    """Safely coerce to float, returning None for None/str non-numeric."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_dataframe(scored: dict) -> pd.DataFrame:
    rows = []
    for pid, s in scored.items():
        p   = s["prediction"]
        loo = s.get("leave_one_out") or {}
        rows.append({
            "prediction_id":              pid,
            "category":                   p.get("category"),
            "entity":                     p.get("entity"),
            "disease":                    p.get("disease_context"),
            "cell_type":                  p.get("cell_type"),
            "novelty":                    p.get("novelty"),
            "expected_direction":         s["expected_direction"],
            "n_relevant":                 s.get("n_relevant"),
            "n_directional":              s.get("n_directional"),
            "n_informative":              s.get("n_informative"),
            "n_concordant":               s.get("n_concordant"),
            "n_opposite":                 s.get("n_opposite"),
            "n_neutral":                  s.get("n_neutral"),
            "concordance_simple":         _to_float_or_none(s.get("concordance_simple")),
            "concordance_weighted":       _to_float_or_none(s.get("concordance_weighted")),
            "concordance_quality_weighted": _to_float_or_none(
                                            s.get("concordance_quality_weighted")),
            "ci95_low":                   _to_float_or_none(s.get("ci95_low")),
            "ci95_high":                  _to_float_or_none(s.get("ci95_high")),
            "binomial_p":                 _to_float_or_none(s.get("binomial_p")),
            "loo_min":                    _to_float_or_none(loo.get("min")),
            "loo_max":                    _to_float_or_none(loo.get("max")),
            "loo_range":                  _to_float_or_none(loo.get("range")),
            "loo_fragile":                _to_bool(loo.get("fragile")),
            "tier":                       s["tier"],
            "prediction_type":            p.get("prediction_type", ""),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["category", "novelty", "tier"]).reset_index(drop=True)


# ─── Figures ────────────────────────────────────────────────────────────────

def fig_tier_summary(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    counts  = df["tier"].value_counts()
    order   = ["STRONG", "MODERATE", "MIXED", "WEAK_SUPPORT",
                "WEAK_DISCORDANT", "DESCRIPTIVE", "NO_INFORMATIVE",
                "NO_DIRECTIONAL", "NONE"]
    counts  = counts.reindex(order, fill_value=0)
    colors  = [TIER_COLORS.get(t, STEEL) for t in counts.index]
    bars    = ax.barh(counts.index, counts.values, color=colors,
                      edgecolor=SLATE, linewidth=0.4)
    for b, v in zip(bars, counts.values):
        if v > 0:
            ax.text(v + 0.3, b.get_y() + b.get_height() / 2, str(int(v)),
                    va="center", fontsize=9, color=NAVY, fontweight="bold")
    ax.set_xlabel("Number of predictions")
    ax.set_title("Evidence Tier Distribution Across Predictions", color=NAVY)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_forest(df: pd.DataFrame, out: Path):
    df_p = df[df["concordance_simple"].notna()].copy()
    if df_p.empty:
        return
    df_p = df_p.sort_values(["category", "concordance_quality_weighted"],
                             ascending=[True, True]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, max(4, 0.30 * len(df_p))))
    y = np.arange(len(df_p))
    ax.axvline(0.5, color=SLATE, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvspan(0.75, 1.0, color=GREEN, alpha=0.05)
    ax.axvspan(0.0,  0.25, color=RED,   alpha=0.05)

    for i, row in df_p.iterrows():
        col = CATEGORY_COLORS.get(row["category"], STEEL)
        if row["ci95_low"] is not None and row["ci95_high"] is not None:
            ax.plot([row["ci95_low"], row["ci95_high"]], [i, i],
                    color=col, alpha=0.35, linewidth=2.2)
        if row["loo_min"] is not None and row["loo_max"] is not None:
            ax.plot([row["loo_min"], row["loo_max"]], [i, i],
                    color=col, alpha=0.55, linewidth=4)
        if row["concordance_simple"] is not None:
            ax.scatter([row["concordance_simple"]], [i], marker="o", s=22,
                       color="white", edgecolor=col, linewidth=1.0, zorder=3)
        if row["concordance_weighted"] is not None:
            ax.scatter([row["concordance_weighted"]], [i], marker="s", s=26,
                       color=col, edgecolor=SLATE, linewidth=0.4, zorder=3)
        size = max(40, min(220, int((row["n_informative"] or 0) * 22)))
        if row["concordance_quality_weighted"] is not None:
            ax.scatter([row["concordance_quality_weighted"]], [i], marker="D",
                       s=size, color=col, edgecolor=NAVY, linewidth=0.6, zorder=4)
        if row.get("binomial_p") is not None and row["binomial_p"] < 0.05:
            stars = ("*" if row["binomial_p"] >= 0.01
                     else ("**" if row["binomial_p"] >= 0.001 else "***"))
            ax.text(min(0.99, (row["concordance_quality_weighted"] or 0) + 0.04),
                    i, stars, fontsize=9, color=NAVY, fontweight="bold", va="center")
        if _to_bool(row.get("loo_fragile")):
            ax.text(-0.04, i, "⚠", fontsize=10, color=GOLD, va="center",
                    ha="right", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(df_p["prediction_id"], fontsize=7)
    ax.set_xlabel("Concordance (○ simple · □ relevance-weighted · ◇ quality-weighted, n-sized)\n"
                  "thin line = Wilson 95% CI · thick line = leave-one-out range · ⚠ = LOO-fragile")
    ax.set_xlim(-0.10, 1.15)
    ax.set_title("Predictions: Literature Concordance with 95% CI and LOO Sensitivity",
                 color=NAVY, pad=10)
    handles = [mpatches.Patch(color=v, label=k.title())
               for k, v in CATEGORY_COLORS.items()
               if k in df_p["category"].unique()]
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_evidence_volume(df: pd.DataFrame, out: Path):
    df_p = df.sort_values(["category", "n_relevant"],
                           ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, max(5, 0.30 * len(df_p))))
    y    = np.arange(len(df_p))
    rels = df_p["n_relevant"].fillna(0).values
    inf  = df_p["n_informative"].fillna(0).values
    cols = [CATEGORY_COLORS.get(c, STEEL) for c in df_p["category"]]
    ax.barh(y, rels, color=cols, alpha=0.30, edgecolor=SLATE, linewidth=0.4)
    ax.barh(y, inf,  color=cols, alpha=0.95, edgecolor=NAVY,  linewidth=0.4)
    for i, (r, n) in enumerate(zip(rels, inf)):
        if r > 0:
            ax.text(r + 0.3, i, f"{int(r)}/{int(n)}", va="center",
                    fontsize=7, color=NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels(df_p["prediction_id"], fontsize=6.5)
    ax.set_xlabel("Records (relevant total / informative for concordance test)")
    ax.set_title("Literature Volume per Prediction", color=NAVY)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_novelty_gap(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    df_p    = df.copy()
    novelty_x = df_p["novelty"].map({"known": 0, "extending": 1, "novel": 2}).fillna(1)
    support_y = (df_p["concordance_quality_weighted"].fillna(0)
                 * np.log1p(df_p["n_informative"].fillna(0)))
    np.random.seed(7)
    jitter = np.random.uniform(-0.06, 0.06, len(df_p))
    cols   = [CATEGORY_COLORS.get(c, STEEL) for c in df_p["category"]]
    ax.scatter(novelty_x + jitter, support_y, c=cols, s=80,
               edgecolor=NAVY, linewidth=0.4, alpha=0.85)
    for _, row in df_p.iterrows():
        nx  = {"known": 0, "extending": 1, "novel": 2}.get(row["novelty"], 1)
        sy  = (row["concordance_quality_weighted"] or 0) * np.log1p(row["n_informative"] or 0)
        if row["novelty"] == "novel" and sy < 0.5:
            ax.annotate(row["prediction_id"], xy=(nx, sy),
                        xytext=(nx + 0.18, sy + 0.18),
                        fontsize=6, color=RED,
                        arrowprops=dict(arrowstyle="-", color=SLATE, lw=0.5))
    ax.axhspan(0, 0.5, xmin=0.66, xmax=1.0, color=RED, alpha=0.07)
    ax.text(2, 0.25, "EXPERIMENTAL PRIORITY ZONE\n(novel + weak literature support)",
            ha="center", color=RED, fontsize=9, fontweight="bold")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Known", "Extending", "Novel"])
    ax.set_xlabel("Prediction novelty")
    ax.set_ylabel("Literature support index\n"
                  "(quality-weighted concordance × log(n_informative+1))")
    ax.set_title("Validation Priority Map", color=NAVY)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_category_pooled(pooled: dict, out: Path):
    cats = [c for c, p in pooled.items() if p.get("pooled_concordance") is not None]
    if not cats:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(cats))))
    y       = np.arange(len(cats))
    colors  = [CATEGORY_COLORS.get(c, STEEL) for c in cats]
    centres = [pooled[c]["pooled_concordance"] for c in cats]
    los     = [pooled[c]["ci95"][0] or 0 for c in cats]
    his     = [pooled[c]["ci95"][1] or 1 for c in cats]
    for i, (c, lo, hi, col) in enumerate(zip(centres, los, his, colors)):
        ax.plot([lo, hi], [i, i], color=col, linewidth=3, alpha=0.7)
        ax.scatter([c], [i], s=180, color=col, edgecolor=NAVY,
                   linewidth=0.8, zorder=3)
        n_total = pooled[cats[i]]["n_informative_total"]
        n_conc  = pooled[cats[i]]["n_concordant_total"]
        p       = pooled[cats[i]]["binomial_p"]
        label   = f" {n_conc}/{n_total} ({c:.0%}) p={p}"
        ax.text(min(1.02, hi + 0.02), i, label, va="center", fontsize=8, color=NAVY)
    ax.axvline(0.5, color=SLATE, linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ").title() for c in cats], fontsize=10)
    ax.set_xlabel("Meta-pooled concordance with 95% CI")
    ax.set_xlim(-0.05, 1.4)
    ax.set_title("Cross-prediction Meta-pooling by Mechanistic Category", color=NAVY)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_concordance_heatmap(df: pd.DataFrame, out: Path):
    df2 = df[df["concordance_quality_weighted"].notna()].copy()
    if df2.empty:
        return
    pivot = df2.pivot_table(values="concordance_quality_weighted",
                             index="prediction_id", columns="category",
                             aggfunc="first")
    fig, ax = plt.subplots(figsize=(7.5, max(4.5, 0.28 * len(pivot))))
    cmap    = LinearSegmentedColormap.from_list(
        "rg", [(0.0, RED), (0.5, "#F5F5F5"), (1.0, GREEN)])
    sns.heatmap(pivot, cmap=cmap, vmin=0, vmax=1, center=0.5,
                annot=True, fmt=".2f", linewidths=0.4, linecolor=FOG,
                ax=ax, cbar_kws={"label": "Quality-weighted concordance",
                                 "shrink": 0.6},
                annot_kws={"fontsize": 7})
    ax.set_title("Per-prediction Concordance Grouped by Category", color=NAVY)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity_loo(df: pd.DataFrame, out: Path):
    """Plot LOO range per prediction; flag fragile in red."""
    df_p = df[df["loo_range"].notna()].copy()
    if df_p.empty:
        return
    df_p = df_p.sort_values("loo_range", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.24 * len(df_p))))
    y    = np.arange(len(df_p))
    cols = [RED if _to_bool(f) else TEAL for f in df_p["loo_fragile"]]
    bars = ax.barh(y, df_p["loo_range"], color=cols, edgecolor=SLATE, linewidth=0.4)
    ax.axvline(0.25, color=GOLD, linestyle="--", linewidth=0.8,
               label="Fragility threshold (0.25)")
    for i, (rng, frag) in enumerate(zip(df_p["loo_range"], df_p["loo_fragile"])):
        if _to_bool(frag):
            ax.text(rng + 0.01, i, "⚠", fontsize=10, color=RED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(df_p["prediction_id"], fontsize=7)
    ax.set_xlabel("Leave-one-out concordance range")
    ax.set_title("Sensitivity Analysis: How Fragile Is Each Concordance Estimate?",
                 color=NAVY)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_discordance_counts(df: pd.DataFrame, out: Path):
    """Stacked bar: concordant / opposite / neutral per prediction."""
    df_p = df[df["n_directional"].fillna(0) > 0].copy()
    if df_p.empty:
        return
    df_p = df_p.sort_values(
        ["n_concordant", "n_opposite"], ascending=[False, True]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(4, 0.24 * len(df_p))))
    y = np.arange(len(df_p))

    def _safe_int(x):
        if x is None:
            return 0
        try:
            if pd.isna(x):
                return 0
        except (TypeError, ValueError):
            pass
        try:
            return int(x)
        except (TypeError, ValueError):
            return 0

    conc = df_p["n_concordant"].apply(_safe_int).values
    opp  = df_p["n_opposite"].apply(_safe_int).values
    neu  = df_p["n_neutral"].apply(_safe_int).values

    ax.barh(y, conc, color=GREEN, label="Concordant",
            edgecolor=SLATE, linewidth=0.3)
    ax.barh(y, opp,  left=conc,         color=RED,      label="Opposite",
            edgecolor=SLATE, linewidth=0.3)
    ax.barh(y, neu,  left=conc + opp,   color="#C8C8C8", label="Neutral",
            edgecolor=SLATE, linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(df_p["prediction_id"], fontsize=7)
    ax.set_xlabel("Number of records")
    ax.set_title("Per-prediction Concordance Composition", color=NAVY)
    ax.legend(loc="lower right", frameon=True)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─── Markdown report ────────────────────────────────────────────────────────

def write_report_md(df: pd.DataFrame, scored: dict, pooled: dict, out_path: Path):
    """
    Generate the structured markdown report.

    Changes vs original:
    - Uses best_excerpt (from stage4) instead of abstract_excerpt[:400] when
      available. best_excerpt is the most directionally informative sentence(s),
      potentially sourced from PMC full text.
    - Labels excerpt source as [full text] or [abstract] in the report.
    - Flags direction_conflict (abstract vs PMC full-text disagree) with ⚡.
    - Emits Title + excerpt for ALL papers in Per-prediction details (not just
      discordant ones), so that extract_evidence_keywords.py can show sentence
      context for concordant and neutral papers too.
    """
    lines = []
    lines.append("# PaperTrail Literature Validation Report\n")
    lines.append("Comprehensive automated literature concordance analysis "
                 "(quality-weighted, hedge-discounted, LOO-sensitivity-tested).\n")
    lines.append("## Overall Summary\n")
    lines.append(f"- **Total predictions assessed:** {len(df)}")

    tier_counts = df["tier"].value_counts().to_dict()
    for t in ["STRONG", "MODERATE", "MIXED", "WEAK_SUPPORT", "WEAK_DISCORDANT",
              "DESCRIPTIVE", "NO_INFORMATIVE", "NO_DIRECTIONAL", "NONE"]:
        c = tier_counts.get(t, 0)
        if c > 0:
            lines.append(f"- **{t.replace('_', ' ').title()}:** {c}")
    n_fragile = int(df["loo_fragile"].apply(_to_bool).sum())
    if n_fragile:
        lines.append(f"- **LOO-fragile (single-paper-driven):** {n_fragile}")

    # PMC coverage note (new — only shown if any records have full-text)
    n_pmc = sum(
        1 for s in scored.values()
        for r in s.get("summary_evidence", [])
        if r.get("has_pmc")
    )
    n_total_ev = sum(len(s.get("summary_evidence", [])) for s in scored.values())
    if n_pmc:
        lines.append(f"- **PMC full-text enriched:** {n_pmc}/{n_total_ev} "
                     f"evidence records ({100 * n_pmc // max(n_total_ev, 1)}% open-access)")
    lines.append("")

    lines.append("## Cross-prediction meta-pooling by category\n")
    lines.append("| Category | n_pred | n_inform | n_concord | Pooled | 95% CI | p |")
    lines.append("|---|---|---|---|---|---|---|")
    for cat, p in pooled.items():
        if p["pooled_concordance"] is not None:
            ci = p["ci95"]
            lines.append(f"| {cat} | {p['n_predictions']} | {p['n_informative_total']} | "
                         f"{p['n_concordant_total']} | {p['pooled_concordance']:.0%} | "
                         f"{ci[0]:.2f}–{ci[1]:.2f} | {p['binomial_p']} |")
    lines.append("")

    priority = df[(df["novelty"] == "novel")
                  & (df["tier"].isin(["NONE", "NO_DIRECTIONAL",
                                      "NO_INFORMATIVE", "WEAK_SUPPORT",
                                      "WEAK_DISCORDANT"]))]
    if not priority.empty:
        # If prediction_type varies across the novel predictions, group by it so
        # each type gets its own section.  This is generic: any value the user
        # places in the prediction_type field of predictions.yaml works.
        type_col = "prediction_type"
        has_types = (
            type_col in priority.columns
            and priority[type_col].notna().any()
            and priority[type_col].nunique() > 1
        )

        def _priority_row(lines, r, df):
            """Append one priority-list entry, with a cross-reference note when
            the same entity has a well-supported prediction elsewhere."""
            entity  = r["entity"]
            related = df[(df["entity"] == entity) &
                         (df["tier"].isin(["STRONG", "MODERATE"])) &
                         (df["prediction_id"] != r["prediction_id"])]
            context_note = ""
            if not related.empty:
                rel  = related.iloc[0]
                context_note = (
                    f" *(Note: related prediction `{rel['prediction_id']}` has "
                    f"`{rel['tier']}` support — the entity is known but this "
                    f"specific comparison is novel)*"
                )
            lines.append(
                f"- **{r['prediction_id']}** ({r['entity']}, {r['disease']}, "
                f"{r['cell_type']}) — tier `{r['tier']}`, "
                f"n_relevant={r['n_relevant']}{context_note}"
            )

        if has_types:
            for ptype, group in priority.groupby(type_col, dropna=False):
                type_label = str(ptype).strip() if ptype else "Standard"
                lines.append(f"## Novel Predictions — {type_label}\n")
                lines.append(
                    f"Novel predictions with prediction_type **{type_label}** "
                    "that currently have limited or no prior literature support. "
                    "These are candidates for experimental validation.\n"
                )
                for _, r in group.iterrows():
                    _priority_row(lines, r, df)
                lines.append("")
        else:
            lines.append("## Novel Claims — Limited Prior Literature (Experimental Priority)\n")
            lines.append(
                "Novel predictions without strong prior literature support. "
                "These are experimental priorities: targeted validation could "
                "directly confirm or refute each prediction.\n"
            )
            for _, r in priority.iterrows():
                _priority_row(lines, r, df)
            lines.append("")

    strong = df[df["tier"] == "STRONG"]
    if not strong.empty:
        lines.append("## Strong Literature Support — Cite Rather Than Re-validate\n")
        for _, r in strong.iterrows():
            lines.append(f"- **{r['prediction_id']}** ({r['entity']}, {r['disease']}) — "
                         f"qw-conc={r['concordance_quality_weighted']}, "
                         f"n_informative={r['n_informative']}, p={r['binomial_p']}")
        lines.append("")

    # ── Discordance investigations ──────────────────────────────────────────
    discordance_count = sum(1 for s in scored.values() if s.get("discordance_reports"))
    if discordance_count:
        lines.append("## Discordance Investigations\n")
        lines.append("These predictions have at least one record whose direction "
                     "OPPOSES the prediction. Each warrants individual review.\n")
        for pid, s in scored.items():
            if not s.get("discordance_reports"):
                continue
            lines.append(f"### {pid} — {s['prediction']['entity']} "
                         f"({s['prediction']['disease_context']})\n")
            lines.append(f"Expected: {s['expected_direction']} · "
                         f"Concordance (simple): {s.get('concordance_simple')} · "
                         f"n_opposite={s.get('n_opposite')}\n")
            for d in s["discordance_reports"][:3]:
                conflict_flag = " ⚡" if d.get("direction_conflict") else ""
                lines.append(f"- **PMID [{d['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{d['pmid']}/):** "
                             f"_{d.get('title', '')[:120]}_ "
                             f"({d.get('journal', '')[:30]}, {d.get('year', '')}){conflict_flag}")
                lines.append(f"  - Extracted direction: **{d['extracted_direction']}**")
                # Prefer best_excerpt (may be from PMC full text) over abstract_excerpt
                exc_text   = (d.get("best_excerpt") or d.get("abstract_excerpt") or "")[:400]
                exc_source = d.get("excerpt_source", "abstract")
                src_label  = " [full text]" if exc_source == "full_text" else ""
                if exc_text:
                    lines.append(f"  - Abstract excerpt{src_label}: {exc_text}…")
                if d.get("direction_conflict"):
                    lines.append("  - ⚡ Note: abstract and PMC full-text give conflicting "
                                 "direction signals — review manually")
                lines.append("")

    # ── Per-prediction details ──────────────────────────────────────────────
    # KEY CHANGE: emit Title + best_excerpt for ALL papers in the table
    # (not just discordant ones).  This lets extract_evidence_keywords.py
    # show real sentence context for concordant and neutral papers.
    lines.append("## Per-prediction details\n")
    for pid, s in scored.items():
        p = s["prediction"]
        lines.append(f"### {pid} — {p['entity']} ({p['disease_context']}, {p['cell_type']})\n")
        lines.append(f"**Prediction note:** {p.get('prediction_note', '')}")
        lines.append(f"**Novelty:** {p.get('novelty', '?')} · "
                     f"**Expected direction:** {p['direction']}")
        lines.append(f"**Tier:** `{s['tier']}` · **Reason:** {s['tier_reason']}")
        loo = s.get("leave_one_out") or {}
        if loo.get("range") is not None:
            frag = " ⚠ fragile" if _to_bool(loo.get("fragile")) else ""
            lines.append(f"**LOO range:** {loo.get('min')}–{loo.get('max')} "
                         f"(Δ={loo.get('range'):.2f}){frag}")
        if s.get("magnitude_pool"):
            mp = s["magnitude_pool"]
            lines.append(f"**Pooled fold-changes (n={mp['n_records_with_fold']}):** "
                         f"median={mp['median_fold']:.2f}, IQR={mp['iqr_fold']}")
        if s.get("summary_evidence"):
            lines.append("\n**Top supporting records:**\n")
            lines.append("| PMID | Year | Journal | Direction | Relation | Hedged | Quality |")
            lines.append("|---|---|---|---|---|---|---|")
            for r in s["summary_evidence"][:10]:
                jr        = (r.get("journal") or "")[:32]
                rel       = r.get("relation", "?")
                rel_emoji = "✓" if rel == "concordant" else ("✗" if rel == "opposite" else "—")
                hedge     = ""
                if r.get("hedge_discount"):
                    hedge = f"{int(r['hedge_discount'] * 100)}%"
                qw       = r.get("quality_weight", "")
                pmc_flag = " 🔓" if r.get("has_pmc") else ""
                lines.append(f"| [{r['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/) | "
                             f"{r.get('year', '')} | {jr} | "
                             f"{r.get('extracted_direction', '?')} | {rel_emoji} | {hedge} | {qw}{pmc_flag} |")

                # Emit title + best excerpt for ALL papers
                title = (r.get("title") or "").strip()
                exc   = (r.get("best_excerpt") or r.get("abstract_excerpt") or "").strip()
                src   = r.get("excerpt_source", "abstract")
                if title:
                    title_clean = title.replace("_", " ").replace("*", "").replace("|", "")
                    lines.append(f"  - Title: _{title_clean}_")
                if exc:
                    exc_clean = exc[:500].replace("|", "").rstrip()
                    src_label = " [full text]" if src == "full_text" else " [abstract]"
                    lines.append(f"  - Abstract excerpt{src_label}: {exc_clean}…")

        lines.append("")

    out_path.write_text("\n".join(lines))


# ─── Orchestration ──────────────────────────────────────────────────────────

def generate(scored_path: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    with open(scored_path) as f:
        data = json.load(f)

    scored = data.get("predictions", data)
    pooled = data.get("category_pooled", {})
    df     = to_dataframe(scored)

    df.to_csv(out_dir / "summary.csv", index=False)

    fig_tier_summary(df,      fig_dir / "tier_summary.png")
    fig_forest(df,            fig_dir / "forest_plot.png")
    fig_evidence_volume(df,   fig_dir / "evidence_volume.png")
    fig_novelty_gap(df,       fig_dir / "novelty_gap.png")
    fig_concordance_heatmap(df, fig_dir / "concordance_heatmap.png")
    fig_category_pooled(pooled, fig_dir / "category_pooled.png")
    fig_sensitivity_loo(df,   fig_dir / "sensitivity_loo.png")
    fig_discordance_counts(df, fig_dir / "discordance_counts.png")

    write_report_md(df, scored, pooled, out_dir / "report.md")
    print(f"Report and figures written to {out_dir}/")


if __name__ == "__main__":
    base = Path(__file__).parent
    generate(base / "scored_predictions.json", base / "output")
