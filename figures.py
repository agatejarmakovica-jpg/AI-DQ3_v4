#!/usr/bin/env python3
"""
AI-DQ3 publication figures
==========================

Generates publication-quality figures (vector PDF + 300 dpi PNG) from the
per-dataset result CSVs written by pipeline.py. One figure per analytical point,
plus a multi-panel overview suitable as a manuscript "Figure 1".

Usage
-----
    python figures.py --results results --out figures
    python figures.py --results results --out figures --dataset "Diabetes Missing Data"

Design notes
------------
* Colour-blind-safe Okabe-Ito palette, fixed semantics:
  Completeness=blue, Accuracy=vermillion, Reuse=green; baseline=grey,
  proposed (semantic)=blue.
* No top/right spines, value-axis grid only, direct value labels, serif-free
  clean typography. Figures sized for single/double journal columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Okabe-Ito colour-blind-safe palette
C_BLUE = "#0072B2"     # Completeness / proposed
C_VERM = "#D55E00"     # Accuracy
C_GREEN = "#009E73"    # Reuse readiness
C_GREY = "#9A9A9A"     # baseline / neutral
C_ORANGE = "#E69F00"
C_PURPLE = "#CC79A7"
C_INK = "#222222"
DIM_COLORS = {"C": C_BLUE, "A": C_VERM, "R": C_GREEN}
DIM_LABELS = {"C": "Completeness", "A": "Accuracy", "R": "Reuse readiness"}


# When False (default), figures carry NO embedded title or suptitle so they can
# be dropped straight into a manuscript (the caption lives in the paper text).
# Panel letters in the overview are kept either way, as captions reference them.
SHOW_TITLES = False


def _title(ax, text: str, **kw) -> None:
    if SHOW_TITLES:
        ax.set_title(text, **kw)


def _suptitle(fig, text: str, **kw) -> None:
    if SHOW_TITLES:
        fig.suptitle(text, **kw)


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": C_INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E2E2E2",
        "grid.linewidth": 0.7,
        "xtick.color": C_INK,
        "ytick.color": C_INK,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "figure.titleweight": "bold",
    })


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


def _read(d: Path, name: str) -> Optional[pd.DataFrame]:
    p = d / name
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df if not df.empty else None
    except Exception:
        return None


def _barlabels(ax, bars, fmt="{:.2f}", pad=3, fontsize=8.5):
    for b in bars:
        w = b.get_width() if b.get_width() else b.get_height()
    ax.bar_label(bars, fmt=fmt, padding=pad, fontsize=fontsize, color=C_INK)


# -----------------------------------------------------------------------------
# Individual figures
# -----------------------------------------------------------------------------


def fig_quality_profile(d: Path, out: Path, dataset: str) -> None:
    prof = _read(d, "rq3_quality_profile.csv")
    if prof is None:
        return
    q = prof.iloc[0]
    dims = ["C(D)", "A(D)", "R(D)", "composite_quality_index"]
    labels = ["Completeness\nC(D)", "Accuracy\nA(D)", "Reuse readiness\nR(D)", "Composite\nindex"]
    vals = [float(q[k]) for k in dims]
    thr = {"C(D)": 0.95, "A(D)": 0.95, "R(D)": 0.80, "composite_quality_index": 0.85}
    colors = [C_BLUE, C_VERM, C_GREEN, C_INK]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(dims))
    bars = ax.bar(x, vals, color=colors, width=0.62, edgecolor="white", linewidth=0.8, zorder=3)
    for xi, k in zip(x, dims):
        ax.hlines(thr[k], xi - 0.34, xi + 0.34, color=C_INK, linestyles=(0, (4, 3)), linewidth=1.1, zorder=4)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9, color=C_INK)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08); ax.set_ylabel("Score (0–1)")
    _title(ax, f"Dataset-level quality profile — {dataset}")
    ax.grid(axis="x", visible=False)
    ax.legend([Line2D([0], [0], color=C_INK, linestyle=(0, (4, 3)), linewidth=1.1)],
              ["Acceptance threshold"], loc="lower right")
    _save(fig, out, "fig_quality_profile")


def fig_component_breakdown(d: Path, out: Path, dataset: str) -> None:
    acc = _read(d, "accuracy_components.csv")
    reuse = _read(d, "reuse_facets.csv")
    comp = _read(d, "completeness_components.csv")
    if acc is None and reuse is None and comp is None:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    def hbar(ax, df, label_col, score_col, color, title):
        if df is None or df.empty:
            ax.axis("off"); _title(ax, title); return
        df = df.iloc[::-1]
        y = np.arange(len(df))
        bars = ax.barh(y, df[score_col], color=color, edgecolor="white", linewidth=0.7, zorder=3, height=0.66)
        ax.set_yticks(y); ax.set_yticklabels([str(v).replace("_", " ") for v in df[label_col]])
        ax.set_xlim(0, 1.12); ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5, color=C_INK)
        _title(ax, title); ax.grid(axis="y", visible=False); ax.set_xlabel("Score")

    hbar(axes[0], comp, "component", "score", C_BLUE, "Completeness components")
    hbar(axes[1], acc, "component", "score", C_VERM, "Accuracy components")
    hbar(axes[2], reuse, "facet", "score", C_GREEN, "Reuse-readiness facets")
    _suptitle(fig, f"Dimension component breakdown — {dataset}", y=1.02)
    _save(fig, out, "fig_component_breakdown")


def fig_anomaly_comparison(d: Path, out: Path, dataset: str) -> None:
    cmp = _read(d, "anomaly_baseline_comparison.csv")
    if cmp is None or len(cmp) < 2:
        return
    g, s = cmp.iloc[0], cmp.iloc[1]
    cats = ["Variables\nused", "Rows flagged", "Flagged rate (%)"]
    generic = [int(g["n_columns"]), int(g["flagged_rows"]), 100 * float(g["flagged_rate"])]
    proposed = [int(s["n_columns"]), int(s["flagged_rows"]), 100 * float(s["flagged_rate"])]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(cats)); w = 0.38
    b1 = ax.bar(x - w / 2, generic, w, label="Generic (uniform, all numeric)",
                color=C_GREY, edgecolor="white", linewidth=0.8, zorder=3)
    b2 = ax.bar(x + w / 2, proposed, w, label="Semantic + HITL (proposed)",
                color=C_BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar_label(b1, fmt="%.1f", padding=3, fontsize=8.5, color=C_INK)
    ax.bar_label(b2, fmt="%.1f", padding=3, fontsize=8.5, color=C_INK)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("Count / rate"); ax.grid(axis="x", visible=False)
    _title(ax, f"RQ2 · Anomaly detection: uniform vs semantic — {dataset}")
    ax.legend(loc="upper right")
    _save(fig, out, "fig_rq2_anomaly_comparison")


def fig_missingness(d: Path, out: Path, dataset: str, top: int = 12) -> None:
    miss = _read(d, "missingness_comparison.csv")
    if miss is None:
        return
    miss = miss.sort_values("semantic_adjusted_unusable_rate", ascending=True).tail(top)
    y = np.arange(len(miss))
    fig, ax = plt.subplots(figsize=(6.8, max(3.0, 0.42 * len(miss) + 1.2)))
    tech = 100 * miss["technical_missing_rate"]
    extra = 100 * (miss["semantic_adjusted_unusable_rate"] - miss["technical_missing_rate"])
    b1 = ax.barh(y, tech, color=C_BLUE, edgecolor="white", linewidth=0.6, zorder=3, height=0.66, label="Technical missing")
    b2 = ax.barh(y, extra, left=tech, color=C_ORANGE, edgecolor="white", linewidth=0.6, zorder=3, height=0.66,
                 label="Semantic-unusable adjustment")
    ax.set_yticks(y); ax.set_yticklabels(miss["column"])
    ax.set_xlabel("Unusable values (% of records)"); ax.grid(axis="y", visible=False)
    for yi, total in zip(y, 100 * miss["semantic_adjusted_unusable_rate"]):
        ax.text(total + 0.6, yi, f"{total:.1f}%", va="center", fontsize=8.5, color=C_INK)
    ax.set_xlim(0, min(100, max(100 * miss["semantic_adjusted_unusable_rate"]) * 1.18 + 4))
    _title(ax, f"Completeness · per-variable missingness — {dataset}")
    ax.legend(loc="lower right")
    _save(fig, out, "fig_completeness_missingness")


def fig_triage(d: Path, out: Path, dataset: str, top: int = 10) -> None:
    reg = _read(d, "hitl_triage_register.csv")
    if reg is None:
        return
    reg = reg.head(top).iloc[::-1]
    labels = []
    for _, r in reg.iterrows():
        col = f" · {r['column']}" if isinstance(r["column"], str) and r["column"] and r["column"].lower() != "nan" else ""
        labels.append(f"{str(r['issue_type']).replace('_', ' ')}{col}")
    colors = [DIM_COLORS.get(str(dm), C_GREY) for dm in reg["dimension"]]
    y = np.arange(len(reg))
    fig, ax = plt.subplots(figsize=(7.4, max(3.0, 0.46 * len(reg) + 1.0)))
    bars = ax.barh(y, reg["priority_score"], color=colors, edgecolor="white", linewidth=0.7, zorder=3, height=0.68)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.bar_label(bars, labels=[f"n={int(c)}" for c in reg["count"]], padding=3, fontsize=8, color=C_INK)
    ax.set_xlabel("Triage priority score"); ax.grid(axis="y", visible=False)
    _title(ax, f"HITL triage — top {top} candidates — {dataset}")
    handles = [Line2D([0], [0], marker="s", linestyle="", markerfacecolor=DIM_COLORS[k],
                      markeredgecolor="white", markersize=9, label=DIM_LABELS[k]) for k in ["C", "A", "R"]]
    ax.legend(handles=handles, loc="lower right", title="Dimension")
    _save(fig, out, "fig_hitl_triage")


def fig_controlled_error(d: Path, out: Path, dataset: str) -> None:
    ce = _read(d, "controlled_error_baseline.csv")
    if ce is None:
        return
    metrics = ["precision", "recall", "f1"]
    methods = ce["method"].tolist()
    name_map = {"generic_anomaly_delta": "Generic", "semantic_hitl_anomaly_delta": "Semantic + HITL"}
    colors = {"generic_anomaly_delta": C_GREY, "semantic_hitl_anomaly_delta": C_BLUE}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.arange(len(metrics)); w = 0.8 / max(len(methods), 1)
    for i, (_, row) in enumerate(ce.iterrows()):
        vals = [float(row[m]) for m in metrics]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w,
                      label=name_map.get(row["method"], row["method"]),
                      color=colors.get(row["method"], C_ORANGE), edgecolor="white", linewidth=0.8, zorder=3)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8, color=C_INK)
    ax.set_xticks(x); ax.set_xticklabels(["Precision", "Recall", "F1"])
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score"); ax.grid(axis="x", visible=False)
    _title(ax, f"Controlled error injection — detector validation — {dataset}")
    ax.legend(loc="upper right")
    _save(fig, out, "fig_controlled_error_validation")


def fig_sensitivity(d: Path, out: Path, dataset: str) -> None:
    s = _read(d, "weight_sensitivity.csv")
    if s is None:
        return
    s = s.iloc[::-1]
    y = np.arange(len(s))
    colors = [C_BLUE if m else C_GREY for m in s["meets_threshold"]]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    bars = ax.barh(y, s["composite"], color=colors, edgecolor="white", linewidth=0.7, zorder=3, height=0.62)
    thr = 0.85
    ax.axvline(thr, color=C_INK, linestyle=(0, (4, 3)), linewidth=1.1, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([str(v).replace("_", " ") for v in s["scenario"]])
    ax.set_xlim(0, 1.05); ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5, color=C_INK)
    ax.set_xlabel("Composite quality index"); ax.grid(axis="y", visible=False)
    _title(ax, f"Weight sensitivity of composite index — {dataset}")
    ax.text(thr + 0.005, len(s) - 0.4, "threshold 0.85", fontsize=8, color=C_INK)
    handles = [Line2D([0], [0], marker="s", linestyle="", markerfacecolor=C_BLUE, markeredgecolor="white",
                      markersize=9, label="meets threshold"),
               Line2D([0], [0], marker="s", linestyle="", markerfacecolor=C_GREY, markeredgecolor="white",
                      markersize=9, label="below threshold")]
    ax.legend(handles=handles, loc="lower right")
    _save(fig, out, "fig_weight_sensitivity")


def fig_overview(d: Path, out: Path, dataset: str) -> None:
    """Multi-panel manuscript Figure 1 combining the four headline panels."""
    prof = _read(d, "rq3_quality_profile.csv")
    cmp = _read(d, "anomaly_baseline_comparison.csv")
    miss = _read(d, "missingness_comparison.csv")
    ce = _read(d, "controlled_error_baseline.csv")
    if prof is None:
        return
    q = prof.iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))

    # (a) profile
    ax = axes[0, 0]
    dims = ["C(D)", "A(D)", "R(D)", "composite_quality_index"]
    labels = ["C(D)", "A(D)", "R(D)", "Composite"]
    vals = [float(q[k]) for k in dims]
    bars = ax.bar(np.arange(4), vals, color=[C_BLUE, C_VERM, C_GREEN, C_INK], width=0.62,
                  edgecolor="white", linewidth=0.8, zorder=3)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5, color=C_INK)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score"); ax.grid(axis="x", visible=False)
    ax.set_title("(a) Quality profile", loc="left", fontsize=10.5)

    # (b) anomaly comparison
    ax = axes[0, 1]
    if cmp is not None and len(cmp) >= 2:
        g, s = cmp.iloc[0], cmp.iloc[1]
        cats = ["Variables", "Rows flagged"]; x = np.arange(2); w = 0.38
        b1 = ax.bar(x - w / 2, [int(g["n_columns"]), int(g["flagged_rows"])], w, color=C_GREY,
                    edgecolor="white", linewidth=0.8, zorder=3, label="Generic")
        b2 = ax.bar(x + w / 2, [int(s["n_columns"]), int(s["flagged_rows"])], w, color=C_BLUE,
                    edgecolor="white", linewidth=0.8, zorder=3, label="Semantic + HITL")
        ax.bar_label(b1, fmt="%d", padding=2, fontsize=8, color=C_INK)
        ax.bar_label(b2, fmt="%d", padding=2, fontsize=8, color=C_INK)
        ax.set_xticks(x); ax.set_xticklabels(cats); ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right")
    ax.set_title("(b) RQ2 · anomaly detection", loc="left", fontsize=10.5)

    # (c) missingness
    ax = axes[1, 0]
    if miss is not None:
        m = miss.sort_values("semantic_adjusted_unusable_rate", ascending=True).tail(8)
        y = np.arange(len(m))
        ax.barh(y, 100 * m["semantic_adjusted_unusable_rate"], color=C_BLUE, edgecolor="white",
                linewidth=0.6, zorder=3, height=0.66)
        ax.set_yticks(y); ax.set_yticklabels(m["column"], fontsize=8); ax.grid(axis="y", visible=False)
        ax.set_xlabel("Unusable (%)")
    ax.set_title("(c) Completeness · missingness", loc="left", fontsize=10.5)

    # (d) controlled error
    ax = axes[1, 1]
    if ce is not None:
        metrics = ["precision", "recall", "f1"]; x = np.arange(3)
        name_map = {"generic_anomaly_delta": "Generic", "semantic_hitl_anomaly_delta": "Semantic + HITL"}
        colors = {"generic_anomaly_delta": C_GREY, "semantic_hitl_anomaly_delta": C_BLUE}
        w = 0.8 / max(len(ce), 1)
        for i, (_, row) in enumerate(ce.iterrows()):
            ax.bar(x + (i - (len(ce) - 1) / 2) * w, [float(row[m]) for m in metrics], w,
                   color=colors.get(row["method"], C_ORANGE), edgecolor="white", linewidth=0.8, zorder=3,
                   label=name_map.get(row["method"], row["method"]))
        ax.set_xticks(x); ax.set_xticklabels(["P", "R", "F1"]); ax.set_ylim(0, 1.1)
        ax.grid(axis="x", visible=False); ax.legend(loc="upper right")
    ax.set_title("(d) Controlled-error validation", loc="left", fontsize=10.5)

    _suptitle(fig, f"AI-DQ3 assessment overview — {dataset}", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out, "fig_overview")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


def build_for_dataset(results_dir: Path, out_root: Path, dataset: str) -> Path:
    d = results_dir / dataset
    out = out_root / dataset
    set_style()
    fig_quality_profile(d, out, dataset)
    fig_component_breakdown(d, out, dataset)
    fig_anomaly_comparison(d, out, dataset)
    fig_missingness(d, out, dataset)
    fig_triage(d, out, dataset)
    fig_controlled_error(d, out, dataset)
    fig_sensitivity(d, out, dataset)
    fig_overview(d, out, dataset)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI-DQ3 publication figures from result CSVs.")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("figures"))
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset folder name (default: all).")
    parser.add_argument("--titles", action="store_true",
                        help="Embed titles in figures (default off, for manuscript-ready output).")
    args = parser.parse_args()
    global SHOW_TITLES
    SHOW_TITLES = bool(args.titles)

    if not args.results.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results}")
    datasets = [args.dataset] if args.dataset else [
        p.name for p in sorted(args.results.iterdir())
        if p.is_dir() and (p / "rq3_quality_profile.csv").exists()
    ]
    if not datasets:
        raise FileNotFoundError(f"No dataset result folders found in {args.results}.")
    for ds in datasets:
        out = build_for_dataset(args.results, args.out, ds)
        print(f"Figures written for '{ds}' -> {out}")


if __name__ == "__main__":
    main()
