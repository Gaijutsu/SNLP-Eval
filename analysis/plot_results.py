"""
Generate evaluation plots for the SWE-bench context-gathering experiment.

Usage:
    python analysis/plot_results.py \
        --csv results/full_all/2026-04-04_124222/results.csv \
        --out analysis/figures
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Gatherer metadata
# ---------------------------------------------------------------------------
DISPLAY = {
    "rag_bm25":       "BM25",
    "rag_dense":      "Dense",
    "rag_hybrid":     "Hybrid",
    "react_agent":    "ReAct",
    "agentic_bm25":   "ReAct-BM25",
    "agentless":      "Agentless",
    "agentless_bm25": "Agentless-BM25",
}

# Marker / colour scheme matching the reference style
STYLE = {
    #                  colour       marker  ms
    "rag_bm25":       ("#1f77b4",   "s",    9),
    "rag_dense":      ("#00bcd4",   "s",    9),
    "rag_hybrid":     ("#2ca02c",   "s",    9),
    "react_agent":    ("#d62728",   "^",   10),
    "agentic_bm25":   ("#8b0000",   "D",   10),
    "agentless":      ("#ff7f0e",   "^",   10),
    "agentless_bm25": ("#e377c2",   "D",   10),
}

ORDER = list(DISPLAY.keys())

RAG_GATHERERS   = ["rag_bm25", "rag_dense", "rag_hybrid"]
AGENT_GATHERERS = ["react_agent", "agentic_bm25", "agentless", "agentless_bm25"]
LLM_GATHERERS   = ["react_agent", "agentic_bm25", "agentless", "agentless_bm25"]


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["gatherer_label"] = df["gatherer"].map(DISPLAY)
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-gatherer mean for all numeric columns."""
    num_cols = df.select_dtypes("number").columns.tolist()
    return df.groupby("gatherer")[num_cols].mean().reindex(ORDER)


# ---------------------------------------------------------------------------
# 1. Pareto front: MRR vs mean latency
# ---------------------------------------------------------------------------
def plot_pareto(s: pd.DataFrame, out: Path, metric: str = "mrr", xlabel_extra: str = ""):
    fig, ax = plt.subplots(figsize=(5.5, 4))

    xs, ys, labels = [], [], []
    for g in ORDER:
        if g not in s.index:
            continue
        x = s.loc[g, "latency_s"]
        y = s.loc[g, metric]
        if pd.isna(x) or pd.isna(y):
            continue
        c, mk, ms = STYLE[g]
        ax.scatter(x, y, color=c, marker=mk, s=ms**2, zorder=5,
                   label=DISPLAY[g], linewidths=0.8, edgecolors="k")
        xs.append(x); ys.append(y); labels.append(g)

    # Pareto front (maximize metric, minimize latency)
    pts = sorted(zip(xs, ys, labels), key=lambda t: t[0])
    pareto = []
    best_y = -np.inf
    for x, y, g in pts:
        if y > best_y:
            pareto.append((x, y))
            best_y = y
    if len(pareto) > 1:
        px, py = zip(*pareto)
        ax.step(px, py, where="post", color="gray", lw=1.2, ls="--",
                zorder=2, label="Pareto front")

    metric_label = {
        "mrr": "MRR",
        "ndcg@1": "NDCG@1",
        "precision@1": "Precision@1",
    }.get(metric, metric.upper())

    ax.set_xscale("log")
    ax.set_xlabel(f"Mean Latency (s, log scale){xlabel_extra}")
    ax.set_ylabel(metric_label)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fname = out / f"pareto_{metric.replace('@','at')}.pdf"
    fig.savefig(fname, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# 2. NDCG@K line chart
# ---------------------------------------------------------------------------
def plot_metric_vs_k(s: pd.DataFrame, out: Path,
                     metric_prefix: str = "ndcg",
                     ylabel: str = "NDCG@K",
                     fname_stem: str = "ndcg_k"):
    ks = [1, 3, 5, 10]
    cols = [f"{metric_prefix}@{k}" for k in ks]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for g in ORDER:
        if g not in s.index:
            continue
        vals = [s.loc[g, c] for c in cols if c in s.columns]
        if not vals:
            continue
        c, mk, ms = STYLE[g]
        ax.plot(ks[:len(vals)], vals, marker=mk, color=c,
                markersize=ms - 1, label=DISPLAY[g], lw=1.5)

    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_xticks(ks)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fname = out / f"{fname_stem}.pdf"
    fig.savefig(fname, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# 3. Grouped bar chart for a single K across gatherers
# ---------------------------------------------------------------------------
def plot_bar_group(s: pd.DataFrame, out: Path,
                   metrics: list[str],
                   ylabel: str,
                   fname_stem: str):
    """Side-by-side bars for multiple metrics across gatherers."""
    gatherers = [g for g in ORDER if g in s.index]
    n_g = len(gatherers)
    n_m = len(metrics)
    x = np.arange(n_g)
    width = 0.8 / n_m
    offsets = np.linspace(-(n_m - 1) / 2, (n_m - 1) / 2, n_m) * width

    fig, ax = plt.subplots(figsize=(8, 4))
    cmap = plt.cm.get_cmap("tab10", n_m)
    for i, met in enumerate(metrics):
        vals = [s.loc[g, met] if met in s.columns else 0.0 for g in gatherers]
        ax.bar(x + offsets[i], vals, width=width * 0.9,
               label=met, color=cmap(i), alpha=0.85, edgecolor="k", lw=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[g] for g in gatherers], rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fname = out / f"{fname_stem}.pdf"
    fig.savefig(fname, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# 4. Token usage vs NDCG@5 (LLM gatherers only)
# ---------------------------------------------------------------------------
def plot_token_quality(s: pd.DataFrame, df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for g in LLM_GATHERERS:
        if g not in s.index:
            continue
        tok_mean = s.loc[g, "token_usage"]
        ndcg_mean = s.loc[g, "ndcg@5"] if "ndcg@5" in s.columns else np.nan
        if pd.isna(tok_mean) or pd.isna(ndcg_mean):
            continue
        c, mk, ms = STYLE[g]
        ax.scatter(tok_mean / 1000, ndcg_mean, color=c, marker=mk,
                   s=ms**2, zorder=5, label=DISPLAY[g],
                   edgecolors="k", linewidths=0.8)

    ax.set_xlabel("Mean Token Usage (k tokens)")
    ax.set_ylabel("NDCG@5")
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fname = out / "token_vs_ndcg5.pdf"
    fig.savefig(fname, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# 5. Edit similarity distribution (Agentless variants)
# ---------------------------------------------------------------------------
def plot_edit_similarity(df: pd.DataFrame, out: Path):
    patch_gatherers = ["agentless", "agentless_bm25"]
    data = []
    labels = []
    for g in patch_gatherers:
        sub = df[df["gatherer"] == g]["edit_similarity"].dropna()
        if not sub.empty:
            data.append(sub.values)
            labels.append(DISPLAY[g])

    if not data:
        return

    fig, ax = plt.subplots(figsize=(4.5, 4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.4,
                    medianprops=dict(color="black", lw=2))
    colors = ["#ff7f0e", "#e377c2"]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax.set_ylabel("Edit Similarity (vs. reference patch)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fname = out / "edit_similarity.pdf"
    fig.savefig(fname, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# 6. Success@K heatmap
# ---------------------------------------------------------------------------
def plot_success_k(s: pd.DataFrame, out: Path):
    ks = [1, 3, 5, 10]
    cols = [f"success@{k}" for k in ks]
    gatherers = [g for g in ORDER if g in s.index]

    mat = np.array([[s.loc[g, c] if c in s.columns else np.nan
                     for c in cols] for g in gatherers])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"@{k}" for k in ks])
    ax.set_yticks(range(len(gatherers)))
    ax.set_yticklabels([DISPLAY[g] for g in gatherers])
    ax.set_xlabel("K")
    ax.set_title("Success@K (at least one gold file in top-K)")

    for i in range(len(gatherers)):
        for j in range(len(ks)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if v < 0.6 else "white")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    fname = out / "success_k_heatmap.pdf"
    fig.savefig(fname, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/full_all/2026-04-04_124222/results.csv")
    ap.add_argument("--out", default="analysis/figures")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading CSV …")
    df = load(args.csv)
    s = summary(df)

    print(f"\n{'Gatherer':<20}  {'MRR':>6}  {'NDCG@5':>7}  {'Latency(s)':>11}  {'Tokens(k)':>10}")
    print("-" * 65)
    for g in ORDER:
        if g not in s.index:
            continue
        print(f"{DISPLAY[g]:<20}  "
              f"{s.loc[g,'mrr']:>6.4f}  "
              f"{s.loc[g,'ndcg@5']:>7.4f}  "
              f"{s.loc[g,'latency_s']:>11.2f}  "
              f"{s.loc[g,'token_usage']/1000:>10.1f}")

    print("\nGenerating figures …")
    plot_pareto(s, out, metric="mrr")
    plot_pareto(s, out, metric="ndcg@1")
    plot_pareto(s, out, metric="ndcg@3")
    plot_metric_vs_k(s, out, "ndcg",     "NDCG@K",      "ndcg_k")
    plot_metric_vs_k(s, out, "recall",   "Recall@K",    "recall_k")
    plot_metric_vs_k(s, out, "precision","Precision@K", "precision_k")
    plot_metric_vs_k(s, out, "f1",       "F1@K",        "f1_k")
    plot_bar_group(s, out,
                   ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"],
                   "NDCG", "ndcg_bars")
    plot_bar_group(s, out,
                   ["recall@1", "recall@5", "recall@10"],
                   "Recall", "recall_bars")
    plot_token_quality(s, df, out)
    plot_edit_similarity(df, out)
    plot_success_k(s, out)

    print(f"\nAll figures written to {out}/")


if __name__ == "__main__":
    main()
