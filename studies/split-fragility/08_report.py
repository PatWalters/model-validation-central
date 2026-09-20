#!/usr/bin/env python
"""Step 8: the statistics and the figures.

Follows "Even More Thoughts on ML Method Comparisons"
(https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html):
the unit of comparison is the distribution of a metric over folds rather than a
single number, and "is this method actually better?" is answered by a test that
corrects for multiple comparisons rather than by bolding the largest value.

What differs from the other studies here is the shape of the design, and
`split_stats` carries the reasoning. Each splitting scheme gives fifty folds per
method -- ten targets by five folds -- and the between-target spread is much
larger than anything separating the methods, so the primary test is paired on
the fold, where the target effect cancels, and the Tukey figure is drawn on
target-centred values so its intervals mean what they look like. The raw pooled
Tukey, which is how the paper aggregates, is drawn too, next to it, because the
difference between the two is itself worth seeing.

Writes to results/figures and results/tables:

  mae_by_split.png            the paper's Figure 2, rebuilt: fold-level MAE per
                              scheme and method, with the naive baseline
  tukey_<metric>.png          block-adjusted Tukey HSD, one panel per scheme
  tukey_raw_<metric>.png      the same on raw pooled values, for contrast
  gap_to_baseline.png         how far each method beats the naive predictor
  paired_<a>_vs_<b>.png       folds connected, paired test in the header
  summary.csv / summary.md    mean +/- sd per scheme x method x metric, annotated
                              best / equivalent / worse by the paired test
  head_to_head.csv            every pair, mean difference, fold win rate, Holm p
  tally.csv                   best / tied / worse counts per method
  gap_to_baseline.csv         the gap to the naive predictor, per scheme

    python 08_report.py
    SPLIT_COMPARISON=published python 08_report.py    # add the paper's own arms
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg
from model_comparison import best_method, make_tukey_plot
from split_stats import (
    SUBJECT_COL,
    block_center,
    collapse_to_targets,
    gap_to_baseline,
    paired_groups,
    paired_table,
)

# The questions worth asking of these four arms, in the order the report asks
# them. Every one is a paired test over a scheme's fifty folds.
PAIRED_QUESTIONS = [
    # The paper's claim, arm by arm: does anything beat the similarity baseline?
    (cfg.KNN_METHOD, cfg.LGBM_METHOD),
    (cfg.KNN_METHOD, cfg.CHEMELEON_METHOD),
    (cfg.KNN_METHOD, cfg.MONROE35_METHOD),
    # And the newer question the paper could not ask: do the two pre-trained
    # representations separate from each other, and from a fingerprint model?
    (cfg.LGBM_METHOD, cfg.CHEMELEON_METHOD),
    (cfg.LGBM_METHOD, cfg.MONROE35_METHOD),
    (cfg.CHEMELEON_METHOD, cfg.MONROE35_METHOD),
    # The one that decides whether any of it is predictive at all.
    (cfg.MEDIAN_METHOD, cfg.MONROE35_METHOD),
    (cfg.MEDIAN_METHOD, cfg.CHEMELEON_METHOD),
    (cfg.MEDIAN_METHOD, cfg.KNN_METHOD),
]

PALETTE = {
    cfg.KNN_METHOD: "#4C72B0",
    cfg.LGBM_METHOD: "#DD8452",
    cfg.CHEMELEON_METHOD: "#55A868",
    cfg.MONROE35_METHOD: "#61483A",
    cfg.MEDIAN_METHOD: "#7A8783",
    "pub_knn": "#A3B5CE",
    "pub_svm": "#8172B3",
    "pub_rf": "#CCB974",
    "pub_xgboost": "#DA8BC3",
    "pub_mlp": "#C44E52",
    "pub_gnn": "#64B5CD",
}

N_COLS = 3
PANEL_SIZE = (17 / 3, 14 / 3)
TUKEY_PAD_IN = PANEL_SIZE[1] - 0.5 * 7
TUKEY_ROW_H_IN = 0.5


def tukey_panel_size(n_methods: int) -> tuple[float, float]:
    return PANEL_SIZE[0], TUKEY_PAD_IN + TUKEY_ROW_H_IN * n_methods


def grid_shape(n: int) -> tuple[int, int]:
    return (n + N_COLS - 1) // N_COLS, N_COLS


def load_metrics() -> pd.DataFrame:
    if not cfg.FOLD_METRICS_CSV.exists():
        raise SystemExit(f"{cfg.FOLD_METRICS_CSV} not found -- run 07_collect_metrics.py first")
    df = pd.read_csv(cfg.FOLD_METRICS_CSV)

    unknown = set(df["method"]) - set(cfg.ALL_METHODS)
    if unknown:
        raise SystemExit(f"unexpected methods in {cfg.FOLD_METRICS_CSV.name}: {sorted(unknown)}")
    df = df[df["method"].isin(cfg.METHODS)].copy()

    # The subject a paired test pairs on, and the block a Tukey figure centres
    # on. One target, one scheme, one fold: every method in this study fitted
    # the identical training molecules for it.
    df[SUBJECT_COL] = "tid" + df["target"].astype(str) + "_f" + df["fold"].astype(str)
    return df


def methods_for(df: pd.DataFrame, metric: str) -> list[str]:
    """The methods that have a value for this metric, in configured order.

    The naive baseline predicts a constant, so it has an MAE and an R^2 but no
    Spearman rho. It drops out of the rank-based comparisons rather than being
    plotted at zero.
    """
    have = set(df.loc[df[metric].notna(), "method"])
    return [m for m in cfg.METHODS if m in have]


def require_complete(df: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    """One scheme's folds, restricted to the methods that cover all of them."""
    sub = df[df["method"].isin(methods)]
    counts = sub.pivot_table(index="method", columns=SUBJECT_COL, values=metric, aggfunc="size")
    full = counts.notna().all(axis=1)
    if not full.all():
        short = ", ".join(full.index[~full])
        raise SystemExit(
            f"{metric}: {short} do not cover every fold. Finish the sweep, or drop them "
            "-- every statistic on this page is paired or blocked on the fold."
        )
    return sub


# --- figures --------------------------------------------------------------
def plot_mae_by_split(metrics: pd.DataFrame, methods: list[str]) -> None:
    """The paper's Figure 2, rebuilt on this study's arms.

    Six panels, one per scheme in the paper's order of stringency, each a
    fold-level distribution per method with the naive baseline drawn across it.
    The baseline is a line rather than a box because it is not a competitor: it
    is the level the paper says every method converges towards, and the point of
    the figure is how much daylight is left above it.
    """
    arms = [m for m in methods if m != cfg.MEDIAN_METHOD]
    rows, cols = grid_shape(len(cfg.SPLITS))
    fig, axes = plt.subplots(rows, cols, figsize=(PANEL_SIZE[0] * cols, PANEL_SIZE[1] * rows),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()

    lo = metrics.loc[metrics["method"].isin(methods), "mae"].min()
    hi = metrics.loc[metrics["method"].isin(methods), "mae"].max()

    for ax, split in zip(axes, cfg.SPLITS):
        g = metrics[(metrics["split"] == split) & (metrics["method"].isin(arms))]
        sns.boxplot(data=g, x="method", y="mae", order=arms, hue="method", hue_order=arms,
                    palette=PALETTE, legend=False, fliersize=2, ax=ax)
        sns.stripplot(data=g, x="method", y="mae", order=arms, color="black",
                      size=1.6, alpha=0.35, jitter=0.22, ax=ax)

        baseline = metrics.loc[
            (metrics["split"] == split) & (metrics["method"] == cfg.MEDIAN_METHOD), "mae"
        ]
        if len(baseline):
            ax.axhline(baseline.mean(), color="#9E3B3B", ls="--", lw=1.2, zorder=0)
            ax.text(0.99, baseline.mean(), " naive ", color="#9E3B3B", fontsize=7,
                    va="bottom", ha="right", transform=ax.get_yaxis_transform())

        ax.set_title(cfg.SPLIT_LABELS[split], fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("MAE (pIC$_{50}$)" if ax is axes[0] or ax is axes[cols] else "")
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([cfg.METHOD_LABELS[m] for m in arms], rotation=30,
                           ha="right", fontsize=8)
        ax.set_ylim(lo - 0.05, hi + 0.05)

    for ax in axes[len(cfg.SPLITS):]:
        ax.set_visible(False)

    fig.suptitle("MAE over 50 folds per splitting scheme, ten ChEMBL targets", fontsize=13)
    fig.tight_layout()
    path = cfg.FIGURE_DIR / "mae_by_split.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_tukey(metrics: pd.DataFrame, metric: str, methods: list[str], blocked: bool) -> None:
    """Tukey HSD against the best method, one panel per scheme.

    Blue is the best mean, grey a method the correction cannot separate from it,
    red a method that clears its interval and is worse. With `blocked` the values
    are target-centred first, which leaves every method's mean exactly where it
    was and takes the between-target variance out of the intervals.
    """
    higher = cfg.METRIC_HIGHER_IS_BETTER[metric]
    rows, cols = grid_shape(len(cfg.SPLITS))
    size = tukey_panel_size(len(methods))
    fig, axes = plt.subplots(rows, cols, figsize=(size[0] * cols, size[1] * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, split in zip(axes, cfg.SPLITS):
        g = require_complete(metrics[metrics["split"] == split], metric, methods)
        g = g[["method", SUBJECT_COL, "target", metric]].dropna(subset=[metric])
        if blocked:
            g = block_center(g, metric, block_col="target")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            make_tukey_plot(
                g, metric, higher_is_better=higher, ax=ax,
                title=cfg.SPLIT_LABELS[split],
                xlabel=("target-centred " if blocked else "") + metric,
            )
        ax.set_yticklabels(
            [cfg.METHOD_LABELS.get(t.get_text(), t.get_text()) for t in ax.get_yticklabels()],
            fontsize=8,
        )

    for ax in axes[len(cfg.SPLITS):]:
        ax.set_visible(False)

    kind = "target-centred" if blocked else "raw pooled"
    fig.suptitle(f"Tukey HSD on {metric.upper()}, {kind}, 50 folds per scheme", fontsize=13)
    fig.tight_layout()
    name = f"tukey_{metric}.png" if blocked else f"tukey_raw_{metric}.png"
    path = cfg.FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def plot_gap_to_baseline(metrics: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """How much MAE each method takes off the naive predictor, fold by fold.

    The paper's argument in one figure. A raw MAE cannot be read across schemes,
    because the schemes select structurally different training sets and the naive
    baseline drifts with them -- the paper says so and the numbers here bear it
    out. The *gap* to that baseline can be read across schemes, and it is paired
    on the fold, so nothing about a target's difficulty is in it.

    Zero is the naive predictor. A method below the line is worse than predicting
    the training median for every molecule.
    """
    if cfg.MEDIAN_METHOD not in methods:
        return pd.DataFrame()
    arms = [m for m in methods if m != cfg.MEDIAN_METHOD]

    frames = []
    for split in cfg.SPLITS:
        g = require_complete(metrics[metrics["split"] == split], "mae", methods)
        gap = gap_to_baseline(
            g[["method", SUBJECT_COL, "mae"]], "mae", cfg.MEDIAN_METHOD,
            higher_is_better=False,
        )
        gap["split"] = split
        frames.append(gap)
    gaps = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=gaps, x="split", y="gap", order=cfg.SPLITS, hue="method",
                hue_order=arms, palette=PALETTE, fliersize=2, ax=ax)
    ax.axhline(0, color="#9E3B3B", ls="--", lw=1.2, zorder=0)
    ax.set_xticks(range(len(cfg.SPLITS)))
    ax.set_xticklabels([cfg.SPLIT_LABELS[s] for s in cfg.SPLITS], rotation=20, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("MAE taken off the naive predictor (pIC$_{50}$)")
    ax.set_title("How much each method beats predicting the training median\n"
                 "Paired on the fold; zero is the naive predictor", fontsize=12)
    ax.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, fc=PALETTE[m], label=cfg.METHOD_LABELS[m]) for m in arms
    ], fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    path = cfg.FIGURE_DIR / "gap_to_baseline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")

    summary = (
        gaps.groupby(["split", "method"])["gap"]
        .agg(mean="mean", sd="std", beats_baseline=lambda s: float((s > 0).mean()))
        .reset_index()
    )
    summary.to_csv(cfg.TABLE_DIR / "gap_to_baseline.csv", index=False)
    return summary


def plot_paired(metrics: pd.DataFrame, metric: str, left: str, right: str,
                methods: list[str]) -> None:
    """One pair, one panel per scheme, folds connected, Holm p in the header.

    The p value is taken from the correction over *every* pair on the page, not
    from a two-method call, so the number in the panel title is the same one the
    head-to-head table carries. A pair tested on its own would be uncorrected,
    and would read as more significant than the report is willing to claim.
    """
    fig, axes = plt.subplots(*grid_shape(len(cfg.SPLITS)),
                             figsize=(PANEL_SIZE[0] * N_COLS,
                                      PANEL_SIZE[1] * grid_shape(len(cfg.SPLITS))[0]),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, split in zip(axes, cfg.SPLITS):
        g = metrics[(metrics["split"] == split) & (metrics["method"].isin([left, right]))]
        wide = g.pivot(index=SUBJECT_COL, columns="method", values=metric).dropna()
        full = require_complete(metrics[metrics["split"] == split], metric, methods)
        table = paired_table(full[["method", SUBJECT_COL, metric]].dropna(), metric, methods)
        hit = table[(table["left"] == left) & (table["right"] == right)]
        if hit.empty:
            hit = table[(table["left"] == right) & (table["right"] == left)].copy()
            hit["mean_diff"] *= -1
            hit["right_wins"] = 1.0 - hit["right_wins"]
        row = hit.iloc[0]

        for _, pair in wide.iterrows():
            ax.plot([0, 1], [pair[left], pair[right]], color="#7A8783", lw=0.6, alpha=0.6)
        ax.plot([0] * len(wide), wide[left], "o", color=PALETTE[left], ms=3.5)
        ax.plot([1] * len(wide), wide[right], "o", color=PALETTE[right], ms=3.5)
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([cfg.METHOD_LABELS[left], cfg.METHOD_LABELS[right]], fontsize=8)
        ax.set_title(
            f"{cfg.SPLIT_LABELS[split]}\n"
            f"$\\Delta$ = {row['mean_diff']:+.3f}, {row['right_wins']:.0%} of folds, "
            f"p$_{{Holm}}$ = {row['p_holm']:.2g}",
            fontsize=9,
        )
        if ax is axes[0] or ax is axes[N_COLS]:
            ax.set_ylabel(metric)

    for ax in axes[len(cfg.SPLITS):]:
        ax.set_visible(False)
    fig.tight_layout()
    path = cfg.FIGURE_DIR / f"paired_{left}_vs_{right}_{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


# --- tables ---------------------------------------------------------------
def build_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- sd per scheme, method and metric, with the paired-test grouping."""
    rows = []
    for metric in cfg.METRICS:
        methods = methods_for(metrics, metric)
        higher = cfg.METRIC_HIGHER_IS_BETTER[metric]
        for split in cfg.SPLITS:
            g = require_complete(metrics[metrics["split"] == split], metric, methods)
            g = g[["method", SUBJECT_COL, "target", metric]].dropna(subset=[metric])
            groups = paired_groups(g, metric, higher_is_better=higher, methods=methods)
            stats = g.groupby("method")[metric].agg(["mean", "std", "size"])
            for method in methods:
                rows.append(
                    {
                        "split": split,
                        "metric": metric,
                        "method": method,
                        "mean": stats.loc[method, "mean"],
                        "sd": stats.loc[method, "std"],
                        "n_folds": int(stats.loc[method, "size"]),
                        "paired_group": groups[method],
                    }
                )
    return pd.DataFrame(rows)


def build_head_to_head(metrics: pd.DataFrame) -> pd.DataFrame:
    """Every pair of methods, every scheme, every metric, Holm-corrected.

    Three tests per pair, and they are not redundant. `t` is the headline, paired
    over the scheme's fifty folds. `wilcoxon` drops the normality assumption on
    the differences. `t_target` drops the independence assumption instead, by
    averaging each method's five folds within a target first and pairing on the
    ten targets -- which matters because the time split's folds share test
    molecules. See `split_stats.collapse_to_targets`.
    """
    frames = []
    for metric in cfg.METRICS:
        methods = methods_for(metrics, metric)
        for split in cfg.SPLITS:
            g = require_complete(metrics[metrics["split"] == split], metric, methods)
            g = g[["method", SUBJECT_COL, "target", metric]].dropna(subset=[metric])
            for test in ("t", "wilcoxon", "t_target"):
                rows = collapse_to_targets(g, metric) if test == "t_target" else \
                    g[["method", SUBJECT_COL, metric]]
                table = paired_table(rows, metric, methods,
                                     test="t" if test == "t_target" else test)
                table.insert(0, "split", split)
                table.insert(1, "metric", metric)
                table.insert(2, "test", test)
                frames.append(table)
    return pd.concat(frames, ignore_index=True)


def report_test_agreement(head: pd.DataFrame) -> None:
    """How often the three tests reach the same verdict on the same pair.

    The fold-level t-test is the headline; the other two exist to be disagreed
    with. Printed every run so a disagreement is noticed rather than discovered.
    """
    wide = head.pivot_table(
        index=["split", "metric", "left", "right"], columns="test", values="p_holm"
    )
    called = (wide < 0.05)
    rows = []
    for other in ("wilcoxon", "t_target"):
        if other not in called:
            continue
        agree = (called["t"] == called[other])
        rows.append({"against": other, "pairs": len(agree),
                     "agree": int(agree.sum()), "differ": int((~agree).sum())})
    print("\n  agreement of the fold-level t-test with the other two:")
    print("   ", pd.DataFrame(rows).to_string(index=False).replace("\n", "\n    "))


def build_tally(summary: pd.DataFrame) -> pd.DataFrame:
    """Best alone / tied for best / worse, counted over scheme x metric.

    The convention the rest of the repository uses: where two methods cannot be
    separated, both are tied, and neither is called the winner. A method is
    `best alone` only when it has the best mean and every other method is
    separated from it.
    """
    rows = []
    for (split, metric), g in summary.groupby(["split", "metric"]):
        tied = set(g.loc[g["paired_group"].isin(["best", "equivalent"]), "method"])
        winner = g.loc[g["paired_group"] == "best", "method"].iloc[0]
        for _, row in g.iterrows():
            if row["method"] == winner and len(tied) == 1:
                verdict = "best_alone"
            elif row["method"] in tied:
                verdict = "tied"
            else:
                verdict = "worse"
            rows.append({"split": split, "metric": metric,
                         "method": row["method"], "verdict": verdict})

    counted = (
        pd.DataFrame(rows)
        .pivot_table(index="method", columns="verdict", aggfunc="size", values="split")
        .reindex(index=[m for m in cfg.METHODS], columns=["best_alone", "tied", "worse"])
        .fillna(0)
        .astype(int)
    )
    return counted.reset_index()


def write_summary_markdown(summary: pd.DataFrame, methods: list[str]) -> None:
    chip = {"best": "**best**", "equivalent": "tied", "worse": "worse"}
    lines = ["# Mean over each scheme's 50 folds", ""]
    for metric in cfg.METRICS:
        lines += [f"## {metric.upper()}", ""]
        present = [m for m in methods if m in set(summary.loc[summary["metric"] == metric, "method"])]
        lines.append("| scheme | " + " | ".join(cfg.METHOD_LABELS[m] for m in present) + " |")
        lines.append("| --- |" + " ---: |" * len(present))
        for split in cfg.SPLITS:
            cells = []
            for method in present:
                row = summary[(summary["metric"] == metric) & (summary["split"] == split)
                              & (summary["method"] == method)]
                if row.empty:
                    cells.append("")
                    continue
                row = row.iloc[0]
                cells.append(f"{row['mean']:.3f} ± {row['sd']:.3f} {chip[row['paired_group']]}")
            lines.append(f"| {cfg.SPLIT_LABELS[split]} | " + " | ".join(cells) + " |")
        lines.append("")
    (cfg.TABLE_DIR / "summary.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-paired", action="store_true",
                        help="skip the per-pair paired figures")
    args = parser.parse_args()

    cfg.ensure_dirs()
    sns.set_style("whitegrid")
    metrics = load_metrics()
    print(f"{cfg.COMPARISON}: {len(cfg.METHODS)} methods, {len(metrics)} fold metrics")

    print("\nfigures:")
    plot_mae_by_split(metrics, methods_for(metrics, "mae"))
    for metric in cfg.METRICS:
        methods = methods_for(metrics, metric)
        plot_tukey(metrics, metric, methods, blocked=True)
        plot_tukey(metrics, metric, methods, blocked=False)
    gaps = plot_gap_to_baseline(metrics, methods_for(metrics, "mae"))

    if not args.no_paired:
        # Restricted to the arms that have folds, so a partial sweep still
        # produces the figures it can rather than stopping on the first pair
        # whose other half is still training.
        have = set(methods_for(metrics, "mae"))
        ordered = [m for m in cfg.METHODS if m in have]
        for left, right in PAIRED_QUESTIONS:
            if {left, right} <= have:
                plot_paired(metrics, "mae", left, right, ordered)
            else:
                print(f"  (skipped {left} vs {right}: not on disk yet)")

    print("\ntables:")
    summary = build_summary(metrics)
    summary.to_csv(cfg.TABLE_DIR / "summary.csv", index=False)
    write_summary_markdown(summary, cfg.METHODS)
    print("  summary.csv, summary.md")

    head = build_head_to_head(metrics)
    head.to_csv(cfg.TABLE_DIR / "head_to_head.csv", index=False)
    print("  head_to_head.csv")
    report_test_agreement(head)

    tally = build_tally(summary)
    tally.to_csv(cfg.TABLE_DIR / "tally.csv", index=False)
    print("  tally.csv")

    print("\nMAE, mean over each scheme's 50 folds:")
    wide = summary[summary["metric"] == "mae"].pivot(index="split", columns="method", values="mean")
    print(wide.reindex(index=cfg.SPLITS, columns=cfg.METHODS).round(3).to_string())

    print("\nbest alone / tied for best / worse, over 18 scheme x metric combinations:")
    print(tally.to_string(index=False))

    if len(gaps):
        print("\nMAE taken off the naive predictor (mean over 50 folds):")
        print(
            gaps.pivot(index="split", columns="method", values="mean")
            .reindex(index=cfg.SPLITS)
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
