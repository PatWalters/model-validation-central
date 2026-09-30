#!/usr/bin/env python
"""Step 3: report the comparison for the active data set.

Follows "Even More Thoughts on ML Method Comparisons"
(https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html):
the unit of comparison is the distribution of a metric over the 25 folds, not a
single number, and the question "is this method actually better?" is answered by a
test that corrects for multiple comparisons rather than by bolding the largest
value in a table.

Writes to results/<ds>/figures and results/<ds>/tables:

  boxplot_<metric>.png       fold-level distributions per endpoint and method
  tukey_<metric>.png         Tukey HSD against the best method, one panel per endpoint
  paired_<a>_vs_<b>_<metric>.png   paired plots, folds connected, paired t-test in the header
  delta_vs_reference_<metric>.png  every head against Monroe + TabPFN 3, paired by fold
  summary.csv / summary.md   mean +/- sd per endpoint x method x metric, annotated
                             with the Tukey grouping (best / equivalent / worse)
  head_to_head.csv           mean difference, fold win rate and paired p value

    python 03_report.py
    ADME_DATASET=biogen python 03_report.py
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns

import config as cfg
from model_comparison import best_method, make_tukey_plot, paired_test, tukey_groups

# The questions worth asking. The first is the control: one checkpoint through two
# implementations, which should tie. The rest put every NVIDIA head against the
# reference, and the last asks what Kumo's extra parameters buy.
COMPARISONS = [
    (cfg.MONROE_TABICL_METHOD, "sdm_tabiclv2"),
    *[(cfg.MONROE_METHOD, method) for method in cfg.SDM_METHODS],
    ("sdm_kumo_small", "sdm_kumo_large"),
]

PALETTE = {
    cfg.LGBM_METHOD: "#4C72B0",
    cfg.MONROE_METHOD: "#937860",
    # The two TabICLv2 arms share a checkpoint, so they share a hue: the upstream
    # package in Monroe's lightened brown, as on the source page, and NVIDIA's
    # port a shade darker.
    cfg.MONROE_TABICL_METHOD: "#C0A08B",
    "sdm_tabiclv2": "#A0765A",
    # Kumo's three sizes are one model family, drawn as one green ramp.
    "sdm_kumo_small": "#A6D38A",
    "sdm_kumo_medium": "#6DB35A",
    "sdm_kumo_large": "#2F7A2A",
    "sdm_tabfm": "#8172B3",
}
REFERENCE = cfg.MONROE_METHOD

# Three panels to a row, as many rows as the data set needs. ExpansionRx fills a
# 3x3; the Biogen set is six endpoints, so it gets 3x2 rather than a blank row.
N_COLS = 3
PANEL_SIZE = (17 / 3, 14 / 3)  # what one ExpansionRx panel came out at

# The Tukey panels are the one figure that lists every method down the y axis, so
# their height has to grow as methods are added or the labels close up on each
# other. The boxplots put methods on the x axis and the paired plots do not show
# them at all, so both keep PANEL_SIZE as it is.
#
# These two reproduce PANEL_SIZE's height exactly at the seven methods it was
# written for, and grow by one row's worth for every method after that.
TUKEY_PAD_IN = PANEL_SIZE[1] - 0.5 * 7  # title, x axis and the margins
TUKEY_ROW_H_IN = 0.5                    # one method row


def tukey_panel_size() -> tuple[float, float]:
    return PANEL_SIZE[0], TUKEY_PAD_IN + TUKEY_ROW_H_IN * len(cfg.METHODS)


def grid_shape(n: int) -> tuple[int, int]:
    return (n + N_COLS - 1) // N_COLS, N_COLS


def load_metrics(path=None) -> pd.DataFrame:
    path = path or cfg.FOLD_METRICS_CSV
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 02_collect_metrics.py first")
    df = pd.read_csv(path)

    # fold_metrics.csv holds every method that has predictions on this data set,
    # because collection is shared between the reports. A report covers one
    # comparison, so anything outside it is dropped here rather than silently
    # failing the completeness check in usable_metrics below.
    known = set(cfg.ALL_METHODS)
    unknown = set(df["method"]) - known
    if unknown:
        raise SystemExit(f"unexpected methods in {path.name}: {sorted(unknown)}")
    df = df[df["method"].isin(cfg.METHODS)].copy()

    df["fold_id"] = "r" + df["repeat"].astype(str) + "f" + df["fold"].astype(str)
    return df


def usable_metrics(metrics: pd.DataFrame, min_folds: int) -> tuple[pd.DataFrame, list[str]]:
    """Keep the endpoints where every method has at least `min_folds` shared folds.

    Restricting to the folds every method has keeps the comparison paired
    while the sweep is still running; with a complete sweep this is a no-op.
    """
    keep, endpoints = [], []
    for endpoint in cfg.TARGET_COLS:
        sub = metrics[metrics["endpoint"] == endpoint]
        by_method = {m: set(g["fold_id"]) for m, g in sub.groupby("method")}
        if set(by_method) != set(cfg.METHODS):
            continue
        shared = set.intersection(*by_method.values())
        if len(shared) < min_folds:
            continue
        endpoints.append(endpoint)
        keep.append(sub[sub["fold_id"].isin(shared)])

    skipped = [e for e in cfg.TARGET_COLS if e not in endpoints]
    if skipped:
        print(f"skipping endpoints without {min_folds} shared folds per method: {', '.join(skipped)}")
    if not endpoints:
        raise SystemExit(f"no endpoint has all {len(cfg.METHODS)} methods yet")
    return pd.concat(keep, ignore_index=True), endpoints


def panel_grid(endpoints: list[str], figsize=None):
    rows, cols = grid_shape(len(endpoints))
    figsize = figsize or (PANEL_SIZE[0] * cols, PANEL_SIZE[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    flat = axes.flatten()
    for ax in flat[len(endpoints):]:
        ax.set_visible(False)
    return fig, flat


# How much of the figure's height to keep clear for the suptitle. `tight_layout`
# does not know about one, so on a tall figure it lays the panels out over it.
SUPTITLE_IN = 0.5


def save(fig, name: str, reserve_title: bool = False) -> None:
    top = 1 - SUPTITLE_IN / fig.get_size_inches()[1] if reserve_title else 1
    fig.tight_layout(rect=(0, 0, 1, top))
    for suffix in ("png", "svg"):
        fig.savefig(cfg.FIGURE_DIR / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.png")


def boxplots(metrics: pd.DataFrame, endpoints: list[str]) -> None:
    """Fold-level distributions -- variability first, central tendency second."""
    for metric in cfg.METRICS:
        fig, axes = panel_grid(endpoints)
        for ax, endpoint in zip(axes, endpoints):
            sub = metrics[metrics["endpoint"] == endpoint]
            sns.boxplot(
                data=sub, x="method", y=metric, order=cfg.METHODS, hue="method",
                hue_order=cfg.METHODS, palette=PALETTE, legend=False, ax=ax, width=0.6,
                fliersize=0,
            )
            sns.stripplot(
                data=sub, x="method", y=metric, order=cfg.METHODS, ax=ax,
                color="0.25", size=3, alpha=0.6, jitter=0.15,
            )
            ax.set_title(endpoint)
            ax.set_xlabel("")
            ax.set_ylabel(cfg.METRIC_LABELS[metric])
            ax.set_xticks(range(len(cfg.METHODS)))
            ax.set_xticklabels(
                [cfg.METHOD_LABELS[m] for m in cfg.METHODS], rotation=25, ha="right", fontsize=8
            )
        fig.suptitle(
            f"{cfg.METRIC_LABELS[metric]} over 25 folds (5x5 CV, fixed test set)", fontsize=14
        )
        save(fig, f"boxplot_{metric}")


def tukey_figures(metrics: pd.DataFrame, endpoints: list[str]) -> None:
    """Tukey HSD against the best method: blue best, grey equivalent, red worse.

    The panels share both axes: one x range across every endpoint, so a bar's
    length means the same thing everywhere, and method names only down the left
    column, since every panel lists the same methods in the same order.
    """
    n_cols = N_COLS
    rows, cols = grid_shape(len(endpoints))
    panel_w, panel_h = tukey_panel_size()
    for metric in cfg.METRICS:
        higher = cfg.METRIC_HIGHER_IS_BETTER[metric]
        fig, axes = panel_grid(endpoints)
        xlims = []
        for ax, endpoint in zip(axes, endpoints):
            sub = metrics[metrics["endpoint"] == endpoint]
            make_tukey_plot(
                sub, metric, higher_is_better=higher, ax=ax, title=endpoint,
                xlabel=cfg.METRIC_LABELS[metric],
            )
            ax.set_yticklabels([cfg.METHOD_LABELS.get(t.get_text(), t.get_text())
                                for t in ax.get_yticklabels()])
            xlims.append(ax.get_xlim())

        # Shared x: the union of what each panel needed, applied to all of them.
        lo = min(x[0] for x in xlims)
        hi = max(x[1] for x in xlims)
        bottom_row = len(endpoints) - n_cols
        for i, ax in enumerate(axes[: len(endpoints)]):
            ax.set_xlim(lo, hi)
            if i % n_cols:                      # not the leftmost column
                ax.set_yticklabels([])
            if i < bottom_row:                  # not the bottom row
                ax.set_xticklabels([])
                ax.set_xlabel("")

        fig.suptitle(
            f"Tukey HSD on {cfg.METRIC_LABELS[metric]} -- blue: best, grey: indistinguishable, "
            "red: significantly worse",
            fontsize=14,
        )
        # Sized here rather than at creation because statsmodels'
        # `plot_simultaneous` calls `set_size_inches((10, 6))` on whatever figure
        # its axes belong to, even when it is handed an `ax` and is drawing one
        # panel of nine. Whatever size the grid is created at, the last panel
        # drawn resets it, so the size has to be set after the panels are.
        fig.set_size_inches(panel_w * cols, panel_h * rows)
        save(fig, f"tukey_{metric}", reserve_title=True)


def paired_figures(metrics: pd.DataFrame, endpoints: list[str]) -> None:
    """Paired plots: the same 25 folds seen by both methods, connected."""
    for left, right in COMPARISONS:
        if not {left, right} <= set(cfg.METHODS):
            continue
        for metric in cfg.METRICS:
            fig, axes = panel_grid(endpoints)
            for ax, endpoint in zip(axes, endpoints):
                sub = metrics[
                    (metrics["endpoint"] == endpoint) & (metrics["method"].isin([left, right]))
                ]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pg.plot_paired(
                        data=sub, dv=metric, within="method", subject="fold_id",
                        order=[left, right], boxplot=True, ax=ax, colors=["green", "grey", "red"],
                    )
                diff, pval = paired_test(sub, metric, left, right)
                better = cfg.METRIC_HIGHER_IS_BETTER[metric] == (diff > 0)
                arrow = "↑" if better else "↓"
                ax.set_title(f"{endpoint}\n{arrow} {diff:+.3f}, p = {pval:.1e}", fontsize=10)
                ax.set_xlabel("")
                ax.set_ylabel(cfg.METRIC_LABELS[metric])
                ax.set_xticks(range(2))
                ax.set_xticklabels([cfg.METHOD_LABELS[left], cfg.METHOD_LABELS[right]],
                                   rotation=20, ha="right")
            fig.suptitle(
                f"{cfg.METHOD_LABELS[right]} vs {cfg.METHOD_LABELS[left]}, "
                f"{cfg.METRIC_LABELS[metric]} paired by fold "
                "(green: right better, red: left better)",
                fontsize=14,
            )
            save(fig, f"paired_{left}_vs_{right}_{metric}")


def delta_vs_reference(metrics: pd.DataFrame, endpoints: list[str]) -> None:
    """Every head against the reference, one row per endpoint.

    Each marker is the mean paired difference over the 25 folds, arm minus
    reference, with a 95% interval from the paired t distribution. The sign is
    flipped for MAE so that right is better on every panel. An interval that
    crosses zero is a fold-paired tie; this is a per-pair test, uncorrected, and
    is read alongside the Tukey figure, not instead of it.
    """
    from scipy import stats

    heads = [m for m in cfg.METHODS if m not in (REFERENCE, cfg.LGBM_METHOD)]
    for metric in cfg.METRICS:
        sign = 1 if cfg.METRIC_HIGHER_IS_BETTER[metric] else -1
        fig, ax = plt.subplots(figsize=(8, 0.55 * len(endpoints) + 1.6))
        offsets = np.linspace(-0.3, 0.3, len(heads))
        for off, head in zip(offsets, heads):
            for i, endpoint in enumerate(endpoints):
                sub = metrics[metrics["endpoint"] == endpoint]
                wide = sub.pivot(index="fold_id", columns="method", values=metric)
                diff = sign * (wide[head] - wide[REFERENCE])
                half = stats.t.ppf(0.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff))
                ax.errorbar(diff.mean(), i + off, xerr=half, fmt="o", ms=4.5,
                            color=PALETTE[head], ecolor=PALETTE[head], elinewidth=1.2,
                            label=cfg.METHOD_LABELS[head] if i == 0 else None)
        ax.axvline(0, color="0.3", lw=1)
        ax.set_yticks(range(len(endpoints)))
        ax.set_yticklabels(endpoints)
        ax.invert_yaxis()
        better = "higher" if cfg.METRIC_HIGHER_IS_BETTER[metric] else "lower"
        ax.set_xlabel(f"{cfg.METRIC_LABELS[metric]} difference from "
                      f"{cfg.METHOD_LABELS[REFERENCE]}  (right = {better}, better)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.25 / len(endpoints)),
                  ncol=3, fontsize=8, frameon=False)
        ax.set_title(f"{cfg.ACTIVE.label}: each head against the reference, "
                     "mean over 25 paired folds, 95% CI")
        save(fig, f"delta_vs_reference_{metric}")


def summary_table(metrics: pd.DataFrame, endpoints: list[str]) -> pd.DataFrame:
    """Mean +/- sd per endpoint x method x metric, with the Tukey grouping."""
    rows = []
    for endpoint in endpoints:
        sub = metrics[metrics["endpoint"] == endpoint]
        for metric in cfg.METRICS:
            groups = tukey_groups(sub, metric, cfg.METRIC_HIGHER_IS_BETTER[metric])
            stats = sub.groupby("method")[metric].agg(["mean", "std"])
            for method in cfg.METHODS:
                rows.append(
                    {
                        "endpoint": endpoint,
                        "metric": metric,
                        "method": method,
                        "mean": stats.loc[method, "mean"],
                        "sd": stats.loc[method, "std"],
                        "tukey_group": groups[method],
                        "n_folds": int((sub["method"] == method).sum()),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.TABLE_DIR / "summary.csv", index=False)

    # A readable version: value +/- sd, tagged with the Tukey grouping rather than
    # bolded -- the point of the blog post is that the bold table hides the test.
    tag = {"best": " (best)", "equivalent": " (=)", "worse": ""}
    summary["cell"] = [
        f"{m:.3f} +/- {s:.3f}{tag[g]}"
        for m, s, g in zip(summary["mean"], summary["sd"], summary["tukey_group"])
    ]
    lines = []
    for metric in cfg.METRICS:
        wide = (
            summary[summary["metric"] == metric]
            .pivot(index="endpoint", columns="method", values="cell")
            .reindex(index=endpoints, columns=cfg.METHODS)
            .rename(columns=cfg.METHOD_LABELS)
        )
        lines.append(f"\n## {metric}\n")
        lines.append(wide.to_markdown())
    (cfg.TABLE_DIR / "summary.md").write_text(
        "# 5x5 cross-validated comparison\n"
        "\nMean +/- sd over 25 folds. `(best)` marks the best mean; `(=)` a method Tukey HSD "
        "cannot distinguish from it at alpha = 0.05; unmarked methods are significantly worse.\n"
        + "\n".join(lines)
        + "\n"
    )
    print(f"wrote tables/summary.csv and tables/summary.md")
    return summary


def head_to_head(metrics: pd.DataFrame, endpoints: list[str]) -> pd.DataFrame:
    """Mean difference, fold win rate and paired p value for each comparison."""
    rows = []
    for left, right in COMPARISONS:
        if not {left, right} <= set(cfg.METHODS):
            continue
        for endpoint in endpoints:
            for metric in cfg.METRICS:
                sub = metrics[
                    (metrics["endpoint"] == endpoint) & (metrics["method"].isin([left, right]))
                ]
                diff, pval = paired_test(sub, metric, left, right)
                wide = sub.pivot(index="fold_id", columns="method", values=metric)
                higher = cfg.METRIC_HIGHER_IS_BETTER[metric]
                wins = (wide[right] > wide[left]) if higher else (wide[right] < wide[left])
                rows.append(
                    {
                        "endpoint": endpoint,
                        "metric": metric,
                        "left": left,
                        "right": right,
                        "mean_diff": diff,
                        "right_wins": int(wins.sum()),
                        "n_folds": len(wide),
                        "p_value": pval,
                    }
                )
    table = pd.DataFrame(rows)
    table.to_csv(cfg.TABLE_DIR / "head_to_head.csv", index=False)
    print("wrote tables/head_to_head.csv")
    return table


def print_overview(metrics: pd.DataFrame, summary: pd.DataFrame, endpoints: list[str]) -> None:
    print("\nbest method per endpoint (mean over 25 folds):")
    header = f"{'endpoint':<17}" + "".join(f"{cfg.METRIC_LABELS[m][:12]:<28}" for m in cfg.METRICS)
    print(header)
    for endpoint in endpoints:
        sub = metrics[metrics["endpoint"] == endpoint]
        cells = []
        for metric in cfg.METRICS:
            best = best_method(sub, metric, cfg.METRIC_HIGHER_IS_BETTER[metric])
            row = summary[
                (summary["endpoint"] == endpoint)
                & (summary["metric"] == metric)
                & (summary["method"] == best)
            ].iloc[0]
            ties = summary[
                (summary["endpoint"] == endpoint)
                & (summary["metric"] == metric)
                & (summary["tukey_group"] == "equivalent")
            ]["method"].tolist()
            mark = f" (= {','.join(ties)})" if ties else ""
            cells.append(f"{cfg.METHOD_LABELS[best]} {row['mean']:.3f}{mark}")
        print(f"{endpoint:<17}" + "".join(f"{c:<28}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=None,
                        help="fold metrics CSV (default: results/<ds>/fold_metrics.csv)")
    parser.add_argument(
        "--min-folds",
        type=int,
        default=cfg.N_REPEATS * cfg.N_SPLITS,
        help="report on endpoints with at least this many folds per method; lower it to "
             "look at a sweep that is still running",
    )
    args = parser.parse_args()

    cfg.ensure_dirs()
    sns.set_style("whitegrid")

    metrics = load_metrics(args.metrics)
    metrics, endpoints = usable_metrics(metrics, args.min_folds)
    n_folds = metrics.groupby(["endpoint", "method"]).size().min()
    print(f"reporting on {len(endpoints)} endpoints x {len(cfg.METHODS)} methods x "
          f"{n_folds} folds")

    boxplots(metrics, endpoints)
    tukey_figures(metrics, endpoints)
    paired_figures(metrics, endpoints)
    delta_vs_reference(metrics, endpoints)
    summary = summary_table(metrics, endpoints)
    head_to_head(metrics, endpoints)
    print_overview(metrics, summary, endpoints)


if __name__ == "__main__":
    main()
