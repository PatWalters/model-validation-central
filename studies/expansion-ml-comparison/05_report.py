#!/usr/bin/env python
"""Step 5: report the comparison.

Follows "Even More Thoughts on ML Method Comparisons"
(https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html):
the unit of comparison is the distribution of a metric over the 25 folds, not a
single number, and the question "is this method actually better?" is answered by a
test that corrects for multiple comparisons rather than by bolding the largest
value in a table.

Writes to results/figures and results/tables:

  boxplot_<metric>.png       fold-level distributions per endpoint and method
  tukey_<metric>.png         Tukey HSD against the best method, one panel per endpoint
  paired_<a>_vs_<b>_<metric>.png   paired plots, folds connected, paired t-test in the header
  summary.csv / summary.md   mean +/- sd per endpoint x method x metric, annotated
                             with the Tukey grouping (best / equivalent / worse)
  head_to_head.csv           mean difference, fold win rate and paired p value

Endpoints that do not yet have every method are skipped with a warning, so
this can be run against a partial sweep.

    python 05_report.py
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg
import seaborn as sns

import config as cfg
from model_comparison import best_method, make_tukey_plot, paired_test, tukey_groups

# The questions worth asking, which depend on which methods are on the page.
PAIRED_QUESTIONS = {
    "foundation": [
        ("chemprop", "chemeleon"),  # does the foundation model earn its keep?
        (cfg.LGBM_METHOD, "chemeleon"),  # does the GNN beat the fingerprint baseline?
        ("chemprop_st", "chemprop"),  # what does multitask training buy, on its own?
        (cfg.LGBM_METHOD, "chemprop_st"),  # GNN vs fingerprints, both single-task
        ("chemprop_st", cfg.MEGACL_METHOD),  # does MEGA-CL pre-training beat a plain single-task GNN?
        (cfg.MEGACL_METHOD, "chemeleon"),  # foundation model against foundation model
        ("chemeleon", cfg.MONROE_METHOD),  # the two strongest foundation models, head to head
        (cfg.LGBM_METHOD, cfg.MONROE_METHOD),  # frozen embeddings in context vs fingerprints
        (cfg.MONROE_METHOD, cfg.MOLJEPA_METHOD),  # two frozen encoders, two in-context heads
        ("chemeleon", cfg.MOLJEPA_METHOD),  # fine-tuned representation vs multimodal frozen one
    ],
    # The Trimole-Hybrid page asks one question in several ways: is choosing a
    # model per endpoint worth more than choosing a good architecture once?
    "trimole": [
        ("chemeleon", cfg.TRIMOLE_METHOD),  # selection against the best single backbone
        (cfg.LGBM_METHOD, cfg.TRIMOLE_METHOD),  # and against the cheapest baseline
        ("chemprop", cfg.TRIMOLE_METHOD),  # against multitask message passing
        ("chemprop", "chemeleon"),  # does the foundation model earn its keep?
        ("chemprop_st", "chemprop"),  # what does multitask training buy, on its own?
        (cfg.LGBM_METHOD, "chemprop_st"),  # GNN vs fingerprints, both single-task
    ],
}

COMPARISONS = [
    pair
    for pair in PAIRED_QUESTIONS[cfg.COMPARISON]
    if set(pair) <= set(cfg.METHODS)
]

PALETTE = {
    cfg.LGBM_METHOD: "#4C72B0",
    "chemprop_st": "#C44E52",
    "chemprop": "#DD8452",
    "chemeleon": "#55A868",
    cfg.MEGACL_METHOD: "#8172B3",
    cfg.MONROE_METHOD: "#937860",
    cfg.MOLJEPA_METHOD: "#DA8BC3",
    cfg.TRIMOLE_METHOD: "#CCB974",
}
# Three panels to a row, as many rows as the data set needs. ExpansionRx fills a
# 3x3; the Biogen set is six endpoints, so it gets 3x2 rather than a blank row.
N_COLS = 3
PANEL_SIZE = (17 / 3, 14 / 3)  # what one ExpansionRx panel came out at


def grid_shape(n: int) -> tuple[int, int]:
    return (n + N_COLS - 1) // N_COLS, N_COLS


def load_metrics(path=None) -> pd.DataFrame:
    path = path or cfg.FOLD_METRICS_CSV
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 04_collect_metrics.py first")
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
        raise SystemExit(
            f"no endpoint has all {len(cfg.METHODS)} methods of the "
            f"{cfg.COMPARISON} comparison yet"
        )
    return pd.concat(keep, ignore_index=True), endpoints


def panel_grid(endpoints: list[str], figsize=None):
    rows, cols = grid_shape(len(endpoints))
    figsize = figsize or (PANEL_SIZE[0] * cols, PANEL_SIZE[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    flat = axes.flatten()
    for ax in flat[len(endpoints):]:
        ax.set_visible(False)
    return fig, flat


def save(fig, name: str) -> None:
    fig.tight_layout()
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
        save(fig, f"tukey_{metric}")


def paired_figures(metrics: pd.DataFrame, endpoints: list[str]) -> None:
    """Paired plots: the same 25 folds seen by both methods, connected."""
    for left, right in COMPARISONS:
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
    parser.add_argument("--metrics", type=Path, default=None, help="fold metrics CSV (default: results/fold_metrics.csv)")
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
    summary = summary_table(metrics, endpoints)
    head_to_head(metrics, endpoints)
    print_overview(metrics, summary, endpoints)


if __name__ == "__main__":
    main()
