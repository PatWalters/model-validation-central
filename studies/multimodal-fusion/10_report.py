#!/usr/bin/env python
"""Step 10: the figures and tables, one per question the paper asks.

Four questions, in the order the paper puts them:

  1. Does multimodal fusion beat the best unimodal model?
  2. Does early or late fusion do better?
  3. Do added modalities contribute, or repeat what is already there?
  4. What does fusion do to uncertainty and calibration?

Plus two the paper cannot ask of itself, because it has one 80/20 split where
this has 25 paired replicates and four methods already scored on them:

  5. Where does the whole grid sit against a fingerprint baseline and three
     D-MPNNs?
  6. How much of late fusion's behaviour is the in-sample meta-features?

Statistics follow the paper where the paper has them: Wilcoxon signed-rank over
endpoints, Holm corrected within each family of comparisons. Within an endpoint,
where the folds give a real pairing, Tukey HSD is used instead, which is the
convention the reference methods were already reported under.

    python 10_report.py
    ADME_DATASET=biogen python 10_report.py
"""

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import analysis
import config as cfg
from model_comparison import make_tukey_plot, tukey_groups

warnings.filterwarnings("ignore", category=FutureWarning)

FUSION_COLORS = {"early": "#4C72B0", "late": "#DD8452", "unimodal": "#8C8C8C"}
LEARNER_COLORS = {"lgbm": "#4C72B0", "rf": "#C44E52", "attfp": "#55A868"}
REFERENCE_COLORS = {
    "lgbm": "#4C72B0", "chemprop_st": "#C44E52",
    "chemprop": "#DD8452", "chemeleon": "#55A868",
}
COMBO_ORDER = list(cfg.COMBOS)


def save(fig, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(cfg.FIGURE_DIR / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png")


def load(path=None) -> pd.DataFrame:
    path = path or cfg.FOLD_METRICS_CSV
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 06_collect_metrics.py first")
    df = pd.read_csv(path)
    df["fold_id"] = "r" + df["repeat"].astype(str) + "f" + df["fold"].astype(str)
    return df


def complete_endpoints(metrics: pd.DataFrame, methods: list[str]) -> list[str]:
    """Endpoints where every one of `methods` has all 25 folds."""
    want = cfg.N_REPEATS * cfg.N_SPLITS
    counts = metrics.pivot_table(index="endpoint", columns="method",
                                 values="r2", aggfunc="count")
    keep = []
    for endpoint in cfg.TARGET_COLS:
        if endpoint not in counts.index:
            continue
        row = counts.loc[endpoint]
        if all(row.get(m, 0) == want for m in methods):
            keep.append(endpoint)
    return keep


# --- 1. the design grid, at a glance -------------------------------------
def grid_distribution(grid: pd.DataFrame, metric: str = "r2") -> None:
    """One panel per modality combination; learner across, fusion by colour.

    The paper's Figure 2, on our endpoints. Each point is one endpoint's mean
    over its 25 folds, so the spread is spread across chemistry rather than
    across folds, which is what the paper's boxes show too.
    """
    means = (
        grid.groupby(["endpoint", "combo", "fusion", "learner"])[metric]
        .mean().reset_index()
    )
    fig, axes = plt.subplots(1, len(COMBO_ORDER), figsize=(4.2 * len(COMBO_ORDER), 4.6),
                             sharey=True)
    for ax, combo in zip(np.atleast_1d(axes), COMBO_ORDER):
        sub = means[means["combo"] == combo]
        sns.boxplot(data=sub, x="learner", y=metric, hue="fusion",
                    order=cfg.LEARNERS, hue_order=cfg.FUSIONS,
                    palette=FUSION_COLORS, ax=ax, width=0.65, fliersize=0)
        sns.stripplot(data=sub, x="learner", y=metric, hue="fusion",
                      order=cfg.LEARNERS, hue_order=cfg.FUSIONS, dodge=True,
                      ax=ax, color="0.2", size=3, alpha=0.6, legend=False)
        ax.set_title(" + ".join(cfg.MODALITY_LABELS[m] for m in cfg.COMBOS[combo]),
                     fontsize=10)
        ax.set_xlabel("")
        ax.set_xticks(range(len(cfg.LEARNERS)))
        ax.set_xticklabels([cfg.LEARNER_LABELS[m] for m in cfg.LEARNERS])
        ax.set_ylabel(cfg.METRIC_LABELS[metric])
        ax.legend_.remove() if ax.get_legend() else None

    handles = [plt.Rectangle((0, 0), 1, 1, color=FUSION_COLORS[f]) for f in cfg.FUSIONS]
    fig.legend(handles, [cfg.FUSION_LABELS[f] for f in cfg.FUSIONS],
               loc="upper right", ncol=2, frameon=False)
    fig.suptitle(
        f"{cfg.ACTIVE.label}: {cfg.METRIC_LABELS[metric]} by modality set, "
        "fusion strategy and final learner",
        fontsize=13,
    )
    save(fig, f"grid_{metric}")


# --- 2. multimodal against the best unimodal ------------------------------
def unimodal_comparison(metrics: pd.DataFrame, metric: str = "r2") -> pd.DataFrame:
    """Per endpoint, the best unimodal model against the best multimodal one.

    Best-against-best, which is how the paper's Table S16-S18 poses it: the
    question is not whether an average fusion model beats an average unimodal
    one, but whether fusing buys anything over the strongest single view.
    """
    wide = analysis.endpoint_means(metrics, metric)
    higher = cfg.METRIC_HIGHER_IS_BETTER[metric]

    best_uni = analysis.best_per_endpoint(wide, analysis.UNIMODAL, higher)
    best_multi = analysis.best_per_endpoint(wide, analysis.MULTIMODAL, higher)
    table = pd.DataFrame({
        "best_unimodal": best_uni,
        "best_multimodal": best_multi,
        "which_unimodal": analysis.best_name_per_endpoint(wide, analysis.UNIMODAL, higher),
        "which_multimodal": analysis.best_name_per_endpoint(wide, analysis.MULTIMODAL, higher),
    })
    table["difference"] = table["best_multimodal"] - table["best_unimodal"]

    from scipy.stats import wilcoxon
    both = table[["best_unimodal", "best_multimodal"]].dropna()
    p = wilcoxon(both["best_multimodal"], both["best_unimodal"]).pvalue if len(both) > 2 else np.nan

    # MAE is the one metric where a negative difference is the improvement, so
    # the win count and the line colours read the direction rather than the sign.
    improved = (table["difference"] > 0) if higher else (table["difference"] < 0)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for endpoint, row in table.iterrows():
        ax.plot([0, 1], [row["best_unimodal"], row["best_multimodal"]],
                marker="o", color="#4C72B0" if improved[endpoint] else "#C44E52",
                alpha=0.8, linewidth=1.4, markersize=5)
        ax.annotate(endpoint, (1.02, row["best_multimodal"]), fontsize=7,
                    va="center", color="0.35")
    ax.set_xlim(-0.25, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["best unimodal", "best multimodal"])
    ax.set_ylabel(cfg.METRIC_LABELS[metric])
    won = int(improved.sum())
    ax.set_title(
        f"{cfg.ACTIVE.label}: fusion wins {won} of {len(table)} endpoints, "
        f"mean {table['difference'].mean():+.3f}, Wilcoxon p = {p:.3f}",
        fontsize=10,
    )
    save(fig, f"unimodal_vs_multimodal_{metric}")

    table.to_csv(cfg.TABLE_DIR / f"unimodal_vs_multimodal_{metric}.csv")
    return table


# --- 3. the modality ladder ----------------------------------------------
def modality_ladder(grid: pd.DataFrame, metrics: pd.DataFrame,
                    metric: str = "r2") -> None:
    """What each added modality buys, with the reference methods as gridlines.

    The unimodal points sit at one modality, so the ladder starts where the
    paper's Figure 6B starts and the reference lines say what any of it is worth.
    """
    means = (
        grid.groupby(["endpoint", "n_modalities", "fusion", "learner"])[metric]
        .mean().reset_index()
    )
    fig, axes = plt.subplots(1, len(cfg.LEARNERS), figsize=(5.0 * len(cfg.LEARNERS), 4.6),
                             sharey=True)
    for ax, learner in zip(np.atleast_1d(axes), cfg.LEARNERS):
        sub = means[means["learner"] == learner]
        sns.pointplot(data=sub, x="n_modalities", y=metric, hue="fusion",
                      hue_order=["unimodal", *cfg.FUSIONS], palette=FUSION_COLORS,
                      errorbar=("ci", 95), ax=ax, dodge=0.25, markers="o", linestyles="-")
        for method in cfg.REFERENCE_METHODS:
            ref = metrics[metrics["method"] == method]
            if not len(ref):
                continue
            value = ref.groupby("endpoint")[metric].mean().mean()
            ax.axhline(value, color=REFERENCE_COLORS[method], linestyle="--",
                       linewidth=1.1, alpha=0.85)
            ax.annotate(cfg.METHOD_LABELS[method], (0.02, value), xycoords=("axes fraction", "data"),
                        fontsize=7, va="bottom", color=REFERENCE_COLORS[method])
        ax.set_title(cfg.LEARNER_LABELS[learner], fontsize=11)
        ax.set_xlabel("modalities")
        ax.set_ylabel(cfg.METRIC_LABELS[metric])
        if ax.get_legend():
            ax.legend_.remove()

    handles = [plt.Line2D([], [], color=FUSION_COLORS[f], marker="o")
               for f in ["unimodal", *cfg.FUSIONS]]
    fig.legend(handles, ["unimodal", *[cfg.FUSION_LABELS[f] for f in cfg.FUSIONS]],
               loc="upper right", ncol=3, frameon=False)
    fig.suptitle(
        f"{cfg.ACTIVE.label}: {cfg.METRIC_LABELS[metric]} against the number of "
        "modalities, averaged over endpoints; dashed lines are the reference methods",
        fontsize=12,
    )
    save(fig, f"modality_ladder_{metric}")


# --- 4. uncertainty and calibration --------------------------------------
def uncertainty_panels(unc: pd.DataFrame) -> None:
    """The paper's Figure 3: four uncertainty metrics by modality set and strategy."""
    grid = unc[unc["method"].isin(cfg.FUSION_METHODS)].copy()
    spec = grid["method"].map(cfg.GRID_SPEC)
    grid["combo"] = spec.map(lambda s: s["combo"])
    grid["fusion"] = spec.map(lambda s: s["fusion"])
    grid["learner"] = spec.map(lambda s: s["learner"])

    rows = cfg.UNCERTAINTY_METRICS
    fig, axes = plt.subplots(len(rows), len(COMBO_ORDER),
                             figsize=(4.0 * len(COMBO_ORDER), 2.9 * len(rows)),
                             sharey="row")
    for r, quantity in enumerate(rows):
        for c, combo in enumerate(COMBO_ORDER):
            ax = axes[r, c]
            sub = grid[grid["combo"] == combo]
            sns.boxplot(data=sub, x="learner", y=quantity, hue="fusion",
                        order=cfg.LEARNERS, hue_order=cfg.FUSIONS,
                        palette=FUSION_COLORS, ax=ax, width=0.65, fliersize=1)
            if ax.get_legend():
                ax.legend_.remove()
            ax.set_xlabel("")
            ax.set_ylabel(cfg.UNCERTAINTY_LABELS[quantity] if c == 0 else "")
            if r == 0:
                ax.set_title(
                    " + ".join(cfg.MODALITY_LABELS[m] for m in cfg.COMBOS[combo]),
                    fontsize=10,
                )
            ax.set_xticks(range(len(cfg.LEARNERS)))
            ax.set_xticklabels([cfg.LEARNER_LABELS[m] for m in cfg.LEARNERS]
                               if r == len(rows) - 1 else [])

    handles = [plt.Rectangle((0, 0), 1, 1, color=FUSION_COLORS[f]) for f in cfg.FUSIONS]
    fig.legend(handles, [cfg.FUSION_LABELS[f] for f in cfg.FUSIONS],
               loc="upper right", ncol=2, frameon=False)
    fig.suptitle(f"{cfg.ACTIVE.label}: epistemic uncertainty and calibration", fontsize=13)
    save(fig, "uncertainty_panels")


def uncertainty_unimodal_comparison(unc: pd.DataFrame) -> pd.DataFrame:
    """The paper's clearest positive result, tested here: fusion and calibration."""
    rows = []
    for quantity in cfg.UNCERTAINTY_METRICS:
        higher = cfg.UNCERTAINTY_HIGHER_IS_BETTER[quantity]
        if higher is None:
            continue
        wide = unc.pivot_table(index="endpoint", columns="method",
                               values=quantity, aggfunc="mean")
        best_uni = analysis.best_per_endpoint(wide, analysis.UNIMODAL, higher)
        best_multi = analysis.best_per_endpoint(wide, analysis.MULTIMODAL, higher)
        both = pd.DataFrame({"uni": best_uni, "multi": best_multi}).dropna()
        if len(both) < 3:
            continue
        from scipy.stats import wilcoxon
        rows.append({
            "quantity": quantity,
            "best_unimodal": both["uni"].mean(),
            "best_multimodal": both["multi"].mean(),
            "mean_diff": (both["multi"] - both["uni"]).mean(),
            "n_endpoints": len(both),
            "p_value": wilcoxon(both["multi"], both["uni"]).pvalue,
        })
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = analysis.holm(table["p_value"].tolist())
        table.to_csv(cfg.TABLE_DIR / "uncertainty_unimodal_vs_multimodal.csv", index=False)
    return table


# --- 5. against the reference methods ------------------------------------
def reference_panel(metrics: pd.DataFrame, metric: str = "r2") -> list[str]:
    """The four reference methods, the best fusion model, and the best unimodal one.

    Which fusion configuration is 'the best' is decided on the mean over
    endpoints, not per endpoint, so the panel shows one fixed configuration
    everywhere rather than a different winner in each. Choosing per endpoint
    would be a selection procedure, and this comparison is about architectures.
    """
    wide = analysis.endpoint_means(metrics, metric)
    higher = cfg.METRIC_HIGHER_IS_BETTER[metric]

    available_multi = [m for m in analysis.MULTIMODAL if m in wide.columns]
    available_uni = [m for m in analysis.UNIMODAL if m in wide.columns]
    if not available_multi or not available_uni:
        return []
    pick = (lambda s: s.idxmax()) if higher else (lambda s: s.idxmin())
    best_multi = pick(wide[available_multi].mean())
    best_uni = pick(wide[available_uni].mean())

    panel = [*cfg.REFERENCE_METHODS, best_uni, best_multi]
    endpoints = complete_endpoints(metrics, panel)
    if not endpoints:
        print("  reference panel: no endpoint has all six methods complete yet")
        return panel

    sub = metrics[metrics["method"].isin(panel) & metrics["endpoint"].isin(endpoints)]
    n_cols = 3
    n_rows = (len(endpoints) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.7 * n_cols, 4.7 * n_rows),
                             squeeze=False)
    flat = axes.flatten()
    for ax in flat[len(endpoints):]:
        ax.set_visible(False)

    xlims = []
    for ax, endpoint in zip(flat, endpoints):
        panel_df = sub[sub["endpoint"] == endpoint]
        make_tukey_plot(panel_df, metric, higher_is_better=higher, ax=ax,
                        title=endpoint, xlabel=cfg.METRIC_LABELS[metric])
        ax.set_yticklabels([cfg.METHOD_LABELS.get(t.get_text(), t.get_text())
                            for t in ax.get_yticklabels()], fontsize=8)
        xlims.append(ax.get_xlim())

    lo, hi = min(x[0] for x in xlims), max(x[1] for x in xlims)
    for i, ax in enumerate(flat[: len(endpoints)]):
        ax.set_xlim(lo, hi)
        if i % n_cols:
            ax.set_yticklabels([])
        if i < len(endpoints) - n_cols:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    fig.suptitle(
        f"{cfg.ACTIVE.label}: Tukey HSD on {cfg.METRIC_LABELS[metric]} -- the best "
        "fusion and unimodal configurations against four reference methods\n"
        "blue: best, grey: indistinguishable, red: significantly worse",
        fontsize=13,
    )
    save(fig, f"reference_tukey_{metric}")

    rows = []
    for endpoint in endpoints:
        panel_df = sub[sub["endpoint"] == endpoint]
        for m in cfg.METRICS:
            groups = tukey_groups(panel_df, m, cfg.METRIC_HIGHER_IS_BETTER[m])
            stats = panel_df.groupby("method")[m].agg(["mean", "std"])
            for method in panel:
                rows.append({
                    "endpoint": endpoint, "metric": m, "method": method,
                    "label": cfg.METHOD_LABELS[method],
                    "mean": stats.loc[method, "mean"], "sd": stats.loc[method, "std"],
                    "tukey_group": groups[method],
                })
    pd.DataFrame(rows).to_csv(cfg.TABLE_DIR / "reference_panel.csv", index=False)
    print(f"  reference panel: {cfg.METHOD_LABELS[best_multi]} and "
          f"{cfg.METHOD_LABELS[best_uni]} against the four reference methods")
    return panel


# --- 6. the leakage control ----------------------------------------------
def leakage_control(metrics: pd.DataFrame, metric: str = "r2") -> pd.DataFrame | None:
    path = cfg.RESULTS_DIR / "fold_metrics_control.csv"
    if not path.exists():
        return None
    control = pd.read_csv(path)

    released = metrics[metrics["method"].isin(control["method"].unique())]
    a = released.groupby(["endpoint", "method"])[metric].mean().rename("in_sample")
    b = control.groupby(["endpoint", "method"])[metric].mean().rename("held_out")
    table = pd.concat([a, b], axis=1).dropna().reset_index()
    table["difference"] = table["held_out"] - table["in_sample"]
    table["label"] = table["method"].map(cfg.METHOD_LABELS)
    table.to_csv(cfg.TABLE_DIR / f"leakage_control_{metric}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    lo = min(table["in_sample"].min(), table["held_out"].min())
    hi = max(table["in_sample"].max(), table["held_out"].max())
    ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=1)
    for combo, marker in zip(COMBO_ORDER, ["o", "s", "^", "D"]):
        sub = table[table["method"].str.contains(f"_{combo}_")]
        ax.scatter(sub["in_sample"], sub["held_out"], marker=marker, s=34,
                   alpha=0.8, label=" + ".join(
                       cfg.MODALITY_LABELS[m] for m in cfg.COMBOS[combo]))
    ax.set_xlabel(f"{cfg.METRIC_LABELS[metric]}, meta-features in sample (as released)")
    ax.set_ylabel(f"{cfg.METRIC_LABELS[metric]}, meta-features held out")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(
        f"{cfg.ACTIVE.label}: late fusion, mean {table['difference'].mean():+.3f} "
        f"{cfg.METRIC_LABELS[metric]} when the meta-learner stops seeing its base "
        "learners' training predictions",
        fontsize=10,
    )
    save(fig, f"leakage_control_{metric}")
    return table


# --- statistical families -------------------------------------------------
def statistical_tests(metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every family the paper tests, Wilcoxon over endpoints with Holm inside."""
    out = {}
    for metric in cfg.METRICS:
        wide = analysis.endpoint_means(metrics, metric)
        higher = cfg.METRIC_HIGHER_IS_BETTER[metric]
        for name, pairs in [
            ("fusion", analysis.fusion_pairs()),
            ("ladder", analysis.ladder_pairs()),
            ("learner", analysis.learner_pairs()),
        ]:
            table = analysis.paired_wilcoxon(wide, pairs, higher)
            if not len(table):
                continue
            table["left_label"] = table["left"].map(cfg.METHOD_LABELS)
            table["right_label"] = table["right"].map(cfg.METHOD_LABELS)
            table["metric"] = metric
            out[f"{name}_{metric}"] = table
            table.to_csv(cfg.TABLE_DIR / f"wilcoxon_{name}_{metric}.csv", index=False)
    return out


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (endpoint, method), g in metrics.groupby(["endpoint", "method"]):
        row = {"endpoint": endpoint, "method": method,
               "label": cfg.METHOD_LABELS[method], "n_folds": len(g)}
        for metric in cfg.METRICS:
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_sd"] = g[metric].std()
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(cfg.TABLE_DIR / "summary.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", default=None)
    args = parser.parse_args()

    cfg.ensure_dirs()
    sns.set_style("whitegrid")

    metrics = load(args.metrics)
    grid = analysis.grid_frame(metrics)
    present = sorted(set(metrics["method"]))
    print(f"{cfg.ACTIVE.label}: {len(present)} methods, "
          f"{metrics['endpoint'].nunique()} endpoints, {len(metrics):,} fold scores")

    print("\nfigures")
    for metric in ("r2", "mae"):
        grid_distribution(grid, metric)
    modality_ladder(grid, metrics, "r2")

    print("\nis fusion better than the best single view?")
    for metric in cfg.METRICS:
        table = unimodal_comparison(metrics, metric)
        better = (table["difference"] > 0) if cfg.METRIC_HIGHER_IS_BETTER[metric] \
            else (table["difference"] < 0)
        print(f"  {metric:<9} fusion better on {int(better.sum())}/{len(table)} endpoints, "
              f"mean change {table['difference'].mean():+.4f}")

    print("\nreference comparison")
    reference_panel(metrics, "r2")

    print("\nleakage control")
    control = leakage_control(metrics, "r2")
    if control is None:
        print("  not run yet")
    else:
        print(f"  {len(control)} endpoint/configuration pairs, mean "
              f"{control['difference'].mean():+.4f} R squared when held out")

    if cfg.UNCERTAINTY_CSV.exists():
        print("\nuncertainty")
        unc = pd.read_csv(cfg.UNCERTAINTY_CSV)
        uncertainty_panels(unc)
        table = uncertainty_unimodal_comparison(unc)
        for _, row in table.iterrows():
            print(f"  {row['quantity']:<20} unimodal {row['best_unimodal']:7.4f}  "
                  f"multimodal {row['best_multimodal']:7.4f}  "
                  f"p(Holm) {row['p_holm']:.3f}")

    print("\nstatistical tests")
    tests = statistical_tests(metrics)
    for name, table in tests.items():
        if name.endswith("_r2"):
            n_sig = int(table["significant"].sum())
            print(f"  {name:<16} {n_sig}/{len(table)} comparisons survive Holm")

    summary_table(metrics)
    print("\nwrote tables/summary.csv")


if __name__ == "__main__":
    main()
