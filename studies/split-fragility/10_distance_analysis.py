#!/usr/bin/env python
"""Step 10: how far out of distribution each scheme actually puts its test set.

The six schemes are the independent variable of this whole study, and "more
stringent" is so far only a word and an ordering borrowed from the paper. This
step measures it. For every fold, every test molecule's nearest training
neighbour is found by ECFP4 Tanimoto similarity, and the fold is summarised by
the mean of those similarities -- the quantity the paper calls MAX-ECFP-SIM and
plots in its Figures 5 and 6.

Two things come out of that.

The first is a check on the ordering. If the schemes are what they claim to be,
the similarity should fall monotonically along them, and the ones the paper
calls equivalent in stringency should land together.

The second is the more useful one, and it is a comparison rather than a
description: with a continuous measure of how far a fold's test set sits from
its training set, each method's accuracy can be plotted against it directly
instead of against a six-valued label. The slope of that line is how fast a
method decays as the test molecules move away, which is the thing the paper
argues is the same for all methods. It is measured here per method, on the gap
to the naive baseline rather than on raw MAE, because raw MAE confounds the
decay with the baseline drift.

Writes:

  tables/fold_distance.csv       mean and median nearest-neighbour similarity
                                 per (target, scheme, fold), and the fraction of
                                 test molecules with no neighbour above 0.35
  tables/fold_sizes.csv          how many molecules each scheme trains and tests
                                 on, and how evenly, which decides whether a
                                 comparison across schemes is confounded
  tables/distance_decay.csv      per method, the slope of the gap to baseline
                                 against fold similarity, with its correlation
  figures/similarity_by_split.png    the distributions, by scheme
  figures/decay_with_distance.png    the gap to baseline against similarity

    python 10_distance_analysis.py
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, spearmanr

import config as cfg
from fold_data import FoldData
from split_stats import SUBJECT_COL, gap_to_baseline

# The Butina threshold the paper clusters at. A test molecule whose nearest
# training neighbour falls below it would not have been clustered with anything
# it could have learned from, which makes the fraction below it a readable
# summary of how much of a test set is genuinely novel.
NOVEL_BELOW = 0.35

# The four schemes that train on 72% of a target, to within a rounding error.
# Across all six, training-set size and distance are correlated; within these
# four they are not, so they are the comparison that varies distance alone.
SIZE_MATCHED = ["random", "scaffold", "butina", "umap"]


def nearest_neighbour_similarity(bits: np.ndarray, test: np.ndarray,
                                 train: np.ndarray) -> np.ndarray:
    """Each test molecule's Tanimoto similarity to its closest training molecule."""
    A = bits[test].astype(bool).astype(np.float32)
    B = bits[train].astype(bool).astype(np.float32)
    shared = A @ B.T
    union = A.sum(1, keepdims=True) + B.sum(1, keepdims=True).T - shared
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, shared / union, 0.0)
    return similarity.max(axis=1)


def fold_distances(data: FoldData) -> pd.DataFrame:
    records = []
    for target in cfg.TARGETS:
        for split in cfg.SPLITS:
            for fold in cfg.FOLDS:
                rows = data.fold(target, split, fold)
                sim = nearest_neighbour_similarity(data.ecfp_bits, rows.test, rows.train)
                records.append(
                    {
                        "target": target,
                        "split": split,
                        "fold": fold,
                        SUBJECT_COL: f"tid{target}_f{fold}",
                        "n_test": len(rows.test),
                        "n_train": len(rows.train),
                        "mean_max_sim": float(sim.mean()),
                        "median_max_sim": float(np.median(sim)),
                        "frac_novel": float((sim < NOVEL_BELOW).mean()),
                    }
                )
    return pd.DataFrame(records)


def size_table(distances: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """How much data each scheme trains and tests on, and how consistently.

    Worth its own table because it decides what a comparison across schemes can
    mean. If the harder schemes also train on less data, then "accuracy collapses
    as the split gets more stringent" is two effects wearing one label, and the
    paper's conclusion would not separate them.

    Sizes are reported as a fraction of each target, because the ten data sets
    run from 1,000 to 2,273 molecules and an absolute count would mostly measure
    which target it came from. The within-target spread over the five folds is
    reported separately: that is the one a scheme controls, and it is what says
    whether its folds are interchangeable.
    """
    n = master.groupby("target").size().rename("n_total")
    d = distances.merge(n, on="target")
    d["train_pct"] = 100 * d["n_train"] / d["n_total"]
    d["test_pct"] = 100 * d["n_test"] / d["n_total"]

    rows = []
    for split, g in d.groupby("split"):
        within = g.groupby("target")
        rows.append(
            {
                "split": split,
                "train": g["n_train"].mean(),
                "test": g["n_test"].mean(),
                "train_pct": g["train_pct"].mean(),
                "train_pct_sd": g["train_pct"].std(),
                "test_pct": g["test_pct"].mean(),
                "test_pct_sd": g["test_pct"].std(),
                # Spread over the five folds of one target, averaged over targets.
                "within_train_range": within["n_train"].agg(lambda s: s.max() - s.min()).mean(),
                "within_test_cv": within["n_test"].agg(
                    lambda s: 100 * s.std() / s.mean()
                ).mean(),
                "mean_max_sim": g["mean_max_sim"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("split").reindex(cfg.SPLITS).reset_index()


def size_confound(distances: pd.DataFrame) -> pd.DataFrame:
    """Whether training-set size travels with distance, overall and within the
    four schemes that hold it fixed.

    The four cluster-based schemes all train on 72% of a target, to within a
    rounding error, while their nearest-neighbour similarity runs from 0.75 down
    to 0.39. That is the clean experiment in this collection: distance varies and
    nothing else does. The diverse and time schemes are not that -- diverse
    trains on 18% of a target and time on 50% -- so a decay measured across all
    six confounds moving the test set away with taking the training data
    away, and this table says by how much.
    """
    matched = ["random", "scaffold", "butina", "umap"]
    rows = []
    for label, subset in (("all six schemes", distances),
                          ("the four 72%-train schemes",
                           distances[distances["split"].isin(matched)])):
        rho = spearmanr(subset["n_train"], subset["mean_max_sim"])
        rows.append({"over": label, "n_folds": len(subset),
                     "spearman_train_vs_similarity": rho.statistic,
                     "p_value": rho.pvalue})
    return pd.DataFrame(rows)


def plot_similarity(distances: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    sns.boxplot(data=distances, x="split", y="mean_max_sim", order=cfg.SPLITS,
                hue="split", hue_order=cfg.SPLITS, palette="crest", legend=False,
                ax=axes[0])
    axes[0].set_ylabel("mean nearest-neighbour Tanimoto")
    axes[0].set_title("How close a test molecule sits to the training set")

    sns.boxplot(data=distances, x="split", y="frac_novel", order=cfg.SPLITS,
                hue="split", hue_order=cfg.SPLITS, palette="crest", legend=False,
                ax=axes[1])
    axes[1].set_ylabel(f"fraction of test molecules below {NOVEL_BELOW}")
    axes[1].set_title("How much of a test set has no close analogue")

    for ax in axes:
        ax.set_xlabel("")
        ax.set_xticks(range(len(cfg.SPLITS)))
        ax.set_xticklabels([cfg.SPLIT_LABELS[s] for s in cfg.SPLITS], rotation=20, ha="right")

    fig.suptitle("The six splitting schemes, measured rather than named "
                 "(300 folds, ECFP4)", fontsize=12)
    fig.tight_layout()
    path = cfg.FIGURE_DIR / "similarity_by_split.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")


def decay(metrics: pd.DataFrame, distances: pd.DataFrame) -> pd.DataFrame:
    """The gap to the naive baseline against fold similarity, per method.

    Pooled over all six schemes deliberately: the point is to replace the
    six-valued label with the continuous quantity underneath it, and a slope
    fitted within one scheme would have almost no range to fit over.
    """
    frames = []
    for split in cfg.SPLITS:
        g = metrics[(metrics["split"] == split) & (metrics["method"].isin(cfg.METHODS))]
        g = g[["method", SUBJECT_COL, "mae"]].dropna()
        if cfg.MEDIAN_METHOD not in set(g["method"]):
            continue
        gap = gap_to_baseline(g, "mae", cfg.MEDIAN_METHOD, higher_is_better=False)
        gap["split"] = split
        frames.append(gap)
    if not frames:
        return pd.DataFrame()

    gaps = pd.concat(frames, ignore_index=True).merge(
        distances[["split", SUBJECT_COL, "mean_max_sim", "frac_novel"]],
        on=["split", SUBJECT_COL], how="left",
    )

    rows = []
    for method, g in gaps.groupby("method"):
        fit = linregress(g["mean_max_sim"], g["gap"])
        rho = spearmanr(g["mean_max_sim"], g["gap"])
        # The same fit over only the four schemes that hold training-set size
        # fixed. Over all six, distance and training-set size move together
        # (Spearman +0.38), so a slope fitted on all of them is partly a slope
        # against how much data the model was given. Within these four that
        # correlation is gone, so this is the slope against distance alone. It
        # is fitted over a narrower range of similarity and on two thirds of
        # the folds, so it is noisier -- and it is the one to believe where the
        # two disagree.
        m = g[g["split"].isin(SIZE_MATCHED)]
        matched_fit = linregress(m["mean_max_sim"], m["gap"]) if len(m) > 2 else None
        rows.append(
            {
                "method": method,
                "n_folds": len(g),
                "slope": fit.slope,
                "intercept": fit.intercept,
                "r_squared": fit.rvalue ** 2,
                "spearman": rho.statistic,
                "p_value": fit.pvalue,
                # What the fit says the method is worth where the test molecules
                # are most distant, which is where the paper's argument lives.
                "gap_at_sim_0.30": fit.intercept + 0.30 * fit.slope,
                "gap_at_sim_0.60": fit.intercept + 0.60 * fit.slope,
                "n_folds_matched": len(m),
                "slope_matched": matched_fit.slope if matched_fit else float("nan"),
                "gap_at_sim_0.40_matched": (
                    matched_fit.intercept + 0.40 * matched_fit.slope
                    if matched_fit else float("nan")
                ),
            }
        )
    table = pd.DataFrame(rows).set_index("method").reindex(
        [m for m in cfg.METHODS if m != cfg.MEDIAN_METHOD]
    ).dropna(how="all").reset_index()

    # The figure.
    palette = {
        cfg.KNN_METHOD: "#4C72B0", cfg.LGBM_METHOD: "#DD8452",
        cfg.CHEMELEON_METHOD: "#55A868", cfg.MONROE35_METHOD: "#61483A",
        "pub_knn": "#A3B5CE", "pub_svm": "#8172B3", "pub_rf": "#CCB974",
        "pub_xgboost": "#DA8BC3", "pub_mlp": "#C44E52", "pub_gnn": "#64B5CD",
    }
    arms = [m for m in cfg.METHODS if m in set(gaps["method"])]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in arms:
        g = gaps[gaps["method"] == method]
        ax.scatter(g["mean_max_sim"], g["gap"], s=7, alpha=0.30,
                   color=palette.get(method, "#7A8783"))
        fit = linregress(g["mean_max_sim"], g["gap"])
        xs = np.linspace(gaps["mean_max_sim"].min(), gaps["mean_max_sim"].max(), 50)
        ax.plot(xs, fit.intercept + fit.slope * xs, lw=2.2,
                color=palette.get(method, "#7A8783"), label=cfg.METHOD_LABELS[method])
    ax.axhline(0, color="#9E3B3B", ls="--", lw=1.2, zorder=0)
    ax.set_xlabel("mean nearest-neighbour Tanimoto, test to train (one point per fold)")
    ax.set_ylabel("MAE taken off the naive predictor (pIC$_{50}$)")
    ax.set_title("How fast each method decays as the test set moves away\n"
                 "All 300 folds, all six schemes pooled", fontsize=12)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    path = cfg.FIGURE_DIR / "decay_with_distance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}")

    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    cfg.ensure_dirs()
    sns.set_style("whitegrid")

    data = FoldData.load(need="ecfp")
    distances = fold_distances(data)
    distances.to_csv(cfg.TABLE_DIR / "fold_distance.csv", index=False)
    print(f"wrote fold_distance.csv ({len(distances)} folds)")

    sizes = size_table(distances, data.master)
    sizes.to_csv(cfg.TABLE_DIR / "fold_sizes.csv", index=False)
    print("\nhow much data each scheme trains and tests on:")
    print(sizes.round(2).to_string(index=False))

    confound = size_confound(distances)
    confound.to_csv(cfg.TABLE_DIR / "size_confound.csv", index=False)
    print("\nis training-set size confounded with distance?")
    print(confound.round(4).to_string(index=False))

    print("\nnearest-neighbour similarity by scheme:")
    print(
        distances.groupby("split")[["mean_max_sim", "median_max_sim", "frac_novel"]]
        .mean()
        .reindex(cfg.SPLITS)
        .round(3)
        .to_string()
    )

    print("\nfigures:")
    plot_similarity(distances)

    if not cfg.FOLD_METRICS_CSV.exists():
        print(f"\n{cfg.FOLD_METRICS_CSV.name} not found -- run 07_collect_metrics.py "
              "for the decay analysis")
        return
    metrics = pd.read_csv(cfg.FOLD_METRICS_CSV)
    metrics[SUBJECT_COL] = "tid" + metrics["target"].astype(str) + "_f" + metrics["fold"].astype(str)

    table = decay(metrics, distances)
    if len(table):
        table.to_csv(cfg.TABLE_DIR / "distance_decay.csv", index=False)
        print("\nhow the gap to the naive baseline decays with distance:")
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
