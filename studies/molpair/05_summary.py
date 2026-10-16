#!/usr/bin/env python
"""Step 5: MolPAIR against every arm, both data sets, and what the correction did.

Run after 03_collect_metrics.py and 04_report.py for both data sets. Writes, under
results/:

  molpair_vs_arms.csv   for each comparator and endpoint x metric, the paired
                        difference over the 25 shared folds, MolPAIR minus the
                        comparator signed so positive is MolPAIR better, called a
                        win, a loss or no call at p < 0.05. Per-pair tests,
                        uncorrected; the Tukey tables are the corrected reading.
  levers.csv            per endpoint, what the pair stage adds to its own global
                        prediction against what swapping CheMeleon's frozen
                        fingerprint for Monroe's embedding is worth, in R2 and MAE.
  by_similarity.csv     every test prediction, binned by the similarity of its
                        nearest training molecule: how often the correction lowered
                        the absolute error, and the mean change -- the paper's
                        Table 21, on these splits.
  regime.csv            the same correction, scored per fold on only the test
                        molecules the paper's protocol would have kept (nearest
                        training molecule at Tanimoto <= 0.60) and on the rest.
  activity.csv          anchor counts, fallbacks and gate values per data set.
  cost.csv              wall-clock per fold on the GPU.
  figures/correction_by_similarity.png   by_similarity.csv, drawn: how often the
                        correction helps, and by how much, against the nearest
                        training molecule's similarity, with the paper's cutoff.

    python 05_summary.py
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

import config as cfg

FOCUS = cfg.MOLPAIR_METHOD
COMPARATORS = [cfg.GLOBAL_METHOD, cfg.MONROE_METHOD, cfg.CHEMELEON_METHOD, cfg.LGBM_METHOD]
# The paper's primary split drops test molecules more similar than this to any
# training molecule.
PAPER_CUTOFF = 0.60
SIM_BINS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001]
OUT = cfg.PROJECT_DIR / "results"


def load_metrics() -> pd.DataFrame:
    frames = [pd.read_csv(cfg.paths(n).fold_metrics).assign(dataset=cfg.DATASETS[n].label)
              for n in cfg.DATASETS]
    return pd.concat(frames, ignore_index=True)


def calls(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ds, endpoint), sub in metrics.groupby(["dataset", "endpoint"], sort=False):
        for metric in cfg.METRICS:
            wide = sub.pivot_table(index=["repeat", "fold"], columns="method", values=metric)
            sign = 1 if cfg.METRIC_HIGHER_IS_BETTER[metric] else -1
            for other in COMPARATORS:
                pair = wide[[FOCUS, other]].dropna()
                if len(pair) != cfg.N_REPEATS * cfg.N_SPLITS:
                    raise SystemExit(f"{ds} {endpoint} {other}: {len(pair)} folds, not 25")
                gain = sign * (pair[FOCUS] - pair[other])
                p = ttest_rel(pair[FOCUS], pair[other]).pvalue
                call = "win" if p < 0.05 and gain.mean() > 0 else ("loss" if p < 0.05 else "no call")
                rows.append({"dataset": ds, "endpoint": endpoint, "metric": metric,
                             "other": other, "molpair_mean": pair[FOCUS].mean(),
                             "other_mean": pair[other].mean(), "gain": gain.mean(),
                             "folds_won": int((gain > 0).sum()), "p": p, "call": call})
    return pd.DataFrame(rows)


def levers(metrics: pd.DataFrame) -> pd.DataFrame:
    """The pair stage against the representation, on one axis.

    Both are paired differences from the same CheMeleon + TabPFN 3 prediction:
    adding MolPAIR's correction to it, or replacing its input with Monroe's
    embedding under the same TabPFN 3.
    """
    means = metrics.groupby(["dataset", "endpoint", "method"])[["r2", "mae"]].mean()
    rows = []
    for (ds, endpoint), g in means.groupby(level=[0, 1], sort=False):
        g = g.droplevel([0, 1])
        rows.append({
            "dataset": ds, "endpoint": endpoint,
            "pair_stage_r2": g.loc[FOCUS, "r2"] - g.loc[cfg.GLOBAL_METHOD, "r2"],
            "representation_r2": g.loc[cfg.MONROE_METHOD, "r2"] - g.loc[cfg.GLOBAL_METHOD, "r2"],
            "finetune_r2": g.loc[cfg.CHEMELEON_METHOD, "r2"] - g.loc[cfg.GLOBAL_METHOD, "r2"],
            "pair_stage_mae": g.loc[cfg.GLOBAL_METHOD, "mae"] - g.loc[FOCUS, "mae"],
            "representation_mae": g.loc[cfg.GLOBAL_METHOD, "mae"] - g.loc[cfg.MONROE_METHOD, "mae"],
            "finetune_mae": g.loc[cfg.GLOBAL_METHOD, "mae"] - g.loc[cfg.CHEMELEON_METHOD, "mae"],
        })
    return pd.DataFrame(rows)


def load_diagnostics() -> pd.DataFrame:
    frames = []
    for n in cfg.DATASETS:
        d = pd.read_parquet(cfg.paths(n).diagnostics)
        d["dataset"] = cfg.DATASETS[n].label
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["err_global"] = (d["y_true"] - d["mu"]).abs()
    d["err_molpair"] = (d["y_true"] - d["mu"] - d["correction"]).abs()
    d["delta"] = d["err_global"] - d["err_molpair"]  # positive = the correction helped
    return d


def by_similarity(d: pd.DataFrame) -> pd.DataFrame:
    d = d.assign(bin=pd.cut(d["max_sim_all"], SIM_BINS, right=False))
    g = d.groupby(["dataset", "bin"], observed=True)
    totals = d.groupby("dataset").size()
    out = pd.DataFrame({
        "n": g.size(),
        "share_of_test": g.size() / totals.reindex(g.size().index.get_level_values(0)).to_numpy(),
        "improved_pct": 100 * g.apply(lambda x: (x["delta"] > 0).mean(), include_groups=False),
        "worsened_pct": 100 * g.apply(lambda x: (x["delta"] < 0).mean(), include_groups=False),
        "mean_abs_err_change": g["delta"].mean(),
        "mean_gate": g["gate"].mean(),
        "mean_abs_correction": g["correction"].apply(lambda c: c.abs().mean()),
    })
    return out.reset_index().assign(bin=lambda x: x["bin"].astype(str))


def regime(d: pd.DataFrame) -> pd.DataFrame:
    """Per-fold MAE of both predictions on the paper's population and the rest.

    MAE is scored per endpoint and fold on each subset, then paired over the 25
    folds, so a fold where one subset is small is not over-weighted against one
    where it is large.
    """
    d = d.assign(paper=d["max_sim_all"] <= PAPER_CUTOFF)
    rows = []
    for (ds, endpoint, paper), g in d.groupby(["dataset", "endpoint", "paper"], sort=False):
        per_fold = g.groupby(["repeat", "fold"]).agg(
            n=("delta", "size"), mae_global=("err_global", "mean"),
            mae_molpair=("err_molpair", "mean"))
        if len(per_fold) < cfg.N_REPEATS * cfg.N_SPLITS or per_fold["n"].min() < 5:
            continue
        gain = per_fold["mae_global"] - per_fold["mae_molpair"]
        p = ttest_rel(per_fold["mae_global"], per_fold["mae_molpair"]).pvalue
        rows.append({"dataset": ds, "endpoint": endpoint,
                     "population": "paper" if paper else "close",
                     "n_per_fold": per_fold["n"].mean(), "mae_global": per_fold["mae_global"].mean(),
                     "mae_molpair": per_fold["mae_molpair"].mean(), "gain": gain.mean(),
                     "p": p, "call": "win" if p < 0.05 and gain.mean() > 0 else
                     ("loss" if p < 0.05 else "no call")})
    return pd.DataFrame(rows)


def activity(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("dataset")
    return pd.DataFrame({
        "query_predictions": g.size(),
        "no_anchor_pct": 100 * g.apply(lambda x: (x["n_anchors"] == 0).mean(), include_groups=False),
        "full_32_pct": 100 * g.apply(lambda x: (x["n_anchors"] == cfg.MOLPAIR["query_anchors"]).mean(),
                                     include_groups=False),
        "active_pct": 100 * g.apply(lambda x: (x["correction"] != 0).mean(), include_groups=False),
        "mean_anchors": g["n_anchors"].mean(),
        "median_max_sim": g["max_sim_all"].median(),
        "paper_population_pct": 100 * g.apply(lambda x: (x["max_sim_all"] <= PAPER_CUTOFF).mean(),
                                              include_groups=False),
        "mean_gate": g["gate"].mean(),
        "improved_pct": 100 * g.apply(lambda x: (x["delta"] > 0).mean(), include_groups=False),
    })


def similarity_figure(d: pd.DataFrame) -> None:
    """Share of test predictions the correction improved, and the mean change in
    absolute error, by the similarity of each test molecule's nearest training
    molecule. Bins with fewer than 50 predictions, or below the anchor threshold,
    are not drawn."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    centres = (np.array(SIM_BINS[:-1]) + np.minimum(SIM_BINS[1:], 1.0)) / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = {"ExpansionRx": "#5B2C83", "Biogen ADME": "#B7A3D6"}
    for ds, g in d.groupby("dataset"):
        b = pd.cut(g["max_sim_all"], SIM_BINS, right=False, labels=False)
        agg = g.groupby(b).agg(n=("delta", "size"), improved=("delta", lambda x: 100 * (x > 0).mean()),
                               change=("delta", "mean"))
        # Below the 0.20 acceptance threshold no anchor is accepted and nothing is
        # corrected, so those bins are not drawn.
        agg = agg[(agg["n"] >= 50) & (np.array(SIM_BINS)[agg.index.to_numpy()]
                                      >= cfg.MOLPAIR["anchor_min_similarity"])]
        x = centres[agg.index.to_numpy()]
        axes[0].plot(x, agg["improved"], "o-", color=colors[ds], label=ds)
        axes[1].plot(x, agg["change"], "o-", color=colors[ds], label=ds)
    for ax in axes:
        ax.axvline(PAPER_CUTOFF, color="0.4", ls="--", lw=1)
        ax.set_xlabel("Tanimoto to the nearest training molecule (Morgan r2)")
        ax.set_xlim(0.15, 1.0)
    axes[0].axhline(50, color="0.3", lw=1)
    axes[0].set_ylabel("test predictions improved, %")
    axes[1].axhline(0, color="0.3", lw=1)
    axes[1].set_ylabel("mean drop in absolute error\n(positive = correction helped)")
    axes[0].text(PAPER_CUTOFF - 0.01, axes[0].get_ylim()[1], "paper's split keeps\nonly what is left of here",
                 ha="right", va="top", fontsize=8, color="0.3")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    out = OUT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "correction_by_similarity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/correction_by_similarity.png")


def cost() -> pd.DataFrame:
    frames = [pd.read_csv(cfg.paths(n).timing).assign(dataset=cfg.DATASETS[n].label)
              for n in cfg.DATASETS if cfg.paths(n).timing.exists()]
    t = pd.concat(frames, ignore_index=True)
    return t.groupby("dataset").agg(
        folds=("seconds", "size"), total_h=("seconds", lambda s: s.sum() / 3600),
        median_s=("seconds", "median"), max_s=("seconds", "max"),
        median_query_pairs=("n_query_pairs", "median"),
        relation_features=("relation_features", "median"))


def main() -> None:
    pd.set_option("display.width", 200)
    metrics = load_metrics()

    table = calls(metrics)
    table.to_csv(OUT / "molpair_vs_arms.csv", index=False)
    print("MolPAIR against each arm, 15 endpoints x 3 metrics = 45 calls each")
    print(pd.crosstab(table["other"], table["call"]).reindex(
        index=COMPARATORS, columns=["win", "no call", "loss"], fill_value=0).to_string(), "\n")
    mae = table[table["metric"] == "mae"]
    print("the same, MAE only, by data set")
    print(pd.crosstab(mae["other"], [mae["dataset"], mae["call"]]).to_string(), "\n")

    lv = levers(metrics)
    lv.to_csv(OUT / "levers.csv", index=False)
    print("per endpoint, against CheMeleon + TabPFN 3")
    print(lv.round(3).to_string(index=False))
    print(f"  median: pair stage R2 {lv.pair_stage_r2.median():+.3f}, representation "
          f"{lv.representation_r2.median():+.3f}, fine-tuning {lv.finetune_r2.median():+.3f}\n")

    d = load_diagnostics()
    sim = by_similarity(d)
    sim.to_csv(OUT / "by_similarity.csv", index=False)
    similarity_figure(d)
    print(sim.round(3).to_string(index=False), "\n")

    reg = regime(d)
    reg.to_csv(OUT / "regime.csv", index=False)
    print("the correction on the paper's population (nearest train <= 0.60) and the rest, MAE")
    print(pd.crosstab([reg["population"]], reg["call"]).to_string())
    print(reg.groupby("population")[["gain", "n_per_fold"]].mean().round(4).to_string(), "\n")

    act = activity(d)
    act.to_csv(OUT / "activity.csv")
    print(act.round(2).to_string(), "\n")

    c = cost()
    c.to_csv(OUT / "cost.csv")
    print("GPU wall-clock, RTX 5070 Ti")
    print(c.round(1).to_string())


if __name__ == "__main__":
    main()
