#!/usr/bin/env python
"""Step 7: Atom-JEPA against every arm, both data sets, and what decides it.

Run after 05_collect_metrics.py and 06_report.py for both data sets. Writes,
under results/:

  calls.csv     for each pair and endpoint x metric, the paired difference over
                the 25 shared folds, signed so positive is the challenger
                better, called a win, a loss or no call at p < 0.05. Per-pair
                tests, uncorrected; the Tukey tables are the corrected reading.
                The pairs are fine-tuned Atom-JEPA against each comparator, and
                the frozen Atom-JEPA arm against Monroe, which holds the head
                fixed and changes only the representation.
  levers.csv    per endpoint, the two steps from Monroe + TabPFN 3 to fine-tuned
                Atom-JEPA on one axis: swapping Monroe's frozen embedding for
                Atom-JEPA's under the same TabPFN 3, and then fine-tuning that
                encoder instead of freezing it.
  size.csv      per endpoint, how many molecules a fold trains on, beside the
                fine-tuned arm's margin over Monroe, and the rank correlation
                between the two.
  cost.csv      wall-clock per fine-tuned fold, by data set and machine.

    python 07_summary.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

import config as cfg

FT, PFN, MONROE = cfg.FT_METHOD, cfg.TABPFN_METHOD, cfg.MONROE_METHOD
PAIRS = [(FT, MONROE), (FT, PFN), (FT, cfg.CHEMELEON_METHOD), (FT, cfg.LGBM_METHOD),
         (PFN, MONROE)]
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
            for focus, other in PAIRS:
                pair = wide[[focus, other]].dropna()
                if len(pair) != cfg.N_REPEATS * cfg.N_SPLITS:
                    raise SystemExit(f"{ds} {endpoint} {focus}/{other}: {len(pair)} folds, not 25")
                gain = sign * (pair[focus] - pair[other])
                p = ttest_rel(pair[focus], pair[other]).pvalue
                call = "win" if p < 0.05 and gain.mean() > 0 else ("loss" if p < 0.05 else "no call")
                rows.append({"dataset": ds, "endpoint": endpoint, "metric": metric,
                             "focus": focus, "other": other,
                             "focus_mean": pair[focus].mean(), "other_mean": pair[other].mean(),
                             "gain": gain.mean(), "folds_won": int((gain > 0).sum()),
                             "p": p, "call": call})
    return pd.DataFrame(rows)


def levers(metrics: pd.DataFrame) -> pd.DataFrame:
    """Monroe + TabPFN 3 -> Atom-JEPA + TabPFN 3 -> Atom-JEPA fine-tuned.

    The first step changes the representation and holds the head; the second
    holds the encoder and changes how it is used. They add up to the whole gap
    between Monroe and the paper's arm. Signed so positive is better.
    """
    means = metrics.groupby(["dataset", "endpoint", "method"])[["r2", "mae"]].mean()
    rows = []
    for (ds, endpoint), g in means.groupby(level=[0, 1], sort=False):
        g = g.droplevel([0, 1])
        rows.append({
            "dataset": ds, "endpoint": endpoint,
            "representation_r2": g.loc[PFN, "r2"] - g.loc[MONROE, "r2"],
            "finetune_r2": g.loc[FT, "r2"] - g.loc[PFN, "r2"],
            "net_r2": g.loc[FT, "r2"] - g.loc[MONROE, "r2"],
            "representation_mae": g.loc[MONROE, "mae"] - g.loc[PFN, "mae"],
            "finetune_mae": g.loc[PFN, "mae"] - g.loc[FT, "mae"],
            "net_mae": g.loc[MONROE, "mae"] - g.loc[FT, "mae"],
        })
    return pd.DataFrame(rows)


def size(lv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Molecules per training fold against the fine-tuned arm's margin over Monroe."""
    rows = []
    for name, ds in cfg.DATASETS.items():
        master = pd.read_csv(cfg.PROJECT_DIR / "data" / name / "master.csv")
        folds = pd.read_csv(cfg.PROJECT_DIR / "folds" / name / "fold_assignments.csv")
        train = master[master[cfg.SET_COL] == "train"]
        for endpoint in ds.targets:
            labelled = set(train.loc[train[endpoint].notna(), cfg.ID_COL])
            per_fold = [len(labelled - set(g.loc[g["fold"] == f, cfg.ID_COL]))
                        for _, g in folds.groupby("repeat") for f in cfg.FOLDS]
            rows.append({"dataset": ds.label, "endpoint": endpoint,
                         "n_train": int(np.median(per_fold)),
                         "n_test": int((master[cfg.SET_COL].eq("test")
                                        & master[endpoint].notna()).sum())})
    t = pd.DataFrame(rows).merge(lv[["dataset", "endpoint", "net_r2", "net_mae"]],
                                 on=["dataset", "endpoint"])
    stats = {f"rho_{m}": spearmanr(t["n_train"], t[f"net_{m}"]).statistic for m in ("r2", "mae")}
    stats.update({f"p_{m}": spearmanr(t["n_train"], t[f"net_{m}"]).pvalue for m in ("r2", "mae")})
    return t.sort_values("n_train"), stats


def cost() -> pd.DataFrame:
    frames = [pd.read_csv(cfg.paths(n).diagnostics).assign(dataset=cfg.DATASETS[n].label)
              for n in cfg.DATASETS]
    d = pd.concat(frames, ignore_index=True)
    # The first Biogen folds ran before the host was recorded; they ran on apollo.
    d["host"] = d["host"].fillna("apollo").replace({"spark-4950": "spark"})
    d["gpu"] = d["gpu"].fillna("NVIDIA GeForce RTX 5070 Ti")
    return (d.groupby(["dataset", "host", "gpu"])
            .agg(folds=("seconds", "size"), median_min=("seconds", lambda s: s.median() / 60),
                 total_h=("seconds", lambda s: s.sum() / 3600),
                 median_best_epoch=("best_epoch", "median"),
                 peak_mem_gb=("peak_mem_gb", "max"))
            .reset_index())


def main() -> None:
    pd.set_option("display.width", 200)
    metrics = load_metrics()

    table = calls(metrics)
    table.to_csv(OUT / "calls.csv", index=False)
    print("15 endpoints x 3 metrics = 45 calls per pair, paired t over 25 folds")
    print(pd.crosstab([table["focus"], table["other"]], [table["dataset"], table["call"]])
          .to_string(), "\n")

    lv = levers(metrics)
    lv.to_csv(OUT / "levers.csv", index=False)
    print(lv.round(3).to_string(index=False))
    for d, g in lv.groupby("dataset"):
        print(f"  {d} median R2: representation {g.representation_r2.median():+.3f}, "
              f"fine-tuning {g.finetune_r2.median():+.3f}, net {g.net_r2.median():+.3f}")
    print()

    sz, stats = size(lv)
    sz.to_csv(OUT / "size.csv", index=False)
    print(sz.round(3).to_string(index=False))
    print(f"  Spearman n_train vs net R2 {stats['rho_r2']:+.2f} (p {stats['p_r2']:.2f}), "
          f"vs net MAE {stats['rho_mae']:+.2f} (p {stats['p_mae']:.2f})\n")
    pd.Series(stats).to_csv(OUT / "size_stats.csv", header=["value"])

    c = cost()
    c.to_csv(OUT / "cost.csv", index=False)
    print(c.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
