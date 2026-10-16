#!/usr/bin/env python
"""Step 7: QuADMET-Former against every arm, both data sets, and internal ablations.

Run after 05_collect_metrics.py and 06_report.py for both data sets. Writes,
under results/:

  calls.csv              for each pair and endpoint x metric, the paired difference
                         over the 25 shared folds, signed so positive is the
                         challenger better, called a win, a loss or no call at
                         p < 0.05.
  levers.csv             per endpoint, the two steps from Monroe + TabPFN 3 to
                         fine-tuned QuADMET-Former: swapping Monroe's frozen
                         embedding for QuADMET-Former's under the same TabPFN 3,
                         and then fine-tuning that encoder with LP-FT instead of
                         reading it in context.
  size.csv               per endpoint, how many molecules a fold trains on, beside
                         the fine-tuned arm's margin over Monroe.
  ablations_summary.csv  internal QuADMET-Former ablations across all 15 endpoints:
                         LP-FT vs Scratch (testing the paper's central claim),
                         LP-FT vs Stage-1 LP, 6-layer vs 3-layer truncation, and
                         per-endpoint selected checkpoint vs fixed `Single [All]`.
  cost.csv               wall-clock per fine-tuned fold, by data set and machine.

    python 07_summary.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

import config as cfg

FT, PFN, MONROE = cfg.FT_METHOD, cfg.TABPFN_METHOD, cfg.MONROE_METHOD
AJ_FT = cfg.ATOMJEPA_FT_METHOD
PAIRS = [
    (FT, AJ_FT),
    (FT, MONROE),
    (FT, PFN),
    (FT, cfg.CHEMELEON_METHOD),
    (FT, cfg.LGBM_METHOD),
    (PFN, MONROE),
    (PFN, cfg.CHEMELEON_METHOD),
    (PFN, cfg.LGBM_METHOD),
]
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
    """Monroe + TabPFN 3 -> QuADMET-Former + TabPFN 3 -> QuADMET-Former fine-tuned."""
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


def summarise_ablations(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare fine-tuned QuADMET-Former against Scratch, Stage-1 LP, 3-layer truncation, and fixed [All]."""
    frames = []
    for n in cfg.DATASETS:
        p = cfg.paths(n).ablations
        if p.exists():
            frames.append(pd.read_csv(p).assign(dataset=cfg.DATASETS[n].label))
    if not frames:
        return pd.DataFrame()
    abl = pd.concat(frames, ignore_index=True)

    ft_sub = (
        metrics[metrics["method"] == FT][["dataset", "endpoint", "repeat", "fold", "r2", "spearman", "mae"]]
        .assign(variant="quadmet_ft")
    )
    all_var = pd.concat([ft_sub, abl[["dataset", "endpoint", "repeat", "fold", "variant", "r2", "spearman", "mae"]]],
                        ignore_index=True)

    rows = []
    for (ds, endpoint), sub in all_var.groupby(["dataset", "endpoint"], sort=False):
        wide_r2 = sub.pivot_table(index=["repeat", "fold"], columns="variant", values="r2")
        wide_mae = sub.pivot_table(index=["repeat", "fold"], columns="variant", values="mae")
        row = {"dataset": ds, "endpoint": endpoint}
        for v in ("quadmet_ft", "quadmet_scratch", "quadmet_lp", "quadmet_lp_3layer", "quadmet_lp_fixed_all"):
            if v in wide_r2:
                row[f"{v}_r2_mean"] = float(wide_r2[v].mean())
                row[f"{v}_r2_sd"] = float(wide_r2[v].std())
                row[f"{v}_mae_mean"] = float(wide_mae[v].mean())
                row[f"{v}_mae_sd"] = float(wide_mae[v].std())
        if {"quadmet_ft", "quadmet_scratch"} <= set(wide_r2.columns):
            pair_r2 = wide_r2[["quadmet_ft", "quadmet_scratch"]].dropna()
            pair_mae = wide_mae[["quadmet_ft", "quadmet_scratch"]].dropna()
            if len(pair_r2) >= 2:
                diff_r2 = pair_r2["quadmet_ft"] - pair_r2["quadmet_scratch"]
                diff_mae = pair_mae["quadmet_scratch"] - pair_mae["quadmet_ft"]
                p_r2 = float(ttest_rel(pair_r2["quadmet_ft"], pair_r2["quadmet_scratch"]).pvalue)
                row["ft_vs_scratch_r2"] = float(diff_r2.mean())
                row["ft_vs_scratch_mae"] = float(diff_mae.mean())
                row["ft_vs_scratch_p"] = p_r2
                row["ft_vs_scratch_call"] = ("win" if p_r2 < 0.05 and diff_r2.mean() > 0
                                             else ("loss" if p_r2 < 0.05 else "no call"))
        if {"quadmet_ft", "quadmet_lp"} <= set(wide_r2.columns):
            pair_lp = wide_r2[["quadmet_ft", "quadmet_lp"]].dropna()
            if len(pair_lp) >= 2:
                diff_lp = pair_lp["quadmet_ft"] - pair_lp["quadmet_lp"]
                row["ft_vs_lp_r2"] = float(diff_lp.mean())
                row["ft_vs_lp_p"] = float(ttest_rel(pair_lp["quadmet_ft"], pair_lp["quadmet_lp"]).pvalue)
        if {"quadmet_lp", "quadmet_lp_3layer"} <= set(wide_r2.columns):
            pair_6v3 = wide_r2[["quadmet_lp", "quadmet_lp_3layer"]].dropna()
            if len(pair_6v3) >= 2:
                diff_6v3 = pair_6v3["quadmet_lp"] - pair_6v3["quadmet_lp_3layer"]
                p_6v3 = float(ttest_rel(pair_6v3["quadmet_lp"], pair_6v3["quadmet_lp_3layer"]).pvalue)
                row["lp6_vs_lp3_r2"] = float(diff_6v3.mean())
                row["lp6_vs_lp3_p"] = p_6v3
        if {"quadmet_lp", "quadmet_lp_fixed_all"} <= set(wide_r2.columns):
            pair_sel = wide_r2[["quadmet_lp", "quadmet_lp_fixed_all"]].dropna()
            if len(pair_sel) >= 2:
                diff_sel = pair_sel["quadmet_lp"] - pair_sel["quadmet_lp_fixed_all"]
                p_sel = float(ttest_rel(pair_sel["quadmet_lp"], pair_sel["quadmet_lp_fixed_all"]).pvalue)
                row["selected_vs_all_r2"] = float(diff_sel.mean())
                row["selected_vs_all_p"] = p_sel
        rows.append(row)
    return pd.DataFrame(rows)


def cost() -> pd.DataFrame:
    frames = [pd.read_csv(cfg.paths(n).diagnostics).assign(dataset=cfg.DATASETS[n].label)
              for n in cfg.DATASETS if cfg.paths(n).diagnostics.exists()]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["host"] = d["host"].fillna("apollo").replace({"spark-4950": "spark"})
    d["gpu"] = d["gpu"].fillna("NVIDIA GeForce RTX 5070 Ti")
    return (d.groupby(["dataset", "host", "gpu"])
            .agg(folds=("seconds", "size"), median_sec=("seconds", "median"),
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

    abl = summarise_ablations(metrics)
    if not abl.empty:
        abl.to_csv(OUT / "ablations_summary.csv", index=False)
        print("internal QuADMET-Former ablations:")
        print(abl.round(3).to_string(index=False), "\n")

    c = cost()
    if not c.empty:
        c.to_csv(OUT / "cost.csv", index=False)
        print(c.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
