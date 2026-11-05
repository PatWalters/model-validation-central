#!/usr/bin/env python
"""Step 7: Atom-JEPA-Bio against every arm, both data sets, and what decides it.

Run after 05_collect_metrics.py and 06_report.py for both data sets. Writes,
under results/:

  calls.csv     for each pair and endpoint x metric, the paired difference over
                the 25 shared folds, signed so positive is the first arm
                better, called a win, a loss or no call at p < 0.05. Per-pair
                tests, uncorrected; the Tukey tables are the corrected reading.
  levers.csv    per endpoint, R^2 and MAE of the steps between arms: the
                bioassay stage, frozen and fine-tuned; and the path from
                Monroe + TabPFN 3 to fine-tuned Bio, first swapping the frozen
                embedding under the same TabPFN 3, then fine-tuning instead.
  size.csv      per endpoint, molecules per training fold beside Bio's margin
                over Monroe, and the rank correlation between the two.
  rerun.csv     the reproduction check: on the folds the stage-1 recipe was
                re-run on this study's GPU, how far the re-run moved from the
                imported atomjepa_ft, against how far bio_ft moved from it on
                the same folds.
  exposure.csv  Biogen only: the test set split into molecules PCBA has an
                exact InChI match for and molecules it does not (from
                09_check_pcba_overlap.py), and each arm scored on each half.
  cost.csv      wall-clock per fine-tuned fold, by data set and machine.
  paper_tally.csv  the authors' own scoreboard, recomputed: Tukey HSD over the
                six arms of their tables only (no stage-1 frozen arm), each arm
                counted best alone, tied for best or worse per endpoint x metric.

    python 07_summary.py
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

import config as cfg
from model_comparison import tukey_groups

FT, PFN, MONROE = cfg.FT_METHOD, cfg.TABPFN_METHOD, cfg.MONROE_METHOD
BASE_FT, BASE_PFN = cfg.BASE_FT_METHOD, cfg.BASE_TABPFN_METHOD
PAIRS = [(FT, BASE_FT), (PFN, BASE_PFN), (FT, MONROE), (FT, PFN),
         (FT, cfg.CHEMELEON_METHOD), (FT, cfg.LGBM_METHOD), (PFN, MONROE),
         (BASE_PFN, MONROE), (BASE_FT, MONROE)]
OUT = cfg.PROJECT_DIR / "results"
# The arms of the authors' Tukey tables, "Atom-JEPA-Bio" on their page
PAPER_ARMS = [FT, PFN, MONROE, BASE_FT, cfg.LGBM_METHOD, cfg.CHEMELEON_METHOD]
PAPER_CLAIM = {"ExpansionRx": {FT: 20, MONROE: 9}, "Biogen ADME": {FT: 14, MONROE: 9}}


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
    """The steps between arms, mean over folds, signed so positive is better.

    bio_stage_*      what the bioassay stage buys: Bio minus stage 1, fine-tuned
                     (ft) and frozen under TabPFN 3 (frozen)
    representation   Monroe + TabPFN 3 -> Bio + TabPFN 3, the head held fixed
    finetune         Bio + TabPFN 3 -> Bio fine-tuned, the encoder held fixed
    net              Monroe + TabPFN 3 -> Bio fine-tuned, the sum of the two
    """
    means = metrics.groupby(["dataset", "endpoint", "method"])[["r2", "mae"]].mean()
    rows = []
    for (ds, endpoint), g in means.groupby(level=[0, 1], sort=False):
        g = g.droplevel([0, 1])
        row = {"dataset": ds, "endpoint": endpoint}
        for m, sign in (("r2", 1), ("mae", -1)):
            row[f"bio_stage_ft_{m}"] = sign * (g.loc[FT, m] - g.loc[BASE_FT, m])
            row[f"bio_stage_frozen_{m}"] = sign * (g.loc[PFN, m] - g.loc[BASE_PFN, m])
            row[f"representation_{m}"] = sign * (g.loc[PFN, m] - g.loc[MONROE, m])
            row[f"finetune_{m}"] = sign * (g.loc[FT, m] - g.loc[PFN, m])
            row[f"net_{m}"] = sign * (g.loc[FT, m] - g.loc[MONROE, m])
        rows.append(row)
    return pd.DataFrame(rows)


def rerun() -> pd.DataFrame:
    """How far a change of machine alone moves stage-1 fine-tuning.

    On every fold the stage-1 recipe was re-run here, compare the re-run with
    the imported atomjepa_ft, and bio_ft with the same imported arm. If the
    first gap is as large as the second, the bioassay stage cannot be told
    apart from the machine.
    """
    m = pd.read_csv(cfg.paths("biogen").fold_metrics)
    re = m[m["method"] == cfg.RERUN_METHOD]
    if re.empty:
        return pd.DataFrame()
    keys = ["endpoint", "repeat", "fold"]
    wide = m[m["method"].isin([cfg.RERUN_METHOD, BASE_FT, FT])].pivot_table(
        index=keys, columns="method", values=cfg.METRICS)
    wide = wide[wide[(cfg.METRICS[0], cfg.RERUN_METHOD)].notna()]
    rows = []
    for metric in cfg.METRICS:
        w = wide[metric]
        for endpoint, g in w.groupby(level=0):
            rows.append({"endpoint": endpoint, "metric": metric, "folds": len(g),
                         "imported": g[BASE_FT].mean(), "rerun": g[cfg.RERUN_METHOD].mean(),
                         "bio": g[FT].mean(),
                         "rerun_minus_imported": (g[cfg.RERUN_METHOD] - g[BASE_FT]).mean(),
                         "abs_rerun_gap": (g[cfg.RERUN_METHOD] - g[BASE_FT]).abs().mean(),
                         "bio_minus_imported": (g[FT] - g[BASE_FT]).mean(),
                         "abs_bio_gap": (g[FT] - g[BASE_FT]).abs().mean()})
    return pd.DataFrame(rows)


def exposure() -> pd.DataFrame:
    """Biogen test molecules PCBA has seen against those it has not, per arm."""
    from scipy.stats import spearmanr as rho
    from sklearn.metrics import mean_absolute_error, r2_score
    ov_path = cfg.paths("biogen").results / "pcba_overlap.csv"
    pq = cfg.paths("biogen").results / "predictions_all.parquet"
    if not ov_path.exists() or not pq.exists():
        return pd.DataFrame()
    seen = set(pd.read_csv(ov_path).query("pcba_exact")[cfg.ID_COL])
    preds = pd.read_parquet(pq)
    preds = preds[preds["method"].isin([FT, BASE_FT, MONROE, PFN, BASE_PFN,
                                        cfg.CHEMELEON_METHOD, cfg.LGBM_METHOD])]
    preds["subset"] = np.where(preds[cfg.ID_COL].isin(seen), "in PCBA", "not in PCBA")
    rows = []
    for (subset, method, endpoint, r, f), g in preds.groupby(
            ["subset", "method", "endpoint", "repeat", "fold"]):
        rows.append({"subset": subset, "method": method, "endpoint": endpoint,
                     "repeat": r, "fold": f, "n": len(g),
                     "r2": r2_score(g["y_true"], g["y_pred"]),
                     "spearman": rho(g["y_true"], g["y_pred"]).statistic,
                     "mae": mean_absolute_error(g["y_true"], g["y_pred"])})
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


def paper_tally(metrics: pd.DataFrame) -> pd.DataFrame:
    """Best alone / tied / worse per arm, Tukey over the authors' six arms only."""
    m = metrics[metrics["method"].isin(PAPER_ARMS)]
    rows = []
    for (ds, endpoint), sub in m.groupby(["dataset", "endpoint"], sort=False):
        for metric in cfg.METRICS:
            g = tukey_groups(sub, metric, cfg.METRIC_HIGHER_IS_BETTER[metric])
            n_top = int(g.isin(("best", "equivalent")).sum())
            for method, grp in g.items():
                top = grp in ("best", "equivalent")
                rows.append({"dataset": ds, "endpoint": endpoint, "metric": metric,
                             "method": method,
                             "kind": ("alone" if n_top == 1 else "tied") if top else "worse"})
    return pd.DataFrame(rows)


def cost() -> pd.DataFrame:
    frames = [pd.read_csv(cfg.paths(n).diagnostics).assign(dataset=cfg.DATASETS[n].label)
              for n in cfg.DATASETS]
    d = pd.concat(frames, ignore_index=True)
    return (d.groupby(["dataset", "method", "host", "gpu"])
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
        print(f"  {d} median R2: bio stage ft {g.bio_stage_ft_r2.median():+.3f}, "
              f"frozen {g.bio_stage_frozen_r2.median():+.3f}; representation "
              f"{g.representation_r2.median():+.3f}, fine-tuning {g.finetune_r2.median():+.3f}, "
              f"net {g.net_r2.median():+.3f}")
    print()

    rr = rerun()
    if not rr.empty:
        rr.to_csv(OUT / "rerun.csv", index=False)
        print("re-run check (Biogen):")
        print(rr.round(4).to_string(index=False), "\n")

    ex = exposure()
    if not ex.empty:
        ex.to_csv(OUT / "exposure.csv", index=False)
        agg = ex.groupby(["subset", "method"])[["r2", "spearman", "mae"]].mean().unstack(0)
        print("Biogen test, in PCBA vs not, mean over endpoints and folds:")
        print(agg.round(3).to_string(), "\n")

    sz, stats = size(lv)
    sz.to_csv(OUT / "size.csv", index=False)
    print(sz.round(3).to_string(index=False))
    print(f"  Spearman n_train vs net R2 {stats['rho_r2']:+.2f} (p {stats['p_r2']:.2f}), "
          f"vs net MAE {stats['rho_mae']:+.2f} (p {stats['p_mae']:.2f})\n")
    pd.Series(stats).to_csv(OUT / "size_stats.csv", header=["value"])

    pt = paper_tally(metrics)
    pt.to_csv(OUT / "paper_tally.csv", index=False)
    top = (pt[pt["kind"] != "worse"].groupby(["dataset", "method"]).size()
           .unstack(0).reindex(PAPER_ARMS).fillna(0).astype(int))
    print("best or tied, Tukey over the authors' six arms (their claim: Bio 20/27 + 14/18, Monroe 9 + 9):")
    print(top.to_string(), "\n")

    c = cost()
    c.to_csv(OUT / "cost.csv", index=False)
    print(c.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
