#!/usr/bin/env python
"""Step 6: all four KERMT arms against every arm, both data sets.

Run after 04_collect_metrics.py and 05_report.py for both data sets. Writes,
under results/:

  calls.csv    for each pair and endpoint x metric, the paired difference over
               the 25 shared folds, signed so positive is the KERMT arm
               better, called a win, a loss or no call at p < 0.05
  ranks.csv    per data set, endpoint and metric, each arm's rank by fold mean
               (1 = best), so the field can be read in one line per arm
  cost.csv     wall clock per fine-tuned fold model, by arm, data set and machine

    python 06_summary.py
"""

import pandas as pd
from scipy.stats import ttest_rel

import config as cfg

K, GB, GL, KT = (cfg.KERMT_METHOD, cfg.GROVER_BASE_METHOD, cfg.GROVER_LARGE_METHOD,
                 cfg.KERMT_TABPFN_METHOD)
COMPARATORS = [cfg.ATOMJEPA_FT_METHOD, cfg.MONROE_METHOD, cfg.CHEMELEON_METHOD, cfg.LGBM_METHOD]
# every KERMT arm against the field; then v2 against the GROVER checkpoints it
# descends from (positive: v2's pre-training helps), GROVER large against base,
# and fine-tuned against frozen (positive: fine-tuning helps)
PAIRS = ([(arm, o) for arm in (K, GL, GB, KT) for o in COMPARATORS]
         + [(K, GB), (K, GL), (GL, GB), (K, KT)])
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


def ranks(metrics: pd.DataFrame) -> pd.DataFrame:
    means = (metrics[metrics["method"].isin(cfg.METHODS)]
             .groupby(["dataset", "endpoint", "method"])[cfg.METRICS].mean().reset_index())
    rows = []
    for metric in cfg.METRICS:
        r = means.groupby(["dataset", "endpoint"])[metric].rank(
            ascending=not cfg.METRIC_HIGHER_IS_BETTER[metric], method="min")
        rows.append(means[["dataset", "endpoint", "method"]].assign(metric=metric, rank=r))
    return pd.concat(rows, ignore_index=True)


def cost() -> pd.DataFrame:
    frames = [pd.read_csv(cfg.paths(n).diagnostics).assign(dataset=cfg.DATASETS[n].label)
              for n in cfg.DATASETS if cfg.paths(n).diagnostics.exists()]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["host"] = d["host"].replace({"spark-4950": "spark"})
    return (d.groupby(["method", "dataset", "host", "gpu"])
            .agg(folds=("seconds", "size"), median_sec=("seconds", "median"),
                 total_h=("seconds", lambda s: s.sum() / 3600),
                 median_best_epoch=("best_epoch", "median"),
                 median_epochs=("epochs", "median"),
                 max_epochs=("epochs", "max"))
            .reset_index())


def main() -> None:
    pd.set_option("display.width", 200)
    metrics = load_metrics()

    table = calls(metrics)
    table.to_csv(OUT / "calls.csv", index=False)
    print("endpoints x 3 metrics per pair and data set, paired t over 25 folds")
    print(pd.crosstab([table["focus"], table["other"]], [table["dataset"], table["call"]])
          .to_string(), "\n")

    rk = ranks(metrics)
    rk.to_csv(OUT / "ranks.csv", index=False)
    print("mean rank by fold-mean metric, 1 = best:")
    print(rk.pivot_table(index="method", columns="dataset", values="rank", aggfunc="mean")
          .reindex(cfg.METHODS).round(2).to_string(), "\n")

    c = cost()
    if not c.empty:
        c.to_csv(OUT / "cost.csv", index=False)
        print(c.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
