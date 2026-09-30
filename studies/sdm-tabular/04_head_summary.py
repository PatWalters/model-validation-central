#!/usr/bin/env python
"""Step 4: every head against the reference, both data sets, one table.

Run after 02_collect_metrics.py and 03_report.py for both data sets. For each
head and each endpoint x metric it takes the paired difference from Monroe +
TabPFN 3 over the 25 shared folds, and calls it a win, a loss or no call at
p < 0.05. Those are per-pair tests, uncorrected; the Tukey figures and tables are
where the multiple-comparison-corrected reading lives. This table is the
fold-paired, reference-centred view the page's question needs.

Writes results/head_vs_reference.csv and results/cost.csv, and prints the
tallies the page quotes.

    python 04_head_summary.py
    ADME_SKIP=sdm_tabfm python 04_head_summary.py     # while TabFM is still running
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

import config as cfg

REFERENCE = cfg.MONROE_METHOD
HEADS = [m for m in cfg.METHODS if m not in (REFERENCE, cfg.LGBM_METHOD)]


def paired_rows() -> pd.DataFrame:
    rows = []
    for name in cfg.DATASETS:
        paths = cfg.paths(name)
        metrics = pd.read_csv(paths.fold_metrics)
        for endpoint in paths.dataset.targets:
            sub = metrics[metrics["endpoint"] == endpoint]
            for metric in cfg.METRICS:
                wide = sub.pivot_table(index=["repeat", "fold"], columns="method", values=metric)
                sign = 1 if cfg.METRIC_HIGHER_IS_BETTER[metric] else -1
                for head in HEADS + [cfg.LGBM_METHOD]:
                    pair = wide[[head, REFERENCE]].dropna()
                    if len(pair) != cfg.N_REPEATS * cfg.N_SPLITS:
                        raise SystemExit(f"{name} {endpoint} {head}: {len(pair)} folds, not 25")
                    gain = sign * (pair[head] - pair[REFERENCE])  # positive = head better
                    p = ttest_rel(pair[head], pair[REFERENCE]).pvalue
                    call = "win" if p < 0.05 and gain.mean() > 0 else (
                        "loss" if p < 0.05 else "no call")
                    rows.append({
                        "dataset": paths.dataset.label, "endpoint": endpoint,
                        "metric": metric, "head": head,
                        "head_mean": pair[head].mean(), "ref_mean": pair[REFERENCE].mean(),
                        "gain": gain.mean(), "folds_won": int((gain > 0).sum()),
                        "p": p, "call": call,
                    })
    return pd.DataFrame(rows)


def cost() -> pd.DataFrame:
    frames = []
    for name in cfg.DATASETS:
        path = cfg.paths(name).timing
        if path.exists():
            t = pd.read_csv(path)
            t["dataset"] = cfg.DATASETS[name].label
            frames.append(t)
    t = pd.concat(frames, ignore_index=True)
    t = t[t["method"].isin(HEADS)]
    return (t.groupby("method")
             .agg(folds=("seconds", "size"), total_min=("seconds", lambda s: s.sum() / 60),
                  median_s=("seconds", "median"), max_s=("seconds", "max"))
             .reindex([h for h in HEADS if h in set(t["method"])]))


def main() -> None:
    table = paired_rows()
    out = cfg.PROJECT_DIR / "results" / "head_vs_reference.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out.relative_to(cfg.PROJECT_DIR)}\n")

    pd.set_option("display.width", 200)
    print("calls against Monroe + TabPFN 3, 15 endpoints x 3 metrics = 45 per head")
    tally = pd.crosstab(table["head"], table["call"]).reindex(
        index=HEADS + [cfg.LGBM_METHOD], columns=["win", "no call", "loss"], fill_value=0)
    print(tally.to_string(), "\n")

    print("the same, by data set, MAE only")
    mae = table[table["metric"] == "mae"]
    print(pd.crosstab([mae["head"]], [mae["dataset"], mae["call"]]).to_string(), "\n")

    print("mean gain over the reference (positive = better), by metric")
    print(table.pivot_table(index="head", columns=["dataset", "metric"], values="gain",
                            aggfunc="mean").reindex(HEADS).round(4).to_string(), "\n")

    # The control: one checkpoint, two implementations.
    metrics = pd.concat([pd.read_csv(cfg.paths(n).fold_metrics).assign(dataset=n)
                         for n in cfg.DATASETS])
    rows = []
    for (ds, endpoint), sub in metrics.groupby(["dataset", "endpoint"]):
        for metric in cfg.METRICS:
            wide = sub.pivot_table(index=["repeat", "fold"], columns="method", values=metric)
            a, b = wide[cfg.MONROE_TABICL_METHOD], wide["sdm_tabiclv2"]
            rows.append({"dataset": ds, "endpoint": endpoint, "metric": metric,
                         "diff": (b - a).mean(), "abs_fold_diff": (b - a).abs().mean(),
                         "p": ttest_rel(b, a).pvalue})
    control = pd.DataFrame(rows)
    control.to_csv(cfg.PROJECT_DIR / "results" / "tabicl_control.csv", index=False)
    print("control, sdm TabICLv2 minus tabicl 2.1.1 over the same folds")
    print(f"  p < 0.05 on {int((control.p < 0.05).sum())} of {len(control)} endpoint x metric")
    print(f"  largest mean difference: R2 {control[control.metric=='r2']['diff'].abs().max():.4f}, "
          f"MAE {control[control.metric=='mae']['diff'].abs().max():.4f}")
    print(control[control.p < 0.05].round(4).to_string(index=False), "\n")

    # How much the choice of head moves things, against how much the
    # representation does: spread of mean R2 across the in-context heads per
    # endpoint, beside each head's lead over LightGBM on Morgan counts.
    r2 = table[table["metric"] == "r2"]
    spread = r2.groupby(["dataset", "endpoint"]).apply(
        lambda g: pd.Series({
            "head_spread": max(g[g["head"] != cfg.LGBM_METHOD]["head_mean"].max(), g["ref_mean"].iloc[0])
                           - min(g[g["head"] != cfg.LGBM_METHOD]["head_mean"].min(), g["ref_mean"].iloc[0]),
            "ref_over_lgbm": g["ref_mean"].iloc[0] - g[g["head"] == cfg.LGBM_METHOD]["head_mean"].iloc[0],
        }), include_groups=False)
    spread.to_csv(cfg.PROJECT_DIR / "results" / "head_spread.csv")
    print("R2: range across the in-context heads vs the reference's lead over LightGBM")
    print(spread.round(3).to_string())
    print(f"  median head spread {spread.head_spread.median():.3f}, "
          f"median lead over LightGBM {spread.ref_over_lgbm.median():.3f}\n")

    c = cost()
    c.to_csv(cfg.PROJECT_DIR / "results" / "cost.csv")
    print("GPU seconds, RTX 5070 Ti, both data sets")
    print(c.round(2).to_string())


if __name__ == "__main__":
    main()
