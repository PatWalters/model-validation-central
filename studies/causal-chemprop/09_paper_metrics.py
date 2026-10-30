#!/usr/bin/env python
"""Step 9: the paper's eleven tasks, every arm, scored as the paper scores them.

Reads predictions/paper/<arm>/<task>.csv from steps 1, 5 and 7, forms the
anchor-offset twin of every arm that predicted its anchor, and writes

  results/paper/task_metrics.csv     per (task, arm): n, Pearson r, Spearman ρ,
                                     RMSE, MAE on the non-anchor molecules --
                                     Table 1's metric, plus the error ones
  results/paper/pooled_metrics.csv   per (kinase, arm): the same, pooled over
                                     that kinase's tasks -- what the parity
                                     plots of Figures 2-4 print
  results/paper/head_to_head.csv     Causal-Chemprop against every other arm
                                     over the eleven tasks: wins, mean
                                     difference and a Wilcoxon signed-rank p,
                                     per metric
  results/paper/reproduction.csv     the three released arms against Table 1

With eleven tasks of 7 to 22 molecules each the tests have little power; the
paper's own unit of comparison is the task, so it is the unit here too.

    python 09_paper_metrics.py
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

import config as cfg

PAPER_DIR = cfg.PROJECT_DIR / "predictions" / "paper"
OUT_DIR = cfg.PROJECT_DIR / "results" / "paper"
CAUSAL = "causal_chemprop"
METRICS = {"pearson": True, "spearman": True, "rmse": False, "mae": False}


def load() -> pd.DataFrame:
    frames = []
    for arm_dir in sorted(p for p in PAPER_DIR.iterdir() if p.is_dir()):
        for f in sorted(arm_dir.glob("*.csv")):
            d = pd.read_csv(f)
            d["method"] = arm_dir.name
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    # Offsets, for every arm with a real prediction on its anchor.
    twins = []
    for (method, task), g in df.groupby(["method", "task"]):
        a = g[g["role"] == "anchor"]
        if method.endswith("_anchor") or len(a) != 1 or not np.isfinite(a["y_pred"].iloc[0]):
            continue
        shift = float(a["y_true"].iloc[0] - a["y_pred"].iloc[0])
        t = g[g["role"] == "scored"].copy()
        t["y_pred"] += shift
        t["method"] = method + cfg.OFFSET
        twins.append(t)
    # The constant: every member predicted at its anchor's measured value.
    base = df[(df["method"] == "chemprop")].copy()
    y_anchor = base[base["role"] == "anchor"].set_index("task")["y_true"]
    base["y_pred"] = base["task"].map(y_anchor)
    base["method"] = "anchor_value"
    out = pd.concat([df] + twins + [base], ignore_index=True)
    return out[out["role"] == "scored"]


def score(g: pd.DataFrame) -> dict:
    y, p = g["y_true"].to_numpy(), g["y_pred"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        const = np.std(p) == 0
        return {"n": len(g),
                "pearson": np.nan if const else pearsonr(y, p)[0],
                "spearman": np.nan if const else spearmanr(y, p)[0],
                "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                "mae": float(np.mean(np.abs(y - p)))}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    tasks = pd.DataFrame([{"task": t, "method": m, **score(g)}
                          for (t, m), g in df.groupby(["task", "method"])])
    tasks["kinase"] = tasks["task"].str.split("_").str[0]
    tasks.to_csv(OUT_DIR / "task_metrics.csv", index=False)

    df["kinase"] = df["task"].str.split("_").str[0]
    pooled = pd.DataFrame([{"kinase": k, "method": m, **score(g)}
                           for (k, m), g in df.groupby(["kinase", "method"])])
    pooled.to_csv(OUT_DIR / "pooled_metrics.csv", index=False)

    rows = []
    wide = {m: tasks.pivot(index="task", columns="method", values=m) for m in METRICS}
    for other in sorted(set(tasks["method"]) - {CAUSAL}):
        for metric, higher in METRICS.items():
            w = wide[metric][[CAUSAL, other]].dropna()
            if len(w) < 2:
                continue
            diff = w[CAUSAL] - w[other]
            sign = 1 if higher else -1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = wilcoxon(w[CAUSAL], w[other]).pvalue if (diff != 0).any() else 1.0
            rows.append({"other": other, "metric": metric, "tasks": len(w),
                         "causal_wins": int((sign * diff > 0).sum()),
                         "causal_losses": int((sign * diff < 0).sum()),
                         "mean_diff": float(diff.mean()), "median_diff": float(diff.median()),
                         "wilcoxon_p": float(p)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "head_to_head.csv", index=False)

    rep = []
    for t, vals in cfg.PAPER_TABLE1.items():
        for arm, (r, rs) in zip(cfg.PAPER_RELEASED_ARMS, vals):
            got = tasks[(tasks["task"] == t) & (tasks["method"] == arm)].iloc[0]
            rep.append({"task": t, "method": arm, "paper_r": r, "paper_rs": rs,
                        "ours_r": got["pearson"], "ours_rs": got["spearman"]})
    rep = pd.DataFrame(rep)
    rep["r_match"] = (rep["ours_r"].round(2) == rep["paper_r"])
    rep["rs_match"] = (rep["ours_rs"].round(2) == rep["paper_rs"])
    rep.to_csv(OUT_DIR / "reproduction.csv", index=False)

    n_match = int(rep["r_match"].sum() + rep["rs_match"].sum())
    print(f"Table 1: {n_match} of {2 * len(rep)} values reproduced to two decimals")
    print(rep[~(rep["r_match"] & rep["rs_match"])].to_string(index=False))
    show = tasks.pivot(index="task", columns="method", values="spearman").reindex(cfg.PAPER_TASKS)
    print("\nwithin-task Spearman ρ\n" + show.round(2).to_string())
    print("\n" + pd.DataFrame(rows).query("metric == 'spearman'").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
