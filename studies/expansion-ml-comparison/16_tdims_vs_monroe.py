#!/usr/bin/env python
"""Step 16: TDiMS against Monroe, both data sets, one table.

Run after 04_collect_metrics.py has been run for both data sets.
Uses the folds as the pairing, which is what makes the test legitimate: both
arms saw identical training molecules in every one of the 25 replicates.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

LEFT, RIGHT, BASE = cfg.TDIMS_METHOD, cfg.MONROE35_METHOD, cfg.LGBM_METHOD
rows = []
for name in ("expansion", "biogen"):
    paths = cfg.paths(name)
    metrics = pd.read_csv(paths.fold_metrics)
    chosen = {}
    config_csv = paths.results / "tdims_config.csv"
    if config_csv.exists():
        scores = pd.read_csv(config_csv)
        means = scores.groupby(["endpoint", "config"])["val_r2"].mean()
        chosen = {e: means[e].idxmax() for e in means.index.get_level_values(0).unique()}

    for endpoint in paths.dataset.targets:
        sub = metrics[metrics["endpoint"] == endpoint]
        wide = sub.pivot_table(index=["repeat", "fold"], columns="method", values="r2")
        if not {LEFT, RIGHT}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=[LEFT, RIGHT])
        diff = wide[LEFT] - wide[RIGHT]
        stat = ttest_rel(wide[LEFT], wide[RIGHT])
        mae = sub.pivot_table(index=["repeat", "fold"], columns="method", values="mae")
        rows.append({
            "dataset": paths.dataset.label,
            "endpoint": endpoint,
            "n_folds": len(wide),
            "config": chosen.get(endpoint, ""),
            "tdims_r2": wide[LEFT].mean(),
            "monroe_r2": wide[RIGHT].mean(),
            "lgbm_r2": wide[BASE].mean() if BASE in wide else np.nan,
            "delta_r2": diff.mean(),
            "win_rate": float((diff > 0).mean()),
            "p": stat.pvalue,
            "tdims_mae": mae[LEFT].mean(),
            "monroe_mae": mae[RIGHT].mean(),
        })

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n--- how often each arm is ahead on mean R2 ---")
for dataset, g in out.groupby("dataset", sort=False):
    n = len(g)
    print(f"{dataset:<14} TDiMS ahead on {int((g.delta_r2 > 0).sum())}/{n} endpoints; "
          f"significant at 0.05 in favour of TDiMS on "
          f"{int(((g.delta_r2 > 0) & (g.p < 0.05)).sum())}, of Monroe on "
          f"{int(((g.delta_r2 < 0) & (g.p < 0.05)).sum())}")
print(f"\nmean R2 gap (TDiMS - Monroe) over all {len(out)} endpoints: "
      f"{out.delta_r2.mean():+.3f}   median {out.delta_r2.median():+.3f}")
print(f"TDiMS vs LightGBM+Morgan: TDiMS ahead on "
      f"{int((out.tdims_r2 > out.lgbm_r2).sum())}/{len(out)}")
out.to_csv(cfg.PROJECT_DIR / "results" / "tdims_vs_monroe35.csv", index=False)
print("\nwrote results/tdims_vs_monroe35.csv")
