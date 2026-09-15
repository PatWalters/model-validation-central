#!/usr/bin/env python
"""Step 9b: what the TabPFN 3.5 arm gave up by running at Monroe's temperature.

From `tabpfn` 9.0.0 a checkpoint can declare the softmax temperature it was
trained for, and the TabPFN 3.5 regression checkpoint declares 1.0. Monroe's
`fit_predict_tabpfn` passes 0.9 to every model, which is the value TabPFN applied
before checkpoints could ask for one. Both Monroe arms here run at that 0.9, so
that they differ in the checkpoint and nothing else -- but that leaves the newer
arm a shade off its own default, and the difference should be measured rather
than waved at.

So `09_run_monroe.py --head v3.5 --softmax-temperature auto` runs the same 225
and 150 folds at the temperature the checkpoint asks for, into
`results/<dataset>/sensitivity/monroe35_tauto/`. This reduces that to one table:
per endpoint and metric, the arm's mean over the 25 folds, the control's, and a
paired test over the folds they share.

The control is not an arm. It never enters the figures, its per-fold files are
not tracked, and this table beside them is what the report reads.

    python 09b_monroe_temperature.py
    ADME_DATASET=biogen python 09b_monroe_temperature.py

Writes results/<dataset>/sensitivity/monroe35_temperature.csv.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg

ARM = cfg.MONROE35_METHOD
CONTROL_DIR = cfg.SENSITIVITY_DIR / f"{ARM}_tauto"


def fold_metrics(paths, label: str) -> pd.DataFrame:
    """R2, Spearman rho and MAE for every (endpoint, repeat, fold) in `paths`."""
    if not paths:
        raise SystemExit(
            f"no prediction files for {label} -- run 09_run_monroe.py for it first"
        )
    preds = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    records = []
    for (endpoint, repeat, fold), g in preds.groupby(
        ["endpoint", "repeat", "fold"], sort=True
    ):
        records.append(
            {
                "endpoint": endpoint,
                "fold_id": f"{repeat}_{fold}",
                "r2": r2_score(g["y_true"], g["y_pred"]),
                "spearman": spearmanr(g["y_true"], g["y_pred"]).statistic,
                "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    arm = fold_metrics(sorted((cfg.PRED_DIR / ARM).glob("*.csv")), f"the {ARM} arm")
    control = fold_metrics(sorted(CONTROL_DIR.glob("*.csv")), "the temperature control")

    # The two runs cover the same folds by construction. Say so rather than
    # quietly averaging over whatever both happen to have.
    shared = arm.merge(control, on=["endpoint", "fold_id"], suffixes=("_arm", "_ctl"))
    if len(shared) != len(arm) or len(shared) != len(control):
        raise SystemExit(
            f"the arm has {len(arm)} folds and the control {len(control)}, sharing "
            f"{len(shared)} -- the comparison would not be paired"
        )

    rows = []
    for endpoint, g in shared.groupby("endpoint", sort=False):
        for metric in cfg.METRICS:
            at_09 = g[f"{metric}_arm"].to_numpy()
            at_auto = g[f"{metric}_ctl"].to_numpy()
            better = cfg.METRIC_HIGHER_IS_BETTER[metric]
            delta = float(np.mean(at_auto - at_09))
            rows.append(
                {
                    "endpoint": endpoint,
                    "metric": metric,
                    "n_folds": len(g),
                    "t0_9": float(at_09.mean()),
                    "t_auto": float(at_auto.mean()),
                    "delta": delta,
                    "auto_better": bool(delta > 0) == better,
                    "p_paired": float(ttest_rel(at_auto, at_09).pvalue),
                }
            )

    table = pd.DataFrame(rows).reindex(
        columns=["endpoint", "metric", "n_folds", "t0_9", "t_auto", "delta",
                 "auto_better", "p_paired"]
    )
    CONTROL_DIR.parent.mkdir(parents=True, exist_ok=True)
    out = cfg.SENSITIVITY_DIR / f"{ARM}_temperature.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out} ({len(table)} rows)")

    moved = table[table["p_paired"] < 0.05]
    print(f"\n{len(moved)} of {len(table)} endpoint/metric combinations move at p < 0.05, "
          f"{int(moved['auto_better'].sum())} of them in the checkpoint's favour")
    print(table.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))


if __name__ == "__main__":
    main()
