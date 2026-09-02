#!/usr/bin/env python
"""Step 6: gather every prediction and reduce it to per-fold statistics.

Reads every tidy prediction file -- the 33 grid configurations and the four
reference methods carried over from ../expansion-ml-comparison -- concatenates
them into results/<dataset>/predictions_all.parquet, and computes R squared,
Spearman rho and MAE for each (endpoint, method, repeat, fold) on the fixed test
set.

Runs against a partial sweep, reporting what is missing rather than failing, so
progress can be checked while steps 3 and 5 are still going.

    python 06_collect_metrics.py
    python 06_collect_metrics.py --no-parquet    # metrics only, skip the big file
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg


def load_predictions(root=None) -> pd.DataFrame:
    root = root or cfg.PRED_DIR
    paths = sorted(root.glob("*/*.csv"))
    if not paths:
        raise SystemExit(f"no prediction files under {root} -- run step 5 first")
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    print(f"read {len(paths):,} prediction files, {len(df):,} predictions")
    return df


def fold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (endpoint, method, repeat, fold), g in df.groupby(
        ["endpoint", "method", "repeat", "fold"], sort=True
    ):
        records.append(
            {
                "endpoint": endpoint,
                "method": method,
                "repeat": repeat,
                "fold": fold,
                "n": len(g),
                "r2": r2_score(g["y_true"], g["y_pred"]),
                "spearman": spearmanr(g["y_true"], g["y_pred"]).statistic,
                "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
            }
        )
    return pd.DataFrame(records)


def report_coverage(metrics: pd.DataFrame) -> None:
    """How much of the grid exists, by configuration rather than by endpoint.

    Thirty-seven methods by nine endpoints is too wide to print as a table, so
    what comes out is the count per method and the endpoints still short.
    """
    expected = cfg.N_REPEATS * cfg.N_SPLITS
    counts = metrics.pivot_table(
        index="method", columns="endpoint", values="r2", aggfunc="count"
    ).reindex(index=cfg.ALL_METHODS, columns=cfg.TARGET_COLS).fillna(0).astype(int)

    total = counts.sum(axis=1)
    want = expected * len(cfg.TARGET_COLS)
    print(f"\nfolds per method (of {want} = {len(cfg.TARGET_COLS)} endpoints x {expected}):")
    for method in cfg.ALL_METHODS:
        got = int(total.get(method, 0))
        short = [e for e in cfg.TARGET_COLS if counts.loc[method, e] != expected] \
            if method in counts.index else cfg.TARGET_COLS
        flag = "" if got == want else f"   short: {', '.join(short[:5])}"
        print(f"  {cfg.METHOD_LABELS[method]:<38} {got:>4}/{want}{flag}")

    missing = int((want * len(cfg.ALL_METHODS)) - total.sum())
    if missing:
        print(f"\n{missing:,} fold predictions still missing")
    else:
        print(f"\ncomplete: {len(metrics):,} rows")

    if metrics[cfg.METRICS].isna().any().any():
        bad = metrics[metrics[cfg.METRICS].isna().any(axis=1)]
        raise SystemExit(f"NaN metric values in {len(bad)} rows -- check {bad.head()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-parquet", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()
    preds = load_predictions()

    unknown = set(preds["method"]) - set(cfg.ALL_METHODS)
    if unknown:
        raise SystemExit(f"unexpected methods in the prediction files: {sorted(unknown)}")

    if not args.no_parquet:
        preds.to_parquet(cfg.PREDICTIONS_PARQUET, index=False)
        print(f"wrote {cfg.PREDICTIONS_PARQUET.name} "
              f"({cfg.PREDICTIONS_PARQUET.stat().st_size / 1e6:.0f} MB)")

    metrics = fold_metrics(preds)
    metrics.to_csv(cfg.FOLD_METRICS_CSV, index=False)
    print(f"wrote {cfg.FOLD_METRICS_CSV.name} ({len(metrics):,} rows)")
    report_coverage(metrics)

    # The two control runs live outside predictions/ so they cannot be swept in
    # as extra methods, but they are scored exactly the same way.
    for directory, name in ((cfg.CONTROL_DIR, "fold_metrics_control.csv"),
                            (cfg.PAPER_GNN_DIR, "fold_metrics_paper_gnn.csv")):
        paths = sorted(directory.glob("*/*.csv"))
        if not paths:
            continue
        frame = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
        scored = fold_metrics(frame)
        scored.to_csv(cfg.RESULTS_DIR / name, index=False)
        print(f"wrote {name} ({len(scored):,} rows, "
              f"{scored['method'].nunique()} configurations)")

    best = (
        metrics[metrics["method"].isin(cfg.GRID_METHODS)]
        .groupby("method")["r2"].mean().sort_values(ascending=False)
    )
    print("\nmean R squared over every endpoint and fold, best ten configurations:")
    for method, value in best.head(10).items():
        print(f"  {cfg.METHOD_LABELS[method]:<40} {value:6.3f}")
    for method in cfg.REFERENCE_METHODS:
        sub = metrics[metrics["method"] == method]
        if len(sub):
            print(f"  {cfg.METHOD_LABELS[method]:<40} {sub['r2'].mean():6.3f}  (reference)")


if __name__ == "__main__":
    main()
