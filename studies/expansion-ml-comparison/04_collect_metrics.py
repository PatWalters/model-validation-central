#!/usr/bin/env python
"""Step 4: gather every prediction and reduce it to per-fold statistics.

Reads every tidy prediction file written by steps 2 and 3, concatenates them into
results/predictions_all.parquet (every individual prediction is kept), and
computes R^2, Spearman rho and MAE for each (endpoint, method, repeat, fold) on
the fixed test set -> results/fold_metrics.csv, 9 x 3 x 25 = 675 rows when the
sweep is complete.

Runs against a partial sweep as well; it reports what is missing rather than
failing, so progress can be checked while step 3 is still going.

    python 04_collect_metrics.py
    python 04_collect_metrics.py --no-parquet   # metrics only, skip the big file
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg


def load_predictions() -> pd.DataFrame:
    paths = sorted(cfg.PRED_DIR.glob("*/*.csv"))
    if not paths:
        raise SystemExit(f"no prediction files under {cfg.PRED_DIR} -- run steps 2 and 3 first")
    df = pd.concat((pd.read_csv(p) for p in paths), ignore_index=True)
    print(f"read {len(paths)} prediction files, {len(df):,} predictions")
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
    expected = cfg.N_REPEATS * cfg.N_SPLITS
    counts = metrics.pivot_table(
        index="endpoint", columns="method", values="r2", aggfunc="count"
    ).reindex(index=cfg.TARGET_COLS, columns=cfg.ALL_METHODS)
    print(f"\nfolds per endpoint and method (of {expected}):")
    print(counts.fillna(0).astype(int).to_string())

    incomplete = int((counts.fillna(0) != expected).sum().sum())
    if incomplete:
        print(f"\n{incomplete} endpoint/method combinations are still incomplete")
    else:
        print(f"\ncomplete: {len(metrics)} rows ({len(cfg.TARGET_COLS)} x {len(cfg.ALL_METHODS)} x {expected})")

    if metrics[cfg.METRICS].isna().any().any():
        raise SystemExit("NaN metric values -- check the prediction files")


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric over the folds."""
    summary = (
        metrics.groupby(["endpoint", "method"])[cfg.METRICS]
        .agg(["mean", "std"])
        .reindex(pd.MultiIndex.from_product([cfg.TARGET_COLS, cfg.ALL_METHODS], names=["endpoint", "method"]))
        .dropna(how="all")
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-parquet", action="store_true", help="skip writing predictions_all.parquet")
    args = parser.parse_args()

    cfg.ensure_dirs()
    preds = load_predictions()

    unknown = set(preds["method"]) - set(cfg.ALL_METHODS)
    if unknown:
        raise SystemExit(f"unexpected methods in the prediction files: {sorted(unknown)}")

    if not args.no_parquet:
        preds.to_parquet(cfg.PREDICTIONS_PARQUET, index=False)
        size_mb = cfg.PREDICTIONS_PARQUET.stat().st_size / 1e6
        print(f"wrote {cfg.PREDICTIONS_PARQUET.name} ({size_mb:.0f} MB)")

    metrics = fold_metrics(preds)
    metrics.to_csv(cfg.FOLD_METRICS_CSV, index=False)
    print(f"wrote {cfg.FOLD_METRICS_CSV.name} ({len(metrics)} rows)")

    report_coverage(metrics)

    summary = summarise(metrics)
    summary_path = cfg.TABLE_DIR / "summary_raw.csv"
    summary.to_csv(summary_path, index=False)

    print("\nmean +/- sd over folds:")
    for metric in cfg.METRICS:
        wide = summary.pivot(index="endpoint", columns="method", values=f"{metric}_mean")
        sd = summary.pivot(index="endpoint", columns="method", values=f"{metric}_std")
        shown = wide.reindex(index=cfg.TARGET_COLS, columns=cfg.ALL_METHODS).copy()
        for col in shown.columns:
            shown[col] = [
                "" if np.isnan(m) else f"{m:6.3f} +/- {s:.3f}"
                for m, s in zip(wide.get(col, pd.Series(dtype=float)).reindex(cfg.TARGET_COLS),
                                sd.get(col, pd.Series(dtype=float)).reindex(cfg.TARGET_COLS))
            ]
        print(f"\n{cfg.METRIC_LABELS[metric]}")
        print(shown.to_string())


if __name__ == "__main__":
    main()
