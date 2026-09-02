#!/usr/bin/env python
"""Step 7: epistemic uncertainty and calibration, for every configuration.

The five folds of a repeat form one ensemble: five models fit on overlapping
four-fifths of the same training molecules, every one of them predicting the same
fixed test set. Their spread is the epistemic sigma, and how well that spread
tracks the error it is supposed to anticipate is what the paper's clearest result
is about -- fusion barely moves accuracy but does move calibration.

Five repeats give five values per configuration and endpoint, which is what the
report's statistics are run over. The four reference methods are scored the same
way, from predictions that already exist, so the fusion grid can be read against
them on calibration as well as on accuracy.

    python 07_uncertainty.py
    ADME_DATASET=biogen python 07_uncertainty.py
"""

import argparse

import numpy as np
import pandas as pd

import config as cfg
from uncertainty import reliability_curve, repeat_metrics


def load_predictions(path=None) -> pd.DataFrame:
    path = path or cfg.PREDICTIONS_PARQUET
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 06_collect_metrics.py first")
    return pd.read_parquet(path)


def collect(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (endpoint, method, repeat), g in preds.groupby(
        ["endpoint", "method", "repeat"], sort=True
    ):
        n_folds = g["fold"].nunique()
        if n_folds < 2:
            continue                       # a spread of one model is not a spread
        rows.append({
            "endpoint": endpoint,
            "method": method,
            "repeat": repeat,
            "n_folds": n_folds,
            "n": g[cfg.ID_COL].nunique(),
            **repeat_metrics(g),
        })
    return pd.DataFrame(rows)


def curves(preds: pd.DataFrame) -> pd.DataFrame:
    """Reliability curves, pooling every repeat, for the figure in step 10."""
    rows = []
    for (endpoint, method), g in preds.groupby(["endpoint", "method"], sort=True):
        curve = reliability_curve(g)
        curve["endpoint"] = endpoint
        curve["method"] = method
        rows.append(curve)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default=None)
    args = parser.parse_args()

    cfg.ensure_dirs()
    preds = load_predictions(args.predictions)

    table = collect(preds)
    table.to_csv(cfg.UNCERTAINTY_CSV, index=False)
    print(f"wrote {cfg.UNCERTAINTY_CSV.name} ({len(table):,} rows, "
          f"{table['method'].nunique()} methods x {table['endpoint'].nunique()} "
          f"endpoints x {cfg.N_REPEATS} repeats)")

    curve_path = cfg.TABLE_DIR / "reliability_curves.csv"
    curves(preds).to_csv(curve_path, index=False)
    print(f"wrote tables/{curve_path.name}")

    print("\nmean over every endpoint and repeat:")
    header = f"{'configuration':<40}" + "".join(
        f"{m:>22}" for m in cfg.UNCERTAINTY_METRICS
    )
    print(header)
    summary = table.groupby("method")[cfg.UNCERTAINTY_METRICS].mean()
    order = [m for m in cfg.ALL_METHODS if m in summary.index]
    for method in sorted(order, key=lambda m: -summary.loc[m, "err_unc_corr"]):
        cells = "".join(f"{summary.loc[method, m]:>22.4f}" for m in cfg.UNCERTAINTY_METRICS)
        tag = "  (reference)" if method in cfg.REFERENCE_METHODS else ""
        print(f"{cfg.METHOD_LABELS[method]:<40}{cells}{tag}")

    if table[cfg.UNCERTAINTY_METRICS].isna().any().any():
        n_bad = int(table[cfg.UNCERTAINTY_METRICS].isna().any(axis=1).sum())
        print(f"\n{n_bad} rows carry a NaN, which happens when sigma is constant "
              "across a test set and the correlation is undefined")


if __name__ == "__main__":
    main()
