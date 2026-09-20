#!/usr/bin/env python
"""Step 1: the k-NN arm -- 10 targets x 6 schemes x 5 folds = 300 fold models.

The paper's own baseline, re-implemented from its released code
(src/models/run_baselines.py in the upstream repository): a
`KNeighborsRegressor` with `weights="distance"` on a precomputed Tanimoto
distance matrix over ECFP4 bit vectors, with k chosen from {1, 3, 5} by MAE on
the fold's validation molecules. That is the one hyperparameter any arm in this
study tunes per fold, and it is tuned the way the paper tuned it.

Because the paper also releases its own per-fold k-NN metrics, this arm is the
one place where the study can check itself: 06_import_published.py brings those
numbers in as `pub_knn`, and 07_collect_metrics.py prints the two side by side.
A reimplementation that agrees with the source on 300 folds is evidence the rest
of the harness -- the folds, the masks, the metric -- is wired up correctly.

Predictions go to predictions/knn/tid_<target>_<split>_f<fold>.csv in the tidy
schema shared by every method. Existing files are skipped, so this is resumable.

    python 01_run_knn.py
    python 01_run_knn.py --target 284 --split umap      # one target, one scheme
    python 01_run_knn.py --force                       # refit everything
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

import config as cfg
from fold_data import FoldData, add_fold_arguments, selected_folds


def tanimoto_distance(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """1 - Tanimoto similarity between every row of A and every row of B.

    The upstream implementation, written out the same way: the dot product of two
    binary matrices counts shared bits, and the union follows from the two
    popcounts. A molecule with an all-zero fingerprint has an undefined
    similarity to everything, and is scored 0 similarity rather than dropped --
    the same choice the source makes.
    """
    A = A.astype(bool).astype(np.float32)
    B = B.astype(bool).astype(np.float32)
    shared = A @ B.T
    union = A.sum(1, keepdims=True) + B.sum(1, keepdims=True).T - shared
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = np.where(union > 0, shared / union, 0.0)
    return (1.0 - similarity).astype(np.float64)


def run_fold(data: FoldData, target: int, split: str, fold: int, force: bool) -> None:
    out_path = cfg.pred_csv(cfg.KNN_METHOD, target, split, fold)
    if out_path.exists() and not force:
        return

    rows = data.fold(target, split, fold)
    X = data.ecfp_bits
    fit, val, test = rows.train, rows.val, rows.test

    D_fit = tanimoto_distance(X[fit], X[fit])
    D_val = tanimoto_distance(X[val], X[fit])
    D_test = tanimoto_distance(X[test], X[fit])

    y = data.y
    # k cannot exceed the number of training molecules. The diverse scheme trains
    # on as few as ~180, so this never bites in practice, but a silent sklearn
    # error on a small fold would be worse than a clipped grid.
    grid = [k for k in cfg.KNN_K_CHOICES if k <= len(fit)]
    if not grid:
        raise SystemExit(f"tid_{target} {split} f{fold}: only {len(fit)} training molecules")

    scored = {}
    for k in grid:
        model = KNeighborsRegressor(n_neighbors=k, metric="precomputed", weights="distance")
        model.fit(D_fit, y[fit])
        scored[k] = (mean_absolute_error(y[val], model.predict(D_val)), model)

    best_k = min(scored, key=lambda k: scored[k][0])
    pred = scored[best_k][1].predict(D_test)

    data.write_predictions(cfg.KNN_METHOD, target, split, fold, test, pred,
                           extra={"best_k": best_k})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_fold_arguments(parser)
    args = parser.parse_args()

    cfg.ensure_dirs()
    (cfg.PRED_DIR / cfg.KNN_METHOD).mkdir(parents=True, exist_ok=True)
    data = FoldData.load(need="ecfp")

    for target in args.target:
        start = time.time()
        for split in args.split:
            for fold in args.fold:
                run_fold(data, target, split, fold, args.force)
        n = len(list((cfg.PRED_DIR / cfg.KNN_METHOD).glob(f"tid_{target}_*_f*.csv")))
        print(f"tid_{target:<6} {n:>3}/30 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
