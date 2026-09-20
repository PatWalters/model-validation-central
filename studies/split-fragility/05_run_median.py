#!/usr/bin/env python
"""Step 5: the naive baseline -- the training median, for every test molecule.

Not a method, and on the page it is drawn as a line rather than a bar. The
paper's claim is that every model's accuracy converges towards this baseline as
the split gets more stringent, so the baseline has to be on the same axis, over
the same folds, scored by the same code as everything else. Putting it through
the same prediction files as the four fitted arms is what makes that possible.

The median of the fold's *training* labels, which is what the paper uses, and
which drifts between schemes because the schemes select structurally different
training sets -- an effect the paper notes and the report reproduces.

R^2 for a constant prediction is not quite zero, because it is measured against
the *test* mean and the constant comes from the training set. Spearman rho is
undefined for a constant vector; 07_collect_metrics.py records it as NaN for
this arm and the rank-based figures leave it out rather than plotting a zero.

    python 05_run_median.py
"""

import argparse
import time

import numpy as np

import config as cfg
from fold_data import FoldData, add_fold_arguments


def run_fold(data: FoldData, target: int, split: str, fold: int, force: bool) -> None:
    out_path = cfg.pred_csv(cfg.MEDIAN_METHOD, target, split, fold)
    if out_path.exists() and not force:
        return

    rows = data.fold(target, split, fold)
    y = data.y
    constant = float(np.median(y[rows.train]))

    data.write_predictions(
        cfg.MEDIAN_METHOD, target, split, fold, rows.test,
        np.full(len(rows.test), constant), extra={"train_median": constant},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_fold_arguments(parser)
    args = parser.parse_args()

    cfg.ensure_dirs()
    (cfg.PRED_DIR / cfg.MEDIAN_METHOD).mkdir(parents=True, exist_ok=True)
    data = FoldData.load()

    for target in args.target:
        start = time.time()
        for split in args.split:
            for fold in args.fold:
                run_fold(data, target, split, fold, args.force)
        n = len(list((cfg.PRED_DIR / cfg.MEDIAN_METHOD).glob(f"tid_{target}_*_f*.csv")))
        print(f"tid_{target:<6} {n:>3}/30 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
