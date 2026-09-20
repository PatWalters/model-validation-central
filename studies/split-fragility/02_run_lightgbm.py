#!/usr/bin/env python
"""Step 2: the LightGBM baseline -- 300 single-task models.

Morgan count fingerprints, radius 2 and 2048 bits, computed once for the whole
collection by 00_prepare_data.py, with LightGBM's library defaults on top. This
is the same arm `studies/expansion-ml-comparison` calls `lgbm`, on this study's
folds, so the two are the same model fit on different molecules.

The validation molecules are unused. LightGBM needs no early-stopping set here
and nothing in the arm is tuned per fold, so it fits on the `train` indices and
nothing else -- the same molecules k-NN and Monroe fit on, and the same ones
ChemProp trains on before it consults its validation fifth.

    python 02_run_lightgbm.py
    python 02_run_lightgbm.py --target 284 --split umap
    python 02_run_lightgbm.py --force
"""

import argparse
import time

from lightgbm import LGBMRegressor

import config as cfg
from fold_data import FoldData, add_fold_arguments


def run_fold(data: FoldData, target: int, split: str, fold: int, force: bool) -> None:
    out_path = cfg.pred_csv(cfg.LGBM_METHOD, target, split, fold)
    if out_path.exists() and not force:
        return

    rows = data.fold(target, split, fold)
    X, y = data.morgan_counts, data.y

    model = LGBMRegressor(
        n_jobs=-1, verbose=-1, random_state=cfg.fold_seed(target, split, fold)
    )
    model.fit(X[rows.train], y[rows.train])
    pred = model.predict(X[rows.test])

    data.write_predictions(cfg.LGBM_METHOD, target, split, fold, rows.test, pred,
                           extra={"n_train": len(rows.train)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_fold_arguments(parser)
    args = parser.parse_args()

    cfg.ensure_dirs()
    (cfg.PRED_DIR / cfg.LGBM_METHOD).mkdir(parents=True, exist_ok=True)
    data = FoldData.load(need="morgan")

    for target in args.target:
        start = time.time()
        for split in args.split:
            for fold in args.fold:
                run_fold(data, target, split, fold, args.force)
        n = len(list((cfg.PRED_DIR / cfg.LGBM_METHOD).glob(f"tid_{target}_*_f*.csv")))
        print(f"tid_{target:<6} {n:>3}/30 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
