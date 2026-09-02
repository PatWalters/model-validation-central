#!/usr/bin/env python
"""Step 2: the LightGBM baseline -- 9 endpoints x 25 folds = 225 single-task models.

Morgan count fingerprints (radius 2, 2048 bits) are computed once for the whole
data set and cached, then each fold model is fit on the molecules the chemprop
models of that fold train on -- the four fifths of `ds == 'train'` that have a
value for the endpoint -- and used to predict the fixed `ds == 'test'` molecules.

The held-out fifth is not used: LightGBM needs no early-stopping set. What matters
for the comparison is that the *training* molecules are identical to chemprop's,
which they are.

Predictions go to predictions/lgbm/<endpoint>_r{r}_f{f}.csv in the tidy schema
shared by every method. Existing files are skipped, so the script is resumable.

    python 02_run_lightgbm.py                    # everything missing
    python 02_run_lightgbm.py --endpoint LogD    # one endpoint
    python 02_run_lightgbm.py --force            # refit everything
"""

import argparse
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

import config as cfg

RDLogger.DisableLog("rdApp.*")


def morgan_counts(df: pd.DataFrame, force: bool = False) -> np.ndarray:
    """Morgan count fingerprints for every molecule, cached to disk.

    Row order matches `df`, which is the row order of data/master.csv.
    """
    if cfg.FINGERPRINT_NPY.exists() and not force:
        fps = np.load(cfg.FINGERPRINT_NPY)
        if len(fps) == len(df):
            print(f"loaded cached fingerprints {fps.shape} from {cfg.FINGERPRINT_NPY.name}")
            return fps
        print("cached fingerprints do not match the data set, recomputing")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=cfg.FP_RADIUS, fpSize=cfg.FP_SIZE)
    start = time.time()
    fps = np.array(
        [gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(smi)) for smi in df[cfg.SMILES_COL]],
        dtype=np.float32,
    )
    np.save(cfg.FINGERPRINT_NPY, fps)
    print(f"computed fingerprints {fps.shape} in {time.time() - start:.1f}s -> {cfg.FINGERPRINT_NPY.name}")
    return fps


def run_fold(
    df: pd.DataFrame,
    fps: np.ndarray,
    folds: pd.DataFrame,
    endpoint: str,
    repeat: int,
    fold: int,
    force: bool,
) -> None:
    out_path = cfg.pred_csv(cfg.LGBM_METHOD, endpoint, repeat, fold)
    if out_path.exists() and not force:
        return

    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for the test molecules

    measured = df[endpoint].notna().to_numpy()
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    # The fit set: training molecules outside the held-out fold with a measured value.
    fit_mask = measured & ~is_test & (fold_of != fold)
    test_mask = measured & is_test

    y = df[endpoint].to_numpy()
    model = LGBMRegressor(n_jobs=-1, verbose=-1, random_state=cfg.fold_seed(repeat, fold))
    model.fit(fps[fit_mask], y[fit_mask])
    pred = model.predict(fps[test_mask])

    test_df = df.loc[test_mask]
    pd.DataFrame(
        {
            "method": cfg.LGBM_METHOD,
            "endpoint": endpoint,
            "repeat": repeat,
            "fold": fold,
            cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(),
            cfg.SMILES_COL: test_df[cfg.SMILES_COL].to_numpy(),
            "y_true": y[test_mask],
            "y_pred": pred,
        },
        columns=cfg.PRED_COLUMNS,
    ).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true", help="refit models that already have predictions")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    cfg.ensure_dirs()
    (cfg.PRED_DIR / cfg.LGBM_METHOD).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    fps = morgan_counts(df)

    for endpoint in args.endpoint:
        start = time.time()
        for repeat in args.repeat:
            for fold in args.fold:
                run_fold(df, fps, folds, endpoint, repeat, fold, args.force)
        n = len(list((cfg.PRED_DIR / cfg.LGBM_METHOD).glob(f"{endpoint}_r*_f*.csv")))
        print(f"{endpoint:<17} {n:>2}/25 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
