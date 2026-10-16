#!/usr/bin/env python
"""Step 2: MolPAIR over every endpoint x repeat x fold.

Each fold is one call to `molpair.MolPAIR.fit_predict`, on the same fit and test
masks every other arm in the project uses: the four fifths of `ds == 'train'`
outside the held-out fold that have a value for the endpoint, scored on every
`ds == 'test'` molecule that has one. The held-out fifth is unused, as it is for
LightGBM and Monroe -- MolPAIR has no training loop to stop, and the folds it
needs for its out-of-fold errors it makes for itself, inside the four fifths.

One call writes two arms, because MolPAIR computes both:

  predictions/<ds>/molpair_global/   mu_q, the CheMeleon + TabPFN 3 prediction
  predictions/<ds>/molpair/          mu_q plus the gated pair correction

and one diagnostics file per fold, under diagnostics/<ds>/, with each test
molecule's anchor count, nearest-anchor similarity, gate and correction.

    python 02_run_molpair.py
    python 02_run_molpair.py --endpoint LOG_HLM --repeat 0 --fold 0   # smoke test
    ADME_DATASET=biogen python 02_run_molpair.py

Runs in an environment where `tabpfn` loads TabPFN 3 (tabpfn 8.x, `monroe` on
apollo) with TABPFN_TOKEN set. Features come from 01_featurize.py.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

import config as cfg
from molpair import MolPAIR, Molecules


def check_tabpfn() -> None:
    """Refuse to run unless the installed tabpfn loads TabPFN 3, as the paper's did."""
    import tabpfn
    from tabpfn.settings import settings

    version = settings.tabpfn.model_version.value
    if version != "v3":
        raise SystemExit(f"tabpfn {tabpfn.__version__} loads TabPFN {version}, not v3")
    print(f"tabpfn {tabpfn.__version__} loads TabPFN {version}", flush=True)


def write_preds(method, endpoint, repeat, fold, test_df, y_true, y_pred) -> None:
    out = cfg.pred_csv(method, endpoint, repeat, fold)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "method": method,
            "endpoint": endpoint,
            "repeat": repeat,
            "fold": fold,
            cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(),
            cfg.SMILES_COL: test_df[cfg.SMILES_COL].to_numpy(),
            "y_true": y_true,
            "y_pred": np.asarray(y_pred, dtype=float),
        },
        columns=cfg.PRED_COLUMNS,
    ).to_csv(out, index=False)


def run_fold(df, mols, folds, endpoint, repeat, fold, force, timings) -> None:
    done = [cfg.pred_csv(m, endpoint, repeat, fold) for m in (cfg.GLOBAL_METHOD, cfg.MOLPAIR_METHOD)]
    if all(p.exists() for p in done) and cfg.diag_csv(endpoint, repeat, fold).exists() and not force:
        return

    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for the test molecules

    measured = df[endpoint].notna().to_numpy()
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    # The same two masks 02_run_lightgbm.py and 09_run_monroe.py use upstream.
    fit_mask = measured & ~is_test & (fold_of != fold)
    test_mask = measured & is_test
    train, query = np.flatnonzero(fit_mask), np.flatnonzero(test_mask)

    y = df[endpoint].to_numpy()
    start = time.time()
    model = MolPAIR(mols, cfg.MOLPAIR, cfg.fold_seed(repeat, fold))
    mu, corrected, diag = model.fit_predict(train, y[train], query)
    seconds = time.time() - start
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(corrected))):
        raise SystemExit(f"{endpoint} r{repeat} f{fold}: non-finite predictions")

    test_df = df.iloc[query]
    write_preds(cfg.GLOBAL_METHOD, endpoint, repeat, fold, test_df, y[query], mu)
    write_preds(cfg.MOLPAIR_METHOD, endpoint, repeat, fold, test_df, y[query], corrected)
    diag.insert(0, cfg.ID_COL, test_df[cfg.ID_COL].to_numpy())
    diag.insert(1, "y_true", y[query])
    diag.to_csv(cfg.diag_csv(endpoint, repeat, fold), index=False)
    timings.append({"method": cfg.MOLPAIR_METHOD, "endpoint": endpoint, "repeat": repeat,
                    "fold": fold, "n_fit": len(train), "n_test": len(query),
                    "seconds": seconds, **model.info})


def save_timings(timings: list[dict]) -> None:
    if not timings:
        return
    new = pd.DataFrame(timings)
    if cfg.TIMING_CSV.exists():
        old = pd.read_csv(cfg.TIMING_CSV)
        key = ["method", "endpoint", "repeat", "fold"]
        old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(cfg.TIMING_CSV, index=False)
    timings.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true", help="refit folds that already have predictions")
    args = parser.parse_args()

    if not cfg.FEATURES_NPZ.exists():
        raise SystemExit(f"{cfg.FEATURES_NPZ} not found -- run 01_featurize.py first")
    check_tabpfn()
    cfg.ensure_dirs()

    df = pd.read_csv(cfg.MASTER_CSV)
    names = np.load(cfg.FEATURES_NPZ)["names"]
    if not np.array_equal(names, df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(f"{cfg.FEATURES_NPZ.name} does not line up with {cfg.MASTER_CSV.name}")
    mols = Molecules.load(cfg.FEATURES_NPZ, len(df))
    folds = pd.read_csv(cfg.FOLD_CSV)
    print(f"{cfg.ACTIVE.label}: {len(df):,} molecules, {int(mols.mmp.nnz / 2):,} matched pairs; "
          f"settings {json.dumps(cfg.MOLPAIR)}", flush=True)

    timings: list[dict] = []
    for endpoint in args.endpoint:
        start = time.time()
        for repeat in args.repeat:
            for fold in args.fold:
                run_fold(df, mols, folds, endpoint, repeat, fold, args.force, timings)
                save_timings(timings)
        n = len(list((cfg.PRED_DIR / cfg.MOLPAIR_METHOD).glob(f"{endpoint}_r*_f*.csv")))
        print(f"{endpoint:<17} {n:>2}/25 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
