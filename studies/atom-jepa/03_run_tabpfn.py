#!/usr/bin/env python
"""Step 3: the frozen arm -- Atom-JEPA's embedding read by TabPFN 3.

Everything but the representation is the `monroe` arm's. The call is Monroe's
own `fit_predict_tabpfn` with its `default_ensemble_specs` (eight estimators,
no PCA) and `output_type="mean"`; the seed is the fold seed; the masks are the
ones 02_run_lightgbm.py and 09_run_monroe.py use in the source comparison. So
against `monroe` this arm changes one thing, which 720-d vector each molecule
is, and against `atomjepa_ft` it changes how the encoder is used: frozen and
read in context, rather than trained.

TabPFN 3 refuses more than 2,000 features and the embedding has 2,048. So in
each fold the 48 lowest-variance columns are dropped, the variance taken over
that fold's training rows only, as MolPAIR's global stage does in the molpair
study. No label is read to choose them.

Runs in the `monroe` environment, where tabpfn 8.x loads TabPFN 3; a preflight
check refuses to run if the installed tabpfn would load anything else.

    python 03_run_tabpfn.py
    ADME_DATASET=biogen python 03_run_tabpfn.py
    python 03_run_tabpfn.py --endpoint LOG_MGMB --repeat 0 --fold 0   # smoke test
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

import config as cfg


def check_tabpfn() -> None:
    import tabpfn
    from tabpfn.settings import settings
    default = settings.tabpfn.model_version.value
    if default != "v3":
        raise SystemExit(f"tabpfn {tabpfn.__version__} would load TabPFN {default}, not v3 -- "
                         "run this in the `monroe` environment")
    print(f"tabpfn {tabpfn.__version__} loads TabPFN {default}")


def load_embeddings(df: pd.DataFrame) -> np.ndarray:
    if not cfg.EMBED_NPZ.exists():
        raise SystemExit(f"{cfg.EMBED_NPZ} not found -- run 02_embed.py first")
    cached = np.load(cfg.EMBED_NPZ, allow_pickle=False)
    if not np.array_equal(cached["names"], df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(f"{cfg.EMBED_NPZ.name} does not line up with {cfg.MASTER_CSV.name}")
    return cached["X"]


def top_variance(X_fit: np.ndarray, k: int) -> np.ndarray:
    """The k columns that vary most across the fold's training rows, in order."""
    return np.sort(np.argsort(-X_fit.var(axis=0), kind="stable")[:k])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    check_tabpfn()
    sys.path.insert(0, str(cfg.MONROE_HOME))
    from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    X = load_embeddings(df)
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    out_dir = cfg.PRED_DIR / cfg.TABPFN_METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = []

    for endpoint in args.endpoint:
        y = df[endpoint].to_numpy()
        measured = df[endpoint].notna().to_numpy()
        for repeat in args.repeat:
            held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
            fold_of = df[cfg.ID_COL].map(held_out).to_numpy()
            for fold in args.fold:
                out_path = cfg.pred_csv(cfg.TABPFN_METHOD, endpoint, repeat, fold)
                if out_path.exists() and not args.force:
                    continue
                fit_mask = measured & ~is_test & (fold_of != fold)
                test_mask = measured & is_test
                start = time.time()
                cols = top_variance(X[fit_mask], cfg.TABPFN_MAX_FEATURES)
                pred = fit_predict_tabpfn(
                    X[fit_mask][:, cols], y[fit_mask], X[test_mask][:, cols],
                    is_classification=False,
                    ensemble_specs=default_ensemble_specs(),
                    seed=cfg.fold_seed(repeat, fold), output_type="mean",
                )
                seconds = time.time() - start
                test_df = df.loc[test_mask]
                pd.DataFrame(
                    {
                        "method": cfg.TABPFN_METHOD,
                        "endpoint": endpoint,
                        "repeat": repeat,
                        "fold": fold,
                        cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(),
                        cfg.SMILES_COL: test_df[cfg.SMILES_COL].to_numpy(),
                        "y_true": y[test_mask],
                        "y_pred": np.asarray(pred, dtype=float),
                    },
                    columns=cfg.PRED_COLUMNS,
                ).to_csv(out_path, index=False)
                timing.append({"method": cfg.TABPFN_METHOD, "endpoint": endpoint,
                               "repeat": repeat, "fold": fold, "seconds": seconds})
                print(f"{endpoint} r{repeat} f{fold}: {int(fit_mask.sum())} fit, "
                      f"{int(test_mask.sum())} test, {seconds:.1f}s", flush=True)

    if timing:
        t = pd.DataFrame(timing)
        t.to_csv(cfg.TIMING_CSV, mode="a", header=not cfg.TIMING_CSV.exists(), index=False)


if __name__ == "__main__":
    main()
