#!/usr/bin/env python
"""Step 6: Monroe + TabPFN 3 with the anchors in context.

The `monroe` arm's call exactly -- Monroe's fit_predict_tabpfn, its
default_ensemble_specs, output_type="mean", the fold seed -- with one change to
the in-context training set: every feasible series' anchor, with its measured
value, is appended to the fold's training rows. That is the information
Causal-Chemprop is given, handed over the way an in-context model takes
information, with no retraining and no correction step.

All anchors of an endpoint go in together, one fit per fold, rather than one
anchor per series per fit. A test molecule therefore also sees the anchors of
other series. In the lead-optimisation setting the paper describes, every
series' anchor has been measured, so that is information the experimenter
holds; and Causal-Chemprop's own encoder was trained on a set those other
anchors are not in, so if anything this favours neither side much. The
predictions are written for every measured test molecule; only the scored ones
are ever scored.

Runs in the `monroe` environment.

    python 06_run_monroe_anchor.py
    ADME_DATASET=biogen python 06_run_monroe_anchor.py
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

import config as cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", nargs="+", default=None, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(cfg.MONROE_HOME))
    from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    emb = np.load(cfg.MONROE_NPZ, allow_pickle=False)
    if not np.array_equal(emb["names"].astype(str), df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(f"{cfg.MONROE_NPZ.name} does not line up with {cfg.MASTER_CSV.name}")
    X = emb["X"]
    series = pd.read_csv(cfg.SERIES_CSV)
    series = series[series["feasible"]]
    endpoints = args.endpoint or [e for e in cfg.TARGET_COLS if e in set(series["endpoint"])]
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    timing = []

    for endpoint in endpoints:
        y = df[endpoint].to_numpy()
        measured = df[endpoint].notna().to_numpy()
        anchors = set(series.loc[(series["endpoint"] == endpoint) & (series["role"] == "anchor"),
                                 cfg.ID_COL])
        is_anchor = df[cfg.ID_COL].isin(anchors).to_numpy()
        for repeat in args.repeat:
            fold_of = df[cfg.ID_COL].map(
                folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]).to_numpy()
            for fold in args.fold:
                out = cfg.pred_csv(cfg.MONROE_ANCHOR_METHOD, endpoint, repeat, fold)
                if out.exists() and not args.force:
                    continue
                fit_mask = (measured & ~is_test & (fold_of != fold)) | is_anchor
                test_mask = measured & is_test
                start = time.time()
                pred = fit_predict_tabpfn(X[fit_mask], y[fit_mask], X[test_mask],
                                          is_classification=False,
                                          ensemble_specs=default_ensemble_specs(),
                                          seed=cfg.fold_seed(repeat, fold), output_type="mean")
                seconds = time.time() - start
                t = df.loc[test_mask]
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "method": cfg.MONROE_ANCHOR_METHOD, "endpoint": endpoint,
                    "repeat": repeat, "fold": fold, cfg.ID_COL: t[cfg.ID_COL].to_numpy(),
                    cfg.SMILES_COL: t[cfg.SMILES_COL].to_numpy(), "y_true": y[test_mask],
                    "y_pred": np.asarray(pred, dtype=float),
                }, columns=cfg.PRED_COLUMNS).to_csv(out, index=False)
                timing.append({"method": cfg.MONROE_ANCHOR_METHOD, "endpoint": endpoint,
                               "repeat": repeat, "fold": fold, "seconds": seconds})
                print(f"{endpoint} r{repeat} f{fold}: {int(fit_mask.sum())} in context "
                      f"({int(is_anchor.sum())} anchors), {seconds:.1f}s", flush=True)

    if timing:
        pd.DataFrame(timing).to_csv(cfg.TIMING_CSV, mode="a",
                                    header=not cfg.TIMING_CSV.exists(), index=False)


if __name__ == "__main__":
    main()
