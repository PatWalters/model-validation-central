#!/usr/bin/env python
"""Step 3: the encoder Causal-Chemprop sits on -- the authors' Chemprop, per fold.

One single-task model per endpoint and fold, on the fold's four fifths, early
stopped on the held-out fifth, with the architecture and schedule of the
release's causal_utils.train (cc_kit.build_model). Its ordinary predictions on
every measured test molecule are the `cc_chemprop` arm; its checkpoint is what
04_run_causal.py builds each series' SCM on.

Only endpoints with at least one feasible series are fitted: on the others
there is nothing to score Causal-Chemprop on, and the arm would be a sixth
from-scratch Chemprop the repository already has.

    python 03_train_encoder.py
    ADME_DATASET=biogen python 03_train_encoder.py
    python 03_train_encoder.py --endpoint LOG_HLM --repeat 0 --fold 0   # smoke test
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

import cc_kit
import config as cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", nargs="+", default=None, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    series = pd.read_csv(cfg.SERIES_CSV)
    feasible = sorted(set(series.loc[series["feasible"], "endpoint"]), key=cfg.TARGET_COLS.index)
    endpoints = args.endpoint or feasible
    is_test = (df[cfg.SET_COL] == "test").to_numpy()

    for endpoint in endpoints:
        measured = df[endpoint].notna().to_numpy()
        for repeat in args.repeat:
            fold_of = df[cfg.ID_COL].map(
                folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]).to_numpy()
            for fold in args.fold:
                out = cfg.pred_csv(cfg.CC_CHEMPROP_METHOD, endpoint, repeat, fold)
                meta = cfg.MODEL_DIR / f"{endpoint}_r{repeat}_f{fold}" / "encoder.json"
                if out.exists() and meta.exists() and not args.force:
                    continue
                fit = df[measured & ~is_test & (fold_of != fold)]
                val = df[measured & ~is_test & (fold_of == fold)]
                test = df[measured & is_test]
                start = time.time()
                ckpt, scaler, epoch = cc_kit.train_encoder(
                    fit, val, endpoint, meta.parent, cfg.fold_seed(repeat, fold))
                fit_s = time.time() - start
                model = cc_kit.load_encoder(ckpt, meta.parent / "scm")
                pred = cc_kit.predict_regular(model, test[cfg.SMILES_COL])
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "method": cfg.CC_CHEMPROP_METHOD, "endpoint": endpoint,
                    "repeat": repeat, "fold": fold,
                    cfg.ID_COL: test[cfg.ID_COL].to_numpy(),
                    cfg.SMILES_COL: test[cfg.SMILES_COL].to_numpy(),
                    "y_true": test[endpoint].to_numpy(), "y_pred": pred,
                }, columns=cfg.PRED_COLUMNS).to_csv(out, index=False)
                meta.write_text(json.dumps({
                    "endpoint": endpoint, "repeat": repeat, "fold": fold,
                    "checkpoint": ckpt.name, "best_epoch": epoch,
                    "scaler_mean": scaler.mean, "scaler_scale": scaler.scale,
                    "n_fit": len(fit), "n_val": len(val), "seconds": fit_s,
                }, indent=1))
                mae = float(np.mean(np.abs(pred - test[endpoint].to_numpy())))
                print(f"{endpoint} r{repeat} f{fold}: {len(fit)} fit, epoch {epoch}, "
                      f"test MAE {mae:.3f}, {fit_s:.0f}s", flush=True)


if __name__ == "__main__":
    main()
