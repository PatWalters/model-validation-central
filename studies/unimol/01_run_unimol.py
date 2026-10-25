#!/usr/bin/env python
"""Step 1: fine-tune Uni-Mol and Uni-Mol2 with unimol_tools on the 25 folds.

For each arm and repeat this is one call to the package's own `MolTrain`, with
nothing of ours inside the training loop:

  the labelled training molecules of the repeat, each tagged with the fold it
  is held out in, go to `MolTrain(split='select', split_group_col='fold',
  kfold=5)`, which fits one model per fold on the other four fifths,
  early-stops it on the held-out fifth and saves it as `model_<fold>.pth`
  (`split='select'` walks the sorted group values, so model k is fold k)

  the fixed test set is then featurised once by the package's `DataHub` and
  predicted by each of the five fold models separately -- `MolPredict` would
  average them, which is not what a fold is -- and inverse-transformed by the
  saved target scaler

Conformers are the package's own: one RDKit ETKDG conformer per molecule,
generated with its fixed seed, so every repeat sees the same coordinates.

The fold models are multitask over all endpoints of the data set, with the
package's NaN-masked L1 loss, and predictions are written per endpoint in the
tidy schema every other arm uses. A repeat whose five folds are all on disk is
skipped, so the sweep can be split between machines by arm and repeat.

    python 01_run_unimol.py --arm unimol
    ADME_DATASET=biogen python 01_run_unimol.py --arm unimol2 --repeat 0 1
"""

import argparse
import json
import shutil
import socket
import time

import numpy as np
import pandas as pd
import torch

import config as cfg


def training_frame(df: pd.DataFrame, folds: pd.DataFrame, repeat: int) -> pd.DataFrame:
    """The labelled training molecules of one repeat, with the fold each is held out in."""
    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    train = df[(df[cfg.SET_COL] == "train") & df[cfg.TARGET_COLS].notna().any(axis=1)].copy()
    train["fold"] = train[cfg.ID_COL].map(held_out)
    if train["fold"].isna().any():
        raise SystemExit(f"repeat {repeat}: {int(train['fold'].isna().sum())} training "
                         "molecules have no fold")
    train["fold"] = train["fold"].astype(int)
    return train[[cfg.ID_COL, cfg.SMILES_COL, "fold", *cfg.TARGET_COLS]].reset_index(drop=True)


def fit(arm: cfg.UniMolArm, train: pd.DataFrame, save_path, seed: int) -> None:
    from unimol_tools import MolTrain

    extra = {"model_size": arm.model_size} if arm.model_size else {}
    clf = MolTrain(
        task="multilabel_regression",
        data_type="molecule",
        epochs=cfg.MAX_EPOCHS,
        learning_rate=cfg.LEARNING_RATE,
        batch_size=arm.batch_size,
        early_stopping=cfg.PATIENCE,
        metrics="mae",
        split="select",
        split_group_col="fold",
        kfold=cfg.N_SPLITS,
        save_path=str(save_path),
        remove_hs=False,
        smiles_col=cfg.SMILES_COL,
        target_cols=cfg.TARGET_COLS,
        target_normalize=cfg.TARGET_NORMALIZE,
        max_norm=cfg.MAX_NORM,
        use_cuda=True,
        use_amp=True,
        use_ddp=False,
        use_gpu="0",
        model_name=arm.model_name,
        conf_cache_level=0,
        **extra,
    )
    clf.config.seed = seed  # MolTrain drops unknown keyword arguments
    clf.fit(train[[cfg.SMILES_COL, "fold", *cfg.TARGET_COLS]])


def predict_folds(save_path, test: pd.DataFrame) -> list[np.ndarray]:
    """Each of the five saved fold models on the test set, in original units."""
    from unimol_tools import MolPredict
    from unimol_tools.data import DataHub
    from unimol_tools.models import NNModel
    from unimol_tools.models.nnmodel import NNDataset
    from unimol_tools.tasks import Trainer

    predictor = MolPredict(load_model=str(save_path))
    config = predictor.config
    hub = DataHub(data=test[[cfg.SMILES_COL, *cfg.TARGET_COLS]], is_train=False,
                  save_path=str(save_path), **config)
    if len(hub.data["unimol_input"]) != len(test):
        raise SystemExit(f"DataHub kept {len(hub.data['unimol_input'])} of {len(test)} "
                         "test molecules -- the rows no longer line up")
    trainer = Trainer(save_path=str(save_path), **config)
    nn_model = NNModel(hub.data, trainer, **config)
    dataset = NNDataset(nn_model.features, np.asarray(hub.data["target"]))
    scaler = hub.data["target_scaler"]

    preds = []
    for fold in cfg.FOLDS:
        y, _, _ = trainer.predict(nn_model.model, dataset, nn_model.loss_func,
                                  nn_model.activation_fn, str(save_path), fold,
                                  scaler, epoch=1, load_model=True)
        preds.append(scaler.inverse_transform(y))
    return preds


def track_epochs() -> dict[int, dict]:
    """Record, per fold, the epoch the package saved and how many it ran.

    `EarlyStopper` saves only the weights, so the epoch it kept is read off it
    as it goes: whenever its best validation score improves, that epoch is the
    one written to `model_<fold>.pth`.
    """
    from unimol_tools.tasks import trainer

    seen: dict[int, dict] = {}
    original = trainer.EarlyStopper.early_stop_choice

    def watched(self, model, epoch, loss, metric_score=None):
        before = self.min_loss
        stop = original(self, model, epoch, loss, metric_score)
        entry = seen.setdefault(self.fold, {"best_epoch": None, "epochs": 0})
        entry["epochs"] = epoch + 1
        if self.min_loss < before:
            entry["best_epoch"] = epoch + 1
            entry["val_mae_z"] = float(self.min_loss)
        return stop

    trainer.EarlyStopper.early_stop_choice = watched
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=list(cfg.UNIMOL_ARMS))
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--keep", action="store_true",
                        help="keep the fold models after predicting (large)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()
    arm = cfg.UNIMOL_ARMS[args.arm]
    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    test = df[(df[cfg.SET_COL] == "test") & df[cfg.TARGET_COLS].notna().any(axis=1)]
    test = test.reset_index(drop=True)
    out_dir = cfg.PRED_DIR / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = track_epochs()

    for repeat in args.repeat:
        wanted = [cfg.pred_csv(args.arm, t, repeat, f) for t in cfg.TARGET_COLS for f in cfg.FOLDS]
        if all(p.exists() for p in wanted) and not args.force:
            print(f"=== {args.arm} r{repeat}: already done", flush=True)
            continue

        save_path = cfg.work_dir(args.arm, repeat)
        if save_path.exists():
            shutil.rmtree(save_path)
        train = training_frame(df, folds, repeat)
        seed = cfg.fold_seed(repeat, 0)
        print(f"=== {args.arm} r{repeat}: {len(train):,} training molecules, "
              f"{len(test):,} test, seed {seed} {time.strftime('%F %T')}", flush=True)

        epochs.clear()
        start = time.time()
        fit(arm, train, save_path, seed)
        fit_seconds = time.time() - start
        preds = predict_folds(save_path, test)
        seconds = time.time() - start

        Y = test[cfg.TARGET_COLS].to_numpy(dtype=float)
        for fold, P in zip(cfg.FOLDS, preds):
            maes = []
            for j, t in enumerate(cfg.TARGET_COLS):
                has = np.isfinite(Y[:, j])
                pd.DataFrame(
                    {
                        "method": args.arm,
                        "endpoint": t,
                        "repeat": repeat,
                        "fold": fold,
                        cfg.ID_COL: test.loc[has, cfg.ID_COL].to_numpy(),
                        cfg.SMILES_COL: test.loc[has, cfg.SMILES_COL].to_numpy(),
                        "y_true": Y[has, j],
                        "y_pred": P[has, j],
                    },
                    columns=cfg.PRED_COLUMNS,
                ).to_csv(cfg.pred_csv(args.arm, t, repeat, fold), index=False)
                maes.append(float(np.abs(P[has, j] - Y[has, j]).mean()))

            n_fit = int((train["fold"] != fold).sum())
            diag = {
                "method": args.arm,
                "model_name": arm.model_name,
                "model_size": arm.model_size,
                "repeat": repeat,
                "fold": fold,
                "seed": seed,
                # one MolTrain call fits all five folds; its time is shared evenly
                "seconds": seconds / cfg.N_SPLITS,
                "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                **epochs.get(fold, {}),
                "test_macro_mae": float(np.mean(maes)),
                "n_train": n_fit,
                "n_val": int(len(train) - n_fit),
                "n_test": int(len(test)),
            }
            cfg.diag_json(args.arm, repeat, fold).write_text(json.dumps(diag, indent=1))
            print(f"[{args.arm}] r{repeat} f{fold}: test macro MAE {np.mean(maes):.4f}", flush=True)

        pd.DataFrame([{
            "method": args.arm,
            "repeat": repeat,
            "host": socket.gethostname(),
            "fit_seconds": fit_seconds,
            "seconds": seconds,
        }]).to_csv(cfg.TIMING_CSV, mode="a", header=not cfg.TIMING_CSV.exists(), index=False)
        print(f"=== {args.arm} r{repeat}: {seconds / 60:.1f} min", flush=True)

        if not args.keep:
            shutil.rmtree(save_path)


if __name__ == "__main__":
    main()
