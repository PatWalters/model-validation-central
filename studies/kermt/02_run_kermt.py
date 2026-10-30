#!/usr/bin/env python
"""Step 2: fine-tune KERMT v2, GROVER base and GROVER large on the 25 folds.

Each fold is one call to the repository's own `main.py finetune`, with nothing
of ours inside the training loop:

  the fold's training molecules (the four fifths outside the held-out fold),
  the held-out fifth and the fixed test set are written as three CSVs, SMILES
  first and then every endpoint of the data set, blank where unmeasured, with
  the matching rows of the RDKit 2D feature cache beside each

  `main.py finetune` fits the target scaler on the training CSV only, trains
  for 100 epochs, keeps the epoch with the best validation MAE on the
  held-out fifth, and predicts the test CSV with it (test_result.csv)

The architecture flags are read from the checkpoint, as the repository's own
kermt-finetune runner does; the training recipe is cfg.FINETUNE for all three
checkpoints. A fold whose prediction files are all on disk is skipped, so the
sweep can be split between machines by arm, repeat and fold.

    python 02_run_kermt.py --arm kermt_ft
    ADME_DATASET=biogen python 02_run_kermt.py --arm grover_large_ft --repeat 3 4
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import config as cfg

# architecture fields the checkpoint dictates, as in
# skills/_shared/scripts/run_finetune_local.py
ARCH_FLAGS = ("hidden_size", "depth", "num_attn_head", "activation", "embedding_output_type")
# recorded per fold from the saved model, since some are the checkpoint's, not ours
EFFECTIVE_ARGS = ("dropout", "bond_drop_rate", "dist_coff", "weight_decay", "select_by_loss",
                  "ffn_hidden_size", "ffn_num_layers", "ffn_num_task_specific_layers",
                  "ffn_task_specific_hidden_size", "epochs", "batch_size", "init_lr", "max_lr",
                  "final_lr", "warmup_epochs", "backbone", "hidden_size")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_arch(ckpt) -> dict:
    """The encoder's shape, from the repository's own checkpoint validator."""
    out = subprocess.run(
        [sys.executable, str(cfg.KERMT_REPO / "skills/_shared/scripts/check_checkpoint.py"),
         "--mode", "finetune_init", "--ckpt", str(ckpt)],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(cfg.KERMT_REPO)},
    ).stdout
    report = json.loads(out[out.index("{"):])
    if not report["ok"]:
        raise SystemExit(f"{ckpt}: {report['errors']}")
    return report["arch"]


def write_split(frame: pd.DataFrame, features: np.ndarray, stem) -> None:
    frame[[cfg.SMILES_COL, *cfg.TARGET_COLS]].rename(columns={cfg.SMILES_COL: "smiles"}).to_csv(
        f"{stem}.csv", index=False)
    np.savez_compressed(f"{stem}.npz", features=features)


def finetune_argv(arch: dict, work, arm: cfg.KermtArm, ckpt, seed: int) -> list[str]:
    argv = [sys.executable, "-u", str(cfg.KERMT_REPO / "main.py"), "finetune",
            "--data_path", str(work / "train.csv"),
            "--features_path", str(work / "train.npz"),
            "--separate_val_path", str(work / "val.csv"),
            "--separate_val_features_path", str(work / "val.npz"),
            "--separate_test_path", str(work / "test.csv"),
            "--separate_test_features_path", str(work / "test.npz"),
            "--split_type", "random",  # required by argparse, unused with separate splits
            "--use_cuikmolmaker_featurization",  # atom and bond features on the GPU, as the README runs it
            "--checkpoint_path", str(ckpt),
            "--dataset_type", "regression",
            "--no_features_scaling",
            "--ensemble_size", "1",
            "--num_folds", "1",
            "--batch_size", str(arm.batch_size),
            "--seed", str(seed),
            "--gpu", "0",
            "--save_dir", str(work / "ckpt")]
    for name in ARCH_FLAGS:
        argv += [f"--{name}", str(arch[name])]
    if arch.get("self_attention"):
        argv += ["--self_attention", "--attn_hidden", str(arch["attn_hidden"]),
                 "--attn_out", str(arch["attn_out"])]
    for name, value in cfg.FINETUNE.items():
        argv += [f"--{name}", str(value)]
    return argv


def read_test_result(work, test: pd.DataFrame) -> np.ndarray:
    found = sorted((work / "ckpt").rglob("test_result.csv"))
    if len(found) != 1:
        raise SystemExit(f"expected one test_result.csv under {work / 'ckpt'}, found {len(found)}")
    result = pd.read_csv(found[0], header=[0, 1], index_col=0)
    if list(result.index) != list(test[cfg.SMILES_COL]):
        raise SystemExit("test_result.csv rows do not line up with the test set")
    return result["preds"][cfg.TARGET_COLS].to_numpy(dtype=float)


def run_fold(args, arm, ckpt, arch, df, X, folds, repeat, fold) -> None:
    wanted = [cfg.pred_csv(args.arm, t, repeat, fold) for t in cfg.TARGET_COLS]
    if all(p.exists() for p in wanted) and not args.force:
        print(f"=== {args.arm} r{repeat} f{fold}: already done", flush=True)
        return

    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for the test molecules
    labelled = df[cfg.TARGET_COLS].notna().any(axis=1).to_numpy()
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    masks = {"train": labelled & ~is_test & (fold_of != fold),
             "val": labelled & ~is_test & (fold_of == fold),
             "test": labelled & is_test}

    work = cfg.work_dir(args.arm, repeat, fold)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for name, mask in masks.items():
        write_split(df.loc[mask], X[mask], work / name)
    test = df.loc[masks["test"]].reset_index(drop=True)

    seed = cfg.fold_seed(repeat, fold)
    print(f"=== {args.arm} r{repeat} f{fold}: {int(masks['train'].sum()):,} train, "
          f"{int(masks['val'].sum()):,} val, {len(test):,} test, seed {seed} "
          f"{time.strftime('%F %T')}", flush=True)
    log_path = cfg.LOG_DIR / f"{args.arm}_r{repeat}_f{fold}.log"
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(finetune_argv(arch, work, arm, ckpt, seed), stdout=log,
                              stderr=subprocess.STDOUT, cwd=cfg.KERMT_REPO,
                              env={**os.environ, "PYTHONPATH": str(cfg.KERMT_REPO),
                                   "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    seconds = time.time() - start
    if proc.returncode:
        raise SystemExit(f"main.py finetune exited {proc.returncode} -- see {log_path}")

    P = read_test_result(work, test)
    Y = test[cfg.TARGET_COLS].to_numpy(dtype=float)
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

    # "best validation mae = x on epoch n", or "best val loss = x on epoch n" when
    # the checkpoint's saved args switch on select_by_loss, as GROVER's do
    best = re.findall(r"best val(?:idation)? (\w+) = ([\d.eE+-]+) on epoch (\d+)", log_path.read_text())
    import torch
    saved = sorted((work / "ckpt").rglob("model.pt"))
    effective = vars(torch.load(saved[0], map_location="cpu", weights_only=False)["args"])
    diag = {
        "method": args.arm,
        "checkpoint": arm.checkpoint,
        "repeat": repeat,
        "fold": fold,
        "seed": seed,
        "seconds": seconds,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        # KERMT counts epochs from 0 and runs them all; there is no early stop
        "best_epoch": int(best[-1][2]) + 1 if best else None,
        "epochs": cfg.FINETUNE["epochs"],
        "selected_on": best[-1][0] if best else None,
        "val_score": float(best[-1][1]) if best else None,
        # what the fold actually trained with: load_checkpoint copies the
        # pre-trained checkpoint's saved model args over the command line
        "effective": {k: effective.get(k) for k in EFFECTIVE_ARGS},
        "test_macro_mae": float(np.mean(maes)),
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": len(test),
    }
    cfg.diag_json(args.arm, repeat, fold).write_text(json.dumps(diag, indent=1))
    pd.DataFrame([{"method": args.arm, "repeat": repeat, "fold": fold,
                   "host": socket.gethostname(), "seconds": seconds}]).to_csv(
        cfg.TIMING_CSV, mode="a", header=not cfg.TIMING_CSV.exists(), index=False)
    print(f"[{args.arm}] r{repeat} f{fold}: test macro MAE {np.mean(maes):.4f}, best epoch "
          f"{diag['best_epoch']}, {seconds / 60:.1f} min", flush=True)

    if not args.keep:
        shutil.rmtree(work)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, choices=list(cfg.KERMT_ARMS))
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--epochs", type=int, help="override, for a smoke test only")
    parser.add_argument("--keep", action="store_true",
                        help="keep the fold's checkpoints and CSVs after predicting (large)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.epochs:
        cfg.FINETUNE["epochs"] = args.epochs

    cfg.ensure_dirs()
    arm = cfg.KERMT_ARMS[args.arm]
    ckpt = cfg.WEIGHTS_DIR / arm.checkpoint
    if sha256(ckpt) != arm.sha256:
        raise SystemExit(f"{ckpt} is not the checkpoint this study used")
    arch = checkpoint_arch(ckpt)
    print(f"{args.arm}: {arm.checkpoint}, arch {arch}", flush=True)

    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    cached = np.load(cfg.FEATURES_NPZ)
    if not np.array_equal(cached["names"], df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(f"{cfg.FEATURES_NPZ.name} does not line up with master.csv -- rerun 01")
    X = cached["X"]
    (cfg.PRED_DIR / args.arm).mkdir(parents=True, exist_ok=True)

    for repeat in args.repeat:
        for fold in args.fold:
            run_fold(args, arm, ckpt, arch, df, X, folds, repeat, fold)


if __name__ == "__main__":
    main()
