#!/usr/bin/env python
"""Step 3: the ChemProp + CheMeleon arm -- 300 fine-tuned D-MPNNs.

A chemprop D-MPNN whose message-passing block is initialised from the CheMeleon
foundation model (Burns et al., J. Chem. Inf. Model. 2026,
doi:10.1021/acs.jcim.6c01546) and then fine-tuned on one target and one fold.
Same settings as the `chemeleon` arm in `studies/expansion-ml-comparison` --
50 epochs, batch 64, one network per fold, no ensembling -- so a fold here is
the same model as a fold there, fit on different molecules.

Every target in this collection is one pIC50 column, so this arm is single-task
by construction. There is no assay family to pool over and so no multitask
variant to compare against; that question belongs to the other study.

`--splits-column split` hands chemprop the assignment 00_prepare_data.py wrote
from the upstream index lists: the fold's own validation molecules are the
early-stopping set and the test molecules are never seen. Unlike the other
studies here the test set rotates between folds, so there is one prediction
input per fold rather than one per target.

Two details worth knowing about `chemprop predict`:

  * the prediction column is named after the model's target, which comes from
    the checkpoint rather than the input file;
  * it copies the input file through to the output, which is why the test inputs
    carry only Name and SMILES -- a file holding `y` would come back with the
    measured values overwritten by the predictions.

Trained weights are discarded once a fold has been predicted: a CheMeleon
checkpoint is ~112 MB and nothing downstream reads it.

    python 03_run_chemprop.py                                # all 300
    python 03_run_chemprop.py --split umap --target 284       # one target, one scheme
    python 03_run_chemprop.py --accelerator cpu              # override the device
"""

import argparse
import shutil
import subprocess
import time
from datetime import timedelta

import pandas as pd

import config as cfg
from fold_data import FoldData, add_fold_arguments

# The chemprop target column in the fold files. One column, because one target
# data set is one endpoint.
TARGET_COLUMN = "y"


def accelerator(override: str | None) -> str:
    """Which device to train on -- a wall-clock choice, not a modelling one.

    CheMeleon's message passing is d_h=2048 / depth=6 against chemprop's default
    300/3, roughly fifteen times the work per epoch, which is why this arm wants
    a GPU and why 300 of them is the long pole of the study.
    """
    if override:
        return override
    import torch

    if torch.cuda.is_available():
        return "gpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_cmd(target: int, split: str, fold: int, out_dir, device: str, opts):
    return [
        "chemprop", "train",
        "--data-path", str(cfg.fold_input(target, split, fold)),
        "--output-dir", str(out_dir),
        "--task-type", "regression",
        "--smiles-columns", cfg.OUT_SMILES_COL,
        "--target-columns", TARGET_COLUMN,
        "--splits-column", cfg.SPLIT_COL,
        "--num-replicates", "1",
        "--ensemble-size", str(cfg.ENSEMBLE_SIZE),
        "--epochs", str(cfg.EPOCHS),
        "--batch-size", str(cfg.BATCH_SIZE),
        "--pytorch-seed", str(cfg.fold_seed(target, split, fold)),
        "--from-foundation", "CHEMELEON",
        "--accelerator", device,
        "--num-workers", str(opts.num_workers),
        *(["--remove-checkpoints"] if cfg.REMOVE_LIGHTNING_CHECKPOINTS else []),
    ]


def predict_cmd(target: int, split: str, fold: int, model_path, raw_path, device: str, opts):
    return [
        "chemprop", "predict",
        "--test-path", str(cfg.test_input(target, split, fold)),
        "--model-paths", str(model_path),
        "--preds-path", str(raw_path),
        "--smiles-columns", cfg.OUT_SMILES_COL,
        "--accelerator", device,
        "--num-workers", str(opts.num_workers),
    ]


def read_raw(raw_path, rows, data: FoldData) -> pd.Series:
    """chemprop's output, realigned to the fold's test rows in master.csv order.

    Joined on Name rather than trusted to come back in order, and checked for
    completeness, because a silent reordering would be invisible in the metrics.
    Name is unique within one target, which is the only scope this join spans.
    """
    raw = pd.read_csv(raw_path)
    if TARGET_COLUMN not in raw.columns:
        raise SystemExit(f"{raw_path} has no {TARGET_COLUMN!r} prediction column")

    want = data.master[cfg.OUT_ID_COL].to_numpy()[rows]
    pred = raw.set_index(cfg.OUT_ID_COL)[TARGET_COLUMN]
    if pred.index.has_duplicates:
        raise SystemExit(f"{raw_path} names are not unique")
    missing = set(want) - set(pred.index)
    if missing:
        raise SystemExit(f"{raw_path} is missing {len(missing)} of the fold's test molecules")
    return pred.reindex(want)


def run_fold(data: FoldData, target: int, split: str, fold: int, device: str, opts) -> bool:
    """Train and predict one fold. True if it ran, False if it was already done."""
    out_path = cfg.pred_csv(cfg.CHEMELEON_METHOD, target, split, fold)
    if out_path.exists() and not opts.force:
        return False

    rows = data.fold(target, split, fold)
    out_dir = cfg.SCRATCH_DIR / cfg.CHEMELEON_METHOD / f"tid_{target}_{split}_f{fold}"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_predictions.csv"

    start = time.time()
    try:
        with open(cfg.LOG_DIR / "chemprop_run.log", "a") as log:
            def run(cmd):
                log.write(" ".join(str(c) for c in cmd) + "\n")
                log.flush()
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)

            log.write(f"\n{'=' * 78}\ntid_{target} {split} fold {fold}\n")
            run(train_cmd(target, split, fold, out_dir, device, opts))

            models = sorted(out_dir.glob("model_*/best.pt"))
            if not models:
                raise SystemExit(f"no checkpoint written under {out_dir}")
            run(predict_cmd(target, split, fold, models[0], raw_path, device, opts))

        pred = read_raw(raw_path, rows.test, data)
        data.write_predictions(cfg.CHEMELEON_METHOD, target, split, fold, rows.test,
                               pred.to_numpy(), extra={"n_train": len(rows.train)})
    finally:
        # ~112 MB of weights per fold, and nothing downstream reads them, so a
        # completed -- or failed -- fold leaves nothing behind.
        shutil.rmtree(out_dir, ignore_errors=True)

    print(f"  tid_{target:<6} {split:<11} f{fold}  {time.time() - start:6.1f}s", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_fold_arguments(parser)
    parser.add_argument("--accelerator", default=None,
                        help="override the device (cpu / gpu / mps); a CUDA device is "
                             "used wherever one is available")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="dataloader workers; affects wall clock, not the model")
    args = parser.parse_args()

    cfg.ensure_dirs()
    (cfg.PRED_DIR / cfg.CHEMELEON_METHOD).mkdir(parents=True, exist_ok=True)
    data = FoldData.load()
    device = accelerator(args.accelerator)
    print(f"training on {device}", flush=True)

    # Split-major, then fold: after one pass over the targets every scheme has a
    # complete fold, so an interrupted run still covers all six schemes rather
    # than finishing the first two and none of the rest.
    todo = [
        (split, fold, target)
        for split in args.split
        for fold in args.fold
        for target in args.target
    ]
    print(f"{len(todo)} (target, split, fold) combinations requested", flush=True)

    started, n_run = time.time(), 0
    for i, (split, fold, target) in enumerate(todo, start=1):
        if run_fold(data, target, split, fold, device, args):
            n_run += 1
            elapsed = time.time() - started
            eta = elapsed / n_run * (len(todo) - i)
            print(f"    {i}/{len(todo)} done, elapsed {timedelta(seconds=int(elapsed))}, "
                  f"eta {timedelta(seconds=int(eta))}", flush=True)

    n = len(list((cfg.PRED_DIR / cfg.CHEMELEON_METHOD).glob("*.csv")))
    print(f"chemeleon {n}/{len(cfg.TARGETS) * len(cfg.SPLITS) * len(cfg.FOLDS)} "
          "fold predictions on disk")


if __name__ == "__main__":
    main()
