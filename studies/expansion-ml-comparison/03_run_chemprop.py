#!/usr/bin/env python
"""Step 3: the chemprop arms of the comparison.

Each run trains one D-MPNN on one unit of work and one fold, then predicts the
fixed `ds == 'test'` molecules. A unit is an assay family for the multitask
variants and a single endpoint for the single-task control:

  chemprop     multitask, message passing initialised from scratch    (3 x 25 = 75)
  chemeleon    multitask, message passing from the CheMeleon foundation model
               (`--from-foundation CHEMELEON`), then fine-tuned        (3 x 25 = 75)
  chemprop_st  single-task, from scratch -- the control that separates
               architecture from multitask transfer                   (9 x 25 = 225)

The multitask variants differ from each other only in how the message-passing
block is initialised, and both read the same folds/<group>_r{r}_f{f}.csv, so a
difference between them is a difference in the representation. `--splits-column
split` hands chemprop the train/val/test assignment made in step 1: the held-out
fifth of the training molecules is the validation set for early stopping, and the
test molecules are never seen.

Two details worth knowing about `chemprop predict`:

  * the prediction columns are named after the model's targets (they come from the
    checkpoint, not from the input file), so a test file of Name + SMILES comes
    back with one prediction column per target;
  * it copies the input file through to the output, which is why the measured
    values are always taken from data/master.csv and joined back by Name --
    passing a file that contains the target columns would have chemprop overwrite
    them with the predictions.

Trained weights are discarded once a fold has been predicted: a CheMeleon
checkpoint is ~112 MB and nothing downstream reads it.

    python 03_run_chemprop.py                                  # both multitask arms
    python 03_run_chemprop.py --variant chemprop_st            # the single-task control
    python 03_run_chemprop.py --unit permeability --repeat 0 --fold 0    # smoke test
"""

import argparse
import shutil
import subprocess
import time
from datetime import timedelta

import pandas as pd

import config as cfg

def run_log(variant: str):
    """One log per variant, so the two variants can run concurrently.

    They contend for nothing: CheMeleon is GPU-bound on MPS and the from-scratch
    variant is CPU-bound, so running them side by side takes the smaller one off
    the critical path.
    """
    return cfg.LOG_DIR / f"chemprop_run_{variant}.log"


def test_input(variant: str, unit: str):
    """A Name + SMILES file of the unit's test molecules, shared by every fold.

    Only these two columns: `chemprop predict` copies its input through to the
    output, and a file carrying the target columns would come back with the
    measured values overwritten by the predictions.
    """
    prefix = "test_st_" if cfg.is_single_task(variant) else "test_"
    path = cfg.FOLD_DIR / f"{prefix}{unit}.csv"
    if not path.exists():
        fold_df = pd.read_csv(cfg.variant_fold_input(variant, unit, 0, 0))
        test_df = fold_df.loc[fold_df[cfg.SPLIT_COL] == "test", [cfg.ID_COL, cfg.SMILES_COL]]
        test_df.to_csv(path, index=False)
        print(f"wrote {path.name} ({len(test_df)} molecules)")
    return path


def train_cmd(variant: str, unit: str, repeat: int, fold: int, out_dir, opts):
    cmd = [
        "chemprop", "train",
        "--data-path", str(cfg.variant_fold_input(variant, unit, repeat, fold)),
        "--output-dir", str(out_dir),
        "--task-type", "regression",
        "--smiles-columns", cfg.SMILES_COL,
        "--target-columns", *cfg.unit_targets(variant, unit),
        "--splits-column", cfg.SPLIT_COL,
        "--num-replicates", "1",
        "--ensemble-size", str(cfg.ENSEMBLE_SIZE),
        "--epochs", str(cfg.EPOCHS),
        "--batch-size", str(cfg.BATCH_SIZE),
        "--pytorch-seed", str(cfg.fold_seed(repeat, fold)),
        "--accelerator", cfg.accelerator(variant, opts.accelerator),
        "--num-workers", str(opts.num_workers),
    ]
    foundation = cfg.VARIANTS[variant]["from_foundation"]
    if foundation is not None:
        # CheMeleon requires --multi-hot-atom-featurizer-mode V2, which is the default.
        cmd += ["--from-foundation", foundation]
    if cfg.REMOVE_LIGHTNING_CHECKPOINTS:
        cmd.append("--remove-checkpoints")
    return cmd


def predict_cmd(variant: str, unit: str, model_path, raw_path, opts):
    return [
        "chemprop", "predict",
        "--test-path", str(test_input(variant, unit)),
        "--model-paths", str(model_path),
        "--preds-path", str(raw_path),
        "--smiles-columns", cfg.SMILES_COL,
        "--accelerator", cfg.accelerator(variant, opts.accelerator),
        "--num-workers", str(opts.num_workers),
    ]


def tidy_predictions(variant: str, unit: str, repeat: int, fold: int, raw_path, master: pd.DataFrame):
    """chemprop's wide output -> the tidy schema shared by every method.

    One row per (molecule, endpoint) for the test molecules that have a measured
    value for that endpoint. The measured values come from master.csv.
    """
    raw = pd.read_csv(raw_path)
    targets = cfg.unit_targets(variant, unit)
    missing = [t for t in targets if t not in raw.columns]
    if missing:
        raise SystemExit(f"{raw_path} has no prediction column for {missing}")

    long = raw.melt(
        id_vars=[cfg.ID_COL, cfg.SMILES_COL],
        value_vars=targets,
        var_name="endpoint",
        value_name="y_pred",
    )
    truth = master.melt(
        id_vars=[cfg.ID_COL],
        value_vars=targets,
        var_name="endpoint",
        value_name="y_true",
    ).dropna(subset=["y_true"])

    out = long.merge(truth, on=[cfg.ID_COL, "endpoint"], how="inner")
    out["method"] = variant
    out["repeat"] = repeat
    out["fold"] = fold
    return out[cfg.PRED_COLUMNS]


def run_one(variant: str, unit: str, repeat: int, fold: int, master: pd.DataFrame, opts) -> bool:
    """Train and predict one fold. Returns True if it ran, False if it was skipped."""
    out_path = cfg.pred_csv(variant, unit, repeat, fold)
    if out_path.exists() and not opts.force:
        return False

    out_dir = cfg.SCRATCH_DIR / variant / f"{unit}_r{repeat}_f{fold}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_predictions.csv"

    start = time.time()
    try:
        with open(run_log(variant), "a") as log:
            def run(cmd):
                log.write(" ".join(str(c) for c in cmd) + "\n")
                log.flush()
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)

            log.write(f"\n{'=' * 78}\n[{variant}] {unit} repeat {repeat} fold {fold}\n")
            run(train_cmd(variant, unit, repeat, fold, out_dir, opts))

            models = sorted(out_dir.glob("model_*/best.pt"))
            if not models:
                raise SystemExit(f"no checkpoint written under {out_dir}")
            run(predict_cmd(variant, unit, models[0], raw_path, opts))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tidy_predictions(variant, unit, repeat, fold, raw_path, master).to_csv(out_path, index=False)
    finally:
        # The weights are ~112 MB apiece for CheMeleon and nothing downstream reads
        # them, so a completed -- or failed -- fold leaves nothing behind.
        shutil.rmtree(out_dir, ignore_errors=True)

    print(f"  [{variant:<11}] {unit:<17} r{repeat} f{fold}  {time.time() - start:6.1f}s", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", nargs="+", default=["chemprop", "chemeleon"], choices=cfg.VARIANT_NAMES)
    parser.add_argument(
        "--unit",
        nargs="+",
        default=None,
        choices=cfg.GROUPS + cfg.TARGET_COLS,
        help="assay families for the multitask variants, endpoints for chemprop_st "
             "(default: every unit the variant covers)",
    )
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true", help="retrain folds that already have predictions")
    parser.add_argument(
        "--accelerator",
        default=None,
        help="override the configured device (cpu / gpu / mps); by default a CUDA "
             "device is used where one is available",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="dataloader workers; only affects wall clock, not the trained model",
    )
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    cfg.ensure_dirs()
    master = pd.read_csv(cfg.MASTER_CSV)

    # Fold-major: after one pass over the groups and variants every endpoint has a
    # complete replicate, so an interrupted run still gives a full picture.
    todo = [
        (repeat, fold, variant, unit)
        for repeat in args.repeat
        for fold in args.fold
        for variant in args.variant
        for unit in (args.unit or cfg.units(variant))
        if unit in cfg.units(variant)
    ]
    print(f"{len(todo)} (variant, unit, fold) combinations requested", flush=True)

    started = time.time()
    n_run = 0
    for i, (repeat, fold, variant, unit) in enumerate(todo, start=1):
        if run_one(variant, unit, repeat, fold, master, args):
            n_run += 1
            elapsed = time.time() - started
            eta = elapsed / n_run * (len(todo) - i)
            print(
                f"    {i}/{len(todo)} done, elapsed {timedelta(seconds=int(elapsed))}, "
                f"eta {timedelta(seconds=int(eta))}",
                flush=True,
            )

    for variant in args.variant:
        n = len(list((cfg.PRED_DIR / variant).glob("*.csv"))) if (cfg.PRED_DIR / variant).exists() else 0
        expected = len(cfg.units(variant)) * cfg.N_REPEATS * cfg.N_SPLITS
        print(f"{variant:<12} {n}/{expected} fold predictions on disk")


if __name__ == "__main__":
    main()
