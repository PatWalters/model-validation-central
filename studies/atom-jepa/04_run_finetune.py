#!/usr/bin/env python
"""Step 4: the paper's own arm -- Atom-JEPA fine-tuned, with the authors' code.

For each of the 25 folds one multitask model is trained over every endpoint of
the data set, as the paper does for ExpansionRx and Biogen: the pre-trained
EquiformerV3 encoder, a shared trunk, a small head per endpoint, a masked Huber
loss weighted by learned per-task uncertainty, the encoder frozen for the first
ten of fifty epochs, L2-SP towards the pre-trained weights, an EMA of the
weights, a random one of ten conformers per molecule each epoch. All of it is
the authors' `train_multitask`, configured by their own
conf/finetune_expansionrx.yaml or conf/finetune_biogen_adme.yaml; the few
overrides are `cfg.FINETUNE_OVERRIDES`, and none of them changes what is
trained.

What is ours is only the split:

  train       the four fifths of `ds == 'train'` outside the held-out fold
  validation  the held-out fifth -- picks the best epoch, as it picks the
              early-stopping epoch for the ChemProp arms
  test        every `ds == 'test'` molecule, the same set every arm is scored on

A molecule enters an endpoint's training loss only where it has a value for
that endpoint, and an endpoint's test predictions are written only for the
test molecules that have one, so the scored rows are exactly the rows every
other arm is scored on. The test prediction is the mean over all ten
conformers: the paper's "Atom-JEPA x10 conf".

Runs in the `atomjepa` environment on a CUDA machine. Each fold is a resume
point.

    python 04_run_finetune.py
    ADME_DATASET=biogen python 04_run_finetune.py
    python 04_run_finetune.py --repeat 0 --fold 0            # one fold, to time it
    python 04_run_finetune.py --repeat 0 --fold 0 --set finetune.epochs=2   # smoke test

Splitting a sweep across machines: list "repeat fold" pairs, one per line, in
folds/<ds>/claimed_folds.txt on a machine, and it skips them. The file is
re-read before every fold, so the split can be changed while a sweep runs.
`--reverse` walks the folds from the last, so two machines started at
opposite ends meet in the middle. A fold in progress leaves a
diagnostics/<ds>/running_r<repeat>_f<fold> marker naming its host. Each
fold records the host that trained it.

ATOMJEPA_NUM_WORKERS sets the data-loader worker count on one machine without
changing the recipe. spark (a GB10, aarch64) deadlocks when the loader forks
workers after cuEquivariance has started its threads, so it runs with 0.
"""

import argparse
import json
import os
import pickle
import socket
import sys
import time

import numpy as np
import pandas as pd

import config as cfg

def claimed_elsewhere() -> set[tuple[int, int]]:
    """(repeat, fold) pairs another machine is running; re-read before every fold."""
    path = cfg.FOLD_DIR / "claimed_folds.txt"
    if not path.exists():
        return set()
    pairs = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            r, f = line.split()
            pairs.add((int(r), int(f)))
    return pairs


CONFIG_NAME = {"expansion": "finetune_expansionrx", "biogen": "finetune_biogen_adme"}


def machine_overrides() -> list[str]:
    """Per-machine execution settings that do not change what is trained."""
    workers = os.environ.get("ATOMJEPA_NUM_WORKERS")
    return [] if workers is None else [f"++data.num_workers={int(workers)}"]


def load_cfg(extra: list[str]):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(cfg.ATOMJEPA_HOME / "conf"), version_base=None):
        return compose(config_name=CONFIG_NAME[cfg.DATASET],
                       overrides=cfg.FINETUNE_OVERRIDES + [
                           f"++finetune.ckpt_path={cfg.ATOMJEPA_CKPT}",
                           f"++misc.checkpoint_dir={cfg.LOG_DIR / 'checkpoints'}",
                       ] + machine_overrides() + extra)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--set", nargs="*", default=[],
                        help="extra Hydra overrides; a run with any is written outside predictions/")
    parser.add_argument("--reverse", action="store_true",
                        help="run the folds from the last one backwards")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(cfg.ATOMJEPA_HOME))
    import torch
    from data.datasets.admet.multitask_finetune import MultiTaskFinetuneDataset, multitask_collate
    from finetuning.admet.training.train import train_multitask
    from finetuning.admet.training.utils import (
        build_equiformer_multitask_factory, load_pretrained_encoder, resolve_device)

    cfg.ensure_dirs()
    hcfg = load_cfg(args.set)
    ftc = hcfg.finetune
    # A run with extra overrides is a test of the plumbing, not the arm.
    out_root = cfg.PRED_DIR if not args.set else cfg.PROJECT_DIR / "scratch" / cfg.DATASET
    print(f"recipe: lr={ftc.lr} epochs={ftc.epochs} freeze={ftc.freeze_encoder_epochs} "
          f"batch={ftc.batch_size} l2sp={ftc.l2sp['lambda']}conformers={ftc.num_conformers} "
          f"precision={ftc.precision} eval={ftc.conformer_eval_mode}", flush=True)

    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    tasks = cfg.TARGET_COLS
    caches = sorted(cfg.CONFORMER_DIR.glob(f"{cfg.DATASET}_*.pkl"))
    if len(caches) != 1:
        raise SystemExit(f"expected one conformer cache in {cfg.CONFORMER_DIR} -- run 01_conformers.py")
    with open(caches[0], "rb") as fh:
        conformers = pickle.load(fh)

    device = torch.device(resolve_device(hcfg.misc.device))
    _, eqv3_cfg, ckpt = load_pretrained_encoder(cfg.ATOMJEPA_CKPT, device, load_weights=False)
    factory, F_in, enc_cfg = build_equiformer_multitask_factory(
        hcfg, eqv3_cfg, ckpt["context_encoder"], device)
    max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))

    def make_ds(part: pd.DataFrame, tag: str, sample: bool):
        ds = MultiTaskFinetuneDataset(
            part, conformers, eqv3_cfg.max_radius, tasks, max_z=max_z,
            num_conformers=int(ftc.num_conformers), sample_conformers=sample,
            smiles_col=cfg.SMILES_COL, tag=tag, features=None)
        if len(ds.keep_index) != len(part):
            raise SystemExit(f"{tag}: {len(part) - len(ds.keep_index)} molecules dropped by the "
                             "dataset -- every arm must see the same molecules")
        return ds

    # A molecule with no label for any endpoint teaches nothing; the authors'
    # loaders drop those rows too.
    labelled = df[tasks].notna().any(axis=1)
    is_test = df[cfg.SET_COL] == "test"
    test_df = df[is_test & labelled].reset_index(drop=True)
    test_ds = make_ds(test_df, f"{cfg.DATASET}/test", False)

    order = [(r, f) for r in args.repeat for f in args.fold]
    for repeat, fold in (reversed(order) if args.reverse else order):
        held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
        fold_of = df[cfg.ID_COL].map(held_out)
        out_dir = out_root / cfg.FT_METHOD
        paths = {t: out_dir / f"{t}_r{repeat}_f{fold}.csv" for t in tasks}
        if all(p.exists() for p in paths.values()) and not args.force:
            continue
        if (repeat, fold) in claimed_elsewhere() and not args.force:
            print(f"=== r{repeat} f{fold}: claimed by another machine, skipped", flush=True)
            continue
        train_df = df[~is_test & labelled & (fold_of != fold)].reset_index(drop=True)
        val_df = df[~is_test & labelled & (fold_of == fold)].reset_index(drop=True)
        train_ds = make_ds(train_df, f"{cfg.DATASET}/train", True)
        val_ds = make_ds(val_df, f"{cfg.DATASET}/valid", False)

        marker = cfg.DIAG_DIR / f"running_r{repeat}_f{fold}"
        if not args.set:
            marker.write_text(socket.gethostname())
        seed = cfg.fold_seed(repeat, fold)
        print(f"=== r{repeat} f{fold} seed {seed}: train {len(train_df)} "
              f"val {len(val_df)} test {len(test_df)}", flush=True)
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        r = train_multitask(hcfg, train_ds, val_ds, test_ds, tasks, factory, F_in,
                            multitask_collate, device, False, dataset_name=cfg.DATASET,
                            run_tag=f"r{repeat}_f{fold}", seed=seed, log_mask=None,
                            encoder_config=enc_cfg, return_test_preds=True)
        seconds = time.time() - start
        preds = np.asarray(r["test_per_mol_preds"], dtype=float)
        if preds.shape != (len(test_df), len(tasks)):
            raise SystemExit(f"test predictions are {preds.shape}, expected "
                             f"{(len(test_df), len(tasks))}")

        out_dir.mkdir(parents=True, exist_ok=True)
        for j, t in enumerate(tasks):
            has = test_df[t].notna().to_numpy()
            pd.DataFrame(
                {
                    "method": cfg.FT_METHOD,
                    "endpoint": t,
                    "repeat": repeat,
                    "fold": fold,
                    cfg.ID_COL: test_df.loc[has, cfg.ID_COL].to_numpy(),
                    cfg.SMILES_COL: test_df.loc[has, cfg.SMILES_COL].to_numpy(),
                    "y_true": test_df.loc[has, t].to_numpy(),
                    "y_pred": preds[has, j],
                },
                columns=cfg.PRED_COLUMNS,
            ).to_csv(paths[t], index=False)

        diag = {"repeat": repeat, "fold": fold, "seed": seed, "seconds": seconds,
                "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(),
                "num_workers": int(hcfg.data.get("num_workers", 4)),
                "best_epoch": r.get("best_epoch"),
                "val_macro_mae": r["val"]["macro"], "test_macro_mae": r["test"]["macro"],
                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
                "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
                "overrides": args.set}
        if not args.set:
            cfg.diag_json(repeat, fold).write_text(json.dumps(diag, indent=1))
            pd.DataFrame([{"method": cfg.FT_METHOD, "endpoint": "all", "repeat": repeat,
                           "fold": fold, "seconds": seconds}]).to_csv(
                cfg.TIMING_CSV, mode="a", header=not cfg.TIMING_CSV.exists(), index=False)
        marker.unlink(missing_ok=True)
        print(f"--- r{repeat} f{fold}: best epoch {diag['best_epoch']}, val MAE "
              f"{diag['val_macro_mae']:.4f}, test MAE {diag['test_macro_mae']:.4f}, "
              f"{seconds / 60:.1f} min, {diag['peak_mem_gb']:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
