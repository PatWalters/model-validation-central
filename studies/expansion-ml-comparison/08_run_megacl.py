#!/usr/bin/env python
"""Step 8: the MEGA-CL arm of the comparison.

MEGA-CL (arXiv 2607.24314) is a graph contrastive-learning foundation model:
a five-layer GCNPlus backbone with external attention and mixed pooling,
pre-trained with NT-Xent on ~100 M molecules, then fine-tuned one endpoint at a
time. The pre-trained checkpoint lives in the authors' repository at
`checkpoints/model_best.pth`; 121 of the fine-tuning model's 129 tensors load
from it, the exceptions being the bond-direction embeddings their loader skips
on purpose and the fresh prediction head.

The model is single-target, so this arm is the direct counterpart of `lgbm` and
`chemprop_st`: one model per endpoint per fold, 9 x 25 = 225 runs, reading the
same `folds/st_<endpoint>_r{r}_f{f}.csv` files that the single-task chemprop arm
reads. Training rows, validation rows and the fixed test set are therefore
identical to every other method's.

Everything about the model and the optimisation is the authors' own: their
`FineTune.train` loop, their `config_finetune.yaml` hyperparameters (100 epochs,
batch 32, Adam at 5e-4 on the head and 1e-4 on the backbone), and their practice
of scoring the checkpoint with the best validation RMSE. Two things are
overridden, both of them about the experiment rather than the model:

  * the split. `MolTestDatasetWrapper` only offers random and scaffold splits,
    so `PresplitWrapper` below hands the trainer the fold's own `split` column.
  * the final evaluation. Their `evaluate_all_datasets` writes timestamped CSVs
    keyed on SMILES; this script predicts the test rows in file order instead,
    so predictions join back to the fold file by position and nothing depends on
    SMILES being unique.

One compatibility shim: MEGA-CL calls `mean_squared_error(..., squared=False)`,
which scikit-learn dropped in 1.6, so the name is wrapped in their module rather
than editing their source.

    MEGACL_HOME=~/software/MEGA-CL python 08_run_megacl.py                  # all 225
    python 08_run_megacl.py --endpoint LOG_MGMB --repeat 0 --fold 0         # smoke test
"""

import argparse
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

import config as cfg

METHOD = "megacl"

MEGACL_HOME = Path(
    os.environ.get("MEGACL_HOME", Path.home() / "software" / "MEGA-CL")
).expanduser().resolve()
CHECKPOINT = MEGACL_HOME / "checkpoints" / "model_best.pth"
FINETUNE_CONFIG = MEGACL_HOME / "config_finetune.yaml"

def run_log(tag: str) -> Path:
    """One log per worker, so several endpoint subsets can run side by side.

    A single fold model uses about 600 MB of GPU memory and leaves the card
    roughly 40 per cent idle, so the 225 runs are split across a handful of
    concurrent processes rather than queued behind one another.
    """
    return cfg.LOG_DIR / (f"megacl_run_{tag}.log" if tag else "megacl_run.log")

# The number of tensors the pre-trained checkpoint is expected to fill. Asserted
# on every run: `load_state_dict(..., strict=False)` in MEGA-CL's loader would
# otherwise let a renamed or mismatched checkpoint through in silence, and the
# arm would quietly become "the architecture trained from scratch".
EXPECTED_LOADED_TENSORS = 121


def megacl_modules():
    """Import MEGA-CL from its checkout, with the scikit-learn shim applied."""
    if not CHECKPOINT.exists():
        raise SystemExit(
            f"no pre-trained checkpoint at {CHECKPOINT}\n"
            f"set MEGACL_HOME to the MEGA-CL checkout (currently {MEGACL_HOME})"
        )
    if str(MEGACL_HOME) not in sys.path:
        sys.path.insert(0, str(MEGACL_HOME))

    import finetune
    from dataset.dataset_test import MolTestDataset
    from models.gcn_plus_atn_mixpool import GCNPlusAtnMixPool

    base_mse = finetune.mean_squared_error

    def mean_squared_error(y_true, y_pred, squared=True, **kwargs):
        value = base_mse(y_true, y_pred, **kwargs)
        return value if squared else float(np.sqrt(value))

    finetune.mean_squared_error = mean_squared_error
    return finetune, MolTestDataset, GCNPlusAtnMixPool


try:  # PyG moved Collater out of the package root after MEGA-CL was written
    from torch_geometric.loader import Collater
except ImportError:
    from torch_geometric.loader.dataloader import Collater


def collate(batch):
    """MEGA-CL's own collate: PyG batching plus the SMILES carried alongside."""
    data = Collater([], None)(batch)
    data.smiles = [item.smiles for item in batch]
    return data


class PresplitWrapper:
    """What `FineTune` asks of a dataset, answered from the fold's split column.

    MEGA-CL's own wrapper draws the split itself, at random or by scaffold. Here
    the split is already fixed: four fifths of the `ds == 'train'` molecules to
    train on, the held-out fifth to select the checkpoint, and the same
    `ds == 'test'` molecules every method is scored on.
    """

    def __init__(self, dataset, split, batch_size, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.index = {
            name: np.flatnonzero((split == name).to_numpy()).tolist()
            for name in ("train", "val", "test")
        }
        for name, idx in self.index.items():
            if not idx:
                raise SystemExit(f"fold has no {name} molecules")

    def _sampled(self, name):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.index[name]),
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=collate,
        )

    def ordered(self, name):
        """The rows of one split in file order, for prediction."""
        return DataLoader(
            Subset(self.dataset, self.index[name]),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate,
        )

    def get_data_loaders(self):
        return self._sampled("train"), self._sampled("val"), self._sampled("test")


def fold_config(endpoint: str, repeat: int, fold: int, opts) -> dict:
    """The authors' fine-tuning configuration, pointed at one fold."""
    config = yaml.safe_load(FINETUNE_CONFIG.read_text())
    config["task_name"] = f"megacl_{endpoint}"
    config["fine_tune_from"] = str(CHECKPOINT)
    config["gpu"] = opts.gpu
    config["repeat_runs"] = 1
    config["epochs"] = opts.epochs
    config["dataset"] = {
        "data_path": str(cfg.st_fold_input(endpoint, repeat, fold)),
        "num_workers": opts.num_workers,
        "splitting": "random",  # unused: PresplitWrapper supplies the split
        "target": endpoint,
        "task": "regression",
        "test_size": 0.1,  # unused
        "valid_size": 0.1,  # unused
    }
    return config


def check_pretrained(GCNPlusAtnMixPool, config) -> None:
    """Fail loudly if the checkpoint no longer fits the fine-tuning model."""
    model_config = dict(config["model"])
    model_config["pred_dim"] = 1
    model_config["num_bond_dir"] = 5
    reference = GCNPlusAtnMixPool(**model_config).state_dict()

    state = torch.load(CHECKPOINT, map_location="cpu")
    loaded = [
        key
        for key, value in state.items()
        if "bond_dir_emb.weight" not in key
        and key in reference
        and reference[key].shape == value.shape
    ]
    if len(loaded) != EXPECTED_LOADED_TENSORS:
        raise SystemExit(
            f"{CHECKPOINT} fills {len(loaded)} tensors, expected "
            f"{EXPECTED_LOADED_TENSORS} -- the checkpoint and the model no longer agree"
        )
    print(f"pre-trained checkpoint fills {len(loaded)}/{len(reference)} tensors")


def build_model(GCNPlusAtnMixPool, config, weights: Path, device):
    model_config = dict(config["model"])
    model_config["pred_dim"] = 1
    model_config["num_bond_dir"] = 5
    model = GCNPlusAtnMixPool(**model_config).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    return model


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for data in loader:
        data = data.to(device)
        _, pred = model(data)
        out.append(pred.detach().cpu().numpy().ravel())
    return np.concatenate(out)


def run_one(endpoint: str, repeat: int, fold: int, modules, opts) -> bool:
    """Fine-tune and predict one fold. True if it ran, False if it was skipped."""
    out_path = cfg.pred_csv(METHOD, endpoint, repeat, fold)
    if out_path.exists() and not opts.force:
        return False

    finetune, MolTestDataset, GCNPlusAtnMixPool = modules
    config = fold_config(endpoint, repeat, fold, opts)

    fold_df = pd.read_csv(config["dataset"]["data_path"])
    work_dir = cfg.SCRATCH_DIR / METHOD / f"{endpoint}_r{repeat}_f{fold}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # MEGA-CL writes its run directories, its TensorBoard events and its copy of
    # the configuration relative to the working directory, so each fold gets one
    # of its own and it is thrown away afterwards.
    shutil.copy(FINETUNE_CONFIG, work_dir / "config_finetune.yaml")
    cwd = Path.cwd()

    start = time.time()
    stdout = sys.stdout
    try:
        os.chdir(work_dir)
        torch.manual_seed(cfg.fold_seed(repeat, fold))
        np.random.seed(cfg.fold_seed(repeat, fold))

        dataset = MolTestDataset(
            data_path=config["dataset"]["data_path"],
            target=endpoint,
            task="regression",
        )
        if len(dataset) != len(fold_df):
            raise SystemExit(
                f"MEGA-CL read {len(dataset)} of {len(fold_df)} rows from "
                f"{config['dataset']['data_path']} -- the split would not line up"
            )

        wrapper = PresplitWrapper(
            dataset, fold_df[cfg.SPLIT_COL], config["batch_size"], opts.num_workers
        )
        trainer = finetune.FineTune(wrapper, config)
        # Their final pass writes timestamped, SMILES-keyed CSVs of every split.
        # This script predicts the test rows itself, in order, just below.
        trainer.evaluate_all_datasets = lambda *args, **kwargs: None

        with open(run_log(opts.tag), "a") as log:
            log.write(f"\n{'=' * 78}\n[{METHOD}] {endpoint} repeat {repeat} fold {fold}\n")
            log.flush()
            sys.stdout = log
            trainer.train()
            sys.stdout = stdout

        weights = Path(trainer.writer.log_dir) / "checkpoints" / "model.pth"
        if not weights.exists():
            raise SystemExit(f"no checkpoint written under {weights.parent}")
        model = build_model(GCNPlusAtnMixPool, config, weights, trainer.device)
        y_pred = predict(model, wrapper.ordered("test"), trainer.device)
    finally:
        sys.stdout = stdout
        os.chdir(cwd)
        shutil.rmtree(work_dir, ignore_errors=True)

    test_df = fold_df.iloc[wrapper.index["test"]]
    out = pd.DataFrame(
        {
            "method": METHOD,
            "endpoint": endpoint,
            "repeat": repeat,
            "fold": fold,
            cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(),
            cfg.SMILES_COL: test_df[cfg.SMILES_COL].to_numpy(),
            "y_true": test_df[endpoint].to_numpy(),
            "y_pred": y_pred,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[cfg.PRED_COLUMNS].to_csv(out_path, index=False)

    print(f"  [{METHOD}] {endpoint:<17} r{repeat} f{fold}  {time.time() - start:6.1f}s", flush=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true", help="refit folds that already have predictions")
    parser.add_argument("--gpu", default="cuda:0", help="'cuda:N', or 'cpu'")
    parser.add_argument("--num-workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--tag", default="", help="suffix for this worker's log file")
    parser.add_argument(
        "--epochs",
        type=int,
        default=yaml.safe_load(FINETUNE_CONFIG.read_text())["epochs"] if FINETUNE_CONFIG.exists() else 100,
        help="fine-tuning epochs (default: the authors' setting)",
    )
    opts = parser.parse_args()

    cfg.ensure_dirs()
    modules = megacl_modules()
    check_pretrained(modules[2], yaml.safe_load(FINETUNE_CONFIG.read_text()))

    jobs = [(e, r, f) for e in opts.endpoint for r in opts.repeat for f in opts.fold]
    print(f"{len(jobs)} MEGA-CL fold models, {opts.epochs} epochs each, log -> {run_log(opts.tag)}")

    start = time.time()
    ran = 0
    for endpoint, repeat, fold in jobs:
        ran += run_one(endpoint, repeat, fold, modules, opts)

    elapsed = timedelta(seconds=round(time.time() - start))
    print(f"\n{ran} fold models trained, {len(jobs) - ran} already present, {elapsed} elapsed")


if __name__ == "__main__":
    main()
