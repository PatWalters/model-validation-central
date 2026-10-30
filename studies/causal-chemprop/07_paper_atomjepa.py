#!/usr/bin/env python
"""Step 7: fine-tuned Atom-JEPA on the paper's eleven tasks.

The `atomjepa_ft` arm of the atom-jepa study, pointed at the kinase files: the
authors' train_multitask at commit bab0853, their Biogen fine-tuning recipe
with the same overrides that study used (none of which changes what is
trained), the released `molecules` encoder, ten conformers per molecule and the
conformer-averaged test prediction. One task, pIC50, so the multitask head is a
single 64-unit head.

The training and validation molecules are exactly those the authors' Chemprop
was trained and early-stopped on (01_reproduce_paper.py writes them), so the
two encoders are selected on the same data. Every row of each counterfactual
file is predicted, the anchor included; 08_paper_report.py forms the
anchor-offset twin from that. There is no retrained-with-the-anchor version:
one molecule added to a 600-molecule fine-tune is the paper's retrained
baseline, and the offset is the cheaper and, in the paper's own Table 1, the
more informative of the two.

Runs in the `atomjepa` environment on the GPU.

    python 07_paper_atomjepa.py
    python 07_paper_atomjepa.py --task ABL1_0 --set finetune.epochs=2   # smoke test
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

# e3nn loads its bundled Wigner constants with a bare torch.load, which torch
# >= 2.6 refuses under weights_only. The file ships with the library.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

OVERRIDES = [
    "++wandb.enabled=false",
    "++finetune.log_transform_tasks=[]",
    "++finetune.report_scale=null",
    "++finetune.conformer_eval_mode=ensemble",
    "++finetune.save_best=false",
    "++finetune.save_last=false",
    "++finetune.seeds=[0]",
    "++finetune.plot_every=0",
    # Kendall uncertainty weighting needs two or more tasks and asserts so. With
    # one, it would only learn a scale on the single loss, so it is turned off.
    "++finetune.mtl_loss=false",
]
TARGET = "pIC50"


def machine_overrides() -> list[str]:
    """Execution settings that do not change what is trained.

    spark (a GB10) deadlocks when the loader forks workers after cuEquivariance
    has started its threads; the atom-jepa study ran it with 0, and so does
    this. That changes which random conformer a step draws, nothing else.
    """
    workers = os.environ.get("ATOMJEPA_NUM_WORKERS")
    return [] if workers is None else [f"++data.num_workers={int(workers)}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", nargs="+", default=cfg.PAPER_TASKS, choices=cfg.PAPER_TASKS)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    home = Path(os.environ.get("ATOMJEPA_HOME", Path.home() / "software" / "atom-jepa")).expanduser()
    sys.path.insert(0, str(home))
    import torch
    from hydra import compose, initialize_config_dir
    from data.datasets.admet.admet_conformers import build_or_load_conformers
    from data.datasets.admet.multitask_finetune import MultiTaskFinetuneDataset, multitask_collate
    from finetuning.admet.training.train import train_multitask
    from finetuning.admet.training.utils import (
        build_equiformer_multitask_factory, load_pretrained_encoder, resolve_device)

    work = cfg.PROJECT_DIR / "work"
    out_dir = (cfg.PROJECT_DIR / "predictions" / "paper" / "atomjepa_ft" if not args.set
               else work / "scratch_atomjepa")
    conf_dir = work / "conformers_paper"
    conf_dir.mkdir(parents=True, exist_ok=True)

    with initialize_config_dir(config_dir=str(home / "conf"), version_base=None):
        hcfg = compose(config_name="finetune_biogen_adme", overrides=OVERRIDES + [
            "++finetune.ckpt_path=molecules",
            f"++misc.checkpoint_dir={work / 'atomjepa_ckpt'}",
        ] + machine_overrides() + args.set)
    ftc = hcfg.finetune
    device = torch.device(resolve_device(hcfg.misc.device))
    _, eqv3_cfg, ckpt = load_pretrained_encoder("molecules", device, load_weights=False)
    factory, F_in, enc_cfg = build_equiformer_multitask_factory(
        hcfg, eqv3_cfg, ckpt["context_encoder"], device)
    max_z = int(getattr(eqv3_cfg, "max_num_elements", 128))

    for i, task in enumerate(cfg.PAPER_TASKS):
        if task not in args.task:
            continue
        out = out_dir / f"{task}.csv"
        if out.exists() and not args.force:
            continue
        split = pd.read_csv(work / "paper_splits" / f"{task}.csv")
        test = pd.read_csv(cfg.UPSTREAM_HOME / "data" /
                           f"{task.split('_')[0].lower()}_counterfactual_{task.split('_')[1]}.csv")
        frame = lambda d: pd.DataFrame({cfg.SMILES_COL: d["SMI"].to_numpy(), TARGET: d["logS"].to_numpy()})
        smiles = list(dict.fromkeys(list(split["SMI"]) + list(test["SMI"])))
        conformers = build_or_load_conformers(f"paper_{task}", smiles, str(conf_dir), seed=42,
                                              n_conf=10, strip_fragments=True, n_workers=-2,
                                              prune_rms=-1.0)
        failed = {s for s in smiles if conformers.get(s) is None}
        if failed & set(test["SMI"]):
            raise SystemExit(f"{task}: {len(failed & set(test['SMI']))} test molecules have no "
                             "conformer -- Atom-JEPA could not be scored on every molecule")
        # A training or validation molecule RDKit cannot embed is left out of
        # this arm's fit only, and counted. (AURKA_0: five bridged-bicyclic
        # stereoisomers, out of 810.) Every arm is still scored on every test molecule.
        n_dropped = int(split["SMI"].isin(failed).sum())
        split = split[~split["SMI"].isin(failed)]

        def make_ds(d, tag, sample):
            ds = MultiTaskFinetuneDataset(d, conformers, eqv3_cfg.max_radius, [TARGET], max_z=max_z,
                                          num_conformers=int(ftc.num_conformers),
                                          sample_conformers=sample, smiles_col=cfg.SMILES_COL,
                                          tag=tag, features=None)
            if len(ds.keep_index) != len(d):
                raise SystemExit(f"{tag}: molecules dropped by the dataset")
            return ds

        train_ds = make_ds(frame(split[split["split"] == "train"]).reset_index(drop=True), f"{task}/train", True)
        val_ds = make_ds(frame(split[split["split"] == "val"]).reset_index(drop=True), f"{task}/val", False)
        test_ds = make_ds(frame(test).reset_index(drop=True), f"{task}/test", False)
        seed = cfg.RANDOM_SEED + i
        start = time.time()
        r = train_multitask(hcfg, train_ds, val_ds, test_ds, [TARGET], factory, F_in,
                            multitask_collate, device, False, dataset_name=f"paper_{task}",
                            run_tag=task, seed=seed, log_mask=None, encoder_config=enc_cfg,
                            return_test_preds=True)
        preds = np.asarray(r["test_per_mol_preds"], dtype=float).reshape(len(test), -1)[:, 0]
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "task": task, "method": "atomjepa_ft", cfg.SMILES_COL: test["SMI"],
            "role": ["anchor"] + ["scored"] * (len(test) - 1),
            "y_true": test["logS"], "y_pred": preds,
        }).to_csv(out, index=False)
        print(f"{task}: best epoch {r.get('best_epoch')}, {(time.time() - start) / 60:.1f} min, "
              f"{n_dropped} training molecules without a conformer", flush=True)
        diag = out_dir.parent.parent.parent / "results" / "paper" / "atomjepa_fits.csv"
        diag.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"task": task, "best_epoch": r.get("best_epoch"),
                       "minutes": (time.time() - start) / 60, "dropped_no_conformer": n_dropped}]
                     ).to_csv(diag, mode="a", header=not diag.exists(), index=False)


if __name__ == "__main__":
    main()
