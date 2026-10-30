#!/usr/bin/env python
"""Step 5: the arms the paper did not run, on the paper's own eleven tasks.

Same training files, same held-out scaffold groups, same anchors as Table 1.
For each task the training set is the file the authors' Chemprop was trained
on (TRAINING_FPATH[i] in causal_utils), and the anchor-aware version of an arm
is trained on their trainmod file -- that training set plus the anchor, which
is what their "retrained Chemprop" saw. So the anchor-aware arms get exactly
the information the paper's retrained baseline got.

  monroe           Monroe's frozen embedding read by TabPFN 3, as in the
                   protocol half: the monroe arm's call, unchanged
  monroe_anchor    the same, with the anchor in the in-context training set
  lgbm             LightGBM on Morgan counts, the reference arm's settings
  lgbm_anchor      the same, refitted with the anchor

Every row of each counterfactual file is predicted, the anchor included, so
06_paper_report.py can also form each arm's anchor-offset correction. Runs in
the `monroe` environment, which has Monroe, tabpfn 8.x (TabPFN 3), LightGBM
and RDKit.

    python 05_paper_arms.py
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import config as cfg

# causal_utils' file lists, by task index, copied rather than imported: the
# module pulls in chemprop, dagma, evomol and a deterministic-algorithms switch
# that the monroe environment has no use for.
FILES = {
    "ABL1": (["abl1.csv", "abl1_train_1.csv", "abl1_train_2.csv", "abl1_train_3.csv"], 4),
    "AURKA": (["aurka.csv", "aurka_train_1.csv", "aurka_train_2.csv", "aurka_train_3.csv"], 4),
    "SRC": (["src_train_0.csv", "src_train_1.csv", "src_train_2.csv"], 3),
}


def task_files(task: str):
    target, k = task.split("_")
    k = int(k)
    data = cfg.UPSTREAM_HOME / "data"
    t = target.lower()
    return (data / FILES[target][0][k], data / f"{t}_trainmod_{k}.csv",
            data / f"{t}_counterfactual_{k}.csv")


def monroe_embedder():
    import torch
    sys.path.insert(0, str(cfg.MONROE_HOME))
    from monroe.eval.embed import embed_smiles
    from monroe.model.ckpt import load_ckpt
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = load_ckpt(str(cfg.MONROE_HOME / "checkpoint")).to(device).eval()

    def embed(smiles):
        lookup = embed_smiles(list(pd.unique(np.asarray(smiles))), enc, device=device,
                              batch_size=64, n_workers=None)
        missing = [s for s in smiles if s not in lookup]
        if missing:
            raise SystemExit(f"Monroe could not featurize {len(missing)} molecules, e.g. {missing[0]}")
        return np.stack([lookup[s] for s in smiles]).astype(np.float32)
    return embed


def morgan_counts(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.stack([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(cfg.MONROE_HOME))
    from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn
    from lightgbm import LGBMRegressor

    embed = monroe_embedder()
    out_root = cfg.PROJECT_DIR / "predictions" / "paper"
    for i, task in enumerate(cfg.PAPER_TASKS):
        train_f, trainmod_f, test_f = task_files(task)
        train, trainmod, test = (pd.read_csv(p) for p in (train_f, trainmod_f, test_f))
        anchor = test["SMI"].iloc[0]
        if len(trainmod) != len(train) + 1 or trainmod["SMI"].iloc[-1] != anchor:
            # Not every trainmod file appends the anchor last; check membership instead.
            if anchor not in set(trainmod["SMI"]) or len(trainmod) != len(train) + 1:
                raise SystemExit(f"{trainmod_f.name} is not {train_f.name} plus the anchor")
        seed = cfg.RANDOM_SEED + i
        fits = {"": train, "_anchor": trainmod}
        X_test_m = embed(test["SMI"].tolist())
        X_test_f = morgan_counts(test["SMI"])
        for suffix, fit in fits.items():
            for arm in ("monroe", "lgbm"):
                out = out_root / f"{arm}{suffix}" / f"{task}.csv"
                if out.exists() and not args.force:
                    continue
                if arm == "monroe":
                    pred = fit_predict_tabpfn(embed(fit["SMI"].tolist()), fit["logS"].to_numpy(),
                                              X_test_m, is_classification=False,
                                              ensemble_specs=default_ensemble_specs(),
                                              seed=seed, output_type="mean")
                else:
                    model = LGBMRegressor(n_jobs=-1, verbose=-1, random_state=seed)
                    model.fit(morgan_counts(fit["SMI"]), fit["logS"].to_numpy())
                    pred = model.predict(X_test_f)
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({
                    "task": task, "method": f"{arm}{suffix}", cfg.SMILES_COL: test["SMI"],
                    "role": ["anchor"] + ["scored"] * (len(test) - 1),
                    "y_true": test["logS"], "y_pred": np.asarray(pred, dtype=float),
                }).to_csv(out, index=False)
                print(f"{task:8s} {arm + suffix:14s} fit {len(fit)}", flush=True)


if __name__ == "__main__":
    main()
