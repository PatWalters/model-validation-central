#!/usr/bin/env python
"""Step 1b: fold files for the single-task ChemProp arm.

The multitask arms train one model per assay family, so part of their margin over
the single-task LightGBM baseline could be multitask transfer rather than the
architecture. This writes the inputs for the control that separates the two: the
same from-scratch D-MPNN, one model per endpoint.

One file per (endpoint, repeat, fold) holding only the molecules with that
endpoint measured -- exactly the rows the LightGBM model of that fold is fit on --
with the `split` column of that fold. The fold assignment is the one made in step
1, so every method still sees identical molecules in every fold.

Depends on data/master.csv and folds/fold_assignments.csv; writes 9 x 25 = 225
small files. Existing files are overwritten (the assignment is deterministic).

    python 01b_make_single_task_folds.py
"""

import numpy as np
import pandas as pd

import config as cfg


def main() -> None:
    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)

    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    n_written = 0

    print("molecules per endpoint (train / val / test), first fold of each:")
    for endpoint in cfg.TARGET_COLS:
        keep = df[endpoint].notna().to_numpy()
        out_cols = [cfg.ID_COL, cfg.SMILES_COL, endpoint, cfg.CLUSTER_COL, cfg.SET_COL, cfg.SPLIT_COL]
        shown = False
        for repeat in cfg.REPEATS:
            held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
            fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for test molecules
            for fold in cfg.FOLDS:
                split = np.where(is_test, "test", "train").astype(object)
                split[fold_of == fold] = "val"
                sub = df.assign(**{cfg.SPLIT_COL: split}).loc[keep, out_cols]
                sub.to_csv(cfg.st_fold_input(endpoint, repeat, fold), index=False)
                n_written += 1
                if not shown:
                    counts = sub[cfg.SPLIT_COL].value_counts()
                    print(
                        f"  {endpoint:<17} {len(sub):>5} "
                        f"({counts.get('train', 0)} / {counts.get('val', 0)} / "
                        f"{counts.get('test', 0)})"
                    )
                    shown = True

    print(f"\nwrote {n_written} single-task fold files to {cfg.FOLD_DIR}")


if __name__ == "__main__":
    main()
