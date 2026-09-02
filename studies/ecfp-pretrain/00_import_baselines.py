#!/usr/bin/env python
"""Step 0: bring the folds and the four baseline arms over from the source repo.

This comparison adds one method to four that already exist. Re-running those four
would answer a different question -- two runs of ChemProp differ by their seeds
even when nothing else does -- so their predictions are copied rather than
recomputed, together with the molecule table and the fold assignments that define
what "the same fold" means.

What is copied, per data set:

  data/<ds>/master.csv              the cleaned molecules, in the row order every
                                    cached embedding is indexed by
  folds/<ds>/fold_assignments.csv   which of the 5 folds each training molecule is
                                    held out in, for each of the 5 repeats
  predictions/<ds>/<method>/*.csv   the tidy per-fold predictions of lgbm,
                                    chemprop_st, chemprop and chemeleon

After this runs, nothing else in this repository reads the source repository. The
copies are verified rather than trusted: every arm has to have the full set of
files for the data set, or the import fails and says which are missing.

    python 00_import_baselines.py
    ADME_DATASET=biogen python 00_import_baselines.py
    ADME_SOURCE_REPO=~/expansion-ml-comparison python 00_import_baselines.py
"""

import argparse
import shutil

import pandas as pd

import config as cfg


def copy_file(src, dst, force: bool) -> None:
    if dst.exists() and not force:
        print(f"  {dst.relative_to(cfg.PROJECT_DIR)} already here")
        return
    if not src.exists():
        raise SystemExit(f"{src} not found -- is ADME_SOURCE_REPO right?")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  {dst.relative_to(cfg.PROJECT_DIR)} <- {src}")


def expected_files(method: str) -> list[str]:
    """The prediction file names one baseline arm should have for this data set.

    Single-task arms fit one model per endpoint, multitask arms one per assay
    family, so the two have different file names for the same 25 folds.
    """
    units = cfg.TARGET_COLS if cfg.BASELINE_IS_SINGLE_TASK[method] else cfg.GROUPS
    return [
        f"{unit}_r{repeat}_f{fold}.csv"
        for unit in units
        for repeat in cfg.REPEATS
        for fold in cfg.FOLDS
    ]


def import_predictions(force: bool) -> None:
    source_root = cfg.SOURCE_REPO / "predictions" / cfg.DATASET
    if not source_root.exists():
        raise SystemExit(f"{source_root} not found -- is ADME_SOURCE_REPO right?")

    for method in cfg.BASELINE_METHODS:
        src_dir = source_root / method
        dst_dir = cfg.PRED_DIR / method
        dst_dir.mkdir(parents=True, exist_ok=True)

        wanted = expected_files(method)
        missing = [name for name in wanted if not (src_dir / name).exists()]
        if missing:
            raise SystemExit(
                f"{method}: {len(missing)} of {len(wanted)} prediction files are "
                f"missing from {src_dir}, starting with {missing[0]}"
            )

        copied = 0
        for name in wanted:
            dst = dst_dir / name
            if dst.exists() and not force:
                continue
            shutil.copyfile(src_dir / name, dst)
            copied += 1
        print(f"  {method:<12} {len(wanted)} folds ({copied} copied, "
              f"{len(wanted) - copied} already here)")


def check_alignment() -> None:
    """The molecule table and the folds have to describe the same molecules."""
    master = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)

    missing_cols = [c for c in (cfg.ID_COL, cfg.SMILES_COL, cfg.SET_COL) if c not in master]
    if missing_cols:
        raise SystemExit(f"{cfg.MASTER_CSV.name} has no {', '.join(missing_cols)} column")
    if master[cfg.ID_COL].duplicated().any():
        raise SystemExit(f"{cfg.MASTER_CSV.name} has duplicate {cfg.ID_COL} values")

    train = set(master.loc[master[cfg.SET_COL] == "train", cfg.ID_COL])
    for repeat, group in folds.groupby("repeat"):
        assigned = set(group[cfg.ID_COL])
        if assigned != train:
            raise SystemExit(
                f"repeat {repeat} assigns {len(assigned)} molecules to folds but "
                f"{cfg.MASTER_CSV.name} has {len(train)} training molecules"
            )

    n_test = int((master[cfg.SET_COL] == "test").sum())
    print(f"\n{len(master):,} molecules: {len(train):,} train, {n_test:,} test, "
          f"{folds['repeat'].nunique()} repeats x {folds['fold'].nunique()} folds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true",
                        help="overwrite files that are already here")
    args = parser.parse_args()

    cfg.ensure_dirs()
    print(f"importing the {cfg.ACTIVE.label} set from {cfg.SOURCE_REPO}")

    copy_file(cfg.SOURCE_REPO / "data" / cfg.DATASET / "master.csv",
              cfg.MASTER_CSV, args.force)
    copy_file(cfg.SOURCE_REPO / "folds" / cfg.DATASET / "fold_assignments.csv",
              cfg.FOLD_CSV, args.force)

    print("\nbaseline predictions:")
    import_predictions(args.force)
    check_alignment()


if __name__ == "__main__":
    main()
