#!/usr/bin/env python
"""Step 0: bring the folds, the embeddings and the imported arms over.

Causal-Chemprop is added to six arms that already exist. Re-running those six
would answer a different question -- two runs differ by their seeds even when
nothing else does -- so their predictions are copied rather than recomputed,
together with the molecule table and the fold assignments that define what
"the same fold" means.

What is copied, per data set:

  data/<ds>/master.csv              the cleaned molecules
  folds/<ds>/fold_assignments.csv   the 5 x 5 GroupKFold assignment
  data/<ds>/monroe_embeddings.npz   Monroe's frozen 720-d embedding, for the
                                    anchors-in-context arm
  predictions/<ds>/<method>/*.csv   lgbm, chemprop_st, chemprop, chemeleon and
                                    monroe from ADME_SOURCE_REPO; atomjepa_ft
                                    from ATOMJEPA_REPO

The copies are verified rather than trusted: every arm has to have the full set
of files for the data set, and the Atom-JEPA study has to have been run on the
same molecules and folds, or the import fails and says why.

    python 00_import_baselines.py
    ADME_DATASET=biogen python 00_import_baselines.py
    # on apollo the atom-jepa sweep carries both, byte-identical to MOLPAIR_SOURCE
    # and the Monroe embeddings live with the source comparison's own sweep
    ADME_SOURCE_REPO=~/ATOM_JEPA ATOMJEPA_REPO=~/ATOM_JEPA \
      MONROE_EMBED_REPO=~/EXPANSION_ML_COMPARISON python 00_import_baselines.py
"""

import argparse
import filecmp
import shutil

import numpy as np
import pandas as pd

import config as cfg


def copy_file(src, dst, force: bool) -> None:
    if dst.exists() and not force:
        print(f"  {dst.relative_to(cfg.PROJECT_DIR)} already here")
        return
    if not src.exists():
        raise SystemExit(f"{src} not found")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  {dst.relative_to(cfg.PROJECT_DIR)} <- {src}")


def expected_files(single_task: bool) -> list[str]:
    units = cfg.TARGET_COLS if single_task else cfg.GROUPS
    return [f"{unit}_r{r}_f{f}.csv" for unit in units for r in cfg.REPEATS for f in cfg.FOLDS]


def import_arm(method: str, src_dir, single_task: bool, force: bool) -> None:
    dst_dir = cfg.PRED_DIR / method
    dst_dir.mkdir(parents=True, exist_ok=True)
    wanted = expected_files(single_task)
    missing = [name for name in wanted if not (src_dir / name).exists()]
    if missing:
        raise SystemExit(f"{method}: {len(missing)} of {len(wanted)} prediction files are "
                         f"missing from {src_dir}, starting with {missing[0]}")
    copied = 0
    for name in wanted:
        dst = dst_dir / name
        if dst.exists() and not force:
            continue
        shutil.copyfile(src_dir / name, dst)
        copied += 1
    print(f"  {method:<12} {len(wanted)} files ({copied} copied)")


def check_alignment() -> None:
    master = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    if master[cfg.ID_COL].duplicated().any():
        raise SystemExit(f"{cfg.MASTER_CSV.name} has duplicate {cfg.ID_COL} values")
    train = set(master.loc[master[cfg.SET_COL] == "train", cfg.ID_COL])
    for repeat, group in folds.groupby("repeat"):
        if set(group[cfg.ID_COL]) != train:
            raise SystemExit(f"repeat {repeat} does not assign exactly the training molecules")
    emb = np.load(cfg.MONROE_NPZ, allow_pickle=False)
    if not np.array_equal(emb["names"].astype(str), master[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(f"{cfg.MONROE_NPZ.name} does not line up with {cfg.MASTER_CSV.name}")
    print(f"\n{len(master):,} molecules: {len(train):,} train, "
          f"{int((master[cfg.SET_COL] == 'test').sum()):,} test; Monroe {emb['X'].shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()
    src = cfg.SOURCE_REPO
    print(f"importing {cfg.ACTIVE.label} from {src} and {cfg.ATOMJEPA_REPO}")
    copy_file(src / "data" / cfg.DATASET / "master.csv", cfg.MASTER_CSV, args.force)
    copy_file(src / "folds" / cfg.DATASET / "fold_assignments.csv", cfg.FOLD_CSV, args.force)
    # The Atom-JEPA sweep has to be on the same molecules and folds.
    for name in ("data/{ds}/master.csv", "folds/{ds}/fold_assignments.csv"):
        theirs = cfg.ATOMJEPA_REPO / name.format(ds=cfg.DATASET)
        ours = cfg.PROJECT_DIR / name.format(ds=cfg.DATASET)
        if not filecmp.cmp(theirs, ours, shallow=False):
            raise SystemExit(f"{theirs} differs from {ours} -- not the same protocol")
    copy_file(cfg.MONROE_EMBED_REPO / "data" / cfg.DATASET / "monroe_embeddings.npz",
              cfg.MONROE_NPZ, args.force)

    print("\nimported predictions:")
    for method, single in cfg.BASELINE_IS_SINGLE_TASK.items():
        import_arm(method, src / "predictions" / cfg.DATASET / method, single, args.force)
    import_arm(cfg.ATOMJEPA_METHOD,
               cfg.ATOMJEPA_REPO / "predictions" / cfg.DATASET / cfg.ATOMJEPA_METHOD,
               True, args.force)
    check_alignment()


if __name__ == "__main__":
    main()
