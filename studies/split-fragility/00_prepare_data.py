#!/usr/bin/env python
"""Step 0: build the master table, the fold assignments and the fingerprints.

The upstream repository (github.com/leekisung1991/bioactivity_prediction, the
release accompanying doi:10.1002/chem.71208) ships, for each target and each of
five folds, six splitting schemes as explicit `train` / `val` / `test` index
lists into that target's raw CSV. Those index lists *are* the protocol, so this
step does not split anything. It reads them, checks them, and writes them out in
long form.

Which copy of them is read, and why, is `config.read_split`'s docstring: the
release ships the splits twice, the two disagree on three of the six schemes,
and only one of the two reproduces the paper's own published numbers.

What comes out:

  data/master.csv             one row per (target, molecule), with the upstream
                              row number kept as `idx` so the split files stay
                              readable against it
  folds/fold_assignments.csv  (target, split, fold, idx, split_role) long form
  folds/tid_*_*_f*.csv        one chemprop input per fold, with a `split` column
  folds/test_tid_*_*_f*.csv   Name + SMILES of each fold's test molecules
  data/ecfp4_bits.npy         ECFP4 bit vectors, master.csv row order  (k-NN)
  data/morgan_counts.npy      Morgan count fingerprints, same order     (LightGBM)

Every molecule is parsed once and every fingerprint computed once, because a
molecule belongs to exactly one target but appears in six schemes x five folds.

    python 00_prepare_data.py
    python 00_prepare_data.py --force   # recompute the fingerprints
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

import config as cfg

RDLogger.DisableLog("rdApp.*")


def load_target(target: int) -> pd.DataFrame:
    """One target's molecules, with the upstream row number kept as `idx`.

    `idx` is the index the split files use. Nothing is dropped or reordered
    here: a molecule RDKit cannot parse is kept with `ok` false and reported, so
    that a fold which would be affected fails loudly later rather than quietly
    training on fewer molecules than another arm.
    """
    raw = pd.read_csv(cfg.upstream_csv(target))
    missing = {cfg.SMILES_COL, cfg.Y_COL, cfg.ID_COL} - set(raw.columns)
    if missing:
        raise SystemExit(f"tid_{target}.csv is missing {sorted(missing)}")

    df = pd.DataFrame(
        {
            "target": target,
            "idx": np.arange(len(raw)),
            cfg.OUT_ID_COL: raw[cfg.ID_COL].astype(str),
            cfg.OUT_SMILES_COL: raw[cfg.SMILES_COL].astype(str),
            "y": raw[cfg.Y_COL].astype(float),
            "year": raw.get(cfg.YEAR_COL),
        }
    )
    df["ok"] = [Chem.MolFromSmiles(smi) is not None for smi in df[cfg.OUT_SMILES_COL]]
    return df


def build_master() -> pd.DataFrame:
    frames = [load_target(target) for target in cfg.TARGETS]
    master = pd.concat(frames, ignore_index=True)

    bad = int((~master["ok"]).sum())
    if bad:
        print(f"warning: {bad} SMILES do not parse and are kept with ok=False")
    if master["y"].isna().any():
        raise SystemExit("missing pIC50 values in the upstream CSVs")

    # A molecule is addressed by (target, idx) everywhere downstream. chembl_cid
    # is not unique across targets -- the same compound is assayed against
    # several of them -- so `Name` is only ever used for display and for joining
    # chemprop's output back within a single fold, where it is unique.
    dup = master.duplicated(["target", "idx"]).sum()
    if dup:
        raise SystemExit(f"{dup} duplicated (target, idx) pairs in the master table")
    return master


def build_folds(master: pd.DataFrame) -> pd.DataFrame:
    """The upstream index lists, checked and flattened to long form.

    The three roles of a scheme must be disjoint, and every index must be a real
    row of that target's CSV. They need not *cover* it: the diverse split is
    drawn from a MaxMin subset of 25% of the data, so most molecules have no role
    in it at all, and the time split's year cutoff leaves the training pool's
    unused remainder out as well.
    """
    sizes = master.groupby("target").size().to_dict()
    records = []

    for target in cfg.TARGETS:
        n = sizes[target]
        for fold in cfg.FOLDS:
            schemes = cfg.read_split(target, fold)   # checks the two copies agree
            unknown = set(schemes) - set(cfg.SPLITS)
            if unknown:
                raise SystemExit(
                    f"tid_{target}_fold_{fold}.json has unexpected schemes {sorted(unknown)}"
                )
            for split in cfg.SPLITS:
                if split not in schemes:
                    raise SystemExit(f"tid_{target}_fold_{fold}.json has no {split!r} scheme")
                roles = schemes[split]
                seen: set[int] = set()
                for role in ("train", "val", "test"):
                    idxs = roles.get(role)
                    if not idxs:
                        raise SystemExit(
                            f"tid_{target} {split} fold {fold}: {role!r} is empty"
                        )
                    idxs = [int(i) for i in idxs]
                    if min(idxs) < 0 or max(idxs) >= n:
                        raise SystemExit(
                            f"tid_{target} {split} fold {fold} {role}: index out of range"
                        )
                    if len(set(idxs)) != len(idxs):
                        raise SystemExit(
                            f"tid_{target} {split} fold {fold} {role}: duplicated indices"
                        )
                    overlap = seen & set(idxs)
                    if overlap:
                        raise SystemExit(
                            f"tid_{target} {split} fold {fold}: {role} overlaps an earlier "
                            f"role on {len(overlap)} molecules"
                        )
                    seen |= set(idxs)
                    records += [
                        {"target": target, "split": split, "fold": fold,
                         "idx": i, "split_role": role}
                        for i in idxs
                    ]

    folds = pd.DataFrame(records)
    print(f"read {len(cfg.TARGETS) * len(cfg.FOLDS)} split files "
          f"-> {len(folds):,} (target, split, fold, molecule) roles")
    return folds


def check_test_coverage(folds: pd.DataFrame) -> None:
    """Say, per scheme, how much of each data set the five test sets cover.

    The four cluster-based schemes rotate a fifth of the data through the test
    set and so cover all of it exactly once. Diverse covers a quarter, and Time
    overlaps itself because each fold picks its own year cutoff. Printed rather
    than asserted, because all three behaviours are the upstream design.
    """
    rows = []
    for split, g in folds[folds["split_role"] == "test"].groupby("split"):
        per_target = g.groupby("target")["idx"]
        rows.append(
            {
                "split": split,
                "test rows/fold": round(len(g) / (len(cfg.TARGETS) * len(cfg.FOLDS))),
                "distinct molecules tested": per_target.nunique().sum(),
                "test assignments": len(g),
            }
        )
    table = pd.DataFrame(rows).set_index("split").reindex(cfg.SPLITS)
    print("\ntest-set coverage over the five folds:")
    print(table.to_string())


def write_fold_inputs(master: pd.DataFrame, folds: pd.DataFrame) -> None:
    """One chemprop input per fold, plus the Name + SMILES file it predicts on.

    The prediction file carries only those two columns: `chemprop predict` copies
    its input through to the output, and a file holding the target column would
    come back with the measured values overwritten by the predictions.
    """
    keyed = master.set_index(["target", "idx"])
    written = 0
    for (target, split, fold), g in folds.groupby(["target", "split", "fold"]):
        path = cfg.fold_input(target, split, fold)
        rows = keyed.loc[list(zip(g["target"], g["idx"]))].copy()
        rows[cfg.SPLIT_COL] = g["split_role"].to_numpy()
        rows = rows.reset_index()[
            [cfg.OUT_ID_COL, cfg.OUT_SMILES_COL, "y", cfg.SPLIT_COL, "target", "idx"]
        ]
        rows.to_csv(path, index=False)

        test = rows.loc[rows[cfg.SPLIT_COL] == "test", [cfg.OUT_ID_COL, cfg.OUT_SMILES_COL]]
        test.to_csv(cfg.test_input(target, split, fold), index=False)
        written += 1
    print(f"\nwrote {written} fold inputs and {written} test inputs to {cfg.FOLD_DIR}")


def fingerprints(master: pd.DataFrame, force: bool) -> None:
    """ECFP4 bit vectors and Morgan counts, both in master.csv row order.

    The same generator serves both: k-NN wants the bit vector its Jaccard
    distance is defined on, LightGBM the counts this framework's baseline uses.
    Unparseable molecules get an all-zero row; `ok` in master.csv is what the
    runners check.
    """
    have = cfg.ECFP_BITS_NPY.exists() and cfg.MORGAN_COUNTS_NPY.exists()
    if have and not force:
        bits = np.load(cfg.ECFP_BITS_NPY, mmap_mode="r")
        if len(bits) == len(master):
            print(f"\nfingerprints already cached {bits.shape}")
            return
        print("\ncached fingerprints do not match the master table, recomputing")

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=cfg.FP_RADIUS, fpSize=cfg.FP_SIZE)
    start = time.time()
    bits = np.zeros((len(master), cfg.FP_SIZE), dtype=np.uint8)
    counts = np.zeros((len(master), cfg.FP_SIZE), dtype=np.float32)
    for row, smi in enumerate(master[cfg.OUT_SMILES_COL]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bits[row] = gen.GetFingerprintAsNumPy(mol)
        counts[row] = gen.GetCountFingerprintAsNumPy(mol)
    np.save(cfg.ECFP_BITS_NPY, bits)
    np.save(cfg.MORGAN_COUNTS_NPY, counts)
    print(f"\ncomputed fingerprints {bits.shape} in {time.time() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute the fingerprints")
    args = parser.parse_args()

    cfg.ensure_dirs()

    master = build_master()
    master.to_csv(cfg.MASTER_CSV, index=False)
    print(f"wrote {cfg.MASTER_CSV.name}: {len(master):,} rows over {master['target'].nunique()} targets")
    print(master.groupby("target").agg(
        n=("y", "size"), mean_pIC50=("y", "mean"), sd=("y", "std")
    ).round(2).to_string())

    folds = build_folds(master)
    folds.to_csv(cfg.FOLD_CSV, index=False)

    check_test_coverage(folds)
    write_fold_inputs(master, folds)
    fingerprints(master, args.force)


if __name__ == "__main__":
    main()
