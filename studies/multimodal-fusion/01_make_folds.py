#!/usr/bin/env python
"""Step 1: clean the data set and lay out the 25 cross-validation folds.

Reads `expansion_log_scaled.csv`, drops the SMILES RDKit cannot parse and the rows
with no measured endpoint, then assigns folds: five repeats of a five-fold
`GroupKFold` over the `ds == 'train'` molecules, grouped by `cluster`. Each
(repeat, fold) pair defines one model:

    train  four fifths of the training molecules
    val    the held-out fifth, used only for early stopping / checkpoint selection
    test   every `ds == 'test'` molecule, the same for all 25 folds

Grouping by cluster keeps a chemotype from straddling the train/validation
boundary, which is what `UNIQUE/scripts/01_prepare_data.py` does with
`GroupShuffleSplit`.

Writes:

    data/master.csv                 the cleaned data set (the source of truth for
                                    the measured values -- chemprop's prediction
                                    files overwrite the target columns)
    folds/fold_assignments.csv      Name, repeat, fold for every training molecule
    folds/<group>_r{r}_f{f}.csv     75 chemprop input files, one per group and fold
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import GroupKFold

import config as cfg

RDLogger.DisableLog("rdApp.*")


def load_clean_data() -> pd.DataFrame:
    if not cfg.RAW_CSV.exists():
        raise SystemExit(f"data set not found: {cfg.RAW_CSV}")
    df = pd.read_csv(cfg.RAW_CSV)
    print(f"read {len(df)} rows from {cfg.RAW_CSV.name}")

    expected = [cfg.SMILES_COL, cfg.ID_COL, cfg.SET_COL, cfg.CLUSTER_COL, *cfg.TARGET_COLS]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SystemExit(f"missing expected columns: {missing}")

    # Drop anything RDKit cannot parse -- chemprop would fail on those rows.
    parsed = df[cfg.SMILES_COL].map(lambda s: Chem.MolFromSmiles(s) is not None)
    if (~parsed).any():
        print(f"dropping {(~parsed).sum()} unparseable SMILES")
        df = df[parsed].reset_index(drop=True)

    # Rows with no measured endpoint at all carry no training signal.
    all_nan = df[cfg.TARGET_COLS].isna().all(axis=1)
    if all_nan.any():
        print(f"dropping {all_nan.sum()} rows with no measured endpoint")
        df = df[~all_nan].reset_index(drop=True)

    if df[cfg.ID_COL].duplicated().any():
        # Every downstream join is on Name, so a duplicate would silently multiply rows.
        raise SystemExit(f"{df[cfg.ID_COL].duplicated().sum()} duplicate {cfg.ID_COL} values")

    df[cfg.SET_COL] = df[cfg.SET_COL].str.lower()
    return df


def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Long table of Name, repeat, fold -- the fold each training molecule is held out in."""
    train_df = df[df[cfg.SET_COL] == "train"]
    records = []
    for repeat in cfg.REPEATS:
        splitter = GroupKFold(
            n_splits=cfg.N_SPLITS, shuffle=True, random_state=cfg.RANDOM_SEED + repeat
        )
        fold_of = np.empty(len(train_df), dtype=int)
        for fold, (_, val_idx) in enumerate(
            splitter.split(train_df, groups=train_df[cfg.CLUSTER_COL])
        ):
            fold_of[val_idx] = fold
        records.append(
            pd.DataFrame(
                {cfg.ID_COL: train_df[cfg.ID_COL].to_numpy(), "repeat": repeat, "fold": fold_of}
            )
        )
    folds = pd.concat(records, ignore_index=True)

    sizes = folds.pivot_table(index="repeat", columns="fold", values=cfg.ID_COL, aggfunc="count")
    print("\nvalidation-fold sizes (molecules):")
    print(sizes.to_string())
    return folds


def check_folds(df: pd.DataFrame, folds: pd.DataFrame) -> None:
    """The invariants the whole comparison rests on."""
    n_train = (df[cfg.SET_COL] == "train").sum()
    cluster_of = df.set_index(cfg.ID_COL)[cfg.CLUSTER_COL]

    for repeat in cfg.REPEATS:
        rep = folds[folds["repeat"] == repeat]
        assert len(rep) == n_train, f"repeat {repeat}: {len(rep)} rows for {n_train} molecules"
        assert rep[cfg.ID_COL].is_unique, f"repeat {repeat}: a molecule lands in two folds"
        assert set(rep["fold"]) == set(cfg.FOLDS), f"repeat {repeat}: missing folds"
        # No cluster may straddle the train/validation boundary.
        clusters = cluster_of.loc[rep[cfg.ID_COL]].to_numpy()
        per_cluster_folds = pd.Series(rep["fold"].to_numpy()).groupby(clusters).nunique()
        split_clusters = per_cluster_folds[per_cluster_folds > 1]
        assert split_clusters.empty, f"repeat {repeat}: clusters split across folds:\n{split_clusters}"
    print("\nfold checks passed: folds partition the training set, no cluster is split")


def write_fold_inputs(df: pd.DataFrame, folds: pd.DataFrame) -> None:
    """One chemprop input file per endpoint group and fold, with a `split` column."""
    out_cols = [
        cfg.ID_COL,
        cfg.SMILES_COL,
        *cfg.TARGET_COLS,
        cfg.CLUSTER_COL,
        cfg.SET_COL,
        cfg.SPLIT_COL,
    ]
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    n_written = 0

    print("\nper-group fold files (molecules: train / val / test):")
    for group, targets in cfg.TARGET_GROUPS.items():
        # A molecule with none of a group's endpoints measured contributes no
        # gradient to that group's model, so it is left out of that group's file.
        keep = df[targets].notna().any(axis=1).to_numpy()
        shown = False
        for repeat in cfg.REPEATS:
            held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
            fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for test molecules
            for fold in cfg.FOLDS:
                split = np.where(is_test, "test", "train").astype(object)
                split[fold_of == fold] = "val"
                sub = df.assign(**{cfg.SPLIT_COL: split}).loc[keep, out_cols]
                sub.to_csv(cfg.fold_input(group, repeat, fold), index=False)
                n_written += 1
                if not shown:
                    counts = sub[cfg.SPLIT_COL].value_counts()
                    print(
                        f"  {group:<17} {len(sub):>5} molecules "
                        f"({counts.get('train', 0)} / {counts.get('val', 0)} / "
                        f"{counts.get('test', 0)})"
                    )
                    shown = True
    print(f"\nwrote {n_written} fold files to {cfg.FOLD_DIR}")


def main() -> None:
    cfg.ensure_dirs()
    df = load_clean_data()

    print(f"\n{cfg.SET_COL} column: " + ", ".join(
        f"{k} {v}" for k, v in df[cfg.SET_COL].value_counts().items()
    ))
    print("\nmeasured values per endpoint and set:")
    print(df.groupby(cfg.SET_COL)[cfg.TARGET_COLS].count().T.to_string())

    folds = assign_folds(df)
    check_folds(df, folds)

    df.to_csv(cfg.MASTER_CSV, index=False)
    folds.to_csv(cfg.FOLD_CSV, index=False)
    print(f"\nwrote {cfg.MASTER_CSV} and {cfg.FOLD_CSV}")

    write_fold_inputs(df, folds)


if __name__ == "__main__":
    main()
