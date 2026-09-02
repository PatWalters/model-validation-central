#!/usr/bin/env python
"""Step 11: is this test set inside Mol-JEPA's pre-training data?

Several of the methods here were published by groups that used these
measurements as an evaluation set, so the obvious question is whether any of
these molecules were also seen during pre-training. Mol-JEPA is the one where
the question can actually be answered: the authors released the whole
pre-training table, 4.66 M rows, as `metadata.csv` in their HuggingFace dataset.
It even has a column for every endpoint of both data sets used here.

This script joins our molecules to it on InChIKey and reports three things:
an exact-key overlap, a looser overlap on the 14-character connectivity block
(so a difference in protonation or stereo layer cannot hide a match, and their
pipeline does normalise protonation to pH 7), and whether any row in the table
carries a value for one of the active data set's endpoints.

    curl -L -o metadata.csv https://huggingface.co/datasets/Flogrammer/Mol-JEPA-dataset/resolve/main/metadata.csv
    python 11_check_pretraining_overlap.py metadata.csv
    ADME_DATASET=biogen python 11_check_pretraining_overlap.py metadata.csv

The file is about 2 GB, so it is streamed in chunks and only the handful of
columns that matter are parsed.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

import config as cfg

RDLogger.DisableLog("rdApp.*")

# metadata.csv carries a label column for each endpoint of both data sets, named
# in the source's own style rather than ours. Checking the right ones matters:
# the Biogen endpoints reach the table through TDC, so their columns existing at
# all is the reason this script is worth running.
LABEL_COLUMNS = {
    "expansion": [
        "LogD",
        "KSOL",
        "HLM CLint",
        "MLM CLint",
        "Caco-2 Permeability Papp A",
        "Caco-2 Permeability Efflux",
        "MPPB",
        "MBPB",
        "MGMB",
    ],
    "biogen": [
        "solubility_ph_6_8",
        "hlm_clint",
        "rlm_clint",
        "mdr1_mdck_er",
        "plasma_protein_binding_human",
        "plasma_protein_binding_rat",
    ],
}

LABELS = LABEL_COLUMNS[cfg.DATASET]

META_COLUMNS = ["inchi_key", "dataset", "is_benchmark", "provided_split"] + LABELS

CHUNK_ROWS = 200_000


def our_keys() -> tuple[dict, dict]:
    """InChIKey and connectivity-block lookups for every molecule in master.csv."""
    df = pd.read_csv(cfg.MASTER_CSV)
    exact, skeleton = {}, {}
    for name, smiles, split in zip(df[cfg.ID_COL], df[cfg.SMILES_COL], df[cfg.SET_COL]):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        key = Chem.MolToInchiKey(mol)
        exact[key] = (name, split)
        skeleton.setdefault(key.split("-")[0], []).append((name, split))
    print(f"{len(exact)} InChIKeys from {len(df)} molecules in {cfg.MASTER_CSV.name}",
          flush=True)
    return exact, skeleton


def scan(path: Path, exact_keys: dict, skeletons: dict):
    hits, near, rows, labelled = [], [], 0, 0
    reader = pd.read_csv(path, usecols=lambda c: c in META_COLUMNS,
                         chunksize=CHUNK_ROWS, low_memory=False)
    for chunk in reader:
        rows += len(chunk)
        keys = chunk["inchi_key"].astype(str)

        matched = chunk[keys.isin(exact_keys)]
        if len(matched):
            hits.append(matched)

        blocks = keys.str.split("-").str[0]
        close = chunk[blocks.isin(skeletons) & ~keys.isin(exact_keys)]
        if len(close):
            near.append(close)

        present = [c for c in LABELS if c in chunk.columns]
        if present:
            labelled += int(chunk[present].notna().any(axis=1).sum())

        print(f"  {rows:,} rows scanned", end="\r", file=sys.stderr, flush=True)

    empty = pd.DataFrame(columns=META_COLUMNS)
    return (rows, labelled,
            pd.concat(hits) if hits else empty,
            pd.concat(near) if near else empty)


def describe(label: str, frame: pd.DataFrame) -> None:
    if not len(frame):
        return
    print(f"\n--- {label} ---")
    for column in ("dataset", "is_benchmark", "provided_split"):
        if column in frame.columns:
            counts = frame[column].value_counts(dropna=False).to_dict()
            print(f"{column}: {counts}")
    present = {c: int(frame[c].notna().sum()) for c in LABELS if c in frame.columns}
    print(f"rows carrying a {cfg.ACTIVE.label} label: {present}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metadata", type=Path,
                        help="metadata.csv from Flogrammer/Mol-JEPA-dataset")
    parser.add_argument("--out", type=Path, default=cfg.SENSITIVITY_DIR,
                        help="where to write the matched rows")
    args = parser.parse_args()

    if not args.metadata.exists():
        raise SystemExit(f"{args.metadata} not found")
    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    exact_keys, skeletons = our_keys()
    rows, labelled, matched, close = scan(args.metadata, exact_keys, skeletons)

    print(f"\nscanned {rows:,} rows of {args.metadata.name}")
    print(f"exact InChIKey matches:     {len(matched)}")
    print(f"same-skeleton-only matches: {len(close)}")
    print(f"rows anywhere in the table carrying a {cfg.ACTIVE.label} label: {labelled:,}")
    describe("exact", matched)
    describe("same skeleton", close)

    args.out.mkdir(parents=True, exist_ok=True)
    matched.to_csv(args.out / "moljepa_overlap_exact.csv", index=False)
    close.to_csv(args.out / "moljepa_overlap_skeleton.csv", index=False)
    print(f"\nwrote the matched rows to {args.out}")


if __name__ == "__main__":
    main()
