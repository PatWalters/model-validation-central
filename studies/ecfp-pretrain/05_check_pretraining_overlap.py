#!/usr/bin/env python
"""Step 5: how much of the pre-training corpus is in our test sets?

Label leakage is not possible here, and it is worth saying why before measuring
anything. PT-GIN's pre-training target is the molecule's own hashed ECFP4,
computed from its structure. There is no experimental measurement in the
objective, so no ADME value from either data set can have reached the encoder as
a training label, whatever molecules it saw.

Molecular overlap is a separate question and it does matter, because a frozen
encoder that has already seen a test molecule has had more chance to build a
useful embedding for it. Two things are checked:

  - exact matches, by standard InChIKey. The same molecule, same stereochemistry.
  - connectivity matches, by the first InChIKey block. The same skeleton with
    different stereochemistry or protonation.

The two data sets are asymmetric in exactly the way this measures. The authors
filtered QMugs at Tanimoto 0.5 against every benchmark they used, which includes
Biogen, so a Biogen-like molecule was removed from pre-training by construction.
ExpansionRx was not one of their benchmarks and got no such filter. If PT-GIN
does better on ExpansionRx than on Biogen, this is the first thing to rule out.

The nearest-neighbour Tanimoto distribution is reported too, since 0.5 is the
threshold their filter used and a test molecule's nearest pre-training neighbour
is the quantity that threshold was about.

A handful of Biogen test molecules come out above 0.5 even though Biogen is one
of the sets the filter was applied to. That is a representation difference, not a
leak. The authors' own `max_tanimoto_*` columns show their filter held exactly:
the highest similarity to any Biogen task among the kept molecules is 0.4958.
They computed it on standardised structures -- canonical tautomer, neutralised,
parent fragment -- and this script compares the SMILES as they appear in
master.csv, so a molecule can sit just under their threshold in their
representation and just over it in this one. Both numbers are worth having: theirs
says the filter did what it claims, and this one says what the overlap looks like
in the form the rest of this project handles molecules in.

QMugs ships with the authors' repository as data/qmugs.csv.gz, carrying a
`tanimoto_filter_0.5` column that marks the 462,189 molecules the released
checkpoints were actually pre-trained on. That file is not in the sparse checkout
the rest of this project uses, so fetch it first:

    curl -L -o qmugs.csv.gz \\
      https://raw.githubusercontent.com/oxpig/topological-pretraining/main/data/qmugs.csv.gz
    python 05_check_pretraining_overlap.py qmugs.csv.gz
    ADME_DATASET=biogen python 05_check_pretraining_overlap.py qmugs.csv.gz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

import config as cfg

RDLogger.DisableLog("rdApp.*")

# The fingerprint the authors' similarity filter used: radius 2, 2048 bits,
# chirality on.
FP_RADIUS = 2
FP_SIZE = 2048
FILTER_THRESHOLD = 0.5
FILTER_COLUMN = f"tanimoto_filter_{FILTER_THRESHOLD}"


def keys_and_fps(smiles, label: str):
    """InChIKeys and fingerprints for a list of SMILES, skipping what will not parse."""
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=FP_RADIUS, fpSize=FP_SIZE, includeChirality=True
    )
    keys, fps, failed = [], [], 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        keys.append(Chem.MolToInchiKey(mol))
        fps.append(generator.GetFingerprint(mol))
    print(f"{label}: {len(keys):,} molecules"
          + (f" ({failed} unparseable)" if failed else ""), flush=True)
    return keys, fps


def load_pretraining(path: Path, sample: int | None) -> pd.Series:
    """The molecules the released checkpoints were pre-trained on.

    The table is all 665,879 of QMugs; `tanimoto_filter_0.5` marks the subset that
    survived the authors' similarity filter, which is what was actually trained on.
    Reading the unfiltered table instead would overstate the overlap by a third.
    """
    table = pd.read_csv(path)
    if FILTER_COLUMN not in table.columns:
        raise SystemExit(
            f"{path.name} has no {FILTER_COLUMN} column -- this should be the "
            "authors' data/qmugs.csv.gz"
        )
    kept = table[table[FILTER_COLUMN].astype(bool)]
    print(f"pre-training corpus: {len(kept):,} of {len(table):,} QMugs molecules "
          f"survive the {FILTER_THRESHOLD} filter")
    if sample:
        kept = kept.sample(n=min(sample, len(kept)), random_state=0)
    return kept["SMILES"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("qmugs", type=Path, help="the authors' data/qmugs.csv.gz")
    parser.add_argument("--sample", type=int, default=None,
                        help="use a random subset of the corpus (for a smoke test)")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 00_import_baselines.py first")
    cfg.ensure_dirs()

    master = pd.read_csv(cfg.MASTER_CSV)
    test = master[master[cfg.SET_COL] == "test"].reset_index(drop=True)
    test_keys, test_fps = keys_and_fps(test[cfg.SMILES_COL], f"{cfg.ACTIVE.label} test set")
    if len(test_keys) != len(test):
        raise SystemExit(
            f"{len(test) - len(test_keys)} test SMILES would not parse, which "
            f"{cfg.MASTER_CSV.name} is not supposed to contain"
        )

    pre_keys, pre_fps = load_pretraining(args.qmugs, args.sample).pipe(
        keys_and_fps, "pre-training corpus"
    )

    exact = set(pre_keys)
    connectivity = {key.split("-")[0] for key in pre_keys}
    in_exact = np.array([k in exact for k in test_keys])
    in_block = np.array([k.split("-")[0] in connectivity for k in test_keys])

    print(f"\nexact InChIKey matches:        {in_exact.sum():,} of {len(test_keys):,} "
          f"({100 * in_exact.mean():.2f}%)")
    print(f"connectivity block matches:   {in_block.sum():,} of {len(test_keys):,} "
          f"({100 * in_block.mean():.2f}%)")

    # Nearest pre-training neighbour of each test molecule, which is the quantity
    # the authors' 0.5 filter thresholded.
    nearest = np.array([
        max(DataStructs.BulkTanimotoSimilarity(fp, pre_fps)) for fp in test_fps
    ])
    over = int((nearest > FILTER_THRESHOLD).sum())
    print(f"\nnearest-neighbour Tanimoto to the corpus, over the "
          f"{len(nearest):,} test molecules:")
    for q in (0.5, 0.75, 0.9, 0.95, 1.0):
        print(f"  {int(q * 100):>3}th percentile  {np.quantile(nearest, q):.3f}")
    print(f"  above the authors' {FILTER_THRESHOLD} filter: {over:,} "
          f"({100 * over / len(nearest):.1f}%)")

    out = cfg.TABLE_DIR / "pretraining_overlap.csv"
    pd.DataFrame({
        cfg.ID_COL: test[cfg.ID_COL],
        "inchikey": test_keys,
        "in_corpus_exact": in_exact,
        "in_corpus_connectivity": in_block,
        "nearest_tanimoto": nearest,
    }).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(cfg.PROJECT_DIR)}")


if __name__ == "__main__":
    main()
