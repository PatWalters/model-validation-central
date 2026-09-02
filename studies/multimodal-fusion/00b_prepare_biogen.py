#!/usr/bin/env python
"""Step 0b: turn Biogen's public ADME set into the table the pipeline expects.

The source is `ADME_public_set_3521.csv` from
github.com/molecularinformatics/Computational-ADME, released under MIT with the
paper of Fang et al. (J. Chem. Inf. Model. 2023, 63, 3263-3274). It holds 3,521
commercially sourced compounds measured on six endpoints, already log
transformed at source.

Two things have to be added before the rest of the pipeline can read it.

  cluster   BitBIRCH-Lean labels, exactly as for the ExpansionRx set, so folds
            can be grouped by chemotype.
  ds        a fixed train/test split, which the file does not carry.

The split is built by holding out whole clusters until the test set reaches the
same 30% fraction the ExpansionRx file has. Clusters are taken in a seeded
random order, so the split is reproducible from the seed alone.

Worth knowing when comparing the two data sets: this holdout is cluster-pure by
construction, while the ExpansionRx split arrived with the challenge and is not
(59 of its 651 clusters straddle the boundary). The Biogen holdout is therefore
the harder of the two. Within either data set every method sees the same split,
so the method rankings are unaffected, but absolute numbers do not transfer.

The repository's own `MPNN/` folder does ship 80/20 splits. They are not used
here: they are per-endpoint rather than global, which a multitask model cannot
consume, and the two plasma protein binding splits contain roughly nine times
more molecules than the public file does, being drawn from in-house data that
was never released.

    python 00b_prepare_biogen.py --download
    python 00b_prepare_biogen.py --source ADME_public_set_3521.csv
"""

import argparse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import config as cfg
from clustering import assign_clusters

RDLogger.DisableLog("rdApp.*")

SOURCE_URL = (
    "https://raw.githubusercontent.com/molecularinformatics/Computational-ADME/"
    "main/ADME_public_set_3521.csv"
)

# Their column names to ours. The short names match the style of the
# ExpansionRx endpoints, and LOG_HLM is deliberately the same name in both data
# sets because it is the same assay.
RENAME = {
    "Internal ID": cfg.ID_COL,
    "SMILES": cfg.SMILES_COL,
    "LOG SOLUBILITY PH 6.8 (ug/mL)": "LOG_SOL",
    "LOG HLM_CLint (mL/min/kg)": "LOG_HLM",
    "LOG RLM_CLint (mL/min/kg)": "LOG_RLM",
    "LOG MDR1-MDCK ER (B-A/A-B)": "LOG_MDR1_ER",
    "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)": "LOG_HPPB",
    "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)": "LOG_RPPB",
}

TEST_FRACTION = 0.30


def load_source(path: Path | None, download: bool) -> pd.DataFrame:
    if download:
        print(f"downloading {SOURCE_URL}")
        with urllib.request.urlopen(SOURCE_URL) as response:
            raw = response.read()
        path = path or Path("ADME_public_set_3521.csv")
        path.write_bytes(raw)
        print(f"saved {path} ({len(raw) / 1024:.0f} kB)")
    if path is None or not path.exists():
        raise SystemExit("pass --source <csv> or --download")
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to our schema, drop what RDKit cannot parse and what has no data."""
    missing = sorted(set(RENAME) - set(df.columns))
    if missing:
        raise SystemExit(f"source file is missing columns: {missing}")

    df = df.rename(columns=RENAME)[[cfg.ID_COL, cfg.SMILES_COL] + cfg.TARGET_COLS].copy()

    parsed = df[cfg.SMILES_COL].map(lambda s: Chem.MolFromSmiles(s) is not None)
    if not parsed.all():
        print(f"dropping {int((~parsed).sum())} molecules RDKit could not parse")
        df = df[parsed]

    measured = df[cfg.TARGET_COLS].notna().any(axis=1)
    if not measured.all():
        print(f"dropping {int((~measured).sum())} molecules with no measured endpoint")
        df = df[measured]

    return df.reset_index(drop=True)


def split_by_cluster(clusters: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Hold out whole clusters until `fraction` of the molecules are in the test set."""
    sizes = pd.Series(clusters).value_counts()
    order = np.random.default_rng(seed).permutation(sizes.index.to_numpy())

    target = fraction * len(clusters)
    held_out, running = set(), 0
    for cluster in order:
        if running >= target:
            break
        held_out.add(cluster)
        running += int(sizes[cluster])

    ds = np.where(pd.Series(clusters).isin(held_out), "test", "train")
    print(f"held out {len(held_out)} of {len(sizes)} clusters, "
          f"{running} of {len(clusters)} molecules ({running / len(clusters):.1%})")
    return ds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=None,
                        help="ADME_public_set_3521.csv, if it is already on disk")
    parser.add_argument("--download", action="store_true",
                        help="fetch the source file from GitHub first")
    parser.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    args = parser.parse_args()

    if cfg.DATASET != "biogen":
        raise SystemExit("run with ADME_DATASET=biogen, so the output path is right")

    df = clean(load_source(args.source, args.download))
    print(f"{len(df)} molecules, {len(cfg.TARGET_COLS)} endpoints")

    df[cfg.CLUSTER_COL] = assign_clusters(df[cfg.SMILES_COL])
    sizes = df[cfg.CLUSTER_COL].value_counts()
    print(f"{len(sizes)} clusters, largest {sizes.iloc[0]}, "
          f"{int((sizes == 1).sum())} singletons")

    df[cfg.SET_COL] = split_by_cluster(df[cfg.CLUSTER_COL].to_numpy(),
                                       args.test_fraction, cfg.RANDOM_SEED)

    counts = pd.DataFrame({
        "train": df[df[cfg.SET_COL] == "train"][cfg.TARGET_COLS].notna().sum(),
        "test": df[df[cfg.SET_COL] == "test"][cfg.TARGET_COLS].notna().sum(),
    })
    counts["total"] = counts.sum(axis=1)
    print("\nmeasurements per endpoint:")
    print(counts.to_string())

    df.to_csv(cfg.RAW_CSV, index=False)
    print(f"\nwrote {cfg.RAW_CSV}")


if __name__ == "__main__":
    main()
