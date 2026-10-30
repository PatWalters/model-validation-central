#!/usr/bin/env python
"""Step 1: KERMT's RDKit 2D features for every molecule, computed once.

KERMT's fine-tuning concatenates a molecule-level descriptor vector to the
encoder's readout before the MLP heads, and the repository's default is
`rdkit_2d_normalized`: descriptastorus's 200 RDKit descriptors, each passed
through a fixed CDF fitted by descriptastorus on a reference set. Nothing is
fitted to these data, so the vectors can be computed once for every molecule
in master.csv and sliced per fold by 02_run_kermt.py without anything leaking
from one fold into another.

The generator is the repository's own, looked up by name from
kermt.data.molfeaturegenerator, so the vectors are exactly what
scripts/save_features.py would write. Non-finite values are left in place;
KERMT replaces them with 0 when it loads a datapoint.

    python 01_features.py
    ADME_DATASET=biogen python 01_features.py
"""

import argparse
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import config as cfg

sys.path.insert(0, str(cfg.KERMT_REPO))
from kermt.data.molfeaturegenerator import get_features_generator  # noqa: E402


def featurize(smiles: str) -> np.ndarray:
    return np.asarray(get_features_generator(cfg.FEATURES_GENERATOR)(smiles), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()
    if cfg.FEATURES_NPZ.exists() and not args.force:
        print(f"{cfg.FEATURES_NPZ} already here")
        return

    df = pd.read_csv(cfg.MASTER_CSV)
    start = time.time()
    with Pool(args.workers) as pool:
        X = np.stack(pool.map(featurize, df[cfg.SMILES_COL], chunksize=64))
    print(f"{cfg.FEATURES_GENERATOR}: {X.shape} in {time.time() - start:.0f}s, "
          f"{int((~np.isfinite(X)).any(axis=1).sum())} molecules with a non-finite value")
    np.savez_compressed(cfg.FEATURES_NPZ, X=X, names=df[cfg.ID_COL].to_numpy().astype(str))
    print(f"wrote {cfg.FEATURES_NPZ}")


if __name__ == "__main__":
    main()
