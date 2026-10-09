#!/usr/bin/env python
"""Step 1: ten 3D conformers for every molecule, with the authors' own builder.

Atom-JEPA reads coordinates, not a graph, so every molecule needs a geometry
before anything else can run. This calls the function the authors' fine-tuning
drivers call, `build_or_load_conformers`, with the settings in their
conf/finetune_expansionrx.yaml: RDKit ETKDGv3, ten conformers, seed 42, no RMS
pruning, MMFF94 optimisation, the largest fragment only. Their fallback --
random coordinates without enforced chirality when ETKDG gives nothing -- is
theirs too.

No label is read. The conformers of every molecule in master.csv are made once
and cached, one pickle per data set, keyed by SMILES; both Atom-JEPA arms read
the same cache.

A molecule that comes back with no conformer at all could not be scored by
either arm. The authors' dataset class drops such molecules silently; this
step instead names them, and fails, so a fold can never quietly be scored on
fewer test molecules than every other method.

    python 01_conformers.py
    ADME_DATASET=biogen python 01_conformers.py
"""

import sys
import time

import pandas as pd

import config as cfg

N_CONF = 10
SEED = 42
PRUNE_RMS = -1.0


def main() -> None:
    sys.path.insert(0, str(cfg.ATOMJEPA_HOME))
    from data.datasets.admet.admet_conformers import build_or_load_conformers

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    smiles = list(dict.fromkeys(df[cfg.SMILES_COL]))
    print(f"{cfg.ACTIVE.label}: {len(smiles):,} unique SMILES -> {cfg.CONFORMER_DIR}")

    start = time.time()
    conformers = build_or_load_conformers(
        cfg.DATASET, smiles, str(cfg.CONFORMER_DIR),
        seed=SEED, n_conf=N_CONF, strip_fragments=True, n_workers=-2, prune_rms=PRUNE_RMS,
    )
    print(f"done in {time.time() - start:.0f}s")

    failed = [s for s in smiles if conformers.get(s) is None]
    short = [s for s in smiles if conformers.get(s) is not None and len(conformers[s][1]) < N_CONF]
    if short:
        print(f"{len(short)} molecules have fewer than {N_CONF} conformers")
    if failed:
        names = df.loc[df[cfg.SMILES_COL].isin(failed), cfg.ID_COL].tolist()
        raise SystemExit(f"{len(failed)} molecules have no conformer: {names[:20]}")
    print(f"all {len(smiles):,} molecules embedded")


if __name__ == "__main__":
    main()
