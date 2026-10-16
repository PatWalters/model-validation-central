#!/usr/bin/env python
"""Step 1: 3D conformers for every molecule in master.csv.

TorchMD-ET reads 3D atomic coordinates and atomic numbers (including explicit
hydrogens) within a 5.0 A cutoff radius. Section 4.2 of the QuADMET-Former
paper states that single-conformer models use RDKit ETKDGv3 followed by MMFF94
force-field optimisation, and multi-conformer models use three conformers per
molecule.

To keep the 3D geometry identical across studies in this repository, this step
uses the same cached RDKit ETKDGv3 + MMFF94 conformer ensemble (seed 42, up to
10 conformers per molecule, largest fragment, random-coordinate fallback so no
molecule is ever dropped) as the Atom-JEPA study.

    python 01_conformers.py
    ADME_DATASET=biogen python 01_conformers.py
"""

import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import config as cfg

N_CONF = 10
SEED = 42
PRUNE_RMS = -1.0


def _embed_one(smi: str) -> tuple[str, tuple[np.ndarray, list[np.ndarray]] | None]:
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi, None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumAtoms())
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    params.pruneRmsThresh = PRUNE_RMS
    params.useSmallRingTorsions = True
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_CONF, params=params))
    if not cids:
        params = AllChem.ETKDGv3()
        params.randomSeed = SEED
        params.useRandomCoords = True
        params.enforceChirality = False
        cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=N_CONF, params=params))
    if not cids:
        return smi, None

    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
    except Exception:
        pass

    z = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=np.int64)
    confs = [
        np.array(mol.GetConformer(cid).GetPositions(), dtype=np.float32)
        for cid in cids
    ]
    return smi, (z, confs)


def main() -> None:
    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    smiles = list(dict.fromkeys(df[cfg.SMILES_COL]))
    cache_path = cfg.CONFORMER_DIR / f"{cfg.DATASET}_seed{SEED}_nconf{N_CONF}_prune{int(PRUNE_RMS)}.pkl"
    print(f"{cfg.ACTIVE.label}: {len(smiles):,} unique SMILES -> {cache_path}")

    start = time.time()
    if cache_path.exists():
        with open(cache_path, "rb") as fh:
            conformers = pickle.load(fh)
        print(f"loaded existing cache in {time.time() - start:.1f}s")
    else:
        conformers = {}
        with ProcessPoolExecutor() as pool:
            for smi, res in pool.map(_embed_one, smiles, chunksize=32):
                conformers[smi] = res
        with open(cache_path, "wb") as fh:
            pickle.dump(conformers, fh)
        print(f"built cache in {time.time() - start:.0f}s")

    failed = [s for s in smiles if conformers.get(s) is None]
    short = [s for s in smiles if conformers.get(s) is not None and len(conformers[s][1]) < 3]
    if short:
        print(f"{len(short)} molecules have fewer than 3 conformers")
    if failed:
        names = df.loc[df[cfg.SMILES_COL].isin(failed), cfg.ID_COL].tolist()
        raise SystemExit(f"{len(failed)} molecules have no conformer: {names[:20]}")
    print(f"all {len(smiles):,} molecules have 3D conformers")


if __name__ == "__main__":
    main()
