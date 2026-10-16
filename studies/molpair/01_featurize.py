#!/usr/bin/env python
"""Step 1: everything MolPAIR needs to know about a molecule, computed once.

None of it depends on a label, so it is computed for every molecule in
master.csv up front and every one of the fold models reads the same cache:

  chemeleon   the frozen CheMeleon fingerprint, 2,048-d -- the global
              representation. CheMeleon's pre-trained message passing with
              mean aggregation, read out through chemprop's `fingerprint`, the
              recipe in the CheMeleon README. No head, nothing fine-tuned.
  rdkit2d     every RDKit 2D descriptor -- the relation representation.
  morgan      Morgan radius-2 bits, packed -- anchor similarity.
  scaffold    the Bemis-Murcko scaffold -- the shared-scaffold pair flag.
  mmp_pairs   every pair of molecules that is an exact one-cut matched pair.

A matched pair here is the Hussain-Rea single cut: break one acyclic single
bond, call the larger side the constant part and the smaller the variable part,
and two molecules are a pair when they share a constant part and differ in the
variable one. The variable part is held to at most ten heavy atoms and to fewer
than the constant part, mmpdb's defaults. Hydrogen replacements, which need an
extra pass that the paper does not describe, are not included.

Runs in an environment with chemprop 2.3 and RDKit (`chemprop` on apollo). The
CheMeleon weights are fetched by chemprop on first use to ~/.chemprop.

    python 01_featurize.py
    ADME_DATASET=biogen python 01_featurize.py
"""

import argparse
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMMPA
from rdkit.Chem.Scaffolds import MurckoScaffold

import config as cfg

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]
MAX_VARIABLE_HEAVIES = 10


def heavy_atoms(smiles: str) -> int:
    mol = Chem.MolFromSmiles(smiles)
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def one_cut_keys(mol) -> list[tuple[str, str]]:
    """(constant, variable) for every admissible single cut of `mol`."""
    keys = set()
    for core, chains in rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False):
        parts = (chains or core).split(".")
        if len(parts) != 2:
            continue
        a, b = sorted(parts, key=heavy_atoms, reverse=True)
        n_const, n_var = heavy_atoms(a), heavy_atoms(b)
        if n_var > MAX_VARIABLE_HEAVIES or n_var >= n_const:
            continue
        keys.add((a, b))
    return sorted(keys)


def per_molecule(smiles: str):
    """Everything but CheMeleon, for one molecule; run in a worker pool."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    desc = Descriptors.CalcMolDescriptors(mol, missingVal=np.nan)
    row = np.array([desc.get(n, np.nan) for n in DESCRIPTOR_NAMES], dtype=np.float64)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=cfg.MOLPAIR["morgan_radius"],
                                                    fpSize=cfg.MOLPAIR["morgan_bits"])
    bits = gen.GetFingerprintAsNumPy(mol).astype(bool)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        scaffold = ""
    return row, np.packbits(bits), scaffold, one_cut_keys(mol)


def matched_pairs(keys: list[list[tuple[str, str]]]) -> np.ndarray:
    """Every (i, j), i < j, sharing a constant part with different variable parts."""
    by_constant = defaultdict(list)
    for i, mol_keys in enumerate(keys):
        for constant, variable in mol_keys:
            by_constant[constant].append((i, variable))
    pairs = set()
    for members in by_constant.values():
        for x in range(len(members)):
            i, vi = members[x]
            for y in range(x + 1, len(members)):
                j, vj = members[y]
                if i != j and vi != vj:
                    pairs.add((min(i, j), max(i, j)))
    return np.array(sorted(pairs), dtype=np.int32).reshape(-1, 2)


def chemeleon_fingerprints(smiles: list[str], batch_size: int) -> np.ndarray:
    import torch
    from chemprop import featurizers, nn
    from chemprop.data import BatchMolGraph
    from chemprop.models import MPNN
    from chemprop.nn import RegressionFFN

    path = cfg.CHEMELEON_CKPT
    if not path.exists():
        from urllib.request import urlretrieve

        path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(cfg.CHEMELEON_URL, path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, weights_only=True, map_location="cpu")
    mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    model = MPNN(message_passing=mp, agg=nn.MeanAggregation(),
                 predictor=RegressionFFN(input_dim=mp.output_dim)).to(device).eval()
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(
        atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())
    print(f"CheMeleon from {path} on {device}, {mp.output_dim}-d")

    out = []
    with torch.inference_mode():
        for start in range(0, len(smiles), batch_size):
            batch = smiles[start:start + batch_size]
            bmg = BatchMolGraph([featurizer(Chem.MolFromSmiles(s)) for s in batch])
            bmg.to(device)
            out.append(model.fingerprint(bmg).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 00_import_baselines.py first")
    df = pd.read_csv(cfg.MASTER_CSV)
    smiles = df[cfg.SMILES_COL].tolist()
    print(f"{cfg.ACTIVE.label}: featurizing {len(smiles):,} molecules")

    start = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(per_molecule, smiles, chunksize=32)
    bad = [i for i, r in enumerate(rows) if r is None]
    if bad:
        raise SystemExit(f"{len(bad)} SMILES do not parse, starting with row {bad[0]}")
    rdkit2d = np.stack([r[0] for r in rows])
    rdkit2d[~np.isfinite(rdkit2d)] = np.nan
    morgan = np.stack([r[1] for r in rows])
    scaffold = np.array([r[2] for r in rows])
    mmp = matched_pairs([r[3] for r in rows])
    print(f"RDKit2D {rdkit2d.shape}, Morgan, scaffolds and {len(mmp):,} one-cut matched "
          f"pairs in {time.time() - start:.0f}s")

    start = time.time()
    chemeleon = chemeleon_fingerprints(smiles, args.batch_size)
    print(f"CheMeleon {chemeleon.shape} in {time.time() - start:.0f}s")
    if not np.all(np.isfinite(chemeleon)):
        raise SystemExit("non-finite CheMeleon fingerprints")

    cfg.FEATURES_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.FEATURES_NPZ,
        names=df[cfg.ID_COL].to_numpy().astype(str),
        chemeleon=chemeleon.astype(np.float32),
        rdkit2d=rdkit2d,
        rdkit2d_names=np.array(DESCRIPTOR_NAMES),
        morgan=morgan,
        scaffold=scaffold,
        mmp_pairs=mmp,
    )
    print(f"wrote {cfg.FEATURES_NPZ.relative_to(cfg.PROJECT_DIR)}")


if __name__ == "__main__":
    main()
