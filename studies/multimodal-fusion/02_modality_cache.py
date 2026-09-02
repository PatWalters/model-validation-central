#!/usr/bin/env python
"""Step 2: the two modalities that do not depend on the fold.

RDKit's descriptor block and a Mol2Vec embedding are properties of a molecule, so
each is computed once for the whole collection and cached in master.csv row
order. The GNN and SMILES modalities are supervised encoders and are built per
fold by step 3.

Mol2Vec needs the authors' pre-trained word2vec model. Point `MOL2VEC_MODEL` at
`model_300dim.pkl` from github.com/samoturk/mol2vec (BSD-3), or pass --model.

    python 02_modality_cache.py
    ADME_DATASET=biogen python 02_modality_cache.py
    python 02_modality_cache.py --force        # recompute both caches
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

import config as cfg
from featurize import mol2vec_embeddings, rdkit_descriptors

DEFAULT_MODEL = os.environ.get(
    "MOL2VEC_MODEL", os.path.expanduser("~/software/mol2vec_model/model_300dim.pkl")
)


def build_rdkit(df: pd.DataFrame, force: bool) -> None:
    if cfg.RDKIT_NPZ.exists() and not force:
        cached = np.load(cfg.RDKIT_NPZ, allow_pickle=True)
        if len(cached["x"]) == len(df):
            print(f"rdkit descriptors cached {cached['x'].shape}")
            return
        print("cached rdkit descriptors do not match the data set, recomputing")

    from featurize import DESCRIPTOR_NAMES

    start = time.time()
    x, names = rdkit_descriptors(df[cfg.SMILES_COL].tolist())
    dropped = [n for n in DESCRIPTOR_NAMES if n not in set(names)]
    if dropped:
        print(f"  dropped {len(dropped)} descriptors non-finite somewhere: "
              f"{', '.join(dropped[:8])}{' ...' if len(dropped) > 8 else ''}")
    np.savez_compressed(cfg.RDKIT_NPZ, x=x, names=np.array(names, dtype=object))
    print(
        f"rdkit descriptors {x.shape} of {len(DESCRIPTOR_NAMES)} available, "
        f"in {time.time() - start:.1f}s -> {cfg.RDKIT_NPZ.name}"
    )


def build_mol2vec(df: pd.DataFrame, model_path: str, force: bool) -> None:
    if cfg.MOL2VEC_NPZ.exists() and not force:
        cached = np.load(cfg.MOL2VEC_NPZ)
        if len(cached["x"]) == len(df):
            print(f"mol2vec embeddings cached {cached['x'].shape}")
            return
        print("cached mol2vec embeddings do not match the data set, recomputing")

    if not os.path.exists(model_path):
        raise SystemExit(
            f"mol2vec model not found at {model_path}\n"
            "download model_300dim.pkl from github.com/samoturk/mol2vec and set "
            "MOL2VEC_MODEL, or pass --model"
        )

    from gensim.models import word2vec

    start = time.time()
    model = word2vec.Word2Vec.load(model_path)
    print(f"  loaded {os.path.basename(model_path)}: "
          f"{len(model.wv):,} identifiers, {model.wv.vector_size}-d")
    x = mol2vec_embeddings(df[cfg.SMILES_COL].tolist(), model.wv)
    np.savez_compressed(cfg.MOL2VEC_NPZ, x=x)
    print(
        f"mol2vec embeddings {x.shape} in {time.time() - start:.1f}s "
        f"-> {cfg.MOL2VEC_NPZ.name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="mol2vec model_300dim.pkl")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-mol2vec", action="store_true")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    print(f"{cfg.ACTIVE.label}: {len(df):,} molecules")

    build_rdkit(df, args.force)
    if not args.skip_mol2vec:
        build_mol2vec(df, args.model, args.force)


if __name__ == "__main__":
    main()
