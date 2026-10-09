#!/usr/bin/env python
"""Step 2: the frozen Atom-JEPA embedding of every molecule, for the TabPFN arm.

The pre-trained encoder is loaded from the Hub through the authors' `atom-jepa`
package and run once over conformer 0 of each molecule from the cache step 1
built, so this arm and the fine-tuned one see the same geometry. Their
`embed(..., layers="all")` returns the l = 0 features of each of the eight
blocks, mean-pooled over every atom including hydrogens; flattened, that is
2,048 numbers per molecule. It is the "all layers" representation of the
paper's frozen baselines, the better of the two it reports on ExpansionRx.

It runs in fp32, as the authors' frozen baseline does. cuEquivariance's fused
kernels are used, and in fp32 they reproduce the plain PyTorch path exactly;
bf16 does not, and moves the eighth block's features by more than 40%.

Nothing here sees a label, so it is done once and reused by all 25 folds of
every endpoint.

    python 02_embed.py
    ADME_DATASET=biogen python 02_embed.py
"""

import pickle
import time

import numpy as np
import pandas as pd

import config as cfg

BATCH = 64


def main() -> None:
    import torch
    from atom_jepa import AtomJEPA

    df = pd.read_csv(cfg.MASTER_CSV)
    caches = sorted(cfg.CONFORMER_DIR.glob(f"{cfg.DATASET}_*.pkl"))
    if len(caches) != 1:
        raise SystemExit(f"expected one conformer cache in {cfg.CONFORMER_DIR}, found {len(caches)}"
                         " -- run 01_conformers.py")
    with open(caches[0], "rb") as fh:
        conformers = pickle.load(fh)

    model = AtomJEPA.from_pretrained(cfg.ATOMJEPA_CKPT, precision="fp32", cuequivariance=True)
    model.eval()

    smiles = list(dict.fromkeys(df[cfg.SMILES_COL]))
    lookup = {}
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(smiles), BATCH):
            chunk = smiles[i:i + BATCH]
            structs = []
            for smi in chunk:
                z, confs = conformers[smi]
                structs.append((np.asarray(z), np.asarray(confs[0], dtype=np.float64)))
            out = model.embed(structs, layers=cfg.EMBED_LAYERS)
            out = out.float().reshape(len(chunk), -1).cpu().numpy()
            lookup.update(zip(chunk, out))
            if (i // BATCH) % 20 == 0:
                print(f"  {i + len(chunk):,}/{len(smiles):,}  {time.time() - start:.0f}s", flush=True)

    X = np.stack([lookup[s] for s in df[cfg.SMILES_COL]]).astype(np.float32)
    if X.shape[1] != cfg.EMBED_DIM:
        raise SystemExit(f"expected {cfg.EMBED_DIM}-d embeddings, got {X.shape[1]}")
    if not np.isfinite(X).all():
        raise SystemExit("non-finite values in the embeddings")
    np.savez_compressed(cfg.EMBED_NPZ, X=X, names=df[cfg.ID_COL].to_numpy().astype(str))
    print(f"wrote {cfg.EMBED_NPZ.name} {X.shape} in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
