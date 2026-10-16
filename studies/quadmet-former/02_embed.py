#!/usr/bin/env python
"""Step 2: frozen QuADMET-Former embeddings across all ten released checkpoints.

Each of the ten Hugging Face checkpoints (five QM target combinations x single-
and multi-conformer pre-training) is a 6-layer, 128-channel TorchMD-ET encoder
(`model.shared_backbone.*`, 1,312,268 parameters). Neither the single-conformer
nor the multi-conformer checkpoints store any downstream head or conformer-
attention parameters on the Hub; all ten have the exact same 122-tensor
`shared_backbone` state dict.

For every molecule in `master.csv` (in row order), this script builds the exact
5.0 A radius graph (`Distance(loop=True)`) on conformer 0 (and on conformers
0, 1, 2 for the multi-conformer checkpoints) and runs each checkpoint in fp32,
writing `data/<ds>/quadmet_embeddings/<checkpoint>.npz` with:
  - `X`:        (N, 128) final-layer LayerNorm + MeanAggregation embedding
  - `X_layers`: (N, 6, 128) per-layer LayerNorm + MeanAggregation embeddings
                (so `X_layers[:, k-1, :]` is the exact output of a k-layer
                truncated TorchMD-ET, and `X_layers.reshape(N, 768)` is the
                all-layers concatenation)
  - `X_conf3`:  (N, 128) mean over the first three conformers (for `multi_*`)
  - `names`:    molecule IDs in `master.csv` order

Nothing here sees a downstream label, so it is run once per data set.

    python 02_embed.py
    ADME_DATASET=biogen python 02_embed.py
"""

import argparse
import pickle
import time

import numpy as np
import pandas as pd
import torch

import config as cfg
from torchmd_backbone import (
    build_mol_graph,
    collate_graphs,
    forward_precomputed,
    load_torchmd_et,
)

BATCH = 128


def embed_graphs(model: torch.nn.Module, graphs: list, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_final (N, 128), X_layers (N, 6, 128)) in fp32."""
    finals, layers = [], []
    with torch.no_grad():
        for i in range(0, len(graphs), BATCH):
            batch = collate_graphs(graphs[i:i + BATCH], device)
            _, aggr, layer_aggrs = forward_precomputed(model, batch, return_all_layers=True)
            finals.append(aggr.float().cpu().numpy())
            layers.append(layer_aggrs.float().cpu().numpy())
    return np.concatenate(finals, axis=0), np.concatenate(layers, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", default=cfg.CHECKPOINTS, choices=cfg.CHECKPOINTS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    caches = sorted(cfg.CONFORMER_DIR.glob(f"{cfg.DATASET}_*.pkl"))
    if len(caches) != 1:
        raise SystemExit(
            f"expected one conformer cache in {cfg.CONFORMER_DIR}, found {len(caches)} "
            "-- run 01_conformers.py"
        )
    with open(caches[0], "rb") as fh:
        conformers = pickle.load(fh)

    todo = [name for name in args.checkpoint if args.force or not cfg.embedding_npz(name).exists()]
    if not todo:
        print("all requested checkpoints already embedded")
        return

    smiles = list(dict.fromkeys(df[cfg.SMILES_COL]))
    smi_to_idx = {s: i for i, s in enumerate(smiles)}
    row_idx = [smi_to_idx[s] for s in df[cfg.SMILES_COL]]
    ids = df[cfg.ID_COL].to_numpy().astype(str)

    t0 = time.time()
    graphs_c0 = []
    graphs_c1 = []
    graphs_c2 = []
    need_multi = any(cfg.CKPT_CONF[name] == "Multi" for name in todo)
    for smi in smiles:
        z, confs = conformers[smi]
        graphs_c0.append(build_mol_graph(z, confs[0]))
        if need_multi:
            c1 = confs[1] if len(confs) > 1 else confs[0]
            c2 = confs[2] if len(confs) > 2 else c1
            graphs_c1.append(build_mol_graph(z, c1))
            graphs_c2.append(build_mol_graph(z, c2))
    print(f"built 5.0 A radius graphs for {len(smiles):,} unique molecules in {time.time() - t0:.1f}s",
          flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in todo:
        start = time.time()
        ckpt = cfg.checkpoint_path(name)
        if not ckpt.exists():
            raise SystemExit(f"checkpoint not found: {ckpt}")
        model = load_torchmd_et(ckpt, num_layers=cfg.PRETRAIN_LAYERS).to(device)
        model.eval()

        u_final, u_layers = embed_graphs(model, graphs_c0, device)
        X = u_final[row_idx].astype(np.float32)
        X_layers = u_layers[row_idx].astype(np.float32)

        payload = {"X": X, "X_layers": X_layers, "names": ids}
        if cfg.CKPT_CONF[name] == "Multi":
            u_c1, _ = embed_graphs(model, graphs_c1, device)
            u_c2, _ = embed_graphs(model, graphs_c2, device)
            u_conf3 = (u_final + u_c1 + u_c2) / 3.0
            payload["X_conf3"] = u_conf3[row_idx].astype(np.float32)

        if not np.isfinite(X).all() or not np.isfinite(X_layers).all():
            raise SystemExit(f"{name}: non-finite values in embeddings")
        np.savez_compressed(cfg.embedding_npz(name), **payload)
        print(f"  {name:<22} ({cfg.checkpoint_label(name):<18}) -> {X.shape} + {X_layers.shape} "
              f"in {time.time() - start:.1f}s", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
