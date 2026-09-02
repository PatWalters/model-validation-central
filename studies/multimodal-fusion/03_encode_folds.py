#!/usr/bin/env python
"""Step 3: the two supervised encoders, once per endpoint and fold.

The GNN and SMILES modalities are not featurizations. Each is an encoder trained
on the fold's own training molecules against the fold's own endpoint, so a
molecule's block changes with the endpoint and the fold and cannot be cached the
way steps 2's two modalities are.

Training them here rather than inside step 5 is what makes the whole grid
affordable. Each fold trains AttentiveFP once and the BiGRU once, and all
thirty-three configurations of that fold then read the same cached blocks. The
alternative -- retraining an encoder inside every configuration that names it --
would be twenty-odd times the work for identical numbers.

Each fold writes one npz holding, for every molecule the fold touches:

    gnn_embed      the 200-wide AttentiveFP readout, the GNN modality
    gnn_atom_mean  the 30-wide mean of raw atom features, which is what the
                   released extractor actually produces; kept as a control
    gnn_pred       AttentiveFP's own prediction, the GNN base learner
    smiles_embed   the 256-wide BiGRU state, the SMILES modality
    smiles_pred    the BiGRU's own prediction, the SMILES base learner

225 folds on ExpansionRx, 150 on Biogen. Resumable: a fold whose npz exists is
skipped.

    python 03_encode_folds.py --gpu 0
    ADME_DATASET=biogen python 03_encode_folds.py --gpu 0
    python 03_encode_folds.py --endpoint LogD --repeat 0 --fold 0   # one fold
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch

import config as cfg
import nets
from featurize import build_vocab, encode_smiles, mol_graph
from folds import fold_masks


def load_graphs(df: pd.DataFrame, cache: dict) -> list:
    """Molecular graphs for every molecule, as shared tensors, built once.

    Every fold of every endpoint reads the same graphs, so the numpy-to-torch
    conversion happens once for the whole process rather than 225 times.
    """
    if "graphs" not in cache:
        start = time.time()
        cache["graphs"] = [nets.as_tensors(mol_graph(s)) for s in df[cfg.SMILES_COL]]
        print(f"featurized {len(cache['graphs']):,} graphs in {time.time() - start:.1f}s",
              flush=True)
    return cache["graphs"]


def run_fold(df, folds, graphs, endpoint, repeat, fold, device, force) -> str:
    out_path = cfg.fold_embeddings(endpoint, repeat, fold)
    if out_path.exists() and not force:
        return "skip"

    masks = fold_masks(df, folds, endpoint, repeat, fold)
    touched = masks.touched
    y = df[endpoint].to_numpy(dtype=float)

    # Row order inside the npz is master.csv order restricted to the touched
    # molecules. Every consumer re-derives it the same way, from the same masks.
    idx_touched = np.flatnonzero(touched)
    position = {row: i for i, row in enumerate(idx_touched)}
    fit_rows = [position[i] for i in np.flatnonzero(masks.fit)]
    val_rows = [position[i] for i in np.flatnonzero(masks.val)]

    graphs_touched = [graphs[i] for i in idx_touched]
    y_touched = y[idx_touched]
    seed = cfg.fold_seed(repeat, fold)

    # --- AttentiveFP ---
    start = time.time()
    gnn = nets.fit_graph_regressor(
        [graphs_touched[i] for i in fit_rows], y_touched[fit_rows],
        [graphs_touched[i] for i in val_rows], y_touched[val_rows],
        seed=seed, device=device,
    )
    gnn_seconds = time.time() - start
    gnn_embed, gnn_pred = nets.graph_embed_and_predict(gnn, graphs_touched, device=device)
    gnn_atom_mean = nets.graph_embed_atom_mean(graphs_touched)

    # --- SMILES BiGRU ---
    smiles = df[cfg.SMILES_COL].to_numpy()[idx_touched]
    # The vocabulary comes from the fitting molecules only, as the released
    # `build_smiles_vocab` does. Characters that appear only in the validation or
    # test molecules become <unk>.
    stoi, itos, max_len = build_vocab([smiles[i] for i in fit_rows])
    ids = encode_smiles(list(smiles), stoi, max_len)

    start = time.time()
    bigru = nets.fit_smiles_regressor(
        ids[fit_rows], y_touched[fit_rows],
        ids[val_rows], y_touched[val_rows],
        vocab_size=len(itos), seed=seed, pad_idx=stoi["<pad>"], device=device,
    )
    smiles_seconds = time.time() - start
    smiles_embed, smiles_pred = nets.smiles_embed_and_predict(bigru, ids, device=device)

    np.savez_compressed(
        out_path,
        rows=idx_touched.astype(np.int32),
        gnn_embed=gnn_embed.astype(np.float32),
        gnn_atom_mean=gnn_atom_mean.astype(np.float32),
        gnn_pred=gnn_pred.astype(np.float32),
        smiles_embed=smiles_embed.astype(np.float32),
        smiles_pred=smiles_pred.astype(np.float32),
        vocab_size=np.int32(len(itos)),
        max_len=np.int32(max_len),
        gnn_seconds=np.float32(gnn_seconds),
        smiles_seconds=np.float32(smiles_seconds),
    )
    return f"{gnn_seconds:5.1f}s gnn + {smiles_seconds:5.1f}s bigru"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    cfg.ensure_dirs()
    device = nets.device_for(args.gpu)
    print(f"{cfg.ACTIVE.label} on {device}", flush=True)

    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    cache: dict = {}
    graphs = load_graphs(df, cache)

    for endpoint in args.endpoint:
        start = time.time()
        for repeat in args.repeat:
            for fold in args.fold:
                note = run_fold(df, folds, graphs, endpoint, repeat, fold, device, args.force)
                if note != "skip":
                    print(f"  {endpoint:<17} r{repeat} f{fold}  {note}", flush=True)
        done = len(list(cfg.EMBED_DIR.glob(f"{endpoint}_r*_f*.npz")))
        print(f"{endpoint:<17} {done:>3}/25 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
