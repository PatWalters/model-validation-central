#!/usr/bin/env python
"""Step 12b: KPGT features for the Trimole-Hybrid arm.

The graph branch of Trimole-Hybrid is KPGT / LiGhT (Li, Zhao and Zeng, KDD 2022;
Nat. Commun. 2023, doi:10.1038/s41467-023-43214-1), Apache-2.0, run from the
authors' pre-trained `base.pth`. It gets its own script because it needs its own
environment: KPGT is built on DGL, whose graphbolt library is compiled per torch
version and stops at torch 2.2, while the rest of this arm runs on current torch.
The two cannot share an interpreter, so this writes a plain .npy that
`12_run_trimole.py --merge-kpgt` folds into the embedding cache.

It also does not use the authors' `preprocess_downstream_dataset.py` and
`extract_features.py` directly, for one specific reason: those write a graph cache
that silently drops any molecule the featurizer rejects, and then extract features
from whatever survived. That is fine when the output is a standalone feature file
and fatal here, where row i of the result has to be molecule i of master.csv. So
this builds the same three inputs their pipeline builds -- the line graph, the
512-bit RDKit path fingerprint and the 200 normalized RDKit descriptors -- keeps
track of which molecules fail, and refuses to write a misaligned file.

Everything about the model is the authors' own: their featurizer, their collator,
their LiGhT and its `generate_fps`, which concatenates the fingerprint virtual
node, the descriptor virtual node and the graph readout into 3 x 768 = 2304
dimensions.

    conda activate kpgt
    KPGT_HOME=~/software/lihan97-KPGT-390f295 \
    KPGT_CHECKPOINT=~/software/lihan97-KPGT-390f295/models/base.pth \
        python 12b_extract_kpgt.py

    ADME_DATASET=biogen python 12b_extract_kpgt.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

KPGT_HOME = Path(
    os.environ.get("KPGT_HOME", Path.home() / "software" / "lihan97-KPGT-390f295")
).expanduser().resolve()
CHECKPOINT = Path(
    os.environ.get("KPGT_CHECKPOINT", KPGT_HOME / "models" / "base.pth")
).expanduser().resolve()

CONFIG_NAME = "base"
PATH_LENGTH = 5
N_VIRTUAL_NODES = 2
# 3 x d_g_feats. Asserted after the first batch, because a different checkpoint
# would still produce features and they would still fit.
FEATURE_DIM = 2304


def kpgt_modules():
    if not KPGT_HOME.exists():
        raise SystemExit(f"no KPGT checkout at {KPGT_HOME} -- set KPGT_HOME")
    if not CHECKPOINT.exists():
        raise SystemExit(
            f"no KPGT checkpoint at {CHECKPOINT}\n"
            "download it from the link in the KPGT README "
            "(https://github.com/lihan97/KPGT) and set KPGT_CHECKPOINT"
        )
    if str(KPGT_HOME) not in sys.path:
        sys.path.insert(0, str(KPGT_HOME))

    # The descriptor block KPGT vendors from descriptastorus looks up
    # scipy.stats.gilbrat, which SciPy renamed to gibrat in 1.11. It is the same
    # distribution under a corrected spelling, so alias it back rather than
    # pinning this whole environment to a four-year-old SciPy.
    import scipy.stats as _stats

    if not hasattr(_stats, "gilbrat") and hasattr(_stats, "gibrat"):
        _stats.gilbrat = _stats.gibrat

    from src.data.collator import preprocess_batch_light
    from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES, smiles_to_graph_tune
    from src.data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
    from src.model.light import LiGhTPredictor
    from src.model_config import config_dict

    return dict(
        preprocess_batch_light=preprocess_batch_light,
        Vocab=Vocab,
        N_ATOM_TYPES=N_ATOM_TYPES,
        N_BOND_TYPES=N_BOND_TYPES,
        smiles_to_graph_tune=smiles_to_graph_tune,
        RDKit2DNormalized=RDKit2DNormalized,
        LiGhTPredictor=LiGhTPredictor,
        config_dict=config_dict,
    )


def build_inputs(smiles: list[str], kpgt: dict, jobs: int):
    """The line graph, path fingerprint and normalized descriptors per molecule."""
    from multiprocessing import Pool

    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    # Serial on purpose. A DGL graph handed back from a worker process travels
    # through torch shared memory, one file descriptor per tensor per graph, and
    # a few thousand molecules exhausts the open-file limit before the pool
    # drains. The descriptors below are plain arrays and parallelise fine.
    print(f"constructing line graphs for {len(smiles)} molecules")
    start = time.time()
    graphs = []
    for i, smi in enumerate(smiles):
        graphs.append(_graph_one(smi, PATH_LENGTH, N_VIRTUAL_NODES))
        if i and i % 1000 == 0:
            print(f"  {i}/{len(smiles)}  ({time.time() - start:.0f}s)", flush=True)
    print(f"  {time.time() - start:.0f}s")

    print("extracting path fingerprints")
    fps = np.asarray(
        [
            list(Chem.RDKFingerprint(Chem.MolFromSmiles(s), minPath=1, maxPath=7, fpSize=512))
            if Chem.MolFromSmiles(s) is not None
            else [0] * 512
            for s in smiles
        ],
        dtype=np.float32,
    )

    print("extracting normalized descriptors")
    start = time.time()
    generator = kpgt["RDKit2DNormalized"]()
    with Pool(jobs) as pool:
        rows = list(pool.imap(generator.process, smiles, chunksize=64))
    # process() returns [ok, d1 ... d200]; the authors keep everything after the flag.
    mds = np.asarray(
        [r[1:] if r is not None else [0.0] * 200 for r in rows], dtype=np.float32
    )
    mds = np.where(np.isnan(mds), 0, mds)
    print(f"  {time.time() - start:.0f}s")

    return graphs, fps, mds


def _graph_one(smiles: str, path_length: int, n_virtual_nodes: int):
    """One line graph, or None if the featurizer rejects the molecule."""
    from src.data.featurizer import smiles_to_graph_tune

    try:
        return smiles_to_graph_tune(
            smiles, max_length=path_length, n_virtual_nodes=n_virtual_nodes
        )
    except Exception:
        return None


def extract(df: pd.DataFrame, batch_size: int, jobs: int) -> None:
    import dgl
    import torch

    kpgt = kpgt_modules()
    smiles = df[cfg.SMILES_COL].tolist()
    graphs, fps, mds = build_inputs(smiles, kpgt, jobs)

    failed = [i for i, g in enumerate(graphs) if g is None]
    if failed:
        raise SystemExit(
            f"{len(failed)} molecules produced no KPGT graph (first at row {failed[0]}, "
            f"{smiles[failed[0]]}). Row alignment with {cfg.MASTER_CSV.name} would be "
            "lost, and this arm would train on a different set of molecules than the "
            "others."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = kpgt["config_dict"][CONFIG_NAME]
    vocab = kpgt["Vocab"](kpgt["N_ATOM_TYPES"], kpgt["N_BOND_TYPES"])

    model = kpgt["LiGhTPredictor"](
        d_node_feats=config["d_node_feats"],
        d_edge_feats=config["d_edge_feats"],
        d_g_feats=config["d_g_feats"],
        d_hpath_ratio=config["d_hpath_ratio"],
        n_mol_layers=config["n_mol_layers"],
        path_length=config["path_length"],
        n_heads=config["n_heads"],
        n_ffn_dense_layers=config["n_ffn_dense_layers"],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size,
    ).to(device)
    state = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    model.eval()
    print(f"loaded KPGT from {CHECKPOINT} on {device}")

    fps_t = torch.from_numpy(fps)
    mds_t = torch.from_numpy(mds)

    out = np.zeros((len(smiles), FEATURE_DIM), dtype=np.float32)
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(graphs), batch_size):
            chunk = graphs[i : i + batch_size]
            batched = dgl.batch(chunk)
            batched.edata["path"][:, :] = kpgt["preprocess_batch_light"](
                batched.batch_num_nodes(), batched.batch_num_edges(),
                batched.edata["path"][:, :],
            )
            feats = model.generate_fps(
                batched.to(device),
                fps_t[i : i + batch_size].to(device),
                mds_t[i : i + batch_size].to(device),
            )
            block = feats.detach().cpu().numpy()
            if block.shape[1] != FEATURE_DIM:
                raise SystemExit(
                    f"expected {FEATURE_DIM}-d KPGT features, got {block.shape[1]} -- "
                    "wrong checkpoint or config"
                )
            out[i : i + len(chunk)] = block
            if (i // batch_size) % 20 == 0:
                done = min(i + batch_size, len(graphs))
                print(f"  {done}/{len(graphs)}  ({time.time() - start:.0f}s)", flush=True)

    cfg.TRIMOLE_KPGT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(cfg.TRIMOLE_KPGT_NPY, out)
    print(
        f"wrote {cfg.TRIMOLE_KPGT_NPY} {out.shape} in {time.time() - start:.0f}s\n"
        f"now run: python 12_run_trimole.py --merge-kpgt"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--jobs", type=int, default=16,
                        help="processes for graph and descriptor construction")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    extract(pd.read_csv(cfg.MASTER_CSV), args.batch_size, args.jobs)


if __name__ == "__main__":
    main()
