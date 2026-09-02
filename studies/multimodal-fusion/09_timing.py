#!/usr/bin/env python
"""Step 9: what each configuration costs to fit.

The paper's Figure 6 puts modelling time against what the extra modality buys,
and finds cost rising much faster than accuracy. Reproducing that needs times
measured under controlled conditions, so this is a separate pass rather than
numbers scraped from the parallel sweep: one process, one thread, one fold per
endpoint, every configuration in sequence on the same machine.

Encoder time is counted too, and counted honestly. A configuration naming the GNN
modality cannot be fit without an AttentiveFP, and one naming SMILES cannot be fit
without a BiGRU, whether or not that cost was paid earlier and cached. Both are
read from the fold's own npz, where step 3 recorded them.

    python 09_timing.py --gpu 0
    ADME_DATASET=biogen python 09_timing.py --gpu 0
"""

import argparse
import time

import numpy as np
import pandas as pd

import config as cfg
import fusion
from folds import fold_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    import json
    import nets
    from featurize import mol_graph

    cfg.ensure_dirs()
    device = nets.device_for(args.gpu)
    df = pd.read_csv(cfg.MASTER_CSV)
    fold_table = pd.read_csv(cfg.FOLD_CSV)
    rdkit_all = np.load(cfg.RDKIT_NPZ, allow_pickle=True)["x"]
    mol2vec_all = np.load(cfg.MOL2VEC_NPZ)["x"]
    graphs_all = [nets.as_tensors(mol_graph(s)) for s in df[cfg.SMILES_COL]]
    hparams = json.loads(cfg.HPARAM_JSON.read_text())

    rows = []
    for endpoint in args.endpoint:
        path = cfg.fold_embeddings(endpoint, args.repeat, args.fold)
        if not path.exists():
            continue
        masks = fold_masks(df, fold_table, endpoint, args.repeat, args.fold)
        npz = np.load(path)
        blocks = fusion.FoldBlocks(
            npz, masks, rdkit_all, mol2vec_all,
            df[endpoint].to_numpy(dtype=float), graphs_all,
        )
        tuned = hparams.get(endpoint, {})
        seed = cfg.fold_seed(args.repeat, args.fold)

        for method in cfg.GRID_METHODS:
            spec = cfg.GRID_SPEC[method]
            params = {"model": tuned.get(method), "base": {}}
            if spec["fusion"] == "late":
                for modality in ("rdkit", "mol2vec"):
                    if modality in spec["modalities"]:
                        params["base"][modality] = tuned.get(
                            cfg.unimodal_method(modality, "lgbm"))
            if spec["learner"] in cfg.TUNED_LEARNERS and params["model"] is None:
                continue
            if any(v is None for v in params["base"].values()):
                continue

            start = time.time()
            fusion.fit_predict(spec, blocks, params, seed, args.threads, device)
            seconds = time.time() - start

            # What the configuration's modalities cost before the final learner
            # ever ran. The graph learner always pays the GNN's, since it is one.
            encoder = 0.0
            if "gnn" in spec["modalities"] or spec["learner"] == "attfp":
                encoder += blocks.seconds["gnn"]
            if "smiles" in spec["modalities"]:
                encoder += blocks.seconds["smiles"]

            rows.append({
                "endpoint": endpoint, "method": method,
                "combo": spec["combo"], "fusion": spec["fusion"],
                "learner": spec["learner"], "n_modalities": len(spec["modalities"]),
                "n_train": int(masks.fit.sum()),
                "fit_seconds": seconds, "encoder_seconds": encoder,
                "total_seconds": seconds + encoder,
            })
            print(f"  {endpoint:<17} {method:<28} {seconds:7.2f}s fit "
                  f"+ {encoder:6.1f}s encoders", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(cfg.TIMING_CSV, index=False)
    print(f"\nwrote {cfg.TIMING_CSV.name} ({len(table):,} rows)")

    if len(table):
        print("\nmedian total seconds by modality count and fusion strategy:")
        pivot = table.pivot_table(
            index="n_modalities", columns="fusion", values="total_seconds", aggfunc="median"
        )
        print(pivot.round(1).to_string())


if __name__ == "__main__":
    main()
