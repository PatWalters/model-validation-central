#!/usr/bin/env python
"""Step 4: choose the hyperparameters, once, for every tuned configuration.

The paper searches 60 sampled settings by mean squared error over a 3-fold split,
once at seed 0, and reuses the winner for every seed afterwards. The same happens
here, with two adaptations: the inner folds are grouped by chemical cluster, the
rule the outer folds already use, and the search runs on the fitting molecules of
repeat 0 fold 0 rather than on a whole training set that has no encoder blocks
attached to it. Nothing in the search touches the test set, and the settings are
then fixed across all 25 folds, so no fold's hyperparameters were chosen with
sight of the molecules it is scored on.

One search per endpoint per tuned configuration: 24 of the 33 grid entries, the
nine `attfp` ones being untuned. Results accumulate in
data/<dataset>/hyperparameters.json, which is rewritten after every search, so the
step is resumable and can be watched while it runs.

    python 04_tune.py                             # everything missing
    python 04_tune.py --endpoint LogD             # one endpoint
    python 04_tune.py --method fus_GRMS_early_rf  # one configuration
    python 04_tune.py --jobs 32
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RandomizedSearchCV

import config as cfg
import fusion
from folds import fold_masks

TUNED_METHODS = [
    m for m in cfg.GRID_METHODS if cfg.GRID_SPEC[m]["learner"] in cfg.TUNED_LEARNERS
]


def load_store() -> dict:
    if cfg.HPARAM_JSON.exists():
        return json.loads(cfg.HPARAM_JSON.read_text())
    return {}


def save_store(store: dict) -> None:
    cfg.HPARAM_JSON.write_text(json.dumps(store, indent=2, sort_keys=True))


def search_matrix(spec: dict, blocks: fusion.FoldBlocks, store: dict,
                  endpoint: str, jobs: int) -> np.ndarray | None:
    """The design matrix a configuration's final learner is searched over.

    Late fusion searches the meta-learner over stacked base predictions, so its
    base learners have to be tuned first; when they are not yet in the store the
    configuration is deferred rather than searched at defaults.
    """
    fusion_kind, modalities = spec["fusion"], spec["modalities"]
    rows = blocks.fit

    if fusion_kind == "unimodal":
        return blocks.block(modalities[0])[rows]
    if fusion_kind == "early":
        return blocks.matrix(modalities, rows)

    base = {}
    for modality in ("rdkit", "mol2vec"):
        if modality not in modalities:
            continue
        key = cfg.unimodal_method(modality, "lgbm")
        settings = store.get(endpoint, {}).get(key)
        if settings is None:
            return None
        base[modality] = settings
    meta = fusion.base_predictions(blocks, modalities, base, cfg.RANDOM_SEED, jobs)
    return meta[rows]


def tune(method: str, blocks: fusion.FoldBlocks, store: dict, endpoint: str,
         groups: np.ndarray, jobs: int, iterations: int) -> dict | None:
    spec = cfg.GRID_SPEC[method]
    x = search_matrix(spec, blocks, store, endpoint, jobs)
    if x is None:
        return None

    y = blocks.y[blocks.fit]
    estimator = fusion.make_tabular(spec["learner"], {}, cfg.RANDOM_SEED, threads=1)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=cfg.SEARCH_SPACES[spec["learner"]],
        n_iter=iterations,
        scoring=cfg.SEARCH_SCORING,
        cv=GroupKFold(n_splits=cfg.SEARCH_CV),
        n_jobs=jobs,
        random_state=cfg.RANDOM_SEED,
    )
    search.fit(x, y, groups=groups)

    best = dict(search.best_params_)
    best.pop("verbose", None)
    best.pop("random_state", None)
    best.pop("n_jobs", None)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--method", nargs="+", default=TUNED_METHODS, choices=TUNED_METHODS)
    parser.add_argument("--jobs", type=int, default=-1, help="cores per search")
    parser.add_argument("--iterations", type=int, default=cfg.SEARCH_ITERATIONS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    fold_table = pd.read_csv(cfg.FOLD_CSV)
    rdkit_all = np.load(cfg.RDKIT_NPZ, allow_pickle=True)["x"]
    mol2vec_all = np.load(cfg.MOL2VEC_NPZ)["x"]
    clusters = df[cfg.CLUSTER_COL].to_numpy()

    from featurize import mol_graph
    graphs_all = [mol_graph(s) for s in df[cfg.SMILES_COL]]

    store = load_store()

    # Two passes over the endpoints, not one. The unimodal LightGBM searches are
    # the cheapest in the grid and they are also late fusion's base learners, so
    # every later configuration -- including the graph meta-learners, which run on
    # another machine entirely -- is blocked until they exist. Doing them for all
    # endpoints first unblocks the GPU side within a few minutes instead of at the
    # end of the whole search.
    base_first = [m for m in args.method
                  if cfg.GRID_SPEC[m]["fusion"] == "unimodal"
                  and cfg.GRID_SPEC[m]["learner"] == "lgbm"]
    rest = [m for m in args.method if m not in base_first]

    for pass_methods, pass_name in ((base_first, "base learners"), (rest, "the rest")):
        if not pass_methods:
            continue
        print(f"\n=== searching {pass_name}: {len(pass_methods)} configurations "
              f"x {len(args.endpoint)} endpoints ===", flush=True)
        run_pass(args, pass_methods, store, df, fold_table, rdkit_all,
                 mol2vec_all, clusters, graphs_all)

    for endpoint in args.endpoint:
        done = len(store.get(endpoint, {}))
        print(f"{endpoint:<17} {done}/{len(TUNED_METHODS)} configurations tuned")


def run_pass(args, methods, store, df, fold_table, rdkit_all, mol2vec_all,
             clusters, graphs_all) -> None:
    """One sweep over the endpoints, searching `methods` on each."""
    for endpoint in args.endpoint:
        npz_path = cfg.fold_embeddings(endpoint, cfg.TUNE_REPEAT, cfg.TUNE_FOLD)
        if not npz_path.exists():
            print(f"{endpoint:<17} no encoder blocks yet -- run 03_encode_folds.py")
            continue

        masks = fold_masks(df, fold_table, endpoint, cfg.TUNE_REPEAT, cfg.TUNE_FOLD)
        blocks = fusion.FoldBlocks(
            np.load(npz_path), masks, rdkit_all, mol2vec_all,
            df[endpoint].to_numpy(dtype=float), graphs_all,
        )
        groups = clusters[blocks.rows][blocks.fit]
        store.setdefault(endpoint, {})

        # Within a pass, unimodal first for the same reason.
        ordered = sorted(methods, key=lambda m: cfg.GRID_SPEC[m]["fusion"] != "unimodal")
        for method in ordered:
            if method in store[endpoint] and not args.force:
                continue
            start = time.time()
            best = tune(method, blocks, store, endpoint, groups,
                        args.jobs, args.iterations)
            if best is None:
                print(f"  {endpoint:<17} {method:<24} deferred (base learners untuned)")
                continue
            store[endpoint][method] = best
            save_store(store)
            print(f"  {endpoint:<17} {method:<24} {time.time() - start:6.1f}s  {best}",
                  flush=True)


if __name__ == "__main__":
    main()
