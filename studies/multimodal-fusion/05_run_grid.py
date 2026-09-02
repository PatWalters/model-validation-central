#!/usr/bin/env python
"""Step 5: fit the design grid -- 33 configurations per endpoint per fold.

Nine unimodal baselines and twenty-four fusion models, over the encoder blocks
step 3 cached and the hyperparameters step 4 chose. 225 folds on ExpansionRx and
150 on Biogen, so 7,425 and 4,950 configuration fits.

Twenty-five of the thirty-three are tabular and run on the CPU in seconds. The
other eight are the graph meta-learner, which trains an AttentiveFP per fold and
wants a GPU. `--learner` splits the two apart so they can run on the machines
that suit them:

    python 05_run_grid.py --learner lgbm rf --jobs 24     # the CPU half
    python 05_run_grid.py --learner attfp --gpu 0         # the GPU half

Both halves are resumable, skipping any configuration whose prediction file
exists, and both write the tidy schema the reference methods already use.

`--control` reruns the eight tabular late-fusion configurations with the
meta-learner fit on the fold's held-out fifth rather than on the molecules its
base learners were fit on. Those go to results/<dataset>/control/ rather than
predictions/, so step 6 cannot sweep them into the comparison as extra methods.
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

import config as cfg
import fusion
from folds import fold_masks, tidy_predictions

CPU_METHODS = [m for m in cfg.GRID_METHODS if cfg.GRID_SPEC[m]["learner"] != "attfp"]
GPU_METHODS = [m for m in cfg.GRID_METHODS if cfg.GRID_SPEC[m]["learner"] == "attfp"]


class Shared:
    """The per-data-set arrays every fold needs, loaded once per process."""

    def __init__(self, paper_gnn_block: bool = False, need_graphs: bool = False):
        import json

        self.df = pd.read_csv(cfg.MASTER_CSV)
        self.folds = pd.read_csv(cfg.FOLD_CSV)
        self.rdkit = np.load(cfg.RDKIT_NPZ, allow_pickle=True)["x"]
        self.mol2vec = np.load(cfg.MOL2VEC_NPZ)["x"]
        self.hparams = json.loads(cfg.HPARAM_JSON.read_text()) if cfg.HPARAM_JSON.exists() else {}
        self.paper_gnn_block = paper_gnn_block
        self.graphs = self._load_graphs() if need_graphs else None

    def _load_graphs(self):
        """Shared graph tensors, built once and reused by every fold."""
        import nets
        from featurize import mol_graph

        return [nets.as_tensors(mol_graph(s)) for s in self.df[cfg.SMILES_COL]]

    def blocks(self, endpoint: str, repeat: int, fold: int):
        npz_path = cfg.fold_embeddings(endpoint, repeat, fold)
        if not npz_path.exists():
            return None, None
        masks = fold_masks(self.df, self.folds, endpoint, repeat, fold)
        blocks = fusion.FoldBlocks(
            np.load(npz_path), masks, self.rdkit, self.mol2vec,
            self.df[endpoint].to_numpy(dtype=float), self.graphs,
            paper_gnn_block=self.paper_gnn_block,
        )
        return blocks, masks

    def params(self, endpoint: str, method: str) -> dict:
        """The tuned settings for one configuration, plus its base learners'.

        A missing entry means step 4 has not reached this configuration; the
        library defaults it falls back to would be a different model wearing the
        same name, so the caller raises rather than quietly substituting them.
        """
        tuned = self.hparams.get(endpoint, {})
        spec = cfg.GRID_SPEC[method]
        out = {"model": tuned.get(method), "base": {}}
        if spec["fusion"] == "late":
            for modality in ("rdkit", "mol2vec"):
                if modality in spec["modalities"]:
                    out["base"][modality] = tuned.get(cfg.unimodal_method(modality, "lgbm"))
        return out


def out_path(method: str, endpoint: str, repeat: int, fold: int,
             control: bool, paper_gnn: bool = False):
    """Where a fold's predictions go.

    The two control runs are deliberately not in predictions/: they are the same
    configurations under a changed assumption, so sweeping them in as extra
    methods would double-count the grid.
    """
    if paper_gnn:
        return cfg.PAPER_GNN_DIR / method / f"{endpoint}_r{repeat}_f{fold}.csv"
    if control:
        return cfg.CONTROL_DIR / method / f"{endpoint}_r{repeat}_f{fold}.csv"
    return cfg.pred_csv(method, endpoint, repeat, fold)


def run_one(shared: Shared, method: str, endpoint: str, repeat: int, fold: int,
            threads: int, device, control: bool, force: bool) -> float | None:
    path = out_path(method, endpoint, repeat, fold, control, shared.paper_gnn_block)
    if path.exists() and not force:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)

    blocks, masks = shared.blocks(endpoint, repeat, fold)
    if blocks is None:
        return None

    spec = cfg.GRID_SPEC[method]
    params = shared.params(endpoint, method)
    if spec["learner"] in cfg.TUNED_LEARNERS and params["model"] is None:
        raise SystemExit(
            f"no tuned hyperparameters for {endpoint}/{method} -- run 04_tune.py first"
        )
    if any(v is None for v in params["base"].values()):
        raise SystemExit(
            f"no tuned base learners for {endpoint}/{method} -- run 04_tune.py first"
        )

    start = time.time()
    y_pred = fusion.fit_predict(
        spec, blocks, params, seed=cfg.fold_seed(repeat, fold),
        threads=threads, device=device, holdout_meta=control,
    )
    seconds = time.time() - start

    frame = tidy_predictions(shared.df, method, endpoint, repeat, fold, masks.test, y_pred)
    frame.to_csv(path, index=False)
    return seconds


def _worker(task):
    """One (method, endpoint, repeat, fold) in a pool process."""
    method, endpoint, repeat, fold, threads, control, force, paper_gnn = task
    global _SHARED
    if "_SHARED" not in globals():
        _SHARED = Shared(paper_gnn_block=paper_gnn)  # tabular half: no graphs
    seconds = run_one(_SHARED, method, endpoint, repeat, fold, threads, None, control, force)
    return method, endpoint, repeat, fold, seconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--learner", nargs="+", default=cfg.LEARNERS, choices=cfg.LEARNERS)
    parser.add_argument("--method", nargs="+", default=None, choices=cfg.GRID_METHODS)
    parser.add_argument("--jobs", type=int, default=1, help="parallel folds (CPU half only)")
    parser.add_argument("--threads", type=int, default=1, help="cores inside one fit")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--control", action="store_true",
                        help="late fusion with the meta-learner on the held-out fifth")
    parser.add_argument("--paper-gnn-block", action="store_true",
                        help="use the 30-wide mean of raw atom features as the GNN "
                             "modality, which is what the released extractor produces")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()

    methods = args.method or [
        m for m in cfg.GRID_METHODS if cfg.GRID_SPEC[m]["learner"] in args.learner
    ]
    if args.control:
        methods = [m for m in methods if m in cfg.CONTROL_METHODS]
        if not methods:
            raise SystemExit("--control covers the tabular late-fusion configurations only")
    if args.paper_gnn_block:
        methods = [m for m in methods if m in cfg.PAPER_GNN_METHODS]
        if not methods:
            raise SystemExit(
                "--paper-gnn-block covers the LightGBM configurations that name the "
                "GNN modality only"
            )

    tasks = [
        (m, e, r, f, args.threads, args.control, args.force, args.paper_gnn_block)
        for e in args.endpoint for r in args.repeat for f in args.fold for m in methods
    ]
    print(f"{cfg.ACTIVE.label}: {len(methods)} configurations x "
          f"{len(args.endpoint)} endpoints x {len(args.repeat) * len(args.fold)} folds "
          f"= {len(tasks):,} fits", flush=True)

    start = time.time()
    done = 0

    if args.jobs > 1 and "attfp" not in args.learner:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(_worker, t) for t in tasks]
            for future in as_completed(futures):
                method, endpoint, repeat, fold, seconds = future.result()
                done += 1
                if seconds is not None and done % 100 == 0:
                    rate = done / (time.time() - start)
                    print(f"  {done:>6,}/{len(tasks):,}  {rate:5.1f}/s  "
                          f"last {endpoint} {method} r{repeat}f{fold} {seconds:.1f}s",
                          flush=True)
    else:
        import nets

        device = nets.device_for(args.gpu)
        print(f"running on {device}", flush=True)
        needs_graphs = any(cfg.GRID_SPEC[m]["learner"] == "attfp" for m in methods)
        shared = Shared(paper_gnn_block=args.paper_gnn_block, need_graphs=needs_graphs)
        for method, endpoint, repeat, fold, threads, control, force, _ in tasks:
            seconds = run_one(shared, method, endpoint, repeat, fold, threads,
                              device, control, force)
            done += 1
            if seconds is not None:
                print(f"  {done:>6,}/{len(tasks):,}  {endpoint:<17} {method:<26} "
                      f"r{repeat}f{fold}  {seconds:6.1f}s", flush=True)

    print(f"done in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
