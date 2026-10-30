#!/usr/bin/env python
"""Step 4: Causal-Chemprop, one SCM per feasible series per fold.

For every (endpoint, repeat, fold) the encoder from 03_train_encoder.py is
loaded, and for every feasible series of that endpoint:

  1. the analog set is the fold's training molecules measured for the endpoint
     whose Murcko scaffold has Tanimoto >= 0.25 to the anchor's (02_make_series);
  2. the authors' fit_causal learns the DAG with DAGMA, fits the mechanisms with
     dowhy's GCM and computes intrinsic causal contributions (cc_kit.fit_scm);
  3. the anchor's measured value is observed and every other member of the
     series is predicted counterfactually, through test_counterfactual's logic
     unchanged (cc_kit.predict_counterfactual).

A job is one series in one fold, a few minutes of single-threaded CPU, and jobs
run in a process pool. DAGMA's 201 x 201 linear algebra is far faster on one
thread than on 32 -- about 1,240 iterations a second against 15 on apollo --
so every worker is pinned to one. Each job writes
work/causal_parts/<ds>/<endpoint>_r<r>_f<f>_s<series>.csv, the resume point,
and a JSON of diagnostics beside it. When every series of a fold is done the
parts are joined into predictions/<ds>/causal_chemprop/<endpoint>_r<r>_f<f>.csv.

    python 04_run_causal.py --workers 28
    python 04_run_causal.py --workers 18 --reverse   # a second machine, from the other end
    ADME_DATASET=biogen python 04_run_causal.py --workers 28
    python 04_run_causal.py --endpoint LOG_HLM --repeat 0 --fold 0 --workers 1   # smoke test
    python 04_run_causal.py --repeat 0 --assemble-only   # after parts arrive from another machine
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"
# The SCM stage is CPU only. Without this every worker opens a CUDA context it
# never uses, a quarter of a gigabyte each, which 28 workers turn into half the
# card the encoder sweep is training on.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import contextlib
import io
import json
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd

import config as cfg

PART_DIR = cfg.PROJECT_DIR / "work" / "causal_parts" / cfg.DATASET


def part_csv(endpoint, repeat, fold, series):
    return PART_DIR / f"{endpoint}_r{repeat}_f{fold}_s{series}.csv"


# A claim names the machine, not the process, so a runner restarted on the same
# machine resumes its own claims. Run one runner per machine.
HOST = os.uname().nodename


def claim_path(endpoint, repeat, fold, series):
    return part_csv(endpoint, repeat, fold, series).with_suffix(".claim")


def claimed_elsewhere(endpoint, repeat, fold, series) -> bool:
    """A series-fold another machine has started. Claims arrive by rsync."""
    p = claim_path(endpoint, repeat, fold, series)
    return p.exists() and p.read_text().strip() != HOST


def run_job(job: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    # dowhy's GCM defaults to joblib n_jobs=-1 for mechanism selection and the
    # Shapley estimates behind the ICCs: inside a pool of 28 workers that is 28
    # x 32 processes. One job per worker; the pool is the parallelism. The
    # vendored gcm reads dowhy.gcm.config, not a copy of its own.
    import dowhy.gcm.config
    dowhy.gcm.config.set_default_n_jobs(1)
    dowhy.gcm.config.disable_progress_bars()
    from rdkit import DataStructs
    import cc_kit
    import networkx as nx
    from dowhy.graph import get_ordered_predecessors

    endpoint, repeat, fold, series = job["endpoint"], job["repeat"], job["fold"], job["series"]
    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    members = pd.read_csv(cfg.SERIES_CSV)
    members = members[(members["endpoint"] == endpoint) & (members["series"] == series)]
    model_dir = cfg.MODEL_DIR / f"{endpoint}_r{repeat}_f{fold}"
    meta = json.loads((model_dir / "encoder.json").read_text())
    scaler = cc_kit.Scaler(meta["scaler_mean"], meta["scaler_scale"])

    fold_of = df[cfg.ID_COL].map(folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"])
    fit = df[df[endpoint].notna() & (df[cfg.SET_COL] == "train") & (fold_of != fold)]
    by_id = df.set_index(cfg.ID_COL)
    anchor_id = members.loc[members["role"] == "anchor", cfg.ID_COL].iloc[0]
    scored = members.loc[members["role"] == "scored", cfg.ID_COL].to_numpy()
    anchor = by_id.loc[anchor_id]

    # The analog set, as 02_make_series.py counted it.
    import importlib
    ms = importlib.import_module("02_make_series")
    a_fp = ms.scaffold_fp(ms.scaffold_smiles(anchor[cfg.SMILES_COL]))
    fit_fps = [ms.scaffold_fp(ms.scaffold_smiles(s)) for s in fit[cfg.SMILES_COL]]
    sims = np.array(DataStructs.BulkTanimotoSimilarity(a_fp, fit_fps))
    analogs = fit[sims >= cfg.ANALOG_SCAFFOLD_SIM]

    start = time.time()
    torch.manual_seed(cfg.fold_seed(repeat, fold) + int(series))
    model = cc_kit.load_encoder(model_dir / meta["checkpoint"],
                                cfg.PROJECT_DIR / "work" / "scm" / cfg.DATASET / f"{endpoint}_r{repeat}_f{fold}_s{series}")
    log = io.StringIO()
    smiles = [by_id.loc[i, cfg.SMILES_COL] for i in scored]
    no_target = False
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            cc_kit.fit_scm(model, analogs, endpoint, scaler, float(anchor[endpoint]),
                           cfg.fold_seed(repeat, fold))
        except nx.NodeNotFound:
            # DAGMA kept no edge into or out of the target, so the target is not
            # in the graph and the SCM has nothing to predict with. The paper's
            # rule for a degenerate group is ordinary Chemprop; so here, flagged.
            no_target = True
        scm_s = time.time() - start
        sim = cc_kit.analog_similarity(
            [by_id.loc[i, cfg.SMILES_COL] for i in members[cfg.ID_COL]], analogs[cfg.SMILES_COL])
        if no_target:
            y = y_int = y_cf = cc_kit.predict_regular(model, smiles)
            switched, var_fallback = np.zeros(len(smiles), dtype=bool), False
        else:
            y, y_int, y_cf, switched, var_fallback = cc_kit.predict_counterfactual(
                model, anchor[cfg.SMILES_COL], smiles, sim)
        y_anchor_regular = float(cc_kit.predict_regular(model, [anchor[cfg.SMILES_COL]])[0])

    if no_target:
        icc, parents, contributors = {"ic50": np.nan}, [], []
    else:
        icc = model.icc
        parents = list(get_ordered_predecessors(model.causal_model.graph, "ic50"))
        contributors = [k for k, v in icc.items() if v > model.icc_threshold * icc["ic50"] and k != "ic50"]
    out = part_csv(endpoint, repeat, fold, series)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "method": cfg.CAUSAL_METHOD, "endpoint": endpoint, "repeat": repeat, "fold": fold,
        cfg.ID_COL: scored, cfg.SMILES_COL: smiles,
        "y_true": [float(by_id.loc[i, endpoint]) for i in scored], "y_pred": y,
        "y_intervention": y_int, "y_counterfactual": y_cf, "switched": switched,
    }).to_csv(out.with_suffix(".tmp"), index=False)
    diag = {
        "endpoint": endpoint, "repeat": repeat, "fold": fold, "series": int(series),
        "n_scored": len(scored), "n_analogs": len(analogs), "sim_analog": sim,
        "icc_threshold": getattr(model, "icc_threshold", np.nan), "n_icc_nodes": len(icc) - 1,
        "n_parents": len(parents), "n_contributors": len(contributors),
        "no_target_node": no_target,
        "degenerate": len(contributors) == 0, "var_fallback": var_fallback,
        "frac_switched": float(np.mean(switched)), "anchor_regular": y_anchor_regular,
        "anchor_true": float(anchor[endpoint]), "scm_seconds": scm_s,
        "seconds": time.time() - start, "host": HOST,
    }
    # Written aside and renamed, the diagnostics first: sync_parts.sh copies
    # whatever it finds and never revisits a file, so it must never see half of one.
    out.with_suffix(".json.tmp").write_text(json.dumps(diag, indent=1))
    os.replace(out.with_suffix(".json.tmp"), out.with_suffix(".json"))
    os.replace(out.with_suffix(".tmp"), out)
    return diag


def assemble(endpoint, repeat, fold, series_ids) -> bool:
    parts = [part_csv(endpoint, repeat, fold, s) for s in series_ids]
    if not all(p.exists() for p in parts):
        return False
    out = cfg.pred_csv(cfg.CAUSAL_METHOD, endpoint, repeat, fold)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.read_csv(p) for p in parts])[cfg.PRED_COLUMNS].to_csv(out, index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", nargs="+", default=None, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS)
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--reverse", action="store_true",
                        help="walk the series-folds from the last, to meet another machine")
    parser.add_argument("--assemble-only", action="store_true",
                        help="only write the fold files of folds whose parts are all present, "
                             "e.g. after parts have arrived from another machine")
    args = parser.parse_args()

    cfg.ensure_dirs()
    series = pd.read_csv(cfg.DATA_DIR / "series_summary.csv")
    series = series[series["feasible"]]
    endpoints = args.endpoint or [e for e in cfg.TARGET_COLS if e in set(series["endpoint"])]
    of_endpoint = {e: sorted(series.loc[series["endpoint"] == e, "series"]) for e in endpoints}

    # Repeats outermost, so a partial sweep is a complete set of folds sooner.
    # A job is ready once its fold's encoder exists; 03_train_encoder.py may
    # still be running, so the queue is re-scanned whenever it runs low.
    def ready_jobs(seen):
        jobs = [
            {"endpoint": e, "repeat": r, "fold": f, "series": int(s)}
            for r in args.repeat for f in args.fold for e in endpoints for s in of_endpoint[e]
            if (e, r, f, int(s)) not in seen and not part_csv(e, r, f, s).exists()
            and not claimed_elsewhere(e, r, f, int(s))
            and (cfg.MODEL_DIR / f"{e}_r{r}_f{f}" / "encoder.json").exists()
        ]
        return jobs[::-1] if args.reverse else jobs

    def all_encoders():
        return all((cfg.MODEL_DIR / f"{e}_r{r}_f{f}" / "encoder.json").exists()
                   for e in endpoints for r in args.repeat for f in args.fold)

    print(f"{cfg.ACTIVE.label}: {sum(len(v) for v in of_endpoint.values())} feasible series, "
          f"repeats {args.repeat}, {args.workers} workers", flush=True)
    if args.assemble_only:
        n = sum(assemble(e, r, f, of_endpoint[e])
                for e in endpoints for r in args.repeat for f in args.fold)
        print(f"{n} complete folds assembled")
        return
    seen, done, futures = set(), 0, {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            if len(futures) < 2 * args.workers:
                for j in ready_jobs(seen)[: 2 * args.workers - len(futures)]:
                    seen.add((j["endpoint"], j["repeat"], j["fold"], j["series"]))
                    claim = claim_path(j["endpoint"], j["repeat"], j["fold"], j["series"])
                    claim.parent.mkdir(parents=True, exist_ok=True)
                    claim.write_text(HOST)
                    futures[pool.submit(run_job, j)] = j
            if not futures:
                if all_encoders():
                    break
                time.sleep(60)
                continue
            finished, _ = wait(futures, timeout=60, return_when=FIRST_COMPLETED)
            for fut in finished:
                j = futures.pop(fut)
                try:
                    d = fut.result()
                except Exception as exc:  # one bad series must not stop the sweep
                    print(f"FAILED {j}: {type(exc).__name__}: {exc}", flush=True)
                    claim_path(j["endpoint"], j["repeat"], j["fold"], j["series"]).unlink(missing_ok=True)
                    continue
                done += 1
                print(f"[{done}] {d['endpoint']} r{d['repeat']} f{d['fold']} s{d['series']}: "
                      f"{d['n_analogs']} analogs, {d['n_contributors']} contributors, "
                      f"{d['n_scored']} scored, {d['seconds']:.0f}s", flush=True)
                if assemble(j["endpoint"], j["repeat"], j["fold"], of_endpoint[j["endpoint"]]):
                    print(f"  fold complete: {j['endpoint']} r{j['repeat']} f{j['fold']}", flush=True)

    for e in endpoints:
        for r in args.repeat:
            for f in args.fold:
                assemble(e, r, f, of_endpoint[e])


if __name__ == "__main__":
    main()
