#!/usr/bin/env python
"""Step 12: the Trimole-Hybrid arm of the comparison.

Trimole-Hybrid (Luo, Huang, Shao, Yu and Li, Bioinformatics 2026,
doi:10.1101/2026.08.24.746660) is the odd one out here. Every other method in
this repository is an architecture: you hand it molecules and it trains. This one
is a *selection procedure*. Its claim is that no single molecular representation
is best for every ADMET endpoint, so instead of committing to one backbone it
builds a pool of candidate predictors over several molecular views, fits all of
them, keeps whichever scores best on the validation split, and only then touches
the test set. What it reports for an endpoint is therefore a model chosen for
that endpoint, not a model chosen for the benchmark.

This is a reimplementation, and that matters
--------------------------------------------
The other pre-trained arms in this repository -- MEGA-CL, Monroe, Mol-JEPA -- run
the authors' own code against the authors' own checkpoints. This one cannot. The
release at github.com/dchen0212/trimole_hybrid says of itself that it is "not a
one-command full rerun bundle": filesystem paths are placeholders (`<PROJECT_ROOT>`,
`<ENV_ROOT>`), no trained weights or cached embeddings are included, every script
is wired to the TDC benchmark's per-task train/valid/test directory layout, and
`LICENSE_PENDING.md` reserves all rights to the authors.

So what follows is written from the paper and from that source read as a
specification, using checkpoints obtained independently from their original
authors. It is a reimplementation of the method, not a rerun of the paper, and
the report never claims otherwise. Anywhere a number here disagrees with the
published benchmark, this file is the more likely culprit.

The four molecular views
-----------------------
Named after the four evidence streams in the paper, with the checkpoint each one
actually uses:

  sequence     ChemBERTa, `seyonec/ChemBERTa-zinc-base-v1`, the CLS token of the
               last hidden layer (768-d). The model their wrapper names.
  graph        KPGT / LiGhT (Li, Zhao and Zeng, KDD 2022; Nat. Commun. 2023),
               the authors' pre-trained `base.pth`, Apache-2.0 (2304-d).
  3D           UniMol via `unimol_tools` (512-d). The paper calls this branch
               "EPT/3D"; the released wrapper is `unimol.py` and calls UniMol,
               so that is what is run here.
  chemistry    Classical priors: Morgan counts, feature-Morgan, MACCS, Avalon,
               ErG, atom pairs, topological torsions and the RDKit descriptor
               block. This is the MapLight-style feature set their sidecar builds
               (Notwell and Wood, arXiv:2310.00174), and the paper's own ablation
               finds it the single most valuable component.

Where this deviates from the paper
----------------------------------
Two deliberate simplifications, both stated in the report:

1. The learned gated-fusion network is replaced by feature-level concatenation
   plus the selection step. The paper's own Figure 3e is the reason: a naive
   learned combiner trained on the same cached candidate predictions performed
   *worse* than task-wise selection on all 22 tasks, which says the selection is
   what carries the method, not the fusion head.
2. The candidate pool is 5 views x 3 chemistry blocks x 4 backends = 60
   predictors per fold, without the seed-bagging, rank-blending and top-k sweeps
   their prediction zoo adds on top. Those multiply the pool without changing
   what it is selecting over.

Fitting the protocol this repository already uses
------------------------------------------------
The method needs a validation split to select on, and every fold here already has
one: the held-out fifth of the training molecules. So per endpoint, repeat and
fold, each candidate is fit on the same four fifths every other method trains on,
scored on the held-out fifth, and the winner predicts the fixed `ds == 'test'`
set. The test set is never seen during selection, and no candidate is refit on
train+val afterwards -- that would give this arm more training molecules than the
other four, and the comparison rests on every method seeing identical rows.

Selection uses validation R^2. The report scores the resulting predictions with
R^2, Spearman and MAE together, so selecting per metric would produce three
mutually inconsistent answers about what the model predicted; the same reasoning
that fixes Monroe's TabPFN output type to "mean" applies here.

    # once per data set, on the GPU
    python 12_run_trimole.py --embed
    python 12_run_trimole.py --chem            # CPU, and the slow one
    # KPGT lives in its own environment, see 12b_extract_kpgt.py

    python 12_run_trimole.py --jobs 8          # all 225 folds
    python 12_run_trimole.py --endpoint LOG_MGMB --repeat 0 --fold 0   # smoke test
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

METHOD = cfg.TRIMOLE_METHOD

CHEMBERTA_MODEL = "seyonec/ChemBERTa-zinc-base-v1"

# Widths are asserted after loading. A silently different checkpoint would still
# produce vectors of some width and still fit, and the result would be wrong in a
# way no metric would reveal.
VIEW_DIMS = {"chemberta": 768, "kpgt": 2304, "unimol": 512}

# --- the candidate pool --------------------------------------------------
# Three blocks of chemistry priors. They overlap on purpose: the point of the
# pool is that different endpoints want different amounts of it, and letting the
# validation split choose is cheaper than arguing about which is right.
CHEM_BLOCKS = {
    "core_maccs_fcfp": ("morgan", "desc", "maccs", "fcfp"),
    "core_pair_torsion": ("morgan", "desc", "pair", "torsion"),
    "wide_chem": ("morgan", "desc", "maccs", "fcfp", "pair", "torsion", "avalon", "erg"),
}

# Which encoders join the chemistry block. "chem" is the sidecar on its own and
# is the control that says whether the deep encoders are earning their place.
VIEWS = {
    "chem": (),
    "chemberta": ("chemberta",),
    "kpgt": ("kpgt",),
    "unimol": ("unimol",),
    "fused": ("chemberta", "kpgt", "unimol"),
}

BACKENDS = ("xgb", "extratrees", "rf", "ridge")

# Univariate filter before fitting, as in their `fit_selector`. They sweep k over
# 256/512/1024/2048; this fixes the largest, which keeps the pool at 60 rather
# than 240 predictors per fold without changing what is being selected over.
SELECT_K = 2048

N_ESTIMATORS_XGB = 220
N_ESTIMATORS_TREE = 400


# --- featurization -------------------------------------------------------
def chem_primitives(smiles: list[str]) -> dict[str, np.ndarray]:
    """Every chemistry prior, computed once and then sliced into blocks.

    Returned as a dict of named matrices rather than one concatenated array so
    the blocks above can be assembled by name, and so a change to one primitive
    does not silently shift the column offsets of the others.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem import MACCSkeys, rdMolDescriptors, Descriptors
    from rdkit.Avalon import pyAvalonTools
    from rdkit.Chem import rdReducedGraphs

    RDLogger.DisableLog("rdApp.*")

    mols = [Chem.MolFromSmiles(s) for s in smiles]
    bad = [i for i, m in enumerate(mols) if m is None]
    if bad:
        raise SystemExit(f"{len(bad)} SMILES did not parse, first at row {bad[0]}")

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg.FP_RADIUS, fpSize=cfg.FP_SIZE
    )
    fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=cfg.FP_RADIUS,
        fpSize=cfg.FP_SIZE,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    pair_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=cfg.FP_SIZE)
    torsion_gen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=cfg.FP_SIZE)

    def counts(gen) -> np.ndarray:
        return np.asarray(
            [gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.float32
        )

    out = {
        "morgan": counts(morgan_gen),
        "fcfp": counts(fcfp_gen),
        "pair": counts(pair_gen),
        "torsion": counts(torsion_gen),
        "maccs": np.asarray(
            [np.asarray(MACCSkeys.GenMACCSKeys(m), dtype=np.float32) for m in mols],
            dtype=np.float32,
        ),
        "avalon": np.asarray(
            [
                np.asarray(pyAvalonTools.GetAvalonFP(m, nBits=1024), dtype=np.float32)
                for m in mols
            ],
            dtype=np.float32,
        ),
        "erg": np.asarray(
            [rdReducedGraphs.GetErGFingerprint(m) for m in mols], dtype=np.float32
        ),
        "desc": np.asarray(
            [list(Descriptors.CalcMolDescriptors(m).values()) for m in mols],
            dtype=np.float32,
        ),
    }
    return {name: sanitize(block) for name, block in out.items()}


def sanitize(X: np.ndarray) -> np.ndarray:
    """Replace the non-finite values RDKit descriptors produce, and clip.

    A handful of RDKit descriptors return inf or NaN for perfectly reasonable
    molecules (Ipc overflows on large rings, for one). Trees tolerate NaN, ridge
    does not, so this is done once for everything rather than per backend.
    """
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0
    return np.clip(X, -1e6, 1e6, out=X)


# --- caches --------------------------------------------------------------
def build_chem(df: pd.DataFrame) -> None:
    smiles = df[cfg.SMILES_COL].tolist()
    print(f"building chemistry priors for {len(smiles)} molecules")
    start = time.time()
    blocks = chem_primitives(smiles)
    cfg.TRIMOLE_CHEM_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.TRIMOLE_CHEM_NPZ,
        names=df[cfg.ID_COL].to_numpy().astype(str),
        **blocks,
    )
    widths = ", ".join(f"{k} {v.shape[1]}" for k, v in sorted(blocks.items()))
    print(f"wrote {cfg.TRIMOLE_CHEM_NPZ.name} in {time.time() - start:.0f}s  ({widths})")


def build_embeddings(df: pd.DataFrame, batch_size: int) -> None:
    """ChemBERTa and UniMol, in master.csv row order. KPGT is merged separately."""
    import torch

    smiles = df[cfg.SMILES_COL].tolist()
    unique = list(dict.fromkeys(smiles))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"embedding {len(unique)} unique molecules ({len(df)} rows) on {device}")

    views: dict[str, np.ndarray] = {}

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
    model = AutoModel.from_pretrained(CHEMBERTA_MODEL, add_pooling_layer=False)
    model = model.to(device).eval()
    start = time.time()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(unique), batch_size):
            batch = tokenizer(
                unique[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            # The CLS token of the last hidden layer, which is what their
            # wrapper takes -- not the pooler, which is randomly initialised.
            cls = model(**batch).last_hidden_state[:, 0, :]
            chunks.append(cls.float().cpu().numpy())
    views["chemberta"] = np.concatenate(chunks).astype(np.float32)
    print(f"  chemberta {views['chemberta'].shape} in {time.time() - start:.0f}s")

    del model
    torch.cuda.empty_cache() if device == "cuda" else None

    workdir = os.environ.get("TRIMOLE_UNIMOL_WORKDIR", "/tmp/trimole_unimol")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    here = os.getcwd()
    start = time.time()
    try:
        os.chdir(workdir)  # unimol_tools writes ./logs at import time
        from unimol_tools import UniMolRepr

        repr_model = UniMolRepr(data_type="molecule", remove_hs=False, batch_size=batch_size)
        reprs = repr_model.get_repr(unique, return_atomic_reprs=False)
    finally:
        os.chdir(here)
    vectors = reprs["cls_repr"] if isinstance(reprs, dict) else reprs
    views["unimol"] = np.asarray(vectors, dtype=np.float32)
    print(f"  unimol    {views['unimol'].shape} in {time.time() - start:.0f}s")

    index = {s: i for i, s in enumerate(unique)}
    rows = np.asarray([index[s] for s in smiles])
    payload = {name: sanitize(X[rows]) for name, X in views.items()}

    for name, X in payload.items():
        if X.shape[1] != VIEW_DIMS[name]:
            raise SystemExit(
                f"{name}: expected {VIEW_DIMS[name]}-d embeddings, got {X.shape[1]}"
            )

    merge_embeddings(df, payload)


def merge_embeddings(df: pd.DataFrame, payload: dict[str, np.ndarray]) -> None:
    """Write or update the embedding cache without dropping views already in it."""
    existing: dict[str, np.ndarray] = {}
    if cfg.TRIMOLE_EMBED_NPZ.exists():
        cached = np.load(cfg.TRIMOLE_EMBED_NPZ, allow_pickle=False)
        if np.array_equal(cached["names"], df[cfg.ID_COL].to_numpy().astype(str)):
            existing = {k: cached[k] for k in cached.files if k != "names"}
        else:
            print("existing embedding cache does not line up with master.csv, rebuilding")

    existing.update(payload)
    cfg.TRIMOLE_EMBED_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.TRIMOLE_EMBED_NPZ,
        names=df[cfg.ID_COL].to_numpy().astype(str),
        **existing,
    )
    have = ", ".join(f"{k} {v.shape[1]}d" for k, v in sorted(existing.items()))
    print(f"wrote {cfg.TRIMOLE_EMBED_NPZ.name}  ({have})")


def merge_kpgt(df: pd.DataFrame) -> None:
    """Fold the separately extracted KPGT features into the embedding cache."""
    if not cfg.TRIMOLE_KPGT_NPY.exists():
        raise SystemExit(
            f"{cfg.TRIMOLE_KPGT_NPY} not found -- run 12b_extract_kpgt.py in the kpgt env"
        )
    X = sanitize(np.load(cfg.TRIMOLE_KPGT_NPY))
    if len(X) != len(df):
        raise SystemExit(f"KPGT features have {len(X)} rows, master.csv has {len(df)}")
    merge_embeddings(df, {"kpgt": X})


def load_caches(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    for path, flag in ((cfg.TRIMOLE_EMBED_NPZ, "--embed"), (cfg.TRIMOLE_CHEM_NPZ, "--chem")):
        if not path.exists():
            raise SystemExit(f"{path} not found -- run 12_run_trimole.py {flag} first")

    names = df[cfg.ID_COL].to_numpy().astype(str)
    embed = np.load(cfg.TRIMOLE_EMBED_NPZ, allow_pickle=False)
    chem = np.load(cfg.TRIMOLE_CHEM_NPZ, allow_pickle=False)
    for path, cached in ((cfg.TRIMOLE_EMBED_NPZ, embed), (cfg.TRIMOLE_CHEM_NPZ, chem)):
        if not np.array_equal(cached["names"], names):
            raise SystemExit(f"{path.name} does not line up with {cfg.MASTER_CSV.name}")

    views = {k: embed[k] for k in embed.files if k != "names"}
    missing = sorted(set(VIEW_DIMS) - set(views))
    if missing:
        raise SystemExit(
            f"embedding cache is missing {missing} -- the candidate pool would be "
            "smaller than the one the report describes"
        )
    return views, {k: chem[k] for k in chem.files if k != "names"}


# --- the pool ------------------------------------------------------------
def make_backend(backend: str, seed: int, threads: int):
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if backend == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=N_ESTIMATORS_XGB,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            objective="reg:squarederror",
            tree_method="hist",
            device="cpu",
            random_state=seed,
            n_jobs=threads,
        )
    # max_features="sqrt" is theirs, and it is not a detail: sklearn's regression
    # default considers every feature at every split, which on a 2048-column
    # design matrix is both a different model and about forty times the work.
    if backend == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS_TREE, max_features="sqrt", min_samples_leaf=2,
            random_state=seed, n_jobs=threads,
        )
    if backend == "rf":
        return RandomForestRegressor(
            n_estimators=N_ESTIMATORS_TREE, max_features="sqrt", min_samples_leaf=2,
            random_state=seed, n_jobs=threads,
        )
    if backend == "ridge":
        # Their "linear" backend: a fixed alpha rather than a tuned one, scaled
        # because the blocks mix count fingerprints with RDKit descriptors whose
        # ranges differ by orders of magnitude.
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0, random_state=seed))
    # ValueError, not SystemExit: make_backend runs inside the worker pool, and a
    # BaseException there kills the worker without the parent ever being told.
    raise ValueError(f"unknown backend {backend!r}")


# Every column any candidate can use, laid out once in this order. The three
# encoders come first so a candidate's columns are a concatenation of contiguous
# spans rather than a scatter.
PART_ORDER = (
    "chemberta", "kpgt", "unimol",
    "morgan", "desc", "maccs", "fcfp", "pair", "torsion", "avalon", "erg",
)


def build_design(
    views: dict[str, np.ndarray], chem: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    """One matrix holding every column, plus where each named block sits in it.

    The obvious implementation materialises one design matrix per (view, chemistry
    block) pair. That is fifteen matrices which between them store the chemistry
    columns five times over -- once per view -- because every view is the same
    chemistry block with different encoders bolted on. It cost 3.8 GB of shared
    memory and, far worse, fifteen separate row-gathers out of it per fold.

    So the columns are stored once and candidates are described by index instead.
    """
    parts, spans, start = [], {}, 0
    for name in PART_ORDER:
        block = views[name] if name in views else chem[name]
        parts.append(block)
        spans[name] = (start, start + block.shape[1])
        start += block.shape[1]
    return np.concatenate(parts, axis=1), spans


def candidate_columns(spans: dict[str, tuple[int, int]]) -> dict[tuple[str, str], np.ndarray]:
    """Which columns of the design matrix each of the 60 candidates draws on."""
    out = {}
    for block, primitives in CHEM_BLOCKS.items():
        chem_cols = np.concatenate([np.arange(*spans[p]) for p in primitives])
        for view, encoders in VIEWS.items():
            encoder_cols = [np.arange(*spans[e]) for e in encoders]
            out[(view, block)] = (
                np.concatenate(encoder_cols + [chem_cols]) if encoder_cols else chem_cols
            )
    return out


def run_fold(args_tuple) -> dict | None:
    """Fit the whole pool for one fold, select on validation, predict the test set.

    Returns the selection record, or None if the fold was already done.
    """
    (endpoint, repeat, fold, force, threads) = args_tuple

    out_path = cfg.pred_csv(METHOD, endpoint, repeat, fold)
    record_path = selection_record(endpoint, repeat, fold)
    if out_path.exists() and record_path.exists() and not force:
        return None

    from sklearn.feature_selection import f_regression
    from sklearn.metrics import mean_squared_error, r2_score

    df, folds = WORKER["df"], WORKER["folds"]
    design, columns = WORKER["design"], WORKER["columns"]

    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for the test molecules

    measured = df[endpoint].notna().to_numpy()
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    # The same fit mask every other method uses, so the training rows match, plus
    # the held-out fifth this method needs in order to select at all.
    fit_mask = measured & ~is_test & (fold_of != fold)
    val_mask = measured & ~is_test & (fold_of == fold)
    test_mask = measured & is_test

    if val_mask.sum() < 1:
        # A normal exception, never SystemExit. SystemExit inherits from
        # BaseException, which multiprocessing's worker loop does not catch, so
        # raising one here kills the worker silently and leaves the parent
        # waiting forever on a result that will never arrive.
        raise RuntimeError(
            f"{endpoint} r{repeat} f{fold}: no validation molecules, "
            "so there is nothing to select on"
        )

    y = df[endpoint].to_numpy()
    y_fit, y_val = y[fit_mask], y[val_mask]
    seed = cfg.fold_seed(repeat, fold)

    # Gather the fold's rows once, out of the one design matrix, instead of once
    # per candidate. This is the single most expensive thing in the fold.
    A_fit, A_val, A_test = design[fit_mask], design[val_mask], design[test_mask]

    # The univariate filter is scored once too. f_regression treats each column
    # independently, so a column's F statistic does not depend on which other
    # columns a candidate happens to include -- selecting the top k within a
    # candidate's own columns from these scores is exactly what fitting
    # SelectKBest on that candidate would have produced. Scored on the training
    # rows only, so the split the selection is judged on stays untouched.
    with np.errstate(divide="ignore", invalid="ignore"):
        f_scores, _ = f_regression(A_fit, y_fit, center=True)
    # Constant columns score NaN; they carry nothing, so they must never win a place.
    f_scores = np.nan_to_num(f_scores, nan=-np.inf, posinf=np.finfo(np.float64).max)

    best = None
    scores = []
    for (view, block), idx in columns.items():
        if len(idx) > SELECT_K:
            top = np.argpartition(f_scores[idx], -SELECT_K)[-SELECT_K:]
            keep = np.sort(idx[top])
        else:
            keep = idx
        X_fit, X_val, X_test = A_fit[:, keep], A_val[:, keep], A_test[:, keep]

        for backend in BACKENDS:
            model = make_backend(backend, seed, threads)
            model.fit(X_fit, y_fit)
            predicted = model.predict(X_val)

            # Rank on negative MSE, not R^2. Within one fold the two give exactly
            # the same ordering -- R^2 is 1 - SSE/SST and SST is a property of
            # the validation labels, identical for every candidate -- but MSE is
            # still defined when SST is zero or the split is down to a couple of
            # molecules, which happens on the smallest endpoint. R^2 is recorded
            # alongside it because it is the readable number.
            score = -float(mean_squared_error(y_val, predicted))
            with np.errstate(invalid="ignore"):
                as_r2 = float(r2_score(y_val, predicted)) if len(y_val) > 1 else float("nan")

            scores.append({"view": view, "block": block, "backend": backend,
                           "val_score": score, "val_r2": as_r2})
            if best is None or score > best["val_score"]:
                best = {
                    "view": view,
                    "block": block,
                    "backend": backend,
                    "val_score": score,
                    "val_r2": as_r2,
                    "n_features": int(X_fit.shape[1]),
                    "y_pred": model.predict(X_test),
                }

    test_df = df.loc[test_mask]
    pd.DataFrame(
        {
            "method": METHOD,
            "endpoint": endpoint,
            "repeat": repeat,
            "fold": fold,
            cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(),
            cfg.SMILES_COL: test_df[cfg.SMILES_COL].to_numpy(),
            "y_true": y[test_mask],
            "y_pred": np.asarray(best["y_pred"], dtype=float),
        },
        columns=cfg.PRED_COLUMNS,
    ).to_csv(out_path, index=False)

    record = {
        "endpoint": endpoint,
        "repeat": repeat,
        "fold": fold,
        "n_fit": int(fit_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(test_mask.sum()),
        "selected_view": best["view"],
        "selected_block": best["block"],
        "selected_backend": best["backend"],
        "selected_val_r2": best["val_r2"],
        "n_features": best["n_features"],
        "pool": scores,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record))
    return record


def selection_record(endpoint: str, repeat: int, fold: int) -> Path:
    """Which candidate won, kept beside the predictions but out of the glob.

    04_collect_metrics.py sweeps predictions/<ds>/*/*.csv, so these live under
    results/ instead -- they are an outcome of the method, not a prediction.
    """
    return cfg.RESULTS_DIR / "trimole_selection" / f"{endpoint}_r{repeat}_f{fold}.json"


WORKER: dict = {}


def init_worker(threads: int) -> None:
    """Cap each worker's own threading.

    The design matrix is deliberately *not* built here. It is built once in the
    parent and inherited through fork, so every worker reads one copy rather than
    allocating its own.
    """
    import os as _os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        _os.environ[var] = str(threads)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--embed", action="store_true",
                        help="build the ChemBERTa and UniMol caches and stop")
    parser.add_argument("--chem", action="store_true",
                        help="build the chemistry-prior cache and stop")
    parser.add_argument("--merge-kpgt", action="store_true",
                        help="fold trimole_kpgt.npy into the embedding cache and stop")
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true",
                        help="refit folds that already have predictions")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="molecules per encoder forward pass while embedding")
    parser.add_argument("--jobs", type=int, default=1,
                        help="folds fitted in parallel (this arm is CPU bound)")
    parser.add_argument("--threads", type=int, default=1,
                        help="threads per fold worker. sklearn's loky backend "
                             "refuses to thread inside pool workers, so with "
                             "--jobs > 1 the cores are better spent on workers")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)

    if args.embed:
        build_embeddings(df, args.batch_size)
        return
    if args.chem:
        build_chem(df)
        return
    if args.merge_kpgt:
        merge_kpgt(df)
        return

    (cfg.PRED_DIR / METHOD).mkdir(parents=True, exist_ok=True)
    views, chem = load_caches(df)
    folds = pd.read_csv(cfg.FOLD_CSV)

    # Fold-major, not endpoint-major. Endpoints differ in size by a factor of
    # twenty here, so grouping by endpoint hands every worker a LogD fold at once
    # and then every worker a LOG_MGMB fold, which both unbalances the pool and
    # means nothing finishes for the first quarter hour. Interleaving gives each
    # batch a mix of sizes and makes the progress line informative early.
    jobs = [
        (endpoint, repeat, fold, args.force, args.threads)
        for repeat in args.repeat
        for fold in args.fold
        for endpoint in args.endpoint
    ]
    pool_size = len(VIEWS) * len(CHEM_BLOCKS) * len(BACKENDS)
    print(
        f"{len(jobs)} folds x {pool_size} candidates "
        f"({len(VIEWS)} views x {len(CHEM_BLOCKS)} chemistry blocks x {len(BACKENDS)} backends)"
    )

    # Built in the parent, before any fork, so every worker shares one copy.
    print("building the design matrix", flush=True)
    design, spans = build_design(views, chem)
    WORKER["df"] = df
    WORKER["folds"] = folds
    WORKER["design"] = design
    WORKER["columns"] = candidate_columns(spans)
    print(
        f"{design.shape[1]:,} columns x {design.shape[0]:,} molecules, "
        f"{design.nbytes / 1e9:.2f} GB, shared across workers"
    )

    start = time.time()
    if args.jobs > 1:
        import multiprocessing as mp

        # fork, so the matrices above are inherited copy-on-write. Nothing writes
        # to them, so the pages are never actually copied.
        ctx = mp.get_context("fork")
        with ctx.Pool(args.jobs, initializer=init_worker, initargs=(args.threads,)) as pool:
            for i, record in enumerate(pool.imap_unordered(run_fold, jobs), 1):
                announce(record, i, len(jobs), start)
    else:
        init_worker(args.threads)
        for i, job in enumerate(jobs, 1):
            announce(run_fold(job), i, len(jobs), start)

    done = len(list((cfg.PRED_DIR / METHOD).glob("*_r*_f*.csv")))
    print(f"\n{done} fold predictions in {(time.time() - start) / 60:.1f} min")


def announce(record: dict | None, i: int, total: int, start: float) -> None:
    if record is None:
        return
    elapsed = (time.time() - start) / 60
    print(
        f"[{i:>3}/{total}] {record['endpoint']:<16} r{record['repeat']} f{record['fold']}  "
        f"{record['selected_view']:>9} / {record['selected_block']:<17} "
        f"{record['selected_backend']:<10} val R2 {record['selected_val_r2']:+.3f}  "
        f"({elapsed:.1f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
