#!/usr/bin/env python
"""Step 1: the PT-GIN arm.

PT-GIN (Money-Kyrle et al., arXiv 2605.10722) is a Graph Isomorphism Network
pre-trained on 462,189 QMugs molecules to predict each molecule's own 2048-bit
hashed ECFP4, as 2048 binary classifications. The pre-training target is computed
from the molecule itself, so no experimental measurement of any kind enters the
objective.

Downstream the encoder is frozen. Each layer's output is graph-pooled, the
per-layer vectors are concatenated into one embedding, and LightGBM predicts the
endpoint from it. Nothing is fine-tuned. That makes this arm the same shape as
the LightGBM baseline -- one single-task regressor per endpoint per fold -- with
the fingerprint swapped for the frozen network's embedding, which is exactly the
comparison the paper is built around.

Three phases:

  --embed    standardise every molecule the way the authors standardised their
             pre-training and benchmark data, then run all ten released
             checkpoints over them and cache one embedding matrix per checkpoint.
             Done once per data set; the encoder never sees a label.
  --select   for every endpoint, checkpoint and fold, fit LightGBM on the four
             fifths and score it on the held-out fifth. The checkpoint with the
             best mean validation R^2 over the 25 folds becomes that endpoint's
             PT-GIN. The test set plays no part in the choice.
  (default)  refit the chosen checkpoint on each of the 25 folds and predict the
             fixed test set.

Why a selection step exists at all: the paper does not have one PT-GIN. It
pre-trains a grid of maximum radius by vocabulary size and picks, per task,
whichever pre-trained model does best in downstream hyperparameter tuning. Ten of
those checkpoints are released and none of them dominates -- on the authors' own
Biogen results four different ones win somewhere. Reproducing the method means
reproducing the choice, and the validation fifth is where this protocol allows a
choice to be made.

The fit and test rows are taken with the same masks as the LightGBM baseline, so
PT-GIN trains on exactly the molecules every other method trains on: the four
fifths of `ds == 'train'` outside the held-out fold that have a value for the
endpoint. The validation fifth is used only by --select.

LightGBM runs at library defaults, as it does for the Morgan-fingerprint
baseline. The paper Optuna-tunes it, but it tunes it identically for every
representation it compares, and leaving both arms here untuned preserves that
while keeping the only difference between `lgbm` and `ptgin` the thing being
studied.

    PTGIN_HOME=~/software/topological-pretraining python 01_run_ptgin.py --embed
    python 01_run_ptgin.py --select --jobs 16
    python 01_run_ptgin.py --jobs 16
    python 01_run_ptgin.py --endpoint LOG_MGMB --repeat 0 --fold 0   # smoke test
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg

# LightGBM names its own columns internally, so sklearn's validation warns on
# every predict from a plain array. It fires once per fit -- 3,750 times in a
# --select sweep -- and says nothing about the model.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

METHOD = cfg.PTGIN_METHOD

# The authors' standardisation, from config/base/standardizer.yaml. Their
# vocabulary was ranked over molecules put through exactly this, so the tokens a
# new molecule maps to only mean what they meant in pre-training if it goes
# through it too.
STANDARDIZER_SETTINGS = {
    "sanitize": True,
    "fragment_parent": True,
    "neutralize": True,
    "reionize": False,
    "canonical_tautomer": True,
    "keep_chirality": True,
}


def ptgin_modules():
    """Import the authors' package, failing loudly if the checkpoints are absent."""
    missing = [n for n in cfg.CHECKPOINTS if not cfg.checkpoint_path(n).exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(cfg.CHECKPOINTS)} checkpoints are not in "
            f"{cfg.CHECKPOINT_DIR}, starting with {missing[0]}.\n"
            f"Clone https://github.com/oxpig/topological-pretraining and point "
            f"PTGIN_HOME at it (currently {cfg.PTGIN_HOME})."
        )
    if str(cfg.PTGIN_HOME) not in sys.path:
        sys.path.insert(0, str(cfg.PTGIN_HOME))

    from topological_pretraining.data.mol import Standardizer
    from topological_pretraining.featurization.pretrained import PreTrainedGNN

    return Standardizer, PreTrainedGNN


def standardised_mols(df: pd.DataFrame, workers: int | None):
    """Every molecule, put through the authors' standardisation.

    Returns a list the length of `df`, with None where a molecule failed. The
    failures are kept as holes rather than dropped so a later fold can name the
    molecule it is missing instead of quietly training on a smaller set.
    """
    from rdkit import Chem, RDLogger

    Standardizer, _ = ptgin_modules()
    RDLogger.DisableLog("rdApp.*")

    n_jobs = workers if workers is not None else -1
    standardizer = Standardizer(n_jobs=n_jobs, **STANDARDIZER_SETTINGS)

    start = time.time()
    mols = [Chem.MolFromSmiles(smi) for smi in df[cfg.SMILES_COL]]
    unparsed = sum(m is None for m in mols)
    out = standardizer([m for m in mols if m is not None])

    result, cursor = [], 0
    for mol in mols:
        if mol is None:
            result.append(None)
        else:
            result.append(out[cursor])
            cursor += 1
    failed = sum(m is None for m in result)
    print(f"standardised {len(result) - failed}/{len(result)} molecules in "
          f"{time.time() - start:.0f}s"
          + (f" ({unparsed} unparseable SMILES)" if unparsed else ""))
    return result


def build_embeddings(df: pd.DataFrame, names: list[str], workers: int | None,
                     batch: int, force: bool) -> None:
    """Cache one embedding matrix per checkpoint, in master.csv row order.

    Standardisation is the expensive half and does not depend on the checkpoint,
    so it happens once and all ten networks read the same molecules.
    """
    import torch

    todo = [n for n in names if force or not cfg.embedding_npz(n).exists()]
    if not todo:
        print("every checkpoint already has a cached embedding")
        return

    _, PreTrainedGNN = ptgin_modules()
    mols = standardised_mols(df, workers)
    rows = [i for i, mol in enumerate(mols) if mol is not None]
    ready = [mols[i] for i in rows]
    ids = df[cfg.ID_COL].to_numpy().astype(str)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.EMBED_DIR.mkdir(parents=True, exist_ok=True)

    for name in todo:
        start = time.time()
        model = PreTrainedGNN(
            path=str(cfg.checkpoint_path(name)),
            embed_state=cfg.EMBED_STATE,
            layer_pool_type=cfg.LAYER_POOL,
            device=device,
            asarray=True,
        )
        width = int(model.model.out_shape)

        X = np.full((len(df), width), np.nan, dtype=np.float32)
        for start_row in range(0, len(ready), batch):
            chunk = ready[start_row:start_row + batch]
            vectors = np.asarray(model(chunk), dtype=np.float32)
            X[rows[start_row:start_row + batch]] = vectors

        # A molecule the featurizer cannot build a graph for comes back as a row
        # of NaN rather than being dropped, so a later fold can name what it is
        # missing instead of quietly training on a smaller set.
        ok = np.isfinite(X).all(axis=1)
        np.savez_compressed(cfg.embedding_npz(name), X=X, ok=ok, names=ids)
        missing = int((~ok).sum())
        print(f"{name:<30} {width:>5}-d  {time.time() - start:6.0f}s"
              + (f"  ({missing} molecules failed)" if missing else ""), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def load_embeddings(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    path = cfg.embedding_npz(name)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 01_run_ptgin.py --embed first")
    cached = np.load(path, allow_pickle=False)
    X, ok, ids = cached["X"], cached["ok"], cached["names"]
    if len(X) != len(df) or not np.array_equal(ids, df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(
            f"{path.name} does not line up with {cfg.MASTER_CSV.name} -- "
            "re-run with --embed"
        )
    return X, ok


def fold_masks(df: pd.DataFrame, folds: pd.DataFrame, endpoint: str,
               repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The fit, validation and test rows of one fold.

    `fit` and `test` are the two masks 02_run_lightgbm.py uses in the source
    repository, so the training molecules match every other arm exactly. `val` is
    the held-out fifth, which the baselines with a training loop early-stop on and
    which --select scores checkpoints on.
    """
    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for the test molecules

    measured = df[endpoint].notna().to_numpy()
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    fit = measured & ~is_test & (fold_of != fold)
    val = measured & ~is_test & (fold_of == fold)
    return fit, val, measured & is_test


def check_embedded(mask: np.ndarray, ok: np.ndarray, what: str, where: str) -> None:
    dropped = int((mask & ~ok).sum())
    if dropped:
        raise SystemExit(
            f"{where}: {dropped} {what} molecules have no PT-GIN embedding, so "
            "this fold would not be comparable with the other methods"
        )


def regressor(repeat: int, fold: int) -> LGBMRegressor:
    """LightGBM at library defaults, as in the Morgan-fingerprint baseline."""
    return LGBMRegressor(n_jobs=1, verbose=-1, random_state=cfg.fold_seed(repeat, fold))


def score_checkpoint(df, X, ok, folds, endpoint, name, repeat, fold) -> dict:
    """Fit one checkpoint on one fold and score it on the validation fifth."""
    fit, val, _ = fold_masks(df, folds, endpoint, repeat, fold)
    check_embedded(fit | val, ok, "training", f"{name} {endpoint} r{repeat} f{fold}")

    y = df[endpoint].to_numpy()
    model = regressor(repeat, fold)
    model.fit(X[fit], y[fit])
    pred = model.predict(X[val])
    return {
        "endpoint": endpoint,
        "checkpoint": name,
        "repeat": repeat,
        "fold": fold,
        "n_fit": int(fit.sum()),
        "n_val": int(val.sum()),
        "val_r2": r2_score(y[val], pred),
        "val_mae": mean_absolute_error(y[val], pred),
    }


def select_checkpoints(df: pd.DataFrame, folds: pd.DataFrame, endpoints: list[str],
                       names: list[str], repeats, fold_ids, jobs: int) -> pd.DataFrame:
    """Score every checkpoint on every fold's validation fifth, and pick one each.

    Writes the whole table, not just the winner: the margin between the best and
    the second best is the thing that says whether the selection step is doing
    any work, and it is only visible if the losers are kept.
    """
    from joblib import Parallel, delayed

    rows = []
    for name in names:
        start = time.time()
        X, ok = load_embeddings(df, name)
        results = Parallel(n_jobs=jobs, prefer="processes")(
            delayed(score_checkpoint)(df, X, ok, folds, endpoint, name, repeat, fold)
            for endpoint in endpoints
            for repeat in repeats
            for fold in fold_ids
        )
        rows.extend(results)
        print(f"{name:<30} {len(results):>4} fits  {time.time() - start:6.0f}s", flush=True)
        del X

    table = pd.DataFrame(rows)
    cfg.SELECTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.SELECTION_CSV, index=False)
    print(f"\nwrote {cfg.SELECTION_CSV.relative_to(cfg.PROJECT_DIR)} ({len(table)} rows)")
    return table


def choose(table: pd.DataFrame) -> pd.DataFrame:
    """Each endpoint's checkpoint: the best mean validation R^2 over its folds."""
    mean = (
        table.groupby(["endpoint", "checkpoint"])["val_r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    if mean["checkpoint"].nunique() < 2:
        raise SystemExit(
            "selection needs at least two checkpoints to choose between -- "
            "re-run --select without restricting --checkpoint"
        )

    rows = []
    for endpoint, group in mean.groupby("endpoint"):
        ranked = group.sort_values("mean", ascending=False).reset_index(drop=True)
        best, runner_up = ranked.iloc[0], ranked.iloc[1]
        rows.append(
            {
                "endpoint": endpoint,
                "checkpoint": best["checkpoint"],
                "radius": cfg.checkpoint_radius(best["checkpoint"]),
                "vocab": cfg.checkpoint_vocab(best["checkpoint"]),
                "val_r2": best["mean"],
                "val_r2_sd": best["std"],
                "n_folds": int(best["count"]),
                "runner_up": runner_up["checkpoint"],
                "runner_up_val_r2": runner_up["mean"],
                "margin": best["mean"] - runner_up["mean"],
                "worst_val_r2": ranked.iloc[-1]["mean"],
            }
        )
    order = {e: i for i, e in enumerate(cfg.TARGET_COLS)}
    choice = pd.DataFrame(rows).sort_values("endpoint", key=lambda s: s.map(order))
    choice.to_csv(cfg.CHOICE_CSV, index=False)
    print(f"wrote {cfg.CHOICE_CSV.relative_to(cfg.PROJECT_DIR)}\n")

    print(f"{'endpoint':<17} {'checkpoint':<30} {'val R2':>7} {'margin':>8} {'spread':>8}")
    for _, row in choice.iterrows():
        print(f"{row['endpoint']:<17} {row['checkpoint']:<30} {row['val_r2']:7.3f} "
              f"{row['margin']:8.3f} {row['val_r2'] - row['worst_val_r2']:8.3f}")
    return choice


def load_choice() -> dict[str, str]:
    if not cfg.CHOICE_CSV.exists():
        raise SystemExit(
            f"{cfg.CHOICE_CSV} not found -- run 01_run_ptgin.py --select first"
        )
    choice = pd.read_csv(cfg.CHOICE_CSV)
    return dict(zip(choice["endpoint"], choice["checkpoint"]))


def run_fold(df, X, ok, folds, endpoint, repeat, fold, force) -> None:
    out_path = cfg.pred_csv(METHOD, endpoint, repeat, fold)
    if out_path.exists() and not force:
        return

    fit, _, test = fold_masks(df, folds, endpoint, repeat, fold)
    check_embedded(fit | test, ok, "fit or test", f"{endpoint} r{repeat} f{fold}")

    y = df[endpoint].to_numpy()
    model = regressor(repeat, fold)
    model.fit(X[fit], y[fit])
    pred = model.predict(X[test])

    test_df = df.loc[test]
    pd.DataFrame(
        {
            "method": METHOD,
            "endpoint": endpoint,
            "repeat": repeat,
            "fold": fold,
            cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(),
            cfg.SMILES_COL: test_df[cfg.SMILES_COL].to_numpy(),
            "y_true": y[test],
            "y_pred": np.asarray(pred, dtype=float),
        },
        columns=cfg.PRED_COLUMNS,
    ).to_csv(out_path, index=False)


def predict(df, folds, endpoints, repeats, fold_ids, force, jobs) -> None:
    from joblib import Parallel, delayed

    chosen = load_choice()
    (cfg.PRED_DIR / METHOD).mkdir(parents=True, exist_ok=True)

    # Grouped by checkpoint so each embedding matrix is loaded once, not once per
    # endpoint that happens to have selected it.
    by_checkpoint: dict[str, list[str]] = {}
    for endpoint in endpoints:
        if endpoint not in chosen:
            raise SystemExit(f"no checkpoint selected for {endpoint} -- re-run --select")
        by_checkpoint.setdefault(chosen[endpoint], []).append(endpoint)

    for name, its_endpoints in by_checkpoint.items():
        start = time.time()
        X, ok = load_embeddings(df, name)
        Parallel(n_jobs=jobs, prefer="processes")(
            delayed(run_fold)(df, X, ok, folds, endpoint, repeat, fold, force)
            for endpoint in its_endpoints
            for repeat in repeats
            for fold in fold_ids
        )
        done = sum(
            len(list((cfg.PRED_DIR / METHOD).glob(f"{e}_r*_f*.csv"))) for e in its_endpoints
        )
        print(f"{name:<30} {', '.join(its_endpoints):<40} {done:>3} folds "
              f"({time.time() - start:.0f}s)", flush=True)
        del X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embed", action="store_true",
                        help="build the embedding caches and stop")
    parser.add_argument("--select", action="store_true",
                        help="score every checkpoint on validation, pick one per endpoint, stop")
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--checkpoint", nargs="+", default=cfg.CHECKPOINTS,
                        choices=cfg.CHECKPOINTS, help="restrict --embed and --select")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                        help="parallel LightGBM fits (default: half the cores)")
    parser.add_argument("--workers", type=int, default=None,
                        help="standardisation processes (default: every core)")
    parser.add_argument("--batch", type=int, default=1000,
                        help="molecules per embedding chunk")
    parser.add_argument("--force", action="store_true",
                        help="rebuild caches / refit folds that already exist")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 00_import_baselines.py first")

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)

    if args.embed:
        build_embeddings(df, args.checkpoint, args.workers, args.batch, args.force)
        return

    folds = pd.read_csv(cfg.FOLD_CSV)

    if args.select:
        table = select_checkpoints(df, folds, args.endpoint, args.checkpoint,
                                   args.repeat, args.fold, args.jobs)
        choose(table)
        return

    predict(df, folds, args.endpoint, args.repeat, args.fold, args.force, args.jobs)


if __name__ == "__main__":
    main()
