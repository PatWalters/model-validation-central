#!/usr/bin/env python
"""Step 9: the Monroe arm of the comparison.

Monroe (Banaszewski and Fitzgibbon, arXiv 2608.18982) is a 58.5 M-parameter GRIT
graph transformer pre-trained on 81 M PM6 molecules and 1,089 PCBA assays. What
makes it different from every other method here is that nothing is trained
downstream. The encoder is frozen, each molecule becomes a single 720-d vector,
and TabPFN predicts the endpoint *in context*: it is handed the training
embeddings together with their labels and produces the test predictions in one
forward pass, with no weight updates and no per-task hyperparameters.

So this arm has two phases:

  --embed   featurize every molecule in data/master.csv (RDKit ETKDG + MMFF94s
            conformer, then the encoder) and cache the embeddings. Done once,
            reused by all 225 folds, because the encoder never sees a label.
  (default) for each endpoint x repeat x fold, fit TabPFN on the training
            embeddings of that fold and predict the fixed test set.

The fit and test rows are taken with the same masks as 02_run_lightgbm.py, so
Monroe trains on exactly the molecules every other method trains on: the four
fifths of `ds == 'train'` outside the held-out fold that have a value for the
endpoint. The held-out fifth is unused, as it is for LightGBM -- there is no
early stopping to do when there is no training loop.

Everything about the model is the authors' own: their checkpoint, their
`embed_smiles`, their `fit_predict_tabpfn` and their `default_ensemble_specs`.
The one choice made here is `output_type="mean"`, TabPFN's default and the one
their own OpenADMET example uses. Their evaluation code switches to "median" for
MAE-scored tasks, which is the right call when MAE is the only metric, but this
report scores one set of predictions with R2, Spearman and MAE together, and
tuning the point estimate per metric would make the three disagree about what
the model predicted.

    MONROE_HOME=~/software/monroe python 09_run_monroe.py --embed
    python 09_run_monroe.py                                 # all 225 folds
    python 09_run_monroe.py --endpoint LOG_MGMB --repeat 0 --fold 0   # smoke test

TabPFN's weights are licence-gated. Set TABPFN_TOKEN (and optionally
TABPFN_MODEL_VERSION=v2, whose weights are not gated) before running.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

METHOD = cfg.MONROE_METHOD

MONROE_HOME = Path(
    os.environ.get("MONROE_HOME", Path.home() / "software" / "monroe")
).expanduser().resolve()
CHECKPOINT_DIR = MONROE_HOME / "checkpoint"

# The width of Monroe's graph-level embedding. Asserted after loading, because a
# silently different checkpoint would still produce vectors and still fit.
EMBEDDING_DIM = 720


def monroe_modules():
    """Import Monroe from its checkout, failing loudly if the weights are absent."""
    if not (CHECKPOINT_DIR / "weights.pt").exists():
        raise SystemExit(
            f"no Monroe checkpoint at {CHECKPOINT_DIR}\n"
            f"set MONROE_HOME to the monroe checkout (currently {MONROE_HOME}) and "
            "run `git lfs pull` in it"
        )
    if str(MONROE_HOME) not in sys.path:
        sys.path.insert(0, str(MONROE_HOME))

    from monroe.eval.embed import embed_smiles
    from monroe.model.ckpt import load_ckpt

    return load_ckpt, embed_smiles


def build_embeddings(df: pd.DataFrame, batch_size: int, workers: int | None) -> None:
    """Embed every molecule once and cache the result in master.csv row order.

    Molecules the featurizer cannot build a graph for come back as NaN rows with
    `ok` false rather than being dropped, so a later fold can say which molecule
    is missing instead of quietly training on a smaller set.
    """
    import torch

    load_ckpt, embed_smiles = monroe_modules()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_ckpt(str(CHECKPOINT_DIR)).to(device).eval()
    print(f"loaded Monroe from {CHECKPOINT_DIR} on {device}")

    smiles = pd.unique(df[cfg.SMILES_COL])
    print(f"embedding {len(smiles)} unique molecules ({len(df)} rows)")

    start = time.time()
    lookup = embed_smiles(list(smiles), encoder, device=device,
                          batch_size=batch_size, n_workers=workers)
    print(f"embedded {len(lookup)}/{len(smiles)} in {time.time() - start:.0f}s")

    width = len(next(iter(lookup.values())))
    if width != EMBEDDING_DIM:
        raise SystemExit(f"expected {EMBEDDING_DIM}-d embeddings, got {width}")

    X = np.full((len(df), width), np.nan, dtype=np.float32)
    ok = np.zeros(len(df), dtype=bool)
    for row, smi in enumerate(df[cfg.SMILES_COL]):
        vector = lookup.get(smi)
        if vector is not None:
            X[row] = vector
            ok[row] = True

    cfg.MONROE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.MONROE_NPZ, X=X, ok=ok,
                        names=df[cfg.ID_COL].to_numpy().astype(str))
    missing = int((~ok).sum())
    print(f"wrote {cfg.MONROE_NPZ.name}  {X.shape}"
          + (f"  ({missing} molecules failed to featurize)" if missing else ""))


def load_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.MONROE_NPZ.exists():
        raise SystemExit(f"{cfg.MONROE_NPZ} not found -- run 09_run_monroe.py --embed first")
    cached = np.load(cfg.MONROE_NPZ, allow_pickle=False)
    X, ok, names = cached["X"], cached["ok"], cached["names"]
    if len(X) != len(df) or not np.array_equal(names, df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(
            f"{cfg.MONROE_NPZ.name} does not line up with {cfg.MASTER_CSV.name} -- "
            "re-run with --embed"
        )
    return X, ok


def run_fold(
    df: pd.DataFrame,
    X: np.ndarray,
    ok: np.ndarray,
    folds: pd.DataFrame,
    endpoint: str,
    repeat: int,
    fold: int,
    force: bool,
    ensemble_specs: list[dict],
    output_type: str,
) -> None:
    out_path = cfg.pred_csv(METHOD, endpoint, repeat, fold)
    if out_path.exists() and not force:
        return

    from monroe.eval.tabpfn import fit_predict_tabpfn

    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for the test molecules

    measured = df[endpoint].notna().to_numpy()
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    # The same two masks 02_run_lightgbm.py uses, so the training molecules match.
    fit_mask = measured & ~is_test & (fold_of != fold)
    test_mask = measured & is_test

    dropped = int((fit_mask & ~ok).sum() + (test_mask & ~ok).sum())
    if dropped:
        raise SystemExit(
            f"{endpoint} r{repeat} f{fold}: {dropped} molecules have no Monroe "
            "embedding, so this fold would not be comparable with the other methods"
        )

    y = df[endpoint].to_numpy()
    pred = fit_predict_tabpfn(
        X[fit_mask],
        y[fit_mask],
        X[test_mask],
        is_classification=False,
        ensemble_specs=ensemble_specs,
        seed=cfg.fold_seed(repeat, fold),
        output_type=output_type,
    )

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
            "y_pred": np.asarray(pred, dtype=float),
        },
        columns=cfg.PRED_COLUMNS,
    ).to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embed", action="store_true",
                        help="build the embedding cache and stop")
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true",
                        help="refit folds that already have predictions")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="molecules per encoder forward pass while embedding")
    parser.add_argument("--workers", type=int, default=None,
                        help="conformer-generation processes (default: every core)")
    parser.add_argument("--output-type", default="mean", choices=["mean", "median"],
                        help="TabPFN point estimate (default: mean, as in Monroe's own example)")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)

    if args.embed:
        build_embeddings(df, args.batch_size, args.workers)
        return

    monroe_modules()  # puts MONROE_HOME on sys.path for monroe.eval.tabpfn
    from monroe.eval.tabpfn import default_ensemble_specs

    (cfg.PRED_DIR / METHOD).mkdir(parents=True, exist_ok=True)
    X, ok = load_embeddings(df)
    folds = pd.read_csv(cfg.FOLD_CSV)
    specs = default_ensemble_specs()

    for endpoint in args.endpoint:
        start = time.time()
        for repeat in args.repeat:
            for fold in args.fold:
                run_fold(df, X, ok, folds, endpoint, repeat, fold,
                         args.force, specs, args.output_type)
        n = len(list((cfg.PRED_DIR / METHOD).glob(f"{endpoint}_r*_f*.csv")))
        print(f"{endpoint:<17} {n:>2}/25 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
