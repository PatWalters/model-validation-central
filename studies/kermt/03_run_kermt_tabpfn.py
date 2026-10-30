#!/usr/bin/env python
"""Step 3: KERMT v2 as a frozen encoder, read by TabPFN 3.

The fine-tuned arms of 02_run_kermt.py change every weight of the pre-trained
network. This arm changes none: each molecule becomes the released checkpoint's
512-d cMIM latent -- the mean of the projection head the contrastive objective
is trained on, `fc_mean_logscale`, which the repository's own extractor writes
as projected.npy -- and TabPFN predicts each endpoint in context from the
fold's training vectors and their labels, in one forward pass with no weight
updates. It is the Monroe arm with Monroe's encoder swapped for KERMT's, so set
against Monroe it asks which representation is better, and set against
kermt_ft it asks what fine-tuning buys.

The latent was chosen before any fold was scored, as the vector the v2
objective shapes and the paper studies, rather than picked from the four
mean-pooled readouts by test score.

Two phases, which need different environments:

  --embed   run task/extract_embeddings.py --projection on every molecule in
            data/master.csv and cache the vectors. Needs the kermt environment.
            Done once; the encoder never sees a label.
  (default) for each endpoint x repeat x fold, fit TabPFN on that fold's
            training vectors and predict the fixed test set. Needs tabpfn 8.x,
            which loads TabPFN 3, the head the Monroe arm used.

TabPFN is called exactly as Monroe's own wrapper calls it for regression with
its default ensemble -- one `TabPFNRegressor` with 8 estimators at softmax
temperature 0.9, no PCA, the fold's seed, the mean as the point estimate -- so
the two arms differ in the vectors and nothing else. The fit and test rows are
the masks expansion-ml-comparison/02_run_lightgbm.py uses: the four fifths of
the training set outside the held-out fold that have a value for the endpoint.
The held-out fifth is unused, as it is for Monroe; there is nothing to early-stop.

    ADME_DATASET=biogen python 03_run_kermt_tabpfn.py --embed
    ADME_DATASET=biogen python 03_run_kermt_tabpfn.py
    ADME_DATASET=biogen python 03_run_kermt_tabpfn.py --endpoint LOG_SOL --repeat 0 --fold 0

TabPFN 3's weights are licence-gated: the checkpoint must already be in the
tabpfn cache, or TABPFN_TOKEN set for an account that has accepted the licence.
"""

import argparse
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

ARM = cfg.KERMT_METHOD
METHOD = cfg.KERMT_TABPFN_METHOD
# the cMIM latent's width, from the checkpoint's saved args; asserted after
# embedding, because a different checkpoint would still produce vectors and fit
EMBEDDING_DIM = 512

# Monroe's default_ensemble_specs(): one ensemble, no PCA.
N_ESTIMATORS = 8
SOFTMAX_TEMPERATURE = 0.9

EMBED_NPZ = cfg.EMBED_DIR / f"{ARM}_projected.npz"


def build_embeddings(df: pd.DataFrame, batch_size: int) -> None:
    """Embed every molecule once and cache the vectors in master.csv row order."""
    arm = cfg.KERMT_ARMS[ARM]
    smiles = pd.unique(df[cfg.SMILES_COL])
    print(f"embedding {len(smiles)} unique molecules ({len(df)} rows) with {arm.checkpoint}",
          flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pd.DataFrame({"smiles": smiles}).to_csv(tmp / "smiles.csv", index=False)
        start = time.time()
        subprocess.run(
            [sys.executable, str(cfg.KERMT_REPO / "task/extract_embeddings.py"),
             "-c", str(cfg.WEIGHTS_DIR / arm.checkpoint), "-i", str(tmp / "smiles.csv"),
             "-o", str(tmp / "out"), "--batch_size", str(batch_size),
             "--projection_only", "--no_progress"],
            check=True, cwd=cfg.KERMT_REPO, env={**os.environ, "PYTHONPATH": str(cfg.KERMT_REPO)})
        vectors = np.load(tmp / "out" / "projected.npy").astype(np.float32)
        with open(tmp / "out" / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)
    print(f"embedded {len(vectors)}/{len(smiles)} in {time.time() - start:.0f}s", flush=True)
    if len(vectors) != len(smiles) or list(meta["smiles"]) != list(smiles):
        raise SystemExit(f"got {len(vectors)} vectors for {len(smiles)} molecules -- "
                         "the rows no longer line up")
    if vectors.shape[1] != EMBEDDING_DIM:
        raise SystemExit(f"expected {EMBEDDING_DIM}-d embeddings, got {vectors.shape[1]}")

    lookup = dict(zip(smiles, vectors))
    valid = dict(zip(smiles, np.asarray(meta["valid"], dtype=bool)))
    X = np.stack([lookup[s] for s in df[cfg.SMILES_COL]])
    ok = np.isfinite(X).all(axis=1) & np.array([valid[s] for s in df[cfg.SMILES_COL]])
    cfg.EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMBED_NPZ, X=X, ok=ok, names=df[cfg.ID_COL].to_numpy().astype(str))
    missing = int((~ok).sum())
    print(f"wrote {EMBED_NPZ}  {X.shape}"
          + (f"  ({missing} molecules have no valid vector)" if missing else ""))


def load_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not EMBED_NPZ.exists():
        raise SystemExit(f"{EMBED_NPZ} not found -- run 03_run_kermt_tabpfn.py --embed")
    cached = np.load(EMBED_NPZ, allow_pickle=False)
    X, ok, names = cached["X"], cached["ok"], cached["names"]
    if len(X) != len(df) or not np.array_equal(names, df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(f"{EMBED_NPZ.name} does not line up with {cfg.MASTER_CSV.name} -- "
                         "re-run with --embed")
    return X, ok


def check_tabpfn() -> None:
    """Refuse to run unless the installed tabpfn loads TabPFN 3, as Monroe's arm did."""
    import tabpfn
    from tabpfn.settings import settings

    loads = settings.tabpfn.model_version.value
    if loads != "v3":
        raise SystemExit(f"tabpfn {tabpfn.__version__} would load TabPFN {loads}; the Monroe "
                         "arm used TabPFN 3 -- run this in an environment with tabpfn 8.x")
    print(f"tabpfn {tabpfn.__version__} loads TabPFN {loads}")


def predict_tabpfn(X_fit, y_fit, X_test, seed: int) -> np.ndarray:
    import torch
    from tabpfn import TabPFNRegressor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = TabPFNRegressor(device=device, n_estimators=N_ESTIMATORS,
                            softmax_temperature=SOFTMAX_TEMPERATURE, random_state=seed)
    model.fit(X_fit, y_fit)
    return model.predict(X_test, output_type="mean")


def run_fold(df, X, ok, folds, method, endpoint, repeat, fold, force) -> None:
    out_path = cfg.pred_csv(method, endpoint, repeat, fold)
    if out_path.exists() and not force:
        return

    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for the test molecules

    measured = df[endpoint].notna().to_numpy()
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    fit_mask = measured & ~is_test & (fold_of != fold)
    test_mask = measured & is_test

    dropped = int((fit_mask & ~ok).sum() + (test_mask & ~ok).sum())
    if dropped:
        raise SystemExit(f"{endpoint} r{repeat} f{fold}: {dropped} molecules have no "
                         "embedding, so this fold would not be comparable with the other methods")

    y = df[endpoint].to_numpy()
    pred = predict_tabpfn(X[fit_mask], y[fit_mask], X[test_mask], cfg.fold_seed(repeat, fold))

    test_df = df.loc[test_mask]
    pd.DataFrame(
        {
            "method": method,
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
    parser.add_argument("--embed", action="store_true", help="build the embedding cache and stop")
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true", help="refit folds that already have predictions")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="molecules per encoder forward pass while embedding")
    args = parser.parse_args()

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    if args.embed:
        build_embeddings(df, args.batch_size)
        return

    check_tabpfn()
    (cfg.PRED_DIR / METHOD).mkdir(parents=True, exist_ok=True)
    X, ok = load_embeddings(df)
    folds = pd.read_csv(cfg.FOLD_CSV)
    for endpoint in args.endpoint:
        start = time.time()
        for repeat in args.repeat:
            for fold in args.fold:
                run_fold(df, X, ok, folds, METHOD, endpoint, repeat, fold, args.force)
        n = len(list((cfg.PRED_DIR / METHOD).glob(f"{endpoint}_r*_f*.csv")))
        print(f"{endpoint:<17} {n:>2}/25 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
