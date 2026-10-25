#!/usr/bin/env python
"""Step 5: Uni-Mol and Uni-Mol2 as frozen encoders, read by TabPFN 3.

The fine-tuned arms of 01_run_unimol.py change every weight of the pre-trained
network. These two arms change none: each molecule becomes the pre-trained
model's molecule-level (CLS) vector, and TabPFN predicts each endpoint in
context from the fold's training vectors and their labels, in one forward pass
with no weight updates. It is the Monroe arm with Monroe's encoder swapped for
Uni-Mol's, so set against Monroe it asks which representation is better, and
set against the fine-tuned arm it asks what fine-tuning buys.

    unimol_tabpfn    Uni-Mol (v1, mol_pre_all_h_220816), 512-d CLS vector
    unimol2_tabpfn   Uni-Mol2 84M, the same size the fine-tuned arm uses

Two phases, which need different environments:

  --embed   featurize every molecule in data/master.csv with the package's own
            `UniMolRepr` -- its RDKit conformer, hydrogens kept, the same input
            the fine-tuned arm sees -- and cache the vectors. Needs the
            unimol_tools environment. Done once; the encoder never sees a label.
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

    ADME_DATASET=biogen python 05_run_unimol_tabpfn.py --embed --arm unimol2
    ADME_DATASET=biogen python 05_run_unimol_tabpfn.py --arm unimol2
    ADME_DATASET=biogen python 05_run_unimol_tabpfn.py --arm unimol --endpoint LOG_SOL --repeat 0 --fold 0

TabPFN 3's weights are licence-gated: the checkpoint must already be in the
tabpfn cache, or TABPFN_TOKEN set for an account that has accepted the licence.
"""

import argparse
import time

import numpy as np
import pandas as pd

import config as cfg

# The width of each model's CLS vector. Asserted after embedding, because a
# different checkpoint would still produce vectors and still fit.
EMBEDDING_DIM = {cfg.UNIMOL_METHOD: 512, cfg.UNIMOL2_METHOD: 768}

# Monroe's default_ensemble_specs(): one ensemble, no PCA.
N_ESTIMATORS = 8
SOFTMAX_TEMPERATURE = 0.9


def embed_npz(arm: str):
    return cfg.EMBED_DIR / f"{arm}.npz"


def build_embeddings(df: pd.DataFrame, arm: str, batch_size: int) -> None:
    """Embed every molecule once and cache the vectors in master.csv row order."""
    from unimol_tools import UniMolRepr

    spec = cfg.UNIMOL_ARMS[arm]
    extra = {"model_size": spec.model_size} if spec.model_size else {}
    encoder = UniMolRepr(data_type="molecule", batch_size=batch_size, remove_hs=False,
                         model_name=spec.model_name, use_gpu="0", **extra)

    smiles = pd.unique(df[cfg.SMILES_COL])
    print(f"embedding {len(smiles)} unique molecules ({len(df)} rows) with {arm}", flush=True)
    start = time.time()
    vectors = np.asarray(encoder.get_repr(list(smiles)), dtype=np.float32)
    print(f"embedded {len(vectors)}/{len(smiles)} in {time.time() - start:.0f}s", flush=True)
    if len(vectors) != len(smiles):
        raise SystemExit(f"got {len(vectors)} vectors for {len(smiles)} molecules -- "
                         "the rows no longer line up")
    if vectors.shape[1] != EMBEDDING_DIM[arm]:
        raise SystemExit(f"expected {EMBEDDING_DIM[arm]}-d embeddings, got {vectors.shape[1]}")

    lookup = dict(zip(smiles, vectors))
    X = np.stack([lookup[s] for s in df[cfg.SMILES_COL]])
    ok = np.isfinite(X).all(axis=1)

    cfg.EMBED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embed_npz(arm), X=X, ok=ok,
                        names=df[cfg.ID_COL].to_numpy().astype(str))
    missing = int((~ok).sum())
    print(f"wrote {embed_npz(arm)}  {X.shape}"
          + (f"  ({missing} molecules have non-finite vectors)" if missing else ""))


def load_embeddings(df: pd.DataFrame, arm: str) -> tuple[np.ndarray, np.ndarray]:
    path = embed_npz(arm)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 05_run_unimol_tabpfn.py --embed --arm {arm}")
    cached = np.load(path, allow_pickle=False)
    X, ok, names = cached["X"], cached["ok"], cached["names"]
    if len(X) != len(df) or not np.array_equal(names, df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(f"{path.name} does not line up with {cfg.MASTER_CSV.name} -- "
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
    parser.add_argument("--arm", required=True, choices=list(cfg.UNIMOL_ARMS),
                        help="which pre-trained encoder supplies the vectors")
    parser.add_argument("--embed", action="store_true", help="build the embedding cache and stop")
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true", help="refit folds that already have predictions")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="molecules per encoder forward pass while embedding")
    args = parser.parse_args()

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)

    if args.embed:
        build_embeddings(df, args.arm, args.batch_size)
        return

    check_tabpfn()
    method = cfg.TABPFN_METHOD_OF[args.arm]
    (cfg.PRED_DIR / method).mkdir(parents=True, exist_ok=True)
    X, ok = load_embeddings(df, args.arm)
    folds = pd.read_csv(cfg.FOLD_CSV)

    for endpoint in args.endpoint:
        start = time.time()
        for repeat in args.repeat:
            for fold in args.fold:
                run_fold(df, X, ok, folds, method, endpoint, repeat, fold, args.force)
        n = len(list((cfg.PRED_DIR / method).glob(f"{endpoint}_r*_f*.csv")))
        print(f"{endpoint:<17} {n:>2}/25 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
