#!/usr/bin/env python
"""Step 1: one NVIDIA structured-data-models head over the Monroe embeddings.

Every model in NVIDIA's `sdm` package is an in-context learner: it is handed the
fold's labelled training rows and the unlabelled test rows together and returns
the test predictions from one forward pass, with no weight updates and nothing
tuned per task. That is exactly what TabPFN does in the `monroe` arm of the
source comparison, so each model here is dropped into the same slot -- the same
cached 720-d Monroe embedding per molecule, the same folds, the same fit and test
masks, the same per-fold seed -- and the only thing that changes is the head.

How each model is run is NVIDIA's own choice, not this project's. The number of
ensemble members and the autocast precision are the defaults their TabArena
adapter (benchmark/tabular/model.py) uses for each model; the preprocessing is
each model's `default_recipe()`; and the point prediction is what that adapter
returns for regression, the mean over the model's output columns after the
recipe has averaged the ensemble. For TabICLv2 and Kumo Tabular those columns are
999 quantiles, so the point estimate is the mean of the predictive distribution,
the same `output_type="mean"` the TabPFN reference uses. TabFM predicts a single
value, so the mean is that value.

Two defaults are worth knowing about. Kumo Tabular and TabFM keep at most 500
feature columns per ensemble member, after shuffling the columns differently for
each member, so no single member sees all 720 dimensions of the embedding and the
ensemble as a whole covers them. TabICLv2 reads all 720.

    python 01_run_sdm.py --model kumo_small
    python 01_run_sdm.py --model tabiclv2 --endpoint LOG_HLM --repeat 0 --fold 0   # smoke test
    ADME_DATASET=biogen python 01_run_sdm.py --model kumo_large
    python 01_run_sdm.py --model tabfm --accept-tabfm-license --query-chunk 512

The embeddings come from the source repository; run
`00_import_baselines.py --inputs-only` on the GPU machine first.
"""

import argparse
import time

import numpy as np
import pandas as pd

import config as cfg

EMBEDDING_DIM = 720


def load_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.MONROE_NPZ.exists():
        raise SystemExit(f"{cfg.MONROE_NPZ} not found -- run 00_import_baselines.py --inputs-only")
    cached = np.load(cfg.MONROE_NPZ, allow_pickle=False)
    X, ok, names = cached["X"], cached["ok"], cached["names"]
    if len(X) != len(df) or not np.array_equal(names, df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(f"{cfg.MONROE_NPZ.name} does not line up with {cfg.MASTER_CSV.name}")
    if X.shape[1] != EMBEDDING_DIM:
        raise SystemExit(f"expected {EMBEDDING_DIM}-d embeddings, got {X.shape[1]}")
    return X, ok


def build_model(key: str, accept_license: bool, device):
    import sdm

    spec = cfg.SDM_MODELS[key]
    kwargs = dict(spec["kwargs"])
    if spec["cls"] == "TabFM":
        if not accept_license:
            raise SystemExit(
                "TabFM's weights are under the TabFM Non-Commercial License v1.0 "
                "(https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/LICENSE). "
                "Read it, and pass --accept-tabfm-license if you accept it."
            )
        kwargs["accept_license"] = True
    return getattr(sdm.models, spec["cls"])(task="regression", device=device, **kwargs)


def to_table(values: np.ndarray, device, name: str | None = None):
    """A float matrix as an all-numerical TableTensor."""
    import sdm

    columns = [name] if name else [f"e{i:03d}" for i in range(values.shape[1])]
    frame = pd.DataFrame(values.reshape(len(values), -1), columns=columns)
    return sdm.TableTensor.from_pandas(
        df=frame, stypes={c: "numerical" for c in columns}, device=device,
    )


# Chunk sizes tried in turn when a fold runs out of GPU memory. None is the single
# pass, which is always tried first unless --query-chunk forces a size.
FALLBACK_CHUNKS = [None, 1024, 512, 256, 128]


def make_predictor(key: str, model, device, estimator_batch_size: int | None):
    """One fold's test predictions from its training rows.

    A `chunk` splits the test rows into pieces for memory, and only for
    memory. In every one of these models a query row attends to the context rows
    and never to another query row, and the generator is reseeded identically for
    each piece, so every piece sees the same ensemble members with the same column
    shuffles. The result is the single pass computed in parts up to half-precision
    rounding: different batch shapes pick different kernels, and on a LogD fold
    the two differ by at most 3e-3 on a 6.9-unit range. Unchunked, a fold is
    bit-for-bit reproducible.
    """
    import torch

    spec = cfg.SDM_MODELS[key]
    dtype = getattr(torch, spec["autocast"])

    def one_pass(x_context, y_context, X_query, seed):
        generator = torch.Generator(device).manual_seed(seed)
        with torch.amp.autocast(device.type, dtype, enabled=device.type == "cuda"):
            out = model(
                x_context=x_context,
                y_context=y_context,
                x_query=to_table(X_query, device),
                num_estimators=spec["num_estimators"],
                estimator_batch_size=estimator_batch_size,
                generator=generator,
            )
        return out.numerical.float().mean(dim=-1).cpu().numpy()

    def predict(X_fit, y_fit, X_test, seed, chunk=None):
        x_context = to_table(X_fit, device)
        y_context = to_table(y_fit.astype(np.float32), device, name="y")
        step = chunk or len(X_test)
        return np.concatenate([
            one_pass(x_context, y_context, X_test[start:start + step], seed)
            for start in range(0, len(X_test), step)
        ])

    return predict


def predict_with_fallback(predict, X_fit, y_fit, X_test, seed, chunks):
    """Try each chunk size in turn, moving on only when the GPU runs out of memory.

    Returns the predictions and the chunk size that produced them, so every fold
    records whether it was computed in one pass.
    """
    import torch

    for chunk in chunks:
        try:
            return predict(X_fit, y_fit, X_test, seed, chunk), chunk
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  out of memory at chunk={chunk or 'none'}, retrying smaller", flush=True)
    raise SystemExit(f"out of memory even at chunk={chunks[-1]}")


def run_fold(df, X, ok, folds, method, endpoint, repeat, fold, force, predict, timings,
             chunks):
    """Fit and predict one fold; returns the chunk size used, or the first of
    `chunks` when the fold was already done."""
    out_path = cfg.pred_csv(method, endpoint, repeat, fold)
    if out_path.exists() and not force:
        return chunks[0]

    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for the test molecules

    measured = df[endpoint].notna().to_numpy()
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    # The same two masks 02_run_lightgbm.py and 09_run_monroe.py use upstream.
    fit_mask = measured & ~is_test & (fold_of != fold)
    test_mask = measured & is_test

    dropped = int((fit_mask & ~ok).sum() + (test_mask & ~ok).sum())
    if dropped:
        raise SystemExit(f"{endpoint} r{repeat} f{fold}: {dropped} molecules have no embedding")

    y = df[endpoint].to_numpy()
    start = time.time()
    pred, chunk = predict_with_fallback(predict, X[fit_mask], y[fit_mask], X[test_mask],
                                        cfg.fold_seed(repeat, fold), chunks)
    seconds = time.time() - start
    if not np.all(np.isfinite(pred)):
        raise SystemExit(f"{endpoint} r{repeat} f{fold}: non-finite predictions")

    test_df = df.loc[test_mask]
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
    timings.append({"method": method, "endpoint": endpoint, "repeat": repeat, "fold": fold,
                    "n_fit": int(fit_mask.sum()), "n_test": int(test_mask.sum()),
                    "seconds": seconds, "query_chunk": chunk or 0})
    return chunk


def save_timings(timings: list[dict]) -> None:
    if not timings:
        return
    new = pd.DataFrame(timings)
    if cfg.TIMING_CSV.exists():
        old = pd.read_csv(cfg.TIMING_CSV)
        key = ["method", "endpoint", "repeat", "fold"]
        old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(cfg.TIMING_CSV, index=False)
    timings.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(cfg.SDM_MODELS))
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true", help="refit folds that already have predictions")
    parser.add_argument("--estimator-batch-size", type=int, default=1,
                        help="ensemble members per forward pass; 1 (sdm's default) is "
                             "the one that fits every fold in 16 GB")
    parser.add_argument("--query-chunk", type=int, default=None,
                        help="always predict the test rows this many at a time (default: one pass, "
                             "falling back to smaller chunks only on out-of-memory)")
    parser.add_argument("--accept-tabfm-license", action="store_true",
                        help="accept the TabFM Non-Commercial License v1.0 for its weights")
    args = parser.parse_args()

    import torch
    import sdm

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 00_import_baselines.py first")
    cfg.ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(cfg.MASTER_CSV)
    X, ok = load_embeddings(df)
    folds = pd.read_csv(cfg.FOLD_CSV)

    method = cfg.SDM_MODELS[args.model]["method"]
    model = build_model(args.model, args.accept_tabfm_license, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{method}: sdm {getattr(sdm, '__version__', '?')}, {n_params / 1e6:.1f} M parameters "
          f"on {device}, {cfg.SDM_MODELS[args.model]['num_estimators']} estimators", flush=True)
    predict = make_predictor(args.model, model, device, args.estimator_batch_size)
    chunks = [args.query_chunk] if args.query_chunk else FALLBACK_CHUNKS

    timings: list[dict] = []
    for endpoint in args.endpoint:
        start = time.time()
        # Folds of one endpoint are within a few rows of each other in size, so
        # once one needs a chunk the rest start from it rather than failing
        # through the larger sizes again.
        remaining = list(chunks)
        for repeat in args.repeat:
            for fold in args.fold:
                used = run_fold(df, X, ok, folds, method, endpoint, repeat, fold,
                                args.force, predict, timings, remaining)
                remaining = remaining[remaining.index(used):]
        save_timings(timings)
        n = len(list((cfg.PRED_DIR / method).glob(f"{endpoint}_r*_f*.csv")))
        print(f"{endpoint:<17} {n:>2}/25 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
