#!/usr/bin/env python
"""Step 10: the Mol-JEPA arm of the comparison.

Mol-JEPA (Rottach et al., arXiv 2608.22642) is a multimodal joint embedding
predictive architecture. Rather than augmenting a molecule and asking for
matching views, it takes fourteen *modalities* of the same molecule -- graph,
ECFP, MOE descriptors, xTB and DFT calculations, embeddings borrowed from UMA,
CLOOME, BioXMol, ChemGPT and Boltz-2, and experimental ChEMBL, PCBA and TDC
label vectors -- masks whole modalities out, and trains a transformer to predict
the missing latents from the ones that remain. Only structure is needed at
inference: the released checkpoint takes SMILES and returns the predicted
modality embeddings, a 512-d CLS summary, and the latent tokens.

Like Monroe, this arm trains nothing downstream. The 45.4 M-parameter encoder is
frozen, every molecule becomes one CLS vector, and a tabular in-context model
predicts the endpoint from the fold's training vectors in a single forward pass.

Two heads are run over the same embeddings:

  tabicl   TabICL, the authors' own recommendation on the model card, and the
           arm that appears in the report.
  tabpfn   TabPFN at the settings the Monroe arm uses. Not a separate method in
           the figures -- it goes to results/sensitivity/ and answers one
           question, how much of the Monroe-versus-Mol-JEPA gap is the
           representation and how much is the head.

The fit and test rows are the masks from 02_run_lightgbm.py, so Mol-JEPA trains
on exactly the molecules every other method trains on. The held-out fifth is
unused, as it is for LightGBM and Monroe: there is no early stopping to do.

    MOLJEPA_HOME=... python 10_run_moljepa.py --embed
    python 10_run_moljepa.py                        # 225 folds, TabICL
    python 10_run_moljepa.py --head tabpfn          # the sensitivity pass
    python 10_run_moljepa.py --endpoint LOG_MGMB --repeat 0 --fold 0

The checkpoint is pulled from the HuggingFace hub on first use and ships custom
modeling code, so `trust_remote_code=True` is required. It needs transformers 4:
under transformers 5 the model is built on the meta device and its Epps-Pulley
buffer construction fails.
"""

import argparse
import time

import numpy as np
import pandas as pd

import config as cfg

METHOD = cfg.MOLJEPA_METHOD

HF_MODEL = "Flogrammer/Mol-JEPA"

# The width of the CLS summary. Asserted after the first batch, because a
# different checkpoint would still produce vectors and still fit.
EMBEDDING_DIM = 512


def out_dir(head: str):
    """Where a head's predictions go.

    Only TabICL is a method in the comparison. The TabPFN pass is a control, and
    it deliberately lands outside `predictions/` so that 04_collect_metrics.py
    does not sweep it up and turn it into a seventh bar in every figure.
    """
    if head == "tabicl":
        return cfg.PRED_DIR / METHOD
    return cfg.SENSITIVITY_DIR / f"{METHOD}_{head}"


def build_embeddings(df: pd.DataFrame, batch_size: int) -> None:
    """Embed every molecule once and cache the CLS tokens in master.csv row order."""
    import torch
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(HF_MODEL, trust_remote_code=True).to(device).eval()
    print(f"loaded {HF_MODEL} on {device}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters")

    smiles = list(pd.unique(df[cfg.SMILES_COL]))
    print(f"embedding {len(smiles)} unique molecules ({len(df)} rows)")

    start = time.time()
    vectors, kept = [], []
    for begin in range(0, len(smiles), batch_size):
        chunk = smiles[begin:begin + batch_size]
        try:
            with torch.no_grad():
                cls = model(chunk).cls
        except Exception as exc:  # one bad molecule should not lose the batch
            print(f"  batch at {begin} failed ({exc}), retrying one at a time")
            for smi in chunk:
                try:
                    with torch.no_grad():
                        vectors.append(model([smi]).cls.cpu().numpy())
                    kept.append(smi)
                except Exception:
                    print(f"  could not embed {smi!r}")
            continue
        vectors.append(cls.cpu().numpy())
        kept.extend(chunk)
        if begin and begin % (batch_size * 20) == 0:
            print(f"  {begin}/{len(smiles)}", flush=True)

    stacked = np.concatenate(vectors)
    print(f"embedded {len(kept)}/{len(smiles)} in {time.time() - start:.0f}s")
    if stacked.shape[1] != EMBEDDING_DIM:
        raise SystemExit(f"expected {EMBEDDING_DIM}-d CLS tokens, got {stacked.shape[1]}")

    lookup = dict(zip(kept, stacked))
    X = np.full((len(df), EMBEDDING_DIM), np.nan, dtype=np.float32)
    ok = np.zeros(len(df), dtype=bool)
    for row, smi in enumerate(df[cfg.SMILES_COL]):
        vector = lookup.get(smi)
        if vector is not None:
            X[row] = vector
            ok[row] = True

    cfg.MOLJEPA_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.MOLJEPA_NPZ, X=X, ok=ok,
                        names=df[cfg.ID_COL].to_numpy().astype(str))
    missing = int((~ok).sum())
    print(f"wrote {cfg.MOLJEPA_NPZ.name}  {X.shape}"
          + (f"  ({missing} molecules failed to embed)" if missing else ""))


def load_embeddings(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.MOLJEPA_NPZ.exists():
        raise SystemExit(f"{cfg.MOLJEPA_NPZ} not found -- run 10_run_moljepa.py --embed first")
    cached = np.load(cfg.MOLJEPA_NPZ, allow_pickle=False)
    X, ok, names = cached["X"], cached["ok"], cached["names"]
    if len(X) != len(df) or not np.array_equal(names, df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(
            f"{cfg.MOLJEPA_NPZ.name} does not line up with {cfg.MASTER_CSV.name} -- "
            "re-run with --embed"
        )
    return X, ok


def predict_tabicl(X_fit, y_fit, X_test, seed: int) -> np.ndarray:
    """The model card's own recommendation: TabICL on the CLS features."""
    import torch
    from tabicl import TabICLRegressor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabICLRegressor(random_state=seed, device=device)
    model.fit(X_fit, y_fit)
    return model.predict(X_test)


def predict_tabpfn(X_fit, y_fit, X_test, seed: int) -> np.ndarray:
    """TabPFN at the settings the Monroe arm uses, so the control is like for like."""
    import torch
    from tabpfn import TabPFNRegressor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(device=device, n_estimators=8, softmax_temperature=0.9,
                            random_state=seed)
    model.fit(X_fit, y_fit)
    return model.predict(X_test, output_type="mean")


HEADS = {"tabicl": predict_tabicl, "tabpfn": predict_tabpfn}


def run_fold(
    df: pd.DataFrame,
    X: np.ndarray,
    ok: np.ndarray,
    folds: pd.DataFrame,
    endpoint: str,
    repeat: int,
    fold: int,
    force: bool,
    head: str,
) -> None:
    out_path = out_dir(head) / f"{endpoint}_r{repeat}_f{fold}.csv"
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
        raise SystemExit(
            f"{endpoint} r{repeat} f{fold}: {dropped} molecules have no Mol-JEPA "
            "embedding, so this fold would not be comparable with the other methods"
        )

    y = df[endpoint].to_numpy()
    pred = HEADS[head](X[fit_mask], y[fit_mask], X[test_mask], cfg.fold_seed(repeat, fold))

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
    parser.add_argument("--head", default="tabicl", choices=sorted(HEADS),
                        help="downstream predictor (default: tabicl, the authors' own)")
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true",
                        help="refit folds that already have predictions")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="molecules per encoder forward pass while embedding")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)

    if args.embed:
        build_embeddings(df, args.batch_size)
        return

    destination = out_dir(args.head)
    destination.mkdir(parents=True, exist_ok=True)
    X, ok = load_embeddings(df)
    folds = pd.read_csv(cfg.FOLD_CSV)

    for endpoint in args.endpoint:
        start = time.time()
        for repeat in args.repeat:
            for fold in args.fold:
                run_fold(df, X, ok, folds, endpoint, repeat, fold, args.force, args.head)
        n = len(list(destination.glob(f"{endpoint}_r*_f*.csv")))
        print(f"[{args.head}] {endpoint:<17} {n:>2}/25 folds  ({time.time() - start:.1f}s)",
              flush=True)


if __name__ == "__main__":
    main()
