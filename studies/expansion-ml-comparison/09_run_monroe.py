#!/usr/bin/env python
"""Step 9: the Monroe arm of the comparison.

Monroe (Banaszewski and Fitzgibbon, arXiv 2608.18982) is a 58.5 M-parameter GRIT
graph transformer pre-trained on 81 M PM6 molecules and 1,089 PCBA assays. What
makes it different from every other method here is that nothing is trained
downstream. The encoder is frozen, each molecule becomes a single 720-d vector,
and TabPFN predicts the endpoint *in context*: it is handed the training
embeddings together with their labels and produces the test predictions in one
forward pass, with no weight updates and no per-task hyperparameters.

Three arms come out of this, differing only in the tabular model that reads the
embeddings. `--head v3` is TabPFN 3, the head the Monroe paper was written
against; `--head v3.5` is TabPFN 3.5, released 15 September 2026, which the
`tabpfn` package makes the default from version 9.0.0; `--head tabicl` is
TabICL, which is not a TabPFN at all but the head Mol-JEPA's authors recommend
and the one the `moljepa` arm uses. The encoder, the cached embeddings, the
folds and the masks are identical across all three, so any difference between
them is the head and nothing else -- and with the TabICL head the two
representations and the two heads form a full square.

Each head needs its own environment. The TabPFN checkpoint is chosen by the
installed `tabpfn` version, and a preflight check refuses to run if that and
`--head` disagree; TabICL wants the Mol-JEPA environment, which is where
`tabicl` lives.

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
    python 09_run_monroe.py                                 # all 225 folds, TabPFN 3
    python 09_run_monroe.py --head v3.5                     # the same folds, TabPFN 3.5
    python 09_run_monroe.py --head tabicl                   # the same folds, TabICL
    python 09_run_monroe.py --endpoint LOG_MGMB --repeat 0 --fold 0   # smoke test

The embedding cache is shared: `--embed` is run once and every head reads it.

TabPFN's weights are licence-gated, and each major version is gated separately.
Set TABPFN_TOKEN, and accept the licence for the version being run -- a token
that covers TabPFN 3 does not cover TabPFN 3.5.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

# Which TabPFN checkpoint each head means, the method it is recorded as, and the
# `tabpfn` releases that can serve it. TabPFN 3 was the default up to tabpfn 8.x
# and stays reachable afterwards only by naming it; 3.5 became the default in
# 9.0.0. Neither is installable alongside the other, hence one environment each.
HEADS = {
    "v3": {"method": cfg.MONROE_METHOD, "predictor": "tabpfn",
           "version": "v3", "min_tabpfn": (8, 0)},
    "v3.5": {"method": cfg.MONROE35_METHOD, "predictor": "tabpfn",
             "version": "v3.5", "min_tabpfn": (9, 0)},
    # Not a TabPFN checkpoint at all: TabICL, the head Mol-JEPA's authors
    # recommend, over Monroe's embeddings. Runs in the Mol-JEPA environment,
    # since that is the one with tabicl in it.
    "tabicl": {"method": cfg.MONROE_TABICL_METHOD, "predictor": "tabicl"},
}

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


def out_dir(method: str, softmax_temperature: str | None):
    """Where a run's predictions go.

    A run at the wrapper's own temperature is the arm itself. A run at any other
    temperature is a control, and lands outside `predictions/` so that
    04_collect_metrics.py does not sweep it up and turn it into another bar in
    every figure. The same convention 10_run_moljepa.py uses for its TabPFN head.
    """
    if softmax_temperature is None:
        return cfg.PRED_DIR / method
    return cfg.SENSITIVITY_DIR / f"{method}_t{softmax_temperature}"


def check_head(head: str) -> None:
    """Refuse to run if the installed tabpfn would load a different checkpoint.

    The checkpoint is chosen by the library, not by anything in this script, so a
    run in the wrong environment would quietly write one head's predictions under
    the other's name. `settings.tabpfn.model_version` is what a `TabPFNRegressor`
    built without an explicit `model_path` will load, which is exactly what
    Monroe's wrapper builds.
    """
    spec = HEADS[head]
    if spec["predictor"] == "tabicl":
        import importlib.metadata

        import tabicl  # noqa: F401  -- imported to fail here rather than mid-sweep

        # tabicl does not carry a __version__, so ask the installed distribution.
        print(f"tabicl {importlib.metadata.version('tabicl')}, "
              f"recorded as {spec['method']}")
        return

    import tabpfn

    installed = tuple(int(part) for part in tabpfn.__version__.split(".")[:2])
    if installed < spec["min_tabpfn"]:
        raise SystemExit(
            f"--head {head} needs tabpfn >= {'.'.join(map(str, spec['min_tabpfn']))}, "
            f"found {tabpfn.__version__}"
        )

    from tabpfn.settings import settings

    default = settings.tabpfn.model_version.value
    if default != spec["version"]:
        raise SystemExit(
            f"--head {head} wants TabPFN {spec['version']}, but tabpfn "
            f"{tabpfn.__version__} would load {default} by default. Run this head in "
            f"its own environment, or set TABPFN_MODEL_VERSION={spec['version']}."
        )
    print(f"tabpfn {tabpfn.__version__} loads TabPFN {default}, "
          f"recorded as {spec['method']}")


def make_predictor(head: str, ensemble_specs: list[dict], output_type: str):
    """The function that turns one fold's training rows into test predictions.

    Both heads are in-context: they are handed the fold's training embeddings
    with their labels and return the test predictions from a single forward
    pass, with no weight updates. What differs is which tabular model does it.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if HEADS[head]["predictor"] == "tabicl":
        from tabicl import TabICLRegressor

        def predict(X_fit, y_fit, X_test, seed):
            # The same call 10_run_moljepa.py makes, so the head is identical
            # across the two representations and only the embeddings differ.
            model = TabICLRegressor(random_state=seed, device=device)
            model.fit(X_fit, y_fit)
            return model.predict(X_test)

        return predict

    monroe_modules()  # puts MONROE_HOME on sys.path for monroe.eval.tabpfn
    from monroe.eval.tabpfn import fit_predict_tabpfn

    def predict(X_fit, y_fit, X_test, seed):
        return fit_predict_tabpfn(
            X_fit, y_fit, X_test, is_classification=False,
            ensemble_specs=ensemble_specs, seed=seed, output_type=output_type,
        )

    return predict


def run_fold(
    df: pd.DataFrame,
    X: np.ndarray,
    ok: np.ndarray,
    folds: pd.DataFrame,
    method: str,
    predictions: Path,
    endpoint: str,
    repeat: int,
    fold: int,
    force: bool,
    predict,
) -> None:
    out_path = predictions / f"{endpoint}_r{repeat}_f{fold}.csv"
    if out_path.exists() and not force:
        return

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
    pred = predict(X[fit_mask], y[fit_mask], X[test_mask], cfg.fold_seed(repeat, fold))

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
    parser.add_argument("--embed", action="store_true",
                        help="build the embedding cache and stop")
    parser.add_argument("--head", default="v3", choices=sorted(HEADS),
                        help="which tabular model reads the embeddings (default: v3, "
                             "the TabPFN 3 head the Monroe paper used)")
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
    parser.add_argument("--softmax-temperature", default=None,
                        help="override the wrapper's temperature; \"auto\" takes the "
                             "checkpoint's own. Any value here makes the run a control, "
                             "written to results/<dataset>/sensitivity/ rather than "
                             "becoming an arm of the comparison")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)

    if args.embed:
        build_embeddings(df, args.batch_size, args.workers)
        return

    method = HEADS[args.head]["method"]
    is_tabpfn = HEADS[args.head]["predictor"] == "tabpfn"
    if args.softmax_temperature is not None and not is_tabpfn:
        raise SystemExit(
            f"--softmax-temperature is a TabPFN setting and --head {args.head} is not "
            "a TabPFN head"
        )
    check_head(args.head)
    predictions = out_dir(method, args.softmax_temperature)

    predictions.mkdir(parents=True, exist_ok=True)
    X, ok = load_embeddings(df)
    folds = pd.read_csv(cfg.FOLD_CSV)

    specs: list[dict] = []
    if is_tabpfn:
        monroe_modules()  # puts MONROE_HOME on sys.path for monroe.eval.tabpfn
        from monroe.eval.tabpfn import default_ensemble_specs

        specs = default_ensemble_specs()
    if args.softmax_temperature is not None:
        # "auto" is TabPFN's own word for "whatever the checkpoint declares"; anything
        # else is a temperature, and is passed as the float the estimator expects.
        value = args.softmax_temperature
        value = value if value == "auto" else float(value)
        specs = [{**spec, "softmax_temperature": value} for spec in specs]
        print(f"control run at softmax_temperature={value!r} -> {predictions}")

    predict = make_predictor(args.head, specs, args.output_type)

    for endpoint in args.endpoint:
        start = time.time()
        for repeat in args.repeat:
            for fold in args.fold:
                run_fold(df, X, ok, folds, method, predictions, endpoint, repeat,
                         fold, args.force, predict)
        n = len(list(predictions.glob(f"{endpoint}_r*_f*.csv")))
        print(f"{endpoint:<17} {n:>2}/25 folds  ({time.time() - start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
