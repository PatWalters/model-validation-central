#!/usr/bin/env python
"""Step 4: the Monroe + TabPFN 3.5 arm -- 300 in-context fits, nothing trained.

Monroe (Banaszewski and Fitzgibbon, arXiv 2608.18982) is a 58.5 M-parameter GRIT
graph transformer pre-trained on 81 M PM6 molecules and 1,089 PCBA assays. What
makes it different from the other arms here is that nothing is trained
downstream: the encoder is frozen, each molecule becomes one 720-d vector, and
TabPFN predicts the endpoint *in context* -- handed the fold's training
embeddings together with their labels, it returns the test predictions in a
single forward pass, with no weight updates and no per-fold hyperparameters.

The head is TabPFN 3.5 (Prior Labs, 15 September 2026), which the `tabpfn`
package makes the default from version 9.0.0 and which is the strongest of the
three heads `studies/expansion-ml-comparison` prices against each other. A
preflight check refuses to run if the installed `tabpfn` would load a different
checkpoint, because the checkpoint is chosen by the library rather than by
anything in this script.

Two phases, as in that study:

  --embed   featurize every molecule in data/master.csv (RDKit ETKDG + MMFF94s
            conformer, then the encoder) and cache the embeddings. Done once and
            reused by all 300 folds, because the encoder never sees a label, and
            because a molecule assayed against two targets is embedded once.
  (default) for each target, scheme and fold, fit TabPFN on that fold's training
            embeddings and predict its test molecules.

The fit rows are the fold's `train` indices and nothing else -- the same
molecules every other arm fits on. The validation molecules are unused: there is
no early stopping to do when there is no training loop, and nothing to tune.

Everything about the model is the authors' own: their checkpoint, their
`embed_smiles`, their `fit_predict_tabpfn` and their `default_ensemble_specs`.
The one choice made here is `output_type="mean"`, TabPFN's default and the one
their own OpenADMET example uses. Their evaluation code switches to "median" for
MAE-scored tasks, which would be the right call if MAE were the only metric, but
this report scores one set of predictions with MAE, R^2 and Spearman together and
tuning the point estimate per metric would make the three disagree about what
the model predicted. MAE is the headline metric here, so `--output-type median`
is available and 07_collect_metrics.py will report it as a control if run.

    MONROE_HOME=~/software/monroe python 04_run_monroe.py --embed
    python 04_run_monroe.py                                  # all 300 folds
    python 04_run_monroe.py --target 284 --split umap --fold 1   # smoke test

TabPFN's weights are licence-gated per major version. Set TABPFN_TOKEN, and
accept the licence for 3.5 -- a token that covers TabPFN 3 does not cover it.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
from fold_data import FoldData, add_fold_arguments

MONROE_HOME = Path(
    os.environ.get("MONROE_HOME", Path.home() / "software" / "monroe")
).expanduser().resolve()
CHECKPOINT_DIR = MONROE_HOME / "checkpoint"

# The width of Monroe's graph-level embedding. Asserted after loading, because a
# silently different checkpoint would still produce vectors and still fit.
EMBEDDING_DIM = 720

# Which TabPFN the arm is, and the `tabpfn` releases that can serve it.
TABPFN_VERSION = "v3.5"
MIN_TABPFN = (9, 0)


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


def check_head() -> None:
    """Refuse to run if the installed tabpfn would load a different checkpoint."""
    import tabpfn

    installed = tuple(int(part) for part in tabpfn.__version__.split(".")[:2])
    if installed < MIN_TABPFN:
        raise SystemExit(
            f"this arm needs tabpfn >= {'.'.join(map(str, MIN_TABPFN))}, "
            f"found {tabpfn.__version__}"
        )

    from tabpfn.settings import settings

    default = settings.tabpfn.model_version.value
    if default != TABPFN_VERSION:
        raise SystemExit(
            f"this arm wants TabPFN {TABPFN_VERSION}, but tabpfn {tabpfn.__version__} "
            f"would load {default} by default. Run it in the environment that has 3.5, "
            f"or set TABPFN_MODEL_VERSION={TABPFN_VERSION}."
        )
    print(f"tabpfn {tabpfn.__version__} loads TabPFN {default}", flush=True)


def build_embeddings(master: pd.DataFrame, batch_size: int, workers: int | None) -> None:
    """Embed every molecule once and cache the result in master.csv row order.

    Deduplicated by SMILES first: 13,444 rows over ten targets hold fewer
    distinct molecules than that, because a compound assayed against two targets
    appears twice, and conformer generation is the expensive half of this step.

    Molecules the featurizer cannot build a graph for come back as NaN rows with
    `ok` false rather than being dropped, so a later fold can say which molecule
    is missing instead of quietly training on a smaller set than the other arms.
    """
    import torch

    load_ckpt, embed_smiles = monroe_modules()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_ckpt(str(CHECKPOINT_DIR)).to(device).eval()
    print(f"loaded Monroe from {CHECKPOINT_DIR} on {device}")

    smiles = pd.unique(master[cfg.OUT_SMILES_COL])
    print(f"embedding {len(smiles):,} distinct molecules ({len(master):,} rows)", flush=True)

    start = time.time()
    lookup = embed_smiles(list(smiles), encoder, device=device,
                          batch_size=batch_size, n_workers=workers)
    print(f"embedded {len(lookup):,}/{len(smiles):,} in {time.time() - start:.0f}s")

    width = len(next(iter(lookup.values())))
    if width != EMBEDDING_DIM:
        raise SystemExit(f"expected {EMBEDDING_DIM}-d embeddings, got {width}")

    X = np.full((len(master), width), np.nan, dtype=np.float32)
    ok = np.zeros(len(master), dtype=bool)
    for row, smi in enumerate(master[cfg.OUT_SMILES_COL]):
        vector = lookup.get(smi)
        if vector is not None:
            X[row] = vector
            ok[row] = True

    cfg.MONROE_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.MONROE_NPZ, X=X, ok=ok,
                        names=master[cfg.OUT_ID_COL].to_numpy().astype(str))
    missing = int((~ok).sum())
    print(f"wrote {cfg.MONROE_NPZ.name}  {X.shape}"
          + (f"  ({missing} rows failed to featurize)" if missing else ""))


def make_predictor(output_type: str):
    """The function that turns one fold's training rows into test predictions."""
    monroe_modules()  # puts MONROE_HOME on sys.path for monroe.eval.tabpfn
    from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn

    specs = default_ensemble_specs()

    def predict(X_fit, y_fit, X_test, seed):
        return fit_predict_tabpfn(
            X_fit, y_fit, X_test, is_classification=False,
            ensemble_specs=specs, seed=seed, output_type=output_type,
        )

    return predict


def run_fold(data: FoldData, target: int, split: str, fold: int, method: str,
             predictions: Path, force: bool, predict) -> bool:
    out_path = predictions / f"tid_{target}_{split}_f{fold}.csv"
    if out_path.exists() and not force:
        return False

    rows = data.fold(target, split, fold)
    data.usable_embeddings(
        np.concatenate([rows.train, rows.test]), f"tid_{target} {split} f{fold}"
    )

    X, y = data.embeddings, data.y
    pred = predict(X[rows.train], y[rows.train], X[rows.test],
                   cfg.fold_seed(target, split, fold))

    if predictions == cfg.PRED_DIR / method:
        data.write_predictions(method, target, split, fold, rows.test, pred,
                               extra={"n_train": len(rows.train)})
    else:
        # A control run at a non-default point estimate. Written outside
        # predictions/ so 07_collect_metrics.py does not sweep it into the arms.
        out = pd.DataFrame({
            "method": method, "target": target, "split": split, "fold": fold,
            cfg.OUT_ID_COL: data.master[cfg.OUT_ID_COL].to_numpy()[rows.test],
            cfg.OUT_SMILES_COL: data.master[cfg.OUT_SMILES_COL].to_numpy()[rows.test],
            "y_true": y[rows.test], "y_pred": np.asarray(pred, dtype=float),
        }, columns=cfg.PRED_COLUMNS)
        out.to_csv(out_path, index=False)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_fold_arguments(parser)
    parser.add_argument("--embed", action="store_true",
                        help="build the embedding cache and stop")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="molecules per encoder forward pass while embedding")
    parser.add_argument("--workers", type=int, default=None,
                        help="conformer-generation processes (default: every core)")
    parser.add_argument("--output-type", default="mean", choices=["mean", "median"],
                        help="TabPFN point estimate (default: mean, as in Monroe's own "
                             "example). Anything else makes the run a control, written "
                             "to results/sensitivity/ rather than becoming the arm")
    args = parser.parse_args()

    cfg.ensure_dirs()

    if args.embed:
        if not cfg.MASTER_CSV.exists():
            raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 00_prepare_data.py first")
        build_embeddings(pd.read_csv(cfg.MASTER_CSV), args.batch_size, args.workers)
        return

    check_head()
    method = cfg.MONROE35_METHOD
    if args.output_type == "mean":
        predictions = cfg.PRED_DIR / method
    else:
        predictions = cfg.RESULTS_DIR / "sensitivity" / f"{method}_{args.output_type}"
        print(f"control run at output_type={args.output_type!r} -> {predictions}")
    predictions.mkdir(parents=True, exist_ok=True)

    data = FoldData.load(need="monroe")
    predict = make_predictor(args.output_type)

    todo = [(split, fold, target)
            for split in args.split for fold in args.fold for target in args.target]
    print(f"{len(todo)} (target, split, fold) combinations requested", flush=True)

    started, n_run = time.time(), 0
    for i, (split, fold, target) in enumerate(todo, start=1):
        if run_fold(data, target, split, fold, method, predictions, args.force, predict):
            n_run += 1
            if n_run % 10 == 0 or n_run == 1:
                elapsed = time.time() - started
                print(f"  {i}/{len(todo)} done, {elapsed / n_run:.1f}s/fold, "
                      f"eta {(elapsed / n_run * (len(todo) - i)) / 60:.0f} min", flush=True)

    n = len(list(predictions.glob("*.csv")))
    print(f"{method} {n}/{len(cfg.TARGETS) * len(cfg.SPLITS) * len(cfg.FOLDS)} "
          "fold predictions on disk")


if __name__ == "__main__":
    main()
