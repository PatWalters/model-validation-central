#!/usr/bin/env python
"""Step 6: the paper's own six arms, imported from its released per-fold metrics.

The upstream release publishes, at exactly the granularity this study works at,
the test MAE, R^2 and Spearman rho of every fold of every one of its six arms:
k-NN, SVM, RF, XGBoost, MLP and a graph neural network, each tuned with 50
Optuna trials per target, scheme and fold. Those numbers sit on the same folds
the four arms here are fitted on, so they can go straight onto the same axis.

They enter as metrics, not as predictions, because predictions were not
released. That has three consequences, and the report states all three wherever
it names a `pub_` arm:

  * they are numbers taken on trust. Nothing here re-computed them, and the
    metric definitions are assumed to be the standard ones. The one arm that can
    be checked is k-NN, which this study also re-runs: 250 of its 300 folds
    reproduce to floating-point tolerance, which is the evidence that the folds
    and the metric agree between the two sources.
  * they cannot be re-scored. A metric this study might want later -- a
    different point estimate, a per-molecule error analysis -- is not
    recoverable for them.
  * on the `diverse_25` scheme they are not strictly on the same training set.
    Upstream did not record that scheme's validation split (see
    `config.read_split`), so the published arms trained on some 237 of a
    263-molecule pool and the arms here train on a different 237 of the same
    pool, against an identical test set. Comparable, but not identical, and the
    50 k-NN folds that fail to reproduce are all on this scheme.

Their arms were each given 50 trials of hyperparameter search per fold and the
four arms here run at their libraries' defaults. That asymmetry favours the
published arms and is worth saying out loud rather than correcting for.

    python 06_import_published.py
"""

import argparse

import pandas as pd

import config as cfg

# Their column names for the three metrics this study scores on.
METRIC_FROM = {"mae": "test_mae", "r2": "test_r2", "spearman": "test_spearman_r"}


def load_published() -> pd.DataFrame:
    """The five ML arms and the k-NN arm, in this study's fold_metrics schema."""
    frames = []
    for path in (cfg.UPSTREAM_2D_METRICS, cfg.UPSTREAM_KNN_METRICS):
        if not path.exists():
            raise SystemExit(f"{path} not found -- it should travel with the study")
        frames.append(pd.read_csv(path))
    raw = pd.concat(frames, ignore_index=True)

    raw = raw[raw["task"] == "regressor"]
    raw = raw[raw["dimension"] == "2d"]

    keep = {model: method for method, model in cfg.PUBLISHED_MODELS.items()}
    unknown = set(raw["model"]) - set(keep)
    if unknown:
        raise SystemExit(f"unexpected models in the published metrics: {sorted(unknown)}")

    out = pd.DataFrame(
        {
            "target": raw["dataset"].astype(int),
            "split": raw["split"],
            "fold": raw["fold"].astype(int),
            "method": raw["model"].map(keep),
            **{metric: raw[column].astype(float) for metric, column in METRIC_FROM.items()},
        }
    )

    bad_target = set(out["target"]) - set(cfg.TARGETS)
    bad_split = set(out["split"]) - set(cfg.SPLITS)
    bad_fold = set(out["fold"]) - set(cfg.FOLDS)
    if bad_target or bad_split or bad_fold:
        raise SystemExit(
            f"published metrics name unknown targets {sorted(bad_target)}, "
            f"schemes {sorted(bad_split)} or folds {sorted(bad_fold)}"
        )

    # `n` is the number of test molecules the metric was computed on. Not
    # released, and not recoverable from a metric, so it is left empty rather
    # than filled in from this study's fold sizes -- which would assert an
    # agreement that has not been checked.
    out["n"] = pd.NA
    out["source"] = "published"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    cfg.ensure_dirs()
    published = load_published()

    expected = len(cfg.TARGETS) * len(cfg.SPLITS) * len(cfg.FOLDS)
    counts = published.groupby("method").size().reindex(cfg.PUBLISHED_METHODS)
    print(f"published per-fold metrics, {expected} folds expected per arm:")
    print(counts.to_string())
    if (counts != expected).any():
        raise SystemExit("a published arm does not cover every fold")

    if published[list(cfg.METRICS)].isna().any().any():
        missing = published[list(cfg.METRICS)].isna().sum()
        raise SystemExit(f"missing published metric values:\n{missing.to_string()}")

    path = cfg.RESULTS_DIR / "published_metrics.csv"
    published.to_csv(path, index=False)
    print(f"\nwrote {path.name} ({len(published)} rows)")

    print("\nmean test MAE by scheme, as published:")
    print(
        published.pivot_table(index="split", columns="method", values="mae")
        .reindex(index=cfg.SPLITS, columns=cfg.PUBLISHED_METHODS)
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
