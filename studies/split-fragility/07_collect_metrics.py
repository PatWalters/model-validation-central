#!/usr/bin/env python
"""Step 7: gather every prediction and reduce it to per-fold statistics.

Reads the tidy prediction files written by steps 1 to 5, concatenates them into
results/predictions_all.parquet, computes MAE, R^2 and Spearman rho for each
(target, scheme, fold, method) on that fold's own test molecules, folds in the
published arms from step 6, and writes results/fold_metrics.csv --
11 methods x 10 targets x 6 schemes x 5 folds = 3,300 rows when complete.

Two checks run here rather than living in a comment.

The first is the reproduction: the k-NN arm re-implemented in step 1 and the
k-NN arm published in step 6 should agree fold by fold. They do on the five
schemes whose splits upstream recorded completely, to within 4e-9 MAE, which is
what licenses putting the published arms on the same axis as the fitted ones. It
is printed every run, and a regression in the folds or the metric would show up
here first.

The second is balance. Every statistic in step 8 is paired or blocked on the
fold, so it needs every method present on every fold. A partial sweep is
reported rather than failed, so progress can be watched, but step 8 will refuse
to run on one.

    python 07_collect_metrics.py
    python 07_collect_metrics.py --no-parquet   # metrics only, skip the big file
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg

# The reproduction the study rests on, and how close it has to be. The tolerance
# is loose enough for float accumulation over a few hundred molecules and far
# tighter than any difference that would matter.
REPRO_PAIR = (cfg.KNN_METHOD, "pub_knn")
# Per metric, because the three do not agree to the same precision and the reason
# is understood. MAE and R^2 are sums over the fold's molecules and reproduce to
# float accumulation. Spearman does not, and cannot: a distance-weighted k-NN
# with k as low as 1 predicts the same value for many molecules at once, so the
# fold is full of rank ties, and how a tie is broken moves rho in the third
# decimal. It is bounded here rather than demanded exact, and the bound is still
# two orders of magnitude below anything the report reads off a rho.
REPRO_TOL = {"mae": 1e-6, "r2": 1e-6, "spearman": 1e-2}
# The one scheme it is not expected to hold on, and why: upstream did not record
# that scheme's validation split, so the two k-NN arms pick k on different
# molecules. See config.read_split.
REPRO_EXEMPT = "diverse_25"


def load_predictions() -> pd.DataFrame:
    paths = sorted(cfg.PRED_DIR.glob("*/*.csv"))
    if not paths:
        raise SystemExit(f"no prediction files under {cfg.PRED_DIR} -- run steps 1 to 5 first")
    df = pd.concat((pd.read_csv(p, usecols=cfg.PRED_COLUMNS) for p in paths), ignore_index=True)
    print(f"read {len(paths)} prediction files, {len(df):,} predictions")
    return df


def check_same_molecules(df: pd.DataFrame) -> None:
    """Every fitted arm must have scored the same molecules on the same fold.

    This is the premise the whole comparison rests on, and it is the one thing
    that would invalidate every number on the page while leaving all of them
    looking perfectly reasonable. ChemProp is the arm to worry about: its
    predictions come back from a subprocess in whatever order the dataloader
    produced them and are joined back on name, so a silent misalignment there
    would show up as a plausible-looking loss of accuracy rather than as an
    error. Comparing the measured values fold by fold catches it, because a
    reordered or truncated join changes which y_true sits in which row.
    """
    fitted = df[df["method"].isin(cfg.RUN_METHODS + [cfg.MEDIAN_METHOD])]
    problems = []
    for (target, split, fold), g in fitted.groupby(["target", "split", "fold"], sort=True):
        reference = None
        for method, h in g.groupby("method", sort=True):
            h = h.sort_values(cfg.OUT_ID_COL, kind="stable")
            signature = (
                tuple(h[cfg.OUT_ID_COL].astype(str)),
                tuple(np.round(h["y_true"].to_numpy(), 9)),
            )
            if reference is None:
                reference = (method, signature)
            elif signature != reference[1]:
                problems.append(
                    f"tid_{target} {split} f{fold}: {method} scored different molecules "
                    f"than {reference[0]} ({len(h)} rows against "
                    f"{len(reference[1][0])})"
                )
    if problems:
        raise SystemExit(
            "the arms are not scoring the same test molecules, so nothing downstream "
            "is a comparison:\n  " + "\n  ".join(problems[:10])
            + (f"\n  ... and {len(problems) - 10} more" if len(problems) > 10 else "")
        )
    n = fitted.groupby(["target", "split", "fold"]).ngroups
    print(f"every fitted arm scored identical molecules on all {n} folds")


def fold_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """One row per fitted (target, scheme, fold, method).

    Spearman rho is undefined against a constant prediction, which is exactly
    what the naive baseline produces, so it comes out NaN for that arm. Recorded
    as NaN rather than filled with zero: the baseline has no ranking to score,
    and a zero would put it on the rank-based figures as though it did.
    """
    records = []
    for (target, split, fold, method), g in df.groupby(
        ["target", "split", "fold", "method"], sort=True
    ):
        constant = g["y_pred"].nunique() == 1
        records.append(
            {
                "target": target,
                "split": split,
                "fold": fold,
                "method": method,
                "n": len(g),
                "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
                "r2": r2_score(g["y_true"], g["y_pred"]),
                "spearman": np.nan if constant else spearmanr(g["y_true"], g["y_pred"]).statistic,
            }
        )
    out = pd.DataFrame(records)
    out["source"] = "fitted"
    return out


def check_reproduction(metrics: pd.DataFrame) -> pd.DataFrame:
    """Re-run k-NN against published k-NN, fold by fold."""
    columns = ["metric", "split", "folds", "tol", "within_tol", "max_abs_diff"]
    mine, theirs = REPRO_PAIR
    if not {mine, theirs} <= set(metrics["method"]):
        print(f"\nreproduction check skipped: need both {mine} and {theirs}")
        # An empty frame *with its columns*, so a downstream reader gets an empty
        # table rather than an unparseable file.
        return pd.DataFrame(columns=columns)

    wide = metrics[metrics["method"].isin(REPRO_PAIR)].pivot_table(
        index=["target", "split", "fold"], columns="method", values=cfg.METRICS
    )
    rows = []
    for metric in cfg.METRICS:
        diff = (wide[(metric, mine)] - wide[(metric, theirs)]).abs()
        for split, g in diff.groupby("split"):
            rows.append(
                {
                    "metric": metric,
                    "split": split,
                    "folds": len(g),
                    "tol": REPRO_TOL[metric],
                    "within_tol": int((g < REPRO_TOL[metric]).sum()),
                    "max_abs_diff": float(g.max()),
                }
            )
    table = pd.DataFrame(rows)

    tols = ", ".join(f"{m} {t:g}" for m, t in REPRO_TOL.items())
    print(f"\nreproduction of the published k-NN by {mine} (tolerances: {tols}):")
    shown = table.pivot(index="split", columns="metric", values="within_tol").reindex(cfg.SPLITS)
    worst = table.pivot(index="split", columns="metric", values="max_abs_diff").reindex(cfg.SPLITS)
    for metric in cfg.METRICS:
        shown[f"{metric} max|d|"] = worst[metric].map(lambda v: f"{v:.1e}")
    print(shown.to_string())

    failed = table[(table["split"] != REPRO_EXEMPT) & (table["within_tol"] < table["folds"])]
    if len(failed):
        print("\nWARNING: the reproduction no longer holds on a scheme it should:")
        print(failed.to_string(index=False))
    else:
        agreed = int(table[(table["split"] != REPRO_EXEMPT) & (table["metric"] == "mae")]["within_tol"].sum())
        total = int(table[(table["split"] != REPRO_EXEMPT) & (table["metric"] == "mae")]["folds"].sum())
        print(f"\n{agreed}/{total} folds reproduce on the five recorded schemes; "
              f"{REPRO_EXEMPT} is exempt by construction")
    return table


def report_coverage(metrics: pd.DataFrame) -> None:
    expected = len(cfg.SPLITS) * len(cfg.FOLDS)
    counts = (
        metrics.pivot_table(index="target", columns="method", values="mae", aggfunc="count")
        .reindex(index=cfg.TARGETS, columns=cfg.ALL_METHODS)
        .fillna(0)
        .astype(int)
    )
    print(f"\nfolds per target and method (of {expected}):")
    print(counts.to_string())

    incomplete = int((counts != expected).sum().sum())
    if incomplete:
        print(f"\n{incomplete} target/method combinations are still incomplete -- "
              "step 8 needs a complete sweep")
    else:
        print(f"\ncomplete: {len(metrics)} rows "
              f"({len(cfg.TARGETS)} targets x {len(cfg.ALL_METHODS)} methods x {expected} folds)")

    # MAE and R^2 must exist for every arm. Spearman is allowed to be missing for
    # the constant baseline and nothing else.
    for metric in ("mae", "r2"):
        if metrics[metric].isna().any():
            raise SystemExit(f"NaN {metric} values -- check the prediction files")
    stray = metrics[metrics["spearman"].isna() & (metrics["method"] != cfg.MEDIAN_METHOD)]
    if len(stray):
        raise SystemExit(
            f"{len(stray)} NaN Spearman values outside the constant baseline:\n"
            f"{stray[['target', 'split', 'fold', 'method']].head().to_string(index=False)}"
        )


def summarise(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric over each scheme's fifty folds."""
    summary = (
        metrics.groupby(["split", "method"])[cfg.METRICS]
        .agg(["mean", "std"])
        .reindex(pd.MultiIndex.from_product([cfg.SPLITS, cfg.ALL_METHODS],
                                            names=["split", "method"]))
        .dropna(how="all")
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-parquet", action="store_true",
                        help="skip writing predictions_all.parquet")
    args = parser.parse_args()

    cfg.ensure_dirs()
    preds = load_predictions()

    unknown = set(preds["method"]) - set(cfg.ALL_METHODS)
    if unknown:
        raise SystemExit(f"unexpected methods in the prediction files: {sorted(unknown)}")

    if not args.no_parquet:
        preds.to_parquet(cfg.PREDICTIONS_PARQUET, index=False)
        print(f"wrote {cfg.PREDICTIONS_PARQUET.name} "
              f"({cfg.PREDICTIONS_PARQUET.stat().st_size / 1e6:.0f} MB)")

    check_same_molecules(preds)
    metrics = fold_metrics(preds)

    published_path = cfg.RESULTS_DIR / "published_metrics.csv"
    if published_path.exists():
        published = pd.read_csv(published_path)
        metrics = pd.concat([metrics, published], ignore_index=True)
    else:
        print(f"\n{published_path.name} not found -- run 06_import_published.py to add "
              "the paper's own arms")

    metrics = metrics.sort_values(["split", "target", "fold", "method"], ignore_index=True)
    metrics.to_csv(cfg.FOLD_METRICS_CSV, index=False)
    print(f"wrote {cfg.FOLD_METRICS_CSV.name} ({len(metrics)} rows)")

    check_reproduction(metrics).to_csv(cfg.SHARED_TABLE_DIR / "reproduction.csv", index=False)
    report_coverage(metrics)

    summary = summarise(metrics)
    summary.to_csv(cfg.SHARED_TABLE_DIR / "summary_raw.csv", index=False)

    print("\nmean over each scheme's fifty folds:")
    for metric in cfg.METRICS:
        wide = summary.pivot(index="split", columns="method", values=f"{metric}_mean")
        print(f"\n{metric}")
        print(wide.reindex(index=cfg.SPLITS, columns=cfg.ALL_METHODS).round(3).to_string())


if __name__ == "__main__":
    main()
