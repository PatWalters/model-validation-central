#!/usr/bin/env python
"""Checks on the statistics, because the statistics are the point of the study.

The whole argument of this directory is that the design here is blocked and
paired and that treating it as fifty independent numbers per method throws the
power away. That claim is worth demonstrating on data whose answer is known
rather than only asserting in a README, so this builds a synthetic table with a
planted target effect and a planted method difference and checks that each
function recovers what was planted.

    python test_split_stats.py
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from split_stats import (
    SUBJECT_COL,
    block_center,
    gap_to_baseline,
    paired_groups,
    paired_table,
    require_balanced,
)

TARGETS = list(range(10))
FOLDS = list(range(5))


def synthetic(effect: float, target_spread: float, noise: float, seed: int = 0) -> pd.DataFrame:
    """Two methods, ten targets, five folds, with everything planted deliberately.

    `b` is better than `a` by `effect` on every single fold. Targets differ from
    each other by `target_spread`, which is the nuisance the design has to remove,
    and each fold carries independent noise of scale `noise` shared between the
    two methods -- shared, because in the real study both methods are fit on the
    same molecules of that fold, which is exactly what makes the pairing valid.
    """
    rng = np.random.default_rng(seed)
    target_of = {t: rng.normal(0, target_spread) for t in TARGETS}
    rows = []
    for t in TARGETS:
        for f in FOLDS:
            shared = rng.normal(0, noise)
            for method, shift in (("a", 0.0), ("b", effect)):
                rows.append(
                    {
                        "method": method,
                        "target": t,
                        SUBJECT_COL: f"tid{t}_f{f}",
                        "score": 1.0 + target_of[t] + shared + shift
                        + rng.normal(0, noise / 10),
                    }
                )
    return pd.DataFrame(rows)


def tukey_p(df: pd.DataFrame) -> float:
    res = pairwise_tukeyhsd(endog=df["score"], groups=df["method"], alpha=0.05)
    return float(res.pvalues[0])


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        raise SystemExit(f"failed: {label}")


def main() -> None:
    # A method difference of 0.05, a target effect four times that, and fold
    # noise in between: roughly the proportions of the real MAE table.
    df = synthetic(effect=0.05, target_spread=0.20, noise=0.08)

    print("block_center")
    centred = block_center(df, "score", block_col="target")
    means_before = df.groupby("method")["score"].mean()
    means_after = centred.groupby("method")["score"].mean()
    check("leaves every method's mean where it was",
          np.allclose(means_before, means_after),
          f"max shift {float((means_before - means_after).abs().max()):.2e}")
    check("removes the between-target variance",
          centred.groupby("target")["score"].mean().std() < 1e-9)
    check("narrows the Tukey interval it is meant to narrow",
          tukey_p(centred) < tukey_p(df),
          f"p {tukey_p(df):.3f} raw -> {tukey_p(centred):.2e} centred")

    print("\npaired_table")
    table = paired_table(df, "score", ["a", "b"])
    row = table.iloc[0]
    check("recovers the planted effect",
          abs(row["mean_diff"] - 0.05) < 0.01, f"{row['mean_diff']:.4f} vs 0.0500")
    check("pairs on all fifty folds", int(row["n_folds"]) == 50)
    check("sees b winning every fold", row["right_wins"] == 1.0)
    check("separates them where the raw pooled test cannot",
          row["p_raw"] < 1e-6 < tukey_p(df),
          f"paired p {row['p_raw']:.1e}, pooled Tukey p {tukey_p(df):.3f}")

    print("\nHolm correction")
    wide = synthetic(effect=0.0, target_spread=0.20, noise=0.08, seed=7)
    many = pd.concat(
        [wide.assign(method=wide["method"] + str(i)) for i in range(4)], ignore_index=True
    )
    corrected = paired_table(many, "score", sorted(many["method"].unique()))
    check("adjusts upwards over many pairs",
          (corrected["p_holm"] >= corrected["p_raw"] - 1e-12).all())
    check("leaves a single pair uncorrected",
          abs(paired_table(df, "score", ["a", "b"]).iloc[0]["p_holm"]
              - paired_table(df, "score", ["a", "b"]).iloc[0]["p_raw"]) < 1e-12)

    print("\npaired_groups")
    labels = paired_groups(df, "score", higher_is_better=True, methods=["a", "b"])
    check("crowns the better method", labels["b"] == "best")
    check("calls the other one worse", labels["a"] == "worse")

    flat = synthetic(effect=0.0, target_spread=0.20, noise=0.08, seed=3)
    tied = paired_groups(flat, "score", higher_is_better=True, methods=["a", "b"])
    check("calls an unseparable pair tied, not a narrow win",
          sorted(tied.tolist()) == ["best", "equivalent"], f"{tied.to_dict()}")

    lower = paired_groups(df, "score", higher_is_better=False, methods=["a", "b"])
    check("respects the metric's direction", lower["a"] == "best")

    print("\ngap_to_baseline")
    gaps = gap_to_baseline(df, "score", baseline="a", higher_is_better=True)
    check("one gap per fold, baseline dropped",
          len(gaps) == 50 and set(gaps["method"]) == {"b"})
    check("signs a better method positive when higher is better",
          abs(gaps["gap"].mean() - 0.05) < 0.01, f"{gaps['gap'].mean():.4f}")
    flipped = gap_to_baseline(df, "score", baseline="a", higher_is_better=False)
    check("and negative when lower is better",
          abs(flipped["gap"].mean() + 0.05) < 0.01, f"{flipped['gap'].mean():.4f}")

    print("\nrequire_balanced")
    try:
        require_balanced(df.iloc[:-1], "score")
    except ValueError:
        check("refuses a table with a fold missing from one method", True)
    else:
        check("refuses a table with a fold missing from one method", False)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
