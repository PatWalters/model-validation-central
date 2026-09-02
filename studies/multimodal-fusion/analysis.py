"""The statistics the paper runs, and the aggregation they run over.

Two levels of comparison, deliberately kept apart.

*Across endpoints* is the paper's own unit: one number per endpoint per
configuration, and a two-sided Wilcoxon signed-rank test over those numbers with
a Holm correction across the family of comparisons. That is how they conclude
that multimodal beats the best unimodal numerically but not significantly, and it
is what "the same test on our data" means.

*Within an endpoint* is what the fold structure allows and their single 80/20
split does not: 25 paired replicates, so Tukey HSD can say which configurations
are distinguishable on one endpoint. That is the machinery the reference methods
were already scored with, and it is what puts the fusion grid on the same axis as
LightGBM and the ChemProp variants.

Both are reported. Where they disagree it is because they are answering different
questions, and the report says which.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config as cfg


def endpoint_means(metrics: pd.DataFrame, value: str) -> pd.DataFrame:
    """One number per endpoint per method: the mean over that method's folds.

    The paper's unit of analysis, since its comparisons are across the fourteen
    properties rather than within one.
    """
    return metrics.pivot_table(
        index="endpoint", columns="method", values=value, aggfunc="mean"
    )


def holm(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p values, in the order given."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def paired_wilcoxon(wide: pd.DataFrame, pairs: list[tuple[str, str]],
                    higher_is_better: bool) -> pd.DataFrame:
    """Two-sided Wilcoxon signed-rank over endpoints, Holm corrected as a family.

    `wide` is `endpoint_means`; each pair is (left, right) and the reported
    difference is right minus left. Pairs whose columns are missing are dropped
    rather than silently scored on a subset.
    """
    rows = []
    for left, right in pairs:
        if left not in wide.columns or right not in wide.columns:
            continue
        both = wide[[left, right]].dropna()
        if len(both) < 3:
            continue
        diff = both[right] - both[left]
        # A test on all-zero differences is undefined; it also cannot happen
        # unless the two configurations are the same model under two names.
        p = 1.0 if np.allclose(diff, 0) else wilcoxon(both[right], both[left]).pvalue
        better = (diff > 0) if higher_is_better else (diff < 0)
        rows.append({
            "left": left, "right": right, "n_endpoints": len(both),
            "mean_diff": float(diff.mean()), "median_diff": float(diff.median()),
            "right_wins": int(better.sum()), "p_value": float(p),
        })

    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm(table["p_value"].tolist())
        table["significant"] = table["p_holm"] < 0.05
    return table


def best_per_endpoint(wide: pd.DataFrame, methods: list[str],
                      higher_is_better: bool) -> pd.Series:
    """For each endpoint, the best value among `methods`."""
    available = [m for m in methods if m in wide.columns]
    sub = wide[available]
    return sub.max(axis=1) if higher_is_better else sub.min(axis=1)


def best_name_per_endpoint(wide: pd.DataFrame, methods: list[str],
                           higher_is_better: bool) -> pd.Series:
    available = [m for m in methods if m in wide.columns]
    sub = wide[available]
    return sub.idxmax(axis=1) if higher_is_better else sub.idxmin(axis=1)


# --- the grid, sliced the ways the paper slices it -----------------------
UNIMODAL = cfg.UNIMODAL_METHODS
MULTIMODAL = cfg.FUSION_METHODS


def by_fusion(fusion: str) -> list[str]:
    return [m for m in MULTIMODAL if cfg.GRID_SPEC[m]["fusion"] == fusion]


def by_learner(learner: str, multimodal_only: bool = True) -> list[str]:
    pool = MULTIMODAL if multimodal_only else cfg.GRID_METHODS
    return [m for m in pool if cfg.GRID_SPEC[m]["learner"] == learner]


def by_combo(combo: str) -> list[str]:
    return [m for m in MULTIMODAL if cfg.GRID_SPEC[m]["combo"] == combo]


def grid_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    """Fold metrics with the three design axes attached as columns.

    Everything downstream groups by `combo`, `fusion` and `learner` rather than
    by parsing method names, so a name change never silently regroups a figure.
    """
    df = metrics[metrics["method"].isin(cfg.GRID_METHODS)].copy()
    spec = df["method"].map(cfg.GRID_SPEC)
    df["combo"] = spec.map(lambda s: s["combo"])
    df["fusion"] = spec.map(lambda s: s["fusion"])
    df["learner"] = spec.map(lambda s: s["learner"])
    df["n_modalities"] = spec.map(lambda s: len(s["modalities"]))
    df["label"] = df["method"].map(cfg.METHOD_LABELS)
    return df


def fusion_pairs() -> list[tuple[str, str]]:
    """Early against late, holding the modality set and the learner fixed."""
    return [
        (cfg.fusion_method(combo, "early", learner),
         cfg.fusion_method(combo, "late", learner))
        for combo in cfg.COMBOS for learner in cfg.LEARNERS
    ]


def ladder_pairs() -> list[tuple[str, str]]:
    """Each step up the modality ladder, holding fusion and learner fixed."""
    steps = [("GR", "GRM"), ("GR", "GRS"), ("GRM", "GRMS"), ("GRS", "GRMS")]
    return [
        (cfg.fusion_method(a, f, learner), cfg.fusion_method(b, f, learner))
        for a, b in steps for f in cfg.FUSIONS for learner in cfg.LEARNERS
    ]


def learner_pairs() -> list[tuple[str, str]]:
    """Each pair of learners, holding the modality set and the strategy fixed."""
    combos = [("lgbm", "rf"), ("lgbm", "attfp"), ("rf", "attfp")]
    return [
        (cfg.fusion_method(combo, f, a), cfg.fusion_method(combo, f, b))
        for a, b in combos for combo in cfg.COMBOS for f in cfg.FUSIONS
    ]
