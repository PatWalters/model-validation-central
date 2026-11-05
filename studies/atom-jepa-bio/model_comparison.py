"""Plots for comparing model performance across cross-validation folds.

`make_tukey_plot` and the fold-statistics convention come from an earlier
project, generalised here to take any metric column and to know which direction
is better for that metric.

The approach follows "Even More Thoughts on ML Method Comparisons"
(https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html):
show the distribution of a metric over folds, and let a test that corrects for
multiple comparisons say which methods are actually distinguishable.
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def best_method(df: pd.DataFrame, y_col: str, higher_is_better: bool, method_col: str = "method") -> str:
    """The method with the best mean value of `y_col`."""
    means = df.groupby(method_col)[y_col].mean()
    return means.idxmax() if higher_is_better else means.idxmin()


def assert_balanced(df: pd.DataFrame, method_col: str = "method") -> None:
    """Refuse to plot unless every method contributed the same number of folds.

    `plot_simultaneous` colours a bar by whether its interval overlaps the
    reference method's, while `tukey_groups` (and so the chips in the tables)
    reads the pairwise `reject` flag. Those are two different decision rules.
    They coincide exactly when the design is balanced: statsmodels builds the
    half-widths from Hochberg's eq 3.32, which reduces to half the pairwise
    critical difference when every group has the same n, so `hw_i + hw_j` is the
    Tukey critical distance to machine precision.

    Drop folds from one method and the levelling stops being exact. The two
    thresholds separate, and a mean difference landing between them gets a red
    bar next to a "tied" chip with nothing to flag the contradiction. Losing a
    fold to a crashed run is the realistic way that happens, so it is worth
    catching here rather than in a reader's eye.
    """
    sizes = df.groupby(method_col).size()
    if sizes.nunique() > 1:
        counts = ", ".join(f"{name} {int(n)}" for name, n in sizes.items())
        raise ValueError(
            "Tukey plot needs the same number of folds per method, got "
            f"{counts}. The bar colours and the table chips would be free to "
            "disagree. Fill in the missing folds, or drop the method."
        )


def make_tukey_plot(
    df: pd.DataFrame,
    y_col: str,
    higher_is_better: bool = True,
    ax=None,
    method_col: str = "method",
    xlim=None,
    title: str = "",
    xlabel: str = None,
):
    """Tukey HSD comparison of every method against the best one.

    Blue is the best method, grey a method it cannot be distinguished from, red a
    method that is significantly worse. The confidence intervals are corrected for
    the number of comparisons, so overlapping bars mean what they appear to mean.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    assert_balanced(df, method_col)
    tukey = pairwise_tukeyhsd(endog=df[y_col], groups=df[method_col], alpha=0.05)
    best = best_method(df, y_col, higher_is_better, method_col)
    tukey.plot_simultaneous(comparison_name=best, ax=ax)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel(xlabel or y_col)
    ax.set_ylabel("")
    ax.set_title(title)
    return tukey


def tukey_groups(
    df: pd.DataFrame, y_col: str, higher_is_better: bool = True, method_col: str = "method"
) -> pd.Series:
    """Label each method 'best', 'equivalent' or 'worse' relative to the best one.

    The annotation the summary table carries instead of a bolded maximum: it says
    which differences survive a multiple-comparison correction.
    """
    tukey = pairwise_tukeyhsd(endog=df[y_col], groups=df[method_col], alpha=0.05)
    results = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    best = best_method(df, y_col, higher_is_better, method_col)

    labels = {}
    for method in df[method_col].unique():
        if method == best:
            labels[method] = "best"
            continue
        row = results[
            ((results["group1"] == best) & (results["group2"] == method))
            | ((results["group1"] == method) & (results["group2"] == best))
        ]
        labels[method] = "worse" if bool(row["reject"].iloc[0]) else "equivalent"
    return pd.Series(labels, name="tukey_group")


def paired_test(df: pd.DataFrame, y_col: str, left: str, right: str, method_col: str = "method",
                subject_col: str = "fold_id") -> tuple[float, float]:
    """Paired t-test of `right` against `left` over the shared folds.

    Returns (mean difference right - left, p value). The folds are the pairing:
    both methods saw the same training molecules, so the comparison is paired.
    """
    wide = df.pivot(index=subject_col, columns=method_col, values=y_col)
    diff = wide[right] - wide[left]
    return float(diff.mean()), float(ttest_rel(wide[right], wide[left]).pvalue)
