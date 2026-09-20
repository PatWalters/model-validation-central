"""The statistics this study needs on top of the shared `model_comparison`.

`model_comparison.py` is byte-identical to the copy in the other studies here,
and it assumes what those studies have: one endpoint at a time, one fixed test
set, and 25 replicate folds over it. A Tukey HSD over those 25 numbers is the
right test, and that module does it.

This study has a different shape. Each of six splitting schemes gives, per
method, ten targets x five folds = fifty numbers, and the between-target spread
is much larger than anything separating the methods. On the ExpansionRx data an
endpoint's MAE is a single distribution; here, pooling ten targets puts a target
effect of roughly 0.2 pIC50 into the residual of a comparison whose interesting
differences are a few hundredths. Tukey HSD on those fifty raw numbers is not
wrong, it is just answering a question nobody asked: it asks whether a method's
performance *across targets* is separable, and the answer is almost always no.

So three things live here.

`block_center` removes the target effect the way a randomized-block design does,
by shifting each target onto the grand mean, so the shared `make_tukey_plot` can
be pointed at the adjusted values and draw the same bars it draws elsewhere.
That is the figure. It is very slightly anti-conservative: statsmodels does not
know nine degrees of freedom went into the block means, and with fifty
observations per method that costs about 4% of the residual df.

`paired_groups` and `paired_table` are the primary test, and they are exact. A
fold is a (target, scheme, fold) triple, and every method fitted in this study
saw the identical training molecules in it, so the fifty differences between any
two methods are genuinely paired and the target effect cancels in each one
rather than being estimated away. Holm's correction is applied within each
(scheme, metric), across every pair of methods being compared, so the p values
survive the same multiplicity discipline Tukey imposes on the figure.

The convention in the reports here is that a difference which does not survive a
correction is a tie, not a smaller win. That is kept: `paired_groups` labels the
method with the best mean `best`, anything Holm cannot separate from it
`equivalent`, and the rest `worse` -- the same three words, and the same chips,
`tukey_groups` produces in the other studies.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

from model_comparison import best_method

# The subject a paired comparison pairs on: one target, one scheme, one fold.
SUBJECT_COL = "fold_id"


def block_center(
    df: pd.DataFrame, y_col: str, block_col: str = "target", method_col: str = "method"
) -> pd.DataFrame:
    """Shift every block onto the grand mean, leaving the method means alone.

    The standard randomized-block adjustment: y' = y - mean(block) + mean(all).
    With a balanced design -- every method present in every block with the same
    number of folds, which `require_balanced` checks -- this changes no method's
    mean at all, and removes the between-block variance from the residual. So
    the bars in a Tukey plot of `y'` sit exactly where a plot of `y` would put
    them, and only the intervals around them change.
    """
    require_balanced(df, y_col, block_col, method_col)
    out = df.copy()
    shifted = out.groupby(block_col)[y_col].transform("mean")
    out[y_col] = out[y_col] - shifted + out[y_col].mean()
    return out


def require_balanced(
    df: pd.DataFrame,
    y_col: str,
    block_col: str = "target",
    method_col: str = "method",
) -> None:
    """Refuse to adjust or test unless every method appears in every block equally.

    `block_center` only leaves the method means untouched when the design is
    balanced, and a paired test needs the same subjects on both sides. An
    unbalanced table here means a sweep is incomplete, which is worth stopping
    for rather than quietly reporting a method on the folds that happened to
    finish.
    """
    counts = df.pivot_table(index=block_col, columns=method_col, values=y_col,
                            aggfunc="size")
    if counts.isna().any().any() or counts.stack().nunique() > 1:
        raise ValueError(
            "the statistics here need every method in every block with the same "
            f"number of folds; got counts\n{counts.fillna(0).astype(int).to_string()}"
        )


def _wide(df: pd.DataFrame, y_col: str, method_col: str, subject_col: str) -> pd.DataFrame:
    wide = df.pivot(index=subject_col, columns=method_col, values=y_col)
    if wide.isna().any().any():
        missing = wide.columns[wide.isna().any()].tolist()
        raise ValueError(f"methods {missing} do not cover every {subject_col}")
    return wide


def paired_table(
    df: pd.DataFrame,
    y_col: str,
    methods: list[str] | None = None,
    method_col: str = "method",
    subject_col: str = SUBJECT_COL,
    test: str = "t",
) -> pd.DataFrame:
    """Every pair of methods, paired over the shared folds, Holm-corrected.

    One row per unordered pair, carrying the mean difference (right - left), the
    fraction of folds `right` wins, the raw p value and the Holm-adjusted one.
    The correction spans every pair in the call, which is what makes it
    comparable with the simultaneous intervals in the Tukey figure -- ask for
    one pair and you get an uncorrected test, which is rarely what is wanted.

    `test` is "t" for a paired t-test, the test the other studies here use, or
    "wilcoxon" for the signed-rank version. MAE differences over fifty folds are
    not far from symmetric, but the rank test is there to be checked against and
    08_report.py writes both.
    """
    wide = _wide(df, y_col, method_col, subject_col)
    names = list(methods or wide.columns)
    missing = [m for m in names if m not in wide.columns]
    if missing:
        raise ValueError(f"no values for {missing}")

    rows = []
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            diff = (wide[right] - wide[left]).to_numpy()
            if np.allclose(diff, 0):
                # Two arms with identical predictions. A test has nothing to say
                # and both tests would raise, so it is recorded as p = 1.
                p = 1.0
            elif test == "wilcoxon":
                p = float(wilcoxon(diff).pvalue)
            else:
                p = float(ttest_rel(wide[right], wide[left]).pvalue)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n_folds": len(diff),
                    "mean_diff": float(diff.mean()),
                    "sd_diff": float(diff.std(ddof=1)),
                    "right_wins": float((diff > 0).mean()),
                    "p_raw": p,
                }
            )

    table = pd.DataFrame(rows)
    table["p_holm"] = multipletests(table["p_raw"], method="holm")[1]
    return table


def paired_groups(
    df: pd.DataFrame,
    y_col: str,
    higher_is_better: bool = True,
    methods: list[str] | None = None,
    method_col: str = "method",
    subject_col: str = SUBJECT_COL,
    alpha: float = 0.05,
    test: str = "t",
) -> pd.Series:
    """Label each method 'best', 'equivalent' or 'worse' by a paired test.

    The same three labels `model_comparison.tukey_groups` produces and the same
    tie rule -- a difference that does not survive the correction is a tie --
    reached by the test this study's design actually licenses. The correction is
    Holm over every pair, so a method is called `worse` only if its gap to the
    best-mean method survives simultaneously with every other gap on the page.
    """
    wide = _wide(df, y_col, method_col, subject_col)
    names = list(methods or wide.columns)
    best = best_method(df[df[method_col].isin(names)], y_col, higher_is_better, method_col)

    table = paired_table(df, y_col, names, method_col, subject_col, test)
    against = table[(table["left"] == best) | (table["right"] == best)]

    labels = {best: "best"}
    for _, row in against.iterrows():
        other = row["right"] if row["left"] == best else row["left"]
        labels[other] = "worse" if row["p_holm"] < alpha else "equivalent"
    return pd.Series({name: labels[name] for name in names}, name="paired_group")


def collapse_to_targets(
    df: pd.DataFrame,
    y_col: str,
    method_col: str = "method",
    block_col: str = "target",
    subject_col: str = SUBJECT_COL,
) -> pd.DataFrame:
    """Average each method's folds within a target, so the target is the subject.

    A paired t-test over fifty folds assumes the fifty differences are
    independent. Within a target they are not quite, and on one scheme they are
    badly not: the time split gives each fold its own publication-year cutoff, so
    a target's five test sets overlap heavily -- 28,057 test assignments over
    9,486 distinct molecules across the whole collection. Differences computed on
    overlapping test molecules are correlated, which inflates the effective
    sample size and makes the fifty-fold test anti-conservative there.

    Collapsing to one number per target first removes the problem entirely,
    whatever the correlation inside a target is, because the ten targets really
    are independent. It costs power -- ten paired observations instead of fifty
    -- so it is the sensitivity check rather than the headline, and 08_report.py
    runs it alongside the fold-level test on every pair. Where the two disagree,
    the fold-level result is the one to distrust.
    """
    out = (
        df.groupby([method_col, block_col])[y_col]
        .mean()
        .reset_index()
        .rename(columns={block_col: subject_col})
    )
    out[subject_col] = out[subject_col].astype(str)
    return out


def gap_to_baseline(
    df: pd.DataFrame,
    y_col: str,
    baseline: str,
    higher_is_better: bool = True,
    method_col: str = "method",
    subject_col: str = SUBJECT_COL,
) -> pd.DataFrame:
    """How far each method beats `baseline`, fold by fold.

    The quantity the paper's argument is actually about. "The models approach
    random guessing" is a claim about the distance between a method and the naive
    predictor, and that distance is what survives being compared across splitting
    schemes: a raw MAE of 0.9 means something different on a scheme whose test
    sets are structurally distant, because the naive baseline moved too.

    Signed so that a positive gap is always a method doing better than the
    baseline, whichever direction the metric runs, and paired on the fold, so
    the target effect cancels rather than being averaged over.
    """
    wide = _wide(df, y_col, method_col, subject_col)
    if baseline not in wide.columns:
        raise ValueError(f"no values for the baseline {baseline!r}")

    sign = 1.0 if higher_is_better else -1.0
    gap = wide.drop(columns=[baseline]).sub(wide[baseline], axis=0) * sign
    return gap.reset_index().melt(id_vars=subject_col, var_name=method_col, value_name="gap")
