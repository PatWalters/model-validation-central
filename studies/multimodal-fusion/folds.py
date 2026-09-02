"""Which molecules a fold fits on, validates on, and is scored on.

One definition, used by every step, so that "the same splits" is a property of
the code rather than a claim in the README. It reproduces the masks
`02_run_lightgbm.py` of ../expansion-ml-comparison builds, which is what the four
reference methods were fit on:

    fit   measured, not in the held-out test set, and not in this fold
    val   measured, not in the test set, and in this fold
    test  measured and in the held-out test set

The test mask does not depend on the fold. All 25 models of an endpoint are
scored on exactly the same molecules, which is what makes the folds a valid
pairing for the statistics and an ensemble for the uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as cfg


@dataclass(frozen=True)
class FoldMasks:
    fit: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def touched(self) -> np.ndarray:
        """Every molecule this fold sees, in master.csv row order."""
        return self.fit | self.val | self.test

    def __repr__(self) -> str:
        return (f"FoldMasks(fit={self.fit.sum()}, val={self.val.sum()}, "
                f"test={self.test.sum()})")


def fold_masks(df: pd.DataFrame, folds: pd.DataFrame, endpoint: str,
               repeat: int, fold: int) -> FoldMasks:
    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    fold_of = df[cfg.ID_COL].map(held_out).to_numpy()   # NaN for test molecules

    measured = df[endpoint].notna().to_numpy()
    is_test = (df[cfg.SET_COL] == "test").to_numpy()

    return FoldMasks(
        fit=measured & ~is_test & (fold_of != fold),
        val=measured & ~is_test & (fold_of == fold),
        test=measured & is_test,
    )


def training_mask(df: pd.DataFrame, endpoint: str) -> np.ndarray:
    """Every training molecule with this endpoint measured, ignoring the folds.

    What the hyperparameter search is run over: it has to be chosen once and
    reused across all 25 folds, so it cannot be tied to any one of them, and it
    must not touch the test set.
    """
    measured = df[endpoint].notna().to_numpy()
    return measured & (df[cfg.SET_COL] != "test").to_numpy()


def tidy_predictions(df: pd.DataFrame, method: str, endpoint: str, repeat: int,
                     fold: int, test_mask: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """The prediction schema every method in both repositories writes."""
    test_df = df.loc[test_mask]
    return pd.DataFrame(
        {
            "method": method,
            "endpoint": endpoint,
            "repeat": repeat,
            "fold": fold,
            cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(),
            cfg.SMILES_COL: test_df[cfg.SMILES_COL].to_numpy(),
            "y_true": test_df[endpoint].to_numpy(),
            "y_pred": np.asarray(y_pred, dtype=float),
        },
        columns=cfg.PRED_COLUMNS,
    )
