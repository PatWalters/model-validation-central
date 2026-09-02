"""Epistemic uncertainty and calibration, as the paper defines them.

Their ensemble is three independently seeded models; ours is the five folds of one
repeat, which predict the same fixed test molecules from five overlapping
four-fifths of the same training set. Both measure the same thing -- how much a
prediction moves when the fit is perturbed -- and ours costs nothing extra and
yields five ensembles per configuration where theirs yields one.

The four quantities are `src/uncertainty_analysis.py` of
github.com/jwasswa2023/Multimodal_Fusion, and Section S6 of the paper's
Supporting Information, unchanged:

    sigma                the population standard deviation across ensemble
                         members, no Bessel correction
    err_unc_corr         Pearson correlation of absolute error with sigma
    ece                  equal-frequency binned mean |error| against mean sigma
    miscalibration_area  the L1 area between the two marginal quantile curves

Two properties of this ECE are worth keeping in mind while reading it. It
compares an error magnitude directly against a standard deviation, so a
perfectly calibrated Gaussian scores about 0.2 sigma rather than zero, and it is
in the units of the endpoint. It is a within-panel comparison, not a number to
carry to another paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


def ensemble(preds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean prediction, sigma, and absolute error for one repeat's ensemble.

    `preds` holds one repeat of one method on one endpoint: every fold's
    prediction for every test molecule. Molecules are aligned on the identifier
    rather than on row order, because nothing guarantees the folds wrote their
    rows in the same order.
    """
    wide = preds.pivot_table(
        index=cfg.ID_COL, columns="fold", values="y_pred", aggfunc="mean"
    )
    truth = preds.groupby(cfg.ID_COL)["y_true"].first().reindex(wide.index)

    members = wide.to_numpy()
    mean = np.nanmean(members, axis=1)
    sigma = np.nanstd(members, axis=1)          # ddof = 0, as the released code
    return mean, sigma, np.abs(truth.to_numpy() - mean)


def error_uncertainty_corr(abs_err: np.ndarray, sigma: np.ndarray) -> float:
    if np.std(abs_err) < 1e-12 or np.std(sigma) < 1e-12:
        return float("nan")
    return float(np.corrcoef(abs_err, sigma)[0, 1])


def regression_ece(abs_err: np.ndarray, sigma: np.ndarray,
                   n_bins: int = cfg.ECE_BINS) -> float:
    """Equal-frequency bins over sigma; mean |error| against mean sigma in each."""
    edges = np.unique(np.quantile(sigma, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return float("nan")

    which = np.digitize(sigma, edges[1:-1], right=True)
    total = 0.0
    for b in range(len(edges) - 1):
        mask = which == b
        if not mask.any():
            continue
        total += mask.mean() * abs(abs_err[mask].mean() - sigma[mask].mean())
    return float(total)


def miscalibration_area(abs_err: np.ndarray, sigma: np.ndarray,
                        n_points: int = 100) -> float:
    """L1 area between the marginal quantile curves of |error| and sigma."""
    if abs_err.size == 0 or sigma.size == 0:
        return float("nan")
    q = np.linspace(0.0, 1.0, n_points)
    return float(np.trapezoid(np.abs(np.quantile(abs_err, q) - np.quantile(sigma, q)), q))


def repeat_metrics(preds: pd.DataFrame) -> dict[str, float]:
    """All four quantities for one method, endpoint and repeat."""
    _, sigma, abs_err = ensemble(preds)
    return {
        "sigma": float(np.mean(sigma)),
        "err_unc_corr": error_uncertainty_corr(abs_err, sigma),
        "ece": regression_ece(abs_err, sigma),
        "miscalibration_area": miscalibration_area(abs_err, sigma),
    }


def reliability_curve(preds: pd.DataFrame, n_bins: int = cfg.ECE_BINS) -> pd.DataFrame:
    """Mean sigma against mean |error| per bin: the ECE, drawn rather than summed."""
    _, sigma, abs_err = ensemble(preds)
    edges = np.unique(np.quantile(sigma, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return pd.DataFrame(columns=["bin", "n", "mean_sigma", "mean_abs_error"])

    which = np.digitize(sigma, edges[1:-1], right=True)
    rows = []
    for b in range(len(edges) - 1):
        mask = which == b
        if not mask.any():
            continue
        rows.append({
            "bin": b,
            "n": int(mask.sum()),
            "mean_sigma": float(sigma[mask].mean()),
            "mean_abs_error": float(abs_err[mask].mean()),
        })
    return pd.DataFrame(rows)
