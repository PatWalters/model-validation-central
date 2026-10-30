#!/usr/bin/env python
"""Step 8: reduce every prediction to per-fold and per-series statistics.

Every arm is scored on the same molecules: the non-anchor members of the
feasible series in data/<ds>/series.csv. Predictions for any other test
molecule are read and ignored. The `_offset` twins are formed here, from each
unanchored arm's own prediction for the anchor:

    y_offset(m) = y_pred(m) + (y_true(anchor) - y_pred(anchor))   for m in the anchor's series

which is the paper's "direct residual-correction" baseline. An offset moves a
whole series by one constant, so it cannot change the ranking within a series:
an arm and its twin have the same within-series Pearson and Spearman by
construction, and differ only in pooled R², pooled Spearman and MAE.

Two kinds of statistic, because the paper and the protocol score differently:

  results/<ds>/fold_metrics.csv    per (endpoint, method, repeat, fold), pooled
                                   over every scored molecule: R², Spearman ρ,
                                   MAE -- the protocol's three -- plus the mean
                                   over series of the within-series Pearson r
                                   and Spearman ρ, the paper's Table 1 metric
  results/<ds>/series_metrics.csv  per (endpoint, method, repeat, fold, series):
                                   n, within-series Pearson, Spearman, MAE

The constant `anchor_value` arm -- every member predicted at its anchor's
measured value -- goes into the pooled metrics only. It has no within-series
ranking to score.

    python 08_collect_metrics.py
    ADME_DATASET=biogen python 08_collect_metrics.py
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg

ANCHOR_VALUE = "anchor_value"
FITTED = [cfg.CC_CHEMPROP_METHOD, cfg.CAUSAL_METHOD, cfg.MONROE_ANCHOR_METHOD]
READ = [cfg.LGBM_METHOD, cfg.CHEMELEON_METHOD, cfg.MONROE_METHOD, cfg.ATOMJEPA_METHOD] + FITTED


def read_arm(method: str, endpoint: str) -> pd.DataFrame:
    """Every fold of one arm for one endpoint; multitask arms file by assay family."""
    unit = endpoint
    if method in cfg.BASELINE_IS_SINGLE_TASK and not cfg.BASELINE_IS_SINGLE_TASK[method]:
        unit = cfg.GROUP_OF[endpoint]
    parts = []
    for r in cfg.REPEATS:
        for f in cfg.FOLDS:
            p = cfg.pred_csv(method, unit, r, f)
            if p.exists():
                d = pd.read_csv(p)
                parts.append(d[d["endpoint"] == endpoint])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cfg.PRED_COLUMNS)


def safe_corr(fn, a, b) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(fn(a, b)[0])


def main() -> None:
    cfg.ensure_dirs()
    series = pd.read_csv(cfg.SERIES_CSV)
    series = series[series["feasible"]]
    endpoints = [e for e in cfg.TARGET_COLS if e in set(series["endpoint"])]
    fold_rows, series_rows, kept = [], [], []

    for endpoint in endpoints:
        s = series[series["endpoint"] == endpoint]
        anchor_of = s[s["role"] == "anchor"].set_index("series")[cfg.ID_COL]
        scored = s[s["role"] == "scored"][[cfg.ID_COL, "series"]]
        frames = {}
        for method in READ:
            d = read_arm(method, endpoint)
            if d.empty:
                continue
            frames[method] = d
            if method in cfg.UNANCHORED:
                a = d[d[cfg.ID_COL].isin(set(anchor_of))].copy()
                a["series"] = a[cfg.ID_COL].map({v: k for k, v in anchor_of.items()})
                a["shift"] = a["y_true"] - a["y_pred"]
                m = d.merge(scored, on=cfg.ID_COL).merge(
                    a[["repeat", "fold", "series", "shift"]], on=["repeat", "fold", "series"])
                m["y_pred"] = m["y_pred"] + m["shift"]
                m["method"] = method + cfg.OFFSET
                frames[method + cfg.OFFSET] = m.drop(columns=["series", "shift"])
        # The constant arm, from the measured anchor values alone.
        if cfg.CC_CHEMPROP_METHOD in frames:
            base = frames[cfg.CC_CHEMPROP_METHOD].merge(scored, on=cfg.ID_COL)
            y_anchor = frames[cfg.CC_CHEMPROP_METHOD].drop_duplicates(cfg.ID_COL).set_index(cfg.ID_COL)["y_true"]
            base["y_pred"] = base["series"].map(anchor_of).map(y_anchor)
            base["method"] = ANCHOR_VALUE
            frames[ANCHOR_VALUE] = base.drop(columns=["series"])

        for method, d in frames.items():
            d = d.merge(scored, on=cfg.ID_COL)
            kept.append(d)
            for (r, f), g in d.groupby(["repeat", "fold"]):
                if len(g) != len(scored):
                    raise SystemExit(f"{method} {endpoint} r{r} f{f}: {len(g)} scored rows, "
                                     f"expected {len(scored)}")
                per = []
                for sid, h in g.groupby("series"):
                    row = {"endpoint": endpoint, "method": method, "repeat": r, "fold": f,
                           "series": sid, "n": len(h),
                           "pearson": safe_corr(pearsonr, h["y_true"], h["y_pred"]),
                           "spearman": safe_corr(spearmanr, h["y_true"], h["y_pred"]),
                           "mae": mean_absolute_error(h["y_true"], h["y_pred"])}
                    per.append(row)
                series_rows += per
                per = pd.DataFrame(per)
                fold_rows.append({
                    "endpoint": endpoint, "method": method, "repeat": r, "fold": f, "n": len(g),
                    "r2": r2_score(g["y_true"], g["y_pred"]),
                    "spearman": safe_corr(spearmanr, g["y_true"], g["y_pred"]),
                    "mae": mean_absolute_error(g["y_true"], g["y_pred"]),
                    "pearson_series": per["pearson"].mean(),
                    "spearman_series": per["spearman"].mean(),
                })

    fm = pd.DataFrame(fold_rows)
    fm.to_csv(cfg.FOLD_METRICS_CSV, index=False)
    pd.DataFrame(series_rows).to_csv(cfg.SERIES_METRICS_CSV, index=False)
    pd.concat(kept, ignore_index=True).to_parquet(cfg.PREDICTIONS_PARQUET, index=False)

    diags = [json.loads(p.read_text()) for p in
             (cfg.PROJECT_DIR / "work" / "causal_parts" / cfg.DATASET).glob("*.json")]
    if diags:
        pd.DataFrame(diags).sort_values(["endpoint", "repeat", "fold", "series"]).to_csv(
            cfg.DIAGNOSTICS_CSV, index=False)

    counts = fm.pivot_table(index="endpoint", columns="method", values="r2", aggfunc="count")
    print(counts.reindex(endpoints).fillna(0).astype(int).to_string())
    print(f"\n{len(fm)} fold rows, {len(series_rows)} series rows; "
          f"{len(diags)} Causal-Chemprop series-fold fits")


if __name__ == "__main__":
    main()
