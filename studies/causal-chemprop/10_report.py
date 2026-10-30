#!/usr/bin/env python
"""Step 10: the protocol half's statistics and figures, for the active data set.

The unit of comparison is the distribution of a metric over folds; Tukey HSD
says which differences survive a correction for multiple comparisons, and a
paired test on the folds says which way a head-to-head goes. Where two arms
cannot be separated, both are tied. Every arm is restricted to the folds every
arm has, so a sweep still running is compared on what it shares.

Because the paper's own unit is the series, not the fold, there is a second
view: each series' within-series Spearman ρ averaged over its folds, and
Causal-Chemprop against each arm by a Wilcoxon signed-rank test over the
series. That is the comparison that answers the paper's claim on its terms.

Writes to results/<ds>/tables and results/<ds>/figures:

  summary.csv              mean, sd and Tukey group per endpoint x arm x metric
  head_to_head.csv         Causal-Chemprop against every arm, paired on folds
  series_head_to_head.csv  the same over series, within-series Spearman ρ
  tukey_<metric>.png       one Tukey panel per endpoint
  series_scatter.png       per-series ρ, Causal-Chemprop against its own
                           encoder and against Monroe with the anchors
  modes.csv                within-series ρ by which of the authors' rules
                           made the prediction, from the fit diagnostics
  cost.csv                 encoder GPU time and SCM CPU time, per endpoint

    python 10_report.py
    ADME_DATASET=biogen python 10_report.py
"""

import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

import config as cfg
from model_comparison import make_tukey_plot, tukey_groups

C = cfg.CAUSAL_METHOD
O = cfg.OFFSET
# The arms on the page. Every unanchored arm appears through its offset twin,
# the version that has been given the anchor; the plain encoder Causal-Chemprop
# sits on and plain Monroe stay as well, as the "no anchor" reference.
REPORT_METHODS = [
    "lgbm" + O, "chemeleon" + O, "atomjepa_ft" + O,
    "monroe", "monroe" + O, cfg.MONROE_ANCHOR_METHOD,
    cfg.CC_CHEMPROP_METHOD, cfg.CC_CHEMPROP_METHOD + O, C,
]
TUKEY_METRICS = ["r2", "spearman", "mae", "spearman_series"]
LABEL = {"r2": "R² (pooled)", "spearman": "Spearman ρ (pooled)", "mae": "MAE (pooled)",
         "spearman_series": "within-series Spearman ρ", "pearson_series": "within-series Pearson r"}
SHORT = {
    "lgbm" + O: "LightGBM +off", "chemeleon" + O: "CheMeleon +off",
    "atomjepa_ft" + O: "Atom-JEPA FT +off", "atomjepa_ft": "Atom-JEPA FT",
    "monroe": "Monroe/PFN3", "monroe" + O: "Monroe/PFN3 +off",
    cfg.MONROE_ANCHOR_METHOD: "Monroe/PFN3 +ctx",
    cfg.CC_CHEMPROP_METHOD: "Chemprop (CC enc.)", cfg.CC_CHEMPROP_METHOD + O: "Chemprop +off",
    C: "Causal-Chemprop", "anchor_value": "anchor value", "lgbm": "LightGBM",
    "chemeleon": "CheMeleon",
}
HIGHER = {**cfg.METRIC_HIGHER_IS_BETTER}


def load():
    fm = pd.read_csv(cfg.FOLD_METRICS_CSV)
    sm = pd.read_csv(cfg.SERIES_METRICS_CSV)
    for d in (fm, sm):
        d["fold_id"] = "r" + d["repeat"].astype(str) + "f" + d["fold"].astype(str)
    # Only the folds Causal-Chemprop has, for every arm, so the design is paired
    # and balanced while its sweep is still running.
    have = fm[fm["method"] == C].groupby("endpoint")["fold_id"].apply(set).to_dict()
    keep = lambda d: d[d.apply(lambda r: r["fold_id"] in have.get(r["endpoint"], set()), axis=1)]
    return keep(fm), keep(sm), have


def summary(fm: pd.DataFrame, endpoints) -> pd.DataFrame:
    rows = []
    for e in endpoints:
        sub = fm[(fm["endpoint"] == e) & fm["method"].isin(REPORT_METHODS)]
        for metric in TUKEY_METRICS:
            groups = tukey_groups(sub, metric, HIGHER[metric])
            for m, g in sub.groupby("method"):
                rows.append({"endpoint": e, "method": m, "metric": metric,
                             "mean": g[metric].mean(), "sd": g[metric].std(),
                             "folds": len(g), "tukey_group": groups[m]})
        for m in ("anchor_value", "monroe", "atomjepa_ft", "lgbm", "chemeleon"):
            g = fm[(fm["endpoint"] == e) & (fm["method"] == m)]
            for metric in ("r2", "spearman", "mae"):
                if m in REPORT_METHODS or g.empty:
                    continue
                rows.append({"endpoint": e, "method": m, "metric": metric, "mean": g[metric].mean(),
                             "sd": g[metric].std(), "folds": len(g), "tukey_group": "not tested"})
    return pd.DataFrame(rows)


def head_to_head(fm: pd.DataFrame, endpoints) -> pd.DataFrame:
    rows = []
    others = sorted(set(fm["method"]) - {C, "anchor_value"})
    for e in endpoints:
        sub = fm[fm["endpoint"] == e]
        for metric in ("r2", "spearman", "mae", "spearman_series", "pearson_series"):
            w = sub.pivot(index="fold_id", columns="method", values=metric)
            for o in others:
                if C not in w or o not in w:
                    continue
                pair = w[[C, o]].dropna()
                if len(pair) < 2:
                    continue
                d = pair[C] - pair[o]
                sign = 1 if HIGHER[metric] else -1
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    p = ttest_rel(pair[C], pair[o]).pvalue if d.std() > 0 else (1.0 if d.mean() == 0 else 0.0)
                rows.append({"endpoint": e, "other": o, "metric": metric, "folds": len(pair),
                             "mean_diff": d.mean(), "causal_fold_wins": int((sign * d > 0).sum()),
                             "p": p,
                             "call": ("ahead" if sign * d.mean() > 0 else "behind") if p < 0.05 else "tied"})
    return pd.DataFrame(rows)


def series_view(sm: pd.DataFrame) -> pd.DataFrame:
    """Each series' within-series ρ averaged over its folds; Wilcoxon across series."""
    per = sm.groupby(["endpoint", "series", "method"])["spearman"].mean().unstack("method")
    rows = []
    for o in sorted(set(per.columns) - {C, "anchor_value"}):
        pair = per[[C, o]].dropna()
        d = pair[C] - pair[o]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = wilcoxon(pair[C], pair[o]).pvalue if (d != 0).any() else 1.0
        rows.append({"other": o, "series": len(pair), "causal_wins": int((d > 0).sum()),
                     "causal_losses": int((d < 0).sum()), "mean_diff": d.mean(),
                     "median_diff": d.median(), "wilcoxon_p": p})
    return pd.DataFrame(rows), per


def tukey_figures(fm, endpoints):
    n = len(endpoints)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    for metric in TUKEY_METRICS:
        fig, axes = plt.subplots(rows, cols, figsize=(5.7 * cols, (0.5 + 0.42 * len(REPORT_METHODS)) * rows),
                                 squeeze=False)
        flat = axes.flatten()
        for ax in flat[n:]:
            ax.set_visible(False)
        for ax, e in zip(flat, endpoints):
            sub = fm[(fm["endpoint"] == e) & fm["method"].isin(REPORT_METHODS)].copy()
            sub["method"] = sub["method"].map(SHORT)
            make_tukey_plot(sub, metric, higher_is_better=HIGHER[metric], ax=ax, title=e,
                            xlabel=LABEL[metric])
        fig.suptitle(f"{cfg.ACTIVE.label}: {LABEL[metric]}, Tukey HSD over folds", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_size_inches()[1]))
        for suffix in ("png", "svg"):
            fig.savefig(cfg.FIGURE_DIR / f"tukey_{metric}.{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)


def series_scatter(per: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, o in zip(axes, (cfg.CC_CHEMPROP_METHOD, cfg.MONROE_ANCHOR_METHOD)):
        pair = per[[C, o]].dropna()
        ax.scatter(pair[o], pair[C], s=18, alpha=0.7, color="#1F6F5C", edgecolor="none")
        ax.plot([-1, 1], [-1, 1], color="0.5", lw=0.8, ls="--")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(f"{SHORT[o]}, within-series ρ")
        ax.set_ylabel("Causal-Chemprop, within-series ρ")
        ahead = int((pair[C] > pair[o]).sum())
        ax.set_title(f"{ahead} of {len(pair)} series above the line", fontsize=10)
    fig.suptitle(f"{cfg.ACTIVE.label}: each point one series, ρ averaged over its folds")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(cfg.FIGURE_DIR / f"series_scatter.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def modes(sm: pd.DataFrame) -> pd.DataFrame:
    """Within-series ρ split by the rule that produced the series' predictions.

    test_counterfactual has three exits: the counterfactual, the plain SCM
    prediction when it is far enough from the anchor's (switched), and all SCM
    predictions when the counterfactuals barely vary (var_fallback). A fit is
    degenerate when no node clears the ICC threshold and the authors' fallback
    picks the nodes instead.
    """
    d = pd.read_csv(cfg.DIAGNOSTICS_CSV)
    c = sm[sm["method"] == C].merge(d, on=["endpoint", "repeat", "fold", "series"])
    e = sm[sm["method"] == cfg.CC_CHEMPROP_METHOD][["endpoint", "repeat", "fold", "series", "spearman"]]
    c = c.merge(e.rename(columns={"spearman": "encoder"}), on=["endpoint", "repeat", "fold", "series"])
    c["mode"] = np.select(
        [c["var_fallback"], c["frac_switched"] == 0, c["frac_switched"] == 1],
        ["all SCM, low variance", "all counterfactual", "all SCM, switched"], "mixed")
    rows = []
    for (key, val), g in [*((("mode", m), g) for m, g in c.groupby("mode")),
                          *((("degenerate", bool(v)), g) for v, g in c.groupby("degenerate")),
                          (("all", "all"), c)]:
        rows.append({"split": key, "value": val, "fits": len(g),
                     "causal_rho": g["spearman"].mean(), "encoder_rho": g["encoder"].mean(),
                     "no_rho": int(g["spearman"].isna().sum()),
                     "median_contributors": g["n_contributors"].median()})
    return pd.DataFrame(rows)


def cost(endpoints) -> pd.DataFrame:
    """Encoder fits (GPU) and SCM fits (one CPU core each) for the folds scored."""
    d = pd.read_csv(cfg.DIAGNOSTICS_CSV)
    rows = []
    for e in endpoints:
        g = d[d["endpoint"] == e]
        enc = [json.loads(p.read_text())["seconds"]
               for f in sorted(set(zip(g["repeat"], g["fold"])))
               if (p := cfg.MODEL_DIR / f"{e}_r{f[0]}_f{f[1]}" / "encoder.json").exists()]
        rows.append({"endpoint": e, "encoders": len(enc), "encoder_gpu_h": sum(enc) / 3600,
                     "scm_fits": len(g), "scm_median_min": g["seconds"].median() / 60,
                     "scm_max_h": g["seconds"].max() / 3600, "scm_cpu_h": g["seconds"].sum() / 3600,
                     "median_analogs": g["n_analogs"].median()})
    return pd.DataFrame(rows)


def main() -> None:
    cfg.ensure_dirs()
    fm, sm, have = load()
    endpoints = [e for e in cfg.TARGET_COLS if len(have.get(e, ())) >= 2]
    if not endpoints:
        raise SystemExit("Causal-Chemprop has fewer than two folds on every endpoint")
    print(f"{cfg.ACTIVE.label}: folds per endpoint {({e: len(have[e]) for e in endpoints})}")
    s = summary(fm, endpoints)
    s.to_csv(cfg.TABLE_DIR / "summary.csv", index=False)
    h = head_to_head(fm, endpoints)
    h.to_csv(cfg.TABLE_DIR / "head_to_head.csv", index=False)
    sv, per = series_view(sm[sm["endpoint"].isin(endpoints)])
    sv.to_csv(cfg.TABLE_DIR / "series_head_to_head.csv", index=False)
    per.reset_index().to_csv(cfg.TABLE_DIR / "series_spearman.csv", index=False)
    tukey_figures(fm, endpoints)
    series_scatter(per)
    md = modes(sm[sm["endpoint"].isin(endpoints)])
    md.to_csv(cfg.TABLE_DIR / "modes.csv", index=False)
    cost(endpoints).to_csv(cfg.TABLE_DIR / "cost.csv", index=False)

    print("\nTukey: endpoints where each arm is in the top group")
    top = s[s["tukey_group"].isin(["best", "equivalent"])]
    print(top.pivot_table(index="method", columns="metric", values="endpoint", aggfunc="count")
          .reindex(REPORT_METHODS).fillna(0).astype(int).to_string())
    print("\nCausal-Chemprop paired on folds (ahead / tied / behind, over endpoints)")
    print(h.pivot_table(index="other", columns=["metric", "call"], values="endpoint", aggfunc="count")
          .fillna(0).astype(int).to_string())
    print("\nover series, within-series Spearman ρ\n" + sv.round(3).to_string(index=False))
    print("\nby prediction rule\n" + md.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
