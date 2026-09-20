#!/usr/bin/env python
"""Step 9: build the shareable HTML report.

Reads the tables and figures from step 8 and writes one self-contained page --
figures embedded as data URIs, no external requests -- to results/report.html.

Every number in the prose is read from the tables rather than typed in, so the
page cannot drift away from the results it describes. Where a sentence depends
on *which* method came out on top, it is phrased from the tally rather than
hard-coded, and the few places that cannot be written that way say so.

    python 09_build_page.py
    SPLIT_COMPARISON=published python 09_build_page.py   # the eleven-arm version
"""

import pandas as pd

import config as cfg
import page_kit
from narrative import NARRATIVE

# One page per comparison, for the same reason the tables are namespaced.
PAGE = cfg.REPORT_DIR / "report.html"

# The browser tab, the bookmark and the Pages site all read this, and every other
# report in the repository opens with one. It comes before the stylesheet because
# the first 8 KB is all some readers scan.
TITLE = "What survives a hard split"

COLOR = {
    cfg.KNN_METHOD: "#4C72B0",
    cfg.LGBM_METHOD: "#DD8452",
    cfg.CHEMELEON_METHOD: "#55A868",
    cfg.MONROE35_METHOD: "#61483A",
    cfg.MEDIAN_METHOD: "#7A8783",
    "pub_knn": "#A3B5CE",
    "pub_svm": "#8172B3",
    "pub_rf": "#CCB974",
    "pub_xgboost": "#DA8BC3",
    "pub_mlp": "#C44E52",
    "pub_gnn": "#64B5CD",
}

SHORT = {
    cfg.KNN_METHOD: "k-NN",
    cfg.LGBM_METHOD: "LightGBM",
    cfg.CHEMELEON_METHOD: "CheMeleon",
    cfg.MONROE35_METHOD: "Monroe/PFN3.5",
    cfg.MEDIAN_METHOD: "naive",
    "pub_knn": "k-NN*",
    "pub_svm": "SVM*",
    "pub_rf": "RF*",
    "pub_xgboost": "XGB*",
    "pub_mlp": "MLP*",
    "pub_gnn": "GNN*",
}

REFERENCES = [
    ("paper", "Lee, K.; Moldagulov, G.; Grzybowski, B. A. ",
     "The Fragility of Bioactivity Prediction: Rigorous Dataset Splits Expose the "
     "Illusion of ML Accuracy.",
     "<span class='venue'>Chem. Eur. J. 2026, e71208.</span>",
     "https://doi.org/10.1002/chem.71208", "doi:10.1002/chem.71208"),
    ("janela", "Janela, T.; Bajorath, J. ",
     "Simple nearest-neighbour analysis meets the accuracy of compound potency "
     "predictions using complex machine learning models.",
     "<span class='venue'>Nat. Mach. Intell. 2022, 4, 1246-1255.</span>",
     "https://doi.org/10.1038/s42256-022-00581-6", "doi:10.1038/s42256-022-00581-6"),
    ("chemeleon", "Burns, J. W.; Zalte, A. S.; Abreu, C. R. A.; et al. ",
     "Deep Learning Foundation Models for Low-Data Regimes from Classical Molecular "
     "Descriptors.",
     "<span class='venue'>J. Chem. Inf. Model. 2026.</span>",
     "https://doi.org/10.1021/acs.jcim.6c01546", "doi:10.1021/acs.jcim.6c01546"),
    ("monroe", "Banaszewski, B.; Fitzgibbon, A. W. ",
     "Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference.",
     "<span class='venue'>Preprint, 2026.</span>",
     "https://arxiv.org/abs/2608.18982", "arXiv:2608.18982"),
    ("chemprop", "Heid, E.; Greenman, K. P.; Chung, Y.; et al. ",
     "Chemprop: A Machine Learning Package for Chemical Property Prediction.",
     "<span class='venue'>J. Chem. Inf. Model. 2024, 64, 9-17.</span>",
     "https://doi.org/10.1021/acs.jcim.3c01250", "doi:10.1021/acs.jcim.3c01250"),
    ("tabpfn", "Hollmann, N.; Müller, S.; Purucker, L.; et al. ",
     "Accurate predictions on small data with a tabular foundation model.",
     "<span class='venue'>Nature 2025, 637, 319-326.</span>",
     "https://doi.org/10.1038/s41586-024-08328-6", "doi:10.1038/s41586-024-08328-6"),
    ("thoughts", "Walters, P. ",
     "Even More Thoughts on ML Method Comparisons.",
     "<span class='venue'>Practical Cheminformatics, 2025.</span>",
     "https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html",
     "practicalcheminformatics.blogspot.com"),
    ("ash", "Ash, J. R.; Wognum, C.; Rodríguez-Pérez, R.; et al. ",
     "Practically significant method comparison protocols for machine learning in small "
     "molecule drug discovery.",
     "<span class='venue'>Preprint, 2025.</span>",
     "https://doi.org/10.26434/chemrxiv-2025-l0h2b", "doi:10.26434/chemrxiv-2025-l0h2b"),
]
NUMBERS = page_kit.reference_numbers(REFERENCES)


def ref(*keys: str) -> str:
    return page_kit.marker(NUMBERS, *keys)


# --- the tables this page reads ------------------------------------------
def load() -> dict:
    tables = {}
    for name in ("summary", "head_to_head", "tally", "gap_to_baseline",
                 "fold_distance", "distance_decay", "fold_sizes", "size_confound"):
        path = cfg.TABLE_DIR / f"{name}.csv"
        if not path.exists():
            step = "10_distance_analysis.py" if name.startswith(("fold_d", "distance")) \
                else "08_report.py"
            raise SystemExit(f"{path} not found -- run {step} first")
        tables[name] = pd.read_csv(path)
    # Written once for the data set rather than per comparison.
    tables["reproduction"] = pd.read_csv(cfg.SHARED_TABLE_DIR / "reproduction.csv")
    tables["metrics"] = pd.read_csv(cfg.FOLD_METRICS_CSV)
    require_every_arm(tables["metrics"])
    return tables


def require_every_arm(metrics: pd.DataFrame) -> None:
    """Refuse to build the page unless every arm of the comparison is complete.

    08_report.py runs happily on a partial sweep, which is what makes it useful
    while the long arm is still training. The page is different: its prose reads
    numbers out of the tables and phrases itself from which arm came out on top,
    so a missing arm would not produce a gap in a table -- it would produce a
    fluent paragraph about the wrong winner. Better to stop.
    """
    expected = len(cfg.TARGETS) * len(cfg.SPLITS) * len(cfg.FOLDS)
    counts = metrics[metrics["method"].isin(cfg.METHODS)].groupby("method").size()
    short = {m: int(counts.get(m, 0)) for m in cfg.METHODS if counts.get(m, 0) != expected}
    if short:
        listed = ", ".join(f"{m} {n}/{expected}" for m, n in short.items())
        raise SystemExit(
            f"the sweep is not complete, so the page would describe it wrongly: {listed}.\n"
            "Finish those arms, or build the page for a comparison that excludes them."
        )


def figure(name: str, caption: str) -> str:
    return page_kit.figure_block(
        page_kit.embed_figure(cfg.FIGURE_DIR / f"{name}.png"), caption, name
    )


def methods_in(summary: pd.DataFrame, metric: str) -> list[str]:
    present = set(summary.loc[summary["metric"] == metric, "method"])
    return [m for m in cfg.METHODS if m in present]


def mean_of(summary: pd.DataFrame, split: str, metric: str, method: str) -> float:
    row = summary[(summary["split"] == split) & (summary["metric"] == metric)
                  & (summary["method"] == method)]
    return float(row["mean"].iloc[0]) if len(row) else float("nan")


def pair(h2h: pd.DataFrame, split: str, metric: str, a: str, b: str,
         test: str = "t") -> pd.Series | None:
    """One pair's row, whichever order it was stored in, sign normalised to b - a."""
    sub = h2h[(h2h["split"] == split) & (h2h["metric"] == metric) & (h2h["test"] == test)]
    hit = sub[(sub["left"] == a) & (sub["right"] == b)]
    if len(hit):
        return hit.iloc[0]
    hit = sub[(sub["left"] == b) & (sub["right"] == a)]
    if not len(hit):
        return None
    row = hit.iloc[0].copy()
    row["mean_diff"] = -row["mean_diff"]
    row["right_wins"] = 1.0 - row["right_wins"]
    return row


# --- page pieces ----------------------------------------------------------
def reproduced(repro: pd.DataFrame) -> str:
    """The k-NN reproduction as an <agreed>/<checked> fact, read from the table."""
    if repro.empty:
        return "n/a"
    # MAE only, so the tile counts folds rather than fold x metric comparisons.
    # The prose beside it gives the fuller figure.
    kept = repro[(repro["split"] != "diverse_25") & (repro["metric"] == "mae")]
    if kept.empty:
        return "n/a"
    return f"{int(kept['within_tol'].sum())}/{int(kept['folds'].sum())}"


def facts(tables: dict) -> str:
    metrics = tables["metrics"]
    fitted = metrics[metrics["source"] == "fitted"]
    items = [
        (f"{len(cfg.METHODS)}", "arms compared"),
        ("10", "ChEMBL targets"),
        ("6", "splitting schemes"),
        (f"{len(fitted):,}", "fold models fitted"),
        (reproduced(tables["reproduction"]), "published k-NN folds reproduced"),
    ]
    cells = "".join(f'<div class="fact"><b>{n}</b><span>{label}</span></div>'
                    for n, label in items)
    return f'<div class="facts">{cells}</div>'


def summary_table(summary: pd.DataFrame, metric: str) -> str:
    """Mean ± sd per scheme and method, with the paired-test grouping as a chip."""
    sub = summary[summary["metric"] == metric]
    methods = methods_in(summary, metric)

    # How many arms share the top of each scheme, so a lone winner and a tie can
    # be labelled differently. Same rule the tally uses.
    tops = sub[sub["paired_group"].isin(("best", "equivalent"))].groupby("split").size()

    rows = []
    for split in cfg.SPLITS:
        g = sub[sub["split"] == split].set_index("method")
        if g.empty:
            continue
        cells = [f'<th scope="row">{cfg.SPLIT_LABELS[split]}</th>']
        for m in methods:
            if m not in g.index:
                cells.append("<td></td>")
                continue
            r = g.loc[m]
            on_top = r["paired_group"] in ("best", "equivalent")
            note = ("tied" if int(tops.get(split, 1)) > 1 else "best") if on_top else ""
            cls = "best" if on_top else "worse"
            chip = f'<span class="chip {cls}">{note}</span>' if note else ""
            cells.append(f'<td class="{cls}"><span class="num">{r["mean"]:.3f}</span>'
                         f'<span class="sd">±{r["sd"]:.3f}</span>{chip}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    head = "".join(
        f'<th><span class="swatch" style="background:{COLOR[m]}"></span>{SHORT[m]}</th>'
        for m in methods
    )
    return ('<div class="bleed tablewrap"><table>'
            f'<thead><tr><th scope="col">Scheme</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def tally_block(tally: pd.DataFrame) -> str:
    n = len(cfg.SPLITS) * len(cfg.METRICS)
    rows = []
    for _, r in tally.iterrows():
        m = r["method"]
        rows.append(
            f'<div class="tally">'
            f'<span class="swatch big" style="background:{COLOR[m]}"></span>'
            f'<span class="tallyname">{cfg.METHOD_LABELS[m]}</span>'
            f'<span class="tallynums">best alone <b>{int(r["best_alone"])}</b> · '
            f'tied <b>{int(r["tied"])}</b> · worse <b>{int(r["worse"])}</b></span>'
            f"</div>"
        )
    return (f'<div class="panel"><div class="tallies">{"".join(rows)}</div>'
            f'<p class="footnote">Over {n} scheme × metric combinations. Where two arms '
            "cannot be separated by a Holm-corrected paired test, both are counted as "
            "tied and neither is called the winner. The naive baseline is in the count "
            "for MAE and R², and out of it for Spearman ρ, which a constant prediction "
            "has none of.</p></div>")


def gap_table(gaps: pd.DataFrame) -> str:
    """The gap to the naive predictor, per scheme, in pIC50 units."""
    methods = [m for m in cfg.METHODS if m in set(gaps["method"])]
    rows = []
    for split in cfg.SPLITS:
        g = gaps[gaps["split"] == split].set_index("method")
        if g.empty:
            continue
        cells = [f'<th scope="row">{cfg.SPLIT_LABELS[split]}</th>']
        for m in methods:
            if m not in g.index:
                cells.append("<td></td>")
                continue
            r = g.loc[m]
            cls = "best" if r["mean"] > 0 else "worse"
            cells.append(f'<td class="{cls}"><span class="num">{r["mean"]:+.3f}</span>'
                         f'<span class="sd">{r["beats_baseline"]:.0%}</span></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    head = "".join(
        f'<th><span class="swatch" style="background:{COLOR[m]}"></span>{SHORT[m]}</th>'
        for m in methods
    )
    return ('<div class="bleed tablewrap"><table>'
            f'<thead><tr><th scope="col">Scheme</th>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="footnote">Mean MAE taken off the naive predictor, in pIC₅₀ units, '
            "paired on the fold; the small figure is the share of the fifty folds on which "
            "the arm beats it. Positive is better than predicting the training median.</p>")


def h2h_table(h2h: pd.DataFrame, a: str, b: str, metric: str = "mae") -> str:
    """One pair across the six schemes."""
    rows = []
    for split in cfg.SPLITS:
        r = pair(h2h, split, metric, a, b)
        if r is None:
            continue
        # MAE: lower is better, so a negative difference means `b` won.
        better = r["mean_diff"] < 0 if not cfg.METRIC_HIGHER_IS_BETTER[metric] else r["mean_diff"] > 0
        sig = r["p_holm"] < 0.05
        verdict = (f"{SHORT[b]} wins" if better else f"{SHORT[a]} wins") if sig else "tied"
        cls = ("win" if better else "loss") if sig else "null"
        rows.append(
            f'<tr><th scope="row">{cfg.SPLIT_LABELS[split]}</th>'
            f'<td class="num">{r["mean_diff"]:+.3f}</td>'
            f'<td class="num">{r["right_wins"]:.0%}</td>'
            f'<td class="num">{r["p_holm"]:.1e}</td>'
            f'<td><span class="chip {cls}">{verdict}</span></td></tr>'
        )
    return ('<div class="tablewrap"><table>'
            f'<thead><tr><th scope="col">Scheme</th>'
            f'<th scope="col">Δ {metric.upper()}</th><th scope="col">Folds to {SHORT[b]}</th>'
            f'<th scope="col">p<sub>Holm</sub></th><th scope="col">Verdict</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def sizes_table(sizes: pd.DataFrame) -> str:
    """How much data each scheme trains on, and how evenly it divides its folds."""
    rows = []
    for _, r in sizes.iterrows():
        matched = abs(r["train_pct"] - 72.0) < 1.0
        cls = "best" if matched else "worse"
        rows.append(
            f'<tr><th scope="row">{cfg.SPLIT_LABELS[r["split"]]}</th>'
            f'<td class="num {cls}">{r["train_pct"]:.1f}<span class="sd">±{r["train_pct_sd"]:.1f}</span></td>'
            f'<td class="num">{r["test_pct"]:.1f}<span class="sd">±{r["test_pct_sd"]:.1f}</span></td>'
            f'<td class="num">{r["train"]:,.0f}</td>'
            f'<td class="num">{r["within_train_range"]:.1f}</td>'
            f'<td class="num">{r["within_test_cv"]:.1f}</td>'
            f'<td class="num">{r["mean_max_sim"]:.3f}</td></tr>'
        )
    return ('<div class="bleed tablewrap"><table>'
            '<thead><tr><th scope="col">Scheme</th><th scope="col">Train %</th>'
            '<th scope="col">Test %</th><th scope="col">Train molecules</th>'
            '<th scope="col">Fold range</th><th scope="col">Test CV %</th>'
            '<th scope="col">NN Tanimoto</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="footnote">Percentages are of each target, because the ten data sets '
            "run from 1,000 to 2,273 molecules. “Fold range” is the largest minus the "
            "smallest training set among one target's five folds, averaged over targets, "
            "and “Test CV %” the same spread for the test sets — both are what a scheme "
            "controls, unlike the differences between targets. Green marks the four "
            "schemes that hold training-set size fixed.</p>")


def stringency_table(distances: pd.DataFrame) -> str:
    """What each scheme does to the distance between train and test, measured."""
    g = distances.groupby("split")[["mean_max_sim", "frac_novel", "n_train", "n_test"]].mean()
    rows = []
    for split in cfg.SPLITS:
        if split not in g.index:
            continue
        r = g.loc[split]
        rows.append(
            f'<tr><th scope="row">{cfg.SPLIT_LABELS[split]}</th>'
            f'<td class="num">{r["mean_max_sim"]:.3f}</td>'
            f'<td class="num">{r["frac_novel"]:.0%}</td>'
            f'<td class="num">{r["n_train"]:,.0f}</td>'
            f'<td class="num">{r["n_test"]:,.0f}</td></tr>'
        )
    return ('<div class="tablewrap"><table>'
            '<thead><tr><th scope="col">Scheme</th>'
            '<th scope="col">Mean NN Tanimoto</th><th scope="col">Below 0.35</th>'
            '<th scope="col">Train</th><th scope="col">Test</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="footnote">Averaged over the fifty folds of each scheme. '
            "“Below 0.35” is the share of test molecules whose nearest training neighbour "
            "falls under the Butina threshold the paper clusters at — molecules with no "
            "close analogue to learn from. Train and test are molecules per fold.</p>")


def decay_table(decay: pd.DataFrame) -> str:
    """Each method's decay with distance, and what it is worth when far out."""
    rows = []
    for method in cfg.METHODS:
        r = decay[decay["method"] == method]
        if r.empty:
            continue
        r = r.iloc[0]
        far = r["gap_at_sim_0.30"]
        cls = "best" if far > 0 else "worse"
        rows.append(
            f'<tr><th scope="row">'
            f'<span class="swatch" style="background:{COLOR[method]}"></span>'
            f'{cfg.METHOD_LABELS[method]}</th>'
            f'<td class="num">{r["slope"]:.3f}</td>'
            f'<td class="num">{r["spearman"]:.3f}</td>'
            f'<td class="num {cls}">{r["gap_at_sim_0.60"]:+.3f}</td>'
            f'<td class="num {cls}">{far:+.3f}</td></tr>'
        )
    return ('<div class="tablewrap"><table>'
            '<thead><tr><th scope="col">Method</th><th scope="col">Slope</th>'
            '<th scope="col">Spearman ρ</th><th scope="col">Gap at 0.60</th>'
            '<th scope="col">Gap at 0.30</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="footnote">A linear fit of the gap to the naive baseline against '
            "each fold's mean nearest-neighbour similarity, over all 300 folds and all six "
            "schemes pooled. The slope is how fast a method loses its advantage as the test "
            "molecules move away; the last two columns are what the fit says it is worth at "
            "a similarity typical of the scaffold split and of the UMAP split. A negative "
            "number is a method that would have done better predicting the training "
            "median.</p>")


def collapse_table(summary: pd.DataFrame, gaps: pd.DataFrame) -> str:
    """The paper's headline trend, arm by arm: raw MAE and the gap behind it.

    Two columns that move in opposite directions are the whole argument. Raw MAE
    rises steeply from the random split to the time split for every arm, which is
    the paper's finding. The gap to the naive predictor is what says whether that
    rise means the models stopped working or the problem got harder.
    """
    rows = []
    for split in cfg.SPLITS:
        naive = mean_of(summary, split, "mae", cfg.MEDIAN_METHOD)
        g = gaps[gaps["split"] == split].set_index("method")
        arms = [m for m in cfg.METHODS if m in g.index]
        if not arms:
            continue
        top = max(arms, key=lambda m: g.loc[m, "mean"])
        cells = (
            f'<th scope="row">{cfg.SPLIT_LABELS[split]}</th>'
            f'<td class="num">{naive:.3f}</td>'
            f'<td class="num">{mean_of(summary, split, "mae", top):.3f}</td>'
            f'<td class="num">{g.loc[top, "mean"]:+.3f}</td>'
            f'<td>{cfg.METHOD_LABELS[top]}</td>'
        )
        rows.append(f"<tr>{cells}</tr>")
    return ('<div class="tablewrap"><table>'
            '<thead><tr><th scope="col">Scheme</th><th scope="col">Naive MAE</th>'
            '<th scope="col">Best MAE</th><th scope="col">Gap</th>'
            '<th scope="col">Which arm</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def build(tables: dict) -> str:
    summary, h2h = tables["summary"], tables["head_to_head"]
    tally, gaps = tables["tally"], tables["gap_to_baseline"]
    distances, decay = tables["fold_distance"], tables["distance_decay"]
    sizes, confound = tables["fold_sizes"], tables["size_confound"]

    parts = [
        f"<title>{TITLE}</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">Model Validation Central</p>',
        "<h1>What survives a hard split</h1>",
        '<p class="lede">Four modelling approaches on the ten ChEMBL potency data sets '
        "and six splitting schemes of Lee, Moldagulov and Grzybowski"
        f"{ref('paper')} — on that paper's own folds, against that paper's own naive "
        "baseline and its own six arms.</p>",
        facts(tables),
        NARRATIVE["intro"](summary, h2h, gaps, tally, distances, decay),
        '<hr class="rule">',
        "<h2>The collapse, and what is left under it</h2>",
        NARRATIVE["collapse"](summary, h2h, gaps, tally, distances, decay),
        collapse_table(summary, gaps),
        figure("mae_by_split",
               "MAE over each scheme's fifty folds — ten targets by five folds — with the "
               "naive baseline drawn across each panel. The schemes run left to right and "
               "top to bottom in the paper's order of stringency."),
        figure("gap_to_baseline",
               "The same numbers read as a distance from the naive predictor, paired on the "
               "fold. Zero is predicting the training median for every test molecule."),
        gap_table(gaps),
        '<hr class="rule">',
        "<h2>What “stringent” actually means</h2>",
        NARRATIVE["stringency"](summary, h2h, gaps, tally, distances, decay),
        stringency_table(distances),
        NARRATIVE["confound"](summary, h2h, gaps, tally, distances, decay),
        sizes_table(sizes),
        figure("similarity_by_split",
               "Each scheme measured rather than named. Left: how close a test molecule "
               "sits to the nearest molecule its model was trained on. Right: how much of "
               "a test set has no analogue above the Butina threshold at all."),
        figure("decay_with_distance",
               "Every fold of every scheme as one point, with the six-valued scheme label "
               "replaced by the quantity underneath it. The line is a least-squares fit "
               "per method; zero is the naive predictor."),
        decay_table(decay),
        '<hr class="rule">',
        "<h2>Which arms can actually be separated</h2>",
        NARRATIVE["separation"](summary, h2h, gaps, tally, distances, decay),
        tally_block(tally),
        figure("tukey_mae",
               "Tukey HSD on target-centred MAE, one panel per scheme. Blue is the best "
               "mean, grey an arm the correction cannot separate from it, red an arm that "
               "clears its interval and is worse."),
        figure("tukey_raw_mae",
               "The same test on the raw pooled numbers, which is how the paper aggregates. "
               "Every interval is wide enough to swallow the field, because the spread "
               "between targets is an order of magnitude larger than the spread between "
               "methods."),
        summary_table(summary, "mae"),
        NARRATIVE["robustness"](summary, h2h, gaps, tally, distances, decay),
        '<hr class="rule">',
        "<h2>k-NN against the rest</h2>",
        NARRATIVE["knn"](summary, h2h, gaps, tally, distances, decay),
        h2h_table(h2h, cfg.KNN_METHOD, cfg.MONROE35_METHOD),
        figure(f"paired_{cfg.KNN_METHOD}_vs_{cfg.MONROE35_METHOD}_mae",
               "Every fold, k-NN to Monroe + TabPFN 3.5, one panel per scheme."),
        '<hr class="rule">',
        "<h2>Ranking, not just error</h2>",
        NARRATIVE["ranking"](summary, h2h, gaps, tally, distances, decay),
        summary_table(summary, "spearman"),
        figure("tukey_spearman",
               "Tukey HSD on target-centred Spearman ρ. The naive baseline is absent: a "
               "constant prediction has no ranking to score."),
        '<hr class="rule">',
        "<h2>How this was run</h2>",
        NARRATIVE["methods"](summary, h2h, gaps, tally, distances, decay),
        "<h2>References</h2>",
        page_kit.render_references(REFERENCES),
        "</div>",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    PAGE.write_text(build(load()))
    print(f"wrote {PAGE} ({PAGE.stat().st_size / 1e6:.1f} MB)")
