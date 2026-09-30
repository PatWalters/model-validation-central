#!/usr/bin/env python
"""Step 5: the report, both data sets on one self-contained page.

One frozen molecular representation, one reference head, and every tabular
foundation model in NVIDIA's structured-data-models package dropped into the
head's slot. Everything is inlined, figures included, so the file stands alone.

This page is deliberately not part of the site index: the study has no
study.json, so site/build_site.py neither lists it nor copies it, and it does not
enter the cross-study table. It is published on its own, by this script writing
a second copy to docs/reports/sdm-tabular.html, which GitHub Pages serves.

    for ds in expansion biogen; do
      ADME_DATASET=$ds python 02_collect_metrics.py
      ADME_DATASET=$ds python 03_report.py
    done
    python 04_head_summary.py
    python 05_build_page.py

With ADME_SKIP=sdm_tabfm on every step, the page is built from the arms that
have finished and says which one has not.
"""

import numpy as np
import pandas as pd

import config as cfg
import page_kit

PAGE = cfg.PROJECT_DIR / "results" / "report.html"
PUBLISHED = cfg.PROJECT_DIR.parent.parent / "docs" / "reports" / "sdm-tabular.html"
MAIN, OTHER = "expansion", "biogen"
METHODS = cfg.METHODS
REF = cfg.MONROE_METHOD
LGBM = cfg.LGBM_METHOD
HEADS = [m for m in METHODS if m not in (REF, LGBM)]
SDM = [m for m in cfg.SDM_METHODS if m in METHODS]
KUMO = [m for m in SDM if m.startswith("sdm_kumo")]

COLOR = {
    LGBM: "#4C72B0", REF: "#937860", cfg.MONROE_TABICL_METHOD: "#C0A08B",
    "sdm_tabiclv2": "#A0765A", "sdm_kumo_small": "#A6D38A",
    "sdm_kumo_medium": "#6DB35A", "sdm_kumo_large": "#2F7A2A", "sdm_tabfm": "#8172B3",
}
SHORT = {
    LGBM: "LightGBM", REF: "TabPFN 3 (ref)", cfg.MONROE_TABICL_METHOD: "TabICL (tabicl)",
    "sdm_tabiclv2": "TabICLv2 (sdm)", "sdm_kumo_small": "Kumo S",
    "sdm_kumo_medium": "Kumo M", "sdm_kumo_large": "Kumo L", "sdm_tabfm": "TabFM",
}
LABEL = {
    **cfg.METHOD_LABELS,
    REF: "Monroe + TabPFN 3 (reference)",
}


# --- loading -------------------------------------------------------------
def figure(name: str, caption: str, ds: str) -> str:
    path = cfg.paths(ds).figures / f"{name}.png"
    return page_kit.figure_block(page_kit.embed_figure(path), caption, name)


def summary(ds: str) -> pd.DataFrame:
    s = pd.read_csv(cfg.paths(ds).tables / "summary.csv")
    return s[s["method"].isin(METHODS)]


def results(name: str) -> pd.DataFrame:
    return pd.read_csv(cfg.PROJECT_DIR / "results" / name)


# --- tallies -------------------------------------------------------------
def tally_counts(s: pd.DataFrame) -> pd.DataFrame:
    """Best alone / tied for best / worse, with the same tie rule as the site."""
    on_top = s["tukey_group"].isin(("best", "equivalent"))
    sizes = s[on_top].groupby(["endpoint", "metric"]).size()
    shared = s.apply(lambda r: int(sizes.get((r["endpoint"], r["metric"]), 1)) > 1, axis=1)
    kind = pd.Series("worse", index=s.index)
    kind[on_top & shared] = "tied"
    kind[on_top & ~shared] = "alone"
    return pd.crosstab(s["method"], kind).reindex(
        index=METHODS, columns=["alone", "tied", "worse"], fill_value=0)


def tukey_tally(s: pd.DataFrame) -> str:
    counts = tally_counts(s)
    cards = []
    for m in METHODS:
        row = counts.loc[m]
        cards.append(
            f'<div class="tally"><span class="swatch big" style="background:{COLOR[m]}"></span>'
            f'<span class="tallyname">{LABEL[m]}</span>'
            f'<span class="tallynums"><b>{int(row["alone"])}</b> best alone · '
            f'<b>{int(row["tied"])}</b> tied for best · <b>{int(row["worse"])}</b> worse'
            "</span></div>")
    return '<div class="tallies">' + "".join(cards) + "</div>"


# --- tables --------------------------------------------------------------
def metric_table(s: pd.DataFrame, metric: str, ds: str) -> str:
    """Mean +/- sd per endpoint and method, the Tukey grouping as a chip."""
    sub = s[s["metric"] == metric]
    on_top = sub[sub["tukey_group"].isin(("best", "equivalent"))]
    sizes = on_top.groupby("endpoint").size()
    rows = []
    for endpoint in cfg.paths(ds).dataset.targets:
        g = sub[sub["endpoint"] == endpoint].set_index("method")
        cells = [f'<th scope="row">{endpoint}</th>']
        for m in METHODS:
            r = g.loc[m]
            top = r["tukey_group"] in ("best", "equivalent")
            note = ("tied" if sizes.get(endpoint, 1) > 1 else "best") if top else ""
            cls = "best" if top else "worse"
            chip = f'<span class="chip {"tied" if note == "tied" else "best"}">{note}</span>' if note else ""
            cells.append(f'<td class="{cls}"><span class="num">{r["mean"]:.3f}</span>'
                         f'<span class="sd">±{r["sd"]:.3f}</span>{chip}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(
        f'<th><span class="swatch" style="background:{COLOR[m]}"></span>{SHORT[m]}</th>'
        for m in METHODS)
    return ('<div class="bleed tablewrap"><table>'
            f'<thead><tr><th scope="col">Endpoint</th>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def calls_table(calls: pd.DataFrame) -> str:
    """Wins / no call / losses against the reference, per head and data set."""
    rows = []
    for head in HEADS:
        cells = [f'<th scope="row"><span class="swatch" style="background:{COLOR[head]}"></span>'
                 f"{LABEL[head]}</th>"]
        for ds in (MAIN, OTHER):
            g = calls[(calls["head"] == head) & (calls["dataset"] == cfg.DATASETS[ds].label)]
            n = g["call"].value_counts()
            cells.append(f'<td class="num"><span class="chip win">{n.get("win", 0)}</span></td>'
                         f'<td class="num">{n.get("no call", 0)}</td>'
                         f'<td class="num"><span class="chip loss">{n.get("loss", 0)}</span></td>')
        r2 = calls[(calls["head"] == head) & (calls["metric"] == "r2")]["gain"].mean()
        mae = calls[(calls["head"] == head) & (calls["metric"] == "mae")]["gain"].mean()
        cells.append(f'<td class="num">{r2:+.3f}</td><td class="num">{-mae:+.4f}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<div class="bleed tablewrap"><table><thead>'
        '<tr><th scope="col" rowspan="2">Head, on Monroe\'s embedding</th>'
        '<th colspan="3">ExpansionRx, 27</th><th colspan="3">Biogen, 18</th>'
        '<th colspan="2">Mean over 15 endpoints</th></tr>'
        "<tr><th>wins</th><th>no call</th><th>losses</th>"
        "<th>wins</th><th>no call</th><th>losses</th>"
        "<th>Δ R²</th><th>Δ MAE</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>")


def control_table(control: pd.DataFrame) -> str:
    rows = []
    for metric in cfg.METRICS:
        g = control[control["metric"] == metric]
        rows.append(
            f'<tr><th scope="row">{ {"r2": "R²", "spearman": "Spearman ρ", "mae": "MAE"}[metric] }</th>'
            f'<td class="num">{g["diff"].abs().median():.4f}</td>'
            f'<td class="num">{g["diff"].abs().max():.4f}</td>'
            f'<td class="num">{g["abs_fold_diff"].mean():.4f}</td>'
            f'<td class="num">{int((g["p"] < 0.05).sum())} / {len(g)}</td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Metric</th>'
            "<th>Median |mean Δ|</th><th>Largest |mean Δ|</th>"
            "<th>Mean |Δ| on one fold</th><th>Endpoints p &lt; 0.05</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def cost_table(cost: pd.DataFrame, calls: pd.DataFrame) -> str:
    rows = []
    for head in [h for h in SDM if h in cost.index]:
        r = cost.loc[head]
        g = calls[calls["head"] == head]["call"].value_counts()
        spec = next(v for v in cfg.SDM_MODELS.values() if v["method"] == head)
        rows.append(
            f'<tr><th scope="row"><span class="swatch" style="background:{COLOR[head]}"></span>'
            f"{LABEL[head]}</th>"
            f'<td class="num">{spec["num_estimators"]}</td>'
            f'<td class="num">{r["median_s"]:.1f}</td><td class="num">{r["max_s"]:.1f}</td>'
            f'<td class="num">{r["total_min"]:.0f}</td>'
            f'<td class="num">{g.get("win", 0)} / {g.get("loss", 0)}</td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Head</th>'
            "<th>Estimators</th><th>Median s / fold</th><th>Slowest fold, s</th>"
            "<th>All 375 folds, min</th><th>Wins / losses vs ref</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# --- references ----------------------------------------------------------
REFERENCES = [
    ("sdm", "NVIDIA. ", "Structured Data Models.",
     "GitHub repository, 2026, Apache 2.0; run here at commit 5d66336.",
     "https://github.com/NVIDIA/structured-data-models", "github.com/NVIDIA/structured-data-models"),
    ("tabicl", "Qu, J.; <i>et al.</i> ",
     "TabICLv2: A Better, Faster, Scalable, and Open Tabular Foundation Model.",
     "ICML 2026. Weights BSD-3-Clause, at huggingface.co/jingang/TabICL.",
     "https://arxiv.org/abs/2602.11139", "arXiv:2602.11139"),
    ("kumo", "Qu, J.; <i>et al.</i> ",
     "NVIDIA Kumo Tabular Sets a New Accuracy-Efficiency Frontier for Tabular Prediction.",
     "Hugging Face blog, 2026. Weights under OpenMDW 1.1, at huggingface.co/nvidia/Kumo-Tabular.",
     "https://huggingface.co/blog/nvidia/kumo-tabular", "huggingface.co/blog/nvidia/kumo-tabular"),
    ("tabfm", "Kong, W.; <i>et al.</i> ",
     "Introducing TabFM: A Zero-shot Foundation Model for Tabular Data.",
     "Google Research blog, 2026. Weights under the TabFM Non-Commercial License v1.0.",
     "https://research.google/blog/introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data",
     "research.google/blog"),
    ("monroe", "Banaszewski, B.; Fitzgibbon, A. W. ",
     "Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference.",
     "Preprint, 2026. The frozen encoder every arm on this page reads.",
     "https://arxiv.org/abs/2608.18982", "arXiv:2608.18982"),
    ("tabpfn", "Hollmann, N.; Müller, S.; Purucker, L.; <i>et al.</i> ",
     "Accurate Predictions on Small Data with a Tabular Foundation Model.",
     '<span class="venue">Nature</span> 2025, 637 (8045), 319–326. TabPFN 3 is the reference head.',
     "https://doi.org/10.1038/s41586-024-08328-6", "10.1038/s41586-024-08328-6"),
    ("expansion", "OpenADMET; Expansion Therapeutics. ",
     "OpenADMET–ExpansionRx Blind Challenge Data.",
     "Hugging Face, 2026, CC BY 4.0.",
     "https://doi.org/10.57967/hf/9687", "10.57967/hf/9687"),
    ("biogen", "Fang, C.; Wang, Y.; Grater, R.; <i>et al.</i> ",
     "Prospective Validation of Machine Learning Algorithms for Absorption, Distribution, "
     "Metabolism, and Excretion Prediction: An Industrial Perspective.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2023, 63 (11), 3263–3274.',
     "https://doi.org/10.1021/acs.jcim.3c00160", "10.1021/acs.jcim.3c00160"),
    ("tukey", "Ash, J. R.; Wognum, C.; Rodríguez-Pérez, R.; <i>et al.</i> ",
     "Practically Significant Method Comparison Protocols for Machine Learning in Small "
     "Molecule Drug Discovery.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2025, 65 (18), 9398–9411.',
     "https://doi.org/10.1021/acs.jcim.5c01609", "10.1021/acs.jcim.5c01609"),
]
REF_NUMBER = page_kit.reference_numbers(REFERENCES)


def ref(*keys: str) -> str:
    return page_kit.marker(REF_NUMBER, *keys)


LINK_BLOG = ('<a href="https://practicalcheminformatics.blogspot.com/2025/03/'
             'even-more-thoughts-on-ml-method.html">“Even More Thoughts on ML Method '
             "Comparisons”</a>")
LINK_SOURCE = ('<a href="expansion-ml-comparison.html">the nine-method foundation-model '
               "comparison</a>")


def tabfm_paragraph(calls: pd.DataFrame, cost: pd.DataFrame) -> str:
    """TabFM's place in the ranking, and what it costs to get there."""
    if "sdm_tabfm" not in METHODS:
        return ""
    n = calls[calls["head"] == "sdm_tabfm"]["call"].value_counts()
    large_w = large_l = 0
    for ds in (MAIN, OTHER):
        m = pd.read_csv(cfg.paths(ds).fold_metrics)
        for _, g in m.groupby("endpoint"):
            for metric in cfg.METRICS:
                wide = g.pivot_table(index=["repeat", "fold"], columns="method", values=metric)
                sign = 1 if cfg.METRIC_HIGHER_IS_BETTER[metric] else -1
                d = sign * (wide["sdm_kumo_large"] - wide["sdm_tabfm"])
                from scipy.stats import ttest_rel
                if ttest_rel(wide["sdm_kumo_large"], wide["sdm_tabfm"]).pvalue < 0.05:
                    large_w += d.mean() > 0
                    large_l += d.mean() < 0
    ratio = cost.loc["sdm_tabfm", "total_min"] / cost.loc["sdm_kumo_large", "total_min"]
    efflux = calls[(calls["head"] == "sdm_tabfm") & (calls["endpoint"] == "LOG_Caco_Efflux")
                   & (calls["metric"] == "r2")]["gain"].iloc[0]
    return (
        f"<p>TabFM, at 1.6 billion parameters the largest model here, lands among the "
        f"Kumo models rather than above them: {n.get('win', 0)} wins against "
        f"{n.get('loss', 0)} losses, more wins than Kumo small and more than twice the "
        f"losses. Its best endpoint is ExpansionRx Caco-2 efflux, {efflux:+.3f} R² over "
        "the reference. It loses to the reference on mouse microsomal stability, as every "
        "head does, and on mouse plasma binding, as the smaller Kumo models do. Against "
        f"Kumo large directly it loses: Kumo large is significantly ahead on {large_w} of "
        f"45 endpoint × metric combinations and behind on {large_l}, and TabFM takes "
        f"{ratio:.1f} times the GPU time to get there.</p>")


def build() -> str:
    s_main, s_other = summary(MAIN), summary(OTHER)
    calls = results("head_vs_reference.csv")
    calls = calls[calls["head"].isin(HEADS)]
    control = results("tabicl_control.csv")
    spread = results("head_spread.csv")
    cost = results("cost.csv").set_index("method")

    counts = calls.groupby(["head", "call"]).size().unstack(fill_value=0)
    best = max(SDM, key=lambda h: counts.loc[h, "win"] - counts.loc[h, "loss"])
    best_w, best_l = int(counts.loc[best, "win"]), int(counts.loc[best, "loss"])
    n_calls = len(calls[calls["head"] == best])
    tab_w = int(counts.loc["sdm_tabiclv2", "win"])
    tab_l = int(counts.loc["sdm_tabiclv2", "loss"])
    tab_bio_l = int(((calls["head"] == "sdm_tabiclv2") & (calls["call"] == "loss")
                     & (calls["dataset"] == "Biogen ADME")).sum())
    gain = calls[calls["head"] == best].pivot_table(index="dataset", columns="metric",
                                                    values="gain", aggfunc="mean")
    ctl_max_r2 = control[control["metric"] == "r2"]["diff"].abs().max()
    ctl_max_mae = control[control["metric"] == "mae"]["diff"].abs().max()
    ctl_sig = int((control["p"] < 0.05).sum())
    med_spread = spread["head_spread"].median()
    med_lead = spread["ref_over_lgbm"].median()
    n_models = sum(int(cost.loc[h, "folds"]) for h in SDM if h in cost.index)

    tally_m, tally_o = tally_counts(s_main), tally_counts(s_other)
    n_large_top = int(tally_m.loc["sdm_kumo_large", ["alone", "tied"]].sum()
                      + tally_o.loc["sdm_kumo_large", ["alone", "tied"]].sum())
    top_m = {m: int(tally_m.loc[m, "alone"] + tally_m.loc[m, "tied"]) for m in METHODS}
    top_o = {m: int(tally_o.loc[m, "alone"] + tally_o.loc[m, "tied"]) for m in METHODS}

    pending = ""
    if "sdm_tabfm" not in METHODS:
        pending = ('<div class="panel howto"><h3>One arm is still running</h3>'
                   "<p>TabFM, Google's 1.6-billion-parameter model and the fifth head in "
                   "the package, takes about twenty times as long per fold as the others "
                   "and is still being fitted. This page covers the four heads that have "
                   "finished and will be rebuilt with TabFM when it lands.</p></div>")

    kumo_rows = []
    for ds in (MAIN, OTHER):
        h2h = pd.read_csv(cfg.paths(ds).tables / "head_to_head.csv")
        g = h2h[(h2h["left"] == "sdm_kumo_small") & (h2h["right"] == "sdm_kumo_large")]
        hib = g["metric"].ne("mae")
        better = (g["mean_diff"] > 0) == hib
        kumo_rows.append((cfg.DATASETS[ds].label, int(((g["p_value"] < .05) & better).sum()),
                          int(((g["p_value"] < .05) & ~better).sum()), len(g)))

    parts = [
        "<title>Swapping the Head</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">Tabular foundation models · 5×5 cross validation · two data sets</p>',
        "<h1>Swapping the head</h1>",
        '<p class="lede">NVIDIA\'s structured-data-models package'
        + ref("sdm")
        + " puts several tabular foundation models behind one in-context interface: "
        "TabICLv2" + ref("tabicl") + ", Kumo Tabular in three sizes" + ref("kumo")
        + " and Google's TabFM" + ref("tabfm") + ". Each is the same kind of object as "
        "the TabPFN" + ref("tabpfn") + " that reads Monroe's frozen molecular embedding"
        + ref("monroe") + " in the strongest arm of " + LINK_SOURCE + ". So each one is "
        "dropped into that slot here, with the representation, the folds and the test "
        "sets held fixed, and scored against Monroe + TabPFN 3 on fifteen ADME endpoints. "
        f"Kumo large is the best of them: on top of all {n_large_top} endpoint × metric "
        f"combinations after Tukey's correction, and ahead of the reference on {best_w} "
        f"of {n_calls} in a paired test, behind on {best_l}. It is also a small effect "
        "next to what the representation is worth.</p>",
        '<div class="facts">'
        f'<div class="fact"><b>{len(SDM)}</b><span>NVIDIA heads</span></div>'
        '<div class="fact"><b>1</b><span>frozen encoder</span></div>'
        '<div class="fact"><b>15</b><span>endpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b><span>cross validation</span></div>'
        f'<div class="fact"><b>{n_models:,}</b><span>fold models fitted</span></div>'
        "</div>",
        pending,
        '<div class="panel howto"><h3>What differs between the arms, and what does not</h3>'
        "<p>Every arm on this page reads the same 720-dimensional Monroe embedding of each "
        "molecule, cached once and never updated. Every arm is handed the same training "
        "molecules in every fold — the four fifths of the training set outside the "
        "held-out fifth that have a value for the endpoint — and is scored on the same "
        "untouched test set. Nothing is trained downstream in any of them: each head "
        "takes the labelled training rows as context and predicts the test rows in one "
        "forward pass. So a difference between two arms is the head and nothing else.</p>"
        "<p>Nothing is tuned either. Each NVIDIA model runs at the ensemble size and "
        "precision NVIDIA's own TabArena adapter uses for it, with its own default "
        "preprocessing, and the point prediction is what that adapter returns: the mean "
        "of the predictive distribution, which is also what the reference head reports. "
        "LightGBM on Morgan counts is carried along as the floor every arm has to clear.</p>"
        "</div>",

        "<h2>The scoreboard</h2>",
        "<p>Tukey HSD per endpoint and metric, counted the way every page on this site "
        "counts: a method is on top if the correction cannot separate it from the best "
        "mean, and it is only <em>best alone</em> if nothing else is on top with it. The "
        "tables show MAE, in the units of each endpoint.</p>",
        "<h3>ExpansionRx" + ref("expansion") + "</h3>",
        tukey_tally(s_main),
        metric_table(s_main, "mae", MAIN),
        figure("tukey_mae", "Tukey HSD on MAE, ExpansionRx. Each bar is a method's mean "
               "over 25 folds with an interval widened to cover every pairwise comparison "
               "at once. Blue is the best mean, grey cannot be separated from it, red is "
               "worse.", MAIN),
        "<h3>Biogen ADME" + ref("biogen") + "</h3>",
        tukey_tally(s_other),
        metric_table(s_other, "mae", OTHER),
        figure("tukey_mae", "Tukey HSD on MAE, Biogen ADME.", OTHER),
        f"<p>With seven heads on one representation, most combinations end in a tie, "
        f"and one arm is never out of it. Kumo large is on top of "
        f"{top_m['sdm_kumo_large']} of 27 ExpansionRx combinations and "
        f"{top_o['sdm_kumo_large']} of 18 Biogen ones — no other arm on either data set "
        f"is never significantly worse than the best. The TabPFN 3 reference is on top "
        f"of {top_m[REF]} and {top_o[REF]}, and the two TabICLv2 arms of the same number "
        f"or one more. LightGBM on Morgan counts reaches the top of "
        f"{top_m[LGBM] + top_o[LGBM]} of 45.</p>",

        "<h2>Against the reference</h2>",
        "<p>The scoreboard asks who is best. The question here is narrower: does swapping "
        "TabPFN 3 for one of NVIDIA's heads help? The folds are the pairing, which is "
        "legitimate because every arm saw identical training molecules in all 25. A win "
        "or a loss is a paired t-test at p &lt; 0.05 on one endpoint and one metric, "
        "uncorrected, so read the counts as a direction rather than a verdict on any "
        "single cell.</p>",
        calls_table(calls),
        figure("delta_vs_reference_mae", "Each head against Monroe + TabPFN 3 on "
               "ExpansionRx: mean paired MAE difference over 25 folds with a 95% interval, "
               "signed so that right of zero is better than the reference.", MAIN),
        figure("delta_vs_reference_mae", "The same on Biogen ADME.", OTHER),
        "<p>All three Kumo Tabular sizes beat the reference more often than they lose "
        f"to it, and the large one most clearly: {best_w} wins against {best_l} losses, "
        f"a mean R² gain of {gain.loc['ExpansionRx', 'r2']:+.3f} on ExpansionRx and "
        f"{gain.loc['Biogen ADME', 'r2']:+.3f} on Biogen. The gains are largest on LogD, "
        "human microsomal stability and the two Caco-2 endpoints. The reference holds "
        "on ExpansionRx mouse microsomal stability, where every head is behind it, and "
        "on Biogen rat plasma protein binding, where every head but Kumo small and TabFM "
        "is. "
        "Tissue binding is otherwise mixed: on ExpansionRx brain and muscle binding the "
        "two TabICLv2 heads are ahead of it, Kumo large on brain binding and TabFM on "
        "muscle binding.</p>"
        f"<p>TabICLv2 goes the other way: {tab_w} wins against {tab_l} losses, and the "
        f"split is by data set. {tab_bio_l} of those losses are on Biogen, where it trails "
        "the reference on every endpoint but human plasma binding. On ExpansionRx it "
        "wins more than it loses, apart from mouse microsomal stability and Caco-2 A→B "
        "permeability, where it gives back 0.026 MAE — the widest gap to the reference "
        "anywhere on the page.</p>",
        tabfm_paragraph(calls, cost),

        "<h2>The control: one checkpoint, two implementations</h2>",
        "<p>The source comparison already had a Monroe + TabICL arm, fitted with the "
        "TabICL authors' own <code>tabicl</code> package, version 2.1.1. That release "
        "loads the same checkpoint NVIDIA's port does, "
        "<code>tabicl-regressor-v2-20260212.ckpt</code>. Putting the two side by side "
        "tests the wrapper rather than the model. If NVIDIA's reimplementation were "
        "subtly wrong, the other heads' numbers would mean less.</p>",
        control_table(control),
        f"<p>They agree. The largest mean difference on any endpoint is {ctl_max_r2:.4f} "
        f"in R² and {ctl_max_mae:.4f} in MAE, an order of magnitude below the gaps "
        f"between heads. {ctl_sig} of 45 endpoint × metric combinations do reach p "
        "&lt; 0.05, and all of them favour NVIDIA's port by a hair. Paired folds are "
        "sensitive enough to see differences of a few thousandths, which is what two "
        "implementations with different preprocessing and random ensemble members "
        "produce. Against the reference the two land in exactly the same place, "
        f"{tab_l} losses each.</p>",

        "<h2>Does size help?</h2>",
        "<p>Kumo Tabular ships in three sizes: 28 million, 63 million and 216 million "
        "parameters, the largest also running twice the ensemble. Paired over the same "
        "folds, large against small:</p>",
        "<ul>" + "".join(
            f"<li><b>{label}</b>: large significantly better on {w} of {n} endpoint × "
            f"metric combinations, worse on {l}.</li>" for label, w, l, n in kumo_rows)
        + "</ul>",
        "<p>Size helps, but not monotonically: the medium model buys nothing over the "
        f"small one, losing to the reference {int(counts.loc['sdm_kumo_medium', 'loss'])} "
        f"times to the small model's {int(counts.loc['sdm_kumo_small', 'loss'])}. The "
        "large model's "
        "edge over the small one is real but a few thousandths of MAE on most endpoints. "
        "It costs about seven times as much GPU time.</p>",
        cost_table(cost, calls),
        "<p>Wall-clock time on one RTX 5070 Ti, per fold, context and test set in one "
        "forward pass. The largest fold is ExpansionRx LogD, with 4,230 training rows and "
        "2,270 test rows. Kumo small is the fastest head on the page and beats the "
        "reference more often than it loses to it.</p>"
        + ("<p>TabFM is the exception to one forward pass. On the two largest "
           "ExpansionRx endpoints, LogD and LogS, 4,230 context rows and 2,270 test rows "
           "do not fit in 16 GB at once, so those 50 folds predict the test rows 1,024 at a "
           "time. A test row attends only to the context, never to another test row, and "
           "every chunk is seeded identically, so this is the single pass computed in "
           "parts; checked on a fold that does fit, the two agree to within 3×10<sup>−3</sup> "
           "on a 6.9-unit range, which is half-precision rounding. Every other fold, TabFM "
           "and otherwise, is one pass and reproduces bit for bit.</p>"
           if "sdm_tabfm" in METHODS else ""),

        "<h2>The head is the small lever</h2>",
        "<p>The source comparison found that the representation is worth about ten times "
        "the head, from two representations and two heads. This page has seven heads on "
        "one representation, which prices the head more precisely.</p>",
        f"<p>On each endpoint, the spread in mean R² between the best and worst "
        f"in-context head has a median of {med_spread:.3f}. Monroe + TabPFN 3's lead "
        f"over LightGBM on Morgan counts — a change of representation and of model — has "
        f"a median of {med_lead:.3f}. Choosing among modern tabular foundation models is "
        f"worth about a fifth of what choosing the representation is worth. Kumo large "
        "is the best head here, but a better head does not turn a frozen encoder into a "
        "fine-tuned one: on the three ExpansionRx tissue-binding endpoints, CheMeleon's "
        "fine-tuned graph network is still ahead of every head on this page by 0.02 to "
        "0.05 MAE.</p>",

        "<h2>What this does and does not say</h2>",
        '<div class="panel howto">'
        "<h3>Scope</h3>"
        "<p><b>One representation.</b> Every head reads Monroe's embedding. Whether "
        "Kumo's lead holds on fingerprints, descriptors or another encoder is not tested "
        "here, and heads can rank differently on different inputs.</p>"
        "<p><b>Defaults only.</b> No head is tuned, and none is fine-tuned, although "
        "the package supports fine-tuning Kumo small and TabICLv2. The Kumo and TabFM "
        "recipes keep at most 500 of the 720 embedding dimensions per ensemble member, "
        "shuffled differently for each, so no single member sees the whole embedding. "
        "TabICLv2 reads all 720.</p>"
        "<p><b>Two of the package's models are not arms.</b> TimesFM 3 forecasts time "
        "series and Kumo Relational models linked tables. Neither has anything to say "
        "about one table of molecules with one numeric target.</p>"
        "<p><b>Licences.</b> TabICLv2's weights are BSD-3-Clause and Kumo Tabular's are "
        "OpenMDW. TabFM's are under Google's TabFM Non-Commercial License, which allows "
        "benchmarking like this and nothing commercial; it was accepted for this study "
        "on that basis.</p>"
        "</div>",

        f'<p class="footnote">Reported the way {LINK_BLOG} and the protocol paper it '
        "points at" + ref("tukey") + " argue comparisons should be: distributions over "
        "folds, simultaneous confidence intervals, paired tests on the folds, and no "
        "bolded maxima. Where two methods cannot be separated, both are reported as "
        "tied.</p>",
        '<p class="repo">Code, per-fold metrics and every figure: '
        '<a href="https://github.com/patwalters/model-validation-central/tree/main/studies/sdm-tabular">'
        "studies/sdm-tabular</a>.</p>",
        "<h2>References</h2>",
        page_kit.render_references(REFERENCES),
        "</div>",
    ]
    return "\n".join(p for p in parts if p)


def main() -> None:
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    html = build()
    PAGE.write_text(html)
    print(f"wrote {PAGE} ({len(html) / 1e6:.1f} MB)")
    if "sdm_tabfm" in METHODS:
        PUBLISHED.write_text(html)
        print(f"published {PUBLISHED.relative_to(PUBLISHED.parents[2])}")
    else:
        print("not published: an interim page, built with arms skipped")


if __name__ == "__main__":
    main()
