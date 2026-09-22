#!/usr/bin/env python
"""Step 17: the TDiMS comparison as one self-contained page.

The other two reports out of this directory ask which pre-trained representation
transfers, and whether picking a model per endpoint beats picking an
architecture. This one asks something narrower and older: is a hand-designed
molecular descriptor still competitive with a pre-trained encoder, when both are
read by the same model?

Three arms, because three is what the question needs. TDiMS and Monroe share a
head -- the same TabPFN 3.5, the same ensemble settings, the same folds, the same
fit and test masks, the same per-fold seeds -- so the difference between them is
the representation and nothing else. LightGBM on Morgan counts is the third,
carried along as the baseline both of them are supposed to beat.

This page is deliberately not wired into the site index: study.json is untouched,
so site/build_site.py does not know about it. It is published on its own.

Everything is inlined, figures included, so the file stands alone.

    ADME_COMPARISON=tdims python 05_report.py
    ADME_COMPARISON=tdims ADME_DATASET=biogen python 05_report.py
    python 16_tdims_vs_monroe.py
    python 17_build_tdims_page.py
"""

import numpy as np
import pandas as pd

import config as cfg
import page_kit

PAGE = cfg.PROJECT_DIR / "results" / "tdims_report.html"

COMPARISON = "tdims"
MAIN = "expansion"
OTHER = "biogen"

METHOD_COLOR = page_kit.METHOD_COLOR
SHORT = page_kit.SHORT
METHODS = cfg.COMPARISONS[COMPARISON]

TDIMS = cfg.TDIMS_METHOD
MONROE = cfg.MONROE35_METHOD
LGBM = cfg.LGBM_METHOD

# How the configuration tags read in prose.
DIS_LABEL = {"dm2": "x<sup>−2</sup>", "dm1": "x<sup>−1</sup>", "dp1": "x"}


# --- loading -------------------------------------------------------------
def figure(name: str, caption: str, ds: str = MAIN) -> str:
    path = cfg.paths(ds, COMPARISON).figures / f"{name}.png"
    return page_kit.figure_block(page_kit.embed_figure(path), caption, name)


def load(ds: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = cfg.paths(ds, COMPARISON)
    return (
        pd.read_csv(paths.tables / "summary.csv"),
        pd.read_csv(paths.tables / "head_to_head.csv"),
        pd.read_csv(paths.fold_metrics),
    )


def chosen(ds: str) -> pd.DataFrame:
    """The configuration each endpoint chose, and the spread it chose across."""
    scores = pd.read_csv(cfg.paths(ds, COMPARISON).results / "tdims_config.csv")
    means = scores.groupby(["endpoint", "config"])["val_r2"].mean()
    rows = []
    for endpoint, g in means.groupby(level="endpoint"):
        rows.append({
            "endpoint": endpoint,
            "config": g.idxmax()[1],
            "val_r2": g.max(),
            "spread": g.max() - g.min(),
            "n_tested": len(g),
        })
    return pd.DataFrame(rows).set_index("endpoint")


def screen(ds: str) -> pd.DataFrame:
    return pd.read_csv(cfg.paths(ds, COMPARISON).results / "tdims_screen.csv")


# --- tallies -------------------------------------------------------------
def top_sizes(summary: pd.DataFrame) -> pd.Series:
    top = summary[summary["tukey_group"].isin(("best", "equivalent"))]
    return top.groupby(["endpoint", "metric"]).size()


def tally_counts(summary: pd.DataFrame) -> pd.DataFrame:
    sizes = top_sizes(summary)
    shared = summary.apply(
        lambda r: int(sizes.get((r["endpoint"], r["metric"]), 1)) > 1, axis=1
    )
    on_top = summary["tukey_group"].isin(("best", "equivalent"))
    kind = pd.Series("worse", index=summary.index)
    kind[on_top & shared] = "tied"
    kind[on_top & ~shared] = "alone"
    return pd.crosstab(summary["method"], kind).reindex(
        index=METHODS, columns=["alone", "tied", "worse"], fill_value=0
    )


def fold_models(metrics: pd.DataFrame, method: str) -> int:
    return len(metrics[metrics["method"] == method])


# --- tables --------------------------------------------------------------
def metric_table(summary: pd.DataFrame, metric: str, ds: str) -> str:
    """Mean +/- sd per endpoint and method, with the Tukey grouping as a chip."""
    sizes = top_sizes(summary)
    sub = summary[summary["metric"] == metric]
    methods = [m for m in METHODS if m in set(sub["method"])]
    rows = []
    for endpoint in cfg.paths(ds, COMPARISON).dataset.targets:
        g = sub[sub["endpoint"] == endpoint].set_index("method")
        if g.empty:
            continue
        cells = [f'<th scope="row">{endpoint}</th>']
        for m in methods:
            if m not in g.index:
                cells.append("<td></td>")
                continue
            r = g.loc[m]
            on_top = r["tukey_group"] in ("best", "equivalent")
            shared = int(sizes.get((endpoint, metric), 1)) > 1
            cls = "best" if on_top else "worse"
            note = ("tied" if shared else "best") if on_top else ""
            chip = f'<span class="chip {cls}">{note}</span>' if note else ""
            cells.append(
                f'<td class="{cls}"><span class="num">{r["mean"]:.3f}</span>'
                f'<span class="sd">±{r["sd"]:.3f}</span>{chip}</td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")

    head = "".join(
        f'<th><span class="swatch" style="background:{METHOD_COLOR[m]}"></span>{SHORT[m]}</th>'
        for m in methods
    )
    return (
        '<div class="bleed tablewrap"><table>'
        f'<thead><tr><th scope="col">Endpoint</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def h2h_table(h2h: pd.DataFrame, left: str, right: str, ds: str,
              metric: str = "r2") -> str:
    sub = h2h[(h2h["left"] == left) & (h2h["right"] == right) & (h2h["metric"] == metric)]
    if sub.empty:
        return '<p class="missing">comparison not available</p>'
    rows = []
    for endpoint in cfg.paths(ds, COMPARISON).dataset.targets:
        r = sub[sub["endpoint"] == endpoint]
        if r.empty:
            continue
        r = r.iloc[0]
        sig = r["p_value"] < 0.05
        # mean_diff is right minus left, so which sign favours `right` depends on
        # the metric. Only R² is shown here, but MAE would silently invert every
        # verdict on this table if the direction were assumed rather than read.
        higher_is_better = cfg.METRIC_HIGHER_IS_BETTER[metric]
        better = (r["mean_diff"] > 0) if higher_is_better else (r["mean_diff"] < 0)
        verdict = ("wins" if better else "loses") if sig else "no call"
        cls = ("win" if better else "loss") if sig else "null"
        rows.append(
            f'<tr><th scope="row">{endpoint}</th>'
            f'<td class="num">{r["mean_diff"]:+.3f}</td>'
            f'<td class="num">{int(r["right_wins"])}/{int(r["n_folds"])}</td>'
            f'<td class="num">{r["p_value"]:.1e}</td>'
            f'<td><span class="chip {cls}">{verdict}</span></td></tr>'
        )
    right_name = cfg.METHOD_LABELS[right]
    return (
        '<div class="bleed tablewrap"><table>'
        f'<thead><tr><th scope="col">Endpoint</th>'
        f'<th scope="col">Δ R² for {right_name}</th>'
        f'<th scope="col">Folds won</th><th scope="col">p</th>'
        '<th scope="col">Verdict</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def tukey_tally(summary: pd.DataFrame) -> str:
    counts = tally_counts(summary)
    cards = []
    for m in METHODS:
        row = counts.loc[m]
        cards.append(
            f'<div class="tally"><span class="swatch big" style="background:{METHOD_COLOR[m]}"></span>'
            f'<span class="tallyname">{cfg.METHOD_LABELS[m]}</span>'
            f'<span class="tallynums">'
            f'<b>{int(row["alone"])}</b> best alone · '
            f'<b>{int(row["tied"])}</b> tied for best · '
            f'<b>{int(row["worse"])}</b> worse</span></div>'
        )
    return '<div class="tallies">' + "".join(cards) + "</div>"


def rank_vs_fit_table(metrics: pd.DataFrame, ds: str) -> str:
    """The R² gap beside the Spearman gap, which is the calibration story."""
    rows = []
    for endpoint in cfg.paths(ds, COMPARISON).dataset.targets:
        sub = metrics[metrics["endpoint"] == endpoint]
        wide_r2 = sub.pivot_table(index=["repeat", "fold"], columns="method", values="r2")
        wide_sp = sub.pivot_table(index=["repeat", "fold"], columns="method", values="spearman")
        if not {TDIMS, MONROE}.issubset(wide_r2.columns):
            continue
        gap_r2 = (wide_r2[MONROE] - wide_r2[TDIMS]).mean()
        gap_sp = (wide_sp[MONROE] - wide_sp[TDIMS]).mean()
        ratio = gap_sp / gap_r2 if gap_r2 else np.nan
        rows.append(
            f'<tr><th scope="row">{endpoint}</th>'
            f'<td class="num">{gap_r2:.3f}</td>'
            f'<td class="num">{gap_sp:.3f}</td>'
            f'<td class="num">{ratio:.2f}</td></tr>'
        )
    return (
        '<div class="bleed tablewrap"><table>'
        '<thead><tr><th scope="col">Endpoint</th>'
        '<th scope="col">R² gap</th><th scope="col">Spearman gap</th>'
        '<th scope="col">Ratio</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def config_table(ds: str) -> str:
    """What each endpoint chose, and how much the choice was worth."""
    picks = chosen(ds)
    rows = []
    for endpoint in cfg.paths(ds, COMPARISON).dataset.targets:
        if endpoint not in picks.index:
            continue
        r = picks.loc[endpoint]
        radius, dis, merge, frag = r["config"].split("_")
        rows.append(
            f'<tr><th scope="row">{endpoint}</th>'
            f'<td class="num">{radius[1:]}</td>'
            f'<td class="num">{DIS_LABEL.get(dis, dis)}</td>'
            f'<td class="num">{merge}</td>'
            f'<td class="num">{"yes" if frag == "cep" else "no"}</td>'
            f'<td class="num">{r["val_r2"]:.3f}</td>'
            f'<td class="num">{r["spread"]:.3f}</td></tr>'
        )
    return (
        '<div class="bleed tablewrap"><table>'
        '<thead><tr><th scope="col">Endpoint</th><th scope="col">Radius</th>'
        '<th scope="col"><i>f</i><sub>dis</sub></th>'
        '<th scope="col"><i>f</i><sub>dup</sub></th>'
        '<th scope="col">CEP fragments</th>'
        '<th scope="col">Validation R²</th>'
        '<th scope="col">Spread across shortlist</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


# --- references ----------------------------------------------------------
REFERENCES = [
    (
        "tdims",
        "Hamada, L.; Kishimoto, A.; Miyaguchi, K.; <i>et al.</i> ",
        "Revisiting Molecular Descriptors with TDiMS for Interpretable "
        "Intramolecular Interactions Based on Substructure Pairs.",
        '<span class="venue">Nat. Comput. Sci.</span> 2026, 6, 945–953. '
        "Descriptor code at "
        '<a href="https://github.com/IBM/materials/tree/main/models/tdims">'
        "github.com/IBM/materials</a>, Apache 2.0.",
        "https://doi.org/10.1038/s43588-026-01036-3",
        "10.1038/s43588-026-01036-3",
    ),
    (
        "monroe",
        "Banaszewski, B.; Fitzgibbon, A. W. ",
        "Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference.",
        "Preprint, 2026. A GRIT graph transformer pre-trained on 81 million PM6 molecules "
        "and 1,089 PCBA assays, adapted downstream by a frozen encoder and TabPFN rather "
        "than by fine-tuning. Code and weights at "
        '<a href="https://github.com/blazejba/monroe">github.com/blazejba/monroe</a>.',
        "https://arxiv.org/abs/2608.18982",
        "arXiv:2608.18982",
    ),
    (
        "tabpfn",
        "Hollmann, N.; Müller, S.; Purucker, L.; <i>et al.</i> ",
        "Accurate Predictions on Small Data with a Tabular Foundation Model.",
        '<span class="venue">Nature</span> 2025, 637 (8045), 319–326. '
        "TabPFN 3.5, released 15 September 2026, is the head both in-context arms use.",
        "https://doi.org/10.1038/s41586-024-08328-6",
        "10.1038/s41586-024-08328-6",
    ),
    (
        "morgan",
        "Rogers, D.; Hahn, M. ",
        "Extended-Connectivity Fingerprints.",
        '<span class="venue">J. Chem. Inf. Model.</span> 2010, 50 (5), 742–754. '
        "The baseline representation, as radius-2 counts over 2,048 bits.",
        "https://doi.org/10.1021/ci100050t",
        "10.1021/ci100050t",
    ),
    (
        "cep",
        "Hachmann, J.; Olivares-Amaya, R.; Atahan-Evrenk, S.; <i>et al.</i> ",
        "The Harvard Clean Energy Project: Large-Scale Computational Screening and "
        "Design of Organic Photovoltaics on the World Community Grid.",
        '<span class="venue">J. Phys. Chem. Lett.</span> 2011, 2 (17), 2241–2251. '
        "Source of the 26 ring fragments TDiMS optionally enumerates over.",
        "https://doi.org/10.1021/jz200866s",
        "10.1021/jz200866s",
    ),
    (
        "biogen",
        "Fang, C.; Wang, Y.; Grater, R.; <i>et al.</i> ",
        "Prospective Validation of Machine Learning Algorithms for Absorption, "
        "Distribution, Metabolism, and Excretion Prediction: An Industrial Perspective.",
        '<span class="venue">J. Chem. Inf. Model.</span> 2023, 63 (11), 3263–3274. '
        "The 3,521-compound public set is at "
        '<a href="https://github.com/molecularinformatics/Computational-ADME">'
        "github.com/molecularinformatics/Computational-ADME</a>.",
        "https://doi.org/10.1021/acs.jcim.3c00160",
        "10.1021/acs.jcim.3c00160",
    ),
    (
        "tukey",
        "Ash, J. R.; Wognum, C.; Rodríguez-Pérez, R.; <i>et al.</i> ",
        "Practically Significant Method Comparison Protocols for Machine Learning "
        "in Small Molecule Drug Discovery.",
        '<span class="venue">J. Chem. Inf. Model.</span> 2025, 65 (18), 9398–9411.',
        "https://doi.org/10.1021/acs.jcim.5c01609",
        "10.1021/acs.jcim.5c01609",
    ),
]

REF_NUMBER = page_kit.reference_numbers(REFERENCES)


def ref(*keys: str) -> str:
    return page_kit.marker(REF_NUMBER, *keys)


LINK_BLOG = (
    '<a href="https://practicalcheminformatics.blogspot.com/2025/03/'
    'even-more-thoughts-on-ml-method.html">“Even More Thoughts on ML Method '
    "Comparisons”</a>"
)


def build() -> str:
    summary, h2h, metrics = load(MAIN)
    bio_summary, bio_h2h, bio_metrics = load(OTHER)

    tally = tally_counts(summary)
    bio_tally = tally_counts(bio_summary)
    combos = len(summary) // len(METHODS)
    bio_combos = len(bio_summary) // len(METHODS)

    gaps = pd.read_csv(cfg.PROJECT_DIR / "results" / "tdims_vs_monroe35.csv")
    mean_gap = gaps["delta_r2"].mean()
    exp_gap = gaps[gaps["dataset"] == "ExpansionRx"]["delta_r2"].mean()
    bio_gap = gaps[gaps["dataset"] == "Biogen ADME"]["delta_r2"].mean()
    clean_sweeps = int((gaps["win_rate"] == 0).sum())
    below_lgbm = int((gaps["tdims_r2"] < gaps["lgbm_r2"]).sum())
    negative = int((gaps["tdims_r2"] < 0).sum())

    picks = pd.concat([chosen(MAIN), chosen(OTHER)])
    n_cep = int(picks["config"].str.contains("_cep").sum())
    n_r1 = int(picks["config"].str.startswith("r1").sum())
    max_spread = picks["spread"].max()

    n_models = fold_models(metrics, TDIMS) + fold_models(bio_metrics, TDIMS)
    n_endpoints = len(picks)

    tdims_top = int(tally.loc[TDIMS, "alone"] + tally.loc[TDIMS, "tied"])
    bio_tdims_top = int(bio_tally.loc[TDIMS, "alone"] + bio_tally.loc[TDIMS, "tied"])

    parts = [
        "<title>A descriptor against an encoder</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">5×5 cross validation · two data sets · one head</p>',
        "<h1>Does a hand-designed descriptor still compete?</h1>",
        '<p class="lede">TDiMS'
        + ref("tdims")
        + " enumerates every pair of substructures inside a molecule and stores a "
        "function of the topological distance between them. It exists to capture "
        "the long-range intramolecular interactions that local fingerprints miss, "
        "and on chromophores its paper reports it beating Mordred, MolFormer, "
        "MolCLR, Atom-Pair and MAP4. Here it is put on ADME data against a "
        "representation nobody designed — Monroe's frozen encoder"
        + ref("monroe")
        + " — with both read by the same TabPFN 3.5"
        + ref("tabpfn")
        + f". {n_models} fold models over {n_endpoints} endpoints, every one scored "
        "on molecules it never saw. The descriptor loses everywhere, but not for "
        "the reason the scoreboard suggests.</p>",
        '<div class="facts">'
        f'<div class="fact"><b>{len(METHODS)}</b><span>methods</span></div>'
        f'<div class="fact"><b>2</b><span>data sets</span></div>'
        f'<div class="fact"><b>{n_endpoints}</b><span>endpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b><span>cross validation</span></div>'
        f'<div class="fact"><b>24</b><span>descriptor configurations</span></div>'
        f'<div class="fact"><b>{n_models}</b><span>fold models</span></div>'
        "</div>",

        '<div class="panel howto">'
        "<h3>Why these two arms are comparable</h3>"
        "<p>Both arms are the same shape: a fixed representation, handed to "
        "TabPFN 3.5 with the fold's training labels, which returns the test "
        "predictions from one forward pass. Nothing is trained downstream in "
        "either. They share the call, the ensemble settings, "
        "<code>output_type=\"mean\"</code>, the per-fold seed, the folds, and the "
        "fit and test masks that every other arm in this project uses.</p>"
        "<p>So the gap between them is not a modelling choice, a tuning budget or "
        "a training schedule. It is what a molecule is turned into, and nothing "
        "else. LightGBM on Morgan counts"
        + ref("morgan")
        + " is here as the reference point both are supposed to beat.</p>"
        "</div>",

        "<h2>The scoreboard</h2>",
        f"<p>Monroe is on top of all {combos + bio_combos} endpoint × metric "
        f"combinations across the two data sets. TDiMS reaches the top of "
        f"{tdims_top} of {combos} on ExpansionRx and {bio_tdims_top} of "
        f"{bio_combos} on Biogen, and in the one place it gets there it is tied "
        "rather than ahead.</p>",
        "<h3>ExpansionRx</h3>",
        tukey_tally(summary),
        metric_table(summary, "r2", MAIN),
        figure("tukey_r2", "Tukey HSD on R², ExpansionRx. Each bar is a method's "
               "mean over 25 folds with an interval widened to cover every "
               "pairwise comparison at once.", MAIN),
        "<h3>Biogen ADME</h3>",
        tukey_tally(bio_summary),
        metric_table(bio_summary, "r2", OTHER),
        figure("tukey_r2", "Tukey HSD on R², Biogen ADME.", OTHER),

        "<h2>Head to head</h2>",
        "<p>The folds are the pairing, which is legitimate here because both arms "
        "saw identical training molecules in every one of the 25 replicates. A "
        "positive difference favours Monroe.</p>",
        "<h3>ExpansionRx</h3>",
        h2h_table(h2h, TDIMS, MONROE, MAIN),
        "<h3>Biogen ADME</h3>",
        h2h_table(bio_h2h, TDIMS, MONROE, OTHER),
        f"<p>The mean R² gap is {abs(exp_gap):.3f} on ExpansionRx and "
        f"{abs(bio_gap):.3f} on Biogen, {abs(mean_gap):.3f} over all "
        f"{n_endpoints} endpoints. On {clean_sweeps} of them Monroe wins all 25 "
        f"folds individually. TDiMS also finishes below LightGBM on Morgan counts "
        f"on {below_lgbm} of {n_endpoints}, and goes negative — worse than "
        f"predicting the training mean — on {negative}.</p>",
        figure("paired_tdims35_vs_monroe35_r2",
               "Paired folds, ExpansionRx. Each line joins the two arms on one "
               "fold; both saw the same training molecules.", MAIN),

        "<h2>It ranks better than it predicts</h2>",
        "<p>The scoreboard above is R², and R² punishes a model twice: once for "
        "ordering molecules wrongly and once for placing them wrongly. Splitting "
        "those apart changes the picture. Set the R² gap beside the Spearman gap "
        "on the same folds.</p>",
        "<h3>ExpansionRx</h3>",
        rank_vs_fit_table(metrics, MAIN),
        "<h3>Biogen ADME</h3>",
        rank_vs_fit_table(bio_metrics, OTHER),
        "<p>On the endpoints where TDiMS looks worst, the Spearman gap is a "
        "fraction of the R² gap: ExpansionRx microsomal stability loses 0.405 R² "
        "but only 0.084 Spearman. TDiMS is ordering those molecules far better "
        "than it is placing them. Most of what it loses on this page is "
        "calibration, not chemistry, and a use that only needs a ranking would "
        "see a much smaller difference than the first table implies.</p>"
        "<p>That is not true everywhere. On both Biogen binding endpoints and on "
        "Caco-2 efflux the two gaps are closer together, and there the descriptor "
        "really is failing to order the molecules.</p>",

        "<h2>The configuration is not what decides it</h2>",
        "<p>TDiMS is not one descriptor but a family. Its paper searches four "
        "axes — the CEP ring fragments"
        + ref("cep")
        + " in or out, Morgan radius 1 or 2, <i>f</i><sub>dis</sub> ∈ "
        "{x<sup>−2</sup>, x<sup>−1</sup>, x} and <i>f</i><sub>dup</sub> ∈ "
        "{sum, max} — and passes the most promising of the 24 to the estimator. "
        "This arm searches the same 24, on the fifth of the training molecules "
        "each fold already holds out and that neither LightGBM nor Monroe ever "
        "uses. A ridge regression screens all 24 on one fold, and the best four "
        "per endpoint are then scored by the real TabPFN head over three.</p>",
        "<h3>ExpansionRx</h3>",
        config_table(MAIN),
        "<h3>Biogen ADME</h3>",
        config_table(OTHER),
        f"<p>No configuration wins everywhere: {n_cep} of {n_endpoints} endpoints "
        f"keep the CEP fragments, {n_r1} take radius 1, and Biogen reaches for the "
        "raw-distance <i>f</i><sub>dis</sub> = x on four of six endpoints where "
        "ExpansionRx picks it exactly once. The organic-electronics ring list "
        "earning its place on drug-like molecules is the one genuine surprise "
        "here.</p>",
        "<p>But the last column is the point. Across the four configurations "
        f"actually scored with TabPFN, best minus worst never exceeds "
        f"{max_spread:.3f}, and on ExpansionRx it stays under 0.1. The deficit to "
        f"Monroe is {abs(mean_gap):.3f}. Even if the screen discarded a "
        "configuration better than every one tested, the configuration is an "
        "order of magnitude too small a lever to explain this result — and the "
        "elaborate per-endpoint search the method calls for is buying very "
        "little.</p>",

        '<div class="panel howto">'
        "<h3>Two contradictions in the authors' release</h3>"
        "<p>The convenience API in <code>tdims_ext</code> is not the pipeline "
        "behind the paper, and the difference is not cosmetic. "
        "<code>experiments/run_nested_cv_experiment.py</code> adds a step that "
        "<em>zeroes</em> — rather than clamps — any feature value above a "
        "threshold, at 1.0 for <i>f</i><sub>dis</sub> = x<sup>−2</sup> and "
        "x<sup>−1</sup> and 1000.0 for <i>f</i><sub>dis</sub> = x. Without that "
        "special case the entire raw-distance third of the grid would be silently "
        "blanked, since raw distances all exceed 1. Applied correctly it removes "
        "3.3% of the nonzeros at radius 1, almost nothing at radius 2, and empties "
        "no molecule's vector.</p>"
        "<p>The LASSO selector's iteration cap differs too: scikit-learn's default "
        "in the experiment script, <code>max_iter=100000</code> in "
        "<code>tdims_ext</code>. For <i>f</i><sub>dis</sub> = x<sup>−2</sup> and "
        "x<sup>−1</sup> this is a no-op — the selector converges in 256 iterations "
        "and both settings pick the identical feature set, Jaccard 1.0000. For "
        "<i>f</i><sub>dis</sub> = x it is not: those features are standardised bond "
        "counts, strongly collinear, and coordinate descent crawls. At 1,000 "
        "iterations it stops unconverged with 233 features; at 100,000 it converges "
        "after 7,201 with 126; the two sets overlap at Jaccard 0.27. This page "
        "follows the experiment script, because that is the code behind the "
        "published numbers. It is not the cheap choice in disguise — on Biogen it "
        "scored <em>higher</em> on every endpoint.</p>"
        "</div>",

        "<h2>What this does and does not say</h2>",
        "<p>It does not say TDiMS is a bad descriptor. It says the thing it was "
        "built to capture is not what these endpoints are made of. TDiMS was "
        "designed for, and validated on, properties where the distance between two "
        "substructures <em>is</em> the physics: absorption maxima and Stokes shifts "
        "of chromophores, frontier orbital energies. Solubility, microsomal "
        "stability and permeability are not those properties. Its own paper reports "
        "the mirror image of this result, finding that TDiMS has difficulty "
        "providing useful features for molecules as small as QM9's — the descriptor "
        "has a range, and both ends of it are now mapped.</p>"
        "<p>It also does not say that hand-designed descriptors are finished. It "
        "says that on these fifteen endpoints, a representation learned from 81 "
        "million unlabelled molecules carries about 0.3 R² that a carefully "
        "reasoned enumeration of substructure pairs does not — and that the "
        "enumeration's own tuning knobs cannot close a tenth of it.</p>",

        f"<p class=\"note\">Reported the way {LINK_BLOG} and the protocol paper it "
        "points at"
        + ref("tukey")
        + " argue comparisons should be: distributions over folds, simultaneous "
        "confidence intervals, paired tests on the folds, and no bolded maxima. "
        "Where two methods cannot be separated, both are reported as tied.</p>",
        "<h2>References</h2>",
        page_kit.render_references(REFERENCES),
        "</div>",
    ]
    return "\n".join(parts)


def main() -> None:
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    html = build()
    PAGE.write_text(html)
    print(f"wrote {PAGE} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
