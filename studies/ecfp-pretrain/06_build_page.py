#!/usr/bin/env python
"""Step 6: the comparison as one self-contained page.

Reads the tables and figures produced by steps 2, 3 and 4 and writes a single
HTML document -- figures embedded as data URIs, no external requests -- covering
both data sets. Regenerate it whenever the underlying results change.

    for ds in expansion biogen; do
      ADME_DATASET=$ds python 02_collect_metrics.py
      ADME_DATASET=$ds python 03_report.py
      ADME_DATASET=$ds python 04_ptgin_selection.py
    done
    python 06_build_page.py
"""

import pandas as pd

import config as cfg
import page_kit

PAGE = cfg.PROJECT_DIR / "results" / "report.html"

MAIN = "expansion"
OTHER = "biogen"

METHOD_COLOR = {
    "lgbm": "#4C72B0",
    "chemprop_st": "#C44E52",
    "chemprop": "#DD8452",
    "chemeleon": "#55A868",
    "ptgin": "#8172B3",
}
SHORT = {
    "lgbm": "LightGBM",
    "chemprop_st": "ChemProp ST",
    "chemprop": "ChemProp MT",
    "chemeleon": "CheMeleon",
    "ptgin": "PT-GIN",
}

METHODS = cfg.METHODS
PTGIN = cfg.PTGIN_METHOD
LGBM = cfg.LGBM_METHOD


# --- loading -------------------------------------------------------------
def load(ds: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = cfg.paths(ds)
    return (
        pd.read_csv(paths.tables / "summary.csv"),
        pd.read_csv(paths.tables / "head_to_head.csv"),
        pd.read_csv(paths.fold_metrics),
    )


def selection(ds: str) -> pd.DataFrame:
    return pd.read_csv(cfg.paths(ds).tables / "selection_summary.csv")


def figure(name: str, caption: str, ds: str = MAIN) -> str:
    path = cfg.paths(ds).figures / f"{name}.png"
    return page_kit.figure_block(page_kit.embed_figure(path), caption, name)


# --- tables --------------------------------------------------------------
def top_sizes(summary: pd.DataFrame) -> pd.Series:
    """How many methods share the top of each endpoint x metric combination.

    `tukey_groups` calls the method with the leading mean 'best' and anything it
    cannot separate from that method 'equivalent'. The asymmetry is an artefact
    of how the label is computed, not a finding: if the correction cannot tell
    two methods apart, a table has no business crowning one of them. So both
    count as top here, and a row says 'tied' whenever more than one is up there.
    """
    top = summary[summary["tukey_group"].isin(("best", "equivalent"))]
    return top.groupby(["endpoint", "metric"]).size()


def metric_table(summary: pd.DataFrame, metric: str, ds: str = MAIN) -> str:
    """Mean +/- sd per endpoint and method, with the Tukey grouping as a chip."""
    sizes = top_sizes(summary)
    sub = summary[summary["metric"] == metric]
    methods = [m for m in METHODS if m in set(sub["method"])]
    rows = []
    for endpoint in cfg.paths(ds).dataset.targets:
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


def h2h_table(h2h: pd.DataFrame, left: str, right: str, metric: str = "r2",
              ds: str = MAIN) -> str:
    """One paired comparison: mean difference, folds won, p value, verdict."""
    sub = h2h[(h2h["left"] == left) & (h2h["right"] == right) & (h2h["metric"] == metric)]
    if sub.empty:
        return '<p class="missing">comparison not available</p>'
    rows = []
    for endpoint in cfg.paths(ds).dataset.targets:
        r = sub[sub["endpoint"] == endpoint]
        if r.empty:
            continue
        r = r.iloc[0]
        sig = r["p_value"] < 0.05
        better = r["mean_diff"] > 0
        verdict = ("wins" if better else "loses") if sig else "no call"
        cls = ("win" if better else "loss") if sig else "null"
        rows.append(
            f'<tr><th scope="row">{endpoint}</th>'
            f'<td class="num">{r["mean_diff"]:+.3f}</td>'
            f'<td class="num">{int(r["right_wins"])}/{int(r["n_folds"])}</td>'
            f'<td class="num">{r["p_value"]:.1e}</td>'
            f'<td><span class="chip {cls}">{verdict}</span></td></tr>'
        )
    return (
        '<div class="bleed tablewrap"><table>'
        '<thead><tr><th scope="col">Endpoint</th><th scope="col">Δ R²</th>'
        '<th scope="col">Folds won</th><th scope="col">p</th>'
        '<th scope="col">Verdict</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def tally_counts(summary: pd.DataFrame) -> pd.DataFrame:
    """Best alone / tied for best / worse, per method, over endpoint x metric."""
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


def tukey_tally(summary: pd.DataFrame) -> str:
    counts = tally_counts(summary)
    cards = []
    for m in METHODS:
        row = counts.loc[m]
        cards.append(
            f'<div class="tally">'
            f'<span class="swatch big" style="background:{METHOD_COLOR[m]}"></span>'
            f'<span class="tallyname">{cfg.METHOD_LABELS[m]}</span>'
            f'<span class="tallynums">'
            f'<b>{int(row["alone"])}</b> best alone · '
            f'<b>{int(row["tied"])}</b> tied for best · '
            f'<b>{int(row["worse"])}</b> worse</span></div>'
        )
    return '<div class="tallies">' + "".join(cards) + "</div>"


def selection_table(sel: pd.DataFrame, ds: str = MAIN) -> str:
    """What was chosen per endpoint, how far ahead it was, and how far apart the grid is.

    `margin` against `fold sd` is the column pair that matters. A margin well
    inside one fold's worth of noise is not a decision the data supports, however
    confidently the procedure made it.
    """
    rows = []
    indexed = sel.set_index("endpoint")
    for endpoint in cfg.paths(ds).dataset.targets:
        if endpoint not in indexed.index:
            continue
        r = indexed.loc[endpoint]
        decisive = r["margin_over_runner_up"] > r["fold_sd"]
        rows.append(
            f'<tr><th scope="row">{endpoint}</th>'
            f'<td class="num">{int(r["radius"])}</td>'
            f'<td class="num">{int(r["vocab"]):,}</td>'
            f'<td class="num">{r["val_r2"]:.3f}</td>'
            f'<td class="num">{r["margin_over_runner_up"]:.3f}</td>'
            f'<td class="num">{r["spread_best_to_worst"]:.3f}</td>'
            f'<td class="num">{r["fold_sd"]:.3f}</td>'
            f'<td><span class="chip {"win" if decisive else "null"}">'
            f'{"clear" if decisive else "within noise"}</span></td></tr>'
        )
    return (
        '<div class="bleed tablewrap"><table>'
        '<thead><tr><th scope="col">Endpoint</th><th scope="col">Radius</th>'
        '<th scope="col">Vocab</th><th scope="col">Val R²</th>'
        '<th scope="col">Margin</th><th scope="col">Grid spread</th>'
        '<th scope="col">Fold sd</th><th scope="col">Decisive?</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def fold_models(metrics: pd.DataFrame, ds: str) -> int:
    """One fold model per (method, unit, repeat, fold).

    A unit is an endpoint for the single-task methods and an assay family for the
    two multitask ChemProp arms, which predict a whole family in one pass.
    """
    group_of = cfg.paths(ds).dataset.group_of
    multitask = {"chemprop", "chemeleon"}

    def unit(row):
        return group_of[row["endpoint"]] if row["method"] in multitask else row["endpoint"]

    sub = metrics[metrics["method"].isin(METHODS)]
    return sub.assign(unit=sub.apply(unit, axis=1)).groupby(
        ["method", "unit", "repeat", "fold"]
    ).ngroups


def selection_fits(ds: str) -> int:
    """How many LightGBM models the checkpoint sweep fit, counting the discarded ones."""
    targets = len(cfg.paths(ds).dataset.targets)
    return targets * len(cfg.CHECKPOINTS) * cfg.N_REPEATS * cfg.N_SPLITS


# --- references ----------------------------------------------------------
REFERENCES = [
    (
        "ptgin",
        "Money-Kyrle, S.; Dablander, M.; Hanser, T.; Werner, S.; Deane, C. M.; Morris, G. M. ",
        "On Improving Graph Neural Networks for QSAR by Pre-training on "
        "Extended-Connectivity Fingerprints.",
        '<span class="venue">arXiv</span> 2026, 2605.10722. The method benchmarked '
        "here, run from the authors' own code and released checkpoints.",
        "https://arxiv.org/abs/2605.10722",
        "arXiv:2605.10722",
    ),
    (
        "sns",
        "Dablander, M.; Hanser, T.; Lambiotte, R.; Morris, G. M. ",
        "Sort &amp; Slice: A Simple and Superior Alternative to Hash-Based Folding "
        "for Extended-Connectivity Fingerprints.",
        '<span class="venue">J. Cheminform.</span> 2024, 16, 135. How PT-GIN builds '
        "its atom-token vocabulary.",
        "https://doi.org/10.1186/s13321-024-00932-y",
        "10.1186/s13321-024-00932-y",
    ),
    (
        "gin",
        "Xu, K.; Hu, W.; Leskovec, J.; Jegelka, S. ",
        "How Powerful are Graph Neural Networks?",
        '<span class="venue">ICLR</span> 2019. The architecture.',
        "https://openreview.net/forum?id=ryGs6iA5Km",
        "ICLR 2019",
    ),
    (
        "qmugs",
        "Isert, C.; Atz, K.; Jiménez-Luna, J.; Schneider, G. ",
        "QMugs, Quantum Mechanical Properties of Drug-like Molecules.",
        '<span class="venue">Sci. Data</span> 2022, 9, 273. The pre-training corpus, '
        "665k molecules before the similarity filter and 462,189 after it.",
        "https://doi.org/10.1038/s41597-022-01390-7",
        "10.1038/s41597-022-01390-7",
    ),
    (
        "ecfp",
        "Rogers, D.; Hahn, M. ",
        "Extended-Connectivity Fingerprints.",
        '<span class="venue">J. Chem. Inf. Model.</span> 2010, 50 (5), 742–754. '
        "The pre-training target, and the baseline it is put against.",
        "https://doi.org/10.1021/ci100050t",
        "10.1021/ci100050t",
    ),
    (
        "lightgbm",
        "Ke, G.; Meng, Q.; Finley, T.; <i>et al.</i> ",
        "LightGBM: A Highly Efficient Gradient Boosting Decision Tree.",
        '<span class="venue">NeurIPS</span> 2017. The downstream predictor for both '
        "PT-GIN and the fingerprint baseline.",
        "https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree",
        "NeurIPS 2017",
    ),
    (
        "chemprop",
        "Graff, D. E.; Morgan, N. K.; Burns, J. W.; <i>et al.</i> ",
        "Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical "
        "Property Prediction.",
        '<span class="venue">J. Chem. Inf. Model.</span> 2026, 66 (1), 28–33.',
        "https://doi.org/10.1021/acs.jcim.5c02332",
        "10.1021/acs.jcim.5c02332",
    ),
    (
        "chemeleon",
        "Burns, J. W.; Zalte, A. S.; Abreu, C. R. A.; <i>et al.</i> ",
        "Deep Learning Foundation Models for Low-Data Regimes from Classical "
        "Molecular Descriptors.",
        '<span class="venue">J. Chem. Inf. Model.</span> 2026, articles ASAP.',
        "https://doi.org/10.1021/acs.jcim.6c01546",
        "10.1021/acs.jcim.6c01546",
    ),
    (
        "biogen",
        "Fang, C.; Wang, Y.; Grater, R.; <i>et al.</i> ",
        "Prospective Validation of Machine Learning Algorithms for Absorption, "
        "Distribution, Metabolism, and Excretion Prediction.",
        '<span class="venue">J. Chem. Inf. Model.</span> 2023, 63 (11), 3263–3274. '
        "The second data set, and the paper's own primary benchmark.",
        "https://doi.org/10.1021/acs.jcim.3c00160",
        "10.1021/acs.jcim.3c00160",
    ),
    (
        "tukey",
        "Tukey, J. W. ",
        "Comparing Individual Means in the Analysis of Variance.",
        '<span class="venue">Biometrics</span> 1949, 5 (2), 99–114.',
        "https://doi.org/10.2307/3001913",
        "10.2307/3001913",
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

# Prose that depends on the numbers is written after the run and pasted in here,
# so a claim can never drift from the table above it without someone noticing.
PROSE_LEDE = (
    "PT-GIN"
    + ref("ptgin")
    + " is a graph network pre-trained to reproduce a molecule\u2019s own Morgan "
    "fingerprint, then frozen and handed to LightGBM. Its authors report it "
    "beating hashed ECFP on five of six Biogen ADME tasks. Run here against those "
    "same fingerprints through that same predictor \u2014 fifteen endpoints, two data "
    "sets, 25 folds each \u2014 it wins six endpoints and loses six, and on Biogen, its "
    "own benchmark, it wins one and loses three. Which six it wins is not random, "
    "and it is not about how much data an endpoint has. PT-GIN improves on the "
    "fingerprints precisely where the fingerprints were already doing badly, and "
    "degrades them where they were doing well."
)

PROSE_SELECTION = (
    "<p>The first thing the sweep says is that the grid barely exists. Across all "
    "nine ExpansionRx endpoints the ten checkpoints span a median of "
    "<b>0.027 R\u00b2</b>, and the winner beats the runner-up by a median of "
    "<b>0.002</b>. Set that against the fold-to-fold standard deviation of the "
    "winner itself \u2014 0.037 to 0.287 \u2014 and the comparison is not close: "
    "<b>not one of the fifteen endpoints across both data sets has a margin as "
    "large as a single fold\u2019s worth of noise.</b></p>"
    "<p>Five different checkpoints get chosen here and four on Biogen, which "
    "looks like the per-task variation the method is built around. It is not. It "
    "is what picking the maximum of ten near-identical numbers looks like. A "
    "radius-2 network with a 1,024-token vocabulary takes five of the nine "
    "ExpansionRx endpoints; a radius-0 network \u2014 one token per atom type, no "
    "circular environment at all \u2014 takes LogD here and three of the six Biogen "
    "endpoints. That a tokenisation carrying no substructure information beyond "
    "the atom finishes first as often as it does is the clearest evidence that "
    "the choice is not carrying signal.</p>"
    "<p>The one endpoint with a real spread, LOG_MGMB at 0.184, is the smallest "
    "in either data set at 431 measurements, and its fold standard deviation is "
    "0.287. That is not a grid with structure in it. That is a small endpoint "
    "being noisy in ten directions at once.</p>"
)

PROSE_EXPANSION = (
    "<p>PT-GIN is on top of six of the 27 combinations and best alone on none of "
    "them. CheMeleon is on top of 24, and LightGBM on none. But the tally "
    "flattens the only part of this that is interesting, because PT-GIN\u2019s six "
    "are not scattered at random across the endpoints.</p>"
)

PROSE_H2H = (
    "<p>Not a sweep in either direction. PT-GIN takes five endpoints, loses "
    "three, and cannot be separated on one. The paper\u2019s claim, that pre-training "
    "on fingerprints beats the fingerprints, half survives.</p>"
    "<p>Which half is the result, and the obvious explanation is the wrong one. "
    "It is <em>not</em> about how much data an endpoint carries: across all "
    "fifteen endpoints on both data sets, the PT-GIN advantage has a rank "
    "correlation of <b>\u22120.01</b> with the number of measurements "
    "(p = 0.98). LOG_MLM has 5,692 measurements and PT-GIN wins 23 folds of 25; "
    "LOG_MBPB has 1,426 and it loses.</p>"
    "<p>What predicts it is <b>how well the fingerprints were already doing</b>. "
    "Sort the fifteen endpoints by the LightGBM baseline\u2019s R\u00b2 and the advantage "
    "runs downhill: Spearman \u03c1 = <b>\u22120.65</b> (p = 0.009) over both data sets, "
    "and \u22120.73 (p = 0.025) on ExpansionRx alone. The crossover sits near a "
    "baseline R\u00b2 of 0.3. Caco-2 efflux ratio, where LightGBM scores \u22120.135, gains "
    "0.136 and PT-GIN wins all 25 folds. LogD, where LightGBM reaches 0.510, "
    "loses 0.091 and PT-GIN wins <b>none of 25</b>. LogS, the next strongest, "
    "also none of 25.</p>"
    "<p>That is what a smoothed fingerprint looks like. The embedding was trained "
    "to reconstruct ECFP4 and cannot contain substructure information ECFP4 does "
    "not already have; what it has instead is 2,048 to 3,072 dense, correlated "
    "coordinates in place of 2,048 sparse counts. Where the signal is weak enough "
    "that variance dominates, that is the better-behaved input and it wins. Where "
    "the counts are already carrying a learnable signal, the smoothing is "
    "spending resolution the model needed. The one clear exception is LOG_MGMB, "
    "which gains 0.162 from a baseline of 0.260 \u2014 and which is the smallest "
    "endpoint in either data set, at 431 measurements, with a fold standard "
    "deviation of 0.29.</p>"
)

PROSE_SCRATCH = (
    "<p>Here the pre-training does earn its keep, on six endpoints against two, "
    "and by margins that are not small: +0.405 R\u00b2 on Caco-2 A\u2192B, where a "
    "from-scratch D-MPNN scores \u22120.307 and does worse than predicting the "
    "training mean.</p>"
    "<p>The same rule fits. The two endpoints PT-GIN loses are LogD and LogS, 0 "
    "folds out of 25 apiece \u2014 and those are exactly the two endpoints where a "
    "from-scratch D-MPNN is <em>strongest</em>, at R\u00b2 0.700 and 0.447. A "
    "different architecture, a different comparator, and the boundary lands in "
    "the same place: PT-GIN pulls a weak model up and drags a strong one "
    "down.</p>"
)

PROSE_BIOGEN = (
    "<p>It does not replicate, and this is the data set the paper reports winning "
    "five of six tasks on. Against LightGBM, PT-GIN wins one endpoint, loses "
    "three, and cannot be separated on two. Against CheMeleon it loses all six "
    "endpoints on R\u00b2 and all six on Spearman, and five of six on MAE. The single "
    "combination of 18 where it is on top, it is on top of by being "
    "indistinguishable rather than better.</p>"
    "<p>The rule from ExpansionRx transfers, though less sharply, and the "
    "clearest instance is against a from-scratch D-MPNN rather than against the "
    "fingerprints. The two plasma protein binding endpoints have 128 and 109 "
    "training molecules, and single-task ChemProp scores R\u00b2 of \u22120.064 and "
    "\u22120.005 on them \u2014 worse than predicting the training mean. They are the only "
    "two of the six where PT-GIN beats it, by +0.224 and +0.072, winning 25 folds "
    "of 25 on human PPB. On the four endpoints where that method actually works, "
    "PT-GIN loses to it every time.</p>"
    "<p>One explanation can be ruled out. These checkpoints were pre-trained on "
    "QMugs filtered at Tanimoto 0.5 against Biogen, so a Biogen-like molecule was "
    "excluded by construction \u2014 but ExpansionRx was never one of the paper\u2019s "
    "benchmarks and got no such filter, which would let the ExpansionRx result be "
    "the flattered one. It is not. Neither test set has a single exact or "
    "connectivity-block InChIKey match in the 462,189-molecule corpus, and the "
    "median nearest-neighbour Tanimoto is 0.372 on ExpansionRx against 0.431 on "
    "Biogen. The unfiltered data set is the <em>less</em> overlapping of the "
    "two.</p>"
)

PROSE_CLOSE = (
    "<p>Two things are worth carrying away, and only one of them is about "
    "PT-GIN.</p>"
    "<p>The first is that self-supervision on a fingerprint gives you a "
    "fingerprint. Every result above is consistent with the embedding being a "
    "smoothed, denser ECFP4 and nothing more: its advantage over the sparse "
    "counts is predicted by how badly those counts were doing and by nothing "
    "else, it rescues a graph network wherever that network is starved, it drags "
    "one down wherever it is not, and it never once beats a graph network "
    "initialised from a foundation model that was pre-trained on something other "
    "than a fingerprint. The ceiling of the pre-training target is the ceiling of "
    "the representation, which is a general point about choosing a "
    "self-supervised objective rather than a criticism of this one.</p>"
    "<p>The second is a warning about validation splits that this arm gives for "
    "free. On eight of the nine ExpansionRx endpoints PT-GIN\u2019s validation R\u00b2 "
    "runs 0.11 to 0.44 above the test R\u00b2 it then achieves \u2014 0.720 against 0.420 "
    "on LogD. On Biogen\u2019s four well-populated endpoints the same gap is "
    "\u22120.01 to +0.03. Nothing here is selecting on validation in a way that could "
    "inflate it: the number is one fixed model\u2019s honest score on a held-out "
    "fifth. The gap is a property of the ExpansionRx split, which shipped with "
    "the challenge and is not cluster-pure, against the Biogen split, which was "
    "built by holding out whole BitBIRCH clusters exactly as the folds are. Any "
    "method that tunes anything on the first of those is being told a comfortable "
    "lie.</p>"
)


def build() -> str:
    summary, h2h, metrics = load(MAIN)
    bio_summary, bio_h2h, bio_metrics = load(OTHER)
    sel, bio_sel = selection(MAIN), selection(OTHER)

    n_endpoints = sum(len(cfg.paths(d).dataset.targets) for d in (MAIN, OTHER))
    n_models = fold_models(metrics, MAIN) + fold_models(bio_metrics, OTHER)
    n_selection = selection_fits(MAIN) + selection_fits(OTHER)

    parts = [
        "<title>Pre-training on Fingerprints</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">5×5 cross validation · two data sets</p>',
        "<h1>A graph network taught to predict fingerprints, against the "
        "fingerprints</h1>",
        f'<p class="lede">{PROSE_LEDE}</p>',
        '<div class="facts">'
        f'<div class="fact"><b>{len(METHODS)}</b><span>methods</span></div>'
        f'<div class="fact"><b>2</b><span>data sets</span></div>'
        f'<div class="fact"><b>{n_endpoints}</b><span>endpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b>'
        "<span>cross validation</span></div>"
        f'<div class="fact"><b>{n_models:,}</b><span>fold models</span></div>'
        f'<div class="fact"><b>{len(cfg.CHECKPOINTS)}</b>'
        "<span>released checkpoints</span></div>"
        "</div>",
        "<h2>What PT-GIN is</h2>",
        "<p>Every atom in a molecule is given one token per circular substructure "
        "radius, drawn from a vocabulary that Sort &amp; Slice"
        + ref("sns")
        + " builds by ranking substructures on how many pre-training molecules "
        "they appear in and keeping the most common. A Graph Isomorphism Network"
        + ref("gin")
        + " reads those tokens and is trained, on 462,189 QMugs molecules"
        + ref("qmugs")
        + ", to answer 2,048 yes-or-no questions about the molecule it is looking "
        "at: which bits of that molecule's own hashed ECFP4"
        + ref("ecfp")
        + " are set.</p>",
        '<div class="panel">'
        "<h3>The self-supervision is the point</h3>"
        "<p>The target is computable from the structure. There is no assay, no "
        "measurement and no label scarcity — pre-training data is anything with a "
        "SMILES string. What the network has to learn in order to reconstruct a "
        "fingerprint from a graph is, the argument goes, a representation that "
        "carries the same substructure information the fingerprint does, but in a "
        "form a downstream model can use more flexibly.</p>"
        "<p>Downstream the encoder is frozen. Each layer's output is graph-pooled, "
        "the per-layer vectors are concatenated, and LightGBM"
        + ref("lightgbm")
        + " predicts the endpoint. Nothing is fine-tuned: the authors report that "
        "end-to-end fine tuning cost more, helped negligibly, and sometimes "
        "hurt.</p>"
        "</div>",
        "<p>That last detail is what makes this method unusually easy to judge. "
        "<b>PT-GIN and the LightGBM baseline are the same pipeline with different "
        "inputs.</b> One is handed 2,048 Morgan counts, the other 2,048 to 3,072 "
        "numbers from a frozen network, and both hand them to LightGBM at library "
        "defaults. Whatever separates them is the representation, and nothing "
        "else.</p>",
        "<h2>Which checkpoint?</h2>",
        "<p>The paper does not have one PT-GIN. It pre-trains a grid of maximum "
        "substructure radius by vocabulary size, and for each task it picks "
        "whichever pre-trained model does best in downstream tuning. Ten of those "
        f"checkpoints are released, so the choice is part of the method and gets "
        "reproduced — on the validation fifth of each fold, which is the only "
        "place this protocol allows a choice to be made. Every checkpoint is fit "
        "on the four fifths of all 25 folds and scored on the held-out fifth; the "
        "best mean validation R² wins the endpoint. The test set plays no part in "
        f"it. That is {n_selection:,} LightGBM fits to choose "
        f"{n_endpoints} models.</p>",
        figure("ptgin_selection",
               "Mean validation R² of each of the ten released checkpoints on each "
               "ExpansionRx endpoint, with the chosen one boxed. Colour is distance "
               "from that endpoint's own mean divided by its fold-to-fold noise "
               "(printed beside each row), so a cell is only shaded when it is far "
               "from its row in units that matter. Nothing reaches half a "
               "standard deviation."),
        selection_table(sel),
        PROSE_SELECTION,
        "<h2>How often is each method best?</h2>",
        "<p>Tukey HSD"
        + ref("tukey")
        + " across the 25 folds, counted over every endpoint × metric combination. "
        "<em>Tied</em> means the method could not be distinguished from the leader "
        "at α = 0.05.</p>",
        '<div class="panel">' + tukey_tally(summary) + "</div>",
        PROSE_EXPANSION,
        figure("tukey_r2",
               "Tukey HSD on R², one panel per ExpansionRx endpoint. Bars are "
               "method means over 25 folds; whiskers are simultaneous confidence "
               "intervals covering every pairwise comparison at once."),
        "<h3>R² by endpoint</h3>",
        metric_table(summary, "r2"),
        "<h3>Spearman ρ by endpoint</h3>",
        metric_table(summary, "spearman"),
        "<h2>Against the fingerprints it was trained to predict</h2>",
        "<p>This is the paper's central claim under this protocol, and the "
        "cleanest test on the page: same predictor, same molecules, same folds, "
        "one representation swapped for another. Folds are paired.</p>",
        h2h_table(h2h, LGBM, PTGIN),
        PROSE_H2H,
        "<h2>Against a graph network with no pre-training</h2>",
        "<p>The other half of the claim. ChemProp single-task"
        + ref("chemprop")
        + " is a message-passing network trained from scratch on each endpoint — "
        "a different architecture from a GIN, but the same bet that a learned "
        "molecular representation beats a fixed one, without the pre-training.</p>",
        h2h_table(h2h, "chemprop_st", PTGIN),
        PROSE_SCRATCH,
        "<h2>Does it replicate?</h2>",
        "<p>Everything above is ExpansionRx. The Biogen public ADME set"
        + ref("biogen")
        + " is the paper's own primary benchmark, and the one it reports winning "
        "five of six tasks on. It is not the same experiment: the authors use 200 "
        "repeats of 5-fold Butina-clustered cross validation with no fixed "
        "holdout, where this project holds out whole BitBIRCH clusters once and "
        "cross-validates within what remains. Absolute numbers do not carry across "
        "the two designs. Rankings within each do.</p>"
        "<p>It also matters that these released checkpoints were filtered against "
        "this data set. Any QMugs molecule within Tanimoto 0.5 of a Biogen "
        "compound was excluded from pre-training. ExpansionRx was not one of the "
        "paper's benchmarks and got no such filter, so if anything the "
        "ExpansionRx result above is the one with more opportunity to be "
        "flattered.</p>",
        '<div class="panel">' + tukey_tally(bio_summary) + "</div>",
        figure("tukey_r2", "Tukey HSD on R² for the six Biogen endpoints.", OTHER),
        metric_table(bio_summary, "r2", OTHER),
        "<h3>What it chose here</h3>",
        figure("ptgin_selection",
               "The same ten checkpoints on the six Biogen endpoints, on the same "
               "fold-noise scale.", OTHER),
        selection_table(bio_sel, OTHER),
        "<h3>Against the fingerprints, again</h3>",
        h2h_table(bio_h2h, LGBM, PTGIN, ds=OTHER),
        PROSE_BIOGEN,
        "<h2>What this does and does not say</h2>",
        '<div class="panel howto">'
        "<h3>Two choices that are this project's, not the authors'</h3>"
        "<p><b>LightGBM runs at library defaults.</b> The paper Optuna-tunes it, "
        "50 trials per task — but it tunes it identically for every representation "
        "it compares, so the comparison it draws does not rest on the tuning. "
        "Leaving both arms untuned here preserves that, and keeps the only "
        "difference between the fingerprint baseline and PT-GIN the thing being "
        "studied. A tuned PT-GIN would score higher than the numbers above. So "
        "would a tuned LightGBM.</p>"
        "<p><b>The checkpoint is selected on the validation fifth</b> rather than "
        "on a dedicated tuning repeat, which is how the paper does it. This "
        "protocol has no spare repeat to give it. The selection table above says "
        "how much that choice was worth.</p>"
        "</div>",
        PROSE_CLOSE,
        f'<p class="footnote">Reported the way {LINK_BLOG} argues comparisons '
        "should be: distributions over folds, simultaneous confidence intervals, "
        "paired tests, and no bolded maxima. Every method saw identical training "
        "molecules in every fold.</p>",
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
