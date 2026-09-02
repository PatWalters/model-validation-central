#!/usr/bin/env python
"""Step 14: the Trimole-Hybrid comparison as one self-contained page.

The other report in this repository asks which pre-trained representation
transfers. This one asks a different question, which is why it is a separate
document rather than three more columns on the first: is choosing a model per
endpoint worth more than choosing a good architecture once?

Five methods, and only one of them is a foundation model in the usual sense. The
four baselines are the ones that make the question answerable -- a fingerprint
model, a single-task GNN, a multitask GNN and a foundation-initialised GNN -- and
the fifth, Trimole-Hybrid, is not a model at all but a procedure that fits sixty
candidates per fold and keeps whichever wins on the validation split.

Everything is inlined, figures included, so the file can be published on its own.

    ADME_COMPARISON=trimole python 05_report.py
    ADME_COMPARISON=trimole python 13_trimole_selection.py
    ADME_COMPARISON=trimole ADME_DATASET=biogen python 05_report.py
    ADME_COMPARISON=trimole ADME_DATASET=biogen python 13_trimole_selection.py
    python 14_build_trimole_page.py
"""

import pandas as pd

import config as cfg
import page_kit

PAGE = cfg.PROJECT_DIR / "results" / "trimole_report.html"

COMPARISON = "trimole"
MAIN = "expansion"
OTHER = "biogen"

METHOD_COLOR = page_kit.METHOD_COLOR
SHORT = page_kit.SHORT

# The methods this page covers, in the order the tables print them.
METHODS = cfg.COMPARISONS[COMPARISON]

VIEW_LABEL = {
    "chem": "chemistry only",
    "chemberta": "+ ChemBERTa",
    "kpgt": "+ KPGT",
    "unimol": "+ UniMol 3D",
    "fused": "+ all three",
}


# --- page pieces ---------------------------------------------------------
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


def selection(ds: str) -> pd.DataFrame:
    return pd.read_csv(cfg.paths(ds, COMPARISON).tables / "selection.csv")


def top_sizes(summary: pd.DataFrame) -> pd.Series:
    """How many methods share the top of each endpoint x metric combination."""
    top = summary[summary["tukey_group"].isin(("best", "equivalent"))]
    return top.groupby(["endpoint", "metric"]).size()


def metric_table(summary: pd.DataFrame, metric: str, ds: str = MAIN) -> str:
    """Mean +/- sd per endpoint and method, with the Tukey grouping as a chip.

    A method that cannot be separated from the leader is marked tied, not beaten,
    and when the top is shared nobody is crowned.
    """
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


def h2h_table(h2h: pd.DataFrame, left: str, right: str, metric: str = "r2",
              ds: str = MAIN) -> str:
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


def tukey_tally(summary: pd.DataFrame) -> str:
    sizes = top_sizes(summary)
    shared = summary.apply(
        lambda r: int(sizes.get((r["endpoint"], r["metric"]), 1)) > 1, axis=1
    )
    on_top = summary["tukey_group"].isin(("best", "equivalent"))
    kind = pd.Series("worse", index=summary.index)
    kind[on_top & shared] = "tied"
    kind[on_top & ~shared] = "alone"
    counts = pd.crosstab(summary["method"], kind)

    cards = []
    for m in [x for x in METHODS if x in counts.index]:
        row = counts.loc[m]
        cards.append(
            f'<div class="tally"><span class="swatch big" style="background:{METHOD_COLOR[m]}"></span>'
            f'<span class="tallyname">{cfg.METHOD_LABELS[m]}</span>'
            f'<span class="tallynums">'
            f'<b>{int(row.get("alone", 0))}</b> best alone · '
            f'<b>{int(row.get("tied", 0))}</b> tied for best · '
            f'<b>{int(row.get("worse", 0))}</b> worse</span></div>'
        )
    return '<div class="tallies">' + "".join(cards) + "</div>"


def tally_counts(summary: pd.DataFrame) -> pd.DataFrame:
    """The same numbers as a frame, so the prose can quote them without drifting."""
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


def selection_table(sel: pd.DataFrame, ds: str = MAIN) -> str:
    """The modal choice per endpoint, and how often it held."""
    rows = []
    for endpoint in cfg.paths(ds, COMPARISON).dataset.targets:
        g = sel[sel["endpoint"] == endpoint]
        if g.empty:
            continue
        cells = [f'<th scope="row">{endpoint}</th>']
        for axis, labels in (
            ("selected_view", VIEW_LABEL),
            ("selected_block", {}),
            ("selected_backend", {}),
        ):
            counts = g[axis].value_counts()
            level, n = counts.index[0], int(counts.iloc[0])
            shown = labels.get(level, level).replace("_", " ")
            cells.append(
                f'<td><span class="num">{shown}</span>'
                f'<span class="sd">{n}/{len(g)}</span></td>'
            )
        cells.append(f'<td class="num">{g["selected_val_r2"].mean():.3f}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="bleed tablewrap"><table>'
        '<thead><tr><th scope="col">Endpoint</th>'
        '<th scope="col">Molecular view</th><th scope="col">Chemistry block</th>'
        '<th scope="col">Backend</th><th scope="col">Mean val R²</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def gap_table(ds: str = MAIN) -> str:
    """What the winning candidate scored on validation, against what it then scored.

    The selection is only as good as the split it is made on. This is the table
    that says whether that split was telling the truth.
    """
    paths = cfg.paths(ds, COMPARISON)
    sel = pd.read_csv(paths.tables / "selection.csv")
    tri = pd.read_csv(paths.fold_metrics)
    tri = tri[tri["method"] == cfg.TRIMOLE_METHOD]
    joined = sel.merge(tri, on=["endpoint", "repeat", "fold"])

    rows = []
    for endpoint in paths.dataset.targets:
        g = joined[joined["endpoint"] == endpoint]
        if g.empty:
            continue
        val, test = g["selected_val_r2"].mean(), g["r2"].mean()
        gap = val - test
        cls = "loss" if gap > 0.15 else ("null" if gap > 0.08 else "win")
        rows.append(
            f'<tr><th scope="row">{endpoint}</th>'
            f'<td class="num">{int(g["n_val"].mean())}</td>'
            f'<td class="num">{val:.3f}</td>'
            f'<td class="num">{test:.3f}</td>'
            f'<td><span class="chip {cls}">{gap:+.3f}</span></td></tr>'
        )
    return (
        '<div class="bleed tablewrap"><table>'
        '<thead><tr><th scope="col">Endpoint</th><th scope="col">Val molecules</th>'
        '<th scope="col">Val R² of winner</th><th scope="col">Test R² it got</th>'
        '<th scope="col">Gap</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def fold_models(metrics: pd.DataFrame, ds: str) -> int:
    """One fold model per (method, unit, repeat, fold).

    A unit is an endpoint for the single-task methods and an assay family for the
    multitask ones. Trimole-Hybrid counts as single-task -- it selects and fits
    per endpoint -- though each of its fold models is itself the survivor of 60
    candidate fits, which the prose says rather than the arithmetic.
    """
    group_of = cfg.paths(ds, COMPARISON).dataset.group_of

    def unit(row):
        variant = cfg.VARIANTS.get(row["method"])
        if variant is not None and not variant["single_task"]:
            return group_of[row["endpoint"]]
        return row["endpoint"]

    sub = metrics[metrics["method"].isin(METHODS)]
    return sub.assign(unit=sub.apply(unit, axis=1)).groupby(
        ["method", "unit", "repeat", "fold"]
    ).ngroups


def candidate_fits(metrics: pd.DataFrame, ds: str) -> int:
    """How many models Trimole-Hybrid fit in total, counting the ones it discarded."""
    sub = metrics[metrics["method"] == cfg.TRIMOLE_METHOD]
    folds = sub.groupby(["endpoint", "repeat", "fold"]).ngroups
    return folds * 60


REFERENCES = [
    (
        "trimole",
        "Luo, Z.; Huang, D.; Shao, Y.; Yu, Q.; Li, Y. ",
        "A Multimodal Representation Learning Platform for Accurate Molecular "
        "ADMET Prediction.",
        '<span class="venue">Bioinformatics</span> 2026, in review. The method '
        "this page reimplements. Their release is an audit package rather than a "
        "runnable pipeline, which is why this is a reimplementation.",
        "https://doi.org/10.1101/2026.08.24.746660",
        "10.1101/2026.08.24.746660",
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
        "kpgt",
        "Li, H.; Zhao, D.; Zeng, J. ",
        "A Knowledge-Guided Pre-training Framework for Improving Molecular "
        "Representation Learning.",
        '<span class="venue">Nat. Commun.</span> 2023, 14, 7568. The graph branch '
        "of the candidate pool, run from the authors' pre-trained checkpoint.",
        "https://doi.org/10.1038/s41467-023-43214-1",
        "10.1038/s41467-023-43214-1",
    ),
    (
        "unimol",
        "Zhou, G.; Gao, Z.; Ding, Q.; <i>et al.</i> ",
        "Uni-Mol: A Universal 3D Molecular Representation Learning Framework.",
        '<span class="venue">ICLR</span> 2023. The 3D branch, via '
        "<code>unimol_tools</code>.",
        "https://openreview.net/forum?id=6K2RM6wVqKu",
        "ICLR 2023",
    ),
    (
        "chemberta",
        "Chithrananda, S.; Grand, G.; Ramsundar, B. ",
        "ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property "
        "Prediction.",
        '<span class="venue">arXiv</span> 2020, 2010.09885. The sequence branch, '
        "checkpoint <code>seyonec/ChemBERTa-zinc-base-v1</code>.",
        "https://arxiv.org/abs/2010.09885",
        "arXiv:2010.09885",
    ),
    (
        "maplight",
        "Notwell, J. H.; Wood, M. W. ",
        "ADMET Property Prediction through Combinations of Molecular Fingerprints.",
        '<span class="venue">arXiv</span> 2023, 2310.00174. The chemistry-prior '
        "feature set the candidate pool's sidecars are built from.",
        "https://arxiv.org/abs/2310.00174",
        "arXiv:2310.00174",
    ),
    (
        "biogen",
        "Fang, C.; Wang, Y.; Grater, R.; <i>et al.</i> ",
        "Prospective Validation of Machine Learning Algorithms for Absorption, "
        "Distribution, Metabolism, and Excretion Prediction.",
        '<span class="venue">J. Chem. Inf. Model.</span> 2023, 63 (11), 3263–3274. '
        "The second data set.",
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
PLACEHOLDER_EXPANSION_PROSE = (
    "<p>The answer to the question in the headline, on this data set, is no. "
    "CheMeleon — one fixed architecture, chosen once and applied to every "
    "endpoint — is on top of 25 of the 27 combinations. Trimole-Hybrid, having "
    "fitted sixty candidates per fold and kept the best of them on evidence it "
    "was entitled to use, is on top of 11, and beaten outright on 16.</p>"
    "<p>It is not last. It comfortably beats LightGBM, which is on top of "
    "nothing, and it beats both from-scratch ChemProp variants on the tally. "
    "The selection is buying something. It is just buying less than a "
    "well-initialised message-passing network costs.</p>"
)

PLACEHOLDER_SELECTION_PROSE = (
    "<p>The choice varies by endpoint and holds within one, which is the pattern "
    "the method needs: the modal view accounts for 75% of an endpoint's folds. "
    "Both microsomal stability endpoints take the chemistry priors alone on 24 "
    "of 25 folds; LogD takes KPGT on 20 of 25. So the pool is doing real work "
    "rather than dressing up a fixed answer.</p>"
    "<p>But look at what it declines. Across all 225 folds the classical "
    "chemistry priors win alone 140 times, KPGT 66, and the other two encoders "
    "together 11. Random forest, one of the four backends, wins twice. Most of "
    "the candidate pool exists to be rejected — which is the paper's own "
    "ablation finding, arrived at from the other direction.</p>"
    "<p>One fold is worth singling out. LOG_MGMB r1 f0 has six molecules in its "
    "validation split, and the candidate that won them scored an R² of −0.090. "
    "On the smallest endpoint the selection signal runs out, and what is left "
    "is a coin toss between sixty models.</p>"
)

PLACEHOLDER_H2H_PROSE = (
    "<p>CheMeleon wins seven of the nine endpoints outright, loses one — "
    "LOG_MLM — and draws one. The losses are not close: on LogD, LogS, LOG_MPPB, LOG_MBPB and "
    "LOG_MGMB the selection procedure fails to take a single fold out of 25.</p>"
    "<p>The reason is structural, and it is visible in the R² table above. On "
    "<b>LogD</b>, the endpoint with the most training data, single-task ChemProp "
    "reaches 0.700 and Trimole-Hybrid reaches 0.399 — worse than plain LightGBM "
    "on Morgan fingerprints, at 0.510. Every one of the sixty candidates is a "
    "fixed representation: a frozen encoder or a block of classical descriptors, "
    "handed to a tree ensemble or a ridge. Nothing in the pool <i>learns</i> a "
    "representation from the endpoint. Where there is enough data to learn one, "
    "a network that does so wins, and no amount of choosing between fixed "
    "alternatives closes the gap.</p>"
    "<p>Where Trimole-Hybrid does win — LOG_MLM, and on the draw at Caco-2 A→B — "
    "is where every method is struggling and no representation is much good.</p>"
)

PLACEHOLDER_BIOGEN_PROSE = (
    "<p>The ranking replicates: CheMeleon is on top of 14 of the 18 "
    "combinations, Trimole-Hybrid 6, and the two from-scratch ChemProp variants "
    "2 each. What does not replicate is what the method chose. On ExpansionRx "
    "the chemistry priors alone won 62% of folds and KPGT 29%; here KPGT wins "
    "most often and the priors drop to a third, while the modal choice within "
    "an endpoint falls from 75% of folds to 53%. The same procedure, given "
    "different chemistry, reaches for different evidence — and is less sure "
    "of itself.</p>"
    "<p>Head to head it takes LOG_RPPB from CheMeleon on 19 of 25 paired folds "
    "and draws on LOG_HPPB, its only two results against CheMeleon here that are "
    "not defeats. These are the two smallest endpoints in either data set, 109 "
    "and 128 training molecules, and they are also precisely where CheMeleon is "
    "weakest. "
    "The selection procedure is not winning because it is good at small data; "
    "it is winning where the strongest fixed architecture happens to fail.</p>"
)


def build() -> str:
    summary, h2h, metrics = load(MAIN)
    bio_summary, bio_h2h, bio_metrics = load(OTHER)
    sel, bio_sel = selection(MAIN), selection(OTHER)

    tally = tally_counts(summary)
    bio_tally = tally_counts(bio_summary)

    n_endpoints = sum(len(cfg.paths(d, COMPARISON).dataset.targets) for d in (MAIN, OTHER))
    n_models = fold_models(metrics, MAIN) + fold_models(bio_metrics, OTHER)
    n_candidates = candidate_fits(metrics, MAIN) + candidate_fits(bio_metrics, OTHER)

    tri = cfg.TRIMOLE_METHOD
    tri_top = int(tally.loc[tri, "alone"] + tally.loc[tri, "tied"])
    bio_tri_top = int(bio_tally.loc[tri, "alone"] + bio_tally.loc[tri, "tied"])
    n_combos = len(summary) // len(METHODS)
    bio_combos = len(bio_summary) // len(METHODS)

    parts = [
        "<title>Sixty models per fold</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">5×5 cross validation · two data sets</p>',
        "<h1>Is picking a model per endpoint worth it?</h1>",
        '<p class="lede">Trimole-Hybrid'
        + ref("trimole")
        + " argues that no single molecular representation suits every ADMET "
        "endpoint, so it builds a pool of candidate predictors, fits all of them, "
        "and keeps whichever wins on a validation split. Here it is put against "
        "four fixed architectures on fifteen endpoints across two unrelated data "
        f"sets — {n_candidates:,} candidate fits to produce "
        f"{n_models:,} fold models, every one scored on molecules it never saw. "
        "It loses to a single well-initialised graph network on both, for two "
        "reasons that only show up when you look at what it picked: nothing in "
        "its pool learns a representation, and on one of the two data sets the "
        "split it selects on does not resemble the split it is scored on.</p>",
        '<div class="facts">'
        f'<div class="fact"><b>{len(METHODS)}</b><span>methods</span></div>'
        f'<div class="fact"><b>2</b><span>data sets</span></div>'
        f'<div class="fact"><b>{n_endpoints}</b><span>endpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b><span>cross validation</span></div>'
        f'<div class="fact"><b>60</b><span>candidates per fold</span></div>'
        f'<div class="fact"><b>{n_candidates/1000:.0f}k</b><span>candidate fits</span></div>'
        "</div>",
        '<div class="panel howto">'
        "<h3>This is a reimplementation, and that matters</h3>"
        "<p>The other pre-trained methods benchmarked in this project run their "
        "authors' own code against their authors' own checkpoints. Trimole-Hybrid "
        "cannot: its public release describes itself as “not a one-command full "
        "rerun bundle”, ships filesystem paths as placeholders, includes no "
        "trained weights or cached embeddings, is wired throughout to the TDC "
        "benchmark's directory layout, and reserves all rights pending a licence "
        "decision.</p>"
        "<p>What follows was therefore written from the paper and from that "
        "source read as a specification, using checkpoints obtained independently "
        "from their original authors"
        + ref("kpgt", "unimol", "chemberta")
        + ". It reproduces the method — a candidate pool over four molecular "
        "views, and selection on validation only — not the paper's numbers. Where "
        "a result here disagrees with the published benchmark, this "
        "reimplementation is the more likely explanation.</p>"
        "</div>",
        "<h2>What the method actually is</h2>",
        "<p>Every other method on this page is an architecture. You hand it "
        "molecules, it trains, and the model you get is the model the architecture "
        "implies. Trimole-Hybrid is a procedure instead. For each endpoint and each "
        "fold it assembles sixty candidate predictors from three ingredients, fits "
        "every one on the training molecules, scores them all on the held-out "
        "validation fifth, and keeps exactly one.</p>",
        '<div class="panel">'
        "<h3>The candidate pool, per fold</h3>"
        "<p><b>Five molecular views</b> — the classical chemistry priors alone "
        "(Morgan counts, feature-Morgan, MACCS, Avalon, ErG, atom pairs, "
        "topological torsions and the RDKit descriptor block"
        + ref("maplight")
        + "), then those priors joined by a ChemBERTa sequence embedding"
        + ref("chemberta")
        + ", a KPGT graph embedding"
        + ref("kpgt")
        + ", a UniMol 3D embedding"
        + ref("unimol")
        + ", or all three at once.</p>"
        "<p><b>Three chemistry blocks</b> — different subsets of those priors, "
        "since endpoints differ in how much of them they want.</p>"
        "<p><b>Four backends</b> — XGBoost, extremely randomised trees, a random "
        "forest and ridge regression.</p>"
        "<p>Five times three times four is sixty. The winner predicts the test "
        "set; the other fifty-nine are discarded.</p>"
        "</div>",
        "<p>The protocol this project already uses fits the method without "
        "modification, because every fold here already carries a validation split. "
        "Each candidate trains on the same four fifths of the training molecules "
        "that every other method trains on, is scored on the held-out fifth, and "
        "the winner is applied to the fixed test set. Nothing is refit on "
        "train-plus-validation afterwards, which would have handed this arm more "
        "training data than the other four and quietly broken the comparison.</p>",
        "<h2>How often is each method best?</h2>",
        "<p>Tukey HSD"
        + ref("tukey")
        + " across the 25 folds, counted over every endpoint × metric "
        "combination. <em>Tied</em> means the method could not be distinguished "
        "from the leader at α = 0.05.</p>",
        '<div class="panel">' + tukey_tally(summary) + "</div>",
        PLACEHOLDER_EXPANSION_PROSE,
        figure("tukey_r2", "Tukey HSD on R², one panel per ExpansionRx endpoint. "
               "Bars are method means over 25 folds; whiskers are simultaneous "
               "confidence intervals covering every pairwise comparison at once."),
        "<h3>R² by endpoint</h3>",
        metric_table(summary, "r2"),
        "<h3>MAE by endpoint</h3>",
        metric_table(summary, "mae"),
        "<h2>What did it choose?</h2>",
        "<p>This is the part no other method has. Because the selection happens "
        "before the test set is touched, what it picked is a result rather than a "
        "diagnostic — and it is the result that can falsify the method's premise. "
        "If the choice never varies between endpoints, the pool is decoration. If "
        "it never repeats within an endpoint, the choice is noise.</p>",
        figure("trimole_selection",
               "Which candidate won, over the 25 folds of each ExpansionRx "
               "endpoint, broken out by the three axes of the pool."),
        selection_table(sel),
        PLACEHOLDER_SELECTION_PROSE,
        "<h2>Against the best fixed architecture</h2>",
        "<p>Folds are paired: both methods saw the same molecules in the same "
        "split, so the comparison is within-fold rather than between averages.</p>",
        h2h_table(h2h, "chemeleon", tri),
        PLACEHOLDER_H2H_PROSE,
        "<h2>Does it replicate?</h2>",
        "<p>Everything above is one data set. The Biogen public ADME set"
        + ref("biogen")
        + " is 3,521 commercially sourced compounds on six endpoints, far more "
        "chemically diverse, and split by holding out whole clusters. Same "
        "protocol, same pool, nothing retuned.</p>",
        '<div class="panel">' + tukey_tally(bio_summary) + "</div>",
        figure("tukey_r2", "Tukey HSD on R² for the six Biogen endpoints.", OTHER),
        metric_table(bio_summary, "r2", OTHER),
        figure("trimole_selection",
               "What the method selected on the Biogen endpoints.", OTHER),
        selection_table(bio_sel, OTHER),
        PLACEHOLDER_BIOGEN_PROSE,
        "<h2>Why selecting on validation was the wrong bet</h2>",
        "<p>Putting the two data sets side by side explains the whole result. A "
        "procedure that picks its model on a validation split is making one "
        "assumption: that doing well on that split predicts doing well on the "
        "test set. That assumption is checkable, because every selection record "
        "stores the score the winning candidate earned on validation, and the "
        "fold metrics store what it then achieved.</p>",
        "<h3>ExpansionRx</h3>",
        gap_table(MAIN),
        "<h3>Biogen</h3>",
        gap_table(OTHER),
        "<p>On ExpansionRx the winning candidate is flattered by between 0.16 "
        "and 0.47 R². It looks like a model that explains 82% of the variance in "
        "LogD and delivers 40%. On Biogen's four well-populated endpoints the "
        "same gap is 0.03 to 0.05 — the validation split is telling very nearly "
        "the truth. The two endpoints where Biogen's gap opens up, to 0.23 and "
        "0.25, are the two with about two dozen validation molecules, which is "
        "small-sample noise rather than a split that misleads.</p>",
        "<p>The difference is in how the test sets were made. Biogen's holdout "
        "was built here, by holding out whole BitBIRCH clusters — the same rule "
        "that builds the cross-validation folds, so validation and test are the "
        "same kind of problem and equally hard. The ExpansionRx split arrived "
        "with the challenge and was drawn some other way, so its held-out fifth "
        "and its test set are not the same kind of problem at all.</p>",
        "<p>That is the risk a selection framework carries and a fixed "
        "architecture does not. CheMeleon does not care whether validation "
        "resembles the test set, because it never consults validation to decide "
        "what it is. Trimole-Hybrid stakes its entire output on that "
        "resemblance, sixty candidates deep. When it holds, the method is "
        "competitive. When it does not, the method optimises confidently for "
        "the wrong thing — and nothing in the procedure can tell it so.</p>",
        '<div class="panel howto">'
        "<h3>What this page does not show</h3>"
        "<p>Two components of the published method are absent. The learned "
        "gated-fusion network is replaced by feature-level concatenation plus the "
        "selection step, on the strength of the paper's own ablation, which found "
        "that a naive learned combiner did worse than task-wise selection on all "
        "22 of its benchmark tasks. And the pool omits the seed-bagging, "
        "rank-blending and top-k sweeps of their full prediction zoo, which "
        "enlarge the pool without changing what it selects over.</p>"
        "<p>Both omissions make this arm a little weaker than the published "
        "method. Neither changes what is being asked, which is whether selecting "
        "per endpoint beats committing to an architecture.</p>"
        "</div>",
        f"<p class=\"note\">Reported the way {LINK_BLOG} argues comparisons "
        "should be: distributions over folds, simultaneous confidence intervals, "
        "paired tests, and no bolded maxima.</p>",
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
