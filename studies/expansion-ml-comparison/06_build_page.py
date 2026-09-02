#!/usr/bin/env python
"""Step 6: build the shareable HTML report.

Reads the tables and figures produced by steps 4 and 5 and writes a single
self-contained page -- figures embedded as data URIs, no external requests --
to results/report.html. Regenerate it whenever the underlying results change.

    python 06_build_page.py
"""

import base64
import io

import pandas as pd
from PIL import Image

import config as cfg
import page_kit

# The page covers both data sets, so it does not belong inside either one's
# results directory.
PAGE = cfg.PROJECT_DIR / "results" / "report.html"

# Which collection a helper should read from when nothing says otherwise.
MAIN = "expansion"
FIG_WIDTH = 1700  # px; figures are downscaled to keep the page under the size limit

# Plot colours and short names, shared with the other report so a method keeps
# its colour and its label across both.
METHOD_COLOR = page_kit.METHOD_COLOR
SHORT = page_kit.SHORT


def embed(name: str, ds: str = MAIN) -> str:
    """A figure as a data URI, downscaled to FIG_WIDTH."""
    path = cfg.paths(ds).figures / f"{name}.png"
    if not path.exists():
        return ""
    img = Image.open(path)
    if img.width > FIG_WIDTH:
        img = img.resize((FIG_WIDTH, round(img.height * FIG_WIDTH / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(name: str, caption: str, ds: str = MAIN) -> str:
    uri = embed(name, ds)
    if not uri:
        return f'<p class="missing">figure {name} not found</p>'
    return (
        '<figure class="bleed">'
        f'<div class="figscroll"><img src="{uri}" alt="{caption}"></div>'
        f"<figcaption>{caption}</figcaption>"
        "</figure>"
    )


def top_sizes(summary: pd.DataFrame) -> pd.Series:
    """How many methods share the top of each endpoint x metric.

    `tukey_groups` calls the method with the leading mean 'best' and anything it
    cannot separate from that method 'equivalent'. The asymmetry is an artefact
    of how the label is computed, not a finding: if the correction cannot tell
    two methods apart, a table has no business crowning one of them. So both
    count as top here, and a row says 'tied' whenever more than one method is up
    there.
    """
    top = summary[summary["tukey_group"].isin(("best", "equivalent"))]
    return top.groupby(["endpoint", "metric"]).size()


def metric_table(summary: pd.DataFrame, metric: str, ds: str = MAIN) -> str:
    """Mean +/- sd per endpoint and method, with the Tukey grouping encoded as a chip."""
    sizes = top_sizes(summary)
    sub = summary[summary["metric"] == metric]
    methods = [m for m in cfg.METHODS if m in set(sub["method"])]
    rows = []
    for endpoint in cfg.paths(ds).dataset.targets:
        g = sub[sub["endpoint"] == endpoint].set_index("method")
        if g.empty:
            continue
        cells = [f"<th scope=\"row\">{endpoint}</th>"]
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
        f"<thead><tr><th scope=\"col\">Endpoint</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def h2h_table(h2h: pd.DataFrame, left: str, right: str, metric: str = "r2",
              ds: str = MAIN) -> str:
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
        better = r["mean_diff"] > 0  # r2: higher is better
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
        f'<thead><tr><th scope="col">Endpoint</th><th scope="col">Δ R²</th>'
        f'<th scope="col">Folds won</th><th scope="col">p</th>'
        f'<th scope="col">Verdict</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def transfer_table(h2h: pd.DataFrame, ds: str = MAIN) -> str:
    """Multitask benefit against how much data the endpoint has.

    The clearest single pattern in the study: the fewer molecules an endpoint has,
    the more it gains from being trained alongside its assay family.
    """
    master = pd.read_csv(cfg.paths(ds).master)
    train = master[master[cfg.SET_COL] == "train"]
    sub = h2h[(h2h["left"] == "chemprop_st") & (h2h["right"] == "chemprop") & (h2h["metric"] == "r2")]
    sub = sub.set_index("endpoint")

    order = sorted(cfg.paths(ds).dataset.targets, key=lambda e: train[e].notna().sum())
    rows = []
    for endpoint in order:
        if endpoint not in sub.index:
            continue
        r = sub.loc[endpoint]
        n = int(train[endpoint].notna().sum())
        cls = "win" if (r["mean_diff"] > 0 and r["p_value"] < 0.05) else (
            "loss" if (r["mean_diff"] < 0 and r["p_value"] < 0.05) else "null")
        rows.append(
            f'<tr><th scope="row">{endpoint}</th>'
            f'<td class="num">{n:,}</td>'
            f'<td class="num">{r["mean_diff"]:+.3f}</td>'
            f'<td class="num">{r["p_value"]:.1e}</td>'
            f'<td><span class="chip {cls}">'
            f'{"multi-task" if cls == "win" else ("single-task" if cls == "loss" else "no call")}'
            "</span></td></tr>"
        )
    return (
        '<div class="bleed tablewrap"><table>'
        '<thead><tr><th scope="col">Endpoint</th><th scope="col">Training molecules</th>'
        '<th scope="col">Δ R² from multi-task</th><th scope="col">p</th>'
        '<th scope="col">Favours</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def tukey_tally(summary: pd.DataFrame) -> str:
    """Per method: alone at the top, sharing the top, or significantly worse.

    Counted symmetrically. A method that shares the top of a combination with one
    other is 'tied' there, and so is the other one, whichever of the two happened
    to have the higher mean.
    """
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
    for m in [x for x in cfg.METHODS if x in counts.index]:
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


# The three method papers, in the order the page first leans on them. `ref()`
# writes the superscript marker; `references()` writes the list they point at.
REFERENCES = [
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
        '<span class="venue">J. Chem. Inf. Model.</span> 2026, articles ASAP. '
        "Introduces CheMeleon, the descriptor-pretrained D-MPNN that "
        "<code>chemprop</code> ships as <code>--from-foundation CHEMELEON</code>; "
        'preprint at <a href="https://arxiv.org/abs/2506.15792">arXiv:2506.15792</a>.',
        "https://doi.org/10.1021/acs.jcim.6c01546",
        "10.1021/acs.jcim.6c01546",
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
        "moljepa",
        "Rottach, F.; Schieferdecker, S.; Rudman, W.; <i>et al.</i> ",
        "Mol-JEPA: A Multimodal Joint Embedding Predictive Architecture for Molecules.",
        "Preprint, 2026. Masks whole molecular modalities and predicts their latents "
        "rather than augmenting the structure. Code, checkpoint and the full "
        "pre-training table at "
        '<a href="https://github.com/Boehringer-Ingelheim/mol-jepa">'
        "github.com/Boehringer-Ingelheim/mol-jepa</a>.",
        "https://arxiv.org/abs/2608.22642",
        "arXiv:2608.22642",
    ),
    (
        "tabicl",
        "Qu, J.; Holzmüller, D.; Varoquaux, G.; Le Morvan, M. ",
        "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data.",
        "Preprint, 2025.",
        "https://arxiv.org/abs/2502.05564",
        "arXiv:2502.05564",
    ),
    (
        "tabpfn",
        "Hollmann, N.; Müller, S.; Purucker, L.; <i>et al.</i> ",
        "Accurate Predictions on Small Data with a Tabular Foundation Model.",
        '<span class="venue">Nature</span> 2025, 637 (8045), 319–326.',
        "https://doi.org/10.1038/s41586-024-08328-6",
        "10.1038/s41586-024-08328-6",
    ),
    (
        "megacl",
        "Jin, T.; Jin, K.; Li, Y.; <i>et al.</i> ",
        "MEGA-CL: A Molecular Foundation Model for Generalizable ADMET Prediction "
        "through Graph External Attention and Contrastive Learning.",
        "Preprint, 2026.",
        "https://arxiv.org/abs/2607.24314",
        "arXiv:2607.24314",
    ),
]

REF_NUMBER = {key: n for n, (key, *_) in enumerate(REFERENCES, start=1)}


def ref(*keys: str) -> str:
    """A superscript marker linking into the reference list."""
    links = ", ".join(
        f'<a href="#ref-{key}">{REF_NUMBER[key]}</a>' for key in keys
    )
    return f'<sup class="ref">{links}</sup>'


def references() -> str:
    items = []
    for key, authors, title, detail, url, label in REFERENCES:
        items.append(
            f'<li id="ref-{key}">{authors}<b>{title}</b> {detail} '
            f'<a href="{url}">{label}</a></li>'
        )
    return '<ol class="refs">' + "".join(items) + "</ol>"


LINK_BLOG = (
    '<a href="https://practicalcheminformatics.blogspot.com/2025/03/'
    'even-more-thoughts-on-ml-method.html">“Even More Thoughts on ML Method '
    'Comparisons”</a>'
)


def load(ds: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One data set's summary, head-to-head and per-fold tables.

    fold_metrics.csv is shared between the reports, so it carries every method
    that has predictions on this data set. This page is about seven of them, and
    the counts in its facts strip are counts of those seven.
    """
    paths = cfg.paths(ds)
    metrics = pd.read_csv(paths.fold_metrics)
    return (pd.read_csv(paths.tables / "summary.csv"),
            pd.read_csv(paths.tables / "head_to_head.csv"),
            metrics[metrics["method"].isin(cfg.METHODS)])


def fold_models(metrics: pd.DataFrame, ds: str) -> int:
    """One fold model per (method, unit, repeat, fold).

    A unit is an endpoint for the single-task methods and an assay family for the
    multitask ones, since a multitask model covers its whole group in one fit.
    Counting endpoints instead would multiply-count chemprop and CheMeleon.
    """
    group_of = cfg.paths(ds).dataset.group_of

    def unit(row):
        variant = cfg.VARIANTS.get(row["method"])
        if variant is not None and not variant["single_task"]:
            return group_of[row["endpoint"]]
        return row["endpoint"]

    return (metrics.assign(unit=metrics.apply(unit, axis=1))
            .groupby(["method", "unit", "repeat", "fold"]).ngroups)


def n_predictions(ds: str) -> int:
    """Predictions belonging to this report's methods, not every method on disk."""
    path = cfg.paths(ds).predictions
    if not path.exists():
        return 0
    preds = pd.read_parquet(path, columns=["method"])
    return int(preds["method"].isin(cfg.METHODS).sum())


def build() -> str:
    summary, h2h, metrics = load(MAIN)
    bio_summary, bio_h2h, bio_metrics = load("biogen")
    n_methods = metrics["method"].nunique()
    n_preds = n_predictions(MAIN) + n_predictions("biogen")
    n_datasets = 2
    n_endpoints = sum(len(cfg.paths(d).dataset.targets) for d in (MAIN, "biogen"))

    n_models = fold_models(metrics, MAIN) + fold_models(bio_metrics, "biogen")

    css = page_kit.CSS

    parts = [
        "<title>Seven ways to model ADME</title>",
        f"<style>{css}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">5×5 cross validation · two data sets</p>',
        "<h1>Which foundation model, and does it replicate?</h1>",
        '<p class="lede">Seven modelling approaches, fifteen ADME and physicochemical '
        "endpoints across two unrelated data sets, 25 replicate models each, every one "
        "scored on a held-out test set it never saw. Four of the seven are pre-trained "
        "foundation models, and only two of those four win anything at all. The model that "
        "wins most does no downstream training whatsoever — it freezes its encoder and "
        "predicts in context. And the neatest pattern in the first data set does not "
        "survive the second.</p>",
        '<div class="facts">'
        f'<div class="fact"><b>{n_methods}</b><span>methods</span></div>'
        f'<div class="fact"><b>{n_datasets}</b><span>data sets</span></div>'
        f'<div class="fact"><b>{n_endpoints}</b><span>endpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b><span>cross validation</span></div>'
        f'<div class="fact"><b>{n_models:,}</b><span>fold models</span></div>'
        f'<div class="fact"><b>{n_preds/1e6:.2f}M</b><span>predictions kept</span></div>'
        "</div>",
        "<h2>How often is each method best?</h2>",
        "<p>Tukey HSD across the 25 folds, counted over every endpoint × metric "
        "combination. <em>Tied</em> means the method could not be distinguished from the "
        "best one at α = 0.05 — a distinction a bolded maximum would hide.</p>",
        '<div class="panel">' + tukey_tally(summary) + "</div>",
        "<p>Of the 27 combinations, 19 have a single method at the top and 8 have two or "
        "three that cannot be told apart. Monroe is alone at the top on 12 and shares it "
        "on 6, CheMeleon alone on 7 and shares on 5. Nothing else is ever at the top, "
        "alone or otherwise. The split between those two is not noise and it does not "
        "follow data volume — it follows the assay. Monroe leads on LogD, both microsomal "
        "stability endpoints and both Caco-2 endpoints. CheMeleon leads on all three "
        "tissue-binding endpoints, plus LogS.</p>",
        "<p>The one place a from-scratch model keeps up is <b>LogD</b>, the endpoint with "
        "the most measurements, where single-task ChemProp shares the top with Monroe on "
        "all three metrics. Given enough data, the model learns a representation as "
        "good as the one it would have been handed.</p>",
        "<p>The other end of the table is just as clean. LightGBM, MEGA-CL and Mol-JEPA "
        "are best on nothing and tied with the best on nothing, across all 27 "
        "combinations. Two of those three are pre-trained foundation models. Pre-training "
        "is not by itself worth anything here.</p>",
        '<div class="panel howto">'
        "<h3>Why nothing here is bolded</h3>"
        "<p>Where two methods cannot be separated, the tables below mark <i>both</i> as "
        "tied for best rather than crowning whichever had the higher mean. A leading "
        "average that the correction cannot defend is not a winner, and printing it as "
        "one is the habit these plots exist to avoid.</p>"
        "</div>",
        '<div class="panel howto">'
        "<h3>How to read the Tukey plots</h3>"
        "<p>Each bar is one method's mean over its 25 folds. The whiskers are a "
        "confidence interval widened to cover every pairwise comparison in the panel "
        "at once, which is what stops seven methods and three metrics from "
        "manufacturing a winner by chance.</p>"
        "<ul>"
        '<li><b class="k-best">Blue</b> is the method with the best mean. Two dashed '
        "vertical lines mark where its interval starts and ends.</li>"
        '<li><b class="k-tied">Grey</b> means the interval overlaps the blue one. The '
        "difference does not survive the correction, so the two methods are not "
        "separated by this data.</li>"
        '<li><b class="k-loss">Red</b> means the interval clears the dashed lines '
        "entirely. That method is worse, and the gap is real.</li>"
        "</ul>"
        "<p>Overlapping bars mean what they look like they mean. That is the whole "
        "point of correcting the intervals first. R² and Spearman ρ put the best "
        "method furthest right. MAE is an error, so its best method sits furthest "
        f'left. The convention comes from {LINK_BLOG} and the protocols in '
        f"Ash <i>et al.</i>{ref('tukey')}</p>"
        "</div>",
        figure("tukey_r2", "Tukey HSD on R². Blue is the best method, grey is statistically "
                          "indistinguishable from it, red is significantly worse."),
        "<h2>R² by endpoint</h2>",
        "<p>Mean ± standard deviation over 25 folds.</p>",
        metric_table(summary, "r2"),
        "<h2>Question 1 — does pretraining help?</h2>",
        f'<p>CheMeleon{ref("chemeleon")} against the identical multitask D-MPNN{ref("chemprop")} '
        "trained from scratch. Same "
        "molecules, same folds, same hyperparameters; the only difference is how the "
        "message-passing block was initialised.</p>",
        h2h_table(h2h, "chemprop", "chemeleon"),
        figure("paired_chemprop_vs_chemeleon_r2",
               "Each line is one fold, seen by both methods. Green favours CheMeleon."),
        "<h2>Question 2 — does multi-task training help?</h2>",
        "<p>The same from-scratch architecture, trained one model per assay family versus "
        "one model per endpoint. This is the control that separates multi-task transfer "
        "from the architecture itself, and it produces the cleanest gradient in the study: "
        "sort the endpoints by how much data they have and the benefit falls away almost "
        "monotonically. LOG_MGMB, with 222 training molecules, gains 0.34 R² from being "
        "trained alongside its assay family. LogD, with 5,039, is actively hurt.</p>",
        transfer_table(h2h),
        figure("paired_chemprop_st_vs_chemprop_r2",
               "Multi-task versus single-task ChemProp, paired by fold. Green favours "
               "multi-task."),
        "<h2>Question 3 — graph network or fingerprints?</h2>",
        "<p>LightGBM on Morgan count fingerprints against the single-task D-MPNN — both "
        "single-task, so this is the like-for-like comparison, and it is a split decision. "
        "The graph network wins on the two endpoints with the most data (LogD, LogS) and "
        "loses on five of the remaining seven. Stripped of pretraining and multi-task "
        "transfer, a D-MPNN is not a general improvement on fingerprints for this data "
        "set.</p>",
        h2h_table(h2h, "lgbm", "chemprop_st"),
        figure("boxplot_r2", "Fold-level R² distributions. The spread matters as much as "
                             "the centre: several endpoints overlap heavily."),
        "<h2>Question 4 — does every foundation model help?</h2>",
        f'<p>MEGA-CL{ref("megacl")} is a graph contrastive-learning model pre-trained on '
        "roughly 100 million molecules. It is single-target by construction, so the like-for-like "
        "comparison is against single-task ChemProp: same one-model-per-endpoint setup, "
        "same folds, pre-trained backbone against random initialisation. It loses that "
        "comparison on five endpoints, wins on three and ties on one. Against LightGBM it "
        "is a coin flip, three wins, three ties, three losses. Against CheMeleon it loses "
        "nine out of nine.</p>",
        h2h_table(h2h, "chemprop_st", "megacl"),
        figure("paired_chemprop_st_vs_megacl_r2",
               "MEGA-CL against single-task ChemProp, paired by fold. Both are one model "
               "per endpoint, so this isolates the pre-training."),
        "<p>Where it does help is where the data runs out. On LOG_MGMB, the smallest "
        "endpoint at 222 training molecules, MEGA-CL beats LightGBM by 0.187 R². On the "
        "two Caco-2 endpoints it collapses, and not only on R²: its Spearman ρ of 0.225 on "
        "LOG_Caco_AB is half of what LightGBM manages. The lesson is not that pre-training "
        "fails, since the only two methods that win anything here are both pre-trained. "
        "It is that the "
        "benefit belongs to a particular pre-trained model, not to pre-training as an "
        "idea.</p>",
        "<h2>Question 5 — does a foundation model need fine-tuning at all?</h2>",
        f'<p>Monroe{ref("monroe")} is a 58.5 M-parameter graph transformer pre-trained on 81 '
        "million PM6 molecules and 1,089 PCBA assays. What separates it from everything else "
        "here is that nothing is trained downstream. The encoder is frozen, each molecule "
        "becomes a single 720-dimensional vector, and "
        f'TabPFN{ref("tabpfn")} predicts the endpoint <i>in context</i>: it is handed the '
        "training embeddings together with their labels and returns the test predictions in "
        "one forward pass. No weight updates, no epochs, no per-endpoint hyperparameters. "
        "Embedding all 7,608 molecules took 32 seconds on one GPU and the 225 fold "
        "predictions took 14 minutes. On the same card the CheMeleon arm took about six "
        "hours and MEGA-CL took twenty-one.</p>",
        h2h_table(h2h, "chemeleon", "monroe"),
        figure("paired_chemeleon_vs_monroe_r2",
               "Monroe against CheMeleon, paired by fold. The two split the endpoints "
               "cleanly rather than one dominating."),
        "<p>Caco-2 is where the gap is widest. Monroe reaches R² 0.471 on permeability and "
        "0.275 on efflux; nothing else clears 0.16 and four of the other five land below "
        "zero on both. Its Spearman ρ on permeability is 0.706 against CheMeleon's 0.564 and "
        "LightGBM's 0.469, and it wins all 25 folds on all three metrics. Microsomal "
        "stability goes the same way, if less dramatically.</p>",
        "<p>Protein binding reverses it. On LOG_MPPB, CheMeleon wins all 25 folds on both "
        "R² and Spearman, 0.541 against 0.300, and it takes LOG_MBPB and LOG_MGMB as well. "
        "Whatever the frozen embedding captures about passive permeability and microsomal "
        "turnover, it does not capture what fraction of a compound stays unbound in "
        "plasma — and there, fine-tuning the representation on the assay still wins.</p>",
        "<p>One more thing separates it. Monroe's fold-to-fold standard deviation is the "
        "smallest of the six on all three metrics: 0.052 against CheMeleon's 0.088 on R², "
        "averaged over the endpoints. That follows from the design. With no training loop "
        "there is no initialisation, no early-stopping epoch and no optimiser trajectory to "
        "vary. The only thing that changes between folds is which molecules are in the "
        "support set.</p>",
        '<div class="panel howto">'
        "<h3>Could Monroe have seen these labels?</h3>"
        "<p>It could not. Monroe pre-trains on 1,152 tasks and every one of them is public "
        "and named: 62 graph-level semi-empirical quantum properties from PM6, 1,089 binary "
        "PubChem bioassay calls from PCBA, and one conformer denoising objective. PM6 is "
        "computed chemistry and PCBA is screening activity. Neither carries an ADME "
        "measurement, so LogD, microsomal stability, Caco-2 permeability and plasma protein "
        "binding have no route into that training signal, on this data set or the Biogen "
        "one.</p>"
        "<p>Seeing a molecule is a different matter from seeing its label, and the overlap "
        "has now been counted rather than assumed. Blazej Banaszewski, one of Monroe's "
        "authors, checked both test sets against the pre-training corpora and found no label "
        "overlap at all. Exactly one ExpansionRx test molecule, one carrying LogD and LogS "
        "values, is in PM6. On the Biogen side the structural overlap is much larger: about "
        "53% of the test molecules are in PM6 and about 8% in PCBA. PM6 contributes "
        "quantum-chemical descriptors that are unrelated to these assays, and the PCBA "
        "bioassays that come closest to the Biogen endpoints biologically are still different "
        "experiments reporting different labels. Monroe had seen a good many of these "
        "molecules. It had never seen what they measure. "
        "<span class=\"footnote\">Overlap figures by personal communication, August 2026.</span></p>"
        "<p>One thing is still worth stating. Monroe's authors have run their model on this "
        "data set, and their repository ships a notebook that predicts the untransformed "
        "ExpansionRx measurements against OpenADMET's CheMeleon baseline. The folds, the "
        "transform and the test set here are ours, and no Monroe hyperparameter was tuned "
        "on them.</p>"
        "</div>",
        "<h2>Question 6 — does adding modalities beat adding scale?</h2>",
        f'<p>Mol-JEPA{ref("moljepa")} is the same shape of arm as Monroe and a different bet '
        "about what a molecule is. Instead of augmenting a structure and asking for matching "
        "views, it collects fourteen <i>modalities</i> of the same molecule — graph, ECFP, MOE "
        "descriptors, xTB and DFT calculations, embeddings borrowed from five other pre-trained "
        "models, and experimental ChEMBL, PCBA and TDC label vectors — masks whole modalities "
        "out, and trains a transformer to predict the missing latents from the ones that "
        "remain. The idea is that a molecule is defined by the company it keeps, not by a "
        "perturbation of its own graph. Only structure is needed at inference. The 45.4 M "
        "parameter encoder is frozen here, every molecule becomes one 512-d CLS token, and "
        f'TabICL{ref("tabicl")} predicts in context, which is what the authors recommend.</p>',
        "<p>It is best on nothing and tied with the best on nothing. Monroe beats it on all "
        "nine endpoints and all three metrics, and on most of them it wins all 25 folds. The "
        "mean R² gap is 0.18.</p>",
        h2h_table(h2h, "monroe", "moljepa"),
        figure("paired_monroe_vs_moljepa_r2",
               "Monroe against Mol-JEPA, paired by fold. Two frozen encoders, two in-context "
               "heads, one representation clearly ahead."),
        "<p>The obvious objection is the head, since Monroe uses TabPFN and this arm uses "
        "TabICL. So the same embeddings were run through TabPFN at Monroe's settings. Across "
        "the 225 folds that moves Mol-JEPA by 0.013 R² and 0.013 Spearman ρ — detectable, at "
        "p = 0.003 and 3 × 10⁻¹⁰, and about a fourteenth of the distance to Monroe. The gap "
        "is the representation, not the predictor bolted onto it.</p>",
        "<p>None of which makes it a bad model. On Caco-2 permeability it reaches R² 0.135 "
        "where LightGBM manages −0.014 and MEGA-CL −0.747, and on LOG_MGMB, the smallest "
        "endpoint in the study, it reaches 0.503 against CheMeleon's 0.642. It is a "
        "respectable model that happens to be in a study with two better ones.</p>",
        '<div class="panel howto">'
        "<h3>Checking the pre-training table</h3>"
        "<p>Mol-JEPA's authors also evaluated on this data set, and here the question of "
        "label overlap is a real one. Two of the fourteen modalities are experimental label "
        "vectors from ChEMBL, PCBA and TDC, and TDC does carry ADME tasks. It is also the "
        "one model where the question can be settled exactly, because the authors released "
        "the entire pre-training table, 4,663,780 rows, with an InChIKey for every "
        "molecule.</p>"
        "<p>Joining our 7,608 molecules to it gives <b>no exact key matches at all</b>. Two "
        "molecules share a connectivity block with one of ours, both from PubChem BioAssay, "
        "and neither carries an ADME measurement. The table does have columns named for the "
        "nine ExpansionRx endpoints, but every one of them is empty across all 4.66 million "
        "rows. Whatever this model saw in pre-training, it was not these molecules and it "
        "was not these labels.</p>"
        "<p>The check is <code>11_check_pretraining_overlap.py</code> in the repository. "
        "Monroe's corpus is not published as a table, but its task list is, and it contains "
        "no ADME endpoint to leak.</p>"
        "</div>",
        "<h2>Ranking is healthier than R²</h2>",
        "<p>Several endpoints post R² at or below zero while ranking test compounds "
        "perfectly respectably. The test set is genuinely shifted from the training set — "
        "for LOG_Caco_Efflux the test mean is 0.92 against 0.51 in training — so a model "
        "that hedges toward the training mean is punished by R² while its ordering "
        "survives. Spearman ρ tells the more useful story for triage.</p>",
        metric_table(summary, "spearman"),
        figure("tukey_spearman", "Tukey HSD on Spearman ρ."),
        "<h2>Mean absolute error</h2>",
        "<p>MAE is the metric a chemist reads closest to directly: it is in log units of the "
        "measurement, and it does not depend on how the test set happens to be spread. "
        "That makes it the fairest of the three here, because it is unmoved by the label "
        "shift that drags R² below zero. The ordering barely changes. Monroe has the "
        "lowest error on five of nine endpoints and CheMeleon on the other four, split the "
        "same way as before: metabolism and permeability against binding. LightGBM, "
        "MEGA-CL and Mol-JEPA are significantly worse than the best on all nine.</p>",
        '<div class="panel howto">'
        "<h3>What a log10(x+1) MAE is worth</h3>"
        "<p>The <code>+1</code> is there because six of the nine endpoints report zeros "
        "that plain <code>log10</code> cannot take: 179 of them in human microsomal "
        "stability, 156 in mouse, and fewer than a dozen apiece in the other four.</p>"
        "<p>It also makes the transform nearly linear below x = 1 and only "
        "properly logarithmic above about x = 10, and a good deal of this data sits in "
        "the flat part: 22% of mouse brain binding values and 17% of Caco-2 Papp values "
        "are below 1 in their native units.</p>"
        "<p>So a fixed fold-change is not worth a fixed distance. On plasma protein "
        "binding, a five-fold difference in free fraction spans 0.67 units between 10% "
        "and 50% unbound, but only 0.14 units between 0.1% and 0.5%. The scale "
        "under-weights errors on the most tightly bound compounds, which is the end a "
        "chemist usually cares about most.</p>"
        "<p>Every method is fit and scored on the same transformed target, so the "
        "comparison below is untouched by this. It is the absolute reading that needs "
        "care. An MAE of 0.175 is not a clean 1.5-fold error.</p>"
        "</div>",
        metric_table(summary, "mae"),
        figure("tukey_mae", "Tukey HSD on MAE. Lower is better, so the best method sits "
                            "furthest left."),
        "<p>The gaps are worth reading in absolute terms rather than as ranks. On LOG_MGMB "
        "CheMeleon is at 0.175 against LightGBM's 0.302, so the fingerprint model is wrong "
        "by roughly 73% more per compound. On LOG_HLM the six methods span 0.293 to 0.377, "
        "a spread of 0.084 log units, narrow enough that the assay noise probably matters "
        "more than the choice.</p>",
        figure("boxplot_mae", "Fold-level MAE distributions."),
        '<hr class="rule">',
        "<h2>Does any of it replicate?</h2>",
        "<p>Everything above is one data set. A comparison run once is a hypothesis, so "
        "the same seven methods, the same protocol and the same statistics were run again "
        f'on Biogen\'s public ADME set{ref("biogen")}: 3,521 commercially sourced compounds '
        "on six endpoints, unrelated to the first collection in chemistry, in provenance "
        "and in who measured it.</p>",
        "<p>The two files are not the same kind of thing. ExpansionRx is a lead "
        "optimisation campaign — 7,608 molecules in 651 clusters, 170 of them singletons. "
        "The Biogen set is a diverse commercial selection: 3,521 molecules in 1,905 "
        "clusters, 1,123 singletons. It carries no train/test split, so one was built by "
        "holding out whole clusters to the same 30% the ExpansionRx file uses. That makes "
        "the Biogen holdout cluster-pure, which the ExpansionRx one is not: 59 of its 651 "
        "clusters straddle the boundary. Biogen is the harder test of the two, so absolute "
        "numbers do not transfer between the halves of this page. Rankings within each "
        "half do.</p>",
        figure("tukey_r2", "Tukey HSD on R² for the Biogen endpoints. Same reading as "
                           "before: blue is best, grey indistinguishable from it, red "
                           "significantly worse.", ds="biogen"),
        '<div class="panel">' + tukey_tally(bio_summary) + "</div>",
        "<p>Eighteen combinations this time, six endpoints by three metrics. <b>Monroe "
        "takes all eighteen.</b> Nothing else is best on one, and nothing else is so much "
        "as tied with the best on one. Against CheMeleon it wins every fold of every "
        "endpoint on every metric — 450 out of 450.</p>",
        metric_table(bio_summary, "r2", ds="biogen"),
        "<p>So the headline replicates and the interesting part does not. On ExpansionRx "
        "the wins split by assay, with CheMeleon taking the three tissue-binding "
        "endpoints. That pattern is gone. Biogen's two plasma protein binding endpoints "
        "are precisely where CheMeleon ought to have been strong, and they are where it "
        "loses by the widest margin on the page: R² 0.478 against 0.230 on human, 0.415 "
        "against 0.154 on rat.</p>",
        "<p>What shows through instead is data volume, and the clearest sight of it is "
        "the one comparison where the two frozen-encoder models disagree. Mol-JEPA, best "
        "at nothing in the first half, beats CheMeleon here on the three endpoints with "
        "the least data — solubility, and the two protein binding assays with 128 and 109 "
        "training molecules — and loses to it on the three with the most. In-context "
        "prediction from a frozen encoder degrades gently as the labels run out. "
        "Fine-tuning falls off a cliff: on those two endpoints single-task ChemProp posts "
        "R² of −0.064 and −0.005, worse than predicting the training mean, and MEGA-CL "
        "manages −0.292 and −0.426.</p>",
        h2h_table(bio_h2h, "chemeleon", "moljepa", ds="biogen"),
        "<p>Three things do carry across. Monroe is the most accurate method on both "
        "collections and also the steadiest, with the smallest fold-to-fold standard "
        "deviation on all three metrics in both. MEGA-CL is the worst on both. And a "
        "from-scratch D-MPNN is still not reliably better than a fingerprint baseline: on "
        "Biogen, single-task ChemProp beats LightGBM on the four larger endpoints and "
        "loses to it on the two smallest.</p>",
        figure("boxplot_r2", "Fold-level R² on the Biogen endpoints.", ds="biogen"),
        "<p>MAE says the same thing, which is worth checking rather than assuming: R² is "
        "sensitive to how the test set happens to be spread, and a verdict that held on "
        "only one metric would be a verdict about the split. Monroe has the lowest error "
        "on all six endpoints, by 0.029 log units on rat microsomal stability at the "
        "narrowest and 0.091 on rat protein binding at the widest. The runner-up changes "
        "with the amount of data, and splits the six endpoints the same way R² did: "
        "CheMeleon is second on the three with the most training molecules, Mol-JEPA on "
        "the three with the least.</p>",
        figure("tukey_mae", "Tukey HSD on MAE for the Biogen endpoints. Lower is better, "
                            "so the best method sits furthest left.", ds="biogen"),
        metric_table(bio_summary, "mae", ds="biogen"),
        '<div class="panel howto">'
        "<h3>Two things to know about this data</h3>"
        "<p>31% of the HLM values and 11% of the RLM values sit exactly on the assay "
        "floor, stacked on one number because the compound was too stable to measure a "
        "rate. A large block of ties is easy to order, so that flatters Spearman ρ, and "
        "it makes MAE look better than the assay deserves.</p>"
        "<p>The two protein binding endpoints have 128 and 109 training molecules here. "
        "The repository's own splits for them hold about nine times that, drawn from "
        "in-house measurements that were never released, so these numbers are not "
        "comparable with the ones in the paper.</p>"
        "</div>",
        "<p>The Mol-JEPA pre-training check was run again, and it matters more on this "
        "side: the Biogen set is public and reaches that pre-training table through "
        "Therapeutic Data Commons, which carries a column for each of the six endpoints. "
        "59 of the 3,521 molecules do appear by exact InChIKey, 1.7% of them, through "
        "nabla-DFT, PubChem BioAssay and TDC, with 132 more sharing a connectivity block. "
        "Not one carries a Biogen measurement, and those six label columns are empty "
        "across all 4.66 million rows. Structures overlap slightly. Labels do not.</p>",
        '<hr class="rule">',
        "<h2>How this was run</h2>",
        "<p>The measurements are the "
        '<a href="https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data">'
        "OpenADMET–ExpansionRx blind challenge set</a>, contributed by Expansion "
        "Therapeutics and released by OpenADMET under CC BY 4.0. Every endpoint was put "
        "on a <code>log10(x+1)</code> scale except LogD, which is already a log quantity. "
        "The <code>+1</code> is there because six endpoints report zeros, which plain "
        "<code>log10</code> cannot take. "
        "LogS and Caco-2 Papp carry a further −6, which moves them from µM and "
        "10⁻⁶ cm/s into molar and cm/s. That shift is a constant, so it cancels in "
        "every metric here. "
        "The 651 chemical clusters come from BitBIRCH-Lean at its default settings.</p>",
        "<p>The second collection is "
        '<a href="https://github.com/molecularinformatics/Computational-ADME">'
        "Biogen's public ADME set</a>, 3,521 compounds on six endpoints, released under "
        f'MIT with the paper of Fang <i>et al.</i>{ref("biogen")} Its values arrive already '
        "log transformed, so nothing was done to them. It carries no train/test split, so "
        "<code>00b_prepare_biogen.py</code> builds one: BitBIRCH-Lean clusters the 3,521 "
        "molecules into 1,905 groups, and whole clusters are held out in a seeded order "
        "until the test set reaches the same 30% the ExpansionRx file uses. Everything "
        "downstream of that point is shared between the two collections — the same fold "
        "construction, the same seven arms, the same statistics.</p>",
        "<p>The <code>ds</code> column fixes the train/test split (5,326 / 2,282 molecules "
        "after dropping unparseable SMILES and rows with no measured endpoint). The 25 "
        "replicates come from five repeats of a five-fold <code>GroupKFold</code> over the "
        "training molecules, grouped by <code>cluster</code> so no chemotype straddles the "
        "train/validation boundary. Each fold fits on four fifths, early-stops on the "
        "held-out fifth where there is a training loop to stop, and is scored on the same "
        "untouched test set — so every method sees identical training molecules in every "
        "fold and a difference in the metrics is a difference in the method.</p>",
        "<ul>"
        "<li><b>LightGBM + Morgan</b> — count fingerprints, radius 2, 2048 bits; library "
        "defaults; one model per endpoint.</li>"
        f'<li><b>ChemProp ST / MT</b>{ref("chemprop")} — chemprop 2.3.0 D-MPNN, 50 epochs, batch 64, one '
        "network per fold. Multitask groups are the three assay families: physicochemical "
        "and tissue binding, microsomal stability, Caco-2 permeability.</li>"
        f'<li><b>CheMeleon</b>{ref("chemeleon")} — the same multitask model with message passing initialised '
        "from the CheMeleon foundation model, then fine-tuned.</li>"
        f'<li><b>MEGA-CL</b>{ref("megacl")} — the authors\' pre-trained checkpoint and their own '
        "fine-tuning recipe (100 epochs, batch 32), one model per endpoint. Only the "
        "split was overridden, so it reads the same folds as everything else.</li>"
        f'<li><b>Monroe + TabPFN</b>{ref("monroe")} — the authors\' pre-trained encoder, frozen. '
        "Every molecule becomes one 720-d embedding, computed once for the whole data set, "
        f'and TabPFN{ref("tabpfn")} predicts each endpoint in context from that fold\'s training '
        "embeddings. Nothing is fitted by gradient descent, so the held-out fifth goes "
        "unused, exactly as it does for LightGBM.</li>"
        f'<li><b>Mol-JEPA + TabICL</b>{ref("moljepa")} — the authors\' pre-trained encoder, '
        "frozen. One 512-d CLS token per molecule, then "
        f'TabICL{ref("tabicl")} in context, which is what their model card recommends. '
        "A TabPFN head was run over the same embeddings as a control.</li>"
        "</ul>",
        '<p class="repo">Every script, the data set and all 2,483,250 predictions '
        'are at <a href="https://github.com/PatWalters/model-validation-central/tree/main/studies/expansion-ml-comparison">'
        'model-validation-central/studies/expansion-ml-comparison</a>. The figures and tables '
        'on this page rebuild from the stored predictions in about a minute, with '
        'no retraining.</p>',
        '<p class="footnote">Statistics follow the approach in '
        '<a href="https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html">'
        "“Even More Thoughts on ML Method Comparisons”</a>: distributions of fold-level "
        "metrics, Tukey HSD corrected for multiple comparisons, and paired tests using the "
        "folds as the pairing. Every individual prediction is retained in "
        "<code>predictions_all.parquet</code>.</p>",
        "<h2>References</h2>",
        references(),
        "</div>",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    PAGE.write_text(build())
    print(f"wrote {PAGE} ({PAGE.stat().st_size/1e6:.1f} MB)")
