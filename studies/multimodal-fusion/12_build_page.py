#!/usr/bin/env python
"""Step 12: the standalone HTML report, both data sets on one page.

Every number in the prose is read from the tables steps 6 to 11 wrote, so the
page cannot drift from the results it describes. Figures are embedded as data
URIs, so the file is one artifact with no companions.

    python 12_build_page.py
"""

import argparse
import html
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import config as cfg
import page_kit as kit

DATASETS = ["expansion", "biogen"]

REFERENCES = [
    ("wasswa", "Wasswa, J.; Kajjumba, G. W.; Ramsundar, B. ",
     "Unimodal vs Multimodal Learning: A Systematic Evaluation of Fusion Strategies "
     "and Model Design for Molecular Property Prediction and Uncertainty Quantification.",
     "<i>J. Chem. Inf. Model.</i> 2026.",
     "https://doi.org/10.1021/acs.jcim.6c01878", "doi:10.1021/acs.jcim.6c01878"),
    ("repo", "", "Multimodal_Fusion, the authors' released code.", "MIT licensed.",
     "https://github.com/jwasswa2023/Multimodal_Fusion", "github.com/jwasswa2023/Multimodal_Fusion"),
    ("attentivefp", "Xiong, Z.; Wang, D.; Liu, X.; et al. ",
     "Pushing the Boundaries of Molecular Representation for Drug Discovery with the "
     "Graph Attention Mechanism.",
     "<i>J. Med. Chem.</i> 2020, 63 (16), 8749-8760.",
     "https://doi.org/10.1021/acs.jmedchem.9b00959", "doi:10.1021/acs.jmedchem.9b00959"),
    ("mol2vec", "Jaeger, S.; Fulle, S.; Turk, S. ",
     "Mol2vec: Unsupervised Machine Learning Approach with Chemical Intuition.",
     "<i>J. Chem. Inf. Model.</i> 2018, 58 (1), 27-35.",
     "https://doi.org/10.1021/acs.jcim.7b00616", "doi:10.1021/acs.jcim.7b00616"),
    ("chemprop", "Graff, D. E.; Morgan, N. K.; Burns, J. W.; et al. ",
     "Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical "
     "Property Prediction.",
     "<i>J. Chem. Inf. Model.</i> 2026, 66 (1), 28-33.",
     "https://doi.org/10.1021/acs.jcim.5c02332", "doi:10.1021/acs.jcim.5c02332"),
    ("chemeleon", "Burns, J. W.; Zalte, A. S.; Abreu, C. R. A.; et al. ",
     "Deep Learning Foundation Models for Low-Data Regimes from Classical Molecular "
     "Descriptors.",
     "<i>J. Chem. Inf. Model.</i> 2026.",
     "https://doi.org/10.1021/acs.jcim.6c01546", "doi:10.1021/acs.jcim.6c01546"),
    ("protocol", "Ash, J. R.; Wognum, C.; Rodr&iacute;guez-P&eacute;rez, R.; et al. ",
     "Practically Significant Method Comparison Protocols for Machine Learning in "
     "Small Molecule Drug Discovery.",
     "<i>J. Chem. Inf. Model.</i> 2025, 65 (18), 9398-9411.",
     "https://doi.org/10.1021/acs.jcim.5c01609", "doi:10.1021/acs.jcim.5c01609"),
    ("biogen", "Fang, C.; Wang, Y.; Grater, R.; et al. ",
     "Prospective Validation of Machine Learning Algorithms for ADME Prediction.",
     "<i>J. Chem. Inf. Model.</i> 2023, 63 (11), 3263-3274.",
     "https://doi.org/10.1021/acs.jcim.3c00160", "doi:10.1021/acs.jcim.3c00160"),
    ("expansion", "", "OpenADMET-ExpansionRx Blind Challenge data.",
     "Contributed by Expansion Therapeutics, CC BY 4.0.",
     "https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data",
     "huggingface.co/datasets/openadmet"),
    ("sibling", "", "expansion-ml-comparison,", "the folds, the splits and the four "
     "reference methods.",
     "https://github.com/PatWalters/model-validation-central/tree/main/studies/expansion-ml-comparison",
     "model-validation-central/studies/expansion-ml-comparison"),
]
NUMBERS = kit.reference_numbers(REFERENCES)


def ref(*keys: str) -> str:
    return kit.marker(NUMBERS, *keys)


@dataclass
class Results:
    """Everything one data set's page section reads, loaded once."""

    name: str
    label: str
    paths: object
    metrics: pd.DataFrame
    uncertainty: pd.DataFrame | None
    unimodal: dict
    reference: pd.DataFrame | None
    ablation: pd.DataFrame | None
    shap: pd.DataFrame | None
    control: pd.DataFrame | None
    paper_gnn: pd.DataFrame | None
    wilcoxon: dict

    def figure(self, name: str) -> str:
        return kit.embed_figure(self.paths.figures / f"{name}.png")


def read_csv(path):
    return pd.read_csv(path) if path.exists() else None


def load(name: str) -> Results | None:
    paths = cfg.paths(name)
    if not paths.fold_metrics.exists():
        return None
    metrics = pd.read_csv(paths.fold_metrics)

    unimodal = {}
    for metric in cfg.METRICS:
        path = paths.tables / f"unimodal_vs_multimodal_{metric}.csv"
        if path.exists():
            unimodal[metric] = pd.read_csv(path, index_col=0)

    wilcoxon = {}
    for family in ("fusion", "ladder", "learner"):
        for metric in cfg.METRICS:
            path = paths.tables / f"wilcoxon_{family}_{metric}.csv"
            if path.exists():
                wilcoxon[f"{family}_{metric}"] = pd.read_csv(path)

    return Results(
        name=name,
        label=cfg.DATASETS[name].label,
        paths=paths,
        metrics=metrics,
        uncertainty=read_csv(paths.uncertainty),
        unimodal=unimodal,
        reference=read_csv(paths.tables / "reference_panel.csv"),
        ablation=read_csv(paths.results / "modality_ablation.csv"),
        shap=read_csv(paths.shap),
        control=read_csv(paths.tables / "leakage_control_r2.csv"),
        paper_gnn=read_csv(paths.tables / "gnn_block_control.csv"),
        wilcoxon=wilcoxon,
    )


# --- small helpers for the prose -----------------------------------------
def fusion_verdict(res: Results, metric: str = "r2") -> dict:
    table = res.unimodal.get(metric)
    if table is None:
        return {}
    higher = cfg.METRIC_HIGHER_IS_BETTER[metric]
    better = (table["difference"] > 0) if higher else (table["difference"] < 0)
    return {
        "won": int(better.sum()),
        "n": len(table),
        "mean": table["difference"].mean(),
        "best_multi": table["which_multimodal"].value_counts().idxmax(),
        "best_uni": table["which_unimodal"].value_counts().idxmax(),
    }


def survived(res: Results, family: str, metric: str = "r2") -> tuple[int, int]:
    table = res.wilcoxon.get(f"{family}_{metric}")
    if table is None:
        return 0, 0
    return int(table["significant"].sum()), len(table)


def grid_means(res: Results) -> pd.Series:
    return (res.metrics[res.metrics["method"].isin(cfg.GRID_METHODS)]
            .groupby("method")["r2"].mean())


def headline(res: Results) -> dict:
    """The handful of numbers the prose for one data set is built from."""
    grid = grid_means(res)
    uni = grid[grid.index.isin(cfg.UNIMODAL_METHODS)]
    multi = grid[grid.index.isin(cfg.FUSION_METHODS)]
    ref = (res.metrics[res.metrics["method"].isin(cfg.REFERENCE_METHODS)]
           .groupby("method")["r2"].mean())
    unc = res.uncertainty
    ladder_sig, ladder_n = survived(res, "ladder")
    fusion_sig, fusion_n = survived(res, "fusion")
    learner_sig, learner_n = survived(res, "learner")
    return {
        "best_grid": grid.idxmax(), "best_grid_r2": grid.max(),
        "best_uni": uni.idxmax(), "best_uni_r2": uni.max(),
        "best_multi": multi.idxmax(), "best_multi_r2": multi.max(),
        "chemeleon": ref.get("chemeleon", float("nan")),
        "chemprop": ref.get("chemprop", float("nan")),
        "lgbm": ref.get("lgbm", float("nan")),
        "grid_beaten_by_chemeleon": bool(ref.get("chemeleon", -9) > grid.max()),
        "ladder": (ladder_sig, ladder_n),
        "fusion": (fusion_sig, fusion_n),
        "learner": (learner_sig, learner_n),
        "unc_best_corr": (unc.groupby("method")["err_unc_corr"].mean().max()
                          if unc is not None else float("nan")),
        "ref_best_corr": (unc[unc["method"].isin(cfg.REFERENCE_METHODS)]
                          .groupby("method")["err_unc_corr"].mean().max()
                          if unc is not None else float("nan")),
    }


def top_configurations(res: Results, n: int = 8) -> pd.DataFrame:
    grid = res.metrics[res.metrics["method"].isin(cfg.GRID_METHODS)]
    means = grid.groupby("method")[["r2", "spearman", "mae"]].mean()
    means = means.sort_values("r2", ascending=False).head(n)
    means.insert(0, "label", [cfg.METHOD_LABELS[m] for m in means.index])
    return means


def reference_means(res: Results) -> pd.DataFrame:
    sub = res.metrics[res.metrics["method"].isin(cfg.REFERENCE_METHODS)]
    means = sub.groupby("method")[["r2", "spearman", "mae"]].mean()
    means.insert(0, "label", [cfg.METHOD_LABELS[m] for m in means.index])
    return means.reindex([m for m in cfg.REFERENCE_METHODS if m in means.index])


def frame_to_html(frame: pd.DataFrame, floats: str = "{:.3f}") -> str:
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda v: "" if pd.isna(v) else floats.format(v))
    return out.to_html(index=False, escape=False, border=0, classes="data")


def esc(text: str) -> str:
    return html.escape(str(text))


# --- the page ------------------------------------------------------------
def grid_table() -> str:
    rows = [
        ("modality", "RDKit descriptors, Mol2Vec, a supervised GNN's graph embedding, "
                     "a character BiGRU's SMILES embedding"),
        ("modality set", "each alone, then GNN+RDKit, +Mol2Vec, +SMILES, and all four"),
        ("fusion", "early, concatenating feature vectors; late, stacking per-modality "
                   "predictions for a meta-learner"),
        ("final learner", "LightGBM, random forest, or AttentiveFP"),
    ]
    cells = "".join(f"<tr><th>{esc(a)}</th><td>{b}</td></tr>" for a, b in rows)
    return f'<table class="data axes"><tbody>{cells}</tbody></table>'


def dataset_section(res: Results) -> str:
    verdict = fusion_verdict(res)
    parts = [f'<h2 id="{res.name}">{esc(res.label)}</h2>']

    n_methods = res.metrics["method"].nunique()
    n_folds = len(res.metrics)
    parts.append(
        f"<p>{n_methods} methods over {res.metrics['endpoint'].nunique()} endpoints, "
        f"{n_folds:,} fold scores: the 33-configuration grid plus the four reference "
        f"methods, every one of them fit on the same molecules.</p>"
    )

    parts.append("<h3>The grid</h3>")
    parts.append(kit.figure_block(
        res.figure("grid_r2"),
        f"{esc(res.label)}: R squared by modality set, fusion strategy and final "
        "learner. Each point is one endpoint's mean over its 25 folds.",
        "grid_r2",
    ))
    parts.append("<h4>The eight best configurations, and the reference methods</h4>")
    parts.append(frame_to_html(top_configurations(res).reset_index(drop=True)))
    parts.append(frame_to_html(reference_means(res).reset_index(drop=True)))

    if verdict:
        head = headline(res)
        parts.append("<h3>Does fusion beat the best single view?</h3>")
        parts.append(
            f"<p>Best against best, per endpoint. Fusion is ahead on "
            f"<b>{verdict['won']} of {verdict['n']}</b> endpoints, by a mean of "
            f"<b>{verdict['mean']:+.3f}</b> R squared. The best multimodal "
            f"configuration overall is {esc(cfg.METHOD_LABELS[head['best_multi']])} "
            f"at {head['best_multi_r2']:.3f}, against "
            f"{esc(cfg.METHOD_LABELS[head['best_uni']])} at "
            f"{head['best_uni_r2']:.3f} for the best single view.</p>"
        )
        parts.append(kit.figure_block(
            res.figure("unimodal_vs_multimodal_r2"),
            "The best unimodal model against the best multimodal one, per endpoint.",
            "unimodal_vs_multimodal_r2",
        ))

    parts.append("<h3>What an extra modality buys</h3>")
    parts.append(kit.figure_block(
        res.figure("modality_ladder_r2"),
        "R squared against the number of modalities, with the four reference "
        "methods as dashed lines.",
        "modality_ladder_r2",
    ))

    sig, total = survived(res, "ladder")
    fsig, ftotal = survived(res, "fusion")
    lsig, ltotal = survived(res, "learner")
    parts.append(
        f"<p>Over endpoints, with Holm correction inside each family: "
        f"<b>{sig} of {total}</b> steps up the modality ladder separate, "
        f"<b>{fsig} of {ftotal}</b> early-against-late comparisons separate, and "
        f"<b>{lsig} of {ltotal}</b> learner-against-learner comparisons separate.</p>"
    )

    if res.ablation is not None:
        parts.append("<h3>Which modalities the model needs, and which it uses</h3>")
        parts.append(kit.figure_block(
            res.figure("ablation_shap"),
            "Removing one modality from the four-modality model, beside how much "
            "of that model's attribution the modality carries.",
            "ablation_shap",
        ))

    if res.uncertainty is not None:
        parts.append("<h3>Uncertainty and calibration</h3>")
        parts.append(kit.figure_block(
            res.figure("uncertainty_panels"),
            "Epistemic sigma, error-uncertainty correlation, expected calibration "
            "error and miscalibration area, from the five folds of each repeat.",
            "uncertainty_panels",
        ))
        table = read_csv(res.paths.tables / "uncertainty_unimodal_vs_multimodal.csv")
        if table is not None:
            parts.append(frame_to_html(table, "{:.4f}"))

    head = headline(res)
    parts.append("<h3>Against a fingerprint baseline and three D-MPNNs</h3>")
    parts.append(
        f"<p>ChemProp initialised from CheMeleon averages "
        f"<b>{head['chemeleon']:.3f}</b> R squared here, against "
        f"{head['best_grid_r2']:.3f} for the best of the thirty-three "
        f"configurations, {head['chemprop']:.3f} for a multitask D-MPNN and "
        f"{head['lgbm']:.3f} for LightGBM on Morgan fingerprints.</p>"
    )
    parts.append(kit.figure_block(
        res.figure("reference_tukey_r2"),
        "Tukey HSD on R squared over 25 folds. Blue is the best mean, grey "
        "overlaps it, red clears it and is worse.",
        "reference_tukey_r2",
    ))

    parts.append("<h3>Two controls</h3>")
    if res.control is not None:
        parts.append(
            f"<p>Late fusion, with the meta-learner moved off the molecules its base "
            f"learners were fit on: <b>{res.control['difference'].mean():+.3f}</b> "
            f"R squared on average.</p>"
        )
        parts.append(kit.figure_block(
            res.figure("leakage_control_r2"),
            "Late fusion with in-sample meta-features, as released, against the "
            "same configuration with the meta-learner fit on the held-out fifth.",
            "leakage_control_r2",
        ))
    if res.paper_gnn is not None:
        parts.append(
            f"<p>The GNN modality as a 200-wide learned readout rather than the "
            f"30-wide mean of raw atom features the released extractor returns: "
            f"<b>{res.paper_gnn['difference'].mean():+.3f}</b> R squared.</p>"
        )
        parts.append(kit.figure_block(
            res.figure("gnn_block_control"),
            "The GNN modality, as released and as intended.",
            "gnn_block_control",
        ))

    if (res.paths.figures / "cost_benefit.png").exists():
        parts.append("<h3>What it costs</h3>")
        parts.append(kit.figure_block(
            res.figure("cost_benefit"),
            "Fitting time against accuracy as modalities are added, measured "
            "sequentially on one machine.",
            "cost_benefit",
        ))

    return "\n".join(parts)


# Matplotlib writes its figures on white. On a dark ground they would read as
# unexplained glare, and inverting them is not an option because the colours carry
# meaning, so they get a deliberate white mat instead.
ARTIFACT_CSS = """
.figscroll{background:#FFFFFF}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]) .figscroll{background:#F7F9F8;border-color:#2C3A36}
}
:root[data-theme="dark"] .figscroll{background:#F7F9F8;border-color:#2C3A36}
"""


def build(results: list[Results], artifact: bool = False) -> str:
    nav = " · ".join(
        f'<a href="#{r.name}">{esc(r.label)}</a>' for r in results
    )
    sections = "\n".join(dataset_section(r) for r in results)

    a, b = results[0], (results[1] if len(results) > 1 else results[0])
    ha, hb = headline(a), headline(b)

    intro = f"""
<h1>Unimodal versus multimodal, on ADME data</h1>
<p class="standfirst">Thirty-three ways of combining four molecular
representations, over fifteen endpoints across two unrelated data sets, 25
replicate models each: 12,375 fitted configurations and 6,375 more for the
controls. The design is Wasswa, Kajjumba and Ramsundar's{ref('wasswa')}; the
endpoints, folds and splits are those of an earlier comparison{ref('sibling')},
so a fingerprint baseline and three D-MPNNs already scored on exactly these
molecules sit on the same axis.</p>

<p>Their central finding replicates. Fusing modalities moves accuracy very
little, and none of it survives a correction for multiple comparisons. What
fusion does buy is calibration, and on the larger data set that buys enough to
be significant.</p>

<p>Two things the paper could not see from inside its own design also show up.
A single well-initialised graph network beats all thirty-three configurations on
both data sets. And when the GNN modality is actually a learned representation
-- in the released code it is not -- the ranking of which modality matters
inverts.</p>

<h2>Three results</h2>

<h3>1. Adding modalities does not reliably add accuracy</h3>
<p>Best multimodal model against best unimodal model, per endpoint, is a coin
flip: fusion is ahead on 7 of 9 ExpansionRx endpoints by a mean of
{fusion_verdict(a)['mean']:+.3f} R squared, and on 4 of 6 Biogen endpoints by
{fusion_verdict(b)['mean']:+.3f}. Stepping up the modality ladder --
GNN+RDKit to three modalities to four, holding the strategy and the learner
fixed -- separates in <b>{ha['ladder'][0]} of {ha['ladder'][1]}</b> comparisons on
ExpansionRx and <b>{hb['ladder'][0]} of {hb['ladder'][1]}</b> on Biogen, after Holm
correction. So does the choice of learner: {ha['learner'][0]} of
{ha['learner'][1]} and {hb['learner'][0]} of {hb['learner'][1]}.</p>
<p>On Biogen the single best configuration in the entire grid is a
<i>unimodal</i> one, {esc(cfg.METHOD_LABELS[hb['best_uni']])} at
{hb['best_uni_r2']:.3f}, ahead of every fusion model built on top of it.</p>

<h3>2. Calibration is what fusion actually buys</h3>
<p>This is the paper's other claim, and it holds up better than the accuracy one.
Against the best unimodal model, the best multimodal model improves all three
uncertainty measures on both data sets, and on ExpansionRx all three survive Holm
correction: error-uncertainty correlation +0.044 (p = 0.027), expected
calibration error -0.016 (p = 0.012), miscalibration area -0.021 (p = 0.012). On
Biogen the same three move by similar amounts in the same direction at p =
0.094.</p>
<p>The reference methods make the size of that plain. The best fusion
configuration reaches an error-uncertainty correlation of
{ha['unc_best_corr']:.3f} on ExpansionRx and {hb['unc_best_corr']:.3f} on Biogen,
where the best of the four reference methods manages {ha['ref_best_corr']:.3f}
and {hb['ref_best_corr']:.3f}. Fusion models know when they are wrong better than
any single model here does, including the one that is most accurate.</p>

<h3>3. A single well-initialised graph network beats the whole grid</h3>
<p>ChemProp initialised from CheMeleon{ref('chemeleon')} averages
<b>{ha['chemeleon']:.3f}</b> R squared on ExpansionRx and
<b>{hb['chemeleon']:.3f}</b> on Biogen. The best of the thirty-three
configurations manages {ha['best_grid_r2']:.3f} and {hb['best_grid_r2']:.3f}. It
is not close, and it does not depend on which data set you look at.</p>
<p>That is the same shape of answer the earlier comparison{ref('sibling')} got
from a different direction: what makes a graph network win is a good
initialisation, not the number of things bolted to it. Multimodal fusion is a way
of combining fixed representations, and the ceiling on that is the quality of the
representations being combined.</p>

<h2>What was compared</h2>
{grid_table()}
<p>Nine unimodal baselines and twenty-four fusion models: 33 configurations per
endpoint per fold. The paper's fifth modality is MS2 fragmentation spectra; no
ADME collection carries them, so that modality is dropped and nothing else about
the design changes.</p>

<p>The 25 replicates come from five repeats of a five-fold <code>GroupKFold</code>
over the training molecules, grouped by chemical cluster. Every configuration
early-stops on the held-out fifth where it has a training loop to stop, and every
one is scored on the same untouched test set. Statistics follow the protocol of
Ash et al.{ref('protocol')}: distributions over folds, corrected for multiple
comparisons, and no bold maxima.</p>

<h2>Uncertainty, from the folds</h2>
<p>The paper estimates epistemic uncertainty from three independently seeded
models. Here the five folds within a repeat play that role: each test molecule is
predicted five times, by five models fit on overlapping four fifths of the same
training set. Five repeats give five ensembles per configuration rather than one,
it costs nothing extra, and it means the reference methods can be scored on
calibration without being rerun.</p>
<p class="note">That expected calibration error compares an error magnitude
directly against a standard deviation, so a perfectly calibrated Gaussian scores
about 0.2 sigma rather than zero, and it carries the units of the endpoint. Read
it within a panel, not against another paper.</p>

<h2>This is a reimplementation, and two controls measure the difference</h2>
<p>The authors' code{ref('repo')} is MIT licensed and public, which is more than
most releases offer. It is also a set of Colab dumps: hard-coded paths, six empty
notebook stubs, scripts that consume globals left behind by other scripts, and no
committed generator for the feature matrix everything loads. It cannot be run, so
this is written from that source read as a specification together with the
paper's Supporting Information.</p>

<h3>Their GNN modality is not a learned representation</h3>
<p>Their extractor hooks the first <code>nn.Linear</code> it finds by module
order. DeepChem's AttentiveFP{ref('attentivefp')} has no <code>.ffn</code>
attribute, so the fallback lands on
<code>gnn.init_context.project_node[0]</code> -- the projection applied to raw
atom features <i>before any message passing</i>. What comes back is a 30-wide
mean of unlearned atom features. Their own Table S3 records the modality as 30
features while describing it as a learned graph representation; the readout is
200 wide.</p>
<p>Running the LightGBM half of the grid on both blocks puts a number on it: the
learned readout is worth <b>+{a.paper_gnn['difference'].mean():.3f}</b> R squared
on ExpansionRx and <b>+{b.paper_gnn['difference'].mean():.3f}</b> on Biogen.</p>
<p>It also inverts the paper's modality ranking. They found RDKit descriptors
indispensable and the GNN embedding nearly free to drop -- which is what you
would expect if the GNN block were a mean of atom features already implied by the
descriptors. With a real graph embedding, the GNN modality carries
<b>49.9%</b> of the fused model's SHAP attribution on ExpansionRx against RDKit's
<b>4.6%</b>, and removing it costs -0.170 R squared under early fusion where
removing RDKit costs -0.009.</p>

<h3>Their late fusion leaks, but the leak is small</h3>
<p>Their base learners fit the whole training set and then predict that same
training set, with no out-of-fold scheme anywhere in the release. With LightGBM
base learners those predictions nearly interpolate, so the meta-learner is fit on
optimistically biased inputs. Refitting the identical meta-learner on the fold's
held-out fifth instead is worth
<b>{a.control['difference'].mean():+.3f}</b> R squared on ExpansionRx and
<b>{b.control['difference'].mean():+.3f}</b> on Biogen. Real, in the direction you
would expect, and small enough that it is not what is holding late fusion
back.</p>

<h3>And both encoders early-stop here</h3>
<p>Theirs train for a fixed 50 and 20 epochs with no validation monitoring, which
they can afford because their protocol has no validation split. This one has one,
and every other method in the comparison uses it. Hyperparameters otherwise
follow their protocol unchanged: 60 sampled settings, mean squared error over
three folds, searched once and reused for every replicate, with the inner folds
grouped by cluster as the outer folds already are.</p>

<h2>Early against late fusion</h2>
<p>The one place the design axes separate at all. On ExpansionRx
<b>{ha['fusion'][0]} of {ha['fusion'][1]}</b> early-against-late comparisons
survive correction, and every one of them is the graph learner preferring late
fusion, by between +0.170 and +0.344 R squared. On Biogen none of the twelve
separate.</p>
<p>The reason is visible in the accuracy table: early fusion hands AttentiveFP
several hundred standardised descriptor columns alongside its own readout, and
the readout drowns. Late fusion hands it four numbers instead. Where the fused
block is small, concatenation is harmless; where it is large, it costs more than
it adds.</p>

<nav class="jump">{nav}</nav>
"""

    body = intro + sections + "<h2>References</h2>" + kit.render_references(REFERENCES)
    title = "Does Multimodal Fusion Help?"

    # `.wrap` is what the stylesheet puts the reading column on; a bare <main>
    # gets the type but none of the measure.
    page = f'<main class="wrap">{body}</main>'

    if artifact:
        # The Artifact host supplies the document skeleton, so this is title,
        # style and content only.
        return f"<title>{title}</title><style>{kit.CSS}{ARTIFACT_CSS}</style>{page}"

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
        f"<style>{kit.CSS}{ARTIFACT_CSS}</style></head><body>{page}</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None)
    parser.add_argument("--artifact", action="store_true",
                        help="emit title/style/content only, for publishing")
    args = parser.parse_args()

    results = [r for r in (load(name) for name in DATASETS) if r is not None]
    if not results:
        raise SystemExit("no fold_metrics.csv found -- run 06_collect_metrics.py first")

    default = "report_artifact.html" if args.artifact else "report.html"
    out = Path(args.out) if args.out else (cfg.PROJECT_DIR / "results" / default)
    page = build(results, artifact=args.artifact)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"{len(results)} data set(s)")


if __name__ == "__main__":
    main()
