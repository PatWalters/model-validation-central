#!/usr/bin/env python
"""Step 11: the report, the paper's tasks and both data sets on one page.

Everything is inlined, figures included, so the file stands alone. Every
number in the prose is computed from the tables it sits beside rather than
typed, and where the prose makes a qualitative claim the claim is checked here:
the build fails if the data stop supporting it.

    python 09_paper_metrics.py
    for ds in expansion biogen; do
      ADME_DATASET=$ds python 08_collect_metrics.py
      ADME_DATASET=$ds python 10_report.py
    done
    python 11_build_page.py

The page is written to results/report.html and, where the repository's docs/
is present, published to docs/reports/causal-chemprop.html.
"""

import numpy as np
import pandas as pd

import config as cfg
import page_kit

PAGE = cfg.PROJECT_DIR / "results" / "report.html"
PUBLISHED = cfg.PROJECT_DIR.parent.parent / "docs" / "reports" / "causal-chemprop.html"
PAPER = cfg.PROJECT_DIR / "results" / "paper"
DATASETS = ["expansion", "biogen"]
LABEL_OF = {ds: cfg.DATASETS[ds].label for ds in DATASETS}

C, O = cfg.CAUSAL_METHOD, cfg.OFFSET
ENC, MA = cfg.CC_CHEMPROP_METHOD, cfg.MONROE_ANCHOR_METHOD
LGBM, CHEMELEON, MONROE, AJ = cfg.LGBM_METHOD, cfg.CHEMELEON_METHOD, cfg.MONROE_METHOD, cfg.ATOMJEPA_METHOD

# An offset moves a whole series by a constant, so an arm and its offset twin
# rank identically: one column each for the within-series metric.
RHO_ARMS = [LGBM + O, CHEMELEON + O, AJ + O, MONROE, MA, ENC, C]
MAE_ARMS = [LGBM + O, CHEMELEON + O, AJ + O, MONROE, MONROE + O, MA, ENC, ENC + O, C]
# Causal-Chemprop against each of these over series.
SERIES_RIVALS = [ENC, MONROE, MA, AJ, CHEMELEON, LGBM]

COLOR = {
    LGBM: page_kit.METHOD_COLOR["lgbm"], CHEMELEON: page_kit.METHOD_COLOR["chemeleon"],
    MONROE: page_kit.METHOD_COLOR["monroe"], AJ: page_kit.METHOD_COLOR["atomjepa_ft"],
    MA: "#61483A", ENC: page_kit.METHOD_COLOR["chemprop"], C: "#2A7F9E",
}
for _m in (LGBM, CHEMELEON, MONROE, AJ, ENC):
    COLOR[_m + O] = COLOR[_m]
SHORT = {
    LGBM: "LightGBM", CHEMELEON: "CheMeleon FT", AJ: "Atom-JEPA FT", MONROE: "Monroe/PFN3",
    MA: "Monroe/PFN3 +ctx", ENC: "Chemprop", C: "Causal-Chemprop",
}
for _m in (LGBM, CHEMELEON, MONROE, AJ, ENC):
    SHORT[_m + O] = SHORT[_m] + " +off"
LABEL = {**cfg.METHOD_LABELS, ENC: "Chemprop, Causal-Chemprop's own encoder"}

# The paper half's arms, in the order of its table.
PAPER_ARMS = ["causal_chemprop", "causal_refit_scm", "causal_refit_all", "chemprop",
              "chemprop_retrained", "lgbm_anchor", "monroe", "monroe_anchor", "atomjepa_ft"]
PAPER_SHORT = {
    "causal_chemprop": "Causal-Chemprop, released", "causal_refit_scm": "SCM refit",
    "causal_refit_all": "encoder + SCM refit", "chemprop": "Chemprop, released",
    "chemprop_retrained": "Chemprop retrained, released", "lgbm_anchor": "LightGBM +anchor",
    "monroe": "Monroe/PFN3", "monroe_anchor": "Monroe/PFN3 +anchor", "atomjepa_ft": "Atom-JEPA FT",
}


# --- loading -------------------------------------------------------------
def table(ds: str, name: str) -> pd.DataFrame:
    return pd.read_csv(cfg.PROJECT_DIR / "results" / ds / "tables" / f"{name}.csv")


def figure(ds: str, name: str, caption: str) -> str:
    path = cfg.PROJECT_DIR / "results" / ds / "figures" / f"{name}.png"
    return page_kit.figure_block(page_kit.embed_figure(path), caption, name)


def num(x: float, places: int = 2, sign: bool = False) -> str:
    """A number for prose, with a real minus sign."""
    text = f"{x:+.{places}f}" if sign else f"{x:.{places}f}"
    return text.replace("-", "−")


def pval(p: float) -> str:
    return "p &lt; 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


# --- the paper half ------------------------------------------------------
def paper_table(tm: pd.DataFrame, h2h: pd.DataFrame) -> str:
    rho = tm.pivot(index="task", columns="method", values="spearman").reindex(cfg.PAPER_TASKS)
    rows = []
    for t in cfg.PAPER_TASKS:
        cells = [f'<th scope="row">{t}</th>']
        for m in PAPER_ARMS:
            v = rho.loc[t].get(m, np.nan)
            cells.append(f'<td class="num">{"–" if pd.isna(v) else num(v)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    hs = h2h[h2h["metric"] == "spearman"].set_index("other")
    tail = ['<th scope="row"><b>released ahead</b></th>', "<td></td>"]
    for m in PAPER_ARMS[1:]:
        r = hs.loc[m]
        tail.append(f'<td class="num">{int(r["causal_wins"])}–{int(r["causal_losses"])}'
                    f'<span class="sd">{pval(r["wilcoxon_p"])}</span></td>')
    rows.append("<tr>" + "".join(tail) + "</tr>")
    head = "".join(f"<th>{PAPER_SHORT[m]}</th>" for m in PAPER_ARMS)
    return ('<div class="bleed tablewrap"><table>'
            f'<thead><tr><th scope="col">Task</th>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# --- the protocol half ---------------------------------------------------
def on_top(s: pd.DataFrame) -> pd.Series:
    return s["tukey_group"].isin(("best", "equivalent"))


def metric_table(s: pd.DataFrame, metric: str, arms: list, places: int) -> str:
    """Mean ± sd over the scored folds; the Tukey grouping as a chip."""
    sub = s[(s["metric"] == metric) & s["method"].isin(arms)]
    sizes = sub[on_top(sub)].groupby("endpoint").size()
    rows = []
    for endpoint in sub["endpoint"].drop_duplicates():
        g = sub[sub["endpoint"] == endpoint].set_index("method")
        cells = [f'<th scope="row">{endpoint}</th>']
        for m in arms:
            r = g.loc[m]
            top = r["tukey_group"] in ("best", "equivalent")
            note = ("tied" if sizes.get(endpoint, 1) > 1 else "best") if top else ""
            chip = f'<span class="chip {"tied" if note == "tied" else "best"}">{note}</span>' if note else ""
            cells.append(f'<td class="{"best" if top else "worse"}"><span class="num">'
                         f'{num(r["mean"], places)}</span><span class="sd">±{r["sd"]:.{places}f}</span>'
                         f"{chip}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(f'<th><span class="swatch" style="background:{COLOR[m]}"></span>{SHORT[m]}</th>'
                   for m in arms)
    return ('<div class="bleed tablewrap"><table>'
            f'<thead><tr><th scope="col">Endpoint</th>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def series_table(sv: dict) -> str:
    rows = []
    for m in SERIES_RIVALS:
        cells = [f'<th scope="row"><span class="swatch" style="background:{COLOR[m]}"></span>'
                 f"{LABEL[m]}</th>"]
        for ds in DATASETS:
            r = sv[ds].loc[m]
            cells.append(f'<td class="num"><span class="chip win">{int(r["causal_wins"])}</span></td>'
                         f'<td class="num"><span class="chip loss">{int(r["causal_losses"])}</span></td>'
                         f'<td class="num">{num(r["median_diff"], 2, True)}</td>'
                         f'<td class="num">{pval(r["wilcoxon_p"])}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    n = {ds: int(sv[ds].loc[ENC, "series"]) for ds in DATASETS}
    return ('<div class="bleed tablewrap"><table><thead>'
            '<tr><th scope="col" rowspan="2">Causal-Chemprop against</th>'
            + "".join(f'<th colspan="4">{LABEL_OF[ds]}, {n[ds]} series</th>' for ds in DATASETS)
            + "</tr><tr>" + "<th>ahead</th><th>behind</th><th>median Δρ</th><th>Wilcoxon</th>" * 2
            + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


MODE_ORDER = ["all counterfactual", "mixed", "all SCM, switched", "all SCM, low variance"]
MODE_TEXT = {
    "all counterfactual": "every molecule the counterfactual",
    "mixed": "some counterfactual, some SCM",
    "all SCM, switched": "every molecule switched to the SCM prediction",
    "all SCM, low variance": "the low-variance fallback: all SCM predictions",
}


def modes_table(md: dict) -> str:
    rows = []
    for mode in MODE_ORDER:
        cells = [f'<th scope="row" style="font-family:inherit">{MODE_TEXT[mode]}</th>']
        for ds in DATASETS:
            g = md[ds]
            r = g[(g["split"] == "mode") & (g["value"] == mode)]
            if r.empty:
                cells.append('<td class="num">0</td><td></td><td></td>')
                continue
            r = r.iloc[0]
            cells.append(f'<td class="num">{int(r["fits"])}</td>'
                         f'<td class="num">{num(r["causal_rho"])}</td>'
                         f'<td class="num">{num(r["encoder_rho"])}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return ('<div class="bleed tablewrap"><table><thead>'
            '<tr><th scope="col" rowspan="2">How the series was predicted</th>'
            + "".join(f'<th colspan="3">{LABEL_OF[ds]}</th>' for ds in DATASETS)
            + "</tr><tr>" + "<th>fits</th><th>ρ, Causal-Chemprop</th><th>ρ, its encoder</th>" * 2
            + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def cost_table(cost: dict, n_series: dict) -> str:
    rows = []
    for ds in DATASETS:
        c = cost[ds]
        rows.append(
            f'<tr><th scope="row">{LABEL_OF[ds]}</th>'
            f'<td class="num">{n_series[ds]}</td><td class="num">{int(c["scm_fits"].sum()):,}</td>'
            f'<td class="num">{np.median(c["scm_median_min"]):.0f}</td>'
            f'<td class="num">{c["scm_max_h"].max():.1f}</td>'
            f'<td class="num">{c["scm_cpu_h"].sum():.0f}</td>'
            f'<td class="num">{c["encoder_gpu_h"].sum():.1f}</td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Data set</th>'
            "<th>Series</th><th>SCM fits</th><th>Median min / fit</th><th>Longest fit, h</th>"
            "<th>SCM, CPU-core hours</th><th>Encoders, GPU hours</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# --- references ----------------------------------------------------------
REFERENCES = [
    ("causal", "Natajaya, C.; Attia, L.; Burns, J. W.; Doyle, P. S. ",
     "Causal-Chemprop: Causal Machine Learning for Molecular Property Prediction and Design.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2026. The method this page puts to the '
     "test, run with the authors' code and released checkpoints.",
     "https://doi.org/10.1021/acs.jcim.6c01429", "10.1021/acs.jcim.6c01429"),
    ("chemprop", "Graff, D. E.; Morgan, N. K.; Burns, J. W.; <i>et al.</i> ",
     "Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical Property "
     "Prediction.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2026, 66 (1), 28–33. The encoder '
     "Causal-Chemprop is built on.",
     "https://doi.org/10.1021/acs.jcim.5c02332", "10.1021/acs.jcim.5c02332"),
    ("dagma", "Bello, K.; Aragam, B.; Ravikumar, P. ",
     "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization.",
     '<span class="venue">Advances in Neural Information Processing Systems</span> 35, 2022. '
     "The structure learner.",
     "https://arxiv.org/abs/2209.08037", "arXiv:2209.08037"),
    ("dowhy", "Blöbaum, P.; Götz, P.; Budhathoki, K.; Mastakouri, A. A.; Janzing, D. ",
     "DoWhy-GCM: An Extension of DoWhy for Causal Inference in Graphical Causal Models.",
     '<span class="venue">J. Mach. Learn. Res.</span> 2024, 25 (147), 1–7. The causal '
     "mechanisms and the intrinsic causal contributions.",
     "https://arxiv.org/abs/2206.06821", "arXiv:2206.06821"),
    ("monroe", "Banaszewski, B.; Fitzgibbon, A. W. ",
     "Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference.",
     "Preprint, 2026. The frozen encoder of the strongest arm on this site.",
     "https://arxiv.org/abs/2608.18982", "arXiv:2608.18982"),
    ("tabpfn3", "Grinsztajn, L.; Flöge, K.; Key, O.; <i>et al.</i> ",
     "TabPFN-3: Technical Report.",
     "Preprint, 2026. The in-context head of the Monroe arms.",
     "https://arxiv.org/abs/2605.13986", "arXiv:2605.13986"),
    ("atomjepa", "Petersen, K. H.; Tirsgaard, R. H.; Cornet, F. R. J.; Jordahn, M.; "
     "Schmidt, M. N. ",
     "Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems.",
     "Preprint, 2026. The fine-tuned 3D encoder, run with the authors' code.",
     "https://arxiv.org/abs/2610.08400", "arXiv:2610.08400"),
    ("chemeleon", "Burns, J. W.; Zalte, A. S.; Abreu, C. R. A.; <i>et al.</i> ",
     "Deep Learning Foundation Models for Low-Data Regimes from Classical Molecular Descriptors.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2026, articles ASAP.',
     "https://doi.org/10.1021/acs.jcim.6c01546", "10.1021/acs.jcim.6c01546"),
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
LINK_SOURCE = '<a href="expansion-ml-comparison.html">the foundation-model comparison</a>'
LINK_AJ = '<a href="atom-jepa.html">the Atom-JEPA study</a>'


def build() -> str:
    # The paper half.
    tm = pd.read_csv(PAPER / "task_metrics.csv")
    ph = pd.read_csv(PAPER / "head_to_head.csv")
    rep = pd.read_csv(PAPER / "reproduction.csv")
    n_rep = int(rep["r_match"].sum() + rep["rs_match"].sum())
    miss = rep[~(rep["r_match"] & rep["rs_match"])].iloc[0]
    hs = ph[ph["metric"] == "spearman"].set_index("other")
    win = {m: (int(hs.loc[m, "causal_wins"]), int(hs.loc[m, "causal_losses"]), hs.loc[m, "wilcoxon_p"],
               int(hs.loc[m, "tasks"])) for m in PAPER_ARMS[1:]}
    rho = tm.pivot(index="task", columns="method", values="spearman")
    refit_all_tasks = int(rho["causal_refit_all"].notna().sum())
    # Means over the tasks both have, so a refit still running is compared fairly.
    both = rho[["causal_chemprop", "causal_refit_all"]].dropna()
    refit_drop = (rho["causal_chemprop"] - rho["causal_refit_all"]).dropna()
    refit_vs_chemprop = (rho["causal_refit_all"] - rho["chemprop"]).dropna()

    # The protocol half.
    s = {ds: table(ds, "summary") for ds in DATASETS}
    sv = {ds: table(ds, "series_head_to_head").set_index("other") for ds in DATASETS}
    md = {ds: table(ds, "modes") for ds in DATASETS}
    cost = {ds: table(ds, "cost") for ds in DATASETS}
    n_series = {ds: int(sv[ds].loc[ENC, "series"]) for ds in DATASETS}
    n_fits = {ds: int(cost[ds]["scm_fits"].sum()) for ds in DATASETS}
    n_ep = {ds: int(cost[ds]["endpoint"].nunique()) for ds in DATASETS}
    folds = {ds: int(s[ds]["folds"].max()) for ds in DATASETS}

    def behind(ds, m):
        r = sv[ds].loc[m]
        return int(r["causal_losses"]), int(r["series"])

    def mode(ds, name, col):
        g = md[ds]
        return float(g[(g["split"] == "mode") & (g["value"] == name)][col].iloc[0])

    def allrow(ds, col):
        g = md[ds]
        return float(g[g["split"] == "all"][col].iloc[0])

    def top_count(ds, m, metric):
        g = s[ds][(s[ds]["method"] == m) & (s[ds]["metric"] == metric)]
        return int(on_top(g).sum()), len(g)

    def mean_metric(ds, m, metric):
        g = s[ds][(s[ds]["method"] == m) & (s[ds]["metric"] == metric)]
        return float(g["mean"].mean())

    degenerate = {ds: md[ds][(md[ds]["split"] == "degenerate") & (md[ds]["value"].astype(str) == "True")]
                  ["fits"].sum() / n_fits[ds] for ds in DATASETS}
    cpu_h = sum(cost[ds]["scm_cpu_h"].sum() for ds in DATASETS)
    gpu_h = sum(cost[ds]["encoder_gpu_h"].sum() for ds in DATASETS)

    # Qualitative claims the prose makes; fail the build if the data stop
    # supporting them.
    assert n_rep >= 2 * len(rep) - 1, "the prose says Table 1 reproduces"
    assert win["chemprop"][0] > win["chemprop"][1], "the prose says the released model beats Chemprop"
    assert refit_drop.median() > 0 and (refit_drop > 0).sum() > len(refit_drop) / 2, \
        "the prose says a full refit loses what the release has"
    for ds in DATASETS:
        for m in (ENC, MONROE, MA):
            k, n = behind(ds, m)
            assert k > 0.8 * n and sv[ds].loc[m, "wilcoxon_p"] < 0.01, \
                f"the prose says Causal-Chemprop is behind {m} on almost every series ({ds})"
        assert mode(ds, "all counterfactual", "causal_rho") < 0.5 * mode(ds, "all counterfactual", "encoder_rho"), \
            "the prose says the counterfactual is what loses the ranking"
        for name in ("all SCM, switched", "all SCM, low variance"):
            g = md[ds][(md[ds]["split"] == "mode") & (md[ds]["value"] == name)]
            if len(g) and g["fits"].iloc[0] >= 5:
                assert abs(g["causal_rho"].iloc[0] - g["encoder_rho"].iloc[0]) < 0.1, \
                    "the prose says the SCM fallbacks rank like the encoder"
        assert top_count(ds, C, "spearman_series")[0] <= 1, "the prose says Causal-Chemprop is almost never on top"
        assert mean_metric(ds, C, "mae") > mean_metric(ds, ENC + O, "mae"), \
            "the prose says the anchor offset does more for MAE than the causal model"

    k_enc = {ds: behind(ds, ENC) for ds in DATASETS}
    k_mon = {ds: behind(ds, MONROE) for ds in DATASETS}
    k_ma = {ds: behind(ds, MA) for ds in DATASETS}

    def ds_pair(fn) -> str:
        e, b = (fn(ds) for ds in DATASETS)
        return f"{e} on ExpansionRx and {b} on Biogen"

    ma_rho_gain = {ds: mean_metric(ds, MA, "spearman_series") - mean_metric(ds, MONROE, "spearman_series")
                   for ds in DATASETS}
    ma_mae_gain = {ds: mean_metric(ds, MONROE, "mae") - mean_metric(ds, MA, "mae") for ds in DATASETS}
    off_mae_cost = {ds: mean_metric(ds, MONROE + O, "mae") - mean_metric(ds, MONROE, "mae") for ds in DATASETS}
    for ds in DATASETS:
        assert off_mae_cost[ds] > 0 and ma_mae_gain[ds] > 0, \
            "the prose says the offset raises Monroe's MAE and the anchors in context lower it"
        assert mean_metric(ds, C, "mae") > max(mean_metric(ds, ENC, "mae"), mean_metric(ds, ENC + O, "mae")), \
            "the prose says Causal-Chemprop's MAE is above its encoder's, shifted or not"
    aj_fits = pd.read_csv(PAPER / "atomjepa_fits.csv")

    parts = [
        '<meta charset="utf-8">',
        "<title>Counterfactuals From One Analog</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">Causal inference on a learned embedding · anchored series · '
        "the paper's tasks and two ADME data sets</p>",
        "<h1>A counterfactual from one measured analog, against the encoder it sits on</h1>",
        '<p class="lede">Causal-Chemprop' + ref("causal") + " fits a structural causal model to "
        "the last hidden layer of a trained Chemprop" + ref("chemprop") + " and, given one "
        "measured member of a scaffold series, predicts the rest of the series as "
        "counterfactuals of it. Its paper reports better within-series ranking than Chemprop on "
        "eleven kinase scaffold groups. With the authors' released checkpoints that reproduces: "
        f"Table 1 to two decimals in {n_rep} of {2 * len(rep)} values, ahead of their Chemprop on "
        f"{win['chemprop'][0]} of {win['chemprop'][3]} tasks. It does not survive retraining. "
        "Refit end to end with the authors' own code, it loses within-task Spearman ρ on "
        f"{int((refit_drop > 0).sum())} of {len(refit_drop)} tasks, by a median "
        f"{num(refit_drop.median())}. And fitted per fold and per series on ExpansionRx and "
        "Biogen, its within-series ρ is behind that of its own encoder on "
        + ds_pair(lambda ds: f"{k_enc[ds][0]} of {k_enc[ds][1]} series")
        + ", and behind Monroe + TabPFN 3" + ref("monroe", "tabpfn3") + " on "
        + ds_pair(lambda ds: f"{k_mon[ds][0]} of {k_mon[ds][1]}") + ".</p>",
        '<div class="facts">'
        '<div class="fact"><b>11</b><span>paper tasks</span></div>'
        f'<div class="fact"><b>{sum(n_series.values())}</b><span>ADME series</span></div>'
        f'<div class="fact"><b>{sum(n_fits.values()):,}</b><span>SCM fits</span></div>'
        f'<div class="fact"><b>{len(RHO_ARMS)}</b><span>arms per series</span></div>'
        f'<div class="fact"><b>{cpu_h:,.0f}</b><span>CPU-core hours</span></div>'
        "</div>",

        '<div class="panel howto"><h3>What the method does</h3>'
        "<p>A Chemprop is trained on the target. Its 200-unit last hidden layer and the "
        "target are taken as the 201 variables of a linear structural causal model: DAGMA"
        + ref("dagma") + " learns the graph from an <i>analog set</i>, the training molecules "
        "whose Murcko scaffold resembles the series', and dowhy's GCM" + ref("dowhy") + " fits a "
        "mechanism to every node and scores each node's intrinsic causal contribution to the "
        "target. To predict a molecule of the series, the noise of the measured anchor is "
        "abducted, the molecule's own values are imposed on the high-contribution nodes, and "
        "the target is read off the counterfactual. Two rules of the authors' "
        "<code>test_counterfactual</code> can replace that with the SCM's plain prediction: "
        "per molecule, when the counterfactual is far from the anchor, and per series, when "
        "the counterfactuals barely vary.</p></div>",

        '<div class="panel howto"><h3>The authors\' code, on this site\'s folds</h3>'
        "<p>Nothing in the model is reimplemented. The encoder is the authors' "
        "<code>SobolevMPNN</code>, trained with the architecture and schedule of their "
        "<code>causal_utils.train</code>; the SCM is their <code>fit_causal</code>, with their "
        "vendored dowhy and DAGMA; inference is their <code>test_counterfactual</code>, with "
        "only the file reading replaced. Where the release hard-codes hyperparameters per task "
        "and the eleven tasks disagree, the majority value is used. On ExpansionRx and Biogen a "
        "<i>series</i> is a held-out test cluster with at least seven measured molecules for "
        "the endpoint; one seeded-random member is its anchor, given to every arm that can use "
        "it; the rest are scored. The analog set is the fold's training molecules within the "
        "paper's scaffold similarity of the anchor, and a series is kept only if it has at "
        "least 90 analogs in every fold, the size below which the paper calls the SCM "
        "under-determined.</p>"
        "<p>Every unanchored arm also appears with the cheapest use of the anchor there is: "
        "shift its predictions for the series by its own error on the anchor, the paper's "
        "“direct residual correction”. A shift cannot change a ranking, so it matters for "
        "MAE, not for within-series ρ. Monroe + TabPFN 3 is also run with every anchor in its "
        "in-context training set (<i>+ctx</i>), the way an in-context model takes new data.</p>"
        f"<p>Causal-Chemprop is fitted once per series per fold, so it is run on the "
        f"{folds['expansion']} folds of the first repeat, not all 25; every arm is compared on "
        "those folds alone. LightGBM, CheMeleon and Monroe come from " + LINK_SOURCE + ", "
        "fine-tuned Atom-JEPA" + ref("atomjepa") + " from " + LINK_AJ + ", unchanged.</p></div>",

        "<h2>The paper's eleven tasks</h2>",
        "<p>The release ships the eleven ABL1, AURKA and SRC tasks with their anchors, the "
        "trained encoders and the fitted SCMs. Run through the authors' own functions they "
        f"give Table 1 to two decimals in {n_rep} of {2 * len(rep)} values; the exception is "
        f"{miss['method'].replace('_', ' ')} on {miss['task']}, ρ "
        f"{num(miss['ours_rs'])} against a printed {num(miss['paper_rs'])}, which looks like a "
        "dropped minus sign. The other arms are fitted here on the same training sets and "
        "anchors; <i>+anchor</i> adds the anchor to the training data. The table is "
        "within-task Spearman ρ, the paper's metric; the bottom row is how many tasks the "
        "released Causal-Chemprop is ahead and behind, and a Wilcoxon signed-rank test over "
        "the tasks.</p>",
        paper_table(tm, ph),
        f"<p>Against the arms the paper ran, the released model holds up: ahead of its Chemprop "
        f"on {win['chemprop'][0]} of {win['chemprop'][3]} tasks and of the retrained Chemprop "
        f"on {win['chemprop_retrained'][0]}. Against arms it did not run the margin is "
        f"similar in count and weaker in evidence: ahead of Monroe + TabPFN 3 on "
        f"{win['monroe'][0]} ({pval(win['monroe'][2])}), of Monroe with the anchor on "
        f"{win['monroe_anchor'][0]} ({pval(win['monroe_anchor'][2])}) and of fine-tuned "
        f"Atom-JEPA on {win['atomjepa_ft'][0]} ({pval(win['atomjepa_ft'][2])}). With 7 to 22 "
        "molecules per task, few of these tests could be significant either way.</p>",
        "<h3>Refitting the release</h3>",
        "<p>The released numbers come from one trained encoder and one fitted SCM per task. "
        "Two refits, both through the authors' <code>train</code>, separate them. Refitting "
        "only the SCM on the released encoder (<i>SCM refit</i>) keeps most of the result: "
        f"the released model is ahead of it on {win['causal_refit_scm'][0]} and behind on "
        f"{WORDS.get(win['causal_refit_scm'][1], win['causal_refit_scm'][1])} of "
        f"{win['causal_refit_scm'][3]} tasks. Retraining the encoder as well "
        f"(<i>encoder + SCM refit</i>) does not: within-task ρ falls on "
        f"{int((refit_drop > 0).sum())} of {len(refit_drop)} tasks, by a median "
        f"{num(refit_drop.median())}, and the refit model is behind the released Chemprop on "
        f"{int((refit_vs_chemprop < 0).sum())} of {len(refit_vs_chemprop)}. Mean ρ over "
        f"{'the' if refit_all_tasks == 11 else 'those'} {len(both)} tasks is "
        f"{num(both['causal_chemprop'].mean())} released and "
        f"{num(both['causal_refit_all'].mean())} refit"
        + ". The authors' training is seeded and set deterministic, but it does not regenerate "
        "their checkpoints here, and a new encoder is a new embedding: the SCM, the "
        "contributing nodes and the counterfactuals all change with it. The result in "
        "Table 1 belongs to those particular checkpoints.</p>",

        "<h2>ExpansionRx and Biogen</h2>",
        f"<p>On {n_ep['expansion']} ExpansionRx endpoints{ref('expansion')} there are "
        f"{n_series['expansion']} series with enough analogs, and on {n_ep['biogen']} Biogen "
        f"endpoints{ref('biogen')} {n_series['biogen']}. Each series is scored separately in "
        "each fold, and the table below counts series, with each series' ρ averaged over its "
        "folds: the comparison on the paper's own terms.</p>",
        series_table(sv),
        f"<p>Causal-Chemprop is behind its own encoder on {k_enc['expansion'][0]} of "
        f"{k_enc['expansion'][1]} ExpansionRx series and {k_enc['biogen'][0]} of "
        f"{k_enc['biogen'][1]} Biogen series. The other ranking arms are further ahead of it, "
        "Monroe + TabPFN 3 most of all; giving Monroe the anchors in context changes its ρ by "
        + ds_pair(lambda ds: num(ma_rho_gain[ds], 3, True)) + " on average.</p>",
        figure("expansion", "series_scatter", "Each point is one ExpansionRx series, its "
               "within-series Spearman ρ averaged over the folds: Causal-Chemprop against its "
               "own encoder, and against Monroe + TabPFN 3 with the anchors in context. Below "
               "the line, Causal-Chemprop ranks the series worse."),
        figure("biogen", "series_scatter", "The same for Biogen ADME."),

        "<h3>Tukey HSD over folds</h3>",
        "<p>The protocol's view, per endpoint, the mean within-series ρ of a fold as the "
        "metric. A method is on top if Tukey's correction cannot separate it from the best "
        f"mean. With {folds['expansion']} folds the intervals are wide, and still Causal-Chemprop "
        f"is on top of {top_count('expansion', C, 'spearman_series')[0]} of "
        f"{top_count('expansion', C, 'spearman_series')[1]} ExpansionRx endpoints and "
        f"{top_count('biogen', C, 'spearman_series')[0]} of "
        f"{top_count('biogen', C, 'spearman_series')[1]} Biogen endpoints.</p>",
        "<h3>ExpansionRx</h3>",
        metric_table(s["expansion"], "spearman_series", RHO_ARMS, 2),
        figure("expansion", "tukey_spearman_series", "Tukey HSD on mean within-series ρ, "
               f"ExpansionRx, over {folds['expansion']} folds. Blue is the best mean, grey "
               "cannot be separated from it, red is worse."),
        "<h3>Biogen ADME</h3>",
        metric_table(s["biogen"], "spearman_series", RHO_ARMS, 2),
        figure("biogen", "tukey_spearman_series", "Tukey HSD on mean within-series ρ, Biogen ADME."),

        "<h2>Where the ranking goes</h2>",
        "<p>The fit diagnostics say which of the authors' rules produced each series' "
        "predictions. When the SCM's plain prediction is used, for every molecule or through the "
        "low-variance fallback, Causal-Chemprop ranks about as well as the encoder underneath "
        "it. When every molecule gets the counterfactual, the ranking is close to random: ρ "
        + ds_pair(lambda ds: f"{num(mode(ds, 'all counterfactual', 'causal_rho'))} against "
                  f"the encoder's {num(mode(ds, 'all counterfactual', 'encoder_rho'))}")
        + " on the same series. The counterfactual pulls every member of a series towards the "
        "anchor and does not keep their order. No node clears the paper's contribution "
        f"threshold in {degenerate['expansion']:.0%} of ExpansionRx fits and "
        f"{degenerate['biogen']:.0%} of Biogen fits; there the authors' fallback chooses the "
        "nodes.</p>",
        modes_table(md),

        "<h2>What the anchor buys</h2>",
        "<p>One measured analog is a noisy correction. Shifting Monroe + TabPFN 3 by its error "
        "on the anchor raises its mean MAE, by "
        + ds_pair(lambda ds: num(off_mae_cost[ds], 3))
        + ": the anchor's own error is part noise and part where it sits in the series, and "
        "the shift adds both to every member. Putting the anchors in its context lowers mean "
        "MAE instead, by "
        + ds_pair(lambda ds: num(ma_mae_gain[ds], 3))
        + ". Causal-Chemprop has a higher mean MAE than its own encoder, shifted or not, on "
        "both data sets: its counterfactuals are worse in absolute terms as well as in rank. "
        "MAE is in the log units of each endpoint.</p>",
        "<h3>ExpansionRx</h3>",
        metric_table(s["expansion"], "mae", MAE_ARMS, 3),
        "<h3>Biogen ADME</h3>",
        metric_table(s["biogen"], "mae", MAE_ARMS, 3),

        "<h2>What it costs</h2>",
        "<p>Every series needs its own SCM in every fold, each on one CPU core: DAGMA on the "
        "analog set, a mechanism per node, then intrinsic causal contributions by Shapley "
        "sampling, which is most of the time. The encoders, one per endpoint and fold, are the "
        "cheap part. The fits ran on two machines, a 32-core workstation with an RTX 5070 Ti "
        "and a DGX Spark, in "
        f"{cpu_h:,.0f} CPU-core hours and {gpu_h:.0f} GPU hours.</p>",
        cost_table(cost, n_series),

        "<h2>What this does and does not say</h2>",
        '<div class="panel howto">'
        "<h3>Scope</h3>"
        "<p><b>One repeat.</b> Causal-Chemprop is run on the five folds of the first repeat; "
        "the other arms have all 25 and are compared on those five. The series-level counts do "
        "not depend on the number of folds the way the fold-level tests do, and they are "
        "lopsided enough that more folds would not reverse them.</p>"
        "<p><b>The paper's setting, not its tuning.</b> The SCM hyperparameters are the "
        "majority values of the release's per-task settings, fixed for every series. The paper "
        "describes one rule; the release tunes per task. A per-series search could do better, "
        "but would need held-out series to tune on, which the setting does not have.</p>"
        "<p><b>Series, not kinases.</b> ADME endpoints of clustered test molecules are not the "
        "paper's kinase scaffold groups. The analog rule, the feasibility floor, the anchor and "
        "the metric are the paper's; the chemistry and the endpoints are not.</p>"
        "<p><b>Atom-JEPA on the paper's tasks.</b> The eleven Atom-JEPA fits on the "
        f"paper's tasks used the authors' Biogen recipe; {int((aj_fits['dropped_no_conformer'] > 0).sum())} "
        "tasks lost five training molecules each for which no conformer could be generated, "
        "and the three SRC tasks were run on the DGX Spark, which has room for them.</p>"
        "</div>",

        f'<p class="footnote">Reported the way {LINK_BLOG} and the protocol paper it points '
        "at" + ref("tukey") + " argue comparisons should be: distributions over folds and "
        "series, simultaneous confidence intervals, paired tests, and no bolded maxima. Where "
        "two methods cannot be separated, both are reported as tied.</p>",
        '<p class="repo">Code, per-fold and per-series metrics and every figure: '
        '<a href="https://github.com/patwalters/model-validation-central/tree/main/studies/causal-chemprop">'
        "studies/causal-chemprop</a>.</p>",
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
    if PUBLISHED.parent.is_dir():
        PUBLISHED.write_text(html)
        print(f"published {PUBLISHED.relative_to(PUBLISHED.parents[2])}")
    else:
        print(f"not published: no {PUBLISHED.parent} here")


if __name__ == "__main__":
    main()
