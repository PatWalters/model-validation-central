#!/usr/bin/env python
"""Step 6: the report, both data sets on one self-contained page.

Uni-Mol and Uni-Mol2 84M, each fine-tuned with the package's own recipe and
each frozen under TabPFN 3, against LightGBM + Morgan, ChemProp + CheMeleon,
Monroe + TabPFN 3 and fine-tuned Atom-JEPA. Everything is inlined, figures
included, so the file stands alone. Every number in the prose is computed from
the tables it sits beside rather than typed, and where the prose makes a
qualitative claim, the claim is checked here and the build fails if the data
stop supporting it.

    for ds in expansion biogen; do
      ADME_DATASET=$ds python 02_collect_metrics.py
      ADME_DATASET=$ds python 03_report.py
    done
    python 04_summary.py
    python 06_build_page.py
"""

from pathlib import Path

import pandas as pd

import config as cfg
import page_kit

PAGE = cfg.PROJECT_DIR / "results" / "report.html"
PUBLISHED = cfg.PROJECT_DIR.parent.parent / "docs" / "reports" / "unimol.html"
BLOG_COPY = Path("/Users/pwalters/DATA/BLOG/UNI-MOL/unimol.html")

MAIN, OTHER = "expansion", "biogen"
LABEL_OF = {MAIN: "ExpansionRx", OTHER: "Biogen ADME"}
METHODS = cfg.METHODS
UM, UM2 = cfg.UNIMOL_METHOD, cfg.UNIMOL2_METHOD
UMT, UM2T = cfg.UNIMOL_TABPFN_METHOD, cfg.UNIMOL2_TABPFN_METHOD
AJ_FT = cfg.ATOMJEPA_FT_METHOD
MONROE, CHEMELEON, LGBM = cfg.MONROE_METHOD, cfg.CHEMELEON_METHOD, cfg.LGBM_METHOD
COMPARATORS = [AJ_FT, MONROE, CHEMELEON, LGBM]

COLOR = {m: page_kit.METHOD_COLOR[m] for m in METHODS}
SHORT = {
    LGBM: "LightGBM",
    CHEMELEON: "CheMeleon FT",
    MONROE: "Monroe/PFN3",
    AJ_FT: "Atom-JEPA FT",
    UMT: "Uni-Mol/PFN3",
    UM: "Uni-Mol FT",
    UM2T: "Uni-Mol2/PFN3",
    UM2: "Uni-Mol2 FT",
}
LABEL = {
    **cfg.METHOD_LABELS,
    CHEMELEON: "ChemProp + CheMeleon, fine-tuned",
    UMT: "Uni-Mol frozen + TabPFN 3",
    UM: "Uni-Mol fine-tuned",
    UM2T: "Uni-Mol2 84M frozen + TabPFN 3",
    UM2: "Uni-Mol2 84M fine-tuned",
}


# --- loading -------------------------------------------------------------
def figure(name: str, caption: str, ds: str) -> str:
    return page_kit.figure_block(page_kit.embed_figure(cfg.paths(ds).figures / f"{name}.png"),
                                 caption, name)


def summary(ds: str) -> pd.DataFrame:
    s = pd.read_csv(cfg.paths(ds).tables / "summary.csv")
    return s[s["method"].isin(METHODS)]


def results(name: str) -> pd.DataFrame:
    return pd.read_csv(cfg.PROJECT_DIR / "results" / name)


def num(x: float, places: int = 3, sign: bool = False) -> str:
    """A number for prose, with a real minus sign."""
    text = f"{x:+.{places}f}" if sign else f"{x:.{places}f}"
    return text.replace("-", "−")


WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve",
         13: "thirteen", 14: "fourteen", 15: "fifteen"}


def count(n: int, noun: str) -> str:
    """'one endpoint', 'six endpoints'."""
    return f"{WORDS.get(n, n)} {noun}{'' if n == 1 else 's'}"


def where(endpoints) -> str:
    """'LOG_MLM, LOG_Caco_AB and LOG_Caco_Efflux'."""
    items = [f"<code>{e}</code>" for e in endpoints]
    if not items:
        return "none"
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


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


# Rows of the head-to-head table: (focus, other, row label). Positive gain is
# always the focus arm better.
CALL_ROWS = (
    [(UM2, o, LABEL[o]) for o in COMPARATORS]
    + [(UM2, UM, LABEL[UM])]
    + [(UM, o, f"<i>Uni-Mol fine-tuned against</i> {LABEL[o]}") for o in COMPARATORS]
    + [(UM2, UM2T, f"{LABEL[UM2T]} <i>(fine-tuned against frozen)</i>"),
       (UM, UMT, f"<i>Uni-Mol fine-tuned against</i> {LABEL[UMT]}")]
    + [(UM2T, o, f"<i>frozen Uni-Mol2 + TabPFN 3 against</i> {LABEL[o]}")
       for o in (MONROE, CHEMELEON, LGBM)]
    + [(UMT, o, f"<i>frozen Uni-Mol + TabPFN 3 against</i> {LABEL[o]}")
       for o in (MONROE, LGBM)]
)


def calls_table(calls: pd.DataFrame) -> str:
    """Wins / no call / losses per pair and data set, from the paired t-tests."""
    rows = []
    for focus, other, name in CALL_ROWS:
        cells = [f'<th scope="row"><span class="swatch" style="background:{COLOR[other]}"></span>'
                 f"{name}</th>"]
        g_all = calls[(calls["focus"] == focus) & (calls["other"] == other)]
        for ds in (MAIN, OTHER):
            n = g_all[g_all["dataset"] == LABEL_OF[ds]]["call"].value_counts()
            cells.append(f'<td class="num"><span class="chip win">{n.get("win", 0)}</span></td>'
                         f'<td class="num">{n.get("no call", 0)}</td>'
                         f'<td class="num"><span class="chip loss">{n.get("loss", 0)}</span></td>')
        r2 = g_all[g_all["metric"] == "r2"]["gain"].mean()
        mae = g_all[g_all["metric"] == "mae"]["gain"].mean()
        cells.append(f'<td class="num">{num(r2, 3, True)}</td><td class="num">{num(mae, 4, True)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    n_main = 3 * len(cfg.DATASETS[MAIN].targets)
    n_other = 3 * len(cfg.DATASETS[OTHER].targets)
    return (
        '<div class="bleed tablewrap"><table><thead>'
        '<tr><th scope="col" rowspan="2">Uni-Mol2 84M fine-tuned against</th>'
        f'<th colspan="3">ExpansionRx, {n_main}</th><th colspan="3">Biogen, {n_other}</th>'
        '<th colspan="2">Gain, mean of 15</th></tr>'
        "<tr><th>wins</th><th>no call</th><th>losses</th>"
        "<th>wins</th><th>no call</th><th>losses</th>"
        "<th>Δ R²</th><th>MAE drop</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>")


def levers(calls: pd.DataFrame, frozen: str, tuned: str) -> pd.DataFrame:
    """Per endpoint, the R² of three steps away from Monroe + TabPFN 3.

    representation  swap Monroe's vector for the frozen Uni-Mol one, same TabPFN 3
    finetune        then fine-tune the encoder instead of reading it frozen
    net             the fine-tuned arm against Monroe, the sum of the two
    """
    r2 = calls[calls["metric"] == "r2"]

    def gain(focus, other):
        g = r2[(r2["focus"] == focus) & (r2["other"] == other)]
        return g.set_index(["dataset", "endpoint"])["gain"]

    lv = pd.DataFrame({
        "representation_r2": gain(frozen, MONROE),
        "finetune_r2": gain(tuned, frozen),
        "net_r2": gain(tuned, MONROE),
    }).reset_index()
    order = {(LABEL_OF[ds], t): i for i, ds in enumerate((MAIN, OTHER))
             for t in cfg.DATASETS[ds].targets}
    lv["order"] = [order[(d, e)] for d, e in zip(lv["dataset"], lv["endpoint"])]
    return lv.sort_values("order").drop(columns="order").reset_index(drop=True)


def levers_table(lv: pd.DataFrame) -> str:
    rows = []
    for _, r in lv.iterrows():
        rows.append(
            f'<tr><th scope="row">{r["endpoint"]}</th><td>{r["dataset"]}</td>'
            f'<td class="num">{num(r["representation_r2"], 3, True)}</td>'
            f'<td class="num">{num(r["finetune_r2"], 3, True)}</td>'
            f'<td class="num"><b>{num(r["net_r2"], 3, True)}</b></td></tr>')
    for ds_label, g in lv.groupby("dataset", sort=False):
        med = g[["representation_r2", "finetune_r2", "net_r2"]].median()
        rows.append(
            f'<tr><th scope="row"><b>median</b></th><td>{ds_label}</td>'
            + "".join(f'<td class="num"><b>{num(med[c], 3, True)}</b></td>'
                      for c in ("representation_r2", "finetune_r2", "net_r2")) + "</tr>")
    return ('<div class="bleed tablewrap"><table><thead><tr>'
            '<th scope="col">Endpoint</th><th style="text-align:left">Data set</th>'
            "<th>Swap Monroe's embedding for frozen Uni-Mol2's, same TabPFN 3</th>"
            "<th>Then fine-tune Uni-Mol2 instead</th>"
            "<th>Fine-tuned Uni-Mol2 against Monroe</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def cost_table(cost: pd.DataFrame) -> str:
    rows = []
    for _, r in cost.iterrows():
        rows.append(
            f'<tr><th scope="row">{SHORT[r["method"]]}</th><td>{r["dataset"]}</td>'
            f'<td style="text-align:left">{r["host"]}, {r["gpu"].replace("NVIDIA ", "")}</td>'
            f'<td class="num">{int(r["folds"])}</td>'
            f'<td class="num">{r["median_sec"] / 60:.1f}</td>'
            f'<td class="num">{int(r["median_best_epoch"])}</td>'
            f'<td class="num">{int(r["max_epochs"])}</td>'
            f'<td class="num">{r["total_h"]:.1f}</td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Arm</th>'
            '<th style="text-align:left">Data set</th><th style="text-align:left">Machine</th>'
            "<th>Fold models</th><th>Median minutes / fold</th>"
            "<th>Median best epoch</th><th>Longest run, epochs</th>"
            "<th>Total, GPU hours</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# --- references ----------------------------------------------------------
REFERENCES = [
    ("unimol", "Zhou, G.; Gao, Z.; Ding, Q.; Zheng, H.; Xu, H.; Wei, Z.; Zhang, L.; Ke, G. ",
     "Uni-Mol: A Universal 3D Molecular Representation Learning Framework.",
     '<span class="venue">ICLR</span> 2023. The 3D transformer fine-tuned here as '
     "<code>unimol</code>, from its released <code>mol_pre_all_h_220816</code> weights.",
     "https://openreview.net/forum?id=6K2RM6wVqKu", "openreview:6K2RM6wVqKu"),
    ("unimol2", "Ji, X.; Wang, Z.; Gao, Z.; <i>et al.</i> ",
     "Uni-Mol2: Exploring Molecular Pretraining Model at Scale.",
     '<span class="venue">NeurIPS</span> 2024. The two-track transformer scaled from 84M to '
     "1.1B parameters; its 84M model is fine-tuned here as <code>unimol2</code>.",
     "https://arxiv.org/abs/2406.14969", "arXiv:2406.14969"),
    ("atomjepa", "Petersen, K. H.; Tirsgaard, R. H.; Cornet, F. R. J.; Jordahn, M.; "
     "Schmidt, M. N. ",
     "Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems.",
     "Preprint, 2026. The 39M-parameter 3D self-supervised comparator, fine-tuned.",
     "https://arxiv.org/abs/2610.08400", "arXiv:2610.08400"),
    ("monroe", "Banaszewski, B.; Fitzgibbon, A. W. ",
     "Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference.",
     "Preprint, 2026. The frozen encoder in the strongest in-context arm.",
     "https://arxiv.org/abs/2608.18982", "arXiv:2608.18982"),
    ("tabpfn3", "Grinsztajn, L.; Flöge, K.; Key, O.; <i>et al.</i> ",
     "TabPFN-3: Technical Report.",
     "Preprint, 2026. The head of the Monroe arm and of both frozen Uni-Mol arms.",
     "https://arxiv.org/abs/2605.13986", "arXiv:2605.13986"),
    ("chemeleon", "Burns, J. W.; Zalte, A. S.; Abreu, C. R. A.; <i>et al.</i> ",
     "Deep Learning Foundation Models for Low-Data Regimes from Classical Molecular Descriptors.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2026, articles ASAP. The pre-trained '
     "2D D-MPNN comparator.",
     "https://doi.org/10.1021/acs.jcim.6c01546", "10.1021/acs.jcim.6c01546"),
    ("chemprop", "Graff, D. E.; Morgan, N. K.; Burns, J. W.; <i>et al.</i> ",
     "Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical Property "
     "Prediction.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2026, 66 (1), 28–33.',
     "https://doi.org/10.1021/acs.jcim.5c02332", "10.1021/acs.jcim.5c02332"),
    ("morgan", "Rogers, D.; Hahn, M. ", "Extended-Connectivity Fingerprints.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2010, 50 (5), 742–754. The LightGBM '
     "baseline's representation.",
     "https://doi.org/10.1021/ci100050t", "10.1021/ci100050t"),
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
LINK_ATOMJEPA = '<a href="atom-jepa.html">the Atom-JEPA study</a>'


def build() -> str:
    s_main, s_other = summary(MAIN), summary(OTHER)
    calls = results("calls.csv")
    cost = results("cost.csv")

    def tally(focus, other, ds=None):
        g = calls[(calls["focus"] == focus) & (calls["other"] == other)]
        if ds:
            g = g[g["dataset"] == LABEL_OF[ds]]
        n = g["call"].value_counts()
        return int(n.get("win", 0)), int(n.get("no call", 0)), int(n.get("loss", 0)), len(g)

    def wtl(focus, other, ds=None):
        w, t, l, _ = tally(focus, other, ds)
        return f"{w}–{t}–{l}"

    w2_aj, t2_aj, l2_aj, n_all = tally(UM2, AJ_FT)
    w2_mo, t2_mo, l2_mo, _ = tally(UM2, MONROE)
    w2_cm, t2_cm, l2_cm, _ = tally(UM2, CHEMELEON)
    w2_lg, t2_lg, l2_lg, _ = tally(UM2, LGBM)
    w21, t21, l21, _ = tally(UM2, UM)
    w1_aj, t1_aj, l1_aj, _ = tally(UM, AJ_FT)
    w1_mo, t1_mo, l1_mo, _ = tally(UM, MONROE)
    w1_cm, t1_cm, l1_cm, _ = tally(UM, CHEMELEON)
    w1_lg, t1_lg, l1_lg, _ = tally(UM, LGBM)
    w2_fz, t2_fz, l2_fz, _ = tally(UM2, UM2T)
    w1_fz, t1_fz, l1_fz, _ = tally(UM, UMT)
    w2t_mo, t2t_mo, l2t_mo, _ = tally(UM2T, MONROE)
    w1t_mo, t1t_mo, l1t_mo, _ = tally(UMT, MONROE)
    w2t_lg, t2t_lg, l2t_lg, _ = tally(UM2T, LGBM)
    w1t_lg, t1t_lg, l1t_lg, _ = tally(UMT, LGBM)
    w2t_cm, t2t_cm, l2t_cm, _ = tally(UM2T, CHEMELEON)
    w2m_main, t2m_main, l2m_main, _ = tally(UM2, MONROE, MAIN)
    w2m_other, t2m_other, l2m_other, _ = tally(UM2, MONROE, OTHER)

    tally_m, tally_o = tally_counts(s_main), tally_counts(s_other)
    top = {m: {ds: int(t.loc[m, ["alone", "tied"]].sum()) for ds, t in ((MAIN, tally_m), (OTHER, tally_o))}
           for m in METHODS}
    alone = {m: {ds: int(t.loc[m, "alone"]) for ds, t in ((MAIN, tally_m), (OTHER, tally_o))}
             for m in METHODS}
    n_combo = {MAIN: 3 * len(cfg.DATASETS[MAIN].targets), OTHER: 3 * len(cfg.DATASETS[OTHER].targets)}

    lv2, lv1 = levers(calls, UM2T, UM2), levers(calls, UMT, UM)
    med2 = lv2.groupby("dataset")[["representation_r2", "finetune_r2", "net_r2"]].median()
    med1 = lv1.groupby("dataset")[["representation_r2", "finetune_r2", "net_r2"]].median()
    n_ep = len(lv2)
    n2_rep_down = int((lv2["representation_r2"] < 0).sum())
    n2_ft_up = int((lv2["finetune_r2"] > 0).sum())
    n1_ft_up = int((lv1["finetune_r2"] > 0).sum())

    # Where fine-tuned Uni-Mol2 sits in the top Tukey tier on ExpansionRx, per endpoint
    sm = s_main[(s_main["method"] == UM2) & s_main["tukey_group"].isin(("best", "equivalent"))]
    top_eps_main = [e for e in cfg.DATASETS[MAIN].targets if e in set(sm["endpoint"])]

    # Fold-to-fold spread of R², fine-tuned Uni-Mol arms against the other fine-tuned 3D arm
    r2 = pd.concat([summary(MAIN), summary(OTHER)])
    r2 = r2[r2["metric"] == "r2"].pivot_table(index="endpoint", columns="method", values="sd")
    sd_ratio2 = float((r2[UM2] / r2[AJ_FT]).median())
    sd_ratio1 = float((r2[UM] / r2[AJ_FT]).median())
    r2_main = s_main[s_main["metric"] == "r2"].pivot(index="endpoint", columns="method", values="mean")
    neg_r2 = [e for e in cfg.DATASETS[MAIN].targets if r2_main.loc[e, UM2] < 0]
    caco_sd = s_main[(s_main["metric"] == "r2") & (s_main["method"] == UM2)
                     & s_main["endpoint"].isin(neg_r2)]["sd"]

    n_models = int(cost["folds"].sum())
    gpu_h = float(cost["total_h"].sum())
    gpu_h_v1 = float(cost[cost["method"] == UM]["total_h"].sum())
    gpu_h_v2 = float(cost[cost["method"] == UM2]["total_h"].sum())
    med_epoch = cost.groupby("method")["median_best_epoch"].median()
    apollo_v2 = cost[(cost["method"] == UM2) & (cost["host"] == "apollo")].set_index("dataset")
    spark_v2 = cost[(cost["method"] == UM2) & (cost["host"] == "spark")].set_index("dataset")
    slowdown = float((spark_v2["median_sec"] / apollo_v2["median_sec"]).median())
    apollo_v1 = cost[(cost["method"] == UM) & (cost["host"] == "apollo")].set_index("dataset")
    v2_over_v1 = float((apollo_v2["median_sec"] / apollo_v1["median_sec"]).median())

    # Self-checking assertions so prose never drifts from the computed tables
    assert n_all == 45 and n_models == 100, "prose states 45 combinations and 100 fold models"
    assert l2_aj > w2_aj and l1_aj > w1_aj, "prose states both fine-tuned arms trail Atom-JEPA FT"
    assert w2_cm > l2_cm and l1_cm > w1_cm, "prose states Uni-Mol2 beats CheMeleon and Uni-Mol does not"
    assert w2_lg > l2_lg and w1_lg > l1_lg and l2_lg == 0, "prose states both beat LightGBM, Uni-Mol2 never loses"
    assert w21 > l21, "prose states Uni-Mol2 beats Uni-Mol"
    assert w2m_main > l2m_main and l2m_other == n_combo[OTHER], \
        "prose states Uni-Mol2 beats Monroe on ExpansionRx and loses every Biogen call"
    assert top[UM2][OTHER] == 0 and top[UM][OTHER] == 0 and alone[UM2][MAIN] == 0, \
        "prose states neither fine-tuned arm is ever on top on Biogen, nor Uni-Mol2 alone on ExpansionRx"
    assert top[UM2][MAIN] > top[MONROE][MAIN] and top[AJ_FT][MAIN] > top[UM2][MAIN], \
        "prose states Uni-Mol2 is on top more often than Monroe on ExpansionRx, less than Atom-JEPA"
    assert w2_fz > l2_fz and w1_fz > l1_fz, "prose states fine-tuning beats freezing for both"
    assert w2t_mo == 0 and w1t_mo == 0, "prose states neither frozen arm wins a call against Monroe"
    assert l1t_lg > w1t_lg and w2t_lg > l2t_lg, "prose states frozen Uni-Mol trails LightGBM, frozen Uni-Mol2 beats it"
    assert n2_rep_down == n_ep, "prose states the frozen Uni-Mol2 vector is behind Monroe's on every endpoint"
    assert all(top[m][ds] == 0 for m in (UMT, UM2T) for ds in (MAIN, OTHER)), \
        "prose states the frozen arms are never on top"
    assert set(neg_r2) == {"LOG_Caco_AB", "LOG_Caco_Efflux"}, "prose names the two Caco endpoints"
    assert abs(med_epoch[UM2] - med_epoch[UM]) <= 5 and v2_over_v1 > 3, \
        "prose states the two keep similar epochs and Uni-Mol2's cost is its size"
    assert sd_ratio2 > 1.5, "prose states fine-tuned Uni-Mol2 is noisier fold to fold than Atom-JEPA"

    parts = [
        '<meta charset="utf-8">',
        "<title>Uni-Mol and Uni-Mol2 Against the ADMET Field</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">3D molecular pre-training at scale · 5×5 cross validation · two data sets</p>',
        "<h1>Uni-Mol and Uni-Mol2, fine-tuned the way the package does it, against the ADMET field</h1>",
        '<p class="lede">Uni-Mol' + ref("unimol") + " pre-trains a 3D SE(3)-invariant "
        "transformer on 209 million RDKit conformers by masked-atom prediction and coordinate "
        "denoising; Uni-Mol2" + ref("unimol2") + " scales the idea to a two-track atom/pair "
        "transformer and reports downstream error falling as it grows from 84 million to 1.1 "
        "billion parameters. Both come with a fine-tuning package, <code>unimol_tools</code>, "
        "and that is how they are run here: the package's own recipe at its defaults, on the "
        "same fifteen ExpansionRx and Biogen endpoints, folds and held-out test sets as the "
        "rest of this site, and each encoder also frozen and read by TabPFN 3"
        + ref("tabpfn3") + " exactly as Monroe is. Fine-tuned Uni-Mol2 84M is the stronger of "
        f"the two and a real step up from Uni-Mol ({w21} wins, {t21} no call, {l21} losses "
        f"over {n_all} endpoint × metric calls): it beats fine-tuned CheMeleon" + ref("chemeleon")
        + f" ({w2_cm}–{t2_cm}–{l2_cm}), never loses to LightGBM ({w2_lg}–{t2_lg}–{l2_lg}), and on "
        f"ExpansionRx reaches the top Tukey tier on {top[UM2][MAIN]} of {n_combo[MAIN]} "
        f"combinations, more than Monroe + TabPFN 3" + ref("monroe") + f" ({top[MONROE][MAIN]}). "
        f"But it is never best alone, it trails fine-tuned Atom-JEPA" + ref("atomjepa")
        + f" ({w2_aj}–{t2_aj}–{l2_aj}), and on Biogen it loses all {n_combo[OTHER]} calls to "
        f"Monroe and is on top of none. Frozen, neither Uni-Mol vector wins a single call "
        f"against Monroe's; fine-tuning is what makes these models competitive.</p>",

        '<div class="facts">'
        f'<div class="fact"><b>{len(METHODS)}</b><span>arms compared</span></div>'
        '<div class="fact"><b>15</b><span>endpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b><span>cross validation</span></div>'
        f'<div class="fact"><b>{n_models}</b><span>fine-tuned fold models</span></div>'
        f'<div class="fact"><b>{gpu_h:.0f}</b><span>GPU hours</span></div>'
        "</div>",

        '<div class="panel howto"><h3>The package\'s own recipe, on these folds</h3>'
        "<p>Both fine-tuned arms are a single call to <code>unimol_tools</code> 0.1.5's "
        "<code>MolTrain</code> per repeat, with nothing of ours inside the training loop. "
        "<code>split='select'</code> with <code>kfold=5</code> takes the repeat's training "
        "molecules tagged with the fold each is held out in, and fits one model per fold on "
        "the other four fifths, early-stopping it on the held-out fifth. Uni-Mol starts from "
        "the released <code>mol_pre_all_h_220816</code> weights, Uni-Mol2 from the package's "
        "default 84M model; hydrogens are kept, and each molecule gets the package's own "
        "single RDKit conformer. Every setting is the package default: Adam at 1e−4 with "
        "linear warm-up and decay, batch 16, at most 100 epochs with a patience of 10, "
        "gradient clipping at 5 and mixed precision, one seed per repeat. One model per fold "
        "predicts every endpoint of its data set at once under the package's NaN-masked L1 "
        "loss (<code>multilabel_regression</code>), the multitask setting CheMeleon and "
        "Atom-JEPA are also fine-tuned in. Two choices are ours. Targets are standardised "
        "(<code>target_normalize='standard'</code>), because the default "
        "<code>'auto'</code> applies <code>log1p</code> to skewed columns and these are "
        "already logs that go below −1. And each of the five fold models predicts the test "
        "set separately, rather than through <code>MolPredict</code>, which would average "
        "them and leave five scores where the protocol needs twenty-five.</p>"
        "<p>One thing the package does that the other fine-tuned arms here do not: its target "
        "scaler is fitted on all five fifths of the training set before the folds are split "
        "(<code>unimol_tools/data/datahub.py</code>), so each fold's held-out fifth helps set "
        "the mean and standard deviation the model trains and early-stops in. The test set is "
        "never involved, so the scores below are unaffected; it is a small leak into early "
        "stopping only.</p>"
        "<p>The frozen arms (<code>unimol_tabpfn</code>, <code>unimol2_tabpfn</code>) take the "
        "package's <code>UniMolRepr</code> CLS vector for every molecule — 512-d for Uni-Mol, "
        "768-d for Uni-Mol2 84M, from the same conformers — and hand each fold's training "
        "vectors to TabPFN 3 with Monroe's exact call: one <code>TabPFNRegressor</code>, 8 "
        "estimators, softmax temperature 0.9, no PCA, the mean as the point estimate. Set "
        "against Monroe they differ only in the vector; set against the fine-tuned arm they "
        "show what fine-tuning buys.</p></div>",

        '<div class="panel howto"><h3>Eight arms</h3>'
        "<p>The four comparators are imported unchanged from " + LINK_SOURCE + " and "
        + LINK_ATOMJEPA + " on the identical fold assignments: LightGBM on 2,048-bit Morgan "
        "counts" + ref("morgan") + " (the classical floor), ChemProp" + ref("chemprop")
        + " fine-tuned from CheMeleon" + ref("chemeleon") + " (the strongest 2D message-passing "
        "arm), Monroe + TabPFN 3" + ref("monroe") + " (the strongest frozen encoder read in "
        "context), and fine-tuned Atom-JEPA" + ref("atomjepa") + " (a 39M-parameter 3D "
        "EquiformerV3 pre-trained by latent prediction on 19M Uni-Mol conformers). Uni-Mol2's "
        "310M model was planned and dropped: at the package's batch of 16 it does not fit in "
        "the 16 GB of the faster GPU, and the slower one would have taken days.</p></div>",

        "<h2>The scoreboard</h2>",
        "<p>Tukey HSD per endpoint and metric, counted the same way across every study on this "
        "site: a method is on top if Tukey's simultaneous correction cannot separate it from "
        "the best mean at α = 0.05, and it is only <em>best alone</em> if no other arm shares "
        "the top group. MAE is in the log units of each endpoint, followed by R².</p>",
        "<h3>ExpansionRx" + ref("expansion") + "</h3>",
        tukey_tally(s_main),
        metric_table(s_main, "mae", MAIN),
        figure("tukey_mae", "Tukey HSD on MAE, ExpansionRx. Each bar is a method's mean "
               "over 25 folds with a 95% simultaneous confidence interval. Blue is the best "
               "mean, grey cannot be separated from it, red is significantly worse.", MAIN),
        figure("boxplot_mae", "Fold-level MAE across all 25 folds (5×5 CV) on ExpansionRx.", MAIN),
        metric_table(s_main, "r2", MAIN),
        figure("tukey_r2", "Tukey HSD on R², ExpansionRx.", MAIN),

        "<h3>Biogen ADME" + ref("biogen") + "</h3>",
        tukey_tally(s_other),
        metric_table(s_other, "mae", OTHER),
        figure("tukey_mae", "Tukey HSD on MAE, Biogen ADME.", OTHER),
        figure("boxplot_mae", "Fold-level MAE across all 25 folds (5×5 CV) on Biogen ADME.", OTHER),
        metric_table(s_other, "r2", OTHER),
        figure("tukey_r2", "Tukey HSD on R², Biogen ADME.", OTHER),

        f"<p>On ExpansionRx fine-tuned Atom-JEPA is on top of {top[AJ_FT][MAIN]} of "
        f"{n_combo[MAIN]} combinations ({alone[AJ_FT][MAIN]} of them alone), fine-tuned "
        f"Uni-Mol2 of {top[UM2][MAIN]} — on {count(len(top_eps_main), 'endpoint')}: "
        f"{where(top_eps_main)} — always sharing the tier, and Monroe + TabPFN 3 of "
        f"{top[MONROE][MAIN]}. Fine-tuned Uni-Mol reaches it {count(top[UM][MAIN], 'time')}. "
        f"On Biogen ADME the picture is Monroe's: on top of {top[MONROE][OTHER]} of "
        f"{n_combo[OTHER]} combinations ({alone[MONROE][OTHER]} alone), Atom-JEPA of "
        f"{top[AJ_FT][OTHER]}, and no Uni-Mol arm of any. Neither frozen Uni-Mol arm is on top "
        "anywhere.</p>",

        f"<h2>Head-to-head across all {n_all} endpoint × metric calls</h2>",
        f"<p>Paired by fold, a two-sided t-test over the 25 shared folds at p &lt; 0.05 for "
        f"each endpoint and metric. Fine-tuned Uni-Mol2 against fine-tuned Uni-Mol goes "
        f"{w21}–{t21}–{l21}, so at the same recipe the larger, newer model is better, by a "
        f"mean {num(calls[(calls.focus == UM2) & (calls.other == UM) & (calls.metric == 'r2')]['gain'].mean(), 3, True)} "
        f"R². It beats fine-tuned CheMeleon {w2_cm}–{t2_cm}–{l2_cm} where Uni-Mol does not "
        f"({w1_cm}–{t1_cm}–{l1_cm}), and both beat LightGBM ({w2_lg}–{t2_lg}–{l2_lg} and "
        f"{w1_lg}–{t1_lg}–{l1_lg}). Against Monroe + TabPFN 3 Uni-Mol2 splits by data set: "
        f"{w2m_main}–{t2m_main}–{l2m_main} on ExpansionRx and {w2m_other}–{t2m_other}–{l2m_other} "
        f"on Biogen. Against fine-tuned Atom-JEPA, the other 3D model and one that was "
        f"pre-trained on Uni-Mol's own conformers, both lose clearly: Uni-Mol2 "
        f"{w2_aj}–{t2_aj}–{l2_aj}, Uni-Mol {w1_aj}–{t1_aj}–{l1_aj}.</p>",
        calls_table(calls),
        figure("delta_unimol2_mae", "Fine-tuned Uni-Mol2 against each arm on ExpansionRx: "
               "mean paired MAE difference over 25 folds with 95% confidence intervals, signed "
               "so right of zero favours Uni-Mol2.", MAIN),
        figure("delta_unimol2_mae", "The same paired MAE differences on Biogen ADME.", OTHER),
        figure("paired_unimol_vs_unimol2_mae", "Uni-Mol against Uni-Mol2, both fine-tuned with "
               "the same package defaults, MAE paired by fold on ExpansionRx.", MAIN),
        figure("paired_atomjepa_ft_vs_unimol2_mae", "Two fine-tuned 3D encoders: Atom-JEPA "
               "against Uni-Mol2 84M, MAE paired by fold on ExpansionRx.", MAIN),

        "<h2>Frozen or fine-tuned</h2>",
        "<p>Reading each encoder frozen through Monroe's TabPFN 3 call splits the distance from "
        "Monroe to fine-tuned Uni-Mol2 into two steps: first swap Monroe's vector for "
        "Uni-Mol2's, then fine-tune Uni-Mol2 instead of reading it.</p>",
        levers_table(lv2),
        f"<p>As a frozen vector Uni-Mol2's is behind Monroe's on all {n_ep} endpoints, by a "
        f"median {num(med2.loc['ExpansionRx', 'representation_r2'], 3, True)} R² on ExpansionRx "
        f"and {num(med2.loc['Biogen ADME', 'representation_r2'], 3, True)} on Biogen, and "
        f"neither frozen arm wins one of the {n_all} calls against Monroe (Uni-Mol2 "
        f"{w2t_mo}–{t2t_mo}–{l2t_mo}, Uni-Mol {w1t_mo}–{t1t_mo}–{l1t_mo}). Frozen Uni-Mol is "
        f"behind even LightGBM ({w1t_lg}–{t1t_lg}–{l1t_lg}); frozen Uni-Mol2 edges it "
        f"({w2t_lg}–{t2t_lg}–{l2t_lg}) but loses to CheMeleon {w2t_cm}–{t2t_cm}–{l2t_cm}. "
        f"Fine-tuning then lifts R² on {n2_ft_up} of {n_ep} endpoints for Uni-Mol2 and "
        f"{n1_ft_up} of {n_ep} for Uni-Mol, by a median "
        f"{num(med2.loc['ExpansionRx', 'finetune_r2'], 3, True)} and "
        f"{num(med2.loc['Biogen ADME', 'finetune_r2'], 3, True)} R² for Uni-Mol2 on the two "
        f"data sets ({num(med1.loc['ExpansionRx', 'finetune_r2'], 3, True)} and "
        f"{num(med1.loc['Biogen ADME', 'finetune_r2'], 3, True)} for Uni-Mol), winning "
        f"{w2_fz} of {n_all} calls against its own frozen arm ({w1_fz} for Uni-Mol). The net "
        f"against Monroe is a median {num(med2.loc['ExpansionRx', 'net_r2'], 3, True)} R² on "
        f"ExpansionRx and {num(med2.loc['Biogen ADME', 'net_r2'], 3, True)} on Biogen. Unlike "
        "Monroe, whose encoder never sees these labels, these vectors were never meant to be "
        "used frozen: Uni-Mol is pre-trained to denoise coordinates and recover atoms, not to "
        "summarise a molecule in one token.</p>",
        figure("paired_unimol2_tabpfn_vs_unimol2_mae", "Uni-Mol2 frozen under TabPFN 3 against "
               "Uni-Mol2 fine-tuned, MAE paired by fold on ExpansionRx.", MAIN),
        figure("paired_monroe_vs_unimol2_tabpfn_mae", "Monroe against frozen Uni-Mol2, both read "
               "by the same TabPFN 3 call, MAE paired by fold on Biogen ADME.", OTHER),

        "<h2>Fold to fold</h2>",
        f"<p>The fine-tuned Uni-Mol arms are the least stable fine-tuned arms here. Across the "
        f"fifteen endpoints the median ratio of their fold-to-fold R² standard deviation to "
        f"fine-tuned Atom-JEPA's is {num(sd_ratio2, 1)} for Uni-Mol2 and {num(sd_ratio1, 1)} for "
        f"Uni-Mol. It is worst on the two Caco-2 endpoints of ExpansionRx, {where(neg_r2)}, where "
        f"Uni-Mol2's mean R² is below zero and its standard deviation reaches "
        f"{num(float(caco_sd.max()), 2)}: some folds are fine and some are badly miscalibrated. "
        "A likely contributor is that the package early-stops each fold model on one "
        "validation score pooled over every endpoint, so the epoch it keeps is the one that "
        "suits the data set as a whole, not the hardest endpoint in it.</p>",
        figure("boxplot_r2", "Fold-level R² across all 25 folds on ExpansionRx; note the spread "
               "of the two Uni-Mol fine-tuned arms on the Caco-2 endpoints.", MAIN),

        "<h2>What it costs</h2>",
        f"<p>The {n_models} fine-tuned fold models took {gpu_h:.0f} GPU hours, "
        f"{gpu_h_v1:.0f} for Uni-Mol and {gpu_h_v2:.0f} for Uni-Mol2, across two machines "
        "(<code>apollo</code>, an RTX 5070 Ti; <code>spark</code>, a DGX Spark GB10, which "
        f"took a median {num(slowdown, 1)}× as long per Uni-Mol2 fold). On the same machine a "
        f"Uni-Mol2 fold took a median {num(v2_over_v1, 1)}× as long as a Uni-Mol one, though "
        f"the two keep similar epochs (a median best epoch of {int(med_epoch[UM2])} against "
        f"{int(med_epoch[UM])}), so the cost is the larger model, not longer training. Time per fold is one "
        "<code>MolTrain</code> call's wall clock, five folds and the test predictions, shared "
        "evenly. The frozen arms need one forward pass per molecule and then TabPFN 3, minutes "
        "rather than hours, and are not tabulated.</p>",
        cost_table(cost),

        f'<p class="footnote">Reported the way {LINK_BLOG} and the protocol paper it '
        "points at" + ref("tukey") + " argue comparisons should be: distributions over "
        "folds, simultaneous confidence intervals, paired tests on the folds, and no "
        "bolded maxima. Where two methods cannot be separated, both are reported as "
        "tied.</p>",
        '<p class="repo">Code, per-fold metrics and every figure: '
        '<a href="https://github.com/patwalters/model-validation-central/tree/main/studies/unimol">'
        "studies/unimol</a>.</p>",
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
    if BLOG_COPY.parent.is_dir():
        BLOG_COPY.write_text(html)
        print(f"wrote blog copy {BLOG_COPY}")


if __name__ == "__main__":
    main()
