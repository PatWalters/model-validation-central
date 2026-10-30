#!/usr/bin/env python
"""Step 7: the report, both data sets on one self-contained page.

KERMT v2 and the two GROVER checkpoints it descends from, each fine-tuned by
KERMT's own `main.py finetune` on these folds, and KERMT v2's frozen latent
read by TabPFN 3, against LightGBM + Morgan, ChemProp + CheMeleon, Monroe +
TabPFN 3 and fine-tuned Atom-JEPA. Everything is inlined, figures included, so
the file stands alone. Every number in the prose is computed from the tables it
sits beside rather than typed, and where the prose makes a qualitative claim,
the claim is checked here and the build fails if the data stop supporting it.

    for ds in expansion biogen; do
      ADME_DATASET=$ds python 04_collect_metrics.py
      ADME_DATASET=$ds python 05_report.py
    done
    python 06_summary.py
    python 07_build_page.py
"""

import ast
from pathlib import Path

import pandas as pd

import config as cfg
import page_kit

PAGE = cfg.PROJECT_DIR / "results" / "report.html"
PUBLISHED = cfg.PROJECT_DIR.parent.parent / "docs" / "reports" / "kermt.html"
BLOG_COPY = Path("/Users/pwalters/DATA/BLOG/KERMT/kermt.html")

MAIN, OTHER = "expansion", "biogen"
LABEL_OF = {MAIN: "ExpansionRx", OTHER: "Biogen ADME"}
METHODS = cfg.METHODS
K, KT = cfg.KERMT_METHOD, cfg.KERMT_TABPFN_METHOD
GB, GL = cfg.GROVER_BASE_METHOD, cfg.GROVER_LARGE_METHOD
AJ_FT = cfg.ATOMJEPA_FT_METHOD
MONROE, CHEMELEON, LGBM = cfg.MONROE_METHOD, cfg.CHEMELEON_METHOD, cfg.LGBM_METHOD
COMPARATORS = [AJ_FT, MONROE, CHEMELEON, LGBM]
TUNED = [K, GB, GL]

COLOR = {m: page_kit.METHOD_COLOR[m] for m in METHODS}
SHORT = {m: page_kit.SHORT[m] for m in METHODS}
LABEL = {
    **cfg.METHOD_LABELS,
    CHEMELEON: "ChemProp + CheMeleon, fine-tuned",
    KT: "KERMT v2 frozen + TabPFN 3",
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


def diagnostics() -> pd.DataFrame:
    """Both data sets' per-fold diagnostics, the effective args spread out.

    `effective` is what the fold actually trained with, which is not always
    what was asked for: see the note on load_checkpoint below.
    """
    frames = []
    for ds in (MAIN, OTHER):
        d = pd.read_csv(cfg.paths(ds).diagnostics)
        eff = pd.DataFrame(list(d.pop("effective").map(ast.literal_eval))).add_prefix("eff_")
        frames.append(pd.concat([d.assign(dataset=LABEL_OF[ds]), eff], axis=1))
    return pd.concat(frames, ignore_index=True)


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
    [(K, o, LABEL[o]) for o in COMPARATORS]
    + [(K, GB, LABEL[GB]), (K, GL, LABEL[GL]), (K, KT, LABEL[KT])]
    + [(GB, o, f"<i>GROVER base fine-tuned against</i> {LABEL[o]}") for o in COMPARATORS]
    + [(GL, o, f"<i>GROVER large fine-tuned against</i> {LABEL[o]}") for o in COMPARATORS]
    + [(GL, GB, f"<i>GROVER large against</i> {LABEL[GB]}")]
    + [(KT, o, f"<i>frozen KERMT v2 + TabPFN 3 against</i> {LABEL[o]}")
       for o in (MONROE, CHEMELEON, LGBM)]
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
        '<tr><th scope="col" rowspan="2">KERMT v2 fine-tuned against</th>'
        f'<th colspan="3">ExpansionRx, {n_main}</th><th colspan="3">Biogen, {n_other}</th>'
        '<th colspan="2">Gain, mean of 15</th></tr>'
        "<tr><th>wins</th><th>no call</th><th>losses</th>"
        "<th>wins</th><th>no call</th><th>losses</th>"
        "<th>Δ R²</th><th>MAE drop</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>")


def levers(calls: pd.DataFrame) -> pd.DataFrame:
    """Per endpoint, the R² of two steps away from Monroe + TabPFN 3.

    representation  swap Monroe's vector for KERMT v2's frozen latent, same TabPFN 3
    finetune        then fine-tune the encoder instead of reading it frozen
    net             the fine-tuned arm against Monroe, the sum of the two
    """
    r2 = calls[calls["metric"] == "r2"]

    def gain(focus, other):
        g = r2[(r2["focus"] == focus) & (r2["other"] == other)]
        return g.set_index(["dataset", "endpoint"])["gain"]

    lv = pd.DataFrame({
        "representation_r2": gain(KT, MONROE),
        "finetune_r2": gain(K, KT),
        "net_r2": gain(K, MONROE),
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
            "<th>Swap Monroe's embedding for KERMT v2's frozen latent, same TabPFN 3</th>"
            "<th>Then fine-tune KERMT v2 instead</th>"
            "<th>Fine-tuned KERMT v2 against Monroe</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def arch_table(diag: pd.DataFrame) -> str:
    """What each checkpoint dictated, recorded from the model each fold saved."""
    fields = [("eff_backbone", "Encoder", None), ("eff_hidden_size", "Hidden width", 0),
              ("eff_dropout", "Dropout", None), ("eff_dist_coff", "Distance coefficient", None),
              ("eff_bond_drop_rate", "Bond drop rate", None),
              ("eff_select_by_loss", "Best epoch picked on", None)]
    rows = []
    for key, name, places in fields:
        cells = [f'<th scope="row">{name}</th>']
        for m in TUNED:
            values = sorted(set(diag[diag["method"] == m][key]))
            if len(values) != 1:
                raise SystemExit(f"{m} varies in {key}: {values}")
            v = values[0]
            if key == "eff_select_by_loss":
                text = "validation loss" if v else "validation MAE"
            elif places == 0:
                text = f"{int(v)}"
            else:
                text = f"{v}"
            cells.append(f'<td class="num">{text}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(f'<th><span class="swatch" style="background:{COLOR[m]}"></span>{SHORT[m]}</th>'
                   for m in TUNED)
    return ('<div class="tablewrap"><table><thead>'
            f'<tr><th scope="col">In effect</th>{head}</tr></thead>'
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
            "<th>Median best epoch</th><th>Epoch budget</th>"
            "<th>Total, GPU hours</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# --- references ----------------------------------------------------------
REFERENCES = [
    ("kermt", "Adrian, C.; Mahajan, S. P.; Sirelkhatim, A.; <i>et al.</i> ",
     "KERMT: Accelerating Molecular Property Prediction with GPU-Optimised Graph Transformers.",
     "Preprint, 2025. NVIDIA's accelerated reimplementation of GROVER, and the code that "
     "fine-tunes every arm fitted here.",
     "https://github.com/NVIDIA-BioNeMo/KERMT", "github:NVIDIA-BioNeMo/KERMT"),
    ("kermtv2", "Xue, D.; Mahajan, S. P.; Adrian, C.; <i>et al.</i> ",
     "Contrastive Pre-training for Molecular Property Prediction at Scale.",
     "Preprint, 2026. The re-pretrained encoder released as "
     "<code>NV-KERMT-70M-v2</code>, fine-tuned here as <code>kermt_ft</code> and read "
     "frozen as <code>kermt_tabpfn</code>.",
     "https://huggingface.co/nvidia/NV-KERMT-70M-v2", "hf:nvidia/NV-KERMT-70M-v2"),
    ("grover", "Rong, Y.; Bian, Y.; Xu, T.; Xie, W.; Wei, Y.; Huang, W.; Huang, J. ",
     "Self-Supervised Graph Transformer on Large-Scale Molecular Data.",
     '<span class="venue">NeurIPS</span> 2020. The dual-view graph transformer KERMT '
     "reimplements; its released base and large checkpoints are the two unexposed arms here.",
     "https://proceedings.neurips.cc/paper/2020/hash/94aef38441efa3380a3bed3faf1f9d5d-Abstract.html",
     "NeurIPS 2020"),
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
     "Preprint, 2026. The head of the Monroe arm and of the frozen KERMT arm.",
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
    ("descriptastorus", "Kelley, B. ", "Descriptastorus.",
     "Software, 2018–. The fixed-CDF RDKit 2D descriptors KERMT concatenates to its readout.",
     "https://github.com/bp-kelley/descriptastorus", "github:bp-kelley/descriptastorus"),
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
LINK_UNIMOL = '<a href="unimol.html">the Uni-Mol study</a>'


def build() -> str:
    s_main, s_other = summary(MAIN), summary(OTHER)
    calls = results("calls.csv")
    cost = results("cost.csv")
    diag = diagnostics()

    def tally(focus, other, ds=None):
        g = calls[(calls["focus"] == focus) & (calls["other"] == other)]
        if ds:
            g = g[g["dataset"] == LABEL_OF[ds]]
        n = g["call"].value_counts()
        return int(n.get("win", 0)), int(n.get("no call", 0)), int(n.get("loss", 0)), len(g)

    def wtl(focus, other, ds=None):
        w, t, l, _ = tally(focus, other, ds)
        return f"{w}–{t}–{l}"

    def gain(focus, other, metric="r2", ds=None):
        g = calls[(calls["focus"] == focus) & (calls["other"] == other)
                  & (calls["metric"] == metric)]
        if ds:
            g = g[g["dataset"] == LABEL_OF[ds]]
        return float(g["gain"].mean())

    wk_aj, tk_aj, lk_aj, n_all = tally(K, AJ_FT)
    wk_mo, tk_mo, lk_mo, _ = tally(K, MONROE)
    wk_gb, tk_gb, lk_gb, _ = tally(K, GB)
    wk_gl, tk_gl, lk_gl, _ = tally(K, GL)
    wk_fz, tk_fz, lk_fz, _ = tally(K, KT)
    wgl_gb, tgl_gb, lgl_gb, _ = tally(GL, GB)
    wkt_mo, tkt_mo, lkt_mo, _ = tally(KT, MONROE)
    wkt_cm, tkt_cm, lkt_cm, _ = tally(KT, CHEMELEON)
    wkt_lg, tkt_lg, lkt_lg, _ = tally(KT, LGBM)

    tally_m, tally_o = tally_counts(s_main), tally_counts(s_other)
    top = {m: {ds: int(t.loc[m, ["alone", "tied"]].sum())
               for ds, t in ((MAIN, tally_m), (OTHER, tally_o))} for m in METHODS}
    alone = {m: {ds: int(t.loc[m, "alone"]) for ds, t in ((MAIN, tally_m), (OTHER, tally_o))}
             for m in METHODS}
    n_combo = {MAIN: 3 * len(cfg.DATASETS[MAIN].targets),
               OTHER: 3 * len(cfg.DATASETS[OTHER].targets)}

    lv = levers(calls)
    med = lv.groupby("dataset")[["representation_r2", "finetune_r2", "net_r2"]].median()
    n_ep = len(lv)
    n_rep_down = int((lv["representation_r2"] < 0).sum())
    rep_up = list(lv[lv["representation_r2"] > 0]["endpoint"])
    n_ft_up = int((lv["finetune_r2"] > 0).sum())

    # Where fine-tuned KERMT v2 sits in the top Tukey tier, per endpoint
    def top_eps(s, ds, method):
        hit = s[(s["method"] == method) & s["tukey_group"].isin(("best", "equivalent"))]
        return [e for e in cfg.DATASETS[ds].targets if e in set(hit["endpoint"])]

    k_top_main = top_eps(s_main, MAIN, K)
    k_top_other = top_eps(s_other, OTHER, K)

    n_models = int(cost["folds"].sum())
    gpu_h = float(cost["total_h"].sum())
    med_epoch = diag.groupby("method")["best_epoch"].median()
    late = {m: float((diag[diag["method"] == m]["best_epoch"] >= 90).mean()) for m in TUNED}
    budget = int(cfg.FINETUNE["epochs"])
    apollo = cost[cost["host"] == "apollo"].set_index(["method", "dataset"])["median_sec"]
    spark = cost[cost["host"] == "spark"].set_index(["method", "dataset"])["median_sec"]
    shared = apollo.index.intersection(spark.index)
    slowdown = float((spark[shared] / apollo[shared]).median())
    gl_over_gb = float(cost[(cost["method"] == GL) & (cost["host"] == "apollo")]
                       .set_index("dataset")["median_sec"].div(
        cost[(cost["method"] == GB) & (cost["host"] == "apollo")]
        .set_index("dataset")["median_sec"]).median())

    # Self-checking assertions so prose never drifts from the computed tables
    assert n_all == 45 and n_models == 150, "prose states 45 combinations and 150 fold models"
    assert wk_gb > lk_gb and wk_gl > lk_gl, "prose states KERMT v2 beats both GROVER arms"
    assert lgl_gb > wgl_gb, "prose states GROVER large does not beat GROVER base"
    assert lk_aj > wk_aj, "prose states KERMT v2 trails Atom-JEPA FT"
    assert tally(K, LGBM)[0] == n_all, "prose states KERMT v2 wins every call against LightGBM"
    assert tally(K, CHEMELEON)[0] > tally(K, CHEMELEON)[2], "prose states KERMT v2 beats CheMeleon"
    assert tally(K, MONROE, MAIN)[0] > tally(K, MONROE, MAIN)[2], \
        "prose states KERMT v2 is ahead of Monroe on ExpansionRx"
    assert tally(K, MONROE, OTHER)[0] == 0, \
        "prose states KERMT v2 wins no Biogen call against Monroe"
    assert all(tally(m, MONROE, OTHER)[0] == 0 for m in TUNED), \
        "prose states no fine-tuned arm wins a Biogen call against Monroe"
    assert all(alone[m][ds] == 0 for m in TUNED for ds in (MAIN, OTHER)), \
        "prose states no fine-tuned arm here is ever best alone"
    assert top[AJ_FT][MAIN] > top[K][MAIN] and top[MONROE][OTHER] > top[K][OTHER], \
        "prose states Atom-JEPA leads ExpansionRx and Monroe leads Biogen"
    assert top[K][MAIN] >= top[GB][MAIN] and top[K][MAIN] >= top[GL][MAIN], \
        "prose states KERMT v2 is on top at least as often as the GROVER arms"
    assert wk_fz > lk_fz and lk_fz == 0, "prose states fine-tuning never loses to the frozen arm"
    assert wkt_mo < lkt_mo and wkt_cm < lkt_cm and wkt_lg > lkt_lg, \
        "prose states the frozen latent trails Monroe and CheMeleon but beats LightGBM"
    assert n_rep_down == n_ep - len(rep_up) and len(rep_up) == 2, \
        "prose states the frozen KERMT latent is behind Monroe's on all but two endpoints"
    assert top[KT][OTHER] == 0, "prose states the frozen arm is never on top on Biogen"
    assert gain(K, GB) > 0 and gain(K, GL) > 0 and gain(K, GB) < 0.05, \
        "prose states KERMT v2's edge over GROVER is positive but small"
    assert all(late[m] < 0.35 for m in TUNED), \
        "prose states the best epoch is usually well inside the budget"
    assert set(diag[diag["method"] == K]["selected_on"]) == {"mae"} and \
        set(diag[diag["method"] == GB]["selected_on"]) == {"loss"}, \
        "prose states v2 selects on MAE and GROVER on loss"

    parts = [
        '<meta charset="utf-8">',
        "<title>KERMT and GROVER Against the ADMET Field</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">Graph-transformer pre-training · 5×5 cross validation · two data sets</p>',
        "<h1>KERMT v2 and the two GROVER checkpoints it descends from, fine-tuned on the same "
        "folds</h1>",
        '<p class="lede">KERMT' + ref("kermt") + " is NVIDIA's GPU-optimised reimplementation of "
        "GROVER" + ref("grover") + ", the dual-view graph transformer that pre-trains on 11 "
        "million unlabelled molecules by predicting masked subgraph motifs; KERMT v2"
        + ref("kermtv2") + " re-pretrains that encoder with a contrastive, probabilistic "
        "objective and is the checkpoint NVIDIA released. All three are fine-tuned here by "
        "KERMT's own <code>main.py finetune</code>, one multitask model per fold, on the same "
        "fifteen ExpansionRx and Biogen endpoints, folds and held-out test sets as the rest of "
        "this site, and v2's frozen 512-d latent is also read by TabPFN 3" + ref("tabpfn3")
        + " exactly as Monroe is. The re-pretraining earns its keep inside the family: "
        f"fine-tuned v2 beats fine-tuned GROVER base {wk_gb}–{tk_gb}–{lk_gb} and GROVER large "
        f"{wk_gl}–{tk_gl}–{lk_gl} over {n_all} endpoint × metric calls, and all three clear "
        f"LightGBM and fine-tuned CheMeleon" + ref("chemeleon") + ". Scaling the original "
        f"checkpoint does not: GROVER large goes {wgl_gb}–{tgl_gb}–{lgl_gb} against GROVER "
        "base. But none of the three reaches the front of this field. Fine-tuned v2 trails "
        "fine-tuned Atom-JEPA" + ref("atomjepa") + f" {wk_aj}–{tk_aj}–{lk_aj}, wins no Biogen "
        f"call against Monroe + TabPFN 3" + ref("monroe") + f" ({wtl(K, MONROE, OTHER)}), and is "
        f"never best alone on either data set. Frozen, v2's latent loses "
        f"{lkt_mo} of {n_all} calls to Monroe's vector. And the comparison carries a caveat the "
        "GROVER arms exist to expose: v2 was pre-trained on a corpus that contains every "
        "molecule in both test sets.</p>",

        '<div class="facts">'
        f'<div class="fact"><b>{len(METHODS)}</b><span>arms compared</span></div>'
        '<div class="fact"><b>15</b><span>endpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b><span>cross validation</span></div>'
        f'<div class="fact"><b>{n_models}</b><span>fine-tuned fold models</span></div>'
        f'<div class="fact"><b>{gpu_h:.0f}</b><span>GPU hours</span></div>'
        "</div>",

        '<div class="panel howto"><h3>The released checkpoint has seen these molecules</h3>'
        "<p>The <code>NV-KERMT-70M-v2</code> checkpoint records the corpus it was pre-trained "
        "on, and that corpus includes both data sets used here: its "
        "<code>train_data_path</code> is a hybrid of ZINC15, ChEMBL, Biogen, OpenADMET and "
        "ChEMBL-MT. Structures only, with no labels, and the objective is contrastive rather "
        "than supervised — but every test molecule was in the pre-training set, so the test "
        "split is not blind to the encoder in the way the protocol assumes. That is the reason "
        "the two original GROVER checkpoints are here. Both pre-date these data sets, so they "
        "have never seen a test molecule in any form, and they share the encoder, the "
        "fine-tuning code and the recipe with v2. Read the v2 result as an upper bound and the "
        "GROVER results as the clean ones; the gap between them mixes better pre-training with "
        "whatever structural exposure is worth, and this design cannot separate the two. What "
        f"can be said is that the gap is small — a mean {num(gain(K, GB), 3, True)} R² over "
        f"GROVER base — and that it does not move v2 past Atom-JEPA or Monroe.</p></div>",

        '<div class="panel howto"><h3>KERMT\'s own fine-tuning, on these folds</h3>'
        "<p>Each of the 150 fold models is one call to <code>main.py finetune</code> at commit "
        f"<code>{cfg.KERMT_COMMIT}</code>, with nothing of ours inside the training loop. The "
        "fold's four training fifths, the held-out fifth and the fixed test set go in as three "
        "CSVs with a matching feature file each; KERMT fits its target scaler on the training "
        "CSV only, trains for "
        f"{budget} epochs, keeps the epoch that scored best on the held-out fifth, and predicts "
        "the test set with it. One model covers every endpoint of its data set at once, labels "
        "masked where unmeasured, the multitask setting CheMeleon and Atom-JEPA are also "
        "fine-tuned in. The recipe is the v2 paper's Appendix E: Noam-scheduled Adam from 1e−4 "
        "to 2e−5, batch 32, a 2-layer shared trunk under a 3-layer head per endpoint, every MLP "
        "700 wide. Molecule-level features are descriptastorus' CDF-normalised RDKit 2D "
        "descriptors" + ref("descriptastorus") + ", the repository default, computed once for "
        "every molecule — the normalisation is fitted by descriptastorus on a reference set, not "
        "on these data, so nothing passes between folds.</p>"
        "<p>One thing to know about running this code as released: "
        "<code>load_checkpoint</code> copies the pre-trained checkpoint's own saved model "
        "arguments over the command line (<code>task/train.py:249</code>), so a few settings "
        "are the checkpoint's rather than ours, and they differ between the arms. Each fold "
        "records what it actually trained with, and that is what the table below reports. The "
        "two that matter: v2 trains with dropout 0.1 and keeps the epoch with the best "
        "validation MAE, while both GROVER checkpoints train with no dropout and keep the epoch "
        "with the best validation loss. The code was left unmodified rather than forced to "
        "agree, so the arms differ a little in more than their weights.</p>",
        arch_table(diag),
        "</div>",

        '<div class="panel howto"><h3>Eight arms</h3>'
        "<p>The four comparators are imported unchanged from " + LINK_SOURCE + " and "
        + LINK_ATOMJEPA + " on the identical fold assignments: LightGBM on 2,048-bit Morgan "
        "counts" + ref("morgan") + " (the classical floor), ChemProp" + ref("chemprop")
        + " fine-tuned from CheMeleon" + ref("chemeleon") + " (the strongest 2D message-passing "
        "arm), Monroe + TabPFN 3" + ref("monroe") + " (the strongest frozen encoder read in "
        "context), and fine-tuned Atom-JEPA" + ref("atomjepa") + " (a 39M-parameter 3D "
        "EquiformerV3 pre-trained by latent prediction). The frozen KERMT arm "
        "(<code>kermt_tabpfn</code>) takes v2's 512-d projected cMIM latent for every molecule "
        "and hands each fold's training vectors to TabPFN 3 with Monroe's exact call: one "
        "<code>TabPFNRegressor</code>, 8 estimators, softmax temperature 0.9, no PCA, the mean "
        "as the point estimate. The latent was chosen before any fold was scored.</p></div>",

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
        f"{n_combo[MAIN]} combinations ({alone[AJ_FT][MAIN]} of them alone) and Monroe + TabPFN 3 "
        f"of {top[MONROE][MAIN]} ({alone[MONROE][MAIN]} alone). The three arms fine-tuned here "
        f"land in a band below that and close together: KERMT v2 on top of {top[K][MAIN]} "
        f"combinations, GROVER base {top[GB][MAIN]}, GROVER large {top[GL][MAIN]}, and none of "
        f"the three ever alone. v2's are {count(len(k_top_main), 'endpoint')}: "
        f"{where(k_top_main)}. On Biogen ADME the picture is Monroe's — on top of "
        f"{top[MONROE][OTHER]} of {n_combo[OTHER]} combinations, {alone[MONROE][OTHER]} of them "
        f"alone — and the fine-tuned graph transformers mostly drop out: KERMT v2 reaches the "
        f"top tier on {top[K][OTHER]} of them, all on {where(k_top_other)}, and neither GROVER arm "
        f"nor the frozen latent reaches it at all.</p>",

        f"<h2>Head-to-head across all {n_all} endpoint × metric calls</h2>",
        f"<p>Paired by fold, a two-sided t-test over the 25 shared folds at p &lt; 0.05 for each "
        f"endpoint and metric. Inside the family the ordering is clear and in the direction the "
        f"v2 paper argues: fine-tuned v2 over fine-tuned GROVER base {wk_gb}–{tk_gb}–{lk_gb} "
        f"(mean {num(gain(K, GB), 3, True)} R²) and over GROVER large {wk_gl}–{tk_gl}–{lk_gl} "
        f"({num(gain(K, GL), 3, True)} R²), with the two GROVER arms close enough that most of "
        f"their calls go unmade — GROVER large against base is {wgl_gb}–{tgl_gb}–{lgl_gb}, so "
        f"the bigger original checkpoint is if anything the weaker one. Outside the family it "
        f"gets harder. All three beat LightGBM almost everywhere and CheMeleon more often than "
        f"not. Against Monroe, v2 is ahead on ExpansionRx ({wtl(K, MONROE, MAIN)}) and loses "
        f"everything that is called on Biogen ({wtl(K, MONROE, OTHER)}); the same split, deeper, "
        f"holds for both GROVER arms. Against fine-tuned Atom-JEPA every one of the three loses "
        f"clearly: v2 {wk_aj}–{tk_aj}–{lk_aj}, GROVER base {wtl(GB, AJ_FT)}, GROVER large "
        f"{wtl(GL, AJ_FT)}.</p>",
        calls_table(calls),
        figure("delta_kermt_ft_mae", "Fine-tuned KERMT v2 against each arm on ExpansionRx: "
               "mean paired MAE difference over 25 folds with 95% confidence intervals, signed "
               "so right of zero favours KERMT v2.", MAIN),
        figure("delta_kermt_ft_mae", "The same paired MAE differences on Biogen ADME.", OTHER),
        figure("paired_grover_base_ft_vs_kermt_ft_mae", "The re-pretraining, isolated: GROVER "
               "base against KERMT v2, the same encoder, code and recipe from two different "
               "pre-trained checkpoints, MAE paired by fold on ExpansionRx.", MAIN),
        figure("paired_grover_base_ft_vs_grover_large_ft_mae", "Scaling the original checkpoint: "
               "GROVER base (800-wide) against GROVER large (1200-wide), MAE paired by fold on "
               "ExpansionRx.", MAIN),
        figure("paired_atomjepa_ft_vs_kermt_ft_mae", "Against the strongest fine-tuned arm on "
               "this site: Atom-JEPA against KERMT v2, MAE paired by fold on ExpansionRx.", MAIN),

        "<h2>Frozen or fine-tuned</h2>",
        "<p>Reading v2 frozen through Monroe's TabPFN 3 call splits the distance from Monroe to "
        "fine-tuned v2 into two steps: first swap Monroe's vector for v2's latent, then "
        "fine-tune v2 instead of reading it.</p>",
        levers_table(lv),
        f"<p>As a frozen vector v2's latent is behind Monroe's on {n_rep_down} of the {n_ep} "
        f"endpoints — ahead only on {where(rep_up)} — by a "
        f"median {num(med.loc['ExpansionRx', 'representation_r2'], 3, True)} R² on ExpansionRx "
        f"and {num(med.loc['Biogen ADME', 'representation_r2'], 3, True)} on Biogen, and it wins "
        f"only {wkt_mo} of {n_all} calls against Monroe. It also loses to CheMeleon "
        f"{wkt_cm}–{tkt_cm}–{lkt_cm}, though it does beat LightGBM {wkt_lg}–{tkt_lg}–{lkt_lg}, "
        f"so the latent is not uninformative — just not a drop-in for an encoder built to be "
        f"read frozen. Fine-tuning then lifts R² on {n_ft_up} of {n_ep} endpoints, by a median "
        f"{num(med.loc['ExpansionRx', 'finetune_r2'], 3, True)} R² on ExpansionRx and "
        f"{num(med.loc['Biogen ADME', 'finetune_r2'], 3, True)} on Biogen, and wins {wk_fz} of "
        f"{n_all} calls against the frozen arm while losing {lk_fz}. The net against Monroe is "
        f"a median {num(med.loc['ExpansionRx', 'net_r2'], 3, True)} R² on ExpansionRx and "
        f"{num(med.loc['Biogen ADME', 'net_r2'], 3, True)} on Biogen — the one data set it gains "
        "on and the one it does not. This is the same shape " + LINK_UNIMOL + " found: an "
        "encoder pre-trained to reconstruct or contrast, rather than to summarise a molecule in "
        "one vector, needs its weights moved before it competes.</p>",
        figure("paired_kermt_tabpfn_vs_kermt_ft_mae", "KERMT v2 frozen under TabPFN 3 against "
               "KERMT v2 fine-tuned, MAE paired by fold on ExpansionRx.", MAIN),
        figure("paired_monroe_vs_kermt_tabpfn_mae", "Monroe against frozen KERMT v2, both read "
               "by the same TabPFN 3 call, MAE paired by fold on Biogen ADME.", OTHER),

        "<h2>What it costs</h2>",
        f"<p>The {n_models} fine-tuned fold models took {gpu_h:.0f} GPU hours across two "
        "machines (<code>apollo</code>, an RTX 5070 Ti; <code>spark</code>, a DGX Spark GB10). "
        "The one arm both machines ran, GROVER base on ExpansionRx, took a median "
        f"{num(slowdown, 1)}× as long on <code>spark</code>, though that figure flatters "
        "<code>apollo</code>: <code>spark</code> ran two fine-tuning jobs at a time for most of "
        "the sweep, which raises its time per fold and its throughput together. "
        "KERMT runs its full epoch budget with no early stopping, so a fold costs the same "
        f"whatever epoch it ends up keeping. The kept epoch is usually well inside the "
        f"{budget} available — a median of {int(med_epoch[K])} for v2, {int(med_epoch[GB])} for "
        f"GROVER base and {int(med_epoch[GL])} for GROVER large — so the budget is not the "
        f"binding constraint, and an early-stopping patience would cut most of this bill. "
        f"On ExpansionRx, the one arm pairing both ran on <code>apollo</code>, a GROVER large "
        f"fold costs {num(gl_over_gb, 1)}× a GROVER base one for its 1200-wide encoder, and "
        f"gets nothing for it. The frozen arm needs one "
        "forward pass per molecule and then TabPFN 3, minutes rather than hours, and is not "
        "tabulated.</p>",
        cost_table(cost),

        f'<p class="footnote">Reported the way {LINK_BLOG} and the protocol paper it '
        "points at" + ref("tukey") + " argue comparisons should be: distributions over "
        "folds, simultaneous confidence intervals, paired tests on the folds, and no "
        "bolded maxima. Where two methods cannot be separated, both are reported as "
        "tied.</p>",
        '<p class="repo">Code, per-fold metrics and every figure: '
        '<a href="https://github.com/patwalters/model-validation-central/tree/main/studies/kermt">'
        "studies/kermt</a>.</p>",
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
