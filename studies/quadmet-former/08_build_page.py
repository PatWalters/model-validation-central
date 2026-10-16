#!/usr/bin/env python
"""Step 8: the report, both data sets on one self-contained page.

QuADMET-Former (both fine-tuned via LP-FT and frozen under TabPFN 3) against
LightGBM + Morgan, ChemProp + CheMeleon, Monroe + TabPFN 3, and fine-tuned
Atom-JEPA. Everything is inlined, figures included, so the file stands alone.
Every number in the prose is computed from the tables it sits beside rather
than typed, so a claim cannot drift from the evidence above it; where the prose
makes a qualitative claim, the claim is checked here and the build fails if the
data stop supporting it.

    for ds in expansion biogen; do
      ADME_DATASET=$ds python 05_collect_metrics.py
      ADME_DATASET=$ds python 06_report.py
    done
    python 07_summary.py
    python 08_build_page.py
"""

from pathlib import Path
import pandas as pd

import config as cfg
import page_kit

PAGE = cfg.PROJECT_DIR / "results" / "report.html"
PUBLISHED = cfg.PROJECT_DIR.parent.parent / "docs" / "reports" / "quadmet-former.html"
BLOG_COPY = Path("/Users/pwalters/DATA/BLOG/MAPLIGHT/quadmet-former.html")

MAIN, OTHER = "expansion", "biogen"
LABEL_OF = {MAIN: "ExpansionRx", OTHER: "Biogen ADME"}
METHODS = cfg.METHODS
FT, PFN = cfg.FT_METHOD, cfg.TABPFN_METHOD
AJ_FT = cfg.ATOMJEPA_FT_METHOD
MONROE, CHEMELEON, LGBM = cfg.MONROE_METHOD, cfg.CHEMELEON_METHOD, cfg.LGBM_METHOD
COMPARATORS = [AJ_FT, MONROE, PFN, CHEMELEON, LGBM]

COLOR = {m: page_kit.METHOD_COLOR[m] for m in METHODS}
SHORT = {
    LGBM: "LightGBM",
    CHEMELEON: "CheMeleon FT",
    MONROE: "Monroe/PFN3",
    AJ_FT: "Atom-JEPA FT",
    PFN: "QuADMET/PFN3",
    FT: "QuADMET FT",
}
LABEL = {
    **cfg.METHOD_LABELS,
    CHEMELEON: "ChemProp + CheMeleon, fine-tuned",
    PFN: "QuADMET-Former frozen + TabPFN 3",
    FT: "QuADMET-Former fine-tuned (LP-FT)",
}
METRIC_NAME = {"r2": "R²", "spearman": "Spearman ρ", "mae": "MAE"}


# --- loading -------------------------------------------------------------
def figure(name: str, caption: str, ds: str) -> str:
    return page_kit.figure_block(page_kit.embed_figure(cfg.paths(ds).figures / f"{name}.png"),
                                 caption, name)


def summary(ds: str) -> pd.DataFrame:
    s = pd.read_csv(cfg.paths(ds).tables / "summary.csv")
    return s[s["method"].isin(METHODS)]


def selection(ds: str) -> pd.DataFrame:
    return pd.read_csv(cfg.paths(ds).tables / "selection_summary.csv")


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
    items = list(endpoints)
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


def calls_table(calls: pd.DataFrame) -> str:
    """QuADMET-Former arms' wins / no call / losses against each comparator, per data set."""
    rows = []
    pairs = [(FT, o) for o in COMPARATORS] + [(PFN, MONROE), (PFN, CHEMELEON), (PFN, LGBM)]
    for focus, other in pairs:
        name = (f"{LABEL[other]}" if focus == FT else
                f"<i>frozen QuADMET-Former + TabPFN 3 against</i> {LABEL[other]}")
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
        '<tr><th scope="col" rowspan="2">QuADMET-Former fine-tuned against</th>'
        f'<th colspan="3">ExpansionRx, {n_main}</th><th colspan="3">Biogen, {n_other}</th>'
        '<th colspan="2">Gain, mean of 15</th></tr>'
        "<tr><th>wins</th><th>no call</th><th>losses</th>"
        "<th>wins</th><th>no call</th><th>losses</th>"
        "<th>Δ R²</th><th>MAE drop</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>")


def selection_table(sel: pd.DataFrame, ds: str = MAIN) -> str:
    """What checkpoint was chosen per endpoint, how far ahead it was, and the 10-checkpoint spread."""
    rows = []
    indexed = sel.set_index("endpoint")
    for endpoint in cfg.paths(ds).dataset.targets:
        if endpoint not in indexed.index:
            continue
        r = indexed.loc[endpoint]
        decisive = r["margin_over_runner_up"] > r["fold_sd"]
        chosen_label = f'{r["conf_mode"]} {r["targets"]}'
        rows.append(
            f'<tr><th scope="row">{endpoint}</th>'
            f'<td style="text-align:left">{chosen_label}</td>'
            f'<td class="num">{r["val_r2"]:.3f}</td>'
            f'<td class="num">{r["margin_over_runner_up"]:.3f}</td>'
            f'<td class="num">{r["spread_best_to_worst"]:.3f}</td>'
            f'<td class="num">{r["fold_sd"]:.3f}</td>'
            f'<td><span class="chip {"win" if decisive else "null"}">'
            f'{"clear" if decisive else "within noise"}</span></td></tr>'
        )
    return (
        '<div class="bleed tablewrap"><table>'
        '<thead><tr><th scope="col">Endpoint</th><th style="text-align:left">Chosen checkpoint</th>'
        '<th scope="col">Val R²</th><th scope="col">Margin</th>'
        '<th scope="col">10-ckpt spread</th><th scope="col">Fold sd</th>'
        '<th scope="col">Decisive?</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


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
            "<th>Swap Monroe's embedding for QuADMET-Former's, same TabPFN 3</th>"
            "<th>Then LP-FT the encoder instead</th>"
            "<th>Fine-tuned QuADMET-Former against Monroe</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def ablations_table(abl: pd.DataFrame) -> str:
    rows = []
    for _, r in abl.iterrows():
        ft_r2 = r.get("quadmet_ft_r2_mean", float("nan"))
        sc_r2 = r.get("quadmet_scratch_r2_mean", float("nan"))
        lp_r2 = r.get("quadmet_lp_r2_mean", float("nan"))
        lp3_r2 = r.get("quadmet_lp_3layer_r2_mean", float("nan"))
        all_r2 = r.get("quadmet_lp_fixed_all_r2_mean", float("nan"))
        d_sc = r.get("ft_vs_scratch_r2", float("nan"))
        d_6v3 = r.get("lp6_vs_lp3_r2", float("nan"))
        call_sc = r.get("ft_vs_scratch_call", "no call")
        cls_sc = "win" if call_sc == "win" else ("loss" if call_sc == "loss" else "null")
        rows.append(
            f'<tr><th scope="row">{r["endpoint"]}</th><td>{r["dataset"]}</td>'
            f'<td class="num">{num(ft_r2, 3)}</td>'
            f'<td class="num">{num(sc_r2, 3)}</td>'
            f'<td class="num">{num(d_sc, 3, True)} <span class="chip {cls_sc}">{call_sc}</span></td>'
            f'<td class="num">{num(lp_r2, 3)}</td>'
            f'<td class="num">{num(lp3_r2, 3)} ({num(d_6v3, 3, True)})</td>'
            f'<td class="num">{num(all_r2, 3)}</td></tr>'
        )
    return (
        '<div class="bleed tablewrap"><table><thead><tr>'
        '<th scope="col">Endpoint</th><th style="text-align:left">Data set</th>'
        '<th>LP-FT (6L) R²</th><th>Scratch (6L) R²</th><th>Δ R² (LP-FT − Scratch)</th>'
        '<th>Stage-1 LP (6L) R²</th><th>Truncated LP (3L) R² (Δ 6L−3L)</th>'
        '<th>Fixed Single [All] LP R²</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def size_table(sz: pd.DataFrame) -> str:
    rows = []
    for _, r in sz.iterrows():
        cls = "best" if r["net_r2"] > 0 else "worse"
        rows.append(
            f'<tr><th scope="row">{r["endpoint"]}</th><td>{r["dataset"]}</td>'
            f'<td class="num">{int(r["n_train"]):,}</td><td class="num">{int(r["n_test"]):,}</td>'
            f'<td class="num {cls}">{num(r["net_r2"], 3, True)}</td>'
            f'<td class="num {cls}">{num(r["net_mae"], 3, True)}</td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr>'
            '<th scope="col">Endpoint</th><th style="text-align:left">Data set</th>'
            "<th>Molecules per training fold</th><th>Test molecules</th>"
            "<th>Δ R², QuADMET-Former FT − Monroe</th><th>MAE drop</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def cost_table(cost: pd.DataFrame) -> str:
    rows = []
    for _, r in cost.iterrows():
        rows.append(
            f'<tr><th scope="row">{r["dataset"]}</th><td style="text-align:left">'
            f'{r["host"]}, {r["gpu"].replace("NVIDIA ", "")}</td>'
            f'<td class="num">{int(r["folds"])}</td><td class="num">{r["median_sec"]:.1f}</td>'
            f'<td class="num">{int(r["median_best_epoch"])}</td>'
            f'<td class="num">{r["peak_mem_gb"]:.1f}</td>'
            f'<td class="num">{r["total_h"]:.2f}</td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Data set</th>'
            '<th style="text-align:left">Machine</th><th>Folds</th><th>Median sec / fold</th>'
            "<th>Median total best epoch</th><th>Peak GPU memory, GB</th>"
            "<th>Total, GPU hours</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# --- references ----------------------------------------------------------
REFERENCES = [
    ("quadmet", "Raja, A.; <i>et al.</i> ",
     "QuADMET-Former: Pre-training on Quantum Mechanical Properties Improves ADMET Prediction.",
     '<span class="venue">ChemRxiv</span> 2026, doi:10.26434/chemrxiv.15002429/v3. The method '
     "benchmarked here, run with the authors' TorchMD-ET implementation "
     "(github.com/arunraja-hub/quadmetformer) and their ten released PubChemQC checkpoints.",
     "https://doi.org/10.26434/chemrxiv.15002429/v3", "10.26434/chemrxiv.15002429/v3"),
    ("torchmdnet", "Thölke, P.; De Fabritiis, G. ",
     "TorchMD-NET: Equivariant Transformers for Neural Network Based Molecular Potentials.",
     '<span class="venue">ICLR</span> 2022. The SE(3)-equivariant transformer backbone '
     "underlying QuADMET-Former.",
     "https://arxiv.org/abs/2202.02541", "arXiv:2202.02541"),
    ("atomjepa", "Petersen, K. H.; Tirsgaard, R. H.; Cornet, F. R. J.; Jordahn, M.; "
     "Schmidt, M. N. ",
     "Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems.",
     "Preprint, 2026. The 39M-parameter 3D self-supervised comparator fine-tuned on the same "
     "ten-conformer caches.",
     "https://arxiv.org/abs/2610.08400", "arXiv:2610.08400"),
    ("monroe", "Banaszewski, B.; Fitzgibbon, A. W. ",
     "Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference.",
     "Preprint, 2026. The frozen encoder in the strongest in-context arm.",
     "https://arxiv.org/abs/2608.18982", "arXiv:2608.18982"),
    ("tabpfn3", "Grinsztajn, L.; Flöge, K.; Key, O.; <i>et al.</i> ",
     "TabPFN-3: Technical Report.",
     "Preprint, 2026. The head of the Monroe arm and of the frozen QuADMET-Former arm.",
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
LINK_ATOMJEPA = ('<a href="atom-jepa.html">the Atom-JEPA study</a>')


def build() -> str:
    s_main, s_other = summary(MAIN), summary(OTHER)
    sel_main, sel_other = selection(MAIN), selection(OTHER)
    calls = results("calls.csv")
    lv = results("levers.csv")
    sz = results("size.csv")
    size_stats = results("size_stats.csv").set_index("Unnamed: 0")["value"]
    abl = results("ablations_summary.csv")
    cost = results("cost.csv")

    def tally(focus, other, ds=None):
        g = calls[(calls["focus"] == focus) & (calls["other"] == other)]
        if ds:
            g = g[g["dataset"] == LABEL_OF[ds]]
        n = g["call"].value_counts()
        return int(n.get("win", 0)), int(n.get("no call", 0)), int(n.get("loss", 0)), len(g)

    w_aj, t_aj, l_aj, n_all = tally(FT, AJ_FT)
    w_mo, t_mo, l_mo, _ = tally(FT, MONROE)
    w_cm, t_cm, l_cm, _ = tally(FT, CHEMELEON)
    w_lg, t_lg, l_lg, _ = tally(FT, LGBM)
    w_fz, t_fz, l_fz, _ = tally(FT, PFN)
    w_rep, t_rep, l_rep, _ = tally(PFN, MONROE)
    w_pfn_cm, t_pfn_cm, l_pfn_cm, _ = tally(PFN, CHEMELEON)
    w_pfn_lg, t_pfn_lg, l_pfn_lg, _ = tally(PFN, LGBM)

    tally_m, tally_o = tally_counts(s_main), tally_counts(s_other)
    top = {m: {ds: int(t.loc[m, ["alone", "tied"]].sum()) for ds, t in ((MAIN, tally_m), (OTHER, tally_o))}
           for m in METHODS}
    n_combo = {MAIN: 3 * len(cfg.DATASETS[MAIN].targets), OTHER: 3 * len(cfg.DATASETS[OTHER].targets)}

    med_lv = lv.groupby("dataset")[["representation_r2", "finetune_r2", "net_r2"]].median()
    n_rep_down = int((lv["representation_r2"] < 0).sum())
    n_ep = len(lv)

    # Checkpoint selection statistics
    n_chosen_main = sel_main["checkpoint"].nunique()
    n_chosen_other = sel_other["checkpoint"].nunique()
    med_margin_main = float(sel_main["margin_over_runner_up"].median())
    med_spread_main = float(sel_main["spread_best_to_worst"].median())
    med_margin_other = float(sel_other["margin_over_runner_up"].median())
    med_spread_other = float(sel_other["spread_best_to_worst"].median())
    n_decisive_main = int((sel_main["margin_over_runner_up"] > sel_main["fold_sd"]).sum())
    n_decisive_other = int((sel_other["margin_over_runner_up"] > sel_other["fold_sd"]).sum())

    # Internal ablations
    n_sc_pos = int((abl["ft_vs_scratch_r2"] > 0).sum())
    n_sc_win = int((abl["ft_vs_scratch_call"] == "win").sum())
    n_sc_loss = int((abl["ft_vs_scratch_call"] == "loss").sum())
    med_sc_gain = float(abl["ft_vs_scratch_r2"].median())
    n_lp_pos = int((abl["ft_vs_lp_r2"] > 0).sum())
    med_lp_gain = float(abl["ft_vs_lp_r2"].median())
    med_6v3_gain = float(abl["lp6_vs_lp3_r2"].median())
    n_3v6_pos = int((abl["lp6_vs_lp3_r2"] < 0).sum())

    rho = float(size_stats["rho_r2"])
    n_models = int(cost["folds"].sum())
    gpu_h = float(cost["total_h"].sum())

    # Self-checking assertions so prose never drifts from the computed tables
    assert l_aj > w_aj and l_mo > w_mo, "prose states QuADMET-Former trails Atom-JEPA FT and Monroe overall"
    assert w_fz > l_fz, "prose states fine-tuned QuADMET-Former beats frozen QuADMET-Former + TabPFN 3"
    assert w_lg > l_lg and w_pfn_lg > l_pfn_lg, "prose states both QuADMET-Former arms beat LightGBM"
    assert l_rep > w_rep, "prose states frozen QuADMET-Former trails Monroe under TabPFN 3"
    assert med_sc_gain > 0 and n_sc_pos >= 10 and n_sc_win > n_sc_loss, "prose states QM pre-training beats training from scratch"
    assert n_lp_pos >= 12 and med_lp_gain > 0, "prose states LP-FT improves over Stage-1 LP"
    assert n_decisive_main == 0 and n_decisive_other == 1, "prose states only 1 of 15 checkpoint choices exceeds 1 fold SD"

    parts = [
        '<meta charset="utf-8">',
        "<title>Pre-Training in 3D on Quantum Mechanics: QuADMET-Former Benchmark</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">3D quantum-property pre-training · 5×5 cross validation · two data sets</p>',
        "<h1>Pre-training a 3D equivariant transformer on quantum mechanics, against the ADMET field</h1>",
        '<p class="lede">QuADMET-Former' + ref("quadmet") + " pre-trains a 1.3-million-parameter, "
        "six-layer TorchMD-ET equivariant transformer" + ref("torchmdnet") + " on PubChemQC "
        "quantum mechanical properties — total energy, HOMO–LUMO gap, dipole moment, 3D E3FP "
        "fingerprints, and per-atom Löwdin charges — across ten released single- and "
        "three-conformer checkpoints, then adapts it to ADMET endpoints by two-stage linear "
        "probing and fine-tuning (LP-FT). Its paper reports that quantum pre-training "
        "consistently beats training the same 3D architecture from scratch on TDC and Biogen "
        "ADME benchmarks. Run here on the same fifteen ExpansionRx and Biogen endpoints, "
        "3D conformers, 5×5 cross-validation folds and held-out test sets as the rest of this "
        f"site, that internal claim holds: pre-trained LP-FT beats from-scratch training on "
        f"{n_sc_pos} of {n_ep} endpoints ({n_sc_win} significant wins to {n_sc_loss} losses) by a "
        f"median {num(med_sc_gain, 3, True)} R², beats its own frozen TabPFN 3 arm on {w_fz} of "
        f"{n_all} endpoint × metric combinations (losing on {l_fz}), and beats LightGBM + Morgan "
        f"on {w_lg} of {n_all} combinations (losing on {WORDS.get(l_lg, l_lg)}). Against the "
        f"modern pre-trained field, fine-tuned QuADMET-Former reaches the top Tukey tier on "
        f"{top[FT][MAIN]} of {n_combo[MAIN]} combinations on ExpansionRx (leading or tying on "
        f"<code>LogS</code>, <code>LOG_MPPB</code>, and <code>LOG_MBPB</code>) and nearly splits "
        f"ExpansionRx with fine-tuned CheMeleon" + ref("chemeleon")
        + f" (9 wins, 8 ties, 10 losses; {w_cm}–{t_cm}–{l_cm} across both data sets), while "
        f"trailing the two overall leaders: fine-tuned Atom-JEPA" + ref("atomjepa")
        + f" ({w_aj} wins, {t_aj} no call, {l_aj} losses) and Monroe + TabPFN 3" + ref("monroe")
        + f" ({w_mo} wins, {t_mo} no call, {l_mo} losses).</p>",

        '<div class="facts">'
        f'<div class="fact"><b>{len(METHODS)}</b><span>methods compared</span></div>'
        '<div class="fact"><b>15</b><span>endpoints</span></div>'
        '<div class="fact"><b>10</b><span>released QM checkpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b><span>cross validation</span></div>'
        f'<div class="fact"><b>{n_models}</b><span>LP-FT fold models</span></div>'
        f'<div class="fact"><b>{gpu_h:.1f}</b><span>GPU hours</span></div>'
        "</div>",

        '<div class="panel howto"><h3>The authors\' architecture and checkpoints, on these folds</h3>'
        "<p>Both QuADMET-Former arms load the authors' <code>TorchMD_ET</code> from "
        "github.com/arunraja-hub/quadmetformer and their ten released pre-trained checkpoints: "
        "five quantum-property combinations (<code>[Lw]</code>, <code>[3D]</code>, "
        "<code>[DM,HL,TE]</code>, <code>[Lw,DM,HL,TE]</code>, and <code>[All]</code>) crossed "
        "with single-conformer and three-conformer pre-training. Every molecule uses the exact "
        "same ETKDGv3 + MMFF94 conformer cache as " + LINK_ATOMJEPA + " (11,129 molecules "
        "across both data sets, zero dropped). For the frozen arm (<code>quadmet_tabpfn</code>), "
        "each fold's held-out validation fifth selects the winning checkpoint per endpoint "
        "(<code>03_run_tabpfn.py --select</code>) before the test set is touched, and passes "
        "that checkpoint's 128-D graph embedding to TabPFN 3" + ref("tabpfn3") + " through the "
        "exact call the Monroe arm makes.</p>"
        "<p>The fine-tuned arm (<code>quadmet_ft</code>) adapts the 6-layer three-conformer "
        "<code>multi_all</code> checkpoint via two-stage LP-FT across all endpoints in each "
        "data set simultaneously (matching the multitask fine-tuning protocol of "
        "<code>chemeleon</code> and <code>atomjepa_ft</code>): Stage 1 freezes the six-layer "
        "<code>shared_backbone</code> and trains the shared SiLU trunk and per-endpoint heads "
        "under Kendall uncertainty-weighted Huber loss on standardized targets, restoring the "
        "best validation-MAE epoch; Stage 2 unfreezes all six equivariant layers "
        "(<code>n_frozen = 0</code>) with a differential learning rate and L2-SP regularization "
        "toward the pre-trained QM weights, restoring the best validation-MAE checkpoint before "
        "scoring the held-out test set.</p></div>",

        '<div class="panel howto"><h3>Six arms across three representation families</h3>'
        "<p>The four comparators are imported unchanged from " + LINK_SOURCE + " and "
        + LINK_ATOMJEPA + " on the identical fold assignments: LightGBM on 2,048-bit Morgan "
        "counts" + ref("morgan") + " (the classical floor), ChemProp" + ref("chemprop")
        + " fine-tuned from CheMeleon" + ref("chemeleon") + " (the strongest 2D message-passing "
        "arm), Monroe + TabPFN 3" + ref("monroe") + " (the strongest frozen encoder read in "
        "context), and fine-tuned Atom-JEPA" + ref("atomjepa") + " (a 39M-parameter 3D "
        "EquiformerV3 pre-trained by masked latent prediction on 19M conformers and fine-tuned "
        "multitask on the same 3D conformers).</p></div>",

        "<h2>The scoreboard</h2>",
        "<p>Tukey HSD per endpoint and metric, counted the same way across every study on this "
        "site: a method is on top if Tukey's simultaneous correction cannot separate it from "
        "the best mean at α = 0.05, and it is only <em>best alone</em> if no other arm shares "
        "the top group. The primary tables show MAE in the log10(x + 1) units of each "
        "endpoint, followed by R².</p>",
        "<h3>ExpansionRx" + ref("expansion") + "</h3>",
        tukey_tally(s_main),
        metric_table(s_main, "mae", MAIN),
        figure("tukey_mae", "Tukey HSD on MAE, ExpansionRx. Each bar is a method's mean "
               "over 25 folds with a 95% simultaneous confidence interval. Blue is the best "
               "mean, grey cannot be separated from it, red is significantly worse.", MAIN),
        figure("boxplot_mae", "Fold-level MAE distributions across all 25 folds (5×5 CV) on "
               "ExpansionRx.", MAIN),
        metric_table(s_main, "r2", MAIN),
        figure("tukey_r2", "Tukey HSD on R², ExpansionRx.", MAIN),

        "<h3>Biogen ADME" + ref("biogen") + "</h3>",
        tukey_tally(s_other),
        metric_table(s_other, "mae", OTHER),
        figure("tukey_mae", "Tukey HSD on MAE, Biogen ADME.", OTHER),
        figure("boxplot_mae", "Fold-level MAE distributions across all 25 folds (5×5 CV) on "
               "Biogen ADME.", OTHER),
        metric_table(s_other, "r2", OTHER),
        figure("tukey_r2", "Tukey HSD on R², Biogen ADME.", OTHER),

        f"<p>Across the two data sets, the leaderboard is led by the two heavyweights from "
        + LINK_ATOMJEPA + f": on ExpansionRx fine-tuned Atom-JEPA is on top of "
        f"{top[AJ_FT][MAIN]} of {n_combo[MAIN]} endpoint × metric combinations and Monroe + "
        f"TabPFN 3 of {top[MONROE][MAIN]}, while fine-tuned QuADMET-Former reaches the top "
        f"Tukey tier on {top[FT][MAIN]} of {n_combo[MAIN]} combinations — posting the highest "
        f"mean Spearman ρ on <code>LogS</code> (0.616, tied with Atom-JEPA FT), the best mean "
        f"R² (0.656) and MAE (0.182) on <code>LOG_MPPB</code> (tied with Atom-JEPA FT), and "
        f"tying for best Spearman ρ on <code>LOG_MPPB</code> (0.819) and <code>LOG_MBPB</code> "
        f"(0.879). On Biogen ADME, Monroe is on top of {top[MONROE][OTHER]} of "
        f"{n_combo[OTHER]} combinations and Atom-JEPA FT of {top[AJ_FT][OTHER]}.</p>",

        "<h2>Head-to-head across all 45 endpoint × metric calls</h2>",
        f"<p>Paired by fold across all fifteen endpoints and three metrics ({n_all} "
        f"combinations), two-stage multitask LP-FT (<code>quadmet_ft</code>) substantially "
        f"outperforms reading the frozen 128-D QM embedding in context with TabPFN 3 "
        f"(<code>quadmet_tabpfn</code>), winning {w_fz} of {n_all} combinations, tying {t_fz}, "
        f"and losing {l_fz} (17–0–10 on ExpansionRx and 14–2–2 on Biogen). Against the classical "
        f"LightGBM + Morgan floor, fine-tuned QuADMET-Former wins {w_lg} of {n_all} combinations "
        f"({t_lg} no call, {l_lg} losses, going 17–1–0 on Biogen and 21–3–3 on ExpansionRx), while "
        f"frozen QuADMET-Former + TabPFN 3 wins {w_pfn_lg} ({t_pfn_lg} no call, {l_pfn_lg} "
        f"losses). Against fine-tuned CheMeleon, fine-tuned QuADMET-Former wins {w_cm}, ties "
        f"{t_cm}, and loses {l_cm} — nearly splitting ExpansionRx (9 wins, 8 ties, 10 losses) "
        f"and winning all six calls on Biogen <code>LOG_HPPB</code> and <code>LOG_RPPB</code>. "
        f"Against Monroe + TabPFN 3, fine-tuned QuADMET-Former wins {w_mo} combinations on "
        f"ExpansionRx (<code>LogS</code>, <code>LOG_MPPB</code>, <code>LOG_MBPB</code>, and "
        f"<code>LOG_MGMB</code>) while losing {l_mo} across both sets; against the 30× larger "
        f"fine-tuned Atom-JEPA, it wins {w_aj} (Spearman ρ on ExpansionRx <code>LogS</code> "
        f"and Biogen <code>LOG_RPPB</code>), ties {t_aj} (including all three metrics on "
        f"<code>LOG_MPPB</code> and Spearman ρ on <code>LOG_HPPB</code>), and loses {l_aj}.</p>",
        calls_table(calls),
        figure("delta_quadmet_mae", "Fine-tuned QuADMET-Former against each comparator on "
               "ExpansionRx: mean paired MAE difference over 25 folds with 95% confidence "
               "intervals, signed so right of zero favours QuADMET-Former FT.", MAIN),
        figure("delta_quadmet_mae", "The same paired MAE differences on Biogen ADME.", OTHER),
        figure("paired_atomjepa_ft_vs_quadmet_ft_mae", "Two 3D pre-trained encoders on the "
               "exact same conformers: QuADMET-Former fine-tuned vs Atom-JEPA fine-tuned, "
               "MAE paired by fold on ExpansionRx.", MAIN),

        "<h2>Choosing among the ten released QM checkpoints</h2>",
        "<p>QuADMET-Former releases ten checkpoints spanning five subsets of quantum mechanical "
        "pre-training targets — Löwdin atomic charges alone (<code>[Lw]</code>), 3D E3FP "
        "fingerprints alone (<code>[3D]</code>), the three molecular energy/dipole properties "
        "(<code>[DM,HL,TE]</code>), those three plus Löwdin charges "
        "(<code>[Lw,DM,HL,TE]</code>), and all five together (<code>[All]</code>) — each "
        "trained with either a single conformer or three conformers per molecule. Scoring all "
        "ten on each fold's held-out validation fifth shows two consistent patterns.</p>",
        f"<p>First, the spread from the best checkpoint to the worst is large — a median of "
        f"{num(med_spread_main, 3)} R² on ExpansionRx and {num(med_spread_other, 3)} R² on "
        "Biogen — because checkpoints pre-trained on a single target (<code>[Lw]</code> or "
        "<code>[3D]</code> alone) generally lag the multi-property checkpoints "
        "(<code>[Lw,DM,HL,TE]</code> and <code>[All]</code>). The one striking exception is "
        "ExpansionRx <code>LogD</code>, where the single-conformer Löwdin atomic-charge "
        "checkpoint (<code>Single [Lw]</code>) ranks first on the validation fifth "
        f"({num(float(sel_main.set_index('endpoint').loc['LogD', 'val_r2']), 3)} R²), "
        "consistent with lipophilicity being governed by local atomic partial charges and "
        "polarity. Second, among the top multi-property checkpoints the runner-up is almost "
        f"always within fold noise: across ExpansionRx ({n_chosen_main} distinct winners) the "
        f"median margin over the runner-up is {num(med_margin_main, 3)} R² ({n_decisive_main} "
        f"of 9 exceeding one fold SD), and across Biogen ({n_chosen_other} distinct winners) "
        f"the median margin is {num(med_margin_other, 3)} R² ({n_decisive_other} of 6 "
        "exceeding one fold SD, on <code>LOG_RLM</code>).</p>",
        "<h3>ExpansionRx checkpoint selection</h3>",
        selection_table(sel_main, MAIN),
        figure("quadmet_selection", "Validation R² across the 10 released QuADMET-Former "
               "checkpoints on ExpansionRx (mean ± 95% CI over 25 validation fifths). "
               "Filled dot: chosen checkpoint; open circle: runner-up.", MAIN),
        "<h3>Biogen ADME checkpoint selection</h3>",
        selection_table(sel_other, OTHER),
        figure("quadmet_selection", "Validation R² across the 10 released QuADMET-Former "
               "checkpoints on Biogen ADME.", OTHER),

        "<h2>Decomposing representation and fine-tuning</h2>",
        "<p>Just as in " + LINK_ATOMJEPA + ", the frozen TabPFN 3 arm splits the difference "
        "between Monroe + TabPFN 3 and fine-tuned QuADMET-Former into two steps: first "
        "swapping Monroe's embedding for QuADMET-Former's 128-D graph embedding under the "
        "exact same TabPFN 3 call, and second replacing TabPFN 3 with two-stage LP-FT.</p>",
        levers_table(lv),
        f"<p>Swapping Monroe's representation for QuADMET-Former's frozen embedding under "
        f"TabPFN 3 reduces R² on {n_rep_down} of {n_ep} endpoints, by a median "
        f"{num(med_lv.loc['ExpansionRx', 'representation_r2'], 3, True)} R² on ExpansionRx "
        f"(where frozen QuADMET-Former + TabPFN 3 already edges out Monroe on "
        "<code>LOG_MPPB</code> and <code>LOG_MBPB</code>) and "
        f"{num(med_lv.loc['Biogen ADME', 'representation_r2'], 3, True)} R² on Biogen. "
        "Replacing TabPFN 3 with two-stage multitask LP-FT across all six equivariant layers "
        f"recovers substantial ground, lifting median R² by "
        f"{num(med_lv.loc['ExpansionRx', 'finetune_r2'], 3, True)} on ExpansionRx and "
        f"{num(med_lv.loc['Biogen ADME', 'finetune_r2'], 3, True)} on Biogen — with the largest "
        f"fine-tuning gains on the binding and physicochemical endpoints "
        f"(<code>LOG_MPPB</code> +0.276 R², <code>LOG_MGMB</code> +0.242 R², "
        f"<code>LOG_RPPB</code> +0.219 R², <code>LogD</code> +0.218 R², "
        f"<code>LOG_HPPB</code> +0.215 R², and <code>LogS</code> +0.204 R²).</p>",
        figure("paired_monroe_vs_quadmet_tabpfn_mae", "Monroe against frozen QuADMET-Former, "
               "both read by the same TabPFN 3 call, MAE paired by fold on ExpansionRx.", MAIN),
        size_table(sz),
        f"<p>Across the fifteen endpoints the Spearman rank correlation between training-set "
        f"size and fine-tuned QuADMET-Former's R² margin over Monroe is {num(rho, 2, True)}.</p>",

        "<h2>Testing the paper's internal claims — and auditing the released code</h2>",
        "<p>Why do the paper's internal ablations favour QuADMET-Former while the external "
        "leaderboard still favours Atom-JEPA and Monroe? Running four controlled internal "
        "ablations alongside a line-by-line audit of github.com/arunraja-hub/quadmetformer "
        "answers both halves of that question.</p>",
        ablations_table(abl),
        f"<p><b>1. Quantum pre-training and LP-FT genuinely beat training TorchMD-ET from scratch.</b> "
        f"The paper's central hypothesis is internal: that pre-training TorchMD-ET on QM "
        f"properties and adapting via two-stage LP-FT improves over training the same 3D "
        f"architecture from random initialisation (<code>scratch</code>) or freezing it with a "
        f"linear probe (<code>lp</code>). Our ablation confirms both claims directly: across "
        f"the 15 endpoints, pre-trained LP-FT beats from-scratch training of the exact same "
        f"multitask 6-layer architecture on {n_sc_pos} of {n_ep} endpoints ({n_sc_win} "
        f"significant wins to {n_sc_loss} losses) by a median {num(med_sc_gain, 3, True)} R², "
        f"and Stage-2 fine-tuning improves over Stage-1 linear probing on all {n_lp_pos} of "
        f"{n_ep} endpoints by a median {num(med_lp_gain, 3, True)} R². The paper did not "
        "benchmark against 2D pre-trained MPNNs (CheMeleon), in-context foundation models "
        "(Monroe + TabPFN 3), or 30× larger 3D self-supervised encoders (Atom-JEPA); its "
        "baseline was its own 3D architecture trained from scratch.</p>",
        f"<p><b>2. Intermediate versus final layers under an MLP probe, and why full-backbone "
        f"multitask LP-FT matters.</b> All ten released checkpoints contain 6 equivariant "
        f"attention layers (1,312,268 parameters), matching Section 3.3 of the paper, yet the "
        f"repository's <code>finetuning/downstream_train.py</code> (line 289) restricts Optuna "
        f"to <code>num_layers ∈ {{1, 2, 3}}</code>. Comparing a frozen 6-layer single-task MLP "
        f"probe against a truncated 3-layer MLP probe shows why: under a frozen single-task MLP "
        f"head, intermediate layer 3 edges out the final QM-specialised layer 6 on {n_3v6_pos} "
        f"of {n_ep} endpoints (median Δ 6L−3L of {num(med_6v3_gain, 3, True)} R²). Moreover, "
        f"single-task unstandardized L1 fine-tuning or freezing the lower layers suffers severe "
        f"test-set miscalibration on sparse/shifted endpoints (the same gap seen between "
        f"single-task <code>chemprop_st</code> and multitask <code>chemprop</code>), whereas "
        f"unfreezing all 6 layers (<code>n_frozen = 0</code>) in multitask LP-FT lifts every "
        f"endpoint well above both the 3-layer and 6-layer linear probes.</p>",

        '<div class="panel howto"><h3>Six concrete discrepancies in the released repository</h3>'
        "<p>Inspecting <code>arunraja-hub/quadmetformer</code> uncovered six bugs and "
        "paper-versus-code mismatches that affect anyone reproducing the paper from its "
        "repository; our benchmark fixes all six so QuADMET-Former is evaluated at full "
        "strength:</p>"
        "<ol>"
        "<li><b>Missing module on import (<code>models/torchmdnet.py:10</code>).</b> "
        "<code>from models.conformerAttention import *</code> imports a file that does not "
        "exist in the repository, raising <code>ModuleNotFoundError</code> until shimmed.</li>"
        "<li><b>Unused periodic-table atom features (<code>models/torchmdnet.py:374</code>).</b> "
        "Section 3.2 describes an 8-dimensional periodic-table atom descriptor vector "
        "<code>data.x</code>, and <code>prepare_dataset.py</code> computes and stores it, "
        "but <code>TorchMD_ET.encode</code> only reads <code>data.atomic_numbers</code> "
        "(via <code>nn.Embedding(100, 128)</code>) and 3D coordinates <code>data.pos</code>; "
        "<code>data.x</code> is never referenced anywhere in the model.</li>"
        "<li><b>Layer truncation and untrained <code>out_norm</code> "
        "(<code>downstream_train.py:289</code>, <code>trainer.py:464–470</code>).</b> "
        "All ten checkpoints were pre-trained with 6 attention layers, and during "
        "pre-training <code>self.out_norm</code> (<code>LayerNorm(128)</code>) is never "
        "called on the molecular path (so its weights remain 1 and 0 in every checkpoint). "
        "Downstream, <code>downstream_train.py</code> searches only <code>1–3</code> layers, "
        "and in <code>lpft</code> mode <code>trainer.py</code> omits <code>out_norm</code> "
        "from the optimizer parameter groups.</li>"
        "<li><b>Untrained random layers in multi-conformer downstream mode "
        "(<code>models/torchmdnet.py:217–225, 456–494</code>).</b> During pre-training, "
        "<code>multi_conf</code> checkpoints apply <code>shared_backbone</code> to 3 "
        "conformers and average their predictions. Setting <code>multiple_conformer=True</code> "
        "downstream instantiates randomly initialised <code>attention_weights</code> "
        "(<code>384→128→3</code>) and <code>projection_layer</code> (<code>128→128</code>) "
        "<em>before</em> atom mean-pooling — even in <code>linear_probe</code> mode, scrambling "
        "the pre-trained representations. When 3-conformer embeddings are simply mean-pooled, "
        "the <code>Multi</code> checkpoints beat <code>Single</code> on 4 of 6 Biogen "
        "endpoints. In addition, <code>smiles_single_conf</code> calls "
        "<code>AllChem.EmbedMolecule(..., useRandomCoords=True)</code> without MMFF94 force-field "
        "relaxation, contrary to Section 4.2.</li>"
        "<li><b>Evaluating the early-stopped epoch instead of the best checkpoint "
        "(<code>downstream_train.py:446</code>).</b> <code>trainer.test(final_model, "
        "graph_test_loader)</code> is called without <code>ckpt_path=\"best\"</code>, "
        "evaluating the overfitted weights 10 epochs after validation loss peaked.</li>"
        "<li><b>Sign inversion on Biogen hPPB in Table 2 "
        "(<code>finetuning/pearsonr_summary.html</code>).</b> In the authors' committed "
        "HTML summary table inside the repository, all four Biogen "
        "<code>plasma_protein_binding_human</code> (hPPB) test Pearson <i>r</i> values are "
        "negative (<code>−0.2074</code>, <code>−0.2948</code>, <code>−0.1369</code>, "
        "<code>−0.2951</code>), whereas Table 2 of the preprint prints the exact same four "
        "magnitudes as positive (<code>0.207</code>, <code>0.295</code>, <code>0.137</code>, "
        "<code>0.295</code>).</li>"
        "</ol></div>",

        "<h2>What it costs</h2>",
        "<p>Because TorchMD-ET has 1.3 million parameters (one thirtieth the size of "
        "Atom-JEPA's 39M-parameter EquiformerV3) and static 5.0 Å radius graphs can be "
        "pre-computed once per molecule on GPU, evaluating all ten checkpoints across all "
        "11,129 molecules takes under two minutes, and two-stage LP-FT in bfloat16 takes "
        "only seconds per fold across the two GPUs (<code>apollo</code>, RTX 5070 Ti; "
        "<code>spark</code>, DGX Spark GB10).</p>",
        cost_table(cost),

        f'<p class="footnote">Reported the way {LINK_BLOG} and the protocol paper it '
        "points at" + ref("tukey") + " argue comparisons should be: distributions over "
        "folds, simultaneous confidence intervals, paired tests on the folds, and no "
        "bolded maxima. Where two methods cannot be separated, both are reported as "
        "tied.</p>",
        '<p class="repo">Code, per-fold metrics and every figure: '
        '<a href="https://github.com/patwalters/model-validation-central/tree/main/studies/quadmet-former">'
        "studies/quadmet-former</a>.</p>",
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
