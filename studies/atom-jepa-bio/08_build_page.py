#!/usr/bin/env python
"""Step 8: the report, both data sets on one self-contained page.

Atom-JEPA-Bio, fine-tuned and frozen, against the stage-1 Atom-JEPA it continues,
against Monroe + TabPFN 3, and against the authors' own scoreboard of this
repository's protocol. Everything is inlined, figures included, so the file
stands alone. Every number in the prose is computed from the tables it sits
beside rather than typed; where the prose makes a qualitative claim, the claim
is checked here and the build fails if the data stop supporting it.

The page is listed on the site index through this study's study.json, but it
does not enter the cross-study table. This script also writes the published
copy, docs/reports/atom-jepa-bio.html.

    for ds in expansion biogen; do
      ADME_DATASET=$ds python 05_collect_metrics.py
      ADME_DATASET=$ds python 06_report.py
    done
    python 07_summary.py
    python 08_build_page.py
"""

import pandas as pd

import config as cfg
import page_kit

PAGE = cfg.PROJECT_DIR / "results" / "report.html"
PUBLISHED = cfg.PROJECT_DIR.parent.parent / "docs" / "reports" / "atom-jepa-bio.html"
MAIN, OTHER = "expansion", "biogen"
LABEL_OF = {MAIN: "ExpansionRx", OTHER: "Biogen ADME"}
METHODS = cfg.METHODS
FT, PFN = cfg.FT_METHOD, cfg.TABPFN_METHOD
BASE_FT, BASE_PFN = cfg.BASE_FT_METHOD, cfg.BASE_TABPFN_METHOD
MONROE, CHEMELEON, LGBM = cfg.MONROE_METHOD, cfg.CHEMELEON_METHOD, cfg.LGBM_METHOD
PAPER_ARMS = [FT, PFN, MONROE, BASE_FT, LGBM, CHEMELEON]
# The authors' page: best or tied of 27 ExpansionRx and 18 Biogen combinations.
PAPER_CLAIM = {MAIN: {FT: 20, MONROE: 9}, OTHER: {FT: 14, MONROE: 9}}

COLOR = {m: page_kit.METHOD_COLOR[m] for m in METHODS}
METRIC_NAME = {"r2": "R²", "spearman": "Spearman ρ", "mae": "MAE"}
SHORT = {LGBM: "LightGBM", CHEMELEON: "CheMeleon FT", MONROE: "Monroe/PFN3",
         BASE_PFN: "Atom-JEPA/PFN3", BASE_FT: "Atom-JEPA FT",
         PFN: "Bio/PFN3", FT: "Bio FT"}
LABEL = {
    **cfg.METHOD_LABELS,
    CHEMELEON: "ChemProp + CheMeleon, fine-tuned",
    BASE_PFN: "Atom-JEPA frozen + TabPFN 3",
    BASE_FT: "Atom-JEPA fine-tuned",
    PFN: "Atom-JEPA-Bio frozen + TabPFN 3",
    FT: "Atom-JEPA-Bio fine-tuned",
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


def word(n: int) -> str:
    return WORDS.get(n, str(n))


def count(n: int, noun: str) -> str:
    """'one endpoint', 'six endpoints'."""
    return f"{word(n)} {noun}{'' if n == 1 else 's'}"


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


CALL_ROWS = [(FT, BASE_FT), (PFN, BASE_PFN), (FT, MONROE), (BASE_FT, MONROE), (PFN, MONROE),
             (FT, PFN), (FT, CHEMELEON), (FT, LGBM)]


def calls_table(calls: pd.DataFrame) -> str:
    """Wins / no call / losses for each pair, per data set."""
    rows = []
    for focus, other in CALL_ROWS:
        cells = [f'<th scope="row"><span class="swatch" style="background:{COLOR[focus]}"></span>'
                 f'{LABEL[focus]} <i>against</i> '
                 f'<span class="swatch" style="background:{COLOR[other]}"></span>{LABEL[other]}</th>']
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
        '<tr><th scope="col" rowspan="2">Pair</th>'
        f'<th colspan="3">ExpansionRx, {n_main}</th><th colspan="3">Biogen, {n_other}</th>'
        '<th colspan="2">Gain, mean of 15</th></tr>'
        "<tr><th>wins</th><th>no call</th><th>losses</th>"
        "<th>wins</th><th>no call</th><th>losses</th>"
        "<th>Δ R²</th><th>MAE drop</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>")


def paper_table(top: pd.DataFrame) -> str:
    """Best-or-tied counts over the authors' six arms, beside the authors' own."""
    rows = []
    for m in PAPER_ARMS:
        cells = [f'<th scope="row"><span class="swatch" style="background:{COLOR[m]}"></span>'
                 f"{LABEL[m]}</th>"]
        for ds in (MAIN, OTHER):
            claim = PAPER_CLAIM[ds].get(m)
            cells.append(f'<td class="num">{"" if claim is None else claim}</td>'
                         f'<td class="num"><b>{int(top.loc[m, LABEL_OF[ds]])}</b></td>')
        claim = sum(PAPER_CLAIM[ds][m] for ds in PAPER_CLAIM) if m in PAPER_CLAIM[MAIN] else None
        cells.append(f'<td class="num">{"" if claim is None else claim}</td>'
                     f'<td class="num"><b>{int(top.loc[m].sum())}</b></td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return ('<div class="bleed tablewrap"><table><thead>'
            '<tr><th scope="col" rowspan="2">Best or tied for best</th>'
            '<th colspan="2">ExpansionRx, of 27</th><th colspan="2">Biogen, of 18</th>'
            '<th colspan="2">Both, of 45</th></tr>'
            "<tr><th>authors</th><th>here</th><th>authors</th><th>here</th>"
            "<th>authors</th><th>here</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def levers_table(lv: pd.DataFrame) -> str:
    cols = ["bio_stage_frozen_r2", "bio_stage_ft_r2", "representation_r2", "finetune_r2", "net_r2"]
    rows = []
    for _, r in lv.iterrows():
        rows.append(f'<tr><th scope="row">{r["endpoint"]}</th><td>{r["dataset"]}</td>'
                    + "".join(f'<td class="num">{num(r[c], 3, True)}</td>' for c in cols[:-1])
                    + f'<td class="num"><b>{num(r["net_r2"], 3, True)}</b></td></tr>')
    for ds_label, g in lv.groupby("dataset", sort=False):
        med = g[cols].median()
        rows.append(f'<tr><th scope="row"><b>median</b></th><td>{ds_label}</td>'
                    + "".join(f'<td class="num"><b>{num(med[c], 3, True)}</b></td>' for c in cols)
                    + "</tr>")
    return ('<div class="bleed tablewrap"><table><thead><tr>'
            '<th scope="col">Endpoint</th><th style="text-align:left">Data set</th>'
            "<th>Bioassay stage, frozen: Bio − Atom-JEPA, both + TabPFN 3</th>"
            "<th>Bioassay stage, fine-tuned: Bio − Atom-JEPA</th>"
            "<th>Monroe's embedding → Bio's, same TabPFN 3</th>"
            "<th>Then fine-tune Bio instead</th>"
            "<th>Bio fine-tuned − Monroe</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def rerun_table(rr: pd.DataFrame) -> str:
    rows = []
    g = rr[rr["metric"] == "r2"].set_index("endpoint").reindex(cfg.DATASETS[OTHER].targets)
    for endpoint, r in g.iterrows():
        rows.append(f'<tr><th scope="row">{endpoint}</th>'
                    f'<td class="num">{num(r["imported"])}</td><td class="num">{num(r["rerun"])}</td>'
                    f'<td class="num">{num(r["rerun_minus_imported"], 3, True)}</td>'
                    f'<td class="num">{num(r["bio"])}</td>'
                    f'<td class="num"><b>{num(r["bio_minus_imported"], 3, True)}</b></td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Endpoint</th>'
            "<th>Atom-JEPA FT, imported</th><th>Atom-JEPA FT, re-run here</th>"
            "<th>Re-run − imported</th><th>Bio FT</th><th>Bio − imported</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def exposure_table(ex: pd.DataFrame) -> str:
    arms = [FT, BASE_FT, MONROE]
    t = ex.groupby(["endpoint", "subset", "method"])[["mae", "n"]].mean()
    rows = []
    for endpoint in cfg.DATASETS[OTHER].targets:
        for subset in ("not in PCBA", "in PCBA"):
            r = t.loc[(endpoint, subset)]
            cells = [f'<th scope="row">{endpoint}</th><td style="text-align:left">{subset}</td>'
                     f'<td class="num">{int(r.loc[FT, "n"]):,}</td>']
            cells += [f'<td class="num">{num(r.loc[m, "mae"])}</td>' for m in arms]
            d_base = r.loc[FT, "mae"] - r.loc[BASE_FT, "mae"]
            d_mon = r.loc[FT, "mae"] - r.loc[MONROE, "mae"]
            cells += [f'<td class="num {"best" if d < 0 else "worse"}">{num(d, 3, True)}</td>'
                      for d in (d_base, d_mon)]
            rows.append("<tr>" + "".join(cells) + "</tr>")
    head = "".join(f"<th>{SHORT[m]} MAE</th>" for m in arms)
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Endpoint</th>'
            '<th style="text-align:left">Test molecules</th><th>n</th>' + head
            + "<th>Bio − Atom-JEPA</th><th>Bio − Monroe</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def cost_table(cost: pd.DataFrame) -> str:
    rows = []
    for _, r in cost.iterrows():
        rows.append(
            f'<tr><th scope="row">{r["dataset"]}</th><td style="text-align:left">{LABEL[r["method"]] if r["method"] in LABEL else r["method"]}</td>'
            f'<td style="text-align:left">{r["host"]}, {r["gpu"].replace("NVIDIA ", "")}</td>'
            f'<td class="num">{int(r["folds"])}</td><td class="num">{r["median_min"]:.0f}</td>'
            f'<td class="num">{int(r["median_best_epoch"])}</td>'
            f'<td class="num">{r["peak_mem_gb"]:.1f}</td>'
            f'<td class="num">{r["total_h"]:.1f}</td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Data set</th>'
            '<th style="text-align:left">Arm</th><th style="text-align:left">Machine</th>'
            "<th>Folds</th><th>Median min / fold</th>"
            "<th>Median best epoch of 50</th><th>Peak GPU memory, GB</th>"
            "<th>Total, GPU hours</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# --- references ----------------------------------------------------------
REFERENCES = [
    ("bio", "Petersen, K. H.; Tirsgaard, R. H.; Larsen, L. F.; Cornet, F. R. J.; Jordahn, M.; "
     "Schmidt, M. N. ",
     "Atom-JEPA-Bio: Improving Atom-JEPA's ADMET Predictions with Second-Stage Pretraining.",
     "Blog post, November 2026. The method and the claim this page puts to the test, run "
     "with the authors' code (github.com/khelverskovp/atom-jepa) and their released "
     "<code>bio</code> encoder (huggingface.co/atom-jepa/atom-jepa).",
     "https://mjordahn.github.io/atom-jepa-website/atom-jepa-bio.html", "atom-jepa-website"),
    ("atomjepa", "Petersen, K. H.; Tirsgaard, R. H.; Cornet, F. R. J.; Jordahn, M.; "
     "Schmidt, M. N. ",
     "Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems.",
     "Preprint, 2026. The stage-1 encoder Atom-JEPA-Bio continues.",
     "https://arxiv.org/abs/2610.08400", "arXiv:2610.08400"),
    ("pcba", "Beaini, D.; Huang, S.; Cunha, J. A.; <i>et al.</i> ",
     "Towards Foundational Models for Molecular Learning on Large-Scale Multi-Task Datasets.",
     "ICLR 2024. The 1,328-assay PCBA table both pre-trainings draw on (Zenodo 8024997).",
     "https://arxiv.org/abs/2310.04292", "arXiv:2310.04292"),
    ("monroe", "Banaszewski, B.; Fitzgibbon, A. W. ",
     "Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference.",
     "Preprint, 2026. The frozen encoder of the strongest arm on this site, pre-trained on "
     "the same PCBA preparation.",
     "https://arxiv.org/abs/2608.18982", "arXiv:2608.18982"),
    ("tabpfn3", "Grinsztajn, L.; Flöge, K.; Key, O.; <i>et al.</i> ",
     "TabPFN-3: Technical Report.",
     "Preprint, 2026. The head of the three frozen arms.",
     "https://arxiv.org/abs/2605.13986", "arXiv:2605.13986"),
    ("morgan", "Rogers, D.; Hahn, M. ", "Extended-Connectivity Fingerprints.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2010, 50 (5), 742–754.',
     "https://doi.org/10.1021/ci100050t", "10.1021/ci100050t"),
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
LINK_SOURCE = ('<a href="expansion-ml-comparison.html">the nine-method foundation-model '
               "comparison</a>")
LINK_STAGE1 = '<a href="atom-jepa.html">the Atom-JEPA study</a>'


def build() -> str:
    s_main, s_other = summary(MAIN), summary(OTHER)
    calls = results("calls.csv")
    lv = results("levers.csv")
    rr = results("rerun.csv")
    ex = results("exposure.csv")
    pt = results("paper_tally.csv")
    sz = results("size.csv")
    size_stats = results("size_stats.csv").set_index("Unnamed: 0")["value"]
    cost = results("cost.csv")
    ov = {ds: pd.read_csv(cfg.paths(ds).results / "pcba_overlap.csv") for ds in (MAIN, OTHER)}

    def tally(focus, other, ds=None):
        g = calls[(calls["focus"] == focus) & (calls["other"] == other)]
        if ds:
            g = g[g["dataset"] == LABEL_OF[ds]]
        n = g["call"].value_counts()
        return int(n.get("win", 0)), int(n.get("no call", 0)), int(n.get("loss", 0)), len(g)

    def per_endpoint(focus, other, ds) -> tuple[list, list, list]:
        """Endpoints where focus wins on every metric, loses on every metric, or neither."""
        g = calls[(calls["focus"] == focus) & (calls["other"] == other)
                  & (calls["dataset"] == LABEL_OF[ds])]
        by_ep = g.groupby("endpoint", sort=False)["call"].agg(set)
        ahead = [e for e, c in by_ep.items() if c == {"win"}]
        behind = [e for e, c in by_ep.items() if c == {"loss"}]
        mixed = [e for e in by_ep.index if e not in ahead + behind]
        return ahead, behind, mixed

    def against(focus, other, ds) -> str:
        """'ahead on 6 and behind on 3 of the 9 endpoints on all three metrics, split on
        the other none (18 ahead and 9 behind of 27 endpoint x metric combinations)'."""
        w, _, l, n = tally(focus, other, ds)
        ahead, behind, mixed = per_endpoint(focus, other, ds)
        n_ep_ds = len(ahead) + len(behind) + len(mixed)
        if len(ahead) == n_ep_ds:
            return (f"ahead on all {word(n_ep_ds)} endpoints on all three metrics ({w} of {n} "
                    "endpoint × metric combinations)")
        sides = [f"{side} on {word(len(k))}" for side, k in (("ahead", ahead), ("behind", behind)) if k]
        text = (f"{' and '.join(sides) or 'on neither side'} of the "
                f"{len(ahead) + len(behind) + len(mixed)} endpoints on all three metrics")
        if mixed:
            text += f" and split on the other {word(len(mixed))}"
        return text + f" ({w} ahead and {l} behind of {n} endpoint × metric combinations)"

    # The bioassay stage.
    stage = {ds: tally(FT, BASE_FT, ds) for ds in (MAIN, OTHER)}
    stage_frozen = {ds: tally(PFN, BASE_PFN, ds) for ds in (MAIN, OTHER)}
    med = lv.groupby("dataset")[["bio_stage_ft_r2", "bio_stage_frozen_r2", "representation_r2",
                                 "finetune_r2", "net_r2"]].median()
    n_ep = len(lv)
    n_stage_up = int((lv["bio_stage_ft_r2"] > 0).sum())
    n_frozen_up = int((lv["bio_stage_frozen_r2"] > 0).sum())

    # Against Monroe, before and after.
    mon = {ds: tally(FT, MONROE, ds) for ds in (MAIN, OTHER)}
    mon_base = {ds: tally(BASE_FT, MONROE, ds) for ds in (MAIN, OTHER)}
    mon_frozen = {ds: tally(PFN, MONROE, ds) for ds in (MAIN, OTHER)}
    lost = {ds: per_endpoint(FT, MONROE, ds)[1] for ds in (MAIN, OTHER)}
    w_cm, _, l_cm, n_all = tally(FT, CHEMELEON)
    w_lg, _, l_lg, _ = tally(FT, LGBM)
    w_fz, _, l_fz, _ = tally(FT, PFN)

    # Scoreboards: ours, and the authors' six arms.
    tally_m, tally_o = tally_counts(s_main), tally_counts(s_other)
    top = {m: {ds: int(t.loc[m, ["alone", "tied"]].sum()) for ds, t in ((MAIN, tally_m), (OTHER, tally_o))}
           for m in METHODS}
    n_combo = {MAIN: 3 * len(cfg.DATASETS[MAIN].targets), OTHER: 3 * len(cfg.DATASETS[OTHER].targets)}
    paper_top = (pt[pt["kind"] != "worse"].groupby(["method", "dataset"]).size().unstack()
                 .reindex(index=PAPER_ARMS, columns=[LABEL_OF[MAIN], LABEL_OF[OTHER]]).fillna(0)
                 .astype(int))
    ours_ft, ours_mon = int(paper_top.loc[FT].sum()), int(paper_top.loc[MONROE].sum())
    claim_ft = sum(PAPER_CLAIM[d][FT] for d in PAPER_CLAIM)
    claim_mon = sum(PAPER_CLAIM[d][MONROE] for d in PAPER_CLAIM)

    # The machine check.
    rr2 = rr[rr["metric"] == "r2"]
    rerun_gap = rr2["rerun_minus_imported"].abs().max()
    bio_gap_min = rr2["bio_minus_imported"].min()
    rerun_folds = int(rr2["folds"].iloc[0])

    # Exposure: Biogen test molecules PCBA has an exact InChI for.
    test_o = ov[OTHER][ov[OTHER][cfg.SET_COL] == "test"]
    test_m = ov[MAIN][ov[MAIN][cfg.SET_COL] == "test"]
    n_seen, n_test = int(test_o["pcba_exact"].sum()), len(test_o)
    n_seen_m, n_test_m = int(test_m["pcba_exact"].sum()), len(test_m)
    n_sol = int(test_o["aid1996"].notna().sum())
    e = ex.groupby(["subset", "endpoint", "method"])["mae"].mean()
    unseen = e.loc["not in PCBA"].unstack()
    seen = e.loc["in PCBA"].unstack()
    unseen_vs_base = int((unseen[FT] < unseen[BASE_FT]).sum())
    unseen_vs_mon = unseen.index[unseen[FT] < unseen[MONROE]].tolist()
    harder = int((seen[MONROE] > unseen[MONROE]).sum())

    # Caco-2 A>B.
    s_caco = s_main[(s_main["endpoint"] == "LOG_Caco_AB") & (s_main["metric"] == "r2")].set_index("method")
    caco = {m: (s_caco.loc[m, "mean"], s_caco.loc[m, "sd"]) for m in METHODS}
    rho = size_stats["rho_r2"]

    n_models = int(cost["folds"].sum())
    gpu_h = cost["total_h"].sum()
    bio_cost = cost[cost["method"] == FT]
    by_host = bio_cost.groupby("host")["folds"].sum()

    stage_losses = calls[(calls["focus"] == FT) & (calls["other"] == BASE_FT)
                         & (calls["call"] == "loss")]
    loss_text = "; ".join(f"{r.endpoint} {METRIC_NAME[r.metric]} on {r.dataset}, by "
                          f"{num(abs(r.focus_mean - r.other_mean), 4)} with {r.folds_won} of 25 "
                          f"folds better" for r in stage_losses.itertuples())
    partial = {ds: [e for e in per_endpoint(FT, MONROE, ds)[2]
                    if "loss" in set(calls[(calls["focus"] == FT) & (calls["other"] == MONROE)
                                           & (calls["dataset"] == LABEL_OF[ds])
                                           & (calls["endpoint"] == e)]["call"])]
               for ds in (MAIN, OTHER)}
    old_cost = pd.read_csv(cfg.PROJECT_DIR.parent / "atom-jepa" / "results" / "cost.csv")
    old_min = old_cost.set_index(["dataset", "host"])["median_min"]
    new_min = bio_cost.set_index(["dataset", "host"])["median_min"]

    def side(t) -> str:
        w, _, l, n = t
        return "ahead of" if w > l + n // 6 else ("behind" if l > w + n // 6 else "level with")

    lost_base = {ds: per_endpoint(BASE_FT, MONROE, ds)[1] for ds in (MAIN, OTHER)}

    # Qualitative claims the prose makes; fail the build if the data stop
    # supporting them.
    assert abs(ours_ft - claim_ft) <= 1 and abs(ours_mon - claim_mon) <= 1, \
        "the prose says the count is within one of the authors'"
    assert sum(stage[d][2] for d in stage) <= 1 and all(stage[d][0] > 5 * stage[d][2] for d in stage), \
        "the prose says the bioassay stage almost never makes fine-tuning worse"
    assert all(stage_frozen[d][0] > stage_frozen[d][2] for d in stage_frozen), \
        "the prose says the bioassay stage improves the frozen encoder on both sets"
    assert med.loc[LABEL_OF[OTHER], "bio_stage_ft_r2"] > med.loc[LABEL_OF[MAIN], "bio_stage_ft_r2"], \
        "the prose says the stage buys more on Biogen"
    assert all(mon[d][0] > mon[d][2] for d in mon), "the prose says Bio leads Monroe on both sets"
    assert mon_base[OTHER][2] > mon_base[OTHER][0], "the prose says stage 1 trailed Monroe on Biogen"
    assert w_lg > l_lg and w_cm > l_cm, "the prose says Bio is ahead of LightGBM and CheMeleon"
    assert rerun_gap < bio_gap_min / 3, "the prose says the machine moves results far less than Bio"
    assert unseen_vs_base == len(unseen), "the prose says Bio beats stage 1 on unseen molecules everywhere"
    assert caco[MONROE][0] > caco[FT][0], "the prose says Caco-2 A>B stays Monroe's"
    any_loss_base = {ds: set(calls[(calls["focus"] == BASE_FT) & (calls["other"] == MONROE)
                                  & (calls["dataset"] == LABEL_OF[ds]) & (calls["call"] == "loss")]["endpoint"])
                     for ds in (MAIN, OTHER)}
    assert all(set(partial[d]) <= any_loss_base[d] for d in partial), \
        "the prose says Bio's partial losses to Monroe are where stage 1 lost"
    assert all(set(lost[d]) <= set(lost_base[d]) for d in lost), \
        "the prose says Bio loses to Monroe only where stage 1 lost"
    assert harder > len(unseen) / 2, "the prose says the PCBA molecules are harder"

    pct = 100 * n_seen / n_test
    parts = [
        '<meta charset="utf-8">',
        "<title>Bioassays Before ADMET</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">Second-stage bioassay pre-training · 5×5 cross validation · two data sets</p>',
        "<h1>What a PCBA stage buys a 3D encoder</h1>",
        '<p class="lede">Atom-JEPA-Bio' + ref("bio") + " continues the released Atom-JEPA "
        "encoder" + ref("atomjepa") + " with a second pre-training stage that adds a supervised "
        "loss over 1,089 PCBA bioassays" + ref("pcba") + " to its self-supervised objectives, "
        "and fine-tunes it on ADMET exactly as before. Its authors scored it on this site's "
        f"protocol and report it best or tied on {claim_ft} of 45 endpoint × metric "
        f"combinations, against {claim_mon} for Monroe + TabPFN 3" + ref("monroe") + ". Run "
        "here with their code and weights, on the same folds, the count comes out at "
        f"{ours_ft} against {ours_mon}. Only the checkpoint differs from " + LINK_STAGE1
        + ", so the bioassay stage can be measured directly: fine-tuned, it is ahead of "
        f"stage 1 on {stage[MAIN][0] + stage[OTHER][0]} of the 45 paired calls and behind on "
        f"{word(stage[MAIN][2] + stage[OTHER][2])}. On Biogen it turns a deficit against Monroe into "
        f"a lead: stage 1 was behind on {mon_base[OTHER][2]} of {n_combo[OTHER]} combinations, "
        f"Bio is ahead on {mon[OTHER][0]}.</p>",
        '<div class="facts">'
        f'<div class="fact"><b>{len(METHODS)}</b><span>methods compared</span></div>'
        '<div class="fact"><b>15</b><span>endpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b><span>cross validation</span></div>'
        f'<div class="fact"><b>{n_models}</b><span>fine-tuned fold models</span></div>'
        f'<div class="fact"><b>{gpu_h:.0f}</b><span>GPU hours</span></div>'
        "</div>",

        '<div class="panel howto"><h3>One checkpoint swapped</h3>'
        "<p>The fine-tuned arm is the authors' <code>train_multitask</code> from "
        "github.com/khelverskovp/atom-jepa at commit 64d539b, whose fine-tuning configuration "
        "is identical to the commit the stage-1 study ran. It reads the <code>bio</code> "
        "encoder from the Hugging Face Hub in place of <code>molecules</code>, and nothing "
        "else changes: one multitask model per fold, the encoder frozen for ten of fifty "
        "epochs, ten RDKit conformers per molecule — the very conformer files the stage-1 "
        "study used — and the test prediction averaged over all ten. The best epoch is picked "
        "on the held-out fifth of the training set and every arm is scored on the same "
        "untouched test set.</p>"
        "<p>Two frozen arms read the stage-1 and the Bio encoder, eight layers mean-pooled, "
        "through the TabPFN 3" + ref("tabpfn3") + " call the Monroe arm makes; both are fitted "
        "here, on the same machine. LightGBM on Morgan counts" + ref("morgan") + ", ChemProp "
        "fine-tuned from CheMeleon" + ref("chemeleon") + " and Monroe + TabPFN 3 are imported "
        "from " + LINK_SOURCE + ", and stage-1 Atom-JEPA fine-tuned from " + LINK_STAGE1
        + ", none of them re-run.</p></div>",

        "<h2>The scoreboard</h2>",
        "<p>Tukey HSD per endpoint and metric over all seven arms, counted the way every page "
        "on this site counts: a method is on top if Tukey's correction cannot separate it "
        "from the best mean, and only <em>best alone</em> if nothing else is on top with it. "
        "The tables show MAE, in the units of each endpoint.</p>",
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
        f"<p>Fine-tuned Bio is on top of {top[FT][MAIN]} of {n_combo[MAIN]} ExpansionRx "
        f"combinations and {top[FT][OTHER]} of {n_combo[OTHER]} Biogen ones; Monroe + TabPFN 3 "
        f"of {top[MONROE][MAIN]} and {top[MONROE][OTHER]}; stage-1 Atom-JEPA fine-tuned of "
        f"{top[BASE_FT][MAIN]} and {top[BASE_FT][OTHER]}.</p>",

        "<h2>The authors' count, recomputed</h2>",
        "<p>The authors' tables compare six arms — the stage-1 frozen arm is not among them — "
        "and import the comparators' per-fold results from this site, as this study does. "
        "Restricting the Tukey test to the same six arms, so that the intervals are as wide "
        "as theirs, gives the counts below beside the ones on their page.</p>",
        paper_table(paper_top),
        f"<p>{'The count reproduces' if (ours_ft, ours_mon) == (claim_ft, claim_mon) else 'The count is within one of theirs'}: "
        f"{ours_ft} of 45 for fine-tuned Bio against their {claim_ft}, {ours_mon} for Monroe "
        f"against their {claim_mon}. The fine-tuned folds here are a separate training of "
        "the same recipe, not the authors' fold models, so an exact match was not to be "
        "expected; the comparators' folds are the same in both.</p>",

        "<h2>The bioassay stage, measured</h2>",
        "<p>Paired by fold, fine-tuned Bio against fine-tuned stage-1 Atom-JEPA is "
        f"{against(FT, BASE_FT, MAIN)} on ExpansionRx and {against(FT, BASE_FT, OTHER)} on "
        f"Biogen. "
        + (f"The {'loss is' if len(stage_losses) == 1 else 'losses are'} {loss_text}, one "
           f"of the pair's {n_all} uncorrected tests. " if len(stage_losses) else "")
        + f"It raises mean R² on {n_stage_up} of {n_ep} endpoints, by a median "
        f"{num(med.loc[LABEL_OF[MAIN], 'bio_stage_ft_r2'], 3, True)} on ExpansionRx and "
        f"{num(med.loc[LABEL_OF[OTHER], 'bio_stage_ft_r2'], 3, True)} on Biogen. Frozen and read "
        "by TabPFN 3 the stage does more: the Bio embedding is "
        f"ahead of the stage-1 one on {stage_frozen[MAIN][0] + stage_frozen[OTHER][0]} of "
        f"{n_all} combinations and behind on "
        f"{word(stage_frozen[MAIN][2] + stage_frozen[OTHER][2])}, a median "
        f"{num(med.loc[LABEL_OF[MAIN], 'bio_stage_frozen_r2'], 3, True)} R² on ExpansionRx and "
        f"{num(med.loc[LABEL_OF[OTHER], 'bio_stage_frozen_r2'], 3, True)} on Biogen, and better "
        f"on {n_frozen_up} of {n_ep} endpoints.</p>",
        calls_table(calls),
        figure("delta_bio_mae", "Fine-tuned Bio against each arm on ExpansionRx: mean paired "
               "MAE difference over 25 folds with a 95% interval, drawn in the comparator's "
               "colour and signed so that right of zero is Bio better.", MAIN),
        figure("delta_bio_mae", "The same on Biogen ADME.", OTHER),

        "<h3>Not the machine</h3>",
        "<p>Stage-1 Atom-JEPA was fine-tuned for the earlier study on an RTX 5070 Ti and a DGX "
        "Spark, and every Bio fold on Biogen on an RTX PRO 5000. To see how far a change of machine moves the same recipe on its "
        f"own, the stage-1 arm was re-trained here on the {rerun_folds} folds of Biogen's first "
        "repeat. The re-run lands within "
        f"{num(rerun_gap, 3)} R² of the imported folds on every endpoint; Bio is at least "
        f"{num(bio_gap_min, 3)} above them on every endpoint.</p>",
        rerun_table(rr),

        "<h2>Against Monroe</h2>",
        f"<p>Fine-tuned Bio is {against(FT, MONROE, MAIN)} on ExpansionRx, and "
        f"{against(FT, MONROE, OTHER)} on Biogen. Stage-1 Atom-JEPA, against the same Monroe "
        f"folds, had been {mon_base[MAIN][0]} ahead and {mon_base[MAIN][2]} behind on "
        f"ExpansionRx and {mon_base[OTHER][0]} ahead and {mon_base[OTHER][2]} behind on Biogen. "
        f"Frozen, the Bio embedding is {side(mon_frozen[OTHER])} Monroe's on Biogen "
        f"({mon_frozen[OTHER][0]} ahead, {mon_frozen[OTHER][2]} behind) and "
        f"{side(mon_frozen[MAIN])} it on ExpansionRx ({mon_frozen[MAIN][0]} ahead, "
        f"{mon_frozen[MAIN][2]} behind); fine-tuning "
        f"is still what wins, ahead of the frozen Bio arm on {w_fz} of {n_all} paired calls and "
        f"behind on {word(l_fz)}.</p>",
        levers_table(lv),
        "<p>Where it still loses is where stage 1 lost. "
        + (f"On ExpansionRx it is behind Monroe on every metric of {where(lost[MAIN])}"
           if lost[MAIN] else "On ExpansionRx it is behind on no endpoint outright")
        + (f", on Biogen on {where(lost[OTHER])}" if lost[OTHER] else "")
        + (f", and on some metrics of {where(partial[MAIN] + partial[OTHER])}. "
           if partial[MAIN] + partial[OTHER] else ". ")
        + "Caco-2 A→B remains Monroe's: an R² of "
        f"{num(caco[MONROE][0])} ± {num(caco[MONROE][1])} over the 25 folds against "
        f"{num(caco[FT][0])} ± {num(caco[FT][1])} for fine-tuned Bio and "
        f"{num(caco[BASE_FT][0])} ± {num(caco[BASE_FT][1])} for stage 1. Across the fifteen "
        "endpoints, the rank correlation between molecules per training fold and Bio's margin "
        f"over Monroe is {num(rho, 2, True)}.</p>",
        figure("paired_monroe_vs_bio_ft_mae", "Monroe + TabPFN 3 against fine-tuned Bio, MAE "
               "paired by fold, Biogen ADME. Each line is one fold.", OTHER),

        "<h2>Was it pre-trained on the test set?</h2>",
        "<p>Stage 2 trains on PCBA as Monroe's pipeline prepares it, which moves the test "
        "splits of the public benchmarks out of training — but those are the Polaris splits of "
        "the Biogen data, not this site's. Matching every test molecule to the PCBA table by "
        f"standard InChI, Monroe's own deduplication key, finds {n_seen} of Biogen's "
        f"{n_test:,} test molecules ({pct:.0f}%) and {word(n_seen_m) if n_seen_m < 16 else n_seen_m} "
        f"of ExpansionRx's {n_test_m:,}. That is an upper bound on what reached training. The "
        "assays are rarely ADMET: one solubility assay, AID 1996, labels "
        f"{n_sol} of the matched Biogen molecules. Monroe's pre-training draws on the same "
        "table, so any exposure is shared by the arm Bio is compared with.</p>",
        "<p>Scoring the Biogen test set in two halves:</p>",
        exposure_table(ex),
        "<p>On the molecules PCBA does not have, fine-tuned Bio is still below stage 1 in MAE "
        f"on all {word(unseen_vs_base)} endpoints and below Monroe on "
        f"{where(unseen_vs_mon)}. The molecules PCBA has are harder, not easier: "
        + (f"Monroe's MAE is higher on them on all {word(harder)} endpoints." if harder == len(unseen)
           else f"Monroe's MAE is higher on them on {word(harder)} of {word(len(unseen))} endpoints.")
        + " The gain does "
        "not come from having seen the test set.</p>",

        "<h2>What it costs</h2>",
        "<p>The recipe is unchanged, and so is the cost: a median "
        f"{new_min[(LABEL_OF[MAIN], 'apollo')]:.0f} minutes per ExpansionRx fold on the RTX 5070 Ti "
        f"against {old_min[(LABEL_OF[MAIN], 'apollo')]:.0f} for stage 1 on the same card. The Bio folds ran on two machines, {int(by_host.get('mercury', 0))} on an RTX "
        f"PRO 5000 Blackwell and {int(by_host.get('apollo', 0))} on an RTX 5070 Ti, each "
        "fold from its own seed, with the claimed folds of each machine skipped by the other.</p>",
        cost_table(cost),

        "<h2>What this does and does not say</h2>",
        '<div class="panel howto">'
        "<h3>Scope</h3>"
        "<p><b>One recipe.</b> The fine-tuning hyperparameters are the authors' and are used "
        "unchanged on both data sets, as in the stage-1 study. Nothing is tuned for Monroe "
        "either.</p>"
        "<p><b>Conformers are pinned.</b> RDKit's conformer generator does not give the same "
        "conformers across releases even with a fixed seed: a newer RDKit changed them for "
        "about a third of the Biogen molecules, and that alone moved the frozen stage-1 arm by "
        "up to 0.10 R² on the plasma-binding endpoints. Every Atom-JEPA arm here reads the "
        "stage-1 study's conformer files, checked by hash.</p>"
        "<p><b>An upper bound on exposure.</b> The PCBA match is by structure, before the "
        "authors' hold-out of the benchmark test splits, which could not be reproduced here. "
        "The real overlap with what stage 2 trained on is at most what is reported.</p>"
        "</div>",

        f'<p class="footnote">Reported the way {LINK_BLOG} and the protocol paper it '
        "points at" + ref("tukey") + " argue comparisons should be: distributions over "
        "folds, simultaneous confidence intervals, paired tests on the folds, and no "
        "bolded maxima. A win or a loss in the paired tables is a t-test at p &lt; 0.05 "
        "on one endpoint and one metric, uncorrected.</p>",
        '<p class="repo">Code, per-fold metrics and every figure: '
        '<a href="https://github.com/patwalters/model-validation-central/tree/main/studies/atom-jepa-bio">'
        "studies/atom-jepa-bio</a>.</p>",
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
