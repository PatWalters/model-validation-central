#!/usr/bin/env python
"""Step 8: the report, both data sets on one self-contained page.

Fine-tuned Atom-JEPA against the arms it would have to beat, and against its
own encoder left frozen. Everything is inlined, figures included, so the file
stands alone. Every number in the prose is computed from the tables it sits
beside rather than typed, so a claim cannot drift from the evidence above it;
where the prose makes a qualitative claim, the claim is checked here and the
build fails if the data stop supporting it.

This page is deliberately not part of the site index: the study has no
study.json, so site/build_site.py neither lists it nor copies it, and it does not
enter the cross-study table. It is published on its own, by this script writing
a second copy to docs/reports/atom-jepa.html, which GitHub Pages serves.

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
PUBLISHED = cfg.PROJECT_DIR.parent.parent / "docs" / "reports" / "atom-jepa.html"
MAIN, OTHER = "expansion", "biogen"
LABEL_OF = {MAIN: "ExpansionRx", OTHER: "Biogen ADME"}
METHODS = cfg.METHODS
FT, PFN = cfg.FT_METHOD, cfg.TABPFN_METHOD
MONROE, CHEMELEON, LGBM = cfg.MONROE_METHOD, cfg.CHEMELEON_METHOD, cfg.LGBM_METHOD
COMPARATORS = [MONROE, PFN, CHEMELEON, LGBM]

COLOR = {m: page_kit.METHOD_COLOR[m] for m in METHODS}
SHORT = {LGBM: "LightGBM", CHEMELEON: "CheMeleon FT", MONROE: "Monroe/PFN3",
         PFN: "Atom-JEPA/PFN3", FT: "Atom-JEPA FT"}
LABEL = {
    **cfg.METHOD_LABELS,
    CHEMELEON: "ChemProp + CheMeleon, fine-tuned",
    PFN: "Atom-JEPA frozen + TabPFN 3",
}
METRIC_NAME = {"r2": "R²", "spearman": "Spearman ρ", "mae": "MAE"}


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
    items = list(endpoints)
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
    """Fine-tuned Atom-JEPA's wins / no call / losses against each arm, per data set."""
    rows = []
    for focus, other in [(FT, o) for o in COMPARATORS] + [(PFN, MONROE)]:
        name = (f"{LABEL[other]}" if focus == FT else
                f"<i>frozen Atom-JEPA + TabPFN 3 against</i> {LABEL[other]}")
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
        '<tr><th scope="col" rowspan="2">Atom-JEPA fine-tuned against</th>'
        f'<th colspan="3">ExpansionRx, {n_main}</th><th colspan="3">Biogen, {n_other}</th>'
        '<th colspan="2">Gain, mean of 15</th></tr>'
        "<tr><th>wins</th><th>no call</th><th>losses</th>"
        "<th>wins</th><th>no call</th><th>losses</th>"
        "<th>Δ R²</th><th>MAE drop</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>")


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
            "<th>Swap Monroe's embedding for Atom-JEPA's, same TabPFN 3</th>"
            "<th>Then fine-tune the encoder instead</th>"
            "<th>Fine-tuned Atom-JEPA against Monroe</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


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
            "<th>Δ R², Atom-JEPA FT − Monroe</th><th>MAE drop</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def cost_table(cost: pd.DataFrame) -> str:
    rows = []
    for _, r in cost.iterrows():
        rows.append(
            f'<tr><th scope="row">{r["dataset"]}</th><td style="text-align:left">'
            f'{r["host"]}, {r["gpu"].replace("NVIDIA ", "")}</td>'
            f'<td class="num">{int(r["folds"])}</td><td class="num">{r["median_min"]:.0f}</td>'
            f'<td class="num">{int(r["median_best_epoch"])}</td>'
            f'<td class="num">{r["peak_mem_gb"]:.1f}</td>'
            f'<td class="num">{r["total_h"]:.1f}</td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Data set</th>'
            '<th style="text-align:left">Machine</th><th>Folds</th><th>Median min / fold</th>'
            "<th>Median best epoch of 50</th><th>Peak GPU memory, GB</th>"
            "<th>Total, GPU hours</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# --- references ----------------------------------------------------------
REFERENCES = [
    ("atomjepa", "Petersen, K. H.; Tirsgaard, R. H.; Cornet, F. R. J.; Jordahn, M.; "
     "Schmidt, M. N. ",
     "Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems.",
     "Preprint, 2026. The method this page puts to the test, run with the authors' code "
     "(github.com/khelverskovp/atom-jepa) and their released encoder "
     "(huggingface.co/atom-jepa/atom-jepa).",
     "https://arxiv.org/abs/2610.08400", "arXiv:2610.08400"),
    ("monroe", "Banaszewski, B.; Fitzgibbon, A. W. ",
     "Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference.",
     "Preprint, 2026. The frozen encoder in the strongest arm of the source comparison.",
     "https://arxiv.org/abs/2608.18982", "arXiv:2608.18982"),
    ("tabpfn3", "Grinsztajn, L.; Flöge, K.; Key, O.; <i>et al.</i> ",
     "TabPFN-3: Technical Report.",
     "Preprint, 2026. The head of the Monroe arm and of the frozen Atom-JEPA arm.",
     "https://arxiv.org/abs/2605.13986", "arXiv:2605.13986"),
    ("chemeleon", "Burns, J. W.; Zalte, A. S.; Abreu, C. R. A.; <i>et al.</i> ",
     "Deep Learning Foundation Models for Low-Data Regimes from Classical Molecular Descriptors.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2026, articles ASAP. The pre-trained '
     "encoder of the other fine-tuned arm.",
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


def build() -> str:
    s_main, s_other = summary(MAIN), summary(OTHER)
    calls = results("calls.csv")
    lv = results("levers.csv")
    sz = results("size.csv")
    size_stats = results("size_stats.csv").set_index("Unnamed: 0")["value"]
    cost = results("cost.csv")

    def tally(focus, other, ds=None):
        g = calls[(calls["focus"] == focus) & (calls["other"] == other)]
        if ds:
            g = g[g["dataset"] == LABEL_OF[ds]]
        n = g["call"].value_counts()
        return int(n.get("win", 0)), int(n.get("no call", 0)), int(n.get("loss", 0)), len(g)

    w_m, t_m, l_m, n_m = tally(FT, MONROE, MAIN)
    w_o, t_o, l_o, n_o = tally(FT, MONROE, OTHER)

    def against_monroe(ds: str, w: int, l: int, n: int) -> str:
        """Per endpoint first, then per endpoint x metric, so neither count is mistaken
        for the other: 'ahead on 6 and behind on 3 of the 9 endpoints on all three
        metrics (18 ahead and 9 behind of 27 endpoint x metric combinations)'."""
        g = calls[(calls["focus"] == FT) & (calls["other"] == MONROE)
                  & (calls["dataset"] == LABEL_OF[ds])]
        by_ep = g.groupby("endpoint")["call"].agg(set)
        ahead = int((by_ep == {"win"}).sum())
        behind = int((by_ep == {"loss"}).sum())
        mixed = len(by_ep) - ahead - behind
        sides = [f"{side} on {k}" for side, k in (("ahead", ahead), ("behind", behind)) if k]
        text = f"{' and '.join(sides)} of the {len(by_ep)} endpoints on all three metrics"
        if mixed:
            text += f", and split on the other {WORDS.get(mixed, mixed)}"
        return text + (f" ({w} ahead and {l} behind of {n} endpoint × metric "
                       "combinations)")
    w_cm, _, l_cm, n_all = tally(FT, CHEMELEON)
    w_lg, _, l_lg, _ = tally(FT, LGBM)
    w_fz, _, l_fz, _ = tally(FT, PFN)
    w_rep, t_rep, l_rep, _ = tally(PFN, MONROE)

    tally_m, tally_o = tally_counts(s_main), tally_counts(s_other)
    top = {m: {ds: int(t.loc[m, ["alone", "tied"]].sum()) for ds, t in ((MAIN, tally_m), (OTHER, tally_o))}
           for m in METHODS}
    n_combo = {MAIN: 3 * len(cfg.DATASETS[MAIN].targets), OTHER: 3 * len(cfg.DATASETS[OTHER].targets)}
    others_on_top = [m for m in METHODS if m not in (FT, MONROE)
                     and top[m][MAIN] + top[m][OTHER] > 0]

    # Which endpoints go which way against Monroe, on MAE.
    mae_m = calls[(calls["focus"] == FT) & (calls["other"] == MONROE) & (calls["metric"] == "mae")]
    exp_lost = mae_m[(mae_m["dataset"] == "ExpansionRx") & (mae_m["call"] == "loss")]["endpoint"].tolist()
    exp_won = mae_m[(mae_m["dataset"] == "ExpansionRx") & (mae_m["call"] == "win")]["endpoint"].tolist()
    bio_won = mae_m[(mae_m["dataset"] == "Biogen ADME") & (mae_m["call"] != "loss")]["endpoint"].tolist()
    bio_lost = mae_m[(mae_m["dataset"] == "Biogen ADME") & (mae_m["call"] == "loss")]["endpoint"].tolist()

    med = lv.groupby("dataset")[["representation_r2", "finetune_r2", "net_r2"]].median()
    n_ft_up = int((lv["finetune_r2"] > 0).sum())
    n_rep_down = int((lv["representation_r2"] < 0).sum())
    n_ep = len(lv)

    # Caco-2 A>B: the arms that train on the endpoint against the two in-context ones.
    s_caco = s_main[(s_main["endpoint"] == "LOG_Caco_AB") & (s_main["metric"] == "r2")].set_index("method")
    caco = {m: (s_caco.loc[m, "mean"], s_caco.loc[m, "sd"]) for m in METHODS}

    rho = size_stats["rho_r2"]
    big_wins = sz.nlargest(2, "net_r2")
    bio_lost_n = sz[(sz["dataset"] == "Biogen ADME") & sz["endpoint"].isin(bio_lost)]["n_train"]
    n_bio_bigger = int((bio_lost_n > big_wins["n_train"].max()).sum())
    # Wins on endpoints smaller than any Biogen endpoint it loses.
    small = sz[(sz["dataset"] == "ExpansionRx") & sz["endpoint"].isin(exp_won)
               & (sz["n_train"] < bio_lost_n.min()) & ~sz["endpoint"].isin(big_wins["endpoint"])]
    small_wins, small_wins_n = small["endpoint"].tolist(), small["n_train"].tolist()
    n_models = int(cost["folds"].sum())
    gpu_h = cost["total_h"].sum()
    by_host = cost.groupby("host")["folds"].sum()

    # Qualitative claims the prose makes; fail the build if the data stop
    # supporting them.
    assert w_lg > l_lg and w_cm > l_cm, "the prose says the paper's comparison replicates"
    assert w_fz > 0 and l_fz == 0, "the prose says fine-tuning never loses to freezing"
    assert w_rep == 0 and l_rep > t_rep, "the prose says Atom-JEPA's frozen embedding never beats Monroe's"
    assert w_m > l_m and l_o > w_o, "the prose says the two data sets disagree about Monroe"
    assert top[FT][MAIN] > top[MONROE][MAIN] and top[MONROE][OTHER] > top[FT][OTHER], \
        "the prose says each data set has a different leader"
    assert abs(rho) < 0.3, "the prose says training-set size does not explain the split"
    assert all(med.loc[d, "representation_r2"] < 0 < med.loc[d, "finetune_r2"] for d in med.index), \
        "the prose says the representation costs and fine-tuning pays, on both sets"
    assert caco[MONROE][0] > max(caco[m][0] for m in METHODS if m != MONROE), \
        "the prose says Monroe is alone on Caco-2 A>B"

    parts = [
        '<meta charset="utf-8">',
        "<title>Fine-Tuning in 3D</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">3D self-supervised pre-training · 5×5 cross validation · two data sets</p>',
        "<h1>A 3D encoder, fine-tuned, against a frozen one read in context</h1>",
        '<p class="lede">Atom-JEPA' + ref("atomjepa") + " pre-trains an equivariant graph "
        "network on 19 million 3D conformers by predicting the latent representation of one "
        "part of a molecule from another, and then fine-tunes it on each data set. Its paper "
        "reports it best or tied on every ExpansionRx and Biogen endpoint against fingerprint "
        "models, Chemprop and KERMT. Run here with the authors' own code and weights on the "
        "same fifteen endpoints, folds and test sets as everything else on this site, that "
        f"claim replicates: it beats fine-tuned CheMeleon on {w_cm} of {n_all} endpoint × "
        f"metric combinations and LightGBM on {w_lg}. Against Monroe + TabPFN 3"
        + ref("monroe") + ", which the paper did not run, the two data sets disagree. On "
        f"ExpansionRx fine-tuned Atom-JEPA is {against_monroe(MAIN, w_m, l_m, n_m)}; on "
        f"Biogen it is {against_monroe(OTHER, w_o, l_o, n_o)}.</p>",
        '<div class="facts">'
        f'<div class="fact"><b>{len(METHODS)}</b><span>methods compared</span></div>'
        '<div class="fact"><b>15</b><span>endpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b><span>cross validation</span></div>'
        f'<div class="fact"><b>{n_models}</b><span>fine-tuned fold models</span></div>'
        f'<div class="fact"><b>{gpu_h:.0f}</b><span>GPU hours</span></div>'
        "</div>",

        '<div class="panel howto"><h3>The authors\' code, on these folds</h3>'
        "<p>The fine-tuned arm is the authors' <code>train_multitask</code> from "
        "github.com/khelverskovp/atom-jepa at commit bab0853, reading their released "
        "<code>molecules</code> encoder from the Hugging Face Hub, configured by their own "
        "<code>finetune_expansionrx.yaml</code> and <code>finetune_biogen_adme.yaml</code>. "
        "The recipe is theirs and is not tuned here: one multitask model per fold over every "
        "endpoint of the data set, a shared trunk and a small head per endpoint, a masked "
        "Huber loss weighted by learned per-task uncertainty, the encoder frozen for the "
        "first ten of fifty epochs, L2-SP towards the pre-trained weights, an EMA of the "
        "weights, and one of ten RDKit conformers per molecule drawn each epoch, made by "
        "their own conformer builder. The test prediction is the mean over all ten "
        "conformers, the paper's “×10 conf” arm.</p>"
        "<p>Only the split is ours. The model trains on the four fifths of the training set "
        "outside the held-out fold, picks its best epoch on the held-out fifth — exactly as "
        "the ChemProp arms use it for early stopping — and is scored on the same untouched "
        "test set as every other arm. The overrides of the authors' configuration change "
        "nothing that is trained: the targets are already on a log scale, so their log "
        "transform is off; selection and scoring use the conformer-averaged prediction; "
        "checkpoints are not written. All are listed in the study's README.</p></div>",

        '<div class="panel howto"><h3>What differs between the arms, and what does not</h3>'
        "<p>Two Atom-JEPA arms are fitted here. The fine-tuned one is the paper's. The "
        "frozen one runs the same pre-trained encoder once over each molecule's first "
        "conformer, takes its eight layers mean-pooled — the “all layers” representation of "
        "the paper's frozen baselines — and hands it to TabPFN 3" + ref("tabpfn3") + " "
        "through exactly the call the Monroe arm makes. Against Monroe that arm changes only "
        "the representation; against the fine-tuned arm it changes only how the encoder is "
        "used.</p>"
        "<p>The other three are imported from " + LINK_SOURCE + " unchanged, not re-run: "
        "LightGBM on Morgan counts" + ref("morgan") + ", the floor; ChemProp" + ref("chemprop")
        + " fine-tuned from CheMeleon" + ref("chemeleon") + ", the other pre-trained encoder "
        "that is trained rather than frozen; and Monroe + TabPFN 3, the strongest arm on "
        "this site.</p></div>",

        "<h2>The scoreboard</h2>",
        "<p>Tukey HSD per endpoint and metric, counted the way every page on this site "
        "counts: a method is on top if Tukey's correction cannot separate it from the best "
        "mean, and it is only <em>best alone</em> if nothing else is on top with it. The "
        "tables show MAE, in the units of each endpoint.</p>",
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
        f"<p>Each data set has a different leader. On ExpansionRx fine-tuned Atom-JEPA is on "
        f"top of {top[FT][MAIN]} of {n_combo[MAIN]} endpoint × metric combinations and Monroe "
        f"+ TabPFN 3 of {top[MONROE][MAIN]}. On Biogen it is the other way round: Monroe on "
        f"{top[MONROE][OTHER]} of {n_combo[OTHER]}, Atom-JEPA on {top[FT][OTHER]}. "
        + (f"Of the rest, only {where(LABEL[m] for m in others_on_top)} reaches the top of "
           "anything, " + " and ".join(f"{top[m][MAIN] + top[m][OTHER]}" for m in others_on_top)
           + " in all. " if others_on_top else "Nothing else reaches the top of anything. ")
        + "The frozen Atom-JEPA arm and LightGBM are on top of none.</p>",

        "<h2>The paper's claim replicates</h2>",
        "<p>The paper compares Atom-JEPA with fingerprint models, Chemprop, KERMT and CKERMT, "
        "and finds it best or statistically tied on all fifteen of these endpoints. Two of "
        "those comparators have a counterpart here. Paired by fold, fine-tuned Atom-JEPA is "
        f"ahead of LightGBM on Morgan counts on {w_lg} of {n_all} endpoint × metric "
        f"combinations and behind on {WORDS.get(l_lg, l_lg)}, and ahead of fine-tuned CheMeleon "
        f"on {w_cm} and behind on {WORDS.get(l_cm, l_cm)}. A win or a loss is a paired t-test at p &lt; 0.05 on one endpoint "
        "and one metric, uncorrected, so read the counts as a direction rather than a "
        "verdict on any single cell.</p>",
        calls_table(calls),

        "<h2>Against Monroe, the data sets disagree</h2>",
        f"<p>On ExpansionRx fine-tuned Atom-JEPA has the lower MAE on {where(exp_won)}, "
        f"significantly on all {count(len(exp_won), 'endpoint')}, and is significantly behind "
        f"on {where(exp_lost)}. On Biogen it is behind on {where(bio_lost)} and not behind on "
        f"{where(bio_won)} alone.</p>",
        figure("delta_atomjepa_mae", "Fine-tuned Atom-JEPA against each arm on ExpansionRx: "
               "mean paired MAE difference over 25 folds with a 95% interval, drawn in the "
               "comparator's colour and signed so that right of zero is Atom-JEPA better.", MAIN),
        figure("delta_atomjepa_mae", "The same on Biogen ADME.", OTHER),
        "<p>The obvious explanation is data volume: Biogen's folds are smaller, so perhaps "
        "fine-tuning needs more labels than an in-context head does. It does not hold. Across "
        "the fifteen endpoints the rank correlation between the number of molecules a fold "
        f"trains on and fine-tuned Atom-JEPA's margin over Monroe is {num(rho, 2, True)}. Its "
        f"two largest wins, {where(big_wins['endpoint'])}, train on "
        f"{where(f'{n:,}' for n in big_wins['n_train'])} molecules per fold, fewer than "
        f"{count(n_bio_bigger, 'Biogen endpoint')} it loses"
        + (f", and it wins {where(small_wins)} on {where(f'{n:,}' for n in small_wins_n)}"
           if small_wins else "") + ".</p>",
        size_table(sz),
        "<p>Nor is it simply the endpoints that are hard for everything. On Caco-2 A→B, the "
        "worst of fine-tuned Atom-JEPA's losses, Monroe + TabPFN 3 reaches an R² of "
        f"{num(caco[MONROE][0], 3)} ± {num(caco[MONROE][1], 3)} over the 25 folds. Every "
        "other arm is far below it: fine-tuned Atom-JEPA "
        f"{num(caco[FT][0], 3)} ± {num(caco[FT][1], 3)}, fine-tuned CheMeleon "
        f"{num(caco[CHEMELEON][0], 3)} ± {num(caco[CHEMELEON][1], 3)}, Atom-JEPA frozen "
        f"{num(caco[PFN][0], 3)} ± {num(caco[PFN][1], 3)} and LightGBM "
        f"{num(caco[LGBM][0], 3)} ± {num(caco[LGBM][1], 3)}. Whatever Monroe's embedding "
        "carries about permeability, the 3D encoder does not, frozen or trained.</p>",

        "<h2>The representation costs, the fine-tuning pays</h2>",
        "<p>The frozen Atom-JEPA arm splits the step from Monroe to fine-tuned Atom-JEPA in "
        "two. The first step keeps TabPFN 3 and swaps Monroe's frozen "
        "embedding for Atom-JEPA's. The second keeps the encoder and fine-tunes it instead of "
        "freezing it. Each column is the change in mean R² over 25 folds, positive "
        "better.</p>",
        levers_table(lv),
        f"<p>As a frozen representation Atom-JEPA is worse than Monroe on {n_rep_down} of "
        f"{n_ep} endpoints, by a median {num(med.loc['ExpansionRx', 'representation_r2'], 3, True)} "
        f"R² on ExpansionRx and {num(med.loc['Biogen ADME', 'representation_r2'], 3, True)} on "
        f"Biogen; paired by fold it is behind on {l_rep} of {n_all} endpoint × metric "
        f"combinations and ahead on {WORDS.get(w_rep, w_rep)}. Fine-tuning it then adds a median "
        f"{num(med.loc['ExpansionRx', 'finetune_r2'], 3, True)} R² on ExpansionRx and "
        f"{num(med.loc['Biogen ADME', 'finetune_r2'], 3, True)} on Biogen, improves on the "
        f"frozen version on {n_ft_up} of {n_ep} endpoints, and is behind it on "
        f"{WORDS.get(l_fz, l_fz)} of the {n_all} paired calls. On ExpansionRx "
        "that more than makes up the deficit; on Biogen it does not. What the paper's arm "
        "wins, it wins through fine-tuning, not through what pre-training put in the "
        "frozen encoder.</p>",
        figure("paired_monroe_vs_atomjepa_tabpfn_mae", "Monroe against frozen Atom-JEPA, both "
               "read by the same TabPFN 3 call, MAE paired by fold, ExpansionRx. Each line is one "
               "fold.", MAIN),

        "<h2>What it costs</h2>",
        "<p>Each fold fine-tunes the 39-million-parameter encoder for fifty epochs and scores "
        "ten conformers of every validation molecule after each one. The frozen arm is "
        "negligible next to it: the embedding is made once in about a minute per data set, "
        "and its 375 TabPFN 3 fits took about twenty minutes together. The fine-tuned folds ran on two "
        f"machines, {int(by_host.get('apollo', 0))} on an RTX 5070 Ti and "
        f"{int(by_host.get('spark', 0))} on a DGX Spark, with the same code, the same "
        "conformers and the same seeds.</p>",
        cost_table(cost),

        "<h2>What this does and does not say</h2>",
        '<div class="panel howto">'
        "<h3>Scope</h3>"
        "<p><b>One recipe.</b> The fine-tuning hyperparameters are the authors', selected on "
        "a Biogen cluster split, and are used unchanged on both data sets, as the paper uses "
        "them. Nothing is tuned for Monroe either.</p>"
        "<p><b>Not the paper's numbers.</b> The paper reports on its own splits and on its own "
        "scales for each endpoint. Here every endpoint is on the log10(x + 1) scale and "
        "the splits are this site's, so the MAEs are not comparable with its tables; the "
        "comparison with its claim is by ranking against the same kinds of comparator.</p>"
        "<p><b>Two machines.</b> The ExpansionRx folds were split between two GPUs. Each fold "
        "is trained and scored on one of them from its own seed, so the comparison is not "
        "affected, but a fold run on the other machine would not reproduce it to the last "
        "digit. On the DGX Spark the data loader runs in the main process, which changes "
        "which random conformer a step draws and nothing else.</p>"
        "</div>",

        f'<p class="footnote">Reported the way {LINK_BLOG} and the protocol paper it '
        "points at" + ref("tukey") + " argue comparisons should be: distributions over "
        "folds, simultaneous confidence intervals, paired tests on the folds, and no "
        "bolded maxima. Where two methods cannot be separated, both are reported as "
        "tied.</p>",
        '<p class="repo">Code, per-fold metrics and every figure: '
        '<a href="https://github.com/patwalters/model-validation-central/tree/main/studies/atom-jepa">'
        "studies/atom-jepa</a>.</p>",
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
    # Built on the GPU box, where the study is checked out on its own, there is no
    # docs/ to publish into; copy results/report.html across instead.
    if PUBLISHED.parent.is_dir():
        PUBLISHED.write_text(html)
        print(f"published {PUBLISHED.relative_to(PUBLISHED.parents[2])}")
    else:
        print(f"not published: no {PUBLISHED.parent} here")


if __name__ == "__main__":
    main()
