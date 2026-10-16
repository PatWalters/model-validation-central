#!/usr/bin/env python
"""Step 6: the report, both data sets on one self-contained page.

MolPAIR against the arms it would have to beat, and against the prediction it
corrects. Everything is inlined, figures included, so the file stands alone.
Every number in the prose is computed from the tables it sits beside rather
than typed, so a claim cannot drift from the evidence above it; where the prose
makes a qualitative claim ("on every endpoint"), the claim is checked here and
the build fails if the data stop supporting it.

This page is deliberately not part of the site index: the study has no
study.json, so site/build_site.py neither lists it nor copies it, and it does not
enter the cross-study table. It is published on its own, by this script writing
a second copy to docs/reports/molpair.html, which GitHub Pages serves.

    for ds in expansion biogen; do
      ADME_DATASET=$ds python 03_collect_metrics.py
      ADME_DATASET=$ds python 04_report.py
    done
    python 05_summary.py
    python 06_build_page.py
"""

import pandas as pd

import config as cfg
import page_kit

PAGE = cfg.PROJECT_DIR / "results" / "report.html"
PUBLISHED = cfg.PROJECT_DIR.parent.parent / "docs" / "reports" / "molpair.html"
MAIN, OTHER = "expansion", "biogen"
METHODS = cfg.METHODS
MP, GLOBAL = cfg.MOLPAIR_METHOD, cfg.GLOBAL_METHOD
MONROE, CHEMELEON, LGBM = cfg.MONROE_METHOD, cfg.CHEMELEON_METHOD, cfg.LGBM_METHOD
COMPARATORS = [GLOBAL, MONROE, CHEMELEON, LGBM]
LABELS = {"ExpansionRx": MAIN, "Biogen ADME": OTHER}

COLOR = {m: page_kit.METHOD_COLOR[m] for m in METHODS}
SHORT = {LGBM: "LightGBM", CHEMELEON: "CheMeleon FT", MONROE: "Monroe/PFN3",
         GLOBAL: "CheMeleon/PFN3", MP: "MolPAIR"}
LABEL = {
    **cfg.METHOD_LABELS,
    CHEMELEON: "ChemProp + CheMeleon, fine-tuned",
    GLOBAL: "CheMeleon + TabPFN 3 (MolPAIR's global stage)",
}
METRIC_NAME = {"r2": "R²", "spearman": "Spearman ρ", "mae": "MAE"}


# --- loading -------------------------------------------------------------
def figure(name: str, caption: str, ds: str | None) -> str:
    base = cfg.paths(ds).figures if ds else cfg.PROJECT_DIR / "results" / "figures"
    return page_kit.figure_block(page_kit.embed_figure(base / f"{name}.png"), caption, name)


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
    """MolPAIR's wins / no call / losses against each arm, per data set."""
    rows = []
    for other in COMPARATORS:
        cells = [f'<th scope="row"><span class="swatch" style="background:{COLOR[other]}"></span>'
                 f"{LABEL[other]}</th>"]
        for ds in (MAIN, OTHER):
            g = calls[(calls["other"] == other) & (calls["dataset"] == cfg.DATASETS[ds].label)]
            n = g["call"].value_counts()
            cells.append(f'<td class="num"><span class="chip win">{n.get("win", 0)}</span></td>'
                         f'<td class="num">{n.get("no call", 0)}</td>'
                         f'<td class="num"><span class="chip loss">{n.get("loss", 0)}</span></td>')
        r2 = calls[(calls["other"] == other) & (calls["metric"] == "r2")]["gain"].mean()
        mae = calls[(calls["other"] == other) & (calls["metric"] == "mae")]["gain"].mean()
        cells.append(f'<td class="num">{num(r2, 3, True)}</td><td class="num">{num(mae, 4, True)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    n_main = 3 * len(cfg.DATASETS[MAIN].targets)
    n_other = 3 * len(cfg.DATASETS[OTHER].targets)
    return (
        '<div class="bleed tablewrap"><table><thead>'
        '<tr><th scope="col" rowspan="2">MolPAIR against</th>'
        f'<th colspan="3">ExpansionRx, {n_main}</th><th colspan="3">Biogen, {n_other}</th>'
        '<th colspan="2">MolPAIR gain, mean of 15</th></tr>'
        "<tr><th>wins</th><th>no call</th><th>losses</th>"
        "<th>wins</th><th>no call</th><th>losses</th>"
        "<th>Δ R²</th><th>MAE drop</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>")


def levers_table(lv: pd.DataFrame) -> str:
    rows = []
    for _, r in lv.iterrows():
        rows.append(
            f'<tr><th scope="row">{r["endpoint"]}</th><td>{r["dataset"]}</td>'
            f'<td class="num">{num(r["pair_stage_r2"], 3, True)}</td>'
            f'<td class="num">{num(r["finetune_r2"], 3, True)}</td>'
            f'<td class="num">{num(r["representation_r2"], 3, True)}</td></tr>')
    med = lv[["pair_stage_r2", "finetune_r2", "representation_r2"]].median()
    rows.append(
        '<tr><th scope="row"><b>median</b></th><td></td>'
        + "".join(f'<td class="num"><b>{num(med[c], 3, True)}</b></td>'
                  for c in ("pair_stage_r2", "finetune_r2", "representation_r2")) + "</tr>")
    return ('<div class="bleed tablewrap"><table><thead><tr>'
            '<th scope="col">Endpoint</th><th style="text-align:left">Data set</th>'
            "<th>Add MolPAIR's pair correction</th>"
            "<th>Fine-tune CheMeleon instead</th>"
            "<th>Read Monroe's embedding instead</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def regime_table(reg: pd.DataFrame, act: pd.DataFrame) -> str:
    rows = []
    for ds_label in ("ExpansionRx", "Biogen ADME"):
        for pop, name in (("paper", "Nearest training molecule ≤ 0.60"),
                          ("close", "Nearest training molecule &gt; 0.60")):
            g = reg[(reg["dataset"] == ds_label) & (reg["population"] == pop)]
            if g.empty:
                continue
            share = act.loc[ds_label, "paper_population_pct"]
            share = share if pop == "paper" else 100 - share
            n = g["call"].value_counts()
            rows.append(
                f'<tr><th scope="row">{ds_label}</th><td style="text-align:left">{name}</td>'
                f'<td class="num">{share:.0f}%</td><td class="num">{len(g)}</td>'
                f'<td class="num">{num(g["gain"].mean(), 4, True)}</td>'
                f'<td class="num"><span class="chip win">{n.get("win", 0)}</span> / '
                f'{n.get("no call", 0)} / <span class="chip loss">{n.get("loss", 0)}</span></td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Data set</th>'
            '<th style="text-align:left">Test molecules</th><th>Share of test set</th>'
            "<th>Endpoints scored</th><th>Mean MAE drop</th><th>Wins / no call / losses</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def cost_table(cost: pd.DataFrame) -> str:
    rows = []
    for ds_label, r in cost.iterrows():
        rows.append(
            f'<tr><th scope="row">{ds_label}</th><td class="num">{int(r["folds"])}</td>'
            f'<td class="num">{r["median_s"]:.0f}</td><td class="num">{r["max_s"]:.0f}</td>'
            f'<td class="num">{int(r["median_query_pairs"]):,}</td>'
            f'<td class="num">{r["total_h"]:.1f}</td></tr>')
    return ('<div class="bleed tablewrap"><table><thead><tr><th scope="col">Data set</th>'
            "<th>Folds</th><th>Median s / fold</th><th>Slowest fold, s</th>"
            "<th>Median query–anchor pairs / fold</th><th>Total, GPU hours</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


# --- references ----------------------------------------------------------
REFERENCES = [
    ("molpair", "Lee, J.; Lee, D.; Jeong, M.; Yoo, J. ",
     "Molecular Property Prediction under Structural Shift with Tabular Foundation Models.",
     "Preprint, 2026. The method this page reimplements; its code link, "
     "github.com/nums-ai/MolPAIR, did not resolve at the time of writing.",
     "https://arxiv.org/abs/2609.38744", "arXiv:2609.38744"),
    ("chemeleon", "Burns, J. W.; Zalte, A. S.; Abreu, C. R. A.; <i>et al.</i> ",
     "Deep Learning Foundation Models for Low-Data Regimes from Classical Molecular Descriptors.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2026, articles ASAP. Both MolPAIR\'s '
     "frozen global representation and, fine-tuned, one of the arms it is compared with.",
     "https://doi.org/10.1021/acs.jcim.6c01546", "10.1021/acs.jcim.6c01546"),
    ("monroe", "Banaszewski, B.; Fitzgibbon, A. W. ",
     "Monroe: A Molecular Foundation Model for In-Context Probabilistic Inference.",
     "Preprint, 2026. The frozen encoder in the strongest arm of the source comparison.",
     "https://arxiv.org/abs/2608.18982", "arXiv:2608.18982"),
    ("tabpfn3", "Grinsztajn, L.; Flöge, K.; Key, O.; <i>et al.</i> ",
     "TabPFN-3: Technical Report.",
     "Preprint, 2026. The backbone of both MolPAIR models and of the Monroe arm.",
     "https://arxiv.org/abs/2605.13986", "arXiv:2605.13986"),
    ("chemprop", "Graff, D. E.; Morgan, N. K.; Burns, J. W.; <i>et al.</i> ",
     "Chemprop v2: An Efficient, Modular Machine Learning Package for Chemical Property "
     "Prediction.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2026, 66 (1), 28–33.',
     "https://doi.org/10.1021/acs.jcim.5c02332", "10.1021/acs.jcim.5c02332"),
    ("morgan", "Rogers, D.; Hahn, M. ", "Extended-Connectivity Fingerprints.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2010, 50 (5), 742–754. The LightGBM '
     "baseline's representation, and MolPAIR's anchor similarity.",
     "https://doi.org/10.1021/ci100050t", "10.1021/ci100050t"),
    ("mmp", "Hussain, J.; Rea, C. ",
     "Computationally Efficient Algorithm to Identify Matched Molecular Pairs (MMPs) in Large "
     "Data Sets.",
     '<span class="venue">J. Chem. Inf. Model.</span> 2010, 50 (3), 339–348. The single-cut '
     "fragmentation MolPAIR ranks its anchors by.",
     "https://doi.org/10.1021/ci900450m", "10.1021/ci900450m"),
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


def where(pairs) -> str:
    """'ExpansionRx LOG_MPPB and Biogen LOG_RPPB', from (data set, endpoint) pairs."""
    items = [f"{'Biogen' if d == 'Biogen ADME' else d} {e}" for d, e in sorted(pairs)]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build() -> str:
    s_main, s_other = summary(MAIN), summary(OTHER)
    calls = results("molpair_vs_arms.csv")
    lv = results("levers.csv")
    reg = results("regime.csv")
    act = results("activity.csv").set_index("dataset")
    sim = results("by_similarity.csv")
    cost = results("cost.csv").set_index("dataset")

    n_calls = len(calls[calls["other"] == GLOBAL])
    counts = calls.groupby(["other", "call"]).size().unstack(fill_value=0).reindex(
        columns=["win", "no call", "loss"], fill_value=0)
    w = {o: int(counts.loc[o, "win"]) for o in COMPARATORS}
    l = {o: int(counts.loc[o, "loss"]) for o in COMPARATORS}
    n_ep = len(lv)

    # The pair stage against the prediction it corrects.
    vs_global = calls[calls["other"] == GLOBAL]
    pair_r2 = lv["pair_stage_r2"]
    pair_mae = lv["pair_stage_mae"]
    pair_folds = vs_global[vs_global["metric"] == "mae"]["folds_won"]
    n_pair_up = int((pair_mae > 0).sum())

    # Against Monroe.
    vs_monroe = calls[calls["other"] == MONROE]
    monroe_gap_mae = -vs_monroe[vs_monroe["metric"] == "mae"]["gain"]  # positive = Monroe better
    n_monroe_better = int((vs_monroe[vs_monroe["metric"] == "mae"]["gain"] < 0).sum())

    monroe_ahead = vs_monroe[(vs_monroe["metric"] == "mae") & (vs_monroe["gain"] > 0)]
    lv_key = lv.set_index(["dataset", "endpoint"])
    rep_worse = set(lv_key.index[lv_key["representation_mae"] < 0])
    monroe_ahead_set = set(zip(monroe_ahead["dataset"], monroe_ahead["endpoint"]))

    # Against fine-tuned CheMeleon, by data set, on MAE.
    vs_cm = calls[(calls["other"] == CHEMELEON) & (calls["metric"] == "mae")]
    cm = {ds: vs_cm[vs_cm["dataset"] == ds]["call"].value_counts() for ds in LABELS}
    cm_wins = set(zip(*vs_cm[vs_cm["call"] == "win"][["dataset", "endpoint"]].T.values))
    ft_worse = set(lv_key.index[lv_key["finetune_mae"] < 0])

    # Where the correction hurts.
    pair_losses = vs_global[vs_global["call"] == "loss"]
    hurt = sorted(set(zip(pair_losses["dataset"], pair_losses["endpoint"])))

    med = lv[["pair_stage_r2", "finetune_r2", "representation_r2"]].median()
    ratio = med["representation_r2"] / med["pair_stage_r2"]
    n_rep_beats_pair = int((lv["representation_r2"] > lv["pair_stage_r2"]).sum())
    positive = {c: int((lv[c] > 0).sum()) for c in ("pair_stage_r2", "finetune_r2", "representation_r2")}
    largest = lv[["pair_stage_r2", "finetune_r2", "representation_r2"]].idxmax(axis=1).value_counts()
    n_pair_largest = int(largest.get("pair_stage_r2", 0))

    tally_m, tally_o = tally_counts(s_main), tally_counts(s_other)
    top = {m: int(tally_m.loc[m, ["alone", "tied"]].sum() + tally_o.loc[m, ["alone", "tied"]].sum())
           for m in METHODS}
    n_combos = 3 * n_ep
    n_models = int(cost["folds"].sum())

    # The regime.
    exp_act, bio_act = act.loc["ExpansionRx"], act.loc["Biogen ADME"]
    reg_paper = reg[reg["population"] == "paper"]
    reg_close = reg[reg["population"] == "close"]
    # Bins below the 0.20 acceptance threshold have no anchors and are never
    # corrected, so they say nothing about how often a correction helps.
    corrected_bins = sim[(sim["n"] >= 50) & (sim["mean_gate"] > 0)]
    improved_lo = corrected_bins["improved_pct"].min()
    improved_hi = corrected_bins["improved_pct"].max()
    paper_w = int((reg_paper["call"] == "win").sum())
    paper_l = int((reg_paper["call"] == "loss").sum())
    close_w = int((reg_close["call"] == "win").sum())
    close_l = int((reg_close["call"] == "loss").sum())
    reg_exp = reg[reg["dataset"] == "ExpansionRx"].pivot(index="endpoint", columns="population",
                                                         values="gain")
    paper_losses = reg_paper[reg_paper["call"] == "loss"]["endpoint"].tolist()
    # ExpansionRx's mean MAE drop over its whole test set, against the same over
    # only the molecules the paper's protocol would keep.
    exp_lv = lv[lv["dataset"] == "ExpansionRx"]
    exp_all_gain = exp_lv["pair_stage_mae"].mean()
    exp_paper_gain = reg_exp["paper"].mean()

    # Qualitative claims the prose makes; fail the build if the data stop
    # supporting them.
    assert w[GLOBAL] > l[GLOBAL], "the prose says the pair stage helps"
    assert w[LGBM] > l[LGBM], "the prose says MolPAIR beats LightGBM"
    assert l[MONROE] > w[MONROE], "the prose says Monroe beats MolPAIR"
    assert med["representation_r2"] > med["pair_stage_r2"], "the prose says the representation is the larger lever"
    assert reg_close["gain"].mean() > reg_paper["gain"].mean() > 0, \
        "the prose says the gain is larger on close analogues but positive on the paper's population"
    assert paper_w > paper_l, "the prose says the correction holds on the paper's population"

    pair_claim = (f"Across the three metrics the corrected prediction is significantly ahead "
                  f"on {w[GLOBAL]} of {n_calls} endpoint × metric combinations and behind on "
                  f"{l[GLOBAL]}, and it has the lower mean MAE on {n_pair_up} of {n_ep} "
                  "endpoints.")
    if hurt:
        pair_claim += (f" The exception is {where(hurt)}, where the correction makes things "
                       "worse: " + " and ".join(
                           f"{METRIC_NAME[r['metric']]} "
                           f"{'falls' if r['metric'] != 'mae' else 'rises'} by "
                           f"{num(abs(r['gain']), 3)}"
                           for _, r in pair_losses.iterrows()) + ".")

    parts = [
        # GitHub Pages sends utf-8 in its header; this keeps a copy opened from disk,
        # or from any server that does not, from rendering as mojibake.
        '<meta charset="utf-8">',
        "<title>Borrowing Errors</title>",
        f"<style>{page_kit.CSS}</style>",
        '<div class="wrap">',
        '<p class="eyebrow">Pairwise correction · 5×5 cross validation · two data sets</p>',
        "<h1>Borrowing errors from the neighbours</h1>",
        '<p class="lede">MolPAIR' + ref("molpair") + " corrects a tabular foundation model "
        "with a second one. A TabPFN 3" + ref("tabpfn3") + " reads a frozen CheMeleon"
        + ref("chemeleon") + " fingerprint and predicts each endpoint in context; a second "
        "TabPFN 3 learns from pairs of training molecules how the first one's error moves "
        "from a molecule to its neighbour, and each test molecule borrows the errors of its "
        "closest training molecules to correct its own prediction. Run here on the same "
        "fifteen ADME endpoints, folds and test sets as everything else on this site, the "
        "paper's own claim replicates: the correction beats the prediction it corrects "
        f"on {w[GLOBAL]} of {n_calls} endpoint × metric combinations and loses "
        f"{WORDS.get(l[GLOBAL], l[GLOBAL])}. It still loses to Monroe + TabPFN 3"
        + ref("monroe") + f" on {l[MONROE]} of {n_calls}, because the correction is worth a "
        f"median {num(med['pair_stage_r2'], 3, True)} R² and choosing the representation is "
        f"worth {num(med['representation_r2'], 3, True)}.</p>",
        '<div class="facts">'
        f'<div class="fact"><b>{len(METHODS)}</b><span>methods compared</span></div>'
        '<div class="fact"><b>15</b><span>endpoints</span></div>'
        f'<div class="fact"><b>{cfg.N_REPEATS}×{cfg.N_SPLITS}</b><span>cross validation</span></div>'
        f'<div class="fact"><b>{n_models:,}</b><span>MolPAIR fold models</span></div>'
        f'<div class="fact"><b>{cost["total_h"].sum():.1f}</b><span>GPU hours</span></div>'
        "</div>",

        '<div class="panel howto"><h3>A reimplementation</h3>'
        "<p>The paper points to its code at github.com/nums-ai/MolPAIR. That repository does "
        "not exist — the organisation has four other repositories and nothing by that name — "
        "and nothing is on PyPI. So MolPAIR is written here from the paper's Section 3 and "
        "Appendix A, with every setting the paper states used as stated: CheMeleon's 2,048-d "
        "fingerprint cut to 2,000 features by variance, RDKit 2D descriptor differences for "
        "the pairs, TabPFN 3 with eight estimators for both models, five inner folds for the "
        "out-of-fold errors, eight anchors per training molecule and a relation context of "
        "8,192 pairs, 32 anchors per query at Tanimoto ≥ 0.20, matched pairs ranked first"
        + ref("mmp") + ", weights s² / u and the gate at τ = 0.20. Where the paper is silent — "
        "the matched-pair rules, the fingerprint length, how the inner folds are drawn — the "
        "choice is recorded in the study's README. This tests the method as described, not "
        "the authors' code, and it does not reproduce the paper's numbers, which are on "
        "different benchmarks.</p></div>",

        '<div class="panel howto"><h3>What differs between the arms, and what does not</h3>'
        "<p>Every arm is handed the same training molecules in every fold — the four fifths "
        "of the training set outside the held-out fifth that have a value for the endpoint — "
        "and is scored on the same untouched test set. MolPAIR draws its own inner folds "
        "inside those four fifths and fits nothing on a test label.</p>"
        "<p>One MolPAIR call returns two predictions: the global prediction it starts from, "
        "and the corrected one. Both are arms here. The pair is the paper's own central "
        "comparison and isolates exactly what the correction adds. The other three arms are "
        "the ones it would have to beat to be worth using: LightGBM on Morgan counts"
        + ref("morgan") + ", the floor; ChemProp" + ref("chemprop") + " fine-tuned from "
        "CheMeleon, the same pre-trained encoder trained rather than frozen; and Monroe + "
        "TabPFN 3, the same head on a different frozen representation and the strongest arm "
        "in " + LINK_SOURCE + ". Those three are imported from that comparison unchanged, "
        "not re-run.</p></div>",

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
        f"<p>Monroe + TabPFN 3 is on top of {top[MONROE]} of {n_combos} endpoint × metric "
        f"combinations across the two data sets. MolPAIR reaches the top of {top[MP]}, "
        f"fine-tuned CheMeleon {top[CHEMELEON]}, the global stage {top[GLOBAL]} and "
        f"LightGBM {top[LGBM]}.</p>",

        "<h2>The paper's claim: the correction helps</h2>",
        "<p>The question the paper asks is whether its pair correction improves the "
        "prediction it starts from. The folds are the pairing, which is legitimate because "
        "both predictions come from the same call on the same training molecules. "
        f"{pair_claim} The gain is small and very regular: a median of "
        f"{num(pair_r2.median(), 3, True)} R² and {num(pair_mae.median(), 4)} MAE per "
        f"endpoint, and the corrected prediction has the lower MAE on a median of "
        f"{int(pair_folds.median())} of 25 folds.</p>",
        figure("paired_molpair_global_vs_molpair_mae",
               "MolPAIR against its own global stage on ExpansionRx, MAE paired by fold. Each "
               "line is one fold; green means the correction lowered the error.", MAIN),
        figure("paired_molpair_global_vs_molpair_mae", "The same on Biogen ADME.", OTHER),
        f"<p>Per molecule, the correction is closer to a coin that lands the right way "
        f"slightly more often. Binned by how similar the nearest training molecule is, it "
        f"improves between {improved_lo:.0f}% and {improved_hi:.0f}% of individual test "
        "predictions and worsens the rest; the paper reports 55–57% for regression on its own "
        "splits. Nearly every prediction is touched: "
        f"{num(exp_act['active_pct'], 1)}% on ExpansionRx and "
        f"{num(bio_act['active_pct'], 1)}% on Biogen get a non-zero correction.</p>",

        "<h2>Against the methods it would have to beat</h2>",
        "<p>The paper's comparison stops at its own baseline and at tuned tree models and "
        "graph networks on its own benchmarks. Here MolPAIR is paired against the arms that "
        "set the bar on these endpoints. A win or a loss is a paired t-test at p &lt; 0.05 "
        "on one endpoint and one metric, uncorrected, so read the counts as a direction "
        "rather than a verdict on any single cell.</p>",
        calls_table(calls),
        figure("delta_molpair_mae", "MolPAIR against each arm on ExpansionRx: mean paired MAE "
               "difference over 25 folds with a 95% interval, drawn in the comparator's "
               "colour and signed so that right of zero is MolPAIR better.", MAIN),
        figure("delta_molpair_mae", "The same on Biogen ADME.", OTHER),
        f"<p>MolPAIR beats LightGBM on Morgan counts on {w[LGBM]} of {n_calls} and loses "
        f"{l[LGBM]}. Against fine-tuned CheMeleon on MAE it wins "
        f"{cm['ExpansionRx'].get('win', 0)} and loses {cm['ExpansionRx'].get('loss', 0)} of "
        f"the {len(cfg.DATASETS[MAIN].targets)} ExpansionRx endpoints, and wins "
        f"{cm['Biogen ADME'].get('win', 0)} and loses {cm['Biogen ADME'].get('loss', 0)} of "
        f"the {len(cfg.DATASETS[OTHER].targets)} Biogen ones. "
        + (f"Those wins are on exactly the endpoints where fine-tuning CheMeleon did worse "
           f"than leaving it frozen under TabPFN 3 — {where(cm_wins)} — so they say more about "
           "fine-tuning than about the correction. "
           if cm_wins == ft_worse else
           f"The wins are on {where(cm_wins)}. ")
        + f"Against Monroe + TabPFN 3 it is behind on MAE on {n_monroe_better} of {n_ep} "
        f"endpoints, by a median of {num(monroe_gap_mae.median(), 3)}, which is "
        f"{monroe_gap_mae.median() / pair_mae.median():.0f} times what the correction "
        "itself buys. "
        + (f"The endpoints where it is ahead, {where(monroe_ahead_set)}, are the ones where "
           "Monroe's embedding is itself worse than CheMeleon's fingerprint under the same "
           "TabPFN 3: the representation, not the correction, decides them.</p>"
           if monroe_ahead_set and monroe_ahead_set == rep_worse else
           (f"It is ahead on {where(monroe_ahead_set)}.</p>" if monroe_ahead_set else "</p>")),

        "<h2>The correction against the representation</h2>",
        "<p>All three of the arms below start from the same place, CheMeleon + TabPFN 3, and "
        "each changes one thing about it. MolPAIR keeps the frozen fingerprint and adds the "
        "pair correction. Fine-tuned CheMeleon trains the encoder instead of freezing it, "
        "and swaps TabPFN for a feed-forward head. Monroe keeps TabPFN 3 and replaces the "
        "fingerprint with a different frozen embedding. Each column is the change in mean "
        "R² over 25 folds.</p>",
        levers_table(lv),
        f"<p>The correction is worth a median {num(med['pair_stage_r2'], 3, True)} R², "
        f"fine-tuning the encoder {num(med['finetune_r2'], 3, True)}, and the representation "
        f"{num(med['representation_r2'], 3, True)} — about {ratio:.0f} times the correction, "
        f"and larger than it on {n_rep_beats_pair} of {n_ep} endpoints. What the correction "
        "has is consistency: it raises R² on "
        f"{positive['pair_stage_r2']} of {n_ep} endpoints, against "
        f"{positive['finetune_r2']} for fine-tuning and {positive['representation_r2']} for "
        f"the representation. It is the largest of the three on "
        f"{WORDS.get(n_pair_largest, n_pair_largest)} of them.</p>",

        "<h2>Is this the paper's regime?</h2>",
        "<p>MolPAIR is pitched at structural shift. The paper's primary split holds out "
        "whole scaffold groups and then drops every test molecule more similar than Tanimoto "
        "0.60 to any training molecule. The splits here hold out whole BitBIRCH clusters "
        "with no similarity cutoff, so they keep close analogues the paper's would remove — "
        "which is where borrowing a neighbour's error should work best, not worst. If the "
        "result above were an artefact of easier splits, the gain would be concentrated in "
        "those close molecules.</p>",
        f"<p>The two data sets differ here. The median test molecule's nearest training "
        f"molecule is at Tanimoto {num(bio_act['median_max_sim'], 2)} on Biogen, and "
        f"{bio_act['paper_population_pct']:.0f}% of its test molecules would have survived "
        "the paper's cutoff: Biogen is close to the paper's regime as it stands. On "
        f"ExpansionRx the median is {num(exp_act['median_max_sim'], 2)} and only "
        f"{exp_act['paper_population_pct']:.0f}% would survive. Scoring the correction "
        "separately on the two populations, per endpoint and per fold:</p>",
        regime_table(reg, act),
        figure("correction_by_similarity", "How often the correction lowers a test "
               "molecule's absolute error, and by how much on average, against the "
               "similarity of its nearest training molecule. The dashed line is the paper's "
               "cutoff: its primary split keeps only molecules to the left.", None),
        f"<p>The correction does more for close analogues, as it should: a mean MAE drop of "
        f"{num(reg_close['gain'].mean(), 4)} on test molecules with a training neighbour "
        f"above 0.60, winning {close_w} of {len(reg_close)} endpoints and losing {close_l}, "
        f"against {num(reg_paper['gain'].mean(), 4)} on the molecules the paper's protocol "
        f"would keep. On ExpansionRx the drop among close analogues is larger on "
        f"{int((reg_exp['close'] > reg_exp['paper']).sum())} of {len(reg_exp)} endpoints. But "
        f"on the paper's population it still wins {paper_w} of {len(reg_paper)} endpoints "
        f"and loses {paper_l}"
        + (f" ({', '.join(paper_losses)})" if paper_losses else "")
        + ". So the easier part of these splits flatters the size of the gain — on "
        f"ExpansionRx the mean MAE drop is {num(exp_all_gain, 4)} over the whole test set and "
        f"{num(exp_paper_gain, 4)} over the part the paper would have kept — but the "
        "replication does not depend on it.</p>",

        "<h2>What it costs</h2>",
        "<p>Every fold fits seven TabPFN 3 models — one global, five out-of-fold, one "
        "relation — and predicts every query–anchor pair in both directions. On one RTX "
        "5070 Ti:</p>",
        cost_table(cost),
        "<p>Most of that is the relation model reading tens of thousands of pair rows "
        "against an 8,192-row context. By comparison the source comparison reports the 225 "
        "ExpansionRx folds of Monroe + TabPFN 3 taking 14 minutes on the same card.</p>",

        "<h2>What this does and does not say</h2>",
        '<div class="panel howto">'
        "<h3>Scope</h3>"
        "<p><b>One configuration.</b> MolPAIR is run in the paper's primary configuration, "
        "CheMeleon with TabPFN 3. The paper also runs it over Monroe's embedding, where it "
        "reports a smaller gain than over CheMeleon. That combination — the correction on "
        "top of the representation that wins here — is the obvious next arm and is not "
        "run.</p>"
        "<p><b>A reimplementation.</b> The authors' code is not available, so differences "
        "between this and their implementation in the details the paper does not specify "
        "cannot be ruled out. The direction of the gain over the global stage, its "
        "regularity, and the share of individual predictions it improves all match what "
        "the paper reports, which is some evidence the method is the same one.</p>"
        "<p><b>Not the paper's splits or endpoints.</b> Fifteen ADME endpoints, not "
        "MoleculeACE and Polaris, and cluster splits, not scaffold-plus-similarity. The "
        "section above scores the correction on the paper's population inside these splits, "
        "which narrows the gap but does not close it.</p>"
        "</div>",

        f'<p class="footnote">Reported the way {LINK_BLOG} and the protocol paper it '
        "points at" + ref("tukey") + " argue comparisons should be: distributions over "
        "folds, simultaneous confidence intervals, paired tests on the folds, and no "
        "bolded maxima. Where two methods cannot be separated, both are reported as "
        "tied.</p>",
        '<p class="repo">Code, per-fold metrics and every figure: '
        '<a href="https://github.com/patwalters/model-validation-central/tree/main/studies/molpair">'
        "studies/molpair</a>.</p>",
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
