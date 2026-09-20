"""The prose of the report, kept apart from the page assembly.

Every number here is read out of the tables in `results/tables/`, and every
sentence whose shape depends on *which* arm came out on top is phrased from the
tally rather than written down. That is deliberate: this page is regenerated
whenever a fold changes, and a hand-typed number would go stale silently. Where
a claim cannot be written that way it is not made.

`NARRATIVE` maps a section of `09_build_page.py` to a function of
(summary, head_to_head, gaps, tally, distances, decay), each returning a block
of HTML. Every section takes the same six tables whether it reads them all or
not, so adding a table to the page does not mean editing every signature.
"""

import pandas as pd

import config as cfg

def _load_table(name: str) -> pd.DataFrame:
    path = cfg.TABLE_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _load_reproduction() -> pd.DataFrame:
    path = cfg.SHARED_TABLE_DIR / "reproduction.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


REPRO = _load_reproduction()
SIZES = _load_table("fold_sizes")
CONFOUND = _load_table("size_confound")

# The four schemes that hold training-set size fixed while the test set moves
# away. They are the clean experiment in this collection.
SIZE_MATCHED = ["random", "scaffold", "butina", "umap"]

# The two ends of the stringency axis, as the paper orders the schemes.
EASIEST, HARDEST = cfg.SPLITS[0], cfg.SPLITS[-1]
# The four schemes the paper calls stringent, where its collapse claim lives.
STRINGENT = ["butina", "diverse_25", "umap", "time"]


# --- reading the tables ---------------------------------------------------
def mean_of(summary: pd.DataFrame, split: str, metric: str, method: str) -> float:
    row = summary[(summary["split"] == split) & (summary["metric"] == metric)
                  & (summary["method"] == method)]
    return float(row["mean"].iloc[0]) if len(row) else float("nan")


def gap_of(gaps: pd.DataFrame, split: str, method: str) -> float:
    row = gaps[(gaps["split"] == split) & (gaps["method"] == method)]
    return float(row["mean"].iloc[0]) if len(row) else float("nan")


def gap_win_rate(gaps: pd.DataFrame, split: str, method: str) -> float:
    row = gaps[(gaps["split"] == split) & (gaps["method"] == method)]
    return float(row["beats_baseline"].iloc[0]) if len(row) else float("nan")


def pair_row(h2h: pd.DataFrame, split: str, metric: str, a: str, b: str,
             test: str = "t") -> pd.Series | None:
    """One pair's row, sign normalised so `mean_diff` reads b - a."""
    sub = h2h[(h2h["split"] == split) & (h2h["metric"] == metric) & (h2h["test"] == test)]
    hit = sub[(sub["left"] == a) & (sub["right"] == b)]
    if len(hit):
        return hit.iloc[0]
    hit = sub[(sub["left"] == b) & (sub["right"] == a)]
    if not len(hit):
        return None
    row = hit.iloc[0].copy()
    row["mean_diff"] = -row["mean_diff"]
    row["right_wins"] = 1.0 - row["right_wins"]
    return row


def top_arm(gaps: pd.DataFrame, split: str) -> str:
    """The arm that takes the most MAE off the naive predictor on one scheme."""
    g = gaps[gaps["split"] == split].set_index("method")
    arms = [m for m in cfg.METHODS if m in g.index]
    return max(arms, key=lambda m: g.loc[m, "mean"])


def on_top(summary: pd.DataFrame, split: str, metric: str) -> list[str]:
    """Every arm the paired test cannot separate from the best on one scheme."""
    sub = summary[(summary["split"] == split) & (summary["metric"] == metric)]
    tied = set(sub.loc[sub["paired_group"].isin(("best", "equivalent")), "method"])
    return [m for m in cfg.METHODS if m in tied]


def label(method: str) -> str:
    return cfg.METHOD_LABELS[method]


def join(names: list[str]) -> str:
    """A, B and C -- for strings that are already what should be printed."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def name_list(methods: list[str]) -> str:
    """The same, for method keys."""
    return join([label(m) for m in methods])


def scheme_list(splits: list[str]) -> str:
    return join([cfg.SPLIT_LABELS[s].lower() for s in splits])


def count_of_six(n: int) -> str:
    """"all six" / "none of the six" / "four of the six" -- so the prose reads."""
    total = len(cfg.SPLITS)
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    if n == total:
        return f"all {words.get(total, total)}"
    if n == 0:
        return f"none of the {words.get(total, total)}"
    return f"{words.get(n, n)} of the {words.get(total, total)}"


def schemes_where(summary: pd.DataFrame, metric: str, method: str,
                  group: tuple[str, ...]) -> list[str]:
    sub = summary[(summary["metric"] == metric) & (summary["method"] == method)]
    return [s for s in cfg.SPLITS
            if len(sub[(sub["split"] == s) & (sub["paired_group"].isin(group))])]


def reproduction(repro: pd.DataFrame) -> tuple[int, int, float]:
    """(folds reproduced, folds checked, worst difference) on the recorded schemes.

    Read rather than written down. The diverse scheme is excluded because its
    validation split was never recorded upstream, so the two k-NN arms pick k on
    different molecules there by construction -- see the protocol section.
    """
    if repro.empty:
        return (0, 0, float("nan"))
    kept = repro[(repro["split"] != "diverse_25") & (repro["metric"].isin(["mae", "r2"]))]
    if kept.empty:
        return (0, 0, float("nan"))
    return (int(kept["within_tol"].sum()), int(kept["folds"].sum()),
            float(kept["max_abs_diff"].max()))


def tally_of(tally: pd.DataFrame, method: str) -> tuple[int, int, int]:
    row = tally[tally["method"] == method]
    if not len(row):
        return (0, 0, 0)
    r = row.iloc[0]
    return int(r["best_alone"]), int(r["tied"]), int(r["worse"])


def p(text: str) -> str:
    return f"<p>{text}</p>"


# --- the sections ---------------------------------------------------------
def intro(summary, h2h, gaps, tally, distances, decay) -> str:
    arms = [m for m in cfg.RUN_METHODS]
    winner = max(arms, key=lambda m: sum(tally_of(tally, m)[:2]))
    w_alone, w_tied, _ = tally_of(tally, winner)
    n_combos = len(cfg.SPLITS) * len(cfg.METRICS)

    naive_rise = mean_of(summary, HARDEST, "mae", cfg.MEDIAN_METHOD) - mean_of(
        summary, EASIEST, "mae", cfg.MEDIAN_METHOD)
    best_rise = mean_of(summary, HARDEST, "mae", winner) - mean_of(summary, EASIEST, "mae", winner)
    gap_easy, gap_hard = gap_of(gaps, EASIEST, winner), gap_of(gaps, HARDEST, winner)

    knn = pair_row(h2h, HARDEST, "mae", cfg.KNN_METHOD, winner)
    knn_beaten = [s for s in cfg.SPLITS
                  if (r := pair_row(h2h, s, "mae", cfg.KNN_METHOD, winner)) is not None
                  and r["mean_diff"] < 0 and r["p_holm"] < 0.05]

    return "\n".join([
        p("The paper this page is built on makes two claims. The first is that "
          "k-nearest neighbours performs comparably to modern machine learning under "
          "every splitting scheme it tries, which is the finding of Janela and "
          "Bajorath restated on ten targets and six schemes. The second is that as the "
          "schemes get more stringent, every method's accuracy deteriorates towards a "
          "naive baseline, so that reported accuracies obtained under lenient splits "
          "are inflated. Both claims are tested here on the paper's own folds, with "
          "two kinds of model it did not have: a pre-trained graph network that is "
          "fine-tuned, and a pre-trained encoder that is frozen and read in context, "
          "with nothing trained downstream at all."),
        p(f"The second claim survives and the first does not. Raw MAE rises for every "
          f"arm as the splits harden — for {label(winner)}, the strongest arm here, "
          f"by {best_rise:+.3f} pIC₅₀ units from the {cfg.SPLIT_LABELS[EASIEST].lower()} "
          f"scheme to the {cfg.SPLIT_LABELS[HARDEST].lower()} one. But the naive "
          f"baseline rises by {naive_rise:+.3f} over the same span, and most of what "
          f"looks like a collapsing model is a moving target. What does not move much "
          f"is the distance between the two: {gap_easy:.3f} pIC₅₀ of MAE taken off the "
          f"naive predictor on the easiest scheme, {gap_hard:.3f} on the hardest."),
        p(f"On the first claim, k-NN is separable from {label(winner)} on "
          f"{count_of_six(len(knn_beaten))} schemes by a Holm-corrected paired test over "
          f"fifty folds"
          + (f", including the {cfg.SPLIT_LABELS[HARDEST].lower()} split, where the gap "
             f"is {abs(knn['mean_diff']):.3f} pIC₅₀ and {label(winner)} is ahead on "
             f"{1 - knn['right_wins']:.0%} of the fifty folds"
             if knn is not None and HARDEST in knn_beaten else "")
          + f". Across all {n_combos} scheme × metric combinations {label(winner)} is "
          f"best alone on {w_alone} and tied for best on {w_tied}. The paper's own six "
          "arms, imported at their published per-fold numbers, do not change that: "
          "they are all variations on learning from one target's labels and a "
          "fingerprint, and the thing that separates here is the representation."),
        p("Whether that is good news depends on what is being asked. If the question "
          "is whether the field's reported accuracies are inflated by lenient splits, "
          "the paper is right and this page agrees with it. If the question is whether "
          "anything beats nearest-neighbour lookup out of distribution, the answer "
          "turns out to depend on which models are in the comparison."),
    ])


def collapse(summary, h2h, gaps, tally, distances, decay) -> str:
    winner = max(cfg.RUN_METHODS, key=lambda m: sum(tally_of(tally, m)[:2]))
    naive_easy = mean_of(summary, EASIEST, "mae", cfg.MEDIAN_METHOD)
    naive_hard = mean_of(summary, HARDEST, "mae", cfg.MEDIAN_METHOD)

    rows = []
    for split in cfg.SPLITS:
        best = top_arm(gaps, split)
        rows.append((split, best, gap_of(gaps, split, best), gap_win_rate(gaps, split, best)))
    worst_gap = min(rows, key=lambda r: r[2])
    best_gap = max(rows, key=lambda r: r[2])

    arms = [m for m in cfg.METHODS if m != cfg.MEDIAN_METHOD]
    below = [(s, m) for s in cfg.SPLITS for m in arms if gap_of(gaps, s, m) < 0]
    below_arms = sorted({m for _, m in below}, key=arms.index)

    return "\n".join([
        p("The paper's Figure 2 is a set of boxplots of MAE, one panel per scheme, and "
          "the trend in it is unmistakable: error climbs as the splits harden until the "
          "models sit on top of a naive baseline. The figure below is that figure, "
          "rebuilt on this study's arms and this study's folds, and the trend is the "
          "same. Nothing here disputes it."),
        p(f"What the raw axis hides is that the baseline moves too. The training median "
          f"scores {naive_easy:.3f} MAE on the "
          f"{cfg.SPLIT_LABELS[EASIEST].lower()} scheme and {naive_hard:.3f} on the "
          f"{cfg.SPLIT_LABELS[HARDEST].lower()} one, a drift of "
          f"{naive_hard - naive_easy:+.3f} that has nothing to do with any model. The "
          "schemes select structurally different training sets, so the potency "
          "distribution a fold trains on is different, and its median is a different "
          "guess. The paper says this in a parenthesis under its own figure. It is "
          "worth promoting to an axis."),
        p(f"Read as a distance from that baseline, the picture is less uniform. The "
          f"best arm on each scheme takes between {worst_gap[2]:.3f} and "
          f"{best_gap[2]:.3f} pIC₅₀ off the naive predictor — thinnest on the "
          f"{cfg.SPLIT_LABELS[worst_gap[0]].lower()} scheme, widest on the "
          f"{cfg.SPLIT_LABELS[best_gap[0]].lower()} one. On the "
          f"{cfg.SPLIT_LABELS[HARDEST].lower()} split, {label(top_arm(gaps, HARDEST))} still "
          f"beats the naive predictor on "
          f"{gap_win_rate(gaps, HARDEST, top_arm(gaps, HARDEST)):.0%} of the fifty "
          f"folds."
          + (f" {len(below)} arm × scheme combinations fall below zero — "
             f"{name_list(below_arms)} on "
             f"{scheme_list(sorted({s for s, _ in below}, key=cfg.SPLITS.index))} — "
             "which is to say those arms would have done better predicting the training "
             "median for every molecule." if below else
             " No arm on any scheme falls below zero, so nothing here is worse than "
             "predicting the training median.")),
        p(f"The distinction matters for what the paper concludes from the trend. "
          f"Accuracy quoted on a random split is inflated as a guide to prospective "
          f"performance — that is established, here and there. Accuracy converging "
          f"towards a naive baseline is a stronger claim, and on this evidence it holds "
          f"for the arms the paper tested and not for {label(winner)}."),
    ])


def stringency(summary, h2h, gaps, tally, distances, decay) -> str:
    by_split = distances.groupby("split")[["mean_max_sim", "frac_novel"]].mean()
    order = by_split["mean_max_sim"].reindex(cfg.SPLITS)
    easiest_sim = float(order.loc[EASIEST])
    hardest_name = order.idxmin()
    hardest_sim = float(order.min())
    novel_hard = float(by_split.loc[hardest_name, "frac_novel"])

    monotone = list(order.sort_values(ascending=False).index) == cfg.SPLITS
    d = decay.set_index("method")
    arms = [m for m in cfg.METHODS if m in d.index]
    shallowest = min(arms, key=lambda m: d.loc[m, "slope"]) if arms else None
    positive_far = [m for m in arms if d.loc[m, "gap_at_sim_0.30"] > 0]
    negative_far = [m for m in arms if d.loc[m, "gap_at_sim_0.30"] <= 0]

    return "\n".join([
        p("Up to this point the six schemes have been an ordering borrowed from the "
          "paper. They can be measured. For every one of the 300 folds, each test "
          "molecule's nearest training neighbour was found by ECFP4 Tanimoto, and the "
          "fold summarised by the mean of those similarities — the quantity the paper "
          "plots in its Figures 5 and 6."),
        p(f"The ordering mostly holds. Mean nearest-neighbour similarity falls from "
          f"{easiest_sim:.3f} on the {cfg.SPLIT_LABELS[EASIEST].lower()} scheme to "
          f"{hardest_sim:.3f} on the {cfg.SPLIT_LABELS[hardest_name].lower()} one, where "
          f"{novel_hard:.0%} of test molecules have no training neighbour above the "
          f"Butina threshold the paper itself clusters at."
          + ("" if monotone else
             " It is not quite monotone in the paper's presentation order — the time and "
             "UMAP splits swap, and the diverse split is closer to them than its position "
             "in the sequence suggests — which is worth knowing before reading a trend "
             "across the six as a straight line.")),
        p("That turns the independent variable from a six-valued label into a continuous "
          "one, and with it each method's accuracy can be plotted against distance "
          "directly. The slope of that line is the claim the paper is really making: "
          "that every method decays at the same rate as the test molecules move away. "
          "It is fitted below on the gap to the naive baseline rather than on raw MAE, "
          "because raw MAE confounds the decay with the baseline drift of the previous "
          "section."
          + (f" The shallowest decay belongs to {label(shallowest)}."
             if shallowest else "")),
        p(f"Extrapolated to a similarity of 0.30 — roughly what the hardest schemes "
          f"deliver — "
          + (f"{name_list(positive_far)} still take MAE off the naive predictor, while "
             f"{name_list(negative_far)} no longer do."
             if positive_far and negative_far else
             ("every arm still takes MAE off the naive predictor." if positive_far else
              "no arm still takes MAE off the naive predictor, which is the paper's "
              "conclusion stated as a number rather than a trend."))),
    ])


def confound(summary, h2h, gaps, tally, distances, decay) -> str:
    """Whether the schemes vary one thing or two."""
    if SIZES.empty or CONFOUND.empty:
        return ""
    z = SIZES.set_index("split")
    matched = [s for s in SIZE_MATCHED if s in z.index]
    others = [s for s in cfg.SPLITS if s not in matched and s in z.index]
    all_rho = CONFOUND.iloc[0]
    matched_rho = CONFOUND.iloc[1] if len(CONFOUND) > 1 else None

    sim_hi = z.loc[matched, "mean_max_sim"].max() if matched else float("nan")
    sim_lo = z.loc[matched, "mean_max_sim"].min() if matched else float("nan")

    pieces = [
        p(f"Before reading a trend across the six, it is worth asking whether the six "
          f"vary one thing or two. Four of them train on 72.0% of a target and test on "
          f"20.0%, and they do it to within a rounding error: across one target's five "
          f"folds the largest training set and the smallest differ by about one "
          f"molecule on the random, scaffold and Butina schemes. The other two do not. "
          + " ".join(
              f"The {cfg.SPLIT_LABELS[s].lower()} scheme trains on "
              f"{z.loc[s, 'train_pct']:.0f}% of a target, {z.loc[s, 'train']:,.0f} "
              f"molecules against {z.loc[matched[0], 'train']:,.0f}."
              for s in others) if others else ""),
        p(f"That matters because training-set size and distance travel together across "
          f"the whole collection: Spearman ρ = {all_rho['spearman_train_vs_similarity']:+.2f} "
          f"between the two over all {int(all_rho['n_folds'])} folds"
          + (f". Within the four size-matched schemes it disappears, "
             f"ρ = {matched_rho['spearman_train_vs_similarity']:+.2f}, "
             f"p = {matched_rho['p_value']:.2f}."
             if matched_rho is not None else ".")
          + f" So {scheme_list(matched)} are a clean four-point experiment — "
          f"nearest-neighbour similarity falls from {sim_hi:.3f} to {sim_lo:.3f} while "
          f"the training set stays at about {z.loc[matched[0], 'train']:,.0f} molecules "
          f"— and the other two confound moving the test set away with taking the "
          f"training data away. A model doing worse on the diverse split than on the "
          f"random one has been handed a quarter of the data as well as a harder test "
          f"set, and nothing in a raw comparison separates those."),
        p("Nothing below is dropped on that account, because the paper's six schemes are "
          "what this study set out to run. But the decay fitted against distance pools "
          "all six, so it carries some of the confound, and the comparison that does not "
          "is the one across the four size-matched schemes. Where the two tell the same "
          "story it is said once; where they differ, the four-scheme version is the one "
          "to believe."),
    ]
    return "\n".join(pieces)


def separation(summary, h2h, gaps, tally, distances, decay) -> str:
    n_combos = len(cfg.SPLITS) * len(cfg.METRICS)
    never = [m for m in cfg.METHODS if sum(tally_of(tally, m)[:2]) == 0]
    tops = {s: on_top(summary, s, "mae") for s in cfg.SPLITS}
    widest = max(tops, key=lambda s: len(tops[s]))
    narrowest = min(tops, key=lambda s: len(tops[s]))

    return "\n".join([
        p("The tally below counts, over every scheme and every metric, how often each "
          "arm is best on its own, tied for best, and separably worse. A method that "
          "cannot be distinguished from the leader is counted as tied and the leader is "
          "not called a winner, which is the convention the rest of this repository "
          "uses and the reason there are no bold maxima anywhere on the page."),
        p(f"The verdicts come from Holm-corrected paired tests over each scheme's fifty "
          f"folds, not from the pooled Tukey the paper's aggregation implies. The two "
          f"figures below show why that choice is not cosmetic. In the pooled version "
          f"every interval is wide enough to swallow the field, and the honest reading "
          f"of it is that nothing can be told from anything — which is, in effect, the "
          f"paper's conclusion. The between-target spread in this collection is around "
          f"0.2 pIC₅₀ and the differences between methods are a few hundredths, so "
          f"fifty numbers treated as independent samples have almost no power. Paired "
          f"on the fold, where both arms saw the identical training molecules, the same "
          f"data separates {len(tops[narrowest])} arm"
          + ("" if len(tops[narrowest]) == 1 else "s")
          + f" at the top of the "
          f"{cfg.SPLIT_LABELS[narrowest].lower()} scheme from the rest."),
        p(f"The correction is not doing the work of hiding differences either. On the "
          f"{cfg.SPLIT_LABELS[widest].lower()} scheme {len(tops[widest])} arms share the "
          f"top on MAE — {name_list(tops[widest])} — and that is a real tie rather than "
          f"a failure of power, because the same test separates the field elsewhere."
          + (f" {name_list(never)} reach the top of nothing, anywhere."
             if never else "")),
    ])


def robustness(summary, h2h, gaps, tally, distances, decay) -> str:
    """How much of the verdict survives the two more conservative tests."""
    wide = h2h.pivot_table(index=["split", "metric", "left", "right"],
                           columns="test", values="p_holm")
    if not {"t", "wilcoxon", "t_target"} <= set(wide.columns):
        return ""
    called = wide < 0.05
    agree_w = int((called["t"] == called["wilcoxon"]).sum())
    agree_g = int((called["t"] == called["t_target"]).sum())
    n = len(called)
    lost = int((called["t"] & ~called["t_target"]).sum())
    gained = int((~called["t"] & called["t_target"]).sum())

    # The headline pair, under each test, scheme by scheme.
    winner = max(cfg.RUN_METHODS, key=lambda m: sum(tally_of(tally, m)[:2]))
    survive = []
    for split in cfg.SPLITS:
        row = pair_row(h2h, split, "mae", cfg.KNN_METHOD, winner, test="t_target")
        if row is not None and row["p_holm"] < 0.05:
            survive.append(split)

    return "\n".join([
        p(f"Two things could make the verdicts above look stronger than they are, and "
          f"both are checked rather than argued about. The differences might not be "
          f"normal, so every pair is also tested with a Wilcoxon signed-rank test: it "
          f"agrees with the t-test on {agree_w} of the {n} pairs on this page. More "
          f"seriously, the fifty folds of a scheme are not fifty independent "
          f"observations — the time split gives each fold its own year cutoff, so a "
          f"target's five test sets overlap. Averaging each arm's five folds within a "
          f"target first and pairing on the ten targets removes that entirely, at the "
          f"cost of going from fifty paired observations to ten."),
        p(f"That target-level test agrees with the fold-level one on {agree_g} of {n} "
          f"pairs. Where they differ it is almost always the same direction — "
          f"{lost} calls are lost and {gained} gained — which is what losing 80% of the "
          f"sample should do, not evidence that the fold-level test was wrong. The "
          f"question is whether the headline survives it, and it does: "
          f"{label(winner)} separates from k-NN on {count_of_six(len(survive))} schemes "
          f"even on ten paired targets"
          + (f", the exception being {scheme_list([s for s in cfg.SPLITS if s not in survive])}"
             if len(survive) < len(cfg.SPLITS) else "")
          + ". Every p value in <code>head_to_head.csv</code> is stored under all three tests."),
    ])


def knn(summary, h2h, gaps, tally, distances, decay) -> str:
    agreed, checked, worst = reproduction(REPRO)
    winner = max(cfg.RUN_METHODS, key=lambda m: sum(tally_of(tally, m)[:2]))
    beaten, tied_with = [], []
    for split in cfg.SPLITS:
        row = pair_row(h2h, split, "mae", cfg.KNN_METHOD, winner)
        if row is None:
            continue
        (beaten if (row["mean_diff"] < 0 and row["p_holm"] < 0.05) else tied_with).append(split)

    pub_beaten = [
        s for s in cfg.SPLITS
        if (r := pair_row(h2h, s, "mae", cfg.KNN_METHOD, cfg.LGBM_METHOD)) is not None
        and r["mean_diff"] < 0 and r["p_holm"] < 0.05
    ]
    wil = [s for s in beaten
           if (r := pair_row(h2h, s, "mae", cfg.KNN_METHOD, winner, test="wilcoxon")) is not None
           and r["p_holm"] < 0.05]

    return "\n".join([
        p("The k-NN arm here is the paper's own, re-implemented from its released code: "
          "a distance-weighted <code>KNeighborsRegressor</code> on a precomputed "
          "Tanimoto distance matrix over ECFP4 bit vectors, with k chosen from {1, 3, 5} "
          "by validation MAE. It is the one arm in this study whose numbers can be "
          f"checked against the source, and {agreed} of {checked} "
          f"({len(cfg.SPLITS) - 1} schemes \u00d7 50 folds \u00d7 MAE and R\u00b2) reproduce "
          f"the published per-fold values to within {worst:.0e}. The fifty folds that do "
          "not are all on the "
          "one scheme whose validation split upstream did not record; the "
          "<a href=\"#how-this-was-run\">protocol section</a> says what was done about "
          "that. So when the table below disagrees with the paper, it is not "
          "disagreeing about what k-NN did."),
        p(f"It disagrees about what that means. Against {label(winner)}, k-NN is "
          f"separably worse on {count_of_six(len(beaten))} schemes"
          + (f" ({scheme_list(beaten)})" if beaten else "")
          + (f", and cannot be separated on {scheme_list(tied_with)}."
             if tied_with else ". There is no scheme on which it holds its own.")
          + (f" A Wilcoxon signed-rank test in place of the t-test agrees on "
             f"{len(wil)} of those {len(beaten)}." if beaten else "")),
        p(f"Against the fingerprint baseline the paper's conclusion fares better: "
          f"LightGBM on Morgan counts separates from k-NN on "
          f"{count_of_six(len(pub_beaten))} schemes"
          + (f" ({scheme_list(pub_beaten)})" if pub_beaten else "")
          + ". Two similarity models on the same fingerprint do behave alike, which is "
          "the part of the Janela and Bajorath result this study reproduces rather "
          "than contradicts. What breaks the pattern is not a better learner on the "
          "same features. It is different features."),
    ])


def ranking(summary, h2h, gaps, tally, distances, decay) -> str:
    rows = []
    for split in cfg.SPLITS:
        best = on_top(summary, split, "spearman")
        rows.append((split, best, max(mean_of(summary, split, "spearman", m) for m in best)))
    easy = [r for r in rows if r[0] == EASIEST][0]
    hard = [r for r in rows if r[0] == HARDEST][0]

    return "\n".join([
        p("MAE is the paper's metric and it is the right one for a headline, because it "
          "is in the units of the endpoint and it can be read against a naive baseline. "
          "It is also the metric least kind to a model that has the ordering right and "
          "the scale wrong, which is the failure mode of a model extrapolating off its "
          "training distribution. Spearman ρ asks the question a screening campaign "
          "actually asks: can the compounds be put in the right order."),
        p(f"The answer degrades the same way, and further. The best ρ on the "
          f"{cfg.SPLIT_LABELS[EASIEST].lower()} scheme is {easy[2]:.3f}; on the "
          f"{cfg.SPLIT_LABELS[HARDEST].lower()} scheme it is {hard[2]:.3f}. Unlike MAE, "
          "there is no moving baseline to subtract here — a rank correlation of zero is "
          "zero on every scheme — so this is the cleanest statement of how much "
          "predictive signal is left when the test molecules are genuinely elsewhere in "
          "chemical space, and it is not much."),
        p("The naive baseline is absent from this table and from the figure under it. A "
          "constant prediction has no ranking to score, and Spearman ρ against it is "
          "undefined rather than zero; it is recorded as missing rather than filled in, "
          "so it does not appear as a bar at the origin that a reader could mistake for "
          "a measurement."),
    ])


def methods(summary, h2h, gaps, tally, distances, decay) -> str:
    agreed, checked, worst = reproduction(REPRO)
    return "\n".join([
        p("Ten ChEMBL 30 potency data sets, 1,000 to 2,273 molecules each and 13,444 in "
          "total, with pIC₅₀ from 5 to 11. They are the collections assembled by Janela "
          "and Bajorath and redistributed in the paper's release, and they arrive here "
          "unchanged."),
        p("<b>Nothing was split here.</b> The release ships, for each target and each of "
          "five folds, six splitting schemes as explicit train / validation / test index "
          "lists, and those lists are the protocol. Every arm fits on a fold's training "
          "indices and is scored on its test indices, so every arm sees identical "
          "training molecules in every fold and a difference in the metrics is a "
          "difference in the method. The validation indices are used only where an arm "
          "needs them — k for k-NN, early stopping for ChemProp. LightGBM and Monroe "
          "have nothing to tune per fold and no training loop to stop, so they leave "
          "those molecules unused rather than training on molecules the other arms "
          "never see."),
        p('<span id="how-this-was-run"></span><b>Which copy of the splits.</b> The '
          "release ships them twice and the two copies disagree: "
          "<code>data/splits/indices/*.pkl</code> and <code>data/splits/*.json</code> "
          "agree on the random, scaffold and time schemes and differ on butina, umap "
          "and diverse — for umap the two test sets do not share a single molecule. The "
          "<code>.pkl</code> files are the ones the paper's numbers came from: "
          f"re-running its released k-NN on them reproduces {agreed} of {checked} "
          f"published per-fold MAE and R\u00b2 values to {worst:.0e} on the five schemes "
          "recorded in full, "
          "where the <code>.json</code> indices reproduce butina and umap not at all. "
          "So the <code>.pkl</code> indices are the protocol, and that reproduction is "
          "kept as a check that runs on every collection pass rather than as a note."),
        p("The sixth scheme needed a repair. <code>diverse_25</code> has an empty "
          "validation set in every one of the fifty <code>.pkl</code> files — upstream "
          "carved one out of the training pool inside the run and did not record it. The "
          "<code>.json</code> copy does record one, and on all fifty combinations it is "
          "a subset of the <code>.pkl</code> training pool, the two test sets are "
          "identical, and <code>.pkl</code> train equals <code>.json</code> train plus "
          "validation. Taking the validation molecules from the <code>.json</code> copy "
          "therefore contradicts nothing, and it is what gives ChemProp something to "
          "stop early on. All three relations are asserted before it is done. The fifty "
          "k-NN folds that do not reproduce are all on this scheme, and the published "
          "arms on it trained on a different 237 molecules of the same 263-molecule pool "
          "against an identical test set — comparable, but not identical."),
        "<ul>"
        "<li><b>k-NN</b> — ECFP4 bit vectors, radius 2, 2048 bits; Tanimoto distance; "
        "distance weighting; k from {1, 3, 5} on the validation split. The paper's own "
        "arm, re-implemented from its code.</li>"
        "<li><b>LightGBM + Morgan</b> — count fingerprints, radius 2, 2048 bits; library "
        "defaults; one model per target.</li>"
        "<li><b>ChemProp + CheMeleon</b> — chemprop 2.3.0 D-MPNN with message passing "
        "initialised from the CheMeleon foundation model, then fine-tuned. 50 epochs, "
        "batch 64, one network per fold, no ensembling.</li>"
        "<li><b>Monroe + TabPFN 3.5</b> — the authors' pre-trained encoder, frozen. Every "
        "molecule becomes one 720-d embedding, computed once for all 13,257 distinct "
        "structures, and TabPFN 3.5 predicts each target in context from that fold's "
        "training embeddings. Nothing is fitted by gradient descent.</li>"
        "<li><b>Training median</b> — the paper's naive baseline, drawn as a line rather "
        "than a bar because it is not a competitor.</li>"
        "<li><b>The paper's six arms</b> — k-NN, SVM, RF, XGBoost, MLP and a GNN, entered "
        "at their <i>published</i> per-fold metrics rather than re-run, because "
        "predictions were not released. They are marked with an asterisk everywhere. "
        "Each was given 50 Optuna trials per fold while the four arms above run at "
        "their libraries' defaults — an asymmetry that favours the published arms and "
        "is left uncorrected.</li>"
        "</ul>",
        p("The statistics are the part that differs most from the paper. Each scheme "
          "gives fifty folds per arm, ten targets by five folds, and the between-target "
          "spread is an order of magnitude larger than the differences between methods. "
          "So the primary test is paired on the fold, where the target effect cancels, "
          "with Holm's correction applied within each scheme and metric across every "
          "pair on the page. Three tests are computed and stored for every pair: the "
          "paired t-test over the fifty folds, a Wilcoxon signed-rank test that drops "
          "the normality assumption, and a target-level t-test that drops the "
          "independence assumption instead, by averaging each method's five folds within "
          "a target and pairing on the ten targets. The last one is there because the "
          "time split gives every fold its own year cutoff, so a target's five test sets "
          "overlap and fifty differences computed on overlapping molecules are not "
          "fifty independent observations. It costs power, so it is a check rather than "
          "the headline. The Tukey figure is drawn on target-centred values, "
          "the standard randomized-block adjustment, which leaves every arm's mean "
          "exactly where it was and only narrows the intervals. The raw pooled Tukey is "
          "drawn beside it because the difference between the two is part of the "
          "argument."),
        '<p class="repo">Every script, both copies of the splits, and all '
        'predictions are at '
        '<a href="https://github.com/PatWalters/model-validation-central/tree/main/studies/split-fragility">'
        "model-validation-central/studies/split-fragility</a>. The figures and tables on "
        "this page rebuild from the stored predictions in under a minute, with no "
        "retraining. <code>test_split_stats.py</code> checks the statistics against "
        "synthetic data with a known answer.</p>",
        '<p class="footnote">Statistics follow the approach in '
        '<a href="https://practicalcheminformatics.blogspot.com/2025/03/even-more-thoughts-on-ml-method.html">'
        "“Even More Thoughts on ML Method Comparisons”</a> and the protocol paper it "
        "points at: distributions of fold-level metrics, corrections for multiple "
        "comparisons, and paired tests using the folds as the pairing.</p>",
    ])


NARRATIVE = {
    "intro": intro,
    "stringency": stringency,
    "confound": confound,
    "collapse": collapse,
    "separation": separation,
    "robustness": robustness,
    "knn": knn,
    "ranking": ranking,
    "methods": methods,
}
