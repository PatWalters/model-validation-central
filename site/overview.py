#!/usr/bin/env python3
"""The cross-study comparison at the top of the index page.

Every study here fits its methods on identical folds and scores them on the same
untouched test set, and the four reference arms are not refit between studies --
their per-fold numbers are copied, so they are identical to the last bit. That is
checked here rather than assumed. It means the per-fold metrics from all three
studies can be put into one Tukey HSD without any of the usual objections to
pooling results across papers.

    python site/overview.py                 # MAE, the default
    python site/overview.py --metric r2     # or any of the three

Writes, for `site/build_site.py` to embed:

    docs/assets/tukey_<metric>_<dataset>.png  one panel per endpoint, as the studies draw them
    docs/overview.json                   the tally table, and what went into it

Unlike `build_site.py`, this needs the scientific stack -- pandas, statsmodels,
matplotlib -- so it is a separate script and is run only when a study's numbers
change. `build_site.py` reads what this leaves behind and stays runnable in a
bare interpreter.

Counting is tie-aware, and deliberately so. `tukey_groups` labels the method with
the leading mean 'best' and anything the correction cannot separate from it
'equivalent'; that asymmetry is an artefact of how the label is computed, not a
finding. So both count as being on top, a combination with more than one method
on top makes every one of them 'tied', and only a combination with exactly one
method on top gives anyone a 'best alone'. This is the same rule the study
reports use, and it is the reason no cell here is a bolded maximum.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STUDIES = ROOT / "studies"
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"

# `model_comparison.py` is byte-identical in all three studies. Importing one
# copy beats keeping a fourth here that could drift away from them.
sys.path.insert(0, str(STUDIES / "expansion-ml-comparison"))
from model_comparison import make_tukey_plot, tukey_groups   # noqa: E402

# --- what goes into the comparison ---------------------------------------
#
# The four reference arms come from the study that fit them. Each study then
# contributes the arms it exists to test. multimodal-fusion's 33 configurations
# would swamp a 10-way comparison and widen Tukey's correction for everyone, so
# it is represented by one cell of its grid, chosen in advance rather than on the
# results: all four modalities, early fusion, LightGBM.
SOURCES = {
    "expansion-ml-comparison": ["lgbm", "chemprop_st", "chemprop", "chemeleon",
                                "megacl", "monroe", "moljepa", "trimole"],
    "ecfp-pretrain": ["ptgin"],
    "multimodal-fusion": ["fus_GRMS_early_lgbm"],
}
REFERENCE_ARMS = ["lgbm", "chemprop_st", "chemprop", "chemeleon"]

LABELS = {
    "lgbm": "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    "chemeleon": "ChemProp + CheMeleon",
    "megacl": "MEGA-CL",
    "monroe": "Monroe + TabPFN",
    "moljepa": "Mol-JEPA + TabICL",
    "trimole": "Trimole-Hybrid",
    "ptgin": "PT-GIN + LightGBM",
    "fus_GRMS_early_lgbm": "Fusion, four modalities",
}
# The colours the study reports already give these methods, so a method looks
# the same here as it does on the page it came from.
COLORS = {
    "lgbm": "#4C72B0", "chemprop_st": "#C44E52", "chemprop": "#DD8452",
    "chemeleon": "#55A868", "megacl": "#8172B3", "monroe": "#937860",
    "moljepa": "#DA8BC3", "trimole": "#CCB974", "ptgin": "#4878CF",
    "fus_GRMS_early_lgbm": "#6ACC64",
}
ORDER = list(LABELS)

DATASETS = {"expansion": "ExpansionRx", "biogen": "Biogen ADME"}
METRICS = {"r2": ("R²", True), "spearman": ("Spearman ρ", True), "mae": ("MAE", False)}
N_COLS = 3

# The metric the front page draws. The tally table counts all three regardless;
# this only chooses which one gets a figure. MAE is the default because it is in
# the units of the endpoint, so a reader can tell what a difference costs.
HEADLINE_METRIC = "mae"


# ------------------------------------------------------------------ loading
def load(dataset: str) -> pd.DataFrame:
    """Every contributing arm's per-fold metrics for one data set, in one frame."""
    frames, refs = [], {}
    for study, methods in SOURCES.items():
        path = STUDIES / study / "results" / dataset / "fold_metrics.csv"
        if not path.exists():
            raise SystemExit(f"missing {path.relative_to(ROOT)} -- run that study's collector")
        d = pd.read_csv(path)
        refs[study] = d[d["method"].isin(REFERENCE_ARMS)]
        take = d[d["method"].isin(methods)].copy()
        missing = set(methods) - set(take["method"])
        if missing:
            raise SystemExit(f"{path.relative_to(ROOT)} has no {', '.join(sorted(missing))}")
        take["study"] = study
        frames.append(take)
    check_references_agree(refs, dataset)
    return pd.concat(frames, ignore_index=True)


def check_references_agree(refs: dict, dataset: str) -> None:
    """The shared arms must be the same numbers, or none of this pools.

    They are copied between studies rather than refit, so anything other than an
    exact match means a study is carrying a stale import and the comparison would
    be silently mixing two different runs.
    """
    key = ["endpoint", "method", "repeat", "fold"]
    cols = ["r2", "spearman", "mae"]
    base_name, base = next(iter(refs.items()))
    base = base.set_index(key).sort_index()
    for name, other in refs.items():
        if name == base_name:
            continue
        other = other.set_index(key).sort_index()
        if not base.index.equals(other.index):
            raise SystemExit(f"{dataset}: {name} and {base_name} disagree on which folds exist")
        worst = (base[cols] - other[cols]).abs().to_numpy().max()
        if worst > 0:
            raise SystemExit(
                f"{dataset}: reference arms differ between {base_name} and {name} "
                f"by up to {worst:g}. One of them is a stale import; re-run its "
                f"baseline import before trusting a combined comparison."
            )


# ------------------------------------------------------------------ the tally
def check_combination(endpoint, metric, groups, on_top, n_methods) -> None:
    """The tie rule, asserted rather than trusted.

    Every endpoint x metric combination must put at least one method on top --
    the leading mean is always labelled 'best', so an empty top means the label
    column has changed meaning underneath us. And 'alone' has to mean alone: if
    two methods cannot be separated, neither may be counted as an outright
    winner. Getting this wrong is exactly the bolded-maximum habit these studies
    exist to avoid, so it is worth failing the build over.
    """
    where = f"{endpoint} / {metric}"
    if len(groups) != n_methods:
        raise SystemExit(f"{where}: {len(groups)} methods labelled, expected {n_methods}")
    if len(on_top) == 0:
        raise SystemExit(f"{where}: no method on top, which tukey_groups cannot produce")
    if len(on_top) > n_methods:
        raise SystemExit(f"{where}: {len(on_top)} methods on top of {n_methods}")
    leaders = groups[groups == "best"]
    if len(leaders) != 1:
        raise SystemExit(f"{where}: {len(leaders)} methods labelled 'best', expected exactly one")


def tally(metrics: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, int]:
    """Per method: alone at the top, sharing it, or significantly worse.

    One Tukey HSD per endpoint over the 25 folds, on the same metric the figure
    draws, then counted the way the module docstring describes: ties are shared,
    not broken. Table and figure move together, so a row and the panel above it
    can never disagree about who won.
    """
    _, higher = METRICS[metric]
    endpoints = sorted(metrics["endpoint"].unique())
    n_methods = metrics["method"].nunique()
    rows = []
    for endpoint in endpoints:
        sub = metrics[metrics["endpoint"] == endpoint]
        groups = tukey_groups(sub, metric, higher_is_better=higher)
        on_top = groups[groups.isin(("best", "equivalent"))].index
        for method in groups.index:
            if method not in on_top:
                kind = "worse"
            else:
                kind = "tied" if len(on_top) > 1 else "alone"
            rows.append({"endpoint": endpoint, "metric": metric,
                         "method": method, "kind": kind})
        check_combination(endpoint, metric, groups, on_top, n_methods)
    counts = pd.crosstab(pd.DataFrame(rows)["method"], pd.DataFrame(rows)["kind"])
    for col in ("alone", "tied", "worse"):
        if col not in counts:
            counts[col] = 0
    return counts, len(endpoints)


# ----------------------------------------------------------------- the figure
def figure(metrics: pd.DataFrame, dataset: str, metric: str = "r2") -> Path:
    """One Tukey panel per endpoint, the way each study draws its own."""
    label, higher = METRICS[metric]
    endpoints = sorted(metrics["endpoint"].unique())
    rows = -(-len(endpoints) // N_COLS)
    fig, axes = plt.subplots(rows, N_COLS, figsize=(4.6 * N_COLS, 3.5 * rows), squeeze=False)
    axes = axes.ravel()

    xlims = []
    for ax, endpoint in zip(axes, endpoints):
        sub = metrics[metrics["endpoint"] == endpoint]
        make_tukey_plot(sub, metric, higher_is_better=higher, ax=ax,
                        title=endpoint, xlabel=label)
        ax.set_yticklabels([LABELS.get(t.get_text(), t.get_text())
                            for t in ax.get_yticklabels()])
        xlims.append(ax.get_xlim())
    for ax in axes[len(endpoints):]:
        ax.set_visible(False)

    lo, hi = min(x[0] for x in xlims), max(x[1] for x in xlims)
    bottom_row = len(endpoints) - N_COLS
    for i, ax in enumerate(axes[:len(endpoints)]):
        ax.set_xlim(lo, hi)
        if i % N_COLS:
            ax.set_yticklabels([])
        if i < bottom_row:
            ax.set_xticklabels([])
            ax.set_xlabel("")

    fig.suptitle(
        f"{DATASETS[dataset]} — Tukey HSD on {label}. "
        "Blue: best. Grey: indistinguishable from it. Red: significantly worse.",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    ASSETS.mkdir(parents=True, exist_ok=True)
    out = ASSETS / f"tukey_{metric}_{dataset}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ------------------------------------------------------------------- driver
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metric", choices=list(METRICS), default=HEADLINE_METRIC,
                    help=f"which metric the figure draws (default {HEADLINE_METRIC})")
    args = ap.parse_args()
    metric_label, _ = METRICS[args.metric]

    payload = {
        "methods": [],
        "datasets": {},
        "metric": args.metric,
        "metric_label": metric_label,
        "note": ("One Tukey HSD per endpoint over the 25 folds, on the metric the "
                 "figures draw. A method is counted as being on top whenever the "
                 "correction cannot separate it from the leading mean, so an endpoint "
                 "with several methods on top gives each of them a tie rather than "
                 "crowning one."),
    }
    study_of = {m: s for s, ms in SOURCES.items() for m in ms}
    tallies = {}
    for dataset in DATASETS:
        metrics = load(dataset)
        counts, combos = tally(metrics, args.metric)
        tallies[dataset] = counts
        out = figure(metrics, dataset, metric=args.metric)
        payload["datasets"][dataset] = {
            "label": DATASETS[dataset],
            "endpoints": int(metrics["endpoint"].nunique()),
            "combinations": combos,
            "figure": f"assets/{out.name}",
            "folds_per_method": int(metrics.groupby("method").size().iloc[0]),
        }
        print(f"{dataset}: {metrics['method'].nunique()} methods, {combos} combinations, "
              f"{out.relative_to(ROOT)}")

    for method in ORDER:
        row = {"method": method, "label": LABELS[method], "color": COLORS[method],
               "study": study_of[method], "counts": {}}
        for dataset, counts in tallies.items():
            c = counts.loc[method] if method in counts.index else {}
            row["counts"][dataset] = {k: int(c.get(k, 0)) for k in ("alone", "tied", "worse")}
        payload["methods"].append(row)

    (DOCS / "overview.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {(DOCS / 'overview.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
