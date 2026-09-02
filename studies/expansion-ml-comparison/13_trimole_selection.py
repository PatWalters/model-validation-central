#!/usr/bin/env python
"""Step 13: what did Trimole-Hybrid actually choose?

The other methods in this repository have nothing to report beyond their scores,
because a method is one model and it either fits well or it does not. This arm is
different: its whole claim is that the right model depends on the endpoint, so the
interesting output is not only how well it did but *what it picked*, 25 times per
endpoint, with no knowledge of the test set.

That makes the selection records a result in their own right, and one that can
falsify the claim. If the method is doing what it says, the choice should vary
between endpoints and be reasonably stable within one. If instead it picks the
same candidate everywhere, the pool is decoration and a single well-chosen model
would have done as well for a sixtieth of the compute. If it picks a different
candidate on every fold of the same endpoint, it is fitting the validation split
rather than the chemistry, and the choice carries no signal.

Reads results/<ds>/trimole_selection/*.json and writes, under the trimole
comparison's own directory:

  tables/selection.csv        one row per endpoint x fold, the winner and its score
  tables/selection_counts.csv how often each view, block and backend won, by endpoint
  figures/trimole_selection   the same as a stacked bar, one panel per axis

    python 13_trimole_selection.py
    ADME_DATASET=biogen python 13_trimole_selection.py
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import config as cfg

# The three axes of the candidate pool, in the order the page discusses them.
AXES = ["selected_view", "selected_block", "selected_backend"]
AXIS_LABELS = {
    "selected_view": "molecular view",
    "selected_block": "chemistry block",
    "selected_backend": "backend",
}

# One colour per level, chosen so the view panel reads as a progression from
# chemistry only through to everything fused.
COLORS = {
    "chem": "#4C72B0", "chemberta": "#DD8452", "kpgt": "#55A868",
    "unimol": "#C44E52", "fused": "#8172B3",
    "core_maccs_fcfp": "#4C72B0", "core_pair_torsion": "#DD8452",
    "wide_chem": "#55A868",
    "xgb": "#4C72B0", "extratrees": "#DD8452", "rf": "#55A868", "ridge": "#C44E52",
}


def load_records() -> pd.DataFrame:
    directory = cfg.RESULTS_DIR / "trimole_selection"
    paths = sorted(directory.glob("*_r*_f*.json"))
    if not paths:
        raise SystemExit(
            f"no selection records in {directory} -- run 12_run_trimole.py first"
        )

    rows = []
    for path in paths:
        record = json.loads(path.read_text())
        rows.append({k: v for k, v in record.items() if k != "pool"})
    df = pd.DataFrame(rows)

    expected = len(cfg.TARGET_COLS) * cfg.N_REPEATS * cfg.N_SPLITS
    if len(df) != expected:
        print(f"note: {len(df)} of {expected} folds have a selection record")

    order = {endpoint: i for i, endpoint in enumerate(cfg.TARGET_COLS)}
    return df.sort_values(
        ["endpoint", "repeat", "fold"], key=lambda s: s.map(order) if s.name == "endpoint" else s
    ).reset_index(drop=True)


def counts_table(df: pd.DataFrame) -> pd.DataFrame:
    """How often each level of each axis won, per endpoint."""
    rows = []
    for axis in AXES:
        tally = df.groupby(["endpoint", axis]).size().rename("folds").reset_index()
        tally = tally.rename(columns={axis: "level"})
        tally.insert(1, "axis", axis)
        rows.append(tally)
    out = pd.concat(rows, ignore_index=True)
    out["share"] = out["folds"] / (cfg.N_REPEATS * cfg.N_SPLITS)
    return out


def plot_selection(df: pd.DataFrame, path_stem) -> None:
    endpoints = [e for e in cfg.TARGET_COLS if e in set(df["endpoint"])]
    fig, axes = plt.subplots(
        1, len(AXES), figsize=(5.6 * len(AXES), 0.42 * len(endpoints) + 2.4), sharey=True
    )

    for ax, axis in zip(axes, AXES):
        wide = (
            df.groupby(["endpoint", axis]).size().unstack(fill_value=0)
            .reindex(endpoints)
        )
        left = pd.Series(0, index=wide.index, dtype=float)
        for level in wide.columns:
            ax.barh(
                wide.index, wide[level], left=left,
                color=COLORS.get(level, "#999999"), label=level,
                edgecolor="white", linewidth=0.6,
            )
            left += wide[level]
        ax.set_xlim(0, cfg.N_REPEATS * cfg.N_SPLITS)
        ax.set_xlabel(f"folds selecting each {AXIS_LABELS[axis]}")
        ax.invert_yaxis()
        ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
        ax.set_title(AXIS_LABELS[axis], fontsize=11)

    fig.suptitle(
        f"What Trimole-Hybrid selected on {cfg.ACTIVE.label}, "
        f"over {cfg.N_REPEATS}x{cfg.N_SPLITS} folds",
        fontsize=12,
    )
    fig.tight_layout()
    for suffix in (".png", ".svg"):
        fig.savefig(path_stem.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)


def report_stability(df: pd.DataFrame) -> None:
    """Say plainly whether the choice varies by endpoint and holds within one."""
    print(f"\n{cfg.ACTIVE.label}: {len(df)} folds")
    for axis in AXES:
        # The share held by the most common choice within an endpoint, averaged.
        within = df.groupby("endpoint")[axis].agg(
            lambda s: s.value_counts().iloc[0] / len(s)
        )
        distinct = df.groupby("endpoint")[axis].nunique()
        overall = df[axis].value_counts()
        print(
            f"  {AXIS_LABELS[axis]:<16} most common overall: {overall.index[0]} "
            f"({overall.iloc[0]}/{len(df)} folds); "
            f"within an endpoint the modal choice holds {within.mean():.0%} of folds, "
            f"{distinct.mean():.1f} distinct choices per endpoint"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.parse_args()

    if cfg.COMPARISON != "trimole":
        raise SystemExit("run with ADME_COMPARISON=trimole, so the output path is right")

    cfg.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_records()
    df.to_csv(cfg.TABLE_DIR / "selection.csv", index=False)

    counts = counts_table(df)
    counts.to_csv(cfg.TABLE_DIR / "selection_counts.csv", index=False)

    plot_selection(df, cfg.FIGURE_DIR / "trimole_selection")
    report_stability(df)

    print(f"\nwrote {cfg.TABLE_DIR / 'selection.csv'}")
    print(f"wrote {cfg.TABLE_DIR / 'selection_counts.csv'}")
    print(f"wrote {cfg.FIGURE_DIR / 'trimole_selection.png'}")


if __name__ == "__main__":
    main()
