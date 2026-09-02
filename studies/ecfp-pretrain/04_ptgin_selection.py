#!/usr/bin/env python
"""Step 4: what did the checkpoint selection actually choose, and did it matter?

The paper does not have one PT-GIN. It pre-trains a grid of maximum substructure
radius by vocabulary size and picks, per task, whichever pre-trained model does
best downstream. Ten of those checkpoints are released, so the choice is part of
the method and gets reproduced -- on the validation fifth, never on the test set.

That makes the 10 x endpoint x fold validation table a result rather than
bookkeeping, and one that can deflate the selection step. Three things are worth
knowing and only one of them is the winner's name:

  - the spread. If the best and worst checkpoints are 0.005 apart, the grid is
    decoration and any one of them would have done.
  - the margin. If the winner beats the runner-up by less than the fold-to-fold
    noise, the choice is a coin toss dressed as a decision.
  - whether the winner is the same everywhere. If one checkpoint wins every
    endpoint, per-task selection is not what the method needs; if a different one
    wins each time with a small margin, it is fitting the validation split.

Reads results/<ds>/ptgin_selection.csv and ptgin_choice.csv, written by step 1.
Writes:

  tables/selection_by_endpoint.csv   mean validation R^2, every checkpoint x endpoint
  tables/selection_summary.csv       the winner, its margin, and the grid's spread
  figures/ptgin_selection.png        the same as a heatmap, with the winner marked

    python 04_ptgin_selection.py
    ADME_DATASET=biogen python 04_ptgin_selection.py
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.SELECTION_CSV.exists():
        raise SystemExit(
            f"{cfg.SELECTION_CSV} not found -- run 01_run_ptgin.py --select first"
        )
    table = pd.read_csv(cfg.SELECTION_CSV)
    choice = pd.read_csv(cfg.CHOICE_CSV)

    expected = len(cfg.TARGET_COLS) * len(cfg.CHECKPOINTS) * cfg.N_REPEATS * cfg.N_SPLITS
    if len(table) != expected:
        print(f"note: {len(table)} of {expected} (endpoint x checkpoint x fold) rows present")
    return table, choice


def by_endpoint(table: pd.DataFrame) -> pd.DataFrame:
    """Mean validation R^2 for every checkpoint on every endpoint."""
    wide = (
        table.groupby(["endpoint", "checkpoint"])["val_r2"].mean().unstack()
        .reindex(index=[e for e in cfg.TARGET_COLS if e in set(table["endpoint"])],
                 columns=cfg.CHECKPOINTS)
    )
    wide.to_csv(cfg.TABLE_DIR / "selection_by_endpoint.csv")
    print("wrote tables/selection_by_endpoint.csv")
    return wide


def summarise(table: pd.DataFrame, choice: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """The winner, how far ahead it is, and how far apart the grid is.

    `fold_sd` is the standard deviation of the winner's validation R^2 over its 25
    folds. Comparing the margin against it is the whole question: a margin well
    inside one fold's worth of noise is not a decision the data supports.
    """
    rows = []
    for _, row in choice.iterrows():
        endpoint = row["endpoint"]
        scores = wide.loc[endpoint].dropna()
        winner_folds = table[
            (table["endpoint"] == endpoint) & (table["checkpoint"] == row["checkpoint"])
        ]["val_r2"]
        rows.append(
            {
                "endpoint": endpoint,
                "checkpoint": row["checkpoint"],
                "radius": row["radius"],
                "vocab": row["vocab"],
                "val_r2": scores.max(),
                "margin_over_runner_up": row["margin"],
                "spread_best_to_worst": scores.max() - scores.min(),
                "fold_sd": float(winner_folds.std()),
                "margin_in_fold_sd": row["margin"] / float(winner_folds.std()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.TABLE_DIR / "selection_summary.csv", index=False)
    print("wrote tables/selection_summary.csv\n")

    print(f"{'endpoint':<17} {'chosen':<30} {'val R2':>7} {'margin':>8} "
          f"{'spread':>8} {'fold sd':>8}")
    for _, row in summary.iterrows():
        print(f"{row['endpoint']:<17} {row['checkpoint']:<30} {row['val_r2']:7.3f} "
              f"{row['margin_over_runner_up']:8.3f} {row['spread_best_to_worst']:8.3f} "
              f"{row['fold_sd']:8.3f}")

    counts = summary["checkpoint"].value_counts()
    print(f"\n{len(counts)} distinct checkpoints chosen across "
          f"{len(summary)} endpoints:")
    for name, n in counts.items():
        print(f"  {name:<30} {n}")
    print(f"\nmedian margin over the runner-up: {summary['margin_over_runner_up'].median():.3f} "
          f"R^2, {summary['margin_in_fold_sd'].median():.2f} fold standard deviations")
    print(f"median spread best to worst:      {summary['spread_best_to_worst'].median():.3f} R^2")
    return summary


def fold_noise(table: pd.DataFrame) -> pd.Series:
    """Per endpoint, how much one checkpoint's validation R^2 moves between folds.

    Averaged over the ten checkpoints, so it is a property of the endpoint rather
    than of whichever checkpoint happened to win. This is the yardstick the whole
    selection question turns on: a difference between checkpoints only means
    something if it is large compared with how much a single checkpoint wanders.
    """
    return (
        table.groupby(["endpoint", "checkpoint"])["val_r2"].std()
        .groupby("endpoint").mean()
    )


def plot(table: pd.DataFrame, wide: pd.DataFrame, summary: pd.DataFrame) -> None:
    """A heatmap of the grid, with each endpoint's chosen checkpoint boxed.

    Two decisions make this figure say what the data says rather than what a
    default colour map would say.

    Rows are centred on their own mean, because the interesting variation is
    *within* an endpoint -- which checkpoint suits it -- and the between-endpoint
    variation would otherwise swamp it entirely.

    Then each row is divided by that endpoint's fold-to-fold noise, and the
    colour axis runs from -1 to +1 of it. Colouring the raw R^2 deviations
    instead would stretch a 0.02 spread across the full colour map and paint a
    vivid picture of differences smaller than the measurement wobbles. On this
    scale a cell is only coloured if it is meaningfully far from its row's mean,
    so a pale grid is the honest rendering of a grid that does not matter.
    """
    noise = fold_noise(table).reindex(wide.index)
    centred = wide.sub(wide.mean(axis=1), axis=0)
    scaled = centred.div(noise, axis=0)

    fig, ax = plt.subplots(figsize=(1.05 * len(wide.columns) + 3.4,
                                    0.46 * len(wide.index) + 2.8))
    image = ax.imshow(scaled.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels([cfg.checkpoint_label(c) for c in wide.columns],
                       rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels(
        [f"{e}\n±{noise[e]:.3f}" for e in wide.index], fontsize=8
    )

    chosen = dict(zip(summary["endpoint"], summary["checkpoint"]))
    columns = {name: i for i, name in enumerate(wide.columns)}
    for row, endpoint in enumerate(wide.index):
        for col, name in enumerate(wide.columns):
            value = wide.loc[endpoint, name]
            if np.isfinite(value):
                # White on a saturated cell, dark on a pale one, so nothing is
                # unreadable whichever way a cell lands.
                deep = abs(scaled.loc[endpoint, name]) > 0.75
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if deep else "0.15")
        if endpoint in chosen:
            ax.add_patch(
                plt.Rectangle((columns[chosen[endpoint]] - 0.5, row - 0.5), 1, 1,
                              fill=False, edgecolor="#1F6F5C", linewidth=2.2)
            )

    ax.set_xlabel("pre-trained checkpoint (max substructure radius, vocabulary size)")
    ax.set_title(
        f"{cfg.ACTIVE.label}: mean validation $R^2$ of each PT-GIN checkpoint\n"
        "shaded by distance from the endpoint's own mean, in units of its "
        "fold-to-fold noise; green box is what was chosen",
        fontsize=11,
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02,
                       ticks=[-1, -0.5, 0, 0.5, 1])
    bar.set_label("standard deviations of one checkpoint across folds", fontsize=9)

    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(cfg.FIGURE_DIR / f"ptgin_selection.{suffix}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("\nwrote figures/ptgin_selection.png")


def main() -> None:
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    cfg.ensure_dirs()
    table, choice = load()
    wide = by_endpoint(table)
    summary = summarise(table, choice, wide)
    plot(table, wide, summary)


if __name__ == "__main__":
    main()
