#!/usr/bin/env python
"""Step 4b: what did the 10-checkpoint QuADMET-Former selection choose, and did it matter?

The paper releases ten pre-trained TorchMD-ET checkpoints -- five QM target
combinations (`[Lw]`, `[3D]`, `[DM,HL,TE]`, `[Lw,DM,HL,TE]`, `[All]`) crossed
with single-conformer and multi-conformer pre-training -- and argues that
different downstream ADMET endpoints benefit from different pre-training target
combinations.

`03_run_tabpfn.py --select` scores all ten checkpoints on the held-out
validation fifth of every fold (`10 x endpoints x 25` fits, never touching the
test set). This script summarises that validation table and renders the
row-centred, fold-noise-scaled heatmap:

  tables/selection_by_endpoint.csv   mean validation R^2, every checkpoint x endpoint
  tables/selection_summary.csv       the winner, its margin, and the grid's spread
  figures/quadmet_selection.png      heatmap in units of fold-to-fold noise

    python 04b_quadmet_selection.py
    ADME_DATASET=biogen python 04b_quadmet_selection.py
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
            f"{cfg.SELECTION_CSV} not found -- run `03_run_tabpfn.py --select` first"
        )
    table = pd.read_csv(cfg.SELECTION_CSV)
    choice = pd.read_csv(cfg.CHOICE_CSV)
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
    rows = []
    for _, row in choice.iterrows():
        endpoint = row["endpoint"]
        scores = wide.loc[endpoint].dropna()
        winner_folds = table[
            (table["endpoint"] == endpoint) & (table["checkpoint"] == row["checkpoint"])
        ]["val_r2"]
        single_best = scores[[c for c in cfg.CHECKPOINTS if cfg.CKPT_CONF[c] == "Single"]].max()
        multi_best = scores[[c for c in cfg.CHECKPOINTS if cfg.CKPT_CONF[c] == "Multi"]].max()
        rows.append(
            {
                "endpoint": endpoint,
                "checkpoint": row["checkpoint"],
                "conf_mode": row["conf_mode"],
                "targets": row["targets"],
                "val_r2": scores.max(),
                "margin_over_runner_up": row["margin"],
                "spread_best_to_worst": scores.max() - scores.min(),
                "fold_sd": float(winner_folds.std()),
                "margin_in_fold_sd": row["margin"] / float(winner_folds.std()),
                "single_minus_multi": float(single_best - multi_best),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(cfg.TABLE_DIR / "selection_summary.csv", index=False)
    print("wrote tables/selection_summary.csv\n")

    print(f"{'endpoint':<17} {'chosen':<22} {'val R2':>7} {'margin':>8} "
          f"{'spread':>8} {'fold sd':>8}")
    for _, row in summary.iterrows():
        print(f"{row['endpoint']:<17} {cfg.checkpoint_label(row['checkpoint']):<22} "
              f"{row['val_r2']:7.3f} {row['margin_over_runner_up']:8.3f} "
              f"{row['spread_best_to_worst']:8.3f} {row['fold_sd']:8.3f}")

    counts = summary["checkpoint"].value_counts()
    print(f"\n{len(counts)} distinct checkpoints chosen across {len(summary)} endpoints:")
    for name, n in counts.items():
        print(f"  {cfg.checkpoint_label(name):<24} {n}")
    print(f"\nmedian margin over the runner-up: {summary['margin_over_runner_up'].median():.3f} "
          f"R^2, {summary['margin_in_fold_sd'].median():.2f} fold standard deviations")
    print(f"median spread best to worst:      {summary['spread_best_to_worst'].median():.3f} R^2")
    return summary


def fold_noise(table: pd.DataFrame) -> pd.Series:
    return (
        table.groupby(["endpoint", "checkpoint"])["val_r2"].std()
        .groupby("endpoint").mean()
    )


def plot(table: pd.DataFrame, wide: pd.DataFrame, summary: pd.DataFrame) -> None:
    noise = fold_noise(table).reindex(wide.index)
    centred = wide.sub(wide.mean(axis=1), axis=0)
    scaled = centred.div(noise, axis=0)

    fig, ax = plt.subplots(figsize=(1.08 * len(wide.columns) + 3.4,
                                    0.48 * len(wide.index) + 2.8))
    image = ax.imshow(scaled.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels([cfg.checkpoint_label(c) for c in wide.columns],
                       rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels([f"{e}\n±{noise[e]:.3f}" for e in wide.index], fontsize=8)

    chosen = dict(zip(summary["endpoint"], summary["checkpoint"]))
    columns = {name: i for i, name in enumerate(wide.columns)}
    for row, endpoint in enumerate(wide.index):
        for col, name in enumerate(wide.columns):
            value = wide.loc[endpoint, name]
            if np.isfinite(value):
                deep = abs(scaled.loc[endpoint, name]) > 0.75
                ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if deep else "0.15")
        if endpoint in chosen:
            ax.add_patch(
                plt.Rectangle((columns[chosen[endpoint]] - 0.5, row - 0.5), 1, 1,
                              fill=False, edgecolor="#1F6F5C", linewidth=2.2)
            )

    ax.axvline(4.5, color="0.35", lw=1.2, ls="--")
    ax.set_xlabel("pre-trained checkpoint (five QM target sets: Single-conf left, Multi-conf right)")
    ax.set_title(
        f"{cfg.ACTIVE.label}: mean validation $R^2$ across all 10 released QuADMET-Former checkpoints\n"
        "shaded by distance from the endpoint's own mean, in units of its "
        "fold-to-fold noise; green box is what was chosen",
        fontsize=11,
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02,
                       ticks=[-1, -0.5, 0, 0.5, 1])
    bar.set_label("standard deviations of one checkpoint across folds", fontsize=9)

    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(cfg.FIGURE_DIR / f"quadmet_selection.{suffix}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print("\nwrote figures/quadmet_selection.png")


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
