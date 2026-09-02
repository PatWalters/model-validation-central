#!/usr/bin/env python
"""Render one metric's table as a standalone image.

Mean value over the 25 folds, one row per endpoint, white background and no
standard deviations -- meant to drop into a post or a slide.

Bold marks every method the Tukey correction cannot separate from the leader,
which on some endpoints is more than one. Bolding the single highest mean would
imply a winner the statistics do not support, so where two methods are tied,
either both are bold or neither is.

    python 07_spearman_table_image.py                 # Spearman
    python 07_spearman_table_image.py --metric mae    # MAE
    python 07_spearman_table_image.py --metric r2 --all
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

import config as cfg

# The heading and the file stem for each metric. Spearman keeps its original file
# name so anything already pointing at spearman_table.png still resolves.
TITLES = {"spearman": "Spearman \u03c1", "r2": "R\u00b2", "mae": "MAE"}
SUBTITLE = "mean over 25 folds"

# Layout, in axes coordinates.
LABEL_X = 0.005
COL_LEFT = 0.34
ROW_H = 0.062
HEADER_Y = 0.955


def render(metric: str) -> None:
    summary = pd.read_csv(cfg.TABLE_DIR / "summary.csv")
    sub = summary[summary["metric"] == metric]
    wide = (
        sub.pivot(index="endpoint", columns="method", values="mean")
        .reindex(index=cfg.TARGET_COLS, columns=cfg.METHODS)
    )
    # Every method the Tukey correction cannot separate from the leader, not just
    # the leading mean. Where two are tied, both are bold or neither is.
    on_top = (
        sub[sub["tukey_group"].isin(("best", "equivalent"))]
        .groupby("endpoint")["method"].apply(set)
    )
    methods = [m for m in cfg.METHODS if m in wide.columns and wide[m].notna().any()]
    wide = wide[methods]

    fig, ax = plt.subplots(figsize=(9.4, 4.6), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    col_w = (1.0 - COL_LEFT) / len(methods)
    col_x = [COL_LEFT + col_w * (i + 0.5) for i in range(len(methods))]

    ax.text(LABEL_X, HEADER_Y + 0.055, TITLES[metric], fontsize=13, fontweight="600",
            va="bottom", ha="left")
    ax.text(0.999, HEADER_Y + 0.058, SUBTITLE, fontsize=9, color="#5C6B68",
            va="bottom", ha="right")

    # Column headings, wrapped onto two lines so they stay readable at this width.
    for x, m in zip(col_x, methods):
        label = cfg.METHOD_LABELS[m].replace(" + ", "\n+ ").replace(" single-task", "\nsingle-task")
        label = label.replace(" multi-task", "\nmulti-task")
        ax.text(x, HEADER_Y, label, fontsize=9.5, ha="center", va="top", linespacing=1.35)

    rule_y = HEADER_Y - 0.075
    ax.plot([LABEL_X, 0.999], [rule_y, rule_y], color="#10171A", lw=1.1)

    for i, endpoint in enumerate(cfg.TARGET_COLS):
        if endpoint not in wide.index:
            continue
        y = rule_y - 0.035 - i * ROW_H
        ax.text(LABEL_X, y, endpoint, fontsize=10, ha="left", va="center", family="monospace")
        row = wide.loc[endpoint]
        best = on_top.get(endpoint, set())
        for x, m in zip(col_x, methods):
            ax.text(
                x, y, f"{row[m]:.3f}", fontsize=10.5, ha="center", va="center",
                family="monospace",
                fontweight="bold" if m in best else "normal",
                color="#10171A" if m in best else "#394743",
            )
        ax.plot([LABEL_X, 0.999], [y - ROW_H / 2, y - ROW_H / 2], color="#DDE4E1", lw=0.6)

    # Crop the axes to what was actually drawn, so the image has no dead band
    # below the last row.
    last_line = rule_y - 0.035 - (len(cfg.TARGET_COLS) - 1) * ROW_H - ROW_H / 2
    ax.set_ylim(last_line - 0.02, 1.06)

    out = cfg.FIGURE_DIR / f"{metric}_table.png"
    fig.savefig(out, facecolor="white", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="spearman", choices=cfg.METRICS)
    parser.add_argument("--all", action="store_true", help="render every metric")
    opts = parser.parse_args()
    for metric in cfg.METRICS if opts.all else [opts.metric]:
        render(metric)


if __name__ == "__main__":
    main()
