#!/usr/bin/env python
"""Step 11: modality contribution and the cost of adding one.

Three figures, from what steps 8 and 9 measured.

  ablation_shap   what removing a modality costs, beside how much of the fused
                  model's attribution that modality carries. The paper's Figure
                  4, and the point of putting them side by side is that they
                  disagree: a modality can be heavily used and barely necessary,
                  which is what redundancy looks like from the inside.
  cost_benefit    modelling time against what the extra modality buys. The
                  paper's Figure 6, and its conclusion is the one most likely to
                  matter in practice.
  gnn_block       the 200-wide learned readout against the 30-wide mean of raw
                  atom features the released extractor produces. Only drawn when
                  step 5 has been run with --paper-gnn-block.

    python 11_contribution_report.py
    ADME_DATASET=biogen python 11_contribution_report.py
"""

import argparse

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config as cfg

FUSION_COLORS = {"early": "#4C72B0", "late": "#DD8452"}
MODALITY_COLORS = {
    "rdkit": "#DD8452", "mol2vec": "#4C72B0",
    "gnn": "#55A868", "smiles": "#C44E52",
}


def save(fig, name: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(cfg.FIGURE_DIR / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png")


def ablation_and_shap(ablation: pd.DataFrame, shap_df: pd.DataFrame) -> None:
    """Drop-one necessity beside grouped SHAP usage."""
    deltas = ["delta_r2", "delta_rmse", "delta_mae"]
    titles = [r"$\Delta R^2$", r"$\Delta$RMSE", r"$\Delta$MAE"]

    fig, axes = plt.subplots(2, 4, figsize=(19, 8.4))
    for r, fusion in enumerate(cfg.FUSIONS):
        sub = ablation[ablation["fusion"] == fusion]
        for c, (column, title) in enumerate(zip(deltas, titles)):
            ax = axes[r, c]
            means = (
                sub.groupby(["endpoint", "removed"])[column].mean()
                .reset_index()
            )
            sns.barplot(data=means, x="removed", y=column, order=cfg.MODALITIES,
                        hue="removed", hue_order=cfg.MODALITIES, palette=MODALITY_COLORS,
                        errorbar="sd", ax=ax, legend=False)
            sns.stripplot(data=means, x="removed", y=column, order=cfg.MODALITIES,
                          ax=ax, color="0.2", size=3.5, alpha=0.7)
            ax.axhline(0, color="0.4", linewidth=0.9)
            ax.set_xlabel("")
            ax.set_ylabel(f"{cfg.FUSION_LABELS[fusion]} fusion" if c == 0 else "")
            ax.set_title(title if r == 0 else "", fontsize=11)
            ax.set_xticks(range(len(cfg.MODALITIES)))
            ax.set_xticklabels([cfg.MODALITY_LABELS[m] for m in cfg.MODALITIES],
                               fontsize=9)

        ax = axes[r, 3]
        if len(shap_df) and fusion == "early":
            means = shap_df.groupby(["endpoint", "modality"])["mean_abs_shap"].mean().reset_index()
            sns.barplot(data=means, x="modality", y="mean_abs_shap", order=cfg.MODALITIES,
                        hue="modality", hue_order=cfg.MODALITIES, palette=MODALITY_COLORS,
                        errorbar="sd", ax=ax, legend=False)
            sns.stripplot(data=means, x="modality", y="mean_abs_shap", order=cfg.MODALITIES,
                          ax=ax, color="0.2", size=3.5, alpha=0.7)
            ax.set_title("grouped SHAP, early fusion", fontsize=11)
            ax.set_xticks(range(len(cfg.MODALITIES)))
            ax.set_xticklabels([cfg.MODALITY_LABELS[m] for m in cfg.MODALITIES], fontsize=9)
        else:
            ax.set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("mean |SHAP|, summed over the block")

    fig.suptitle(
        f"{cfg.ACTIVE.label}: removing a modality from the four-modality model "
        "(negative is worse without it), and how much of the fused model's "
        "attribution it carries",
        fontsize=13,
    )
    save(fig, "ablation_shap")


def cost_benefit(timings: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Modelling time against accuracy, as more modalities go in."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    cost = timings[timings["fusion"].isin(cfg.FUSIONS)]
    baseline = timings[
        (timings["fusion"] == "unimodal") & (timings["method"] == cfg.unimodal_method("rdkit", "lgbm"))
    ].set_index("endpoint")["total_seconds"]

    ax = axes[0]
    relative = cost.copy()
    relative["fold_change"] = relative.apply(
        lambda r: r["total_seconds"] / baseline.get(r["endpoint"], np.nan), axis=1
    )
    sns.pointplot(data=relative, x="n_modalities", y="fold_change", hue="fusion",
                  hue_order=cfg.FUSIONS, palette=FUSION_COLORS, errorbar=("ci", 95),
                  ax=ax, dodge=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("modalities")
    ax.set_ylabel("fold change in fitting time\nagainst LightGBM on RDKit alone")
    ax.set_title("cost", fontsize=11)

    ax = axes[1]
    grid = metrics[metrics["method"].isin(cfg.FUSION_METHODS)].copy()
    spec = grid["method"].map(cfg.GRID_SPEC)
    grid["n_modalities"] = spec.map(lambda s: len(s["modalities"]))
    grid["fusion"] = spec.map(lambda s: s["fusion"])
    means = grid.groupby(["endpoint", "n_modalities", "fusion"])["r2"].mean().reset_index()
    sns.pointplot(data=means, x="n_modalities", y="r2", hue="fusion",
                  hue_order=cfg.FUSIONS, palette=FUSION_COLORS, errorbar=("ci", 95),
                  ax=ax, dodge=0.2)
    ax.set_xlabel("modalities")
    ax.set_ylabel(cfg.METRIC_LABELS["r2"])
    ax.set_title("benefit", fontsize=11)

    ax = axes[2]
    joined = (
        relative.groupby(["n_modalities", "fusion"])["fold_change"].median().rename("cost")
        .to_frame()
        .join(means.groupby(["n_modalities", "fusion"])["r2"].mean().rename("r2"))
        .reset_index()
    )
    for fusion in cfg.FUSIONS:
        sub = joined[joined["fusion"] == fusion].sort_values("n_modalities")
        ax.plot(sub["cost"], sub["r2"], marker="o", color=FUSION_COLORS[fusion],
                label=cfg.FUSION_LABELS[fusion])
        for _, row in sub.iterrows():
            ax.annotate(int(row["n_modalities"]), (row["cost"], row["r2"]),
                        fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("median fold change in fitting time")
    ax.set_ylabel(cfg.METRIC_LABELS["r2"])
    ax.set_title("what the time buys", fontsize=11)
    ax.legend(frameon=False, fontsize=9)

    for ax in axes[:2]:
        if ax.get_legend():
            ax.legend_.set_title("")
    fig.suptitle(f"{cfg.ACTIVE.label}: the cost of an extra modality against what it buys",
                 fontsize=13)
    save(fig, "cost_benefit")


def gnn_block_control(metrics: pd.DataFrame) -> pd.DataFrame | None:
    """The learned readout against the block the released extractor produces."""
    path = cfg.RESULTS_DIR / "fold_metrics_paper_gnn.csv"
    if not path.exists():
        return None
    paper = pd.read_csv(path)

    ours = metrics[metrics["method"].isin(paper["method"].unique())]
    a = ours.groupby(["endpoint", "method"])["r2"].mean().rename("readout_200d")
    b = paper.groupby(["endpoint", "method"])["r2"].mean().rename("atom_mean_30d")
    table = pd.concat([a, b], axis=1).dropna().reset_index()
    table["difference"] = table["readout_200d"] - table["atom_mean_30d"]
    table["label"] = table["method"].map(cfg.METHOD_LABELS)
    table.to_csv(cfg.TABLE_DIR / "gnn_block_control.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    lo = min(table["readout_200d"].min(), table["atom_mean_30d"].min())
    hi = max(table["readout_200d"].max(), table["atom_mean_30d"].max())
    ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", linewidth=1)
    ax.scatter(table["atom_mean_30d"], table["readout_200d"], s=34, alpha=0.8,
               color="#4C72B0")
    ax.set_xlabel(r"$R^2$, 30-wide mean of raw atom features (as released)")
    ax.set_ylabel(r"$R^2$, 200-wide learned readout")
    ax.set_title(
        f"{cfg.ACTIVE.label}: the GNN modality, mean {table['difference'].mean():+.3f} "
        r"$R^2$ when it is actually a learned representation",
        fontsize=10,
    )
    save(fig, "gnn_block_control")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    cfg.ensure_dirs()
    sns.set_style("whitegrid")

    metrics_path = cfg.FOLD_METRICS_CSV
    if not metrics_path.exists():
        raise SystemExit(f"{metrics_path} not found -- run 06_collect_metrics.py first")
    metrics = pd.read_csv(metrics_path)

    ablation_path = cfg.RESULTS_DIR / "modality_ablation.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        shap_df = pd.read_csv(cfg.SHAP_CSV) if cfg.SHAP_CSV.exists() else pd.DataFrame()
        print("\nmodality contribution")
        ablation_and_shap(ablation, shap_df)

        order = (
            ablation.groupby(["fusion", "removed"])["delta_r2"].mean().unstack()
            .reindex(columns=cfg.MODALITIES)
        )
        print("\nmean change in R squared on removing a modality:")
        print(order.round(4).to_string())
        if len(shap_df):
            share = shap_df.groupby("modality")["mean_abs_shap"].mean().reindex(cfg.MODALITIES)
            print("\ngrouped SHAP share, early fusion:")
            for modality, value in share.items():
                print(f"  {cfg.MODALITY_LABELS[modality]:<10} {value / share.sum():6.1%}")
    else:
        print("modality contribution: run 08_modality_contribution.py first")

    if cfg.TIMING_CSV.exists():
        print("\ncost and benefit")
        cost_benefit(pd.read_csv(cfg.TIMING_CSV), metrics)
    else:
        print("cost and benefit: run 09_timing.py first")

    print("\nGNN block control")
    table = gnn_block_control(metrics)
    if table is None:
        print("  not run -- rerun step 5 with --paper-gnn-block to compare")
    else:
        print(f"  mean {table['difference'].mean():+.4f} R squared for the learned readout")


if __name__ == "__main__":
    main()
