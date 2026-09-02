#!/usr/bin/env python
"""Step 8: what each modality contributes, by ablation and by grouped SHAP.

The paper asks the question twice, because the two answers mean different things.
Ablation removes a modality, refits, and measures what the model loses: how
*necessary* the modality is. Grouped SHAP leaves the model whole and asks how
much of its attribution the modality's columns carry: how much it is *used*. A
modality that duplicates another will look unnecessary and still be used.

Both run on the four-modality model with a LightGBM final learner, under early
and late fusion, which is the configuration the paper's Section 3.3 uses. Both
run on all 25 folds rather than once, so the answer arrives as a distribution.

Drop-one models reuse the four-modality model's tuned hyperparameters rather than
being retuned, which is what `modality_contribution.py` does. It biases the
comparison against the smaller feature blocks and the report says so.

    python 08_modality_contribution.py --jobs 24
    ADME_DATASET=biogen python 08_modality_contribution.py --jobs 24
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config as cfg
import fusion
from folds import fold_masks

ABLATION_FUSIONS = ["early", "late"]
ABLATION_LEARNER = "lgbm"
SHAP_SAMPLE = 1000


def score(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).statistic,
    }


def fit_combination(blocks, modalities, fusion_kind, params, seed, threads):
    spec = {"modalities": modalities, "fusion": fusion_kind, "learner": ABLATION_LEARNER}
    return fusion.fit_predict(spec, blocks, params, seed, threads)


def grouped_shap(blocks, params, seed, threads, rng) -> dict[str, float]:
    """Mean absolute SHAP summed within each modality's columns.

    Deliberately not divided by the block's width, as in the paper: the number is
    the block's total attribution, so a 300-column Mol2Vec block and a 30-column
    one are not comparable per feature. Read it as how much the model leans on the
    modality, not on any one of its columns.
    """
    import shap

    modalities = cfg.COMBOS[cfg.FULL_COMBO]
    x_fit = blocks.matrix(modalities, blocks.fit)
    x_test = blocks.matrix(modalities, blocks.test)
    model = fusion.make_tabular(ABLATION_LEARNER, params["model"], seed, threads)
    model.fit(x_fit, blocks.y[blocks.fit])

    take = min(SHAP_SAMPLE, len(x_test))
    sample = x_test[rng.choice(len(x_test), size=take, replace=False)]
    values = np.abs(shap.TreeExplainer(model).shap_values(sample)).mean(axis=0)

    slices = blocks.block_slices(modalities)
    return {m: float(values[s].sum()) for m, s in slices.items()}


def run_fold(task):
    endpoint, repeat, fold, threads = task
    global _SHARED
    if "_SHARED" not in globals():
        _SHARED = _Shared()
    shared = _SHARED

    blocks, _ = shared.blocks(endpoint, repeat, fold)
    if blocks is None:
        return []

    seed = cfg.fold_seed(repeat, fold)
    y_test = blocks.y[blocks.test]
    full = cfg.COMBOS[cfg.FULL_COMBO]
    rows = []

    for fusion_kind in ABLATION_FUSIONS:
        params = shared.params(
            endpoint, cfg.fusion_method(cfg.FULL_COMBO, fusion_kind, ABLATION_LEARNER)
        )
        if params["model"] is None:
            continue

        baseline = score(y_test, fit_combination(
            blocks, full, fusion_kind, params, seed, threads))
        for modality in full:
            kept = [m for m in full if m != modality]
            reduced = score(y_test, fit_combination(
                blocks, kept, fusion_kind, params, seed, threads))
            rows.append({
                "endpoint": endpoint, "repeat": repeat, "fold": fold,
                "fusion": fusion_kind, "removed": modality,
                **{f"delta_{k}": reduced[k] - baseline[k] for k in baseline},
                **{f"full_{k}": v for k, v in baseline.items()},
            })

    early_params = shared.params(
        endpoint, cfg.fusion_method(cfg.FULL_COMBO, "early", ABLATION_LEARNER)
    )
    shap_rows = []
    if early_params["model"] is not None:
        rng = np.random.default_rng(seed)
        for modality, value in grouped_shap(blocks, early_params, seed, threads, rng).items():
            shap_rows.append({
                "endpoint": endpoint, "repeat": repeat, "fold": fold,
                "fusion": "early", "modality": modality, "mean_abs_shap": value,
            })

    return [("ablation", r) for r in rows] + [("shap", r) for r in shap_rows]


class _Shared:
    """Same job as `Shared` in step 5, kept local so this step imports cleanly."""

    def __init__(self):
        import json

        self.df = pd.read_csv(cfg.MASTER_CSV)
        self.folds = pd.read_csv(cfg.FOLD_CSV)
        self.rdkit = np.load(cfg.RDKIT_NPZ, allow_pickle=True)["x"]
        self.mol2vec = np.load(cfg.MOL2VEC_NPZ)["x"]
        # Ablation and SHAP are LightGBM only, so no graphs are ever needed.
        self.graphs = None
        self.hparams = json.loads(cfg.HPARAM_JSON.read_text())

    def blocks(self, endpoint, repeat, fold):
        path = cfg.fold_embeddings(endpoint, repeat, fold)
        if not path.exists():
            return None, None
        masks = fold_masks(self.df, self.folds, endpoint, repeat, fold)
        return fusion.FoldBlocks(
            np.load(path), masks, self.rdkit, self.mol2vec,
            self.df[endpoint].to_numpy(dtype=float), self.graphs,
        ), masks

    def params(self, endpoint, method):
        tuned = self.hparams.get(endpoint, {})
        spec = cfg.GRID_SPEC[method]
        out = {"model": tuned.get(method), "base": {}}
        if spec["fusion"] == "late":
            for modality in ("rdkit", "mol2vec"):
                if modality in spec["modalities"]:
                    out["base"][modality] = tuned.get(cfg.unimodal_method(modality, "lgbm"))
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    cfg.ensure_dirs()
    tasks = [
        (e, r, f, args.threads)
        for e in args.endpoint for r in args.repeat for f in args.fold
    ]
    print(f"{cfg.ACTIVE.label}: ablation and grouped SHAP over {len(tasks):,} folds",
          flush=True)

    start = time.time()
    ablation, shap_rows = [], []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_fold, t) for t in tasks]
            for i, future in enumerate(as_completed(futures), 1):
                for kind, row in future.result():
                    (ablation if kind == "ablation" else shap_rows).append(row)
                if i % 25 == 0:
                    print(f"  {i:>4}/{len(tasks)} folds  "
                          f"({time.time() - start:.0f}s)", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            for kind, row in run_fold(task):
                (ablation if kind == "ablation" else shap_rows).append(row)
            if i % 25 == 0:
                print(f"  {i:>4}/{len(tasks)} folds", flush=True)

    ablation_df = pd.DataFrame(ablation)
    ablation_path = cfg.RESULTS_DIR / "modality_ablation.csv"
    ablation_df.to_csv(ablation_path, index=False)
    print(f"wrote {ablation_path.name} ({len(ablation_df):,} rows)")

    shap_df = pd.DataFrame(shap_rows)
    shap_df.to_csv(cfg.SHAP_CSV, index=False)
    print(f"wrote {cfg.SHAP_CSV.name} ({len(shap_df):,} rows)")

    if len(ablation_df):
        print("\nmean change on removing a modality from the four-modality model:")
        pivot = ablation_df.pivot_table(
            index="removed", columns="fusion", values="delta_r2", aggfunc="mean"
        ).reindex(cfg.MODALITIES)
        print(pivot.round(4).to_string())
        print("\n(negative means the model got worse without it)")

    if len(shap_df):
        print("\nmean absolute SHAP by modality, early fusion, four modalities:")
        share = shap_df.groupby("modality")["mean_abs_shap"].mean().reindex(cfg.MODALITIES)
        for modality, value in share.items():
            print(f"  {cfg.MODALITY_LABELS[modality]:<10} {value:8.4f}  "
                  f"{value / share.sum():6.1%}")


if __name__ == "__main__":
    main()
