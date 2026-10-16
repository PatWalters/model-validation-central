#!/usr/bin/env python
"""Step 3: checkpoint selection (`--select`) and the frozen TabPFN 3 arm.

Two phases:

  --select   For every endpoint, released checkpoint (10 in all: 5 QM target
             sets x single/multi-conformer) and fold, train the paper's Stage-1
             linear-probe MLP head (`f_predict`: 128 -> 128 -> 128 -> 1, SiLU,
             dropout 0.1, L1 loss, AdamW) on the four training fifths and score
             it on the held-out validation fifth (`fold_of == fold`). The
             checkpoint with the best mean validation R^2 over the 25 folds
             becomes that endpoint's QuADMET-Former checkpoint. The test set is
             never read during selection. Also records the Stage-1 linear-probe
             ablations (6-layer vs 3-layer truncation, selected vs fixed `[All]`
             checkpoint, single- vs 3-conformer pooling) in
             `results/<ds>/quadmet_ablations.csv`.

  (default)  Run the frozen QuADMET-Former embedding of each endpoint's chosen
             checkpoint through Monroe's `fit_predict_tabpfn` (TabPFN 3, eight
             estimators, no PCA, `output_type="mean"`). Against `monroe` this
             arm changes only the representation; against `quadmet_ft` it
             changes only how the representation is used.

    python 03_run_tabpfn.py --select
    ADME_DATASET=biogen python 03_run_tabpfn.py --select
    python 03_run_tabpfn.py
    ADME_DATASET=biogen python 03_run_tabpfn.py
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

import config as cfg


def check_tabpfn() -> None:
    import tabpfn
    from tabpfn.settings import settings
    default = settings.tabpfn.model_version.value
    if default != "v3":
        raise SystemExit(
            f"tabpfn {tabpfn.__version__} would load TabPFN {default}, not v3 -- "
            "run this in the `monroe` environment"
        )
    print(f"tabpfn {tabpfn.__version__} loads TabPFN {default}")


def load_checkpoint_npz(df: pd.DataFrame, name: str) -> dict[str, np.ndarray]:
    path = cfg.embedding_npz(name)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 02_embed.py first")
    cached = np.load(path, allow_pickle=False)
    if not np.array_equal(cached["names"], df[cfg.ID_COL].to_numpy().astype(str)):
        raise SystemExit(f"{path.name} does not line up with {cfg.MASTER_CSV.name}")
    return {k: cached[k] for k in cached.files if k != "names"}


def fit_linear_probe_batched(X_fit: np.ndarray, y_fit: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             X_test: np.ndarray, seed: int,
                             device) -> tuple[np.ndarray, np.ndarray, int]:
    """Train the authors' 3-layer SiLU MLP head (`f_predict`) with L1 loss on GPU.

    Since `shared_backbone` has zero dropout and is frozen in Stage 1
    (`linear_probe`), `aggr` (`X`) is static across epochs. Restores the epoch
    with the lowest validation MAE on `X_val`.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    in_dim = X_fit.shape[1]
    h_dim = cfg.HEAD_HIDDEN_DIM
    head = nn.Sequential(
        nn.Linear(in_dim, h_dim),
        nn.SiLU(),
        nn.Dropout(cfg.DROPOUT),
        nn.Linear(h_dim, h_dim),
        nn.SiLU(),
        nn.Dropout(cfg.DROPOUT),
        nn.Linear(h_dim, 1),
    ).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=cfg.LR_HEAD, weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    xf = torch.as_tensor(X_fit, dtype=torch.float32, device=device)
    yf = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    xt = torch.as_tensor(X_test, dtype=torch.float32, device=device)

    n = xf.size(0)
    bs = cfg.BATCH_SIZE * cfg.ACCUMULATE_GRAD
    gen = torch.Generator(device=device).manual_seed(seed)

    best_val_mae = float("inf")
    best_epoch = 0
    best_val_pred = None
    best_test_pred = None
    wait = 0

    for epoch in range(1, cfg.LP_EPOCHS + 1):
        head.train()
        perm = torch.randperm(n, generator=gen, device=device)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            opt.zero_grad(set_to_none=True)
            pred = head(xf[idx]).squeeze(-1)
            loss = torch.nn.functional.l1_loss(pred, yf[idx])
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            v_pred = head(xv).squeeze(-1)
            v_mae = float(torch.nn.functional.l1_loss(v_pred, yv).item())
        sched.step(v_mae)

        if v_mae < best_val_mae - 1e-6:
            best_val_mae = v_mae
            best_epoch = epoch
            best_val_pred = v_pred.detach().cpu().numpy()
            with torch.no_grad():
                best_test_pred = head(xt).squeeze(-1).detach().cpu().numpy()
            wait = 0
        else:
            wait += 1
            if epoch >= cfg.MIN_EPOCHS and wait >= cfg.PATIENCE:
                break

    return best_val_pred, best_test_pred, best_epoch


def run_selection(df: pd.DataFrame, folds: pd.DataFrame) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    caches = {name: load_checkpoint_npz(df, name) for name in cfg.CHECKPOINTS}

    sel_rows = []
    test_preds_by_ckpt = {}
    t0 = time.time()

    for name in cfg.CHECKPOINTS:
        t_ckpt = time.time()
        # Use 3-conformer pooled embedding for Multi checkpoints, single conformer for Single
        X = caches[name].get("X_conf3", caches[name]["X"])
        for endpoint in cfg.TARGET_COLS:
            y = df[endpoint].to_numpy()
            measured = df[endpoint].notna().to_numpy()
            test_mask = measured & is_test
            for repeat in cfg.REPEATS:
                held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
                fold_of = df[cfg.ID_COL].map(held_out).to_numpy()
                for fold in cfg.FOLDS:
                    fit_mask = measured & ~is_test & (fold_of != fold)
                    val_mask = measured & ~is_test & (fold_of == fold)
                    seed = cfg.fold_seed(repeat, fold)
                    v_pred, t_pred, best_ep = fit_linear_probe_batched(
                        X[fit_mask], y[fit_mask], X[val_mask], y[val_mask],
                        X[test_mask], seed, device
                    )
                    sel_rows.append({
                        "endpoint": endpoint,
                        "checkpoint": name,
                        "conf_mode": cfg.CKPT_CONF[name],
                        "targets": cfg.CKPT_TARGETS[name],
                        "repeat": repeat,
                        "fold": fold,
                        "n_fit": int(fit_mask.sum()),
                        "n_val": int(val_mask.sum()),
                        "best_epoch": best_ep,
                        "val_r2": r2_score(y[val_mask], v_pred),
                        "val_mae": mean_absolute_error(y[val_mask], v_pred),
                    })
                    test_preds_by_ckpt[(name, endpoint, repeat, fold)] = (y[test_mask], t_pred)
        print(f"  scored {name:<22} ({cfg.checkpoint_label(name):<18}) in {time.time() - t_ckpt:.1f}s",
              flush=True)

    table = pd.DataFrame(sel_rows)
    table.to_csv(cfg.SELECTION_CSV, index=False)
    print(f"wrote {cfg.SELECTION_CSV.relative_to(cfg.PROJECT_DIR)} ({len(table)} rows) "
          f"in {time.time() - t0:.1f}s")

    # Choose the checkpoint with best mean validation R^2 per endpoint
    mean = (
        table.groupby(["endpoint", "checkpoint"])[["val_r2", "val_mae"]]
        .agg(["mean", "std", "count"])
    )
    mean.columns = [f"{m}_{s}" for m, s in mean.columns]
    mean = mean.reset_index()

    choice_rows = []
    for endpoint in cfg.TARGET_COLS:
        group = mean[mean["endpoint"] == endpoint].sort_values("val_r2_mean", ascending=False).reset_index(drop=True)
        best, runner_up = group.iloc[0], group.iloc[1]
        choice_rows.append({
            "endpoint": endpoint,
            "checkpoint": best["checkpoint"],
            "conf_mode": cfg.CKPT_CONF[best["checkpoint"]],
            "targets": cfg.CKPT_TARGETS[best["checkpoint"]],
            "val_r2": best["val_r2_mean"],
            "val_r2_sd": best["val_r2_std"],
            "val_mae": best["val_mae_mean"],
            "n_folds": int(best["val_r2_count"]),
            "runner_up": runner_up["checkpoint"],
            "runner_up_val_r2": runner_up["val_r2_mean"],
            "margin": best["val_r2_mean"] - runner_up["val_r2_mean"],
            "worst_val_r2": group.iloc[-1]["val_r2_mean"],
        })
    choice = pd.DataFrame(choice_rows)
    choice.to_csv(cfg.CHOICE_CSV, index=False)
    print(f"wrote {cfg.CHOICE_CSV.relative_to(cfg.PROJECT_DIR)}\n")
    for _, r in choice.iterrows():
        print(f"  {r['endpoint']:<17} {cfg.checkpoint_label(r['checkpoint']):<20} "
              f"val R2 {r['val_r2']:.3f}  margin {r['margin']:.3f}  "
              f"spread {r['val_r2'] - r['worst_val_r2']:.3f}")

    # Record Stage-1 Linear Probe ablations on the test set:
    #   1. quadmet_lp (selected checkpoint, 6 layers)
    #   2. quadmet_lp_all (fixed `single_all` checkpoint, 6 layers)
    #   3. quadmet_lp_3layer (selected checkpoint, 3-layer truncated `X_layers[:, 2, :]`)
    chosen_map = dict(zip(choice["endpoint"], choice["checkpoint"]))
    ablation_rows = []
    for endpoint in cfg.TARGET_COLS:
        win_ckpt = chosen_map[endpoint]
        X_3layer = caches[win_ckpt]["X_layers"][:, 2, :]
        y = df[endpoint].to_numpy()
        measured = df[endpoint].notna().to_numpy()
        test_mask = measured & is_test
        for repeat in cfg.REPEATS:
            held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
            fold_of = df[cfg.ID_COL].map(held_out).to_numpy()
            for fold in cfg.FOLDS:
                # 1. Selected 6-layer LP
                yt, yp_sel = test_preds_by_ckpt[(win_ckpt, endpoint, repeat, fold)]
                # 2. Fixed single_all 6-layer LP
                _, yp_all = test_preds_by_ckpt[(cfg.DEFAULT_CHECKPOINT, endpoint, repeat, fold)]
                # 3. Selected 3-layer truncated LP
                fit_mask = measured & ~is_test & (fold_of != fold)
                val_mask = measured & ~is_test & (fold_of == fold)
                seed = cfg.fold_seed(repeat, fold)
                _, yp_3l, _ = fit_linear_probe_batched(
                    X_3layer[fit_mask], y[fit_mask], X_3layer[val_mask], y[val_mask],
                    X_3layer[test_mask], seed, device
                )
                for variant, yp in (
                    ("quadmet_lp", yp_sel),
                    ("quadmet_lp_fixed_all", yp_all),
                    ("quadmet_lp_3layer", yp_3l),
                ):
                    ablation_rows.append({
                        "endpoint": endpoint,
                        "variant": variant,
                        "checkpoint": win_ckpt if variant != "quadmet_lp_fixed_all" else cfg.DEFAULT_CHECKPOINT,
                        "repeat": repeat,
                        "fold": fold,
                        "r2": r2_score(yt, yp),
                        "spearman": spearmanr(yt, yp).statistic,
                        "mae": mean_absolute_error(yt, yp),
                    })

    abl_df = pd.DataFrame(ablation_rows)
    if cfg.ABLATION_CSV.exists():
        old = pd.read_csv(cfg.ABLATION_CSV)
        old = old[~old["variant"].isin({"quadmet_lp", "quadmet_lp_fixed_all", "quadmet_lp_3layer"})]
        abl_df = pd.concat([old, abl_df], ignore_index=True)
    abl_df.to_csv(cfg.ABLATION_CSV, index=False)
    print(f"wrote Stage-1 LP ablations to {cfg.ABLATION_CSV.relative_to(cfg.PROJECT_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--select", action="store_true",
                        help="score all 10 checkpoints on validation fifths and stop")
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)

    if args.select:
        run_selection(df, folds)
        return

    if not cfg.CHOICE_CSV.exists():
        raise SystemExit(f"{cfg.CHOICE_CSV} not found -- run `03_run_tabpfn.py --select` first")
    choice = pd.read_csv(cfg.CHOICE_CSV)
    chosen_map = dict(zip(choice["endpoint"], choice["checkpoint"]))

    check_tabpfn()
    sys.path.insert(0, str(cfg.MONROE_HOME))
    from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn

    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    out_dir = cfg.PRED_DIR / cfg.TABPFN_METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = []

    for endpoint in args.endpoint:
        ckpt_name = chosen_map[endpoint]
        cached = load_checkpoint_npz(df, ckpt_name)
        X = cached.get("X_conf3", cached["X"])
        y = df[endpoint].to_numpy()
        measured = df[endpoint].notna().to_numpy()
        for repeat in args.repeat:
            held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
            fold_of = df[cfg.ID_COL].map(held_out).to_numpy()
            for fold in args.fold:
                out_path = cfg.pred_csv(cfg.TABPFN_METHOD, endpoint, repeat, fold)
                if out_path.exists() and not args.force:
                    continue
                fit_mask = measured & ~is_test & (fold_of != fold)
                test_mask = measured & is_test
                start = time.time()
                pred = fit_predict_tabpfn(
                    X[fit_mask], y[fit_mask], X[test_mask],
                    is_classification=False,
                    ensemble_specs=default_ensemble_specs(),
                    seed=cfg.fold_seed(repeat, fold),
                    output_type="mean",
                )
                seconds = time.time() - start
                test_df = df.loc[test_mask]
                pd.DataFrame(
                    {
                        "method": cfg.TABPFN_METHOD,
                        "endpoint": endpoint,
                        "repeat": repeat,
                        "fold": fold,
                        cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(),
                        cfg.SMILES_COL: test_df[cfg.SMILES_COL].to_numpy(),
                        "y_true": y[test_mask],
                        "y_pred": np.asarray(pred, dtype=float),
                    },
                    columns=cfg.PRED_COLUMNS,
                ).to_csv(out_path, index=False)
                timing.append({
                    "method": cfg.TABPFN_METHOD,
                    "endpoint": endpoint,
                    "repeat": repeat,
                    "fold": fold,
                    "seconds": seconds,
                })
                print(f"{endpoint} ({ckpt_name}) r{repeat} f{fold}: "
                      f"{int(fit_mask.sum())} fit, {int(test_mask.sum())} test, {seconds:.1f}s",
                      flush=True)

    if timing:
        t = pd.DataFrame(timing)
        t.to_csv(cfg.TIMING_CSV, mode="a", header=not cfg.TIMING_CSV.exists(), index=False)


if __name__ == "__main__":
    main()
