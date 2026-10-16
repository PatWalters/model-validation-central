#!/usr/bin/env python
"""Step 4: the paper's own arm -- QuADMET-Former fine-tuned (`lpft`).

For each of the 25 folds, trains a multitask 6-layer TorchMD-ET model across
all endpoints of the data set (matching the multitask fine-tuning protocol of
Atom-JEPA and ChemProp + CheMeleon on these datasets) in two stages (`LP-FT`):

  Stage 1 (`linear_probe`)
    The pre-trained 6-layer TorchMD-ET encoder (`shared_backbone`, initialised
    from `multi_all`) is frozen, and the shared-trunk + per-task MLP readout
    (`MultiTaskHead`, matching the 3-layer SiLU MLP depth of `f_predict`) is
    trained on the cached 3-conformer embeddings with AdamW (`lr=1e-3`,
    `weight_decay=1e-4`) and Kendall uncertainty-weighted masked Huber loss
    (`MTLLoss`) on train-standardised targets, restoring the epoch with the
    lowest macro validation MAE on the held-out fifth (`fold_of == fold`).

  Stage 2 (`lpft`)
    Initialised from the Stage-1 checkpoint, all 6 layers of `shared_backbone`
    are unfrozen (`lr=3e-4`) alongside `out_norm` and `MultiTaskHead` (`lr=1e-3`)
    with L2-SP regularisation (`1e-4`) towards the pre-trained backbone weights,
    cycling over the 3 conformers per molecule during training and restoring the
    best macro validation-MAE weights before predicting the held-out `ds == 'test'`
    molecules (averaged over the 3 conformers).

Also supports `--scratch` to train the identical 6-layer TorchMD-ET +
`MultiTaskHead` architecture from random initialisation on the same folds,
recording the results in `results/<ds>/quadmet_ablations.csv`
(`variant == 'quadmet_scratch'`) to test the paper's headline claim that QM
pre-training outperforms training from scratch.

    python 04_run_finetune.py
    ADME_DATASET=biogen python 04_run_finetune.py
    python 04_run_finetune.py --scratch --repeat 0
    ADME_DATASET=biogen python 04_run_finetune.py --scratch --repeat 0
"""

import argparse
import json
import pickle
import socket
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg
from torchmd_backbone import (
    BatchedGraph,
    build_mol_graph,
    collate_graphs,
    forward_precomputed,
    load_torchmd_et,
)

MULTITASK_CKPT = "multi_all"
N_CONFS = 3


class MultiTaskHead(nn.Module):
    """Shared SiLU trunk + per-endpoint 2-layer SiLU MLP heads + Kendall MTLLoss.

    Each endpoint path (`trunk -> head[j]`) is a 3-layer SiLU MLP
    (`128 -> 128 -> 128 -> 1`) matching `TorchMD_ET.f_predict`, while sharing the
    first hidden layer across endpoints.
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_tasks: int, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_tasks)
        ])
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return torch.cat([head(h) for head in self.heads], dim=1)

    def mtl_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(target)
        safe_t = torch.where(mask, target, torch.zeros_like(target))
        elem = F.smooth_l1_loss(pred, safe_t, reduction="none", beta=0.5) * mask
        counts = mask.sum(dim=0)
        present = counts > 0
        task_losses = elem.sum(dim=0) / counts.clamp_min(1)
        prec = 0.5 * torch.exp(-2.0 * self.log_sigma)
        terms = prec * task_losses + self.log_sigma
        return terms[present].sum()


def claimed_elsewhere() -> set[tuple[int, int]]:
    """(repeat, fold) pairs another machine is running; re-read before every fold."""
    path = cfg.FOLD_DIR / "claimed_folds.txt"
    if not path.exists():
        return set()
    pairs = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            r, f = line.split()
            pairs.add((int(r), int(f)))
    return pairs


@torch.no_grad()
def predict_batches(model: nn.Module, batches: list[BatchedGraph]) -> np.ndarray:
    model.eval()
    out = []
    use_amp = len(batches) > 0 and batches[0].z.is_cuda
    for b in batches:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            p, _ = forward_precomputed(model, b)
        if p.dim() == 1:
            p = p.unsqueeze(-1)
        out.append(p.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def predict_3conf(model: nn.Module, batches_3: list[list[BatchedGraph]]) -> np.ndarray:
    total = None
    for c in range(len(batches_3)):
        pc = predict_batches(model, batches_3[c])
        total = pc if total is None else (total + pc)
    return total / float(len(batches_3))


def macro_val_mae(vp: np.ndarray, yv: np.ndarray) -> float:
    maes = []
    for j in range(yv.shape[1]):
        mj = np.isfinite(yv[:, j])
        if mj.any():
            maes.append(float(np.abs(vp[mj, j] - yv[mj, j]).mean()))
    return float(np.mean(maes))


def train_stage1_lp(model: nn.Module, X_fit: np.ndarray, Y_fit: np.ndarray,
                    X_val: np.ndarray, Y_val: np.ndarray, seed: int,
                    device: torch.device) -> tuple[float, int]:
    """Train `model.f_predict` (`MultiTaskHead`) on cached frozen 3-conf embeddings."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.f_predict.parameters(), lr=cfg.LR_HEAD,
                            weight_decay=cfg.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    xf = torch.as_tensor(X_fit, dtype=torch.float32, device=device)
    yf = torch.as_tensor(Y_fit, dtype=torch.float32, device=device)
    xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)

    n = xf.size(0)
    bs = cfg.BATCH_SIZE * cfg.ACCUMULATE_GRAD
    gen = torch.Generator(device=device).manual_seed(seed)

    best_val_mae = float("inf")
    best_epoch = 0
    best_state = {k: v.detach().clone() for k, v in model.f_predict.state_dict().items()}

    for epoch in range(1, 61):
        model.f_predict.train()
        perm = torch.randperm(n, generator=gen, device=device)
        for start in range(0, n, bs):
            idx = perm[start:start + bs]
            opt.zero_grad(set_to_none=True)
            pred = model.f_predict(xf[idx])
            loss = model.f_predict.mtl_loss(pred, yf[idx])
            loss.backward()
            opt.step()

        model.f_predict.eval()
        with torch.no_grad():
            vp = model.f_predict(xv).detach().cpu().numpy()
        v_mae = macro_val_mae(vp, Y_val)
        sched.step(v_mae)

        if v_mae < best_val_mae - 1e-6:
            best_val_mae = v_mae
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.f_predict.state_dict().items()}

    model.f_predict.load_state_dict(best_state)
    return best_val_mae, best_epoch


def train_end_to_end_multitask(
    model: nn.Module,
    graphs_3: list[list],
    fit_idx: np.ndarray,
    val_batches: list[BatchedGraph],
    Y_norm: np.ndarray,
    val_idx: np.ndarray,
    seed: int,
    device: torch.device,
    is_lpft: bool,
    init_val_mae: float = float("inf"),
    max_epochs: int | None = None,
) -> tuple[float, int]:
    """Full 6-layer multitask TorchMD-ET training (`lpft` or `scratch`)."""
    torch.manual_seed(seed)
    for p in model.shared_backbone.parameters():
        p.requires_grad = True

    default_ep = 20 if cfg.DATASET == "biogen" else 25
    epochs = max_epochs if max_epochs is not None else default_ep

    if is_lpft:
        init_bb = {n: p.detach().clone() for n, p in model.shared_backbone.named_parameters()}
        opt = torch.optim.AdamW([
            {"params": model.shared_backbone.parameters(), "lr": 3e-4, "weight_decay": 0.0},
            {"params": list(model.out_norm.parameters()) + list(model.f_predict.parameters()),
             "lr": cfg.LR_HEAD, "weight_decay": cfg.WEIGHT_DECAY},
        ])
    else:
        init_bb = None
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR_HEAD, weight_decay=cfg.WEIGHT_DECAY)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    eff_bs = cfg.BATCH_SIZE * cfg.ACCUMULATE_GRAD
    gen = torch.Generator().manual_seed(seed)
    use_amp = device.type == "cuda"

    Yv = Y_norm[val_idx]
    vp0 = predict_batches(model, val_batches)
    best_val_mae = min(init_val_mae, macro_val_mae(vp0, Yv))
    best_epoch = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        model.train()
        order = fit_idx[torch.randperm(len(fit_idx), generator=gen).numpy()]
        c_ep = (epoch - 1) % N_CONFS
        for start in range(0, len(order), eff_bs):
            sub_i = order[start:start + eff_bs]
            b = collate_graphs([graphs_3[i][c_ep] for i in sub_i], device)
            yb = torch.as_tensor(Y_norm[sub_i], dtype=torch.float32, device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred, _ = forward_precomputed(model, b)
                if pred.dim() == 1:
                    pred = pred.unsqueeze(-1)
                loss = model.f_predict.mtl_loss(pred, yb)
            if init_bb is not None:
                l2sp = sum((param - init_bb[name]).pow(2).sum()
                           for name, param in model.shared_backbone.named_parameters())
                loss = loss + 1e-4 * l2sp
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        sched.step()
        vp = predict_batches(model, val_batches)
        v_mae = macro_val_mae(vp, Yv)
        if v_mae < best_val_mae - 1e-5:
            best_val_mae = v_mae
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_val_mae, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--scratch", action="store_true",
                        help="train 6-layer TorchMD-ET from random initialisation and record in quadmet_ablations.csv")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override max fine-tuning epochs")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    tasks = cfg.TARGET_COLS

    caches = sorted(cfg.CONFORMER_DIR.glob(f"{cfg.DATASET}_*.pkl"))
    if len(caches) != 1:
        raise SystemExit(f"expected one conformer cache in {cfg.CONFORMER_DIR} -- run 01_conformers.py")
    with open(caches[0], "rb") as fh:
        conformers = pickle.load(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    graphs_3 = [
        [
            build_mol_graph(
                conformers[smi][0],
                conformers[smi][1][min(c, len(conformers[smi][1]) - 1)],
            ).to(device)
            for c in range(N_CONFS)
        ]
        for smi in df[cfg.SMILES_COL]
    ]
    print(f"built {len(graphs_3):,} x {N_CONFS} conformer graphs on {device} in {time.time() - t0:.1f}s",
          flush=True)

    ckpt_name = "scratch" if args.scratch else MULTITASK_CKPT
    ckpt_path = None if args.scratch else cfg.checkpoint_path(ckpt_name)
    X_cached = None
    if not args.scratch:
        npz = np.load(cfg.embedding_npz(ckpt_name), allow_pickle=False)
        X_cached = npz.get("X_conf3", npz["X"])

    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    any_labelled = df[tasks].notna().any(axis=1).to_numpy()
    Y_raw = df[tasks].to_numpy(dtype=np.float32)

    test_mask = is_test & any_labelled
    test_idx = np.where(test_mask)[0]
    eff_bs = cfg.BATCH_SIZE * cfg.ACCUMULATE_GRAD
    test_batches_3 = [
        [
            collate_graphs([graphs_3[i][c] for i in test_idx[s:s + eff_bs]], device)
            for s in range(0, len(test_idx), eff_bs)
        ]
        for c in range(N_CONFS)
    ]
    test_df = df.iloc[test_idx].reset_index(drop=True)

    out_dir = cfg.PRED_DIR / cfg.FT_METHOD
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch_rows = []

    order = [(r, f) for r in args.repeat for f in args.fold]
    for repeat, fold in (reversed(order) if args.reverse else order):
        paths = {t: cfg.pred_csv(cfg.FT_METHOD, t, repeat, fold) for t in tasks}
        if not args.scratch and all(p.exists() for p in paths.values()) and not args.force:
            continue
        if not args.scratch and (repeat, fold) in claimed_elsewhere() and not args.force:
            print(f"=== r{repeat} f{fold}: claimed by another machine, skipped", flush=True)
            continue

        held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
        fold_of = df[cfg.ID_COL].map(held_out).to_numpy()
        fit_mask = ~is_test & any_labelled & (fold_of != fold)
        val_mask = ~is_test & any_labelled & (fold_of == fold)
        fit_idx = np.where(fit_mask)[0]
        val_idx = np.where(val_mask)[0]

        means = np.nanmean(Y_raw[fit_idx], axis=0)
        stds = np.where(np.nanstd(Y_raw[fit_idx], axis=0) > 1e-8,
                        np.nanstd(Y_raw[fit_idx], axis=0), 1.0)
        Y_norm = (Y_raw - means) / stds

        val_batches = [
            collate_graphs([graphs_3[i][0] for i in val_idx[s:s + eff_bs]], device)
            for s in range(0, len(val_idx), eff_bs)
        ]

        seed = cfg.fold_seed(repeat, fold)
        start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model = load_torchmd_et(
            ckpt_path=ckpt_path,
            num_layers=cfg.PRETRAIN_LAYERS,
            dp_rate=cfg.DROPOUT,
            head_hidden_dim=cfg.HEAD_HIDDEN_DIM,
        ).to(device)
        model.f_predict = MultiTaskHead(
            cfg.HIDDEN_CHANNELS, cfg.HEAD_HIDDEN_DIM, len(tasks), dropout=cfg.DROPOUT
        ).to(device)

        if args.scratch:
            val_mae, best_ep = train_end_to_end_multitask(
                model, graphs_3, fit_idx, val_batches, Y_norm, val_idx,
                seed, device, is_lpft=False, max_epochs=args.epochs,
            )
            lp_val_mae, lp_ep = float("nan"), 0
        else:
            lp_val_mae, lp_ep = train_stage1_lp(
                model, X_cached[fit_idx], Y_norm[fit_idx],
                X_cached[val_idx], Y_norm[val_idx], seed, device,
            )
            val_mae, ft_ep = train_end_to_end_multitask(
                model, graphs_3, fit_idx, val_batches, Y_norm, val_idx,
                seed, device, is_lpft=True, init_val_mae=lp_val_mae,
                max_epochs=args.epochs,
            )
            best_ep = lp_ep + ft_ep

        preds_norm = predict_3conf(model, test_batches_3)
        preds = preds_norm * stds + means
        seconds = time.time() - start
        Yt = Y_raw[test_idx]

        r2s, maes = [], []
        for j, t in enumerate(tasks):
            has = np.isfinite(Yt[:, j])
            yt_j = Yt[has, j]
            yp_j = preds[has, j]
            r2_j = float(r2_score(yt_j, yp_j))
            mae_j = float(mean_absolute_error(yt_j, yp_j))
            r2s.append(r2_j)
            maes.append(mae_j)
            if args.scratch:
                scratch_rows.append({
                    "endpoint": t,
                    "variant": "quadmet_scratch",
                    "checkpoint": "scratch",
                    "repeat": repeat,
                    "fold": fold,
                    "r2": r2_j,
                    "spearman": float(spearmanr(yt_j, yp_j).statistic),
                    "mae": mae_j,
                })
            else:
                pd.DataFrame(
                    {
                        "method": cfg.FT_METHOD,
                        "endpoint": t,
                        "repeat": repeat,
                        "fold": fold,
                        cfg.ID_COL: test_df.loc[has, cfg.ID_COL].to_numpy(),
                        cfg.SMILES_COL: test_df.loc[has, cfg.SMILES_COL].to_numpy(),
                        "y_true": yt_j,
                        "y_pred": yp_j,
                    },
                    columns=cfg.PRED_COLUMNS,
                ).to_csv(paths[t], index=False)

        test_macro_mae = float(np.mean(maes))
        mean_r2 = float(np.mean(r2s))
        if args.scratch:
            print(f"[scratch] r{repeat} f{fold}: best ep {best_ep}, val z-MAE {val_mae:.4f}, "
                  f"test macro MAE {test_macro_mae:.4f}, mean R2 {mean_r2:+.3f}, {seconds:.1f}s",
                  flush=True)
        else:
            diag = {
                "endpoint": "all",
                "checkpoint": ckpt_name,
                "repeat": repeat,
                "fold": fold,
                "seed": seed,
                "seconds": seconds,
                "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                "lp_best_epoch": lp_ep,
                "best_epoch": best_ep,
                "lp_val_mae": lp_val_mae,
                "val_macro_mae": val_mae,
                "test_macro_mae": test_macro_mae,
                "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                                if device.type == "cuda" else 0.0),
                "n_train": int(len(fit_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
            }
            cfg.diag_json("all", repeat, fold).write_text(json.dumps(diag, indent=1))
            pd.DataFrame([{
                "method": cfg.FT_METHOD,
                "endpoint": "all",
                "repeat": repeat,
                "fold": fold,
                "seconds": seconds,
            }]).to_csv(cfg.TIMING_CSV, mode="a", header=not cfg.TIMING_CSV.exists(), index=False)
            print(f"[lpft] r{repeat} f{fold} ({ckpt_name}): LP ep {lp_ep} (z-val {lp_val_mae:.4f}) "
                  f"-> FT ep {best_ep} (z-val {val_mae:.4f}), test macro MAE {test_macro_mae:.4f}, "
                  f"mean R2 {mean_r2:+.3f}, {seconds:.1f}s", flush=True)

        del model

    if args.scratch and scratch_rows:
        abl_df = pd.DataFrame(scratch_rows)
        if cfg.ABLATION_CSV.exists():
            old = pd.read_csv(cfg.ABLATION_CSV)
            old = old[~(old["variant"].eq("quadmet_scratch")
                        & old["endpoint"].isin(abl_df["endpoint"])
                        & old["repeat"].isin(abl_df["repeat"])
                        & old["fold"].isin(abl_df["fold"]))]
            abl_df = pd.concat([old, abl_df], ignore_index=True)
        abl_df.to_csv(cfg.ABLATION_CSV, index=False)
        print(f"updated {cfg.ABLATION_CSV.relative_to(cfg.PROJECT_DIR)} with {len(scratch_rows)} scratch rows")


if __name__ == "__main__":
    main()
