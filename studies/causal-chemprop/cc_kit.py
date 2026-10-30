"""The authors' Causal-Chemprop, pointed at our molecules instead of their files.

`causal_utils.train` and `causal_utils.test_counterfactual` in the release read
their inputs from hard-coded per-task paths (TRAINING_FPATH[i], TEST_FPATH[i],
...) and their hyperparameters from per-task lists. Nothing else in them is
task-specific. The two functions here are those two, with the paths replaced by
data frames and the per-task lists by the fixed values in config.py. The model
class is the authors' own `SobolevMPNN` from causal_chemprop.py, imported, not
copied, and so are DAGMA, the vendored dowhy GCM and Chemprop 2.0.0.

What differs from `train`, and why:

  validation split   theirs is a random 80:20 of the training file; here it is
                     the protocol's held-out fifth, the same molecules every
                     other early-stopped arm in the repository stops on
  accelerator        GPU for the Chemprop fit (theirs runs on an M2 CPU), CPU
                     for the SCM, one thread per process, many processes
  determinism        torch.use_deterministic_algorithms is not set: several
                     CUDA kernels have no deterministic version. Seeds are set.

What differs from `test_counterfactual`: nothing but where the anchor and the
test molecules come from, and that it returns the arrays instead of plotting.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from rdkit import RDLogger

# The authors' fingerprint calls are the deprecated API; one warning per molecule.
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(cfg.UPSTREAM_HOME))


def upstream():
    """The authors' modules. Imported late: gcm and dagma are slow to load."""
    from chemprop import data as cdata
    from chemprop import featurizers, nn
    from chemprop.nn.transforms import UnscaleTransform
    from causal_chemprop import SobolevMPNN, CustomMSEMetric
    return cdata, featurizers, nn, UnscaleTransform, SobolevMPNN, CustomMSEMetric


class Scaler:
    """sklearn's StandardScaler on one column, as train() fits it, kept as two floats."""

    def __init__(self, mean: float, scale: float):
        self.mean, self.scale = float(mean), float(scale)

    @classmethod
    def fit(cls, y: np.ndarray) -> "Scaler":
        from sklearn.preprocessing import StandardScaler
        s = StandardScaler().fit(np.asarray(y, dtype=float).reshape(-1, 1))
        return cls(s.mean_[0], s.scale_[0])

    def transform(self, y) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.mean) / self.scale

    def sklearn(self):
        from sklearn.preprocessing import StandardScaler
        s = StandardScaler()
        s.mean_, s.scale_, s.var_ = np.array([self.mean]), np.array([self.scale]), np.array([self.scale ** 2])
        s.n_features_in_ = 1
        return s


def datapoints(smiles, targets=None):
    cdata, *_ = upstream()
    if targets is None:
        return [cdata.MoleculeDatapoint.from_smi(s, None, x_d=None) for s in smiles]
    return [cdata.MoleculeDatapoint.from_smi(s, [t]) for s, t in zip(smiles, targets)]


def loader(points, batch_size: int, shuffle: bool, seed: int):
    import torch
    cdata, featurizers, *_ = upstream()
    ds = cdata.MoleculeDataset(points, featurizers.SimpleMoleculeMolGraphFeaturizer())
    g = torch.Generator()
    g.manual_seed(seed)
    return cdata.build_dataloader(ds, batch_size=batch_size, num_workers=0, shuffle=shuffle,
                                  seed=seed, generator=g)


def build_model(scaler: Scaler, save_dir: Path):
    """causal_utils.train's architecture, unchanged."""
    _, _, nn, UnscaleTransform, SobolevMPNN, CustomMSEMetric = upstream()
    mcmp = nn.BondMessagePassing(depth=cfg.CC_DEPTH, d_h=cfg.CC_MPNN_HIDDEN, dropout=cfg.CC_DROPOUT)
    agg = nn.MeanAggregation()
    ffn = nn.RegressionFFN(
        input_dim=mcmp.output_dim, hidden_dim=cfg.CC_FFN_HIDDEN, dropout=cfg.CC_DROPOUT,
        n_layers=cfg.CC_FFN_LAYERS, criterion=CustomMSEMetric(),
        output_transform=UnscaleTransform.from_standard_scaler(scaler.sklearn()),
    )
    init_lr, max_lr, final_lr = cfg.CC_LR
    return SobolevMPNN(cfg.CC_FFN_HIDDEN + 1, save_dir, mcmp, agg, ffn, batch_norm=True,
                       metrics=[CustomMSEMetric()], init_lr=init_lr, max_lr=max_lr,
                       final_lr=final_lr)


def train_encoder(fit: pd.DataFrame, val: pd.DataFrame, target: str, out_dir: Path, seed: int):
    """Fit the authors' Chemprop. Returns (best checkpoint path, scaler, best epoch)."""
    from lightning import pytorch as pl
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping

    seed_everything(seed, workers=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler = Scaler.fit(fit[target])
    train_loader = loader(datapoints(fit[cfg.SMILES_COL], scaler.transform(fit[target])),
                          cfg.CC_BATCH, True, seed)
    val_loader = loader(datapoints(val[cfg.SMILES_COL], scaler.transform(val[target])),
                        32, False, seed)
    model = build_model(scaler, out_dir)
    ckpt = ModelCheckpoint(monitor="val/logs_loss", dirpath=out_dir, save_top_k=1, mode="min")
    trainer = pl.Trainer(
        accelerator="gpu", devices=1, max_epochs=cfg.CC_MAX_EPOCHS, logger=False,
        enable_checkpointing=True, check_val_every_n_epoch=1, num_sanity_val_steps=0,
        callbacks=[EarlyStopping(monitor="val/logs_loss", mode="min", patience=cfg.CC_PATIENCE),
                   ckpt],
        inference_mode=False, enable_progress_bar=False, enable_model_summary=False,
    )
    trainer.fit(model, train_loader, val_loader)
    best = Path(ckpt.best_model_path)
    epoch = int(best.stem.split("-")[0].split("=")[1])
    return best, scaler, epoch


def load_encoder(ckpt: Path, save_dir: Path):
    """The checkpoint on CPU, in SobolevMPNN's training mode (save_dir is a directory)."""
    import torch
    *_, SobolevMPNN, _ = upstream()
    save_dir.mkdir(parents=True, exist_ok=True)
    # chemprop's load_from_checkpoint reads the hyperparameters with a bare
    # torch.load, no map_location, so a checkpoint written on the GPU cannot be
    # opened where CUDA is hidden. A CPU copy, written once, can.
    cpu = ckpt.with_suffix(".cpu.ckpt")
    if not cpu.exists():
        tmp = cpu.with_suffix(f".tmp{os.getpid()}")
        torch.save(torch.load(ckpt, map_location="cpu", weights_only=False), tmp)
        os.replace(tmp, cpu)
    return SobolevMPNN.load_from_checkpoint(cpu, causal_pkl=save_dir, map_location="cpu")


def predict_regular(model, smiles) -> np.ndarray:
    """test_regular's inference: predictor(fingerprint), unscaled."""
    from lightning.pytorch import Trainer
    model.inference = "regular"
    trainer = Trainer(accelerator="cpu", logger=False, enable_progress_bar=False,
                      enable_model_summary=False)
    out = trainer.predict(model, loader(datapoints(smiles), 256, False, 0))
    return np.concatenate([np.asarray(o, dtype=float).reshape(-1) for o in out])


def fit_scm(model, analogs: pd.DataFrame, target: str, scaler: Scaler, anchor_y: float, seed: int):
    """train(fit_causal=True)'s second half: DAGMA, mechanisms, ICC."""
    model.w_threshold = cfg.SCM_W_THRESHOLD
    model.lambda_1 = cfg.SCM_LAMBDA
    model.T = cfg.SCM_T
    model.seed_ic50 = float(scaler.transform([anchor_y])[0])
    analog_loader = loader(datapoints(analogs[cfg.SMILES_COL], scaler.transform(analogs[target])),
                           128, True, seed)
    model.icc = model.fit_causal(analog_loader, analog_loader)
    return model


def morgan1048(smiles):
    """data_similarity's fingerprint: radius 2, 1048 bits (sic)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    return [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), radius=2, nBits=1048)
            for s in smiles]


def analog_similarity(series_smiles, analog_smiles) -> float:
    """data_similarity's second number: mean test-to-analog Tanimoto, anchor included."""
    from rdkit import DataStructs
    a = morgan1048(analog_smiles)
    total = sum(sum(DataStructs.BulkTanimotoSimilarity(fp, a)) for fp in morgan1048(series_smiles))
    return total / (len(series_smiles) * len(analog_smiles))


def predict_counterfactual(model, anchor_smiles: str, test_smiles, sim_analog: float):
    """test_counterfactual, returning (y, y_intervention, y_counterfactual, switched, var_fallback)."""
    from lightning.pytorch import Trainer
    model.inference = "counterfactual"
    model.icc_threshold = max(0.5, min(2.5 * sim_analog, 1.0))
    Y, Y_int, Y_count, switched = [], [], [], []
    for smi in test_smiles:
        test_loader = loader(datapoints([anchor_smiles, smi]), 50, False, 0)
        trainer = Trainer(accelerator="cpu", logger=False, enable_progress_bar=False,
                          enable_model_summary=False)
        cf, interv, interv_diff, counter_diff = trainer.predict(model, test_loader)[0]
        y_int, y_cf = float(interv[-1]), float(cf[-1])
        use_int = interv_diff > cfg.SCM_SWITCH * counter_diff
        Y_int.append(y_int)
        Y_count.append(y_cf)
        Y.append(y_int if use_int else y_cf)
        switched.append(bool(use_int))
    var_fallback = bool(np.var(np.array(Y)) < 0.005)
    if var_fallback:
        Y = list(Y_int)
    return np.array(Y), np.array(Y_int), np.array(Y_count), np.array(switched), var_fallback
