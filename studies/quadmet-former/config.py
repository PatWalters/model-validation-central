"""Shared configuration for the QuADMET-Former comparison.

QuADMET-Former (Raja,ikkala, et al., ChemRxiv 2026, 15002429/v3) pre-trains an
equivariant transformer, TorchMD-ET with 6 layers, hidden dimension 128 and
1.31 M parameters, on 665k QMugs molecules to predict a combination of atomic
(Loewdin partial charges) and molecular (dipole moment, HOMO-LUMO gap, total
energy, 3D E3FP fingerprint) quantum-mechanical properties. Ten checkpoints are
released on Hugging Face -- five target combinations crossed with single- and
multi-conformer pre-training -- and for each downstream ADMET task the paper
selects a pre-training target combination and fine-tunes the encoder with a
two-stage linear-probe then fine-tune (LP-FT) recipe.

The paper reports beating from-scratch training on 18 of 22 TDC tasks and 5 of
6 Biogen ADME tasks. It did not run on the fixed-holdout 5x5 BitBIRCH cluster
protocol, and it did not compare against CheMeleon or Monroe + TabPFN 3. This
study puts QuADMET-Former on the molecules, folds and statistics everything
else in this repository uses:

  imported from expansion-ml-comparison, not re-run
    lgbm           LightGBM on Morgan count fingerprints -- the floor
    chemprop_st    single-task D-MPNN
    chemprop       multitask D-MPNN
    chemeleon      multitask D-MPNN initialised from CheMeleon, then fine-tuned
    monroe         Monroe + TabPFN 3, the strongest arm in the source comparison

  fitted here
    quadmet_ft      the paper's own LP-FT recipe: TorchMD-ET + 3-layer SiLU MLP
                    head trained with L1 loss in two stages (linear probe, then
                    end-to-end fine-tuning with lower backbone learning rate),
                    by 04_run_finetune.py
    quadmet_tabpfn  the frozen pre-trained TorchMD-ET encoder read by the same
                    TabPFN 3 call as `monroe`, by 03_run_tabpfn.py

Checkpoint selection
--------------------
Because the paper releases ten pre-trained checkpoints (five QM target sets x
single/multi-conformer) and treats target-set choice as task-dependent, all ten
checkpoints are embedded once by 02_embed.py and scored on the held-out
validation fifth of each fold (never on the test set) to audit whether the ten
checkpoints differ meaningfully relative to fold-to-fold noise.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# --- data sets -----------------------------------------------------------
#     python 04_run_finetune.py                        # expansion
#     ADME_DATASET=biogen python 04_run_finetune.py    # Biogen


@dataclass(frozen=True)
class DataSet:
    """One collection of molecules and endpoints, and how to group them."""

    name: str
    label: str
    targets: list[str]
    groups: dict[str, list[str]]

    @property
    def group_of(self) -> dict[str, str]:
        return {t: g for g, ts in self.groups.items() for t in ts}


DATASETS = {
    "expansion": DataSet(
        name="expansion",
        label="ExpansionRx",
        targets=[
            "LogD",
            "LogS",
            "LOG_HLM",
            "LOG_MLM",
            "LOG_Caco_AB",
            "LOG_Caco_Efflux",
            "LOG_MPPB",
            "LOG_MBPB",
            "LOG_MGMB",
        ],
        groups={
            "physchem_binding": ["LogD", "LogS", "LOG_MPPB", "LOG_MBPB", "LOG_MGMB"],
            "metabolism": ["LOG_HLM", "LOG_MLM"],
            "permeability": ["LOG_Caco_AB", "LOG_Caco_Efflux"],
        },
    ),
    "biogen": DataSet(
        name="biogen",
        label="Biogen ADME",
        targets=[
            "LOG_SOL",
            "LOG_HLM",
            "LOG_RLM",
            "LOG_MDR1_ER",
            "LOG_HPPB",
            "LOG_RPPB",
        ],
        groups={
            "physchem_binding": ["LOG_SOL", "LOG_HPPB", "LOG_RPPB"],
            "metabolism": ["LOG_HLM", "LOG_RLM"],
            "permeability": ["LOG_MDR1_ER"],
        },
    ),
}

DATASET = os.environ.get("ADME_DATASET", "expansion")
if DATASET not in DATASETS:
    raise SystemExit(
        f"unknown ADME_DATASET {DATASET!r} -- choose from {', '.join(DATASETS)}"
    )
ACTIVE = DATASETS[DATASET]

# --- paths ---------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

SOURCE_REPO = Path(
    os.environ.get("ADME_SOURCE_REPO", PROJECT_DIR.parent / "expansion-ml-comparison")
).expanduser()

DATA_DIR = PROJECT_DIR / "data" / DATASET
FOLD_DIR = PROJECT_DIR / "folds" / DATASET
PRED_DIR = PROJECT_DIR / "predictions" / DATASET
DIAG_DIR = PROJECT_DIR / "diagnostics" / DATASET
RESULTS_DIR = PROJECT_DIR / "results" / DATASET
LOG_DIR = PROJECT_DIR / "logs" / DATASET

FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

MASTER_CSV = DATA_DIR / "master.csv"
FOLD_CSV = FOLD_DIR / "fold_assignments.csv"

# Authors' checkout and downloaded Hugging Face checkpoints
QUADMET_HOME = Path(
    os.environ.get("QUADMET_HOME", Path.home() / "software" / "quadmetformer")
).expanduser().resolve()
CKPT_DIR = Path(
    os.environ.get("QUADMET_CKPT_DIR", PROJECT_DIR / "ckpts")
).expanduser().resolve()

CONFORMER_DIR = DATA_DIR / "conformers"
EMBED_DIR = DATA_DIR / "quadmet_embeddings"

SELECTION_CSV = RESULTS_DIR / "quadmet_selection.csv"
CHOICE_CSV = RESULTS_DIR / "quadmet_choice.csv"
ABLATION_CSV = RESULTS_DIR / "quadmet_ablations.csv"

MONROE_HOME = Path(
    os.environ.get("MONROE_HOME", Path.home() / "software" / "monroe")
).expanduser().resolve()

PREDICTIONS_PARQUET = RESULTS_DIR / "predictions_all.parquet"
FOLD_METRICS_CSV = RESULTS_DIR / "fold_metrics.csv"
DIAGNOSTICS_CSV = RESULTS_DIR / "diagnostics.csv"
TIMING_CSV = RESULTS_DIR / "timing.csv"


@dataclass(frozen=True)
class Paths:
    """Where one data set's outputs live, for the summary and the page builder."""

    dataset: DataSet
    results: Path

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def tables(self) -> Path:
        return self.results / "tables"

    @property
    def fold_metrics(self) -> Path:
        return self.results / "fold_metrics.csv"

    @property
    def diagnostics(self) -> Path:
        return self.results / "diagnostics.csv"

    @property
    def timing(self) -> Path:
        return self.results / "timing.csv"

    @property
    def selection(self) -> Path:
        return self.results / "quadmet_selection.csv"

    @property
    def choice(self) -> Path:
        return self.results / "quadmet_choice.csv"

    @property
    def ablations(self) -> Path:
        return self.results / "quadmet_ablations.csv"


def paths(name: str) -> Paths:
    if name not in DATASETS:
        raise SystemExit(f"unknown data set {name!r} -- choose from {', '.join(DATASETS)}")
    return Paths(DATASETS[name], PROJECT_DIR / "results" / name)


# --- data columns --------------------------------------------------------
SMILES_COL = "SMILES"
ID_COL = "Name"
SET_COL = "ds"
CLUSTER_COL = "cluster"

TARGET_COLS = list(ACTIVE.targets)
TARGET_GROUPS = {group: list(targets) for group, targets in ACTIVE.groups.items()}
GROUPS = list(TARGET_GROUPS)
GROUP_OF = dict(ACTIVE.group_of)

assert sorted(GROUP_OF) == sorted(TARGET_COLS), "TARGET_GROUPS must cover TARGET_COLS exactly"

# --- methods -------------------------------------------------------------
LGBM_METHOD = "lgbm"
CHEMELEON_METHOD = "chemeleon"
MONROE_METHOD = "monroe"
ATOMJEPA_FT_METHOD = "atomjepa_ft"
FT_METHOD = "quadmet_ft"
TABPFN_METHOD = "quadmet_tabpfn"

BASELINE_IS_SINGLE_TASK = {
    LGBM_METHOD: True,
    "chemprop_st": True,
    "chemprop": False,
    CHEMELEON_METHOD: False,
    MONROE_METHOD: True,
    ATOMJEPA_FT_METHOD: True,
}
BASELINE_METHODS = list(BASELINE_IS_SINGLE_TASK)

METHOD_LABELS = {
    LGBM_METHOD: "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    CHEMELEON_METHOD: "ChemProp + CheMeleon",
    MONROE_METHOD: "Monroe + TabPFN 3",
    ATOMJEPA_FT_METHOD: "Atom-JEPA fine-tuned",
    TABPFN_METHOD: "QuADMET-Former + TabPFN 3",
    FT_METHOD: "QuADMET-Former fine-tuned",
}

ALL_METHODS = list(METHOD_LABELS)
METHODS = [
    LGBM_METHOD,
    CHEMELEON_METHOD,
    MONROE_METHOD,
    ATOMJEPA_FT_METHOD,
    TABPFN_METHOD,
    FT_METHOD,
]

# --- the 10 released checkpoints -----------------------------------------
# 5 target combinations x 2 conformer regimes (single_conf, multi_conf)
CHECKPOINT_SPECS = [
    ("single_lw", "single_conf/_lowdin.ckpt", "Single", "[Lw]"),
    ("single_3d", "single_conf/e3fp_.ckpt", "Single", "[3D]"),
    ("single_dm_hl_te", "single_conf/dipole_moment_hl_gap_total_energy_.ckpt", "Single", "[DM,HL,TE]"),
    ("single_lw_dm_hl_te", "single_conf/dipole_moment_hl_gap_total_energy_lowdin.ckpt", "Single", "[Lw,DM,HL,TE]"),
    ("single_all", "single_conf/dipole_moment_e3fp_hl_gap_total_energy_lowdin.ckpt", "Single", "[All]"),
    ("multi_lw", "multi_conf/_lowdin.ckpt", "Multi", "[Lw]"),
    ("multi_3d", "multi_conf/e3fp_.ckpt", "Multi", "[3D]"),
    ("multi_dm_hl_te", "multi_conf/dipole_moment_hl_gap_total_energy_.ckpt", "Multi", "[DM,HL,TE]"),
    ("multi_lw_dm_hl_te", "multi_conf/dipole_moment_hl_gap_total_energy_lowdin.ckpt", "Multi", "[Lw,DM,HL,TE]"),
    ("multi_all", "multi_conf/dipole_moment_e3fp_hl_gap_total_energy_lowdin.ckpt", "Multi", "[All]"),
]

CHECKPOINTS = [name for name, _, _, _ in CHECKPOINT_SPECS]
CKPT_RELPATH = {name: rel for name, rel, _, _ in CHECKPOINT_SPECS}
CKPT_CONF = {name: conf for name, _, conf, _ in CHECKPOINT_SPECS}
CKPT_TARGETS = {name: targets for name, _, _, targets in CHECKPOINT_SPECS}
DEFAULT_CHECKPOINT = "single_all"


def checkpoint_path(name: str) -> Path:
    return CKPT_DIR / CKPT_RELPATH[name]


def embedding_npz(name: str) -> Path:
    return EMBED_DIR / f"{name}.npz"


def checkpoint_label(name: str) -> str:
    return f"{CKPT_CONF[name]} {CKPT_TARGETS[name]}"


# --- TorchMD-ET architecture & LP-FT hyperparameters ---------------------
HIDDEN_CHANNELS = 128
PRETRAIN_LAYERS = 6
NUM_RBF = 50
NUM_HEADS = 16
CUTOFF_LOWER = 0.0
CUTOFF_UPPER = 5.0
MAX_ATOM_TYPE = 100
MAX_NUM_NEIGHBORS = 32

# Downstream LP-FT settings from trainer.py and downstream_train.py
LP_EPOCHS = 80
FT_EPOCHS = 12
MIN_EPOCHS = 6
PATIENCE = 4
BATCH_SIZE = 32
ACCUMULATE_GRAD = 2
LR_HEAD = 1e-3
BACKBONE_LR_MULT = 0.1
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
HEAD_HIDDEN_DIM = 128

# --- cross validation ----------------------------------------------------
N_REPEATS = 5
N_SPLITS = 5
RANDOM_SEED = 0xF00D

REPEATS = list(range(N_REPEATS))
FOLDS = list(range(N_SPLITS))


def fold_seed(repeat: int, fold: int) -> int:
    """A distinct, reproducible seed for each of the 25 fold models."""
    return RANDOM_SEED + 100 * repeat + fold


# --- tidy prediction schema ---------------------------------------------
PRED_COLUMNS = [
    "method",
    "endpoint",
    "repeat",
    "fold",
    ID_COL,
    SMILES_COL,
    "y_true",
    "y_pred",
]

METRICS = ["r2", "spearman", "mae"]
METRIC_HIGHER_IS_BETTER = {"r2": True, "spearman": True, "mae": False}
METRIC_LABELS = {"r2": "$R^2$", "spearman": "Spearman $\\rho$", "mae": "MAE"}


def pred_csv(method: str, name: str, repeat: int, fold: int) -> Path:
    """Tidy predictions for one method and fold -- also the resume point."""
    return PRED_DIR / method / f"{name}_r{repeat}_f{fold}.csv"


def diag_json(endpoint: str, repeat: int, fold: int) -> Path:
    return DIAG_DIR / f"{FT_METHOD}_{endpoint}_r{repeat}_f{fold}.json"


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, DIAG_DIR, RESULTS_DIR, LOG_DIR,
                 FIGURE_DIR, TABLE_DIR, CONFORMER_DIR, EMBED_DIR):
        path.mkdir(parents=True, exist_ok=True)
