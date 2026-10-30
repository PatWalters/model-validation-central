"""Shared configuration for the KERMT comparison.

KERMT (Adrian et al., 2025) is NVIDIA's accelerated reimplementation of GROVER,
a dual-view graph transformer, and its claim is that pre-trained graph models
fine-tuned *multitask* beat non-pre-trained graph networks on ADMET. KERMT v2
(Xue et al., 2026) re-pretrains the same encoder with a contrastive,
probabilistic objective, fine-tunes it with a task-specific head per endpoint,
and is the released checkpoint, NV-KERMT-70M-v2. This study fine-tunes it, and
the two original GROVER checkpoints it descends from, on the molecules, folds
and statistics everything else here uses:

  imported from expansion-ml-comparison, not re-run
    lgbm             LightGBM on Morgan count fingerprints -- the floor
    chemprop_st      single-task D-MPNN
    chemprop         multitask D-MPNN
    chemeleon        multitask D-MPNN initialised from CheMeleon, then fine-tuned
    monroe           Monroe + TabPFN 3, the strongest frozen arm
    atomjepa_ft      Atom-JEPA fine-tuned, the strongest fine-tuned 3D arm

  fitted here, by 02_run_kermt.py -- one recipe, three starting points
    kermt_ft         NV-KERMT-70M-v2 (contrastive hybrid) fine-tuned
    grover_base_ft   GROVER base (Rong et al., 2020) fine-tuned with KERMT's code
    grover_large_ft  GROVER large, the same way

  fitted here, by 03_run_kermt_tabpfn.py
    kermt_tabpfn     NV-KERMT-70M-v2 frozen, its 512-d cMIM latent read by
                     TabPFN 3 -- the Monroe recipe with Monroe's encoder swapped

The released v2 checkpoint was pre-trained on a corpus that includes every
Biogen and ExpansionRx molecule, test molecules too (structures, no labels).
The two GROVER checkpoints were pre-trained before either data set existed in
this form, so they are the arms that have never seen a test molecule.

Each fine-tuned arm is KERMT's own `main.py finetune`, once per repeat and
fold: train on four fifths, keep the epoch with the best validation MAE on the
held-out fifth, predict the fixed test set. One multitask model per fold covers
all endpoints of a data set, labels masked where missing, as CheMeleon does.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# --- data sets -----------------------------------------------------------
#     python 02_run_kermt.py --arm kermt_ft                        # expansion
#     ADME_DATASET=biogen python 02_run_kermt.py --arm kermt_ft    # Biogen


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
# KERMT's per-fold CSVs, feature files, checkpoints and log while a fold runs;
# large, and only needed until the test set is predicted
WORK_DIR = PROJECT_DIR / "work" / DATASET
# the RDKit 2D features and the frozen encoder's vectors, in master.csv order
FEATURE_DIR = PROJECT_DIR / "features" / DATASET
EMBED_DIR = PROJECT_DIR / "embeddings" / DATASET
FEATURES_NPZ = FEATURE_DIR / "rdkit_2d_normalized.npz"

FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

MASTER_CSV = DATA_DIR / "master.csv"
FOLD_CSV = FOLD_DIR / "fold_assignments.csv"

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
KERMT_METHOD = "kermt_ft"
GROVER_BASE_METHOD = "grover_base_ft"
GROVER_LARGE_METHOD = "grover_large_ft"
KERMT_TABPFN_METHOD = "kermt_tabpfn"

BASELINE_IS_SINGLE_TASK = {
    LGBM_METHOD: True,
    "chemprop_st": True,
    "chemprop": False,
    CHEMELEON_METHOD: False,
    MONROE_METHOD: True,
    ATOMJEPA_FT_METHOD: True,
}
BASELINE_METHODS = list(BASELINE_IS_SINGLE_TASK)

# Checkpoints live outside the study; none is ours to redistribute.
WEIGHTS_DIR = Path(os.environ.get("KERMT_WEIGHTS", PROJECT_DIR / "weights")).expanduser()
# The KERMT checkout whose main.py runs the fine-tuning (NVIDIA-BioNeMo/KERMT).
KERMT_REPO = Path(os.environ.get("KERMT_REPO", PROJECT_DIR / "upstream" / "KERMT")).expanduser()
KERMT_COMMIT = "8828743"


@dataclass(frozen=True)
class KermtArm:
    """One pre-trained checkpoint, and the sha256 it has to have."""

    checkpoint: str
    sha256: str
    batch_size: int


KERMT_ARMS = {
    KERMT_METHOD: KermtArm(
        "NV-KERMT-70M-v2/kermt_contrastive_v2.0.pt",  # HF revision 7df5eb3
        "e9e6649bc96503fbdb3023e312764ecbbbafd686d9a62865a1fec9466cea6be3", 32),
    GROVER_BASE_METHOD: KermtArm(
        "grover_base.pt",
        "47e095880d71baf29ea6f6253473cd56d5406213fa82959c6e14ea469e06b1de", 32),
    GROVER_LARGE_METHOD: KermtArm(
        "grover_large.pt",
        "4b0c436fbd6ed8539fa92a0c9f890878f5e3dd8591d959315e773efb6302baaa", 32),
}

METHOD_LABELS = {
    LGBM_METHOD: "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    CHEMELEON_METHOD: "ChemProp + CheMeleon",
    MONROE_METHOD: "Monroe + TabPFN 3",
    ATOMJEPA_FT_METHOD: "Atom-JEPA fine-tuned",
    KERMT_METHOD: "KERMT v2 fine-tuned",
    GROVER_BASE_METHOD: "GROVER base fine-tuned",
    GROVER_LARGE_METHOD: "GROVER large fine-tuned",
    KERMT_TABPFN_METHOD: "KERMT v2 + TabPFN 3",
}

ALL_METHODS = list(METHOD_LABELS)
METHODS = [
    LGBM_METHOD,
    CHEMELEON_METHOD,
    MONROE_METHOD,
    ATOMJEPA_FT_METHOD,
    GROVER_BASE_METHOD,
    GROVER_LARGE_METHOD,
    KERMT_METHOD,
    KERMT_TABPFN_METHOD,
]

# --- fine-tuning ---------------------------------------------------------
# The recipe of the KERMT v2 paper (Appendix E): end-to-end fine-tuning for
# 100 epochs, a 2-layer shared MLP trunk under a 3-layer head per endpoint,
# every MLP 700 wide, the epoch with the best validation MAE kept. What the
# paper does not state comes from the repository's own kermt-finetune defaults
# (skills/_shared/config/defaults_finetune.json), and the encoder's shape from
# the checkpoint. The same flags are passed for all three checkpoints, but
# KERMT's load_checkpoint copies the pre-trained checkpoint's saved model args
# over them (task/train.py:249), so dropout, bond_drop_rate, dist_coff,
# weight_decay and select_by_loss are in effect the checkpoint's: dropout 0.1
# for v2 and 0 for GROVER, bond_drop_rate 0 and dist_coff 0.1 for all three,
# and GROVER's best epoch picked on validation loss rather than MAE. The code
# is run as released; 02_run_kermt.py records what each fold actually used.
FINETUNE = {
    "epochs": 100,
    "init_lr": 1e-4,
    "max_lr": 1e-4,
    "final_lr": 2e-5,
    "dropout": 0.0,
    "bond_drop_rate": 0.1,
    "dist_coff": 0.15,
    "ffn_hidden_size": 700,
    "ffn_num_layers": 2,
    "ffn_num_task_specific_layers": 3,
    "ffn_task_specific_hidden_size": 700,
    "metric": "mae",
}
# per-molecule CDF-normalised RDKit 2D descriptors, the repository default; the
# normalisation is fixed, not fitted, so computing them once for every molecule
# leaks nothing between folds
FEATURES_GENERATOR = "rdkit_2d_normalized"

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


def diag_json(method: str, repeat: int, fold: int) -> Path:
    return DIAG_DIR / f"{method}_r{repeat}_f{fold}.json"


def work_dir(method: str, repeat: int, fold: int) -> Path:
    return WORK_DIR / f"{method}_r{repeat}_f{fold}"


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, DIAG_DIR, RESULTS_DIR, LOG_DIR,
                 WORK_DIR, FEATURE_DIR, EMBED_DIR, FIGURE_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)
