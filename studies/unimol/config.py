"""Shared configuration for the Uni-Mol comparison.

Uni-Mol (Zhou et al., ICLR 2023) pre-trains a 3D SE(3)-invariant transformer by
masked-atom prediction and coordinate denoising on 209M RDKit conformers of
19M molecules; Uni-Mol2 (Ji et al., NeurIPS 2024) scales the same idea to a
two-track atom/pair transformer pre-trained on 884M conformers and reports that
downstream error keeps falling as the model grows from 84M to 1.1B parameters.
Both are released, with a fine-tuning recipe, as the `unimol_tools` package, and
that package is how most people meet either model. This study fine-tunes both
with it, on the molecules, folds and statistics everything else here uses:

  imported from expansion-ml-comparison, not re-run
    lgbm           LightGBM on Morgan count fingerprints -- the floor
    chemprop_st    single-task D-MPNN
    chemprop       multitask D-MPNN
    chemeleon      multitask D-MPNN initialised from CheMeleon, then fine-tuned
    monroe         Monroe + TabPFN 3, the strongest frozen arm
    atomjepa_ft    Atom-JEPA fine-tuned, the strongest fine-tuned 3D arm

  fitted here, by 01_run_unimol.py
    unimol         Uni-Mol (v1, 47M, mol_pre_all_h_220816) fine-tuned
    unimol2        Uni-Mol2 84M fine-tuned, the package's default size

  fitted here, by 05_run_unimol_tabpfn.py
    unimol_tabpfn  Uni-Mol frozen, its CLS vector read by TabPFN 3
    unimol2_tabpfn Uni-Mol2 84M frozen, the same way -- the Monroe recipe with
                   Monroe's encoder swapped out

Each Uni-Mol arm is the package's own `MolTrain` with `split='select'`: given
the training molecules and the fold each is held out in, it fits one model per
fold, early-stops on that fold's held-out fifth and saves it, which is exactly
the 25-model protocol here. Every fold model then predicts the fixed test set.
One multitask model per fold covers all endpoints of a data set, with the
package's NaN-masked L1 loss (`multilabel_regression`), as CheMeleon does.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# --- data sets -----------------------------------------------------------
#     python 01_run_unimol.py                        # expansion
#     ADME_DATASET=biogen python 01_run_unimol.py    # Biogen


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
# unimol_tools writes its fold models, scaler and config here, one directory
# per arm and repeat; large, and only needed until the test set is predicted
WORK_DIR = PROJECT_DIR / "work" / DATASET
# the frozen encoders' vectors for every molecule, for the TabPFN arms
EMBED_DIR = PROJECT_DIR / "embeddings" / DATASET

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
UNIMOL_METHOD = "unimol"
UNIMOL2_METHOD = "unimol2"
UNIMOL_TABPFN_METHOD = "unimol_tabpfn"
UNIMOL2_TABPFN_METHOD = "unimol2_tabpfn"
TABPFN_METHOD_OF = {
    UNIMOL_METHOD: UNIMOL_TABPFN_METHOD,
    UNIMOL2_METHOD: UNIMOL2_TABPFN_METHOD,
}

BASELINE_IS_SINGLE_TASK = {
    LGBM_METHOD: True,
    "chemprop_st": True,
    "chemprop": False,
    CHEMELEON_METHOD: False,
    MONROE_METHOD: True,
    ATOMJEPA_FT_METHOD: True,
}
BASELINE_METHODS = list(BASELINE_IS_SINGLE_TASK)


@dataclass(frozen=True)
class UniMolArm:
    """One Uni-Mol model as `MolTrain` is asked for it."""

    model_name: str
    model_size: str | None
    batch_size: int


UNIMOL_ARMS = {
    UNIMOL_METHOD: UniMolArm("unimolv1", None, 16),
    UNIMOL2_METHOD: UniMolArm("unimolv2", "84m", 16),
}

METHOD_LABELS = {
    LGBM_METHOD: "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    CHEMELEON_METHOD: "ChemProp + CheMeleon",
    MONROE_METHOD: "Monroe + TabPFN 3",
    ATOMJEPA_FT_METHOD: "Atom-JEPA fine-tuned",
    UNIMOL_METHOD: "Uni-Mol fine-tuned",
    UNIMOL2_METHOD: "Uni-Mol2 84M fine-tuned",
    UNIMOL_TABPFN_METHOD: "Uni-Mol + TabPFN 3",
    UNIMOL2_TABPFN_METHOD: "Uni-Mol2 84M + TabPFN 3",
}

ALL_METHODS = list(METHOD_LABELS)
METHODS = [
    LGBM_METHOD,
    CHEMELEON_METHOD,
    MONROE_METHOD,
    ATOMJEPA_FT_METHOD,
    UNIMOL_METHOD,
    UNIMOL2_METHOD,
    UNIMOL_TABPFN_METHOD,
    UNIMOL2_TABPFN_METHOD,
]

# --- fine-tuning ---------------------------------------------------------
# The package's own defaults from unimol_tools/config/default.yaml, which is
# what a user who does not tune gets: Adam at 1e-4 with 3% linear warm-up and
# linear decay, gradient clipping at 5, mixed precision, early stopping on the
# held-out fifth with a patience of 10 epochs out of at most 100. Targets are
# standardised per endpoint rather than left to `auto`, which picks log1p for
# skewed columns and these columns are already logs that go below -1.
LEARNING_RATE = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
MAX_NORM = 5.0
TARGET_NORMALIZE = "standard"
UNIMOL_SEED = 42

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


def work_dir(method: str, repeat: int) -> Path:
    return WORK_DIR / f"{method}_r{repeat}"


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, DIAG_DIR, RESULTS_DIR, LOG_DIR,
                 WORK_DIR, EMBED_DIR, FIGURE_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)
