"""Shared configuration for the PT-GIN comparison.

Five modelling approaches on the same fifteen endpoints, the same 5x5 folds and
the same held-out test sets used in `expansion-ml-comparison`:

  lgbm         LightGBM on Morgan count fingerprints, one model per endpoint
  chemprop_st  a D-MPNN trained from scratch, one model per endpoint
  chemprop     the same D-MPNN, one model per assay family
  chemeleon    the same multitask D-MPNN, message passing initialised from the
               CheMeleon foundation model
  ptgin        a GIN pre-trained to predict hashed ECFP4 on QMugs, frozen, its
               concatenated layer embeddings handed to LightGBM
               (Money-Kyrle et al., arXiv 2605.10722)

The four baselines are not re-run here. Their per-fold predictions are copied
from that repository by 00_import_baselines.py, so the numbers on this page are
the same numbers that appear on the other one and a difference between arms is a
difference between methods, not between runs.

What PT-GIN is
--------------
A Graph Isomorphism Network whose atoms are tokenised by Sort & Slice: every
circular substructure up to a maximum radius is ranked by how many pre-training
molecules it appears in, the vocabulary is sliced to a fixed size, and each atom
carries one learned token embedding per radius. The network is pre-trained on
462,189 QMugs molecules to predict that molecule's own 2048-bit hashed ECFP4 as
a multitask binary classification, with no experimental label anywhere in the
objective.

Downstream the encoder is frozen. Every layer's output is graph-pooled, the
per-layer vectors are concatenated, and LightGBM predicts the endpoint from that.
Nothing is fine-tuned -- the authors report that end-to-end fine tuning was more
expensive and no better, and sometimes worse.

That makes `ptgin` the same shape of arm as `lgbm`: one single-task LightGBM
regressor per endpoint per fold, differing only in what it is handed. The
comparison between the two is a comparison of representations with the predictor
held fixed, which is the comparison the paper itself is built around.

Checkpoint selection
--------------------
The paper does not have one PT-GIN. It pre-trains a grid of maximum radius by
vocabulary size and selects per task, during downstream hyperparameter tuning,
whichever pre-trained model does best. Ten of those checkpoints are released.

That selection is part of the method, so it is reproduced here, but it is made
where this protocol allows a choice to be made: on the validation fifth. For each
endpoint every one of the ten checkpoints is fit on the four fifths of all 25
folds and scored on the held-out fifth, and the checkpoint with the best mean
validation R^2 becomes that endpoint's PT-GIN. The test set is untouched by the
choice. 01_run_ptgin.py writes the whole 10-way validation table, so what was
selected -- and by how much -- is auditable.

Evaluation protocol
-------------------
Unchanged from the source repository. The `ds` column of the raw file fixes a
train/test split. The 25 replicates come from 5 repeats of 5-fold `GroupKFold`
over the `ds == 'train'` molecules, grouped by `cluster`: four fifths are the
training set, the held-out fifth is the validation set, and every model is scored
on the *same* held-out `ds == 'test'` molecules.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# --- data sets -----------------------------------------------------------
# Both collections from the source comparison, with the same endpoints and the
# same assay families. Which one is active is read from the environment once, at
# import, so every script picks it up without threading a flag through:
#
#     python 01_run_ptgin.py                      # expansion, the default
#     ADME_DATASET=biogen python 01_run_ptgin.py  # the Biogen set


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

# Where the folds, the cleaned molecule table and the four baseline arms come
# from. 00_import_baselines.py copies them in once; after that this project runs
# on its own and the variable is only read again if the import is repeated.
SOURCE_REPO = Path(
    os.environ.get("ADME_SOURCE_REPO", PROJECT_DIR.parent / "expansion-ml-comparison")
).expanduser()

DATA_DIR = PROJECT_DIR / "data" / DATASET
FOLD_DIR = PROJECT_DIR / "folds" / DATASET
PRED_DIR = PROJECT_DIR / "predictions" / DATASET
RESULTS_DIR = PROJECT_DIR / "results" / DATASET
LOG_DIR = PROJECT_DIR / "logs" / DATASET

FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

MASTER_CSV = DATA_DIR / "master.csv"
FOLD_CSV = FOLD_DIR / "fold_assignments.csv"

# One .npz per pre-trained checkpoint, each holding a frozen embedding per
# molecule in master.csv row order. Ten files, written once by --embed.
EMBED_DIR = DATA_DIR / "ptgin_embeddings"

# The 10 x endpoint x fold validation sweep that picks each endpoint's
# checkpoint, and the resulting choice.
SELECTION_CSV = RESULTS_DIR / "ptgin_selection.csv"
CHOICE_CSV = RESULTS_DIR / "ptgin_choice.csv"

PREDICTIONS_PARQUET = RESULTS_DIR / "predictions_all.parquet"
FOLD_METRICS_CSV = RESULTS_DIR / "fold_metrics.csv"


@dataclass(frozen=True)
class Paths:
    """Where one data set's outputs live, whichever one is currently active.

    Every script works on the active data set through the module-level constants
    above. The page builder is the exception: it puts both collections on one
    page, so it needs to reach the other one's tables and figures without
    re-importing this module under a different environment.
    """

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
    def selection(self) -> Path:
        return self.results / "ptgin_selection.csv"

    @property
    def choice(self) -> Path:
        return self.results / "ptgin_choice.csv"

    @property
    def master(self) -> Path:
        return PROJECT_DIR / "data" / self.dataset.name / "master.csv"


def paths(name: str) -> Paths:
    if name not in DATASETS:
        raise SystemExit(f"unknown data set {name!r} -- choose from {', '.join(DATASETS)}")
    return Paths(DATASETS[name], PROJECT_DIR / "results" / name)

# --- data columns --------------------------------------------------------
SMILES_COL = "SMILES"
ID_COL = "Name"
SET_COL = "ds"           # 'train' / 'test' in the raw file
CLUSTER_COL = "cluster"  # groups the CV folds and the validation split

TARGET_COLS = list(ACTIVE.targets)
TARGET_GROUPS = {group: list(targets) for group, targets in ACTIVE.groups.items()}
GROUPS = list(TARGET_GROUPS)
GROUP_OF = dict(ACTIVE.group_of)

assert sorted(GROUP_OF) == sorted(TARGET_COLS), "TARGET_GROUPS must cover TARGET_COLS exactly"

# --- methods -------------------------------------------------------------
LGBM_METHOD = "lgbm"
PTGIN_METHOD = "ptgin"

# The four arms imported from the source comparison, and how each one names its
# prediction files there: single-task arms write one file per endpoint, multitask
# arms one per assay family.
BASELINE_METHODS = [LGBM_METHOD, "chemprop_st", "chemprop", "chemeleon"]
BASELINE_IS_SINGLE_TASK = {
    LGBM_METHOD: True,
    "chemprop_st": True,
    "chemprop": False,
    "chemeleon": False,
}

METHOD_LABELS = {
    LGBM_METHOD: "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    "chemeleon": "ChemProp + CheMeleon",
    PTGIN_METHOD: "PT-GIN + LightGBM",
}

METHODS = list(METHOD_LABELS)
ALL_METHODS = list(METHODS)

# --- the pre-trained checkpoints -----------------------------------------
# Where the authors' checkout lives. It ships the ten released PT-GIN
# checkpoints under pt_models/vocab_size/, so nothing is pre-trained here.
PTGIN_HOME = Path(
    os.environ.get("PTGIN_HOME", Path.home() / "software" / "topological-pretraining")
).expanduser()
CHECKPOINT_DIR = PTGIN_HOME / "pt_models" / "vocab_size"

# The grid the paper searched, as released: maximum substructure radius crossed
# with vocabulary size. Radius 0 is the MolE-style one-token-per-atom-type
# featurisation; radius 1 and 2 add a token per larger circular environment.
CHECKPOINTS = [
    "pt_gin_radius_0_vocab_2048",
    "pt_gin_radius_1_vocab_2048",
    "pt_gin_radius_1_vocab_4096",
    "pt_gin_radius_1_vocab_8192",
    "pt_gin_radius_1_vocab_16384",
    "pt_gin_radius_2_vocab_1024",
    "pt_gin_radius_2_vocab_2048",
    "pt_gin_radius_2_vocab_4096",
    "pt_gin_radius_2_vocab_8192",
    "pt_gin_radius_2_vocab_16384",
]


def checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"{name}.pt"


def embedding_npz(name: str) -> Path:
    return EMBED_DIR / f"{name}.npz"


def checkpoint_radius(name: str) -> int:
    return int(name.split("_radius_")[1].split("_")[0])


def checkpoint_vocab(name: str) -> int:
    return int(name.split("_vocab_")[1])


def checkpoint_label(name: str) -> str:
    return f"r={checkpoint_radius(name)}, {checkpoint_vocab(name)}"

# How the frozen embedding is read out of the network. Both are the paper's
# downstream defaults: every layer's graph-pooled output, concatenated.
LAYER_POOL = "concat"
EMBED_STATE = "global"

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
    """Tidy predictions for one method and fold.

    `name` is the endpoint for the single-task arms and the assay family for the
    multitask ones, since a multitask model predicts its whole group in one pass.
    These files are the resume points: a run skips any combination whose file
    already exists.
    """
    return PRED_DIR / method / f"{name}_r{repeat}_f{fold}.csv"


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, RESULTS_DIR, LOG_DIR,
                 FIGURE_DIR, TABLE_DIR, EMBED_DIR):
        path.mkdir(parents=True, exist_ok=True)
