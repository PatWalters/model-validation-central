"""Shared configuration for the NVIDIA structured-data-models comparison.

NVIDIA's `structured-data-models` package (github.com/NVIDIA/structured-data-models)
collects tabular foundation models behind one in-context interface: TabICLv2,
Kumo Tabular in three sizes, and Google's TabFM. Every one of them is the same
kind of object as the TabPFN head in the `monroe` arm of
`expansion-ml-comparison` -- handed a table of labelled rows and a table of
unlabelled ones, it predicts the second from the first in a single forward pass,
with no weights updated.

So this study holds the representation fixed and swaps the head. Every arm reads
the same frozen 720-d Monroe embedding of each molecule, on the same folds, with
the same fit and test masks. The reference is `monroe`: that embedding read by
TabPFN 3, which is the configuration the Monroe paper itself was written
against. A difference between an arm here and the reference is the head and
nothing else.

  imported from expansion-ml-comparison, not re-run
    lgbm           LightGBM on Morgan count fingerprints -- the floor
    chemprop_st    single-task D-MPNN
    chemprop       multitask D-MPNN
    chemeleon      multitask D-MPNN initialised from CheMeleon
    monroe         Monroe + TabPFN 3, the reference
    monroe_tabicl  Monroe + TabICL, from the authors' own `tabicl` 2.1.1 package

  fitted here, every one through the `sdm` package, over the same embeddings
    sdm_tabiclv2       TabICLv2 (Qu et al., ICML 2026)
    sdm_kumo_small     Kumo Tabular, small
    sdm_kumo_medium    Kumo Tabular, medium
    sdm_kumo_large     Kumo Tabular, large
    sdm_tabfm          TabFM (Google Research, 2026)

`monroe_tabicl` and `sdm_tabiclv2` load the same checkpoint,
`tabicl-regressor-v2-20260212.ckpt`, through two different implementations. The
pair is the study's control: if NVIDIA's port reproduces the upstream model, the
two should be indistinguishable, and any gap between the other `sdm` arms and the
reference can be read as the model rather than the wrapper.

The package also ships TimesFM 3 (a time-series forecaster) and Kumo Relational
(a model over linked tables). Neither has anything to say about one table of
molecules with one numeric target, so neither is an arm.

The four chemprop/LightGBM arms are carried for the cross-study overview, which
refuses to pool a study that does not reproduce them exactly. The report focuses
on the head comparison and shows LightGBM alone as the floor.

Evaluation protocol
-------------------
Unchanged from the source repository. The `ds` column fixes a train/test split.
The 25 replicates come from 5 repeats of 5-fold `GroupKFold` over the
`ds == 'train'` molecules, grouped by cluster: four fifths are the training set
and every model is scored on the *same* held-out `ds == 'test'` molecules. The
held-out fifth is unused, as it is for LightGBM and Monroe -- an in-context model
has no training loop to stop early.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# --- data sets -----------------------------------------------------------
#     python 01_run_sdm.py --model kumo_small                      # expansion
#     ADME_DATASET=biogen python 01_run_sdm.py --model kumo_small  # Biogen


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

# Where the folds, the molecule table, the Monroe embeddings and the imported
# arms come from. 00_import_baselines.py copies them in once.
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
# One 720-d Monroe embedding per molecule, in master.csv row order, written by
# `09_run_monroe.py --embed` in the source repository. Every arm here reads it.
MONROE_NPZ = DATA_DIR / "monroe_embeddings.npz"

PREDICTIONS_PARQUET = RESULTS_DIR / "predictions_all.parquet"
FOLD_METRICS_CSV = RESULTS_DIR / "fold_metrics.csv"
# Wall-clock per fold, one row per (method, endpoint, repeat, fold).
TIMING_CSV = RESULTS_DIR / "timing.csv"


@dataclass(frozen=True)
class Paths:
    """Where one data set's outputs live, for the page builder."""

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
MONROE_METHOD = "monroe"
MONROE_TABICL_METHOD = "monroe_tabicl"

# The arms imported from the source comparison, and whether each writes one
# prediction file per endpoint (single-task) or one per assay family.
BASELINE_IS_SINGLE_TASK = {
    LGBM_METHOD: True,
    "chemprop_st": True,
    "chemprop": False,
    "chemeleon": False,
    MONROE_METHOD: True,
    MONROE_TABICL_METHOD: True,
}
BASELINE_METHODS = list(BASELINE_IS_SINGLE_TASK)

# The heads fitted here. `key` is the --model name, `method` what the arm is
# recorded as, and the rest is how the model is built and run.
#
# `num_estimators` and `autocast` are NVIDIA's own defaults for each model, taken
# from benchmark/tabular/model.py in their repository -- the settings behind the
# TabArena numbers they report. Nothing here is tuned.
SDM_MODELS = {
    "tabiclv2": {"method": "sdm_tabiclv2", "cls": "TabICLv2", "kwargs": {},
                 "num_estimators": 8, "autocast": "float16"},
    "kumo_small": {"method": "sdm_kumo_small", "cls": "KumoTabular",
                   "kwargs": {"size": "small"}, "num_estimators": 8,
                   "autocast": "float16"},
    "kumo_medium": {"method": "sdm_kumo_medium", "cls": "KumoTabular",
                    "kwargs": {"size": "medium"}, "num_estimators": 8,
                    "autocast": "float16"},
    "kumo_large": {"method": "sdm_kumo_large", "cls": "KumoTabular",
                   "kwargs": {"size": "large"}, "num_estimators": 16,
                   "autocast": "float16"},
    # TabFM's weights are under Google's TabFM Non-Commercial License v1.0, and
    # the package will not download them until that licence is accepted. See the
    # README.
    "tabfm": {"method": "sdm_tabfm", "cls": "TabFM", "kwargs": {},
              "num_estimators": 32, "autocast": "bfloat16"},
}
SDM_METHODS = [spec["method"] for spec in SDM_MODELS.values()]

METHOD_LABELS = {
    LGBM_METHOD: "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    "chemeleon": "ChemProp + CheMeleon",
    MONROE_METHOD: "Monroe + TabPFN 3",
    MONROE_TABICL_METHOD: "Monroe + TabICL (tabicl)",
    "sdm_tabiclv2": "Monroe + TabICLv2 (sdm)",
    "sdm_kumo_small": "Monroe + Kumo small",
    "sdm_kumo_medium": "Monroe + Kumo medium",
    "sdm_kumo_large": "Monroe + Kumo large",
    "sdm_tabfm": "Monroe + TabFM",
}

ALL_METHODS = list(METHOD_LABELS)

# What the report compares. The chemprop arms are here for the cross-study
# overview, not for this page's question; LightGBM stays as the floor every arm
# is supposed to clear.
METHODS = [LGBM_METHOD, MONROE_METHOD, MONROE_TABICL_METHOD, *SDM_METHODS]

# Arms to leave off the report while they are still running, so an interim page
# can be built from the ones that are finished:
#
#     ADME_SKIP=sdm_tabfm python 03_report.py
SKIP = [m for m in os.environ.get("ADME_SKIP", "").split(",") if m]
unknown_skip = set(SKIP) - set(METHODS)
if unknown_skip:
    raise SystemExit(f"ADME_SKIP names unknown arms: {', '.join(sorted(unknown_skip))}")
METHODS = [m for m in METHODS if m not in SKIP]

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


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, RESULTS_DIR, LOG_DIR,
                 FIGURE_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)
