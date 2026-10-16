"""Shared configuration for the MolPAIR comparison.

MolPAIR (Lee, Lee, Jeong and Yoo, arXiv 2609.38744) corrects a tabular foundation
model's prediction with a second one. A global TabPFN reads a frozen CheMeleon
fingerprint of each molecule and predicts the endpoint in context. A relation
TabPFN is then shown pairs of training molecules -- their RDKit descriptor
difference, their similarity, whether they are a matched pair, and the global
model's out-of-fold predictions for both -- and learns how the global model's
error changes from one to the other. At query time each test molecule borrows
the known errors of up to 32 similar training molecules, shifted by the
predicted error difference, and the global prediction is corrected by their
weighted mean, scaled back when the nearest anchor is not very similar.

The paper reports it improving its own global baseline on 46 of 58 MoleculeACE
and Polaris tasks under scaffold-plus-similarity splits. This study asks the
question the paper does not: put on the same molecules, folds and statistics as
everything else here, does it beat the methods it would actually have to beat?

  imported from expansion-ml-comparison, not re-run
    lgbm           LightGBM on Morgan count fingerprints -- the floor
    chemprop_st    single-task D-MPNN
    chemprop       multitask D-MPNN
    chemeleon      multitask D-MPNN initialised from CheMeleon, then fine-tuned
    monroe         Monroe + TabPFN 3, the strongest arm in the source comparison

  fitted here, by 02_run_molpair.py
    molpair_global CheMeleon fingerprint + TabPFN 3: MolPAIR's own first stage,
                   which the paper calls the global baseline
    molpair        the full method, the global prediction plus the pair correction

`molpair_global` is not an extra arm bolted on. It is the prediction MolPAIR
corrects, computed in the same call, so the pair `molpair` against
`molpair_global` is the paper's own central comparison and isolates exactly
what the relation stage adds. Against `monroe` the global stage is the same
head on a different frozen representation; against `chemeleon` it is the same
pre-trained encoder, frozen rather than fine-tuned.

The authors' code link (github.com/nums-ai/MolPAIR) does not resolve, so the
method is reimplemented from the paper, its Section 3 and Appendix A. Every
setting it states is used as stated; see `MOLPAIR` below.

Evaluation protocol
-------------------
Unchanged from the source repository. The `ds` column fixes a train/test split.
The 25 replicates come from 5 repeats of 5-fold `GroupKFold` over the
`ds == 'train'` molecules, grouped by cluster: four fifths are the training set
and every model is scored on the *same* held-out `ds == 'test'` molecules. The
held-out fifth is unused, as it is for LightGBM and Monroe. MolPAIR's own inner
five folds, the out-of-fold predictions its relation context is built from, are
taken inside the four fifths and never see the test set.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# --- data sets -----------------------------------------------------------
#     python 02_run_molpair.py                         # expansion
#     ADME_DATASET=biogen python 02_run_molpair.py     # Biogen


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

# Where the folds, the molecule table and the imported arms come from.
# 00_import_baselines.py copies them in once.
SOURCE_REPO = Path(
    os.environ.get("ADME_SOURCE_REPO", PROJECT_DIR.parent / "expansion-ml-comparison")
).expanduser()

DATA_DIR = PROJECT_DIR / "data" / DATASET
FOLD_DIR = PROJECT_DIR / "folds" / DATASET
PRED_DIR = PROJECT_DIR / "predictions" / DATASET
# Per-molecule bookkeeping from every MolPAIR fold: how many anchors were
# accepted, the nearest one's similarity, the gate and the correction. Kept out
# of predictions/ so 03_collect_metrics.py does not read it as another arm.
DIAG_DIR = PROJECT_DIR / "diagnostics" / DATASET
RESULTS_DIR = PROJECT_DIR / "results" / DATASET
LOG_DIR = PROJECT_DIR / "logs" / DATASET

FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

MASTER_CSV = DATA_DIR / "master.csv"
FOLD_CSV = FOLD_DIR / "fold_assignments.csv"
# Everything MolPAIR reads about a molecule, in master.csv row order, written by
# 01_featurize.py. None of it depends on a label, so it is computed once.
FEATURES_NPZ = DATA_DIR / "molpair_features.npz"

# CheMeleon's pre-trained message passing, where chemprop keeps it after
# `--from-foundation CHEMELEON`, and where it fetches it from otherwise.
CHEMELEON_CKPT = Path(os.environ.get("CHEMELEON_CKPT", "~/.chemprop/chemeleon_mp.pt")).expanduser()
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"

PREDICTIONS_PARQUET = RESULTS_DIR / "predictions_all.parquet"
FOLD_METRICS_CSV = RESULTS_DIR / "fold_metrics.csv"
DIAGNOSTICS_PARQUET = RESULTS_DIR / "diagnostics_all.parquet"
# Wall-clock per fold, one row per (method, endpoint, repeat, fold).
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
        return self.results / "diagnostics_all.parquet"

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
GLOBAL_METHOD = "molpair_global"
MOLPAIR_METHOD = "molpair"

# The arms imported from the source comparison, and whether each writes one
# prediction file per endpoint (single-task) or one per assay family.
BASELINE_IS_SINGLE_TASK = {
    LGBM_METHOD: True,
    "chemprop_st": True,
    "chemprop": False,
    CHEMELEON_METHOD: False,
    MONROE_METHOD: True,
}
BASELINE_METHODS = list(BASELINE_IS_SINGLE_TASK)

METHOD_LABELS = {
    LGBM_METHOD: "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    CHEMELEON_METHOD: "ChemProp + CheMeleon",
    MONROE_METHOD: "Monroe + TabPFN 3",
    GLOBAL_METHOD: "CheMeleon + TabPFN 3",
    MOLPAIR_METHOD: "MolPAIR",
}

ALL_METHODS = list(METHOD_LABELS)

# What the report compares. The two from-scratch ChemProp arms are carried for
# the cross-study overview, which refuses to pool a study that does not
# reproduce the reference arms exactly; they are not this page's question.
METHODS = [LGBM_METHOD, CHEMELEON_METHOD, MONROE_METHOD, GLOBAL_METHOD, MOLPAIR_METHOD]

# --- MolPAIR -------------------------------------------------------------
# Every value is the paper's default (Table 4 and Sections 3-4), not tuned here.
MOLPAIR = {
    # Global context: frozen CheMeleon, 2,048 features, at most 2,000 kept after
    # median imputation, constant filtering and variance selection.
    "global_max_features": 2000,
    # Relation context: signed and absolute RDKit2D differences plus eight
    # auxiliary signals, (2000 - 8) / 2 = 996 descriptors at most.
    "relation_max_descriptors": 996,
    # Both TabPFN 3 models, eight estimators each.
    "n_estimators": 8,
    # The relation model's seed is the global one plus this.
    "relation_seed_offset": 104_729,
    # Out-of-fold predictions from five folds inside the training set.
    "oof_folds": 5,
    # Up to eight non-self anchors per training molecule, inside its OOF fold.
    "train_anchors": 8,
    # Relation context of at most 8,192 pair rows, deterministically subsampled.
    "relation_context": 8192,
    # Up to 32 anchors per query, Tanimoto on Morgan radius 2 at least 0.20.
    "query_anchors": 32,
    "anchor_min_similarity": 0.20,
    # Reliability gate threshold tau.
    "gate_tau": 0.20,
    # Pair rows per relation-model prediction call.
    "relation_batch": 8192,
    # Morgan radius-2 bits for anchor similarity. The paper gives the radius,
    # not the length; 2,048 is the length used everywhere else here.
    "morgan_radius": 2,
    "morgan_bits": 2048,
    # (Q0.9 - Q0.1) / 2.563 is a standard deviation under a normal.
    "quantile_to_sd": 2.563,
}

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


def diag_csv(name: str, repeat: int, fold: int) -> Path:
    return DIAG_DIR / f"{name}_r{repeat}_f{fold}.csv"


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, DIAG_DIR, RESULTS_DIR, LOG_DIR,
                 FIGURE_DIR, TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)
