"""Shared configuration for the 5x5 cross-validated method comparison.

Six modelling approaches are compared on the nine endpoints in
`expansion_log_scaled.csv`:

  lgbm         LightGBM regression on Morgan count fingerprints, one model per endpoint
  chemprop_st  the same D-MPNN from scratch, one model per endpoint
  chemprop     multitask D-MPNN, message passing initialised from scratch
  chemeleon    the same multitask D-MPNN, message passing initialised from the
               CheMeleon foundation model (Burns et al., J. Chem. Inf. Model. 2026,
               doi:10.1021/acs.jcim.6c01546)
  megacl       MEGA-CL fine-tuned from the authors' checkpoint (arXiv 2607.24314)
  monroe       Monroe's frozen encoder plus TabPFN in context, nothing trained
               downstream (arXiv 2608.18982)
  moljepa      Mol-JEPA's frozen multimodal encoder plus TabICL, likewise nothing
               trained downstream (arXiv 2608.22642)

The chemprop side of the pipeline follows the setup in
~/software/applicability/UNIQUE/scripts/config.py: the same endpoint groups, the
same hyperparameters, the same accelerator choice, and the same cluster-aware
handling of the validation split.

Evaluation protocol
-------------------
The `ds` column of the raw file fixes a train/test split. The 25 replicate models
per method come from 5 repeats of 5-fold `GroupKFold` over the `ds == 'train'`
molecules, grouped by `cluster`: four fifths are the training set, the held-out
fifth is the validation set used for early stopping, and every model is scored on
the *same* held-out `ds == 'test'` molecules. Every method sees identical
molecules in every fold, so a difference in the metrics is a difference in the
method.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# --- data sets -----------------------------------------------------------
# The same seven methods and the same protocol are run over two collections.
# Which one is active is read from the environment once, at import, so every
# script picks it up without threading a flag through its internals:
#
#     python 02_run_lightgbm.py                      # expansion, the default
#     ADME_DATASET=biogen python 02_run_lightgbm.py  # the Biogen set
#
# Everything a data set needs to differ in lives here. Paths are namespaced by
# name, so the two never write into each other's folds or predictions.


@dataclass(frozen=True)
class DataSet:
    """One collection of molecules and endpoints, and how to group them."""

    name: str
    raw: str                       # the source table, at the project root
    label: str                     # how the report names it
    targets: list[str]
    groups: dict[str, list[str]]

    @property
    def group_of(self) -> dict[str, str]:
        return {t: g for g, ts in self.groups.items() for t in ts}


DATASETS = {
    # Nine endpoints from the OpenADMET-ExpansionRx blind challenge, put on a
    # log10(x+1) scale. The `ds` column ships with the file.
    "expansion": DataSet(
        name="expansion",
        raw="expansion_log_scaled.csv",
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
    # Six endpoints on 3,521 commercially sourced compounds, released by Biogen
    # (Fang et al., J. Chem. Inf. Model. 2023). Already log-transformed at
    # source. It carries no train/test split, so 00b_prepare_biogen.py builds
    # one the same way the rest of the pipeline splits folds: whole BitBIRCH
    # clusters held out, to the same 30% test fraction the ExpansionRx file has.
    "biogen": DataSet(
        name="biogen",
        raw="biogen_adme_3521.csv",
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
            # One endpoint on its own is a degenerate multitask group: for
            # permeability the multitask arm is the single-task arm with a
            # different name. Said plainly in the report rather than hidden by
            # folding MDR1-MDCK efflux in with assays it has nothing to do with.
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

# Which set of methods the report covers. Read here so the paths below can
# namespace the figures; validated against COMPARISONS once those are defined.
#
#     python 05_report.py                          # the seven-method report
#     ADME_COMPARISON=trimole python 05_report.py  # the five-method one
COMPARISON = os.environ.get("ADME_COMPARISON", "foundation")

# --- paths ---------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

RAW_CSV = PROJECT_DIR / ACTIVE.raw

DATA_DIR = PROJECT_DIR / "data" / DATASET
FOLD_DIR = PROJECT_DIR / "folds" / DATASET
PRED_DIR = PROJECT_DIR / "predictions" / DATASET
RESULTS_DIR = PROJECT_DIR / "results" / DATASET
LOG_DIR = PROJECT_DIR / "logs" / DATASET
SCRATCH_DIR = PROJECT_DIR / "scratch" / DATASET  # chemprop model dirs, deleted after use

# Predictions and per-fold metrics are shared: they describe a method on a data
# set, not a report. Figures and tables belong to one comparison, so every report
# but the default gets its own subdirectory and none of them overwrite each other.
REPORT_DIR = RESULTS_DIR if COMPARISON == "foundation" else RESULTS_DIR / COMPARISON

FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
# Control runs that are not methods in the comparison. Kept out of PRED_DIR so
# 04_collect_metrics.py does not sweep them into the figures.
SENSITIVITY_DIR = RESULTS_DIR / "sensitivity"

# The cleaned data set: the raw file minus unparseable SMILES and rows with no
# measured endpoint, with the fold assignments joined on. Written by step 1.
MASTER_CSV = DATA_DIR / "master.csv"
FOLD_CSV = FOLD_DIR / "fold_assignments.csv"
FINGERPRINT_NPY = DATA_DIR / "morgan_counts.npy"
# One 720-d Monroe embedding per molecule, in master.csv row order. Computed once
# by 09_run_monroe.py --embed and reused by all 225 of its folds.
MONROE_NPZ = DATA_DIR / "monroe_embeddings.npz"
# One 512-d Mol-JEPA CLS token per molecule, same convention.
MOLJEPA_NPZ = DATA_DIR / "moljepa_embeddings.npz"

# Trimole-Hybrid's four molecular views, each cached once per data set in
# master.csv row order. The three encoders go in one file because they are always
# read together; the chemistry priors are separate because they are much larger
# and are built on the CPU while the encoders want the GPU.
TRIMOLE_EMBED_NPZ = DATA_DIR / "trimole_embeddings.npz"
TRIMOLE_CHEM_NPZ = DATA_DIR / "trimole_chem.npz"
# KPGT runs in its own environment (DGL pins an old torch), so its features are
# written by a separate step and merged into TRIMOLE_EMBED_NPZ.
TRIMOLE_KPGT_NPY = DATA_DIR / "trimole_kpgt.npy"

PREDICTIONS_PARQUET = RESULTS_DIR / "predictions_all.parquet"
FOLD_METRICS_CSV = RESULTS_DIR / "fold_metrics.csv"


@dataclass(frozen=True)
class Paths:
    """Where one data set's outputs live, whichever one is currently active.

    Every script works on the active data set through the module-level constants
    above. The report is the exception: it puts both collections on one page, so
    it needs to reach the other one's tables and figures without re-importing
    this module under a different environment.
    """

    dataset: DataSet
    results: Path
    comparison: str = "foundation"

    @property
    def report(self) -> Path:
        """Where one comparison's figures and tables live, mirroring REPORT_DIR."""
        return self.results if self.comparison == "foundation" else self.results / self.comparison

    @property
    def figures(self) -> Path:
        return self.report / "figures"

    @property
    def tables(self) -> Path:
        return self.report / "tables"

    @property
    def sensitivity(self) -> Path:
        return self.results / "sensitivity"

    @property
    def fold_metrics(self) -> Path:
        return self.results / "fold_metrics.csv"

    @property
    def predictions(self) -> Path:
        return self.results / "predictions_all.parquet"

    @property
    def master(self) -> Path:
        return PROJECT_DIR / "data" / self.dataset.name / "master.csv"


def paths(name: str, comparison: str | None = None) -> Paths:
    """One data set's outputs, for a comparison (the active one by default)."""
    if name not in DATASETS:
        raise SystemExit(f"unknown data set {name!r} -- choose from {', '.join(DATASETS)}")
    return Paths(DATASETS[name], PROJECT_DIR / "results" / name, comparison or COMPARISON)

# --- data columns --------------------------------------------------------
SMILES_COL = "SMILES"
ID_COL = "Name"
SET_COL = "ds"           # 'train' / 'test' in the raw file
CLUSTER_COL = "cluster"  # groups the CV folds and the validation split
SPLIT_COL = "split"      # 'train' / 'val' / 'test' -- consumed by chemprop

# --- endpoints -----------------------------------------------------------
# Every endpoint of the active data set, in file order. Used for reporting and
# for the LightGBM models, which are single-task.
TARGET_COLS = list(ACTIVE.targets)

# The chemprop models are multitask over *related* endpoints, the assay families
# defined with the data set above. Grouping this way keeps multitask transfer
# where it is plausible -- human and rodent microsomal stability, permeability
# with permeability -- without forcing unrelated endpoints through shared weights.
TARGET_GROUPS = {group: list(targets) for group, targets in ACTIVE.groups.items()}

GROUPS = list(TARGET_GROUPS)
GROUP_OF = dict(ACTIVE.group_of)

assert sorted(GROUP_OF) == sorted(TARGET_COLS), "TARGET_GROUPS must cover TARGET_COLS exactly"

# --- methods -------------------------------------------------------------
# The two chemprop variants differ *only* in how the message-passing block is
# initialised, so a difference between them is a difference in the representation.
#
# `accelerator` is a wall-clock choice, not a modelling one: CheMeleon's message
# passing is d_h=2048 / depth=6 against the default 300/3, roughly 15x the work per
# epoch, which is worth moving to the GPU.
VARIANTS = {
    "chemprop": {
        "label": "ChemProp multi-task",
        "from_foundation": None,
        "accelerator": "cpu",
        "single_task": False,
    },
    "chemeleon": {
        "label": "ChemProp + CheMeleon",
        "from_foundation": "CHEMELEON",
        "accelerator": "mps",
        "single_task": False,
    },
    # The control that separates architecture from multitask transfer: the same
    # from-scratch D-MPNN, one model per endpoint instead of one per assay family.
    # Against `chemprop` it isolates what multitask training buys; against `lgbm` it
    # is a like-for-like single-task comparison of a GNN with a fingerprint model.
    "chemprop_st": {
        "label": "ChemProp single-task",
        "from_foundation": None,
        "accelerator": "cpu",
        "single_task": True,
    },
}

VARIANT_NAMES = list(VARIANTS)

LGBM_METHOD = "lgbm"

# MEGA-CL is not a chemprop variant, so it has no entry in VARIANTS. It is a
# single-task graph contrastive-learning foundation model (arXiv 2607.24314),
# fine-tuned one endpoint at a time from the authors' pre-trained checkpoint by
# 08_run_megacl.py, and it reads the same st_<endpoint>_r{r}_f{f}.csv folds as
# the single-task chemprop arm.
MEGACL_METHOD = "megacl"

# Monroe (arXiv 2608.18982) is a graph-transformer foundation model whose downstream
# head is not trained at all: the pre-trained encoder is frozen, every molecule
# becomes one 720-d embedding, and TabPFN predicts each endpoint in context from
# the labelled training embeddings. That makes it single-task like `lgbm`, and it
# reads the same rows -- 09_run_monroe.py mirrors the LightGBM fit/test masks
# exactly rather than the st_ fold files, which amounts to the same molecules.
MONROE_METHOD = "monroe"

# Mol-JEPA (arXiv 2608.22642) is the same shape of arm as Monroe: a frozen
# multimodal encoder, one 512-d CLS token per molecule, and a tabular in-context
# model on top. The head is TabICL, which is what the authors recommend on their
# model card. 10_run_moljepa.py also runs a TabPFN head as a control, so the
# gap against Monroe can be split into representation and head.
MOLJEPA_METHOD = "moljepa"

# Trimole-Hybrid (Luo et al., Bioinformatics 2026, doi:10.1101/2026.08.24.746660) is
# not a model but a selection framework, so it is the one arm here whose "method"
# is a procedure rather than an architecture. For each task it builds a pool of
# candidate predictors over four molecular views -- a ChemBERTa sequence encoder,
# a KPGT graph encoder, a UniMol 3D encoder and blocks of classical chemistry
# priors -- fits every candidate, keeps whichever scores best on the validation
# split, and only then touches the test set.
#
# Unlike MEGA-CL, Monroe and Mol-JEPA, this arm is *not* the authors' code. Their
# release is an audit package: the paths are placeholders, no weights or cached
# embeddings are included, every script is wired to the TDC benchmark's directory
# layout, and LICENSE_PENDING.md reserves all rights. 12_run_trimole.py is
# therefore a reimplementation of the method from the paper and that source, using
# independently obtained checkpoints. The report says so wherever it names it.
TRIMOLE_METHOD = "trimole"

METHOD_LABELS = {
    LGBM_METHOD: "LightGBM + Morgan",
    "chemprop_st": VARIANTS["chemprop_st"]["label"],
    "chemprop": VARIANTS["chemprop"]["label"],
    "chemeleon": VARIANTS["chemeleon"]["label"],
    MEGACL_METHOD: "MEGA-CL",
    MONROE_METHOD: "Monroe + TabPFN",
    MOLJEPA_METHOD: "Mol-JEPA + TabICL",
    TRIMOLE_METHOD: "Trimole-Hybrid",
}

# Every method that can appear in predictions/. Collection works over all of them,
# so one sweep of 04_collect_metrics.py serves every report.
ALL_METHODS = list(METHOD_LABELS)

# A report covers a *subset* of them. The two questions being asked are different
# enough that putting eight methods on one page would answer neither: the first
# asks which pre-trained representation transfers, the second asks whether picking
# a model per endpoint beats committing to one architecture.
COMPARISONS = {
    "foundation": [
        LGBM_METHOD, "chemprop_st", "chemprop", "chemeleon",
        MEGACL_METHOD, MONROE_METHOD, MOLJEPA_METHOD,
    ],
    "trimole": [
        LGBM_METHOD, "chemprop_st", "chemprop", "chemeleon", TRIMOLE_METHOD,
    ],
}
DEFAULT_COMPARISON = "foundation"

if COMPARISON not in COMPARISONS:
    raise SystemExit(
        f"unknown ADME_COMPARISON {COMPARISON!r} -- choose from {', '.join(COMPARISONS)}"
    )

METHODS = list(COMPARISONS[COMPARISON])


def is_single_task(variant: str) -> bool:
    return VARIANTS[check_variant(variant)]["single_task"]


def units(variant: str) -> list[str]:
    """What one model covers: an endpoint for single-task, an assay family otherwise."""
    return TARGET_COLS if is_single_task(variant) else GROUPS


def unit_targets(variant: str, unit: str) -> list[str]:
    return [unit] if is_single_task(variant) else TARGET_GROUPS[unit]

# --- cross validation ----------------------------------------------------
N_REPEATS = 5
N_SPLITS = 5
RANDOM_SEED = 0xF00D

REPEATS = list(range(N_REPEATS))
FOLDS = list(range(N_SPLITS))


def fold_seed(repeat: int, fold: int) -> int:
    """A distinct, reproducible seed for each of the 25 fold models."""
    return RANDOM_SEED + 100 * repeat + fold


# --- model hyperparameters ----------------------------------------------
# chemprop: UNIQUE's settings, with the 10-member ensemble replaced by the 25 CV
# replicates -- each fold trains a single network.
EPOCHS = 50
BATCH_SIZE = 64
ENSEMBLE_SIZE = 1
REMOVE_LIGHTNING_CHECKPOINTS = True

# LightGBM: Morgan count fingerprints, library defaults otherwise.
FP_RADIUS = 2
FP_SIZE = 2048

# --- tidy prediction schema ---------------------------------------------
# Every method writes the same columns, one row per (molecule, endpoint, fold).
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
# True where a larger value is a better model.
METRIC_HIGHER_IS_BETTER = {"r2": True, "spearman": True, "mae": False}
METRIC_LABELS = {"r2": "$R^2$", "spearman": "Spearman $\\rho$", "mae": "MAE"}


# --- helpers -------------------------------------------------------------
def check_variant(variant: str) -> str:
    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r} -- choose from {', '.join(VARIANTS)}")
    return variant


def accelerator(variant: str, override: str = None) -> str:
    """Which device to train on -- a wall-clock choice, not a modelling one.

    The UNIQUE pipeline verified that the accelerator does not affect the result
    (a control run of the permeability group agreed to four decimals between cpu
    and mps), so a fold trained on a CUDA machine is comparable with one trained
    here. A CUDA device is preferred wherever one is available, since CheMeleon's
    d_h=2048 / depth=6 message passing is roughly 15x the work per epoch of the
    default backbone.
    """
    if override:
        return override

    name = VARIANTS[check_variant(variant)]["accelerator"]
    import torch

    if torch.cuda.is_available():
        return "gpu"
    if name == "mps" and not torch.backends.mps.is_available():
        print(f"[{variant}] mps unavailable, falling back to cpu")
        return "cpu"
    return name


def fold_input(group: str, repeat: int, fold: int) -> Path:
    """The chemprop training file for one group and one fold.

    Holds the molecules with at least one of that group's endpoints measured and a
    `split` column of 'train' / 'val' / 'test'. Shared by both chemprop variants --
    the comparison rests on the models seeing identical molecules and splits.
    """
    return FOLD_DIR / f"{group}_r{repeat}_f{fold}.csv"


def st_fold_input(endpoint: str, repeat: int, fold: int) -> Path:
    """The chemprop training file for one endpoint and one fold, single-task.

    Holds only the molecules with that endpoint measured -- the same rows the
    LightGBM model of that fold is fit on -- carrying the fold's `split` column.
    """
    return FOLD_DIR / f"st_{endpoint}_r{repeat}_f{fold}.csv"


def variant_fold_input(variant: str, unit: str, repeat: int, fold: int) -> Path:
    if is_single_task(variant):
        return st_fold_input(unit, repeat, fold)
    return fold_input(unit, repeat, fold)


def pred_csv(method: str, name: str, repeat: int, fold: int) -> Path:
    """Tidy predictions for one method and fold.

    `name` is the endpoint for LightGBM and the endpoint group for chemprop, since
    a multitask model predicts its whole group in one pass. These files are the
    resume points: a run skips any combination whose file already exists.
    """
    return PRED_DIR / method / f"{name}_r{repeat}_f{fold}.csv"


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, RESULTS_DIR, LOG_DIR, FIGURE_DIR, TABLE_DIR,
                 SENSITIVITY_DIR):
        path.mkdir(parents=True, exist_ok=True)
