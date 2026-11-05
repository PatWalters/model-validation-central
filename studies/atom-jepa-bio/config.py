"""Shared configuration for the Atom-JEPA-Bio comparison.

Atom-JEPA-Bio (Petersen, Tirsgaard, Larsen, Cornet, Jordahn and Schmidt, Nov 2026)
continues the released Atom-JEPA `molecules` encoder with a second pre-training
stage: the same self-supervised latent-prediction objectives, plus a supervised
focal loss over 1,089 PCBA bioassays on ~1.5 M molecules, prepared with Monroe's
own pipeline. The architecture, 39 M-parameter EquiformerV3, and the ADMET
fine-tuning recipe are unchanged. The authors evaluated it on this repository's
protocol and report it best or tied on 34 of 45 endpoint x metric combinations,
against 18 for Monroe + TabPFN 3.

Because only the encoder weights differ from the atom-jepa study, the question
this study can answer cleanly is what the bioassay stage buys: the same recipe,
the same folds, one checkpoint swapped.

  imported, not re-run
    lgbm             LightGBM on Morgan counts -- the floor
    chemprop_st      single-task D-MPNN           (site overview only)
    chemprop         multitask D-MPNN             (site overview only)
    chemeleon        D-MPNN initialised from CheMeleon, fine-tuned
    monroe           Monroe + TabPFN 3
    atomjepa_ft      stage-1 Atom-JEPA fine-tuned, from the atom-jepa study

  fitted here
    bio_ft           the paper's arm: the bio encoder fine-tuned with the
                     authors' code and recipe, 04_run_finetune.py
    bio_tabpfn       the frozen bio encoder read by Monroe's TabPFN 3 call
    atomjepa_tabpfn  the frozen stage-1 encoder, refit here by the same code
                     as bio_tabpfn so the two frozen arms share conformers,
                     machine and library versions

  reproduction check, not a compared arm
    atomjepa_ft_rerun  stage-1 fine-tuning re-run on this study's GPU for a
                       few folds, to measure how far a change of machine
                       alone moves the imported atomjepa_ft

Evaluation protocol
-------------------
Unchanged from the source repository. The `ds` column fixes a train/test split.
The 25 replicates come from 5 repeats of 5-fold `GroupKFold` over the
`ds == 'train'` molecules, grouped by cluster: four fifths are the training set
and every model is scored on the *same* held-out `ds == 'test'` molecules. The
fine-tuned arms use the held-out fifth to pick the best epoch.
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

# Where the folds, the molecule table and the imported arms come from.
# 00_import_baselines.py copies them in once.
SOURCE_REPO = Path(
    os.environ.get("ADME_SOURCE_REPO", PROJECT_DIR.parent / "kermt")
).expanduser()

DATA_DIR = PROJECT_DIR / "data" / DATASET
FOLD_DIR = PROJECT_DIR / "folds" / DATASET
PRED_DIR = PROJECT_DIR / "predictions" / DATASET
# Per-fold bookkeeping for the fine-tuned arm: best epoch, validation MAE,
# molecules dropped by conformer generation. Kept out of predictions/ so
# 05_collect_metrics.py does not read it as another arm.
DIAG_DIR = PROJECT_DIR / "diagnostics" / DATASET
RESULTS_DIR = PROJECT_DIR / "results" / DATASET
LOG_DIR = PROJECT_DIR / "logs" / DATASET

FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

MASTER_CSV = DATA_DIR / "master.csv"
FOLD_CSV = FOLD_DIR / "fold_assignments.csv"
# The authors' checkout, for their fine-tuning code and Hydra configs. The
# encoder weights come from the Hugging Face Hub on first use.
ATOMJEPA_HOME = Path(
    os.environ.get("ATOMJEPA_HOME", Path.home() / "software" / "atom-jepa")
).expanduser().resolve()

# Ten conformers per molecule, made once by 01_conformers.py with the authors'
# own builder; one pickle per data set, keyed by SMILES.
CONFORMER_DIR = DATA_DIR / "conformers"
# The atom-jepa study's caches, built on apollo under RDKit 2026.03.6 and read
# by every one of its fine-tuned folds on both machines. 01_conformers.py
# refuses any other.
CONFORMER_MD5 = {
    "biogen": "3636e982661535ea0141343b23d547f3",
    "expansion": "993fdd0e8845b8d1eaf945df22821ca3",
}
# The frozen all-layer embeddings of conformer 0, in master.csv row order,
# one file per encoder, written by 02_embed.py: see embed_npz() below.

# Monroe's checkout, for the TabPFN wrapper the frozen arm shares with `monroe`.
MONROE_HOME = Path(
    os.environ.get("MONROE_HOME", Path.home() / "software" / "monroe")
).expanduser().resolve()

PREDICTIONS_PARQUET = RESULTS_DIR / "predictions_all.parquet"
FOLD_METRICS_CSV = RESULTS_DIR / "fold_metrics.csv"
DIAGNOSTICS_CSV = RESULTS_DIR / "diagnostics.csv"
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
BASE_FT_METHOD = "atomjepa_ft"          # stage 1, imported
BASE_TABPFN_METHOD = "atomjepa_tabpfn"  # stage 1 frozen, refit here
FT_METHOD = "bio_ft"                    # the paper's arm
TABPFN_METHOD = "bio_tabpfn"
RERUN_METHOD = "atomjepa_ft_rerun"      # reproduction check only

# The arms imported from earlier studies, and whether each writes one
# prediction file per endpoint (single-task) or one per assay family.
# atomjepa_ft writes one file per endpoint although it is one multitask model.
BASELINE_IS_SINGLE_TASK = {
    LGBM_METHOD: True,
    "chemprop_st": True,
    "chemprop": False,
    CHEMELEON_METHOD: False,
    MONROE_METHOD: True,
    BASE_FT_METHOD: True,
}
BASELINE_METHODS = list(BASELINE_IS_SINGLE_TASK)

# Which released encoder each fitted arm reads (atom-jepa/atom-jepa on the Hub).
CKPT_OF = {
    FT_METHOD: "bio",
    TABPFN_METHOD: "bio",
    BASE_TABPFN_METHOD: "molecules",
    RERUN_METHOD: "molecules",
}
FT_METHODS = [FT_METHOD, RERUN_METHOD]
TABPFN_METHODS = [TABPFN_METHOD, BASE_TABPFN_METHOD]
# The Hub revision both encoders were downloaded at, recorded for the README.
HF_REVISION_USED = "bf14d025fc864d689dbad1ac9c55b8b7054af11e"
ATOMJEPA_COMMIT = "64d539b"

METHOD_LABELS = {
    LGBM_METHOD: "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    CHEMELEON_METHOD: "ChemProp + CheMeleon",
    MONROE_METHOD: "Monroe + TabPFN 3",
    BASE_TABPFN_METHOD: "Atom-JEPA + TabPFN 3",
    BASE_FT_METHOD: "Atom-JEPA fine-tuned",
    TABPFN_METHOD: "Atom-JEPA-Bio + TabPFN 3",
    FT_METHOD: "Atom-JEPA-Bio fine-tuned",
    RERUN_METHOD: "Atom-JEPA fine-tuned (re-run)",
}

ALL_METHODS = list(METHOD_LABELS)

# What the report compares. The two from-scratch ChemProp arms are carried for
# the cross-study overview only, and the re-run is a reproduction check.
METHODS = [LGBM_METHOD, CHEMELEON_METHOD, MONROE_METHOD, BASE_TABPFN_METHOD,
           BASE_FT_METHOD, TABPFN_METHOD, FT_METHOD]

# --- Atom-JEPA -----------------------------------------------------------
# The fine-tuning recipe is the authors' conf/finetune_expansionrx.yaml (and the
# identical conf/finetune_biogen_adme.yaml), unchanged between commit bab0853,
# which the atom-jepa study ran, and 64d539b, which this one runs. These
# overrides are the atom-jepa study's, and the authors' own eval_mvc.py uses
# the same list; none of them changes what is trained. `++` sets a key whether
# or not the data set's yaml declares it:
FINETUNE_OVERRIDES = [
    "++wandb.enabled=false",
    # Our endpoints are already on a log scale (log10(x + 1), or LogD as is).
    "++finetune.log_transform_tasks=[]",
    # A second set of metrics on raw assay units; never touches training.
    "++finetune.report_scale=null",
    # Select and score on the conformer-averaged prediction ("x10 conf").
    "++finetune.conformer_eval_mode=ensemble",
    "++finetune.save_best=false",
    "++finetune.save_last=false",
    "++finetune.seeds=[0]",
    "++finetune.plot_every=0",
]

# The frozen arms: the l = 0 output of each of the eight blocks, mean-pooled
# over atoms, 2,048 features, from conformer 0, in fp32 -- exactly the atom-jepa
# study's frozen arm, for both encoders.
EMBED_LAYERS = "all"
EMBED_DIM = 8 * 256
TABPFN_MAX_FEATURES = 2000


def embed_npz(ckpt: str) -> Path:
    return DATA_DIR / f"{ckpt}_embeddings.npz"


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


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, DIAG_DIR, RESULTS_DIR, LOG_DIR,
                 FIGURE_DIR, TABLE_DIR, CONFORMER_DIR):
        path.mkdir(parents=True, exist_ok=True)
