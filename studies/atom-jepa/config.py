"""Shared configuration for the Atom-JEPA comparison.

Atom-JEPA (Petersen, Tirsgaard, Cornet, Jordahn and Schmidt, arXiv 2610.08400)
pre-trains a 3D equivariant graph network, EquiformerV3 with 39 M parameters,
on 19 M Uni-Mol conformers by predicting the latent representation of one part
of a molecule from another, at the level of single atoms and of whole
substructures. No labels are used in pre-training. For ADMET it is then
fine-tuned, multitask, one small head per endpoint, on ten conformers per
molecule. The paper reports it best or statistically tied on all nine
ExpansionRx endpoints and all six Biogen ones, ahead of CKERMT, KERMT, Chemprop
and every fingerprint baseline it ran.

It did not run Monroe. This study puts it on the molecules, folds and
statistics everything else here uses, against Monroe + TabPFN 3.

  imported from expansion-ml-comparison, not re-run
    lgbm           LightGBM on Morgan count fingerprints -- the floor
    chemprop_st    single-task D-MPNN
    chemprop       multitask D-MPNN
    chemeleon      multitask D-MPNN initialised from CheMeleon, then fine-tuned
    monroe         Monroe + TabPFN 3, the strongest arm in the source comparison

  fitted here
    atomjepa_ft      the paper's own recipe: the pre-trained encoder fine-tuned
                     with the authors' code at the authors' hyperparameters,
                     by 04_run_finetune.py
    atomjepa_tabpfn  the frozen encoder, all eight layers mean-pooled, read by
                     the same TabPFN 3 call as `monroe`, by 03_run_tabpfn.py

`atomjepa_ft` is the claim the paper makes. `atomjepa_tabpfn` is the control
that prices it: against `monroe` it changes the representation and nothing
else, and against `atomjepa_ft` it changes how the representation is used.

Evaluation protocol
-------------------
Unchanged from the source repository. The `ds` column fixes a train/test split.
The 25 replicates come from 5 repeats of 5-fold `GroupKFold` over the
`ds == 'train'` molecules, grouped by cluster: four fifths are the training set
and every model is scored on the *same* held-out `ds == 'test'` molecules. The
fine-tuned arm uses the held-out fifth to pick its best epoch, exactly as the
ChemProp arms use it for early stopping; the TabPFN arm, like Monroe, has no
training loop and leaves it unused.
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
    os.environ.get("ADME_SOURCE_REPO", PROJECT_DIR.parent / "expansion-ml-comparison")
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
ATOMJEPA_CKPT = "molecules"            # atom-jepa/atom-jepa on the Hub, Uni-Mol 19M

# Ten conformers per molecule, made once by 01_conformers.py with the authors'
# own builder; one pickle per data set, keyed by SMILES.
CONFORMER_DIR = DATA_DIR / "conformers"
# The frozen all-layer embedding of conformer 0, in master.csv row order,
# written by 02_embed.py. The encoder never sees a label, so it is made once.
EMBED_NPZ = DATA_DIR / "atomjepa_embeddings.npz"

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
FT_METHOD = "atomjepa_ft"
TABPFN_METHOD = "atomjepa_tabpfn"

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
    TABPFN_METHOD: "Atom-JEPA + TabPFN 3",
    FT_METHOD: "Atom-JEPA fine-tuned",
}

ALL_METHODS = list(METHOD_LABELS)

# What the report compares. The two from-scratch ChemProp arms are carried for
# the cross-study overview, which refuses to pool a study that does not
# reproduce the reference arms exactly; they are not this page's question.
METHODS = [LGBM_METHOD, CHEMELEON_METHOD, MONROE_METHOD, TABPFN_METHOD, FT_METHOD]

# --- Atom-JEPA -----------------------------------------------------------
# The fine-tuning recipe is the authors' conf/finetune_expansionrx.yaml (and the
# identical conf/finetune_biogen_adme.yaml) taken as it is, values the paper
# says were selected on the Biogen cluster validation split. These overrides
# are the only departures, and none of them changes what is trained. `++`
# sets a key whether or not the data set's yaml declares it:
FINETUNE_OVERRIDES = [
    "++wandb.enabled=false",
    # Our endpoints are already on a log scale (log10(x + 1), or LogD as is).
    # The ExpansionRx default would log1p them a second time.
    "++finetune.log_transform_tasks=[]",
    # Off: the KERMT log10 report scale is a second set of metrics on their
    # raw units, which these data are not in. It never touches training.
    "++finetune.report_scale=null",
    # Score and select on the conformer-averaged prediction -- what the paper
    # calls "Atom-JEPA x10 conf" and what is written to disk -- rather than on
    # the mean of per-conformer errors.
    "++finetune.conformer_eval_mode=ensemble",
    # The best-validation EMA weights are kept in memory either way; writing
    # two 0.5 GB checkpoints per fold buys nothing here.
    "++finetune.save_best=false",
    "++finetune.save_last=false",
    # One model per fold, seeded per fold, instead of five seeds on one split.
    "++finetune.seeds=[0]",
    "++finetune.plot_every=0",
]

# The frozen arm. "All layers" is the better of the paper's two frozen
# variants on ExpansionRx under both of its heads (Table 18): the l = 0 output
# of each of the eight blocks, mean-pooled over atoms, 8 x 256 = 2,048
# features, from conformer 0 of the same cache the fine-tuned arm reads. Run
# in fp32, as the authors' own frozen baseline is: bf16 moves the last block's
# features by 44% against fp32 on this hardware.
EMBED_LAYERS = "all"
EMBED_DIM = 8 * 256
# TabPFN 3's limit. 03_run_tabpfn.py keeps the highest-variance columns, chosen
# on each fold's training rows.
TABPFN_MAX_FEATURES = 2000

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


def diag_json(repeat: int, fold: int) -> Path:
    return DIAG_DIR / f"{FT_METHOD}_r{repeat}_f{fold}.json"


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, DIAG_DIR, RESULTS_DIR, LOG_DIR,
                 FIGURE_DIR, TABLE_DIR, CONFORMER_DIR):
        path.mkdir(parents=True, exist_ok=True)
