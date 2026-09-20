"""Shared configuration for the split-stringency comparison.

Four modelling approaches are compared on the ten ChEMBL pIC50 data sets and the
six splitting schemes of Lee, Moldagulov and Grzybowski (Chem. Eur. J. 2026,
doi:10.1002/chem.71208), on that paper's own folds:

  knn        k-nearest neighbours on ECFP4 Jaccard distance, k chosen on the
             validation split -- the paper's own baseline, re-implemented
  lgbm       LightGBM on Morgan count fingerprints, radius 2, 2048 bits
  chemeleon  ChemProp D-MPNN, message passing initialised from CheMeleon
             (Burns et al., J. Chem. Inf. Model. 2026, doi:10.1021/acs.jcim.6c01546)
  monroe35   Monroe's frozen encoder plus TabPFN 3.5 in context, nothing trained
             downstream (Banaszewski and Fitzgibbon, arXiv 2608.18982)

Two reference arms come with them:

  median     the paper's naive baseline: the training median for every test
             molecule. The claim being tested is that the models collapse
             towards this line, so it belongs on the axis rather than in a
             footnote.
  pub_*      the paper's own six arms -- k-NN, SVM, RF, XGBoost, MLP and a GNN --
             imported from its released per-fold metrics. Not re-run here, and
             named so on every figure and table.

Evaluation protocol
-------------------
The upstream repository ships, for each of the ten targets, five folds of each of
six splitting schemes as explicit train / validation / test index lists into its
own raw CSV. Those files are the protocol: nothing is re-split here. Every method
fits on the `train` indices of a fold and is scored on that fold's `test`
indices, so a difference in the metrics is a difference in the method.

The `val` indices are used only where a method needs them -- k for k-NN, early
stopping for ChemProp. LightGBM and Monroe have nothing to tune per fold and
nothing to stop early, so they leave the validation molecules unused rather than
training on molecules the other arms do not see. This is the same convention
02_run_lightgbm.py follows in `studies/expansion-ml-comparison`.

Ten targets x six schemes x five folds = 300 fold models per method.
"""

import json
import os
from pathlib import Path

# --- which comparison ----------------------------------------------------
# Which set of arms a report covers. Read here, before the paths, so those can
# namespace the figures and tables; validated against COMPARISONS further down,
# once they are defined.
#
#     python 08_report.py                            # the eleven-arm page
#     SPLIT_COMPARISON=core python 08_report.py      # only the arms fitted here
DEFAULT_COMPARISON = "published"
COMPARISON = os.environ.get("SPLIT_COMPARISON", DEFAULT_COMPARISON)

# --- paths ---------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

# What travels with the study: the upstream data and split indices, copied
# verbatim, plus the two per-fold metric files the published arms are read from.
UPSTREAM_DIR = PROJECT_DIR / "upstream"
UPSTREAM_RAW = UPSTREAM_DIR / "raw"
# Two copies of the splits travel with the study because the upstream release
# ships two and they disagree. See `read_split` below, and the README.
UPSTREAM_INDICES = UPSTREAM_DIR / "indices"   # the .pkl files -- authoritative
UPSTREAM_SPLITS = UPSTREAM_DIR / "splits"     # the .json files -- val sets only
UPSTREAM_2D_METRICS = UPSTREAM_DIR / "regressor_2d_results.csv"
UPSTREAM_KNN_METRICS = UPSTREAM_DIR / "knn_regressor_results.csv"

DATA_DIR = PROJECT_DIR / "data"
FOLD_DIR = PROJECT_DIR / "folds"
PRED_DIR = PROJECT_DIR / "predictions"
RESULTS_DIR = PROJECT_DIR / "results"
LOG_DIR = PROJECT_DIR / "logs"
SCRATCH_DIR = PROJECT_DIR / "scratch"   # chemprop model dirs, deleted after use

# Predictions and per-fold metrics describe a method on this data set, not a
# report, so they are shared. Figures and tables belong to one comparison: the
# multiplicity correction spans every pair on the page, so the same folds give
# different verdicts under an 11-arm page than under a 5-arm one, and the two
# must not overwrite each other. The default comparison keeps the top-level
# directory and every other one gets its own, as in `expansion-ml-comparison`.
REPORT_DIR = RESULTS_DIR if COMPARISON == DEFAULT_COMPARISON else RESULTS_DIR / COMPARISON

FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
# Two tables describe the data rather than a comparison -- the k-NN reproduction
# and the raw per-scheme means -- so they are written once and read by every
# report. Under the default comparison this is the same directory as TABLE_DIR.
SHARED_TABLE_DIR = RESULTS_DIR / "tables"

# One row per molecule per target, with a stable integer `idx` that is the row
# number in the upstream CSV -- which is what the split files index into.
MASTER_CSV = DATA_DIR / "master.csv"
# train / val / test membership for every (target, split, fold), long form.
FOLD_CSV = FOLD_DIR / "fold_assignments.csv"

# ECFP4 bit vectors (k-NN's Jaccard distance) and Morgan counts (LightGBM),
# both in master.csv row order.
ECFP_BITS_NPY = DATA_DIR / "ecfp4_bits.npy"
MORGAN_COUNTS_NPY = DATA_DIR / "morgan_counts.npy"
# One 720-d Monroe embedding per molecule, same convention.
MONROE_NPZ = DATA_DIR / "monroe_embeddings.npz"

PREDICTIONS_PARQUET = RESULTS_DIR / "predictions_all.parquet"
FOLD_METRICS_CSV = RESULTS_DIR / "fold_metrics.csv"

# --- data ----------------------------------------------------------------
# The ten ChEMBL target ids, as the upstream repository names them. Each is a
# single-target pIC50 regression problem, so there is no multitask grouping to
# make here -- every arm is single-task by construction.
TARGETS = [220, 230, 260, 262, 279, 284, 1865, 2409, 4005, 4822]

# What each target is, for the report. ChEMBL 30 target ids.
TARGET_NAMES = {
    220: "Acetylcholinesterase",
    230: "Serotonin transporter",
    260: "MAP kinase p38 alpha",
    262: "Glycogen synthase kinase-3 beta",
    279: "VEGFR-2",
    284: "Dipeptidyl peptidase IV",
    1865: "Histone deacetylase 6",
    2409: "Epoxide hydratase",
    4005: "PI3-kinase p110-alpha",
    4822: "Beta-secretase 1",
}

# The six schemes, from least to most stringent as the paper orders them. The
# names are the keys inside the upstream split JSON, unchanged.
SPLITS = ["random", "scaffold", "butina", "diverse_25", "umap", "time"]

SPLIT_LABELS = {
    "random": "Random",
    "scaffold": "Bemis-Murcko scaffold",
    "butina": "Butina cluster",
    "diverse_25": "Diverse (MaxMin 25%)",
    "umap": "UMAP cluster",
    "time": "Time",
}

FOLDS = [1, 2, 3, 4, 5]

# Upstream column names in the raw CSVs.
SMILES_COL = "nonstereo_aromatic_smiles"
Y_COL = "pIC50"
ID_COL = "chembl_cid"
YEAR_COL = "year"

# What this study calls them once master.csv is built. The framework's other
# studies use SMILES / Name, and chemprop is handed these column names.
OUT_SMILES_COL = "SMILES"
OUT_ID_COL = "Name"
SPLIT_COL = "split"        # 'train' / 'val' / 'test' -- consumed by chemprop

RANDOM_SEED = 0xF00D


def fold_seed(target: int, split: str, fold: int) -> int:
    """A distinct, reproducible seed for each of the 300 fold models."""
    return RANDOM_SEED + 1000 * (SPLITS.index(split) + 1) + 10 * fold + TARGETS.index(target)


# --- methods -------------------------------------------------------------
KNN_METHOD = "knn"
LGBM_METHOD = "lgbm"
CHEMELEON_METHOD = "chemeleon"
MONROE35_METHOD = "monroe35"
MEDIAN_METHOD = "median"

# The four methods actually fitted here.
RUN_METHODS = [KNN_METHOD, LGBM_METHOD, CHEMELEON_METHOD, MONROE35_METHOD]

# The paper's own arms, imported from its released per-fold metrics rather than
# re-run. `pub_` marks every one of them, in the data and on the page, because
# they are numbers taken on trust and the other arms are not.
PUBLISHED_MODELS = {
    "pub_knn": "knn",
    "pub_svm": "svm",
    "pub_rf": "rf",
    "pub_xgboost": "xgboost",
    "pub_mlp": "mlp",
    "pub_gnn": "gnn",
}
PUBLISHED_METHODS = list(PUBLISHED_MODELS)

METHOD_LABELS = {
    KNN_METHOD: "k-NN (Tanimoto)",
    LGBM_METHOD: "LightGBM + Morgan",
    CHEMELEON_METHOD: "ChemProp + CheMeleon",
    MONROE35_METHOD: "Monroe + TabPFN 3.5",
    MEDIAN_METHOD: "Training median",
    "pub_knn": "k-NN (published)",
    "pub_svm": "SVM (published)",
    "pub_rf": "RF (published)",
    "pub_xgboost": "XGBoost (published)",
    "pub_mlp": "MLP (published)",
    "pub_gnn": "GNN (published)",
}

ALL_METHODS = list(METHOD_LABELS)

# Which arms a report covers. `published` is the default and is what the page is
# written about: the four methods fitted here, the naive baseline, and the
# paper's own six arms, which sit on the same folds but are numbers rather than
# predictions. `core` drops those six, for a figure that shows only arms this
# study can vouch for end to end.
COMPARISONS = {
    "core": [KNN_METHOD, LGBM_METHOD, CHEMELEON_METHOD, MONROE35_METHOD, MEDIAN_METHOD],
    "published": [
        KNN_METHOD, LGBM_METHOD, CHEMELEON_METHOD, MONROE35_METHOD, MEDIAN_METHOD,
        *PUBLISHED_METHODS,
    ],
}
if COMPARISON not in COMPARISONS:
    raise SystemExit(
        f"unknown SPLIT_COMPARISON {COMPARISON!r} -- choose from {', '.join(COMPARISONS)}"
    )
METHODS = list(COMPARISONS[COMPARISON])

# --- metrics -------------------------------------------------------------
# MAE first: it is the paper's metric, and it is in pIC50 units, which is what
# makes the naive baseline readable as a line on the same axis.
METRICS = ["mae", "r2", "spearman"]
METRIC_HIGHER_IS_BETTER = {"mae": False, "r2": True, "spearman": True}
METRIC_LABELS = {"mae": "MAE", "r2": "$R^2$", "spearman": "Spearman $\\rho$"}

# --- model hyperparameters ----------------------------------------------
# k-NN: the upstream grid and the upstream estimator -- distance weighting on a
# precomputed Tanimoto distance matrix, k picked by validation MAE.
KNN_K_CHOICES = [1, 3, 5]

# Fingerprints. The upstream repository uses radius 2 / 2048 bits for both its
# own ECFP4 and, by coincidence, the same shape this framework's LightGBM arm
# uses, so one setting serves both. k-NN reads the bit vector, LightGBM the counts.
FP_RADIUS = 2
FP_SIZE = 2048

# chemprop: the settings `studies/expansion-ml-comparison` uses, so a CheMeleon
# fold here is the same model as a CheMeleon fold there.
EPOCHS = 50
BATCH_SIZE = 64
ENSEMBLE_SIZE = 1
REMOVE_LIGHTNING_CHECKPOINTS = True

# --- tidy prediction schema ---------------------------------------------
# Every method writes the same columns, one row per (molecule, target, split, fold).
PRED_COLUMNS = [
    "method",
    "target",
    "split",
    "fold",
    OUT_ID_COL,
    OUT_SMILES_COL,
    "y_true",
    "y_pred",
]


# --- helpers -------------------------------------------------------------
def upstream_csv(target: int) -> Path:
    return UPSTREAM_RAW / f"tid_{target}.csv"


def upstream_indices(target: int, fold: int) -> Path:
    return UPSTREAM_INDICES / f"regression_db_tid_{target}_splits_fold_{fold}.pkl"


def upstream_json(target: int, fold: int) -> Path:
    return UPSTREAM_SPLITS / f"tid_{target}_fold_{fold}.json"


def read_split(target: int, fold: int) -> dict:
    """One fold's six schemes, each a dict of train / val / test index lists.

    The upstream release ships the splits twice and the two copies disagree:
    `data/splits/indices/*.pkl` carries `{train,val,test}_ids` as sets, and
    `data/splits/*.json` carries them as lists. They agree on the random,
    scaffold and time schemes and differ on butina, umap and diverse_25 -- for
    umap the two test sets do not share a single molecule.

    The `.pkl` files are the ones the paper's own numbers were computed from.
    Re-running the released k-NN on the `.pkl` indices reproduces every
    published per-fold MAE to floating-point tolerance on five of the six
    schemes; on the `.json` indices, butina and umap do not reproduce at all.
    So the `.pkl` indices are the protocol here, and 07_collect_metrics.py keeps
    that reproduction as a standing check rather than a note.

    The sixth scheme needs one repair. `diverse_25` has an empty `val_ids` in
    every one of the fifty `.pkl` files: upstream carved its validation set out
    of the training pool inside the run and did not record it. The `.json` copy
    does record one, and on all fifty combinations it is a subset of the `.pkl`
    training pool while the two test sets are identical and the `.pkl` training
    pool is exactly the `.json` train plus val. Taking the validation molecules
    from the `.json` copy therefore contradicts nothing in the `.pkl` one, and
    it is what gives ChemProp something to stop early on and k-NN something to
    pick k on for that scheme. It is done here, once, for every arm, and
    00_prepare_data.py asserts all three of those relations before doing it.
    """
    import pickle

    with open(upstream_indices(target, fold), "rb") as handle:
        packed = pickle.load(handle)

    split = {
        scheme: {role: sorted(int(i) for i in roles[f"{role}_ids"])
                 for role in ("train", "val", "test")}
        for scheme, roles in packed.items()
    }

    # The one scheme whose validation set upstream did not record.
    if not split["diverse_25"]["val"]:
        with open(upstream_json(target, fold)) as handle:
            recorded = json.load(handle)["diverse_25"]
        pool = set(split["diverse_25"]["train"])
        val = sorted(int(i) for i in recorded["val"])
        train = sorted(int(i) for i in recorded["train"])
        if not set(val) <= pool:
            raise SystemExit(
                f"tid_{target} fold {fold}: the json diverse_25 validation set is not "
                "inside the pkl training pool, so it cannot be used to carve one"
            )
        if set(train) | set(val) != pool:
            raise SystemExit(
                f"tid_{target} fold {fold}: json diverse_25 train+val is not the pkl "
                "training pool"
            )
        if set(map(int, recorded["test"])) != set(split["diverse_25"]["test"]):
            raise SystemExit(
                f"tid_{target} fold {fold}: the two diverse_25 test sets disagree"
            )
        split["diverse_25"]["train"] = train
        split["diverse_25"]["val"] = val

    return split


def fold_input(target: int, split: str, fold: int) -> Path:
    """The chemprop training file for one target, scheme and fold.

    Carries the fold's molecules and a `split` column of 'train' / 'val' /
    'test'. The held-out validation molecules are chemprop's early-stopping set.
    """
    return FOLD_DIR / f"tid_{target}_{split}_f{fold}.csv"


def test_input(target: int, split: str, fold: int) -> Path:
    """A Name + SMILES file of one fold's test molecules, for `chemprop predict`.

    Unlike the other studies here the test set is not fixed across folds -- a
    five-fold scheme rotates it -- so there is one of these per fold rather than
    one per unit.
    """
    return FOLD_DIR / f"test_tid_{target}_{split}_f{fold}.csv"


def pred_csv(method: str, target: int, split: str, fold: int) -> Path:
    """Tidy predictions for one method and fold, and the resume point for a run."""
    return PRED_DIR / method / f"tid_{target}_{split}_f{fold}.csv"


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, RESULTS_DIR, LOG_DIR, FIGURE_DIR, TABLE_DIR,
                 SHARED_TABLE_DIR):
        path.mkdir(parents=True, exist_ok=True)
