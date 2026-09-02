"""Shared configuration for the multimodal fusion evaluation.

This repository asks the question Wasswa, Kajjumba and Ramsundar ask in
*Unimodal vs Multimodal Learning* (J. Chem. Inf. Model. 2026,
doi:10.1021/acs.jcim.6c01878), on two ADME collections instead of their fourteen
environmental-chemistry ones, and on the folds and splits of
`../expansion-ml-comparison` so the answer can be read against methods that have
already been scored on exactly the same molecules.

Their design has three axes.

  modality   RDKit descriptors, Mol2Vec embeddings, a supervised GNN's graph
             embedding, and a character BiGRU's SMILES embedding. Their fifth,
             MS2 fragmentation spectra, does not exist for ADME endpoints and is
             dropped; nothing else about the design changes.
  fusion     early, concatenating modality feature vectors into one design matrix
             before a single predictor, or late, stacking each modality's
             prediction into a meta-feature matrix for a meta-learner.
  learner    LightGBM, random forest, or AttentiveFP as the final predictor.

Four unimodal baselines and four modality combinations run under both fusion
strategies with all three learners. That is the whole grid, not a sample of it,
because the paper's claims are about the *shape* of the grid: that adding
modalities buys less than choosing the learner, that early and late separate only
in places, and that fusion helps calibration more than accuracy.

Reference methods
-----------------
`lgbm`, `chemprop_st`, `chemprop` and `chemeleon` are not run here. Their
predictions are copied in from `../expansion-ml-comparison`, where they were
produced on these same fold files, and they appear in the report as fixed
reference points: a fingerprint baseline, a single-task D-MPNN, a multitask
D-MPNN, and a D-MPNN initialised from a foundation model.

Evaluation protocol
-------------------
Unchanged from that project. The `ds` column fixes a train/test split; the 25
replicates are five repeats of a five-fold `GroupKFold` over the training
molecules, grouped by `cluster`; every model is scored on the same untouched test
set. Early stopping, where a method has any, uses the held-out fifth.

Uncertainty
-----------
The paper estimates epistemic uncertainty from three independently seeded models.
Here the five folds within a repeat play that role: each test molecule is
predicted five times, by five models fit on overlapping four-fifths of the same
training set, and the spread of those predictions is the ensemble sigma. That
gives five ensembles per configuration instead of one, and costs nothing extra --
which also means the four reference methods can be scored on calibration without
being rerun.
"""

import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

# --- data sets -----------------------------------------------------------
# Both collections, and the whole grid, are run over each. Which one is active is
# read from the environment once, at import:
#
#     python 20_run_fusion.py                      # expansion, the default
#     ADME_DATASET=biogen python 20_run_fusion.py  # the Biogen set


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

RAW_CSV = PROJECT_DIR / ACTIVE.raw

DATA_DIR = PROJECT_DIR / "data" / DATASET
FOLD_DIR = PROJECT_DIR / "folds" / DATASET
PRED_DIR = PROJECT_DIR / "predictions" / DATASET
RESULTS_DIR = PROJECT_DIR / "results" / DATASET
LOG_DIR = PROJECT_DIR / "logs" / DATASET
SCRATCH_DIR = PROJECT_DIR / "scratch" / DATASET

FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

MASTER_CSV = DATA_DIR / "master.csv"
FOLD_CSV = FOLD_DIR / "fold_assignments.csv"
FINGERPRINT_NPY = DATA_DIR / "morgan_counts.npy"

# The two modality caches that do not depend on the fold: one row per molecule in
# master.csv row order, computed once by 15_embed_modalities.py.
#   RDKit descriptors, the full RDKit descriptor list minus the ones that are
#   constant or non-finite anywhere in the collection.
RDKIT_NPZ = DATA_DIR / "rdkit_descriptors.npz"
#   Mol2Vec, 300-d, from the authors' pre-trained model_300dim.pkl.
MOL2VEC_NPZ = DATA_DIR / "mol2vec_embeddings.npz"

# The GNN and SMILES modalities are *supervised* representations: their encoders
# are trained on each fold's training molecules, so their embeddings depend on the
# endpoint and the fold and cannot be cached per molecule. They are written per
# fold instead, and reused by every configuration that names them.
EMBED_DIR = DATA_DIR / "fold_embeddings"

PREDICTIONS_PARQUET = RESULTS_DIR / "predictions_all.parquet"
FOLD_METRICS_CSV = RESULTS_DIR / "fold_metrics.csv"
UNCERTAINTY_CSV = RESULTS_DIR / "uncertainty.csv"
TIMING_CSV = RESULTS_DIR / "timings.csv"
SHAP_CSV = RESULTS_DIR / "modality_shap.csv"


@dataclass(frozen=True)
class Paths:
    """Where one data set's outputs live, whichever one is currently active.

    Every script works on the active data set through the module-level constants
    above. The report is the exception: it puts both collections on one page.
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
    def uncertainty(self) -> Path:
        return self.results / "uncertainty.csv"

    @property
    def timings(self) -> Path:
        return self.results / "timings.csv"

    @property
    def shap(self) -> Path:
        return self.results / "modality_shap.csv"

    @property
    def predictions(self) -> Path:
        return self.results / "predictions_all.parquet"

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
SPLIT_COL = "split"      # 'train' / 'val' / 'test' -- consumed by chemprop

TARGET_COLS = list(ACTIVE.targets)
TARGET_GROUPS = {group: list(targets) for group, targets in ACTIVE.groups.items()}
GROUPS = list(TARGET_GROUPS)
GROUP_OF = dict(ACTIVE.group_of)

assert sorted(GROUP_OF) == sorted(TARGET_COLS), "TARGET_GROUPS must cover TARGET_COLS exactly"

# --- the design grid -----------------------------------------------------
# Four modalities. The paper's fifth, MS2 fragmentation spectra, is an
# experimental measurement rather than something computed from a structure, and no
# ADME collection carries it. Everything else follows their Section 2.2.
MODALITIES = ["rdkit", "mol2vec", "gnn", "smiles"]

MODALITY_LABELS = {
    "rdkit": "RDKit",
    "mol2vec": "Mol2Vec",
    "gnn": "GNN",
    "smiles": "SMILES",
}
MODALITY_CODE = {"rdkit": "R", "mol2vec": "M", "gnn": "G", "smiles": "S"}

# The modality combinations the paper builds, in the order it introduces them:
# a bimodal model, two trimodal ones, and the four-modality model. RDKit and the
# GNN embedding are in every combination because those are the two the paper's
# ablation finds structurally necessary.
COMBOS = {
    "GR": ["gnn", "rdkit"],
    "GRM": ["gnn", "rdkit", "mol2vec"],
    "GRS": ["gnn", "rdkit", "smiles"],
    "GRMS": ["gnn", "rdkit", "mol2vec", "smiles"],
}
FULL_COMBO = "GRMS"

FUSIONS = ["early", "late"]
FUSION_LABELS = {"early": "Early", "late": "Late"}

# The three final predictors. LightGBM and random forest consume a feature matrix
# directly; AttentiveFP consumes the molecular graph and takes the fused features
# as extra global descriptors concatenated to its readout.
LEARNERS = ["lgbm", "rf", "attfp"]
LEARNER_LABELS = {"lgbm": "LGBM", "rf": "RF", "attfp": "AttentiveFP"}

# Learners that can be handed a plain feature matrix. A unimodal RDKit "model"
# under AttentiveFP would be AttentiveFP ignoring its own graph, so the unimodal
# arm of a tabular modality is fit by these two only.
TABULAR_LEARNERS = ["lgbm", "rf"]


def unimodal_method(modality: str, learner: str) -> str:
    return f"uni_{MODALITY_CODE[modality]}_{learner}"


def fusion_method(combo: str, fusion: str, learner: str) -> str:
    return f"fus_{combo}_{fusion}_{learner}"


# Nine unimodal baselines: each of the four modalities under LightGBM and random
# forest, plus AttentiveFP trained end to end on the graph, which *is* the GNN
# modality's native unimodal model rather than a learner applied to an embedding.
UNIMODAL_METHODS = [
    unimodal_method(m, learner) for m in MODALITIES for learner in TABULAR_LEARNERS
] + [unimodal_method("gnn", "attfp")]

# Twenty-four fusion models: four combinations x two strategies x three learners.
FUSION_METHODS = [
    fusion_method(combo, fusion, learner)
    for combo, fusion, learner in product(COMBOS, FUSIONS, LEARNERS)
]

# Every configuration this repository fits. Thirty-three per endpoint per fold.
GRID_METHODS = UNIMODAL_METHODS + FUSION_METHODS

# What each configuration is, looked up rather than re-parsed at every call site.
GRID_SPEC = {
    **{
        unimodal_method(m, learner): {
            "modalities": [m],
            "combo": MODALITY_CODE[m],
            "fusion": "unimodal",
            "learner": learner,
        }
        for m in MODALITIES
        for learner in TABULAR_LEARNERS
    },
    unimodal_method("gnn", "attfp"): {
        "modalities": ["gnn"],
        "combo": "G",
        "fusion": "unimodal",
        "learner": "attfp",
    },
    **{
        fusion_method(combo, fusion, learner): {
            "modalities": list(mods),
            "combo": combo,
            "fusion": fusion,
            "learner": learner,
        }
        for (combo, mods), fusion, learner in product(COMBOS.items(), FUSIONS, LEARNERS)
    },
}

assert set(GRID_SPEC) == set(GRID_METHODS)


def method_label(method: str) -> str:
    spec = GRID_SPEC.get(method)
    if spec is None:
        return REFERENCE_LABELS[method]
    mods = " + ".join(MODALITY_LABELS[m] for m in spec["modalities"])
    if spec["fusion"] == "unimodal":
        if spec["learner"] == "attfp" and spec["modalities"] == ["gnn"]:
            return "AttentiveFP"
        return f"{LEARNER_LABELS[spec['learner']]} - {mods}"
    return f"{LEARNER_LABELS[spec['learner']]} | {FUSION_LABELS[spec['fusion']]} | {mods}"


# --- reference methods ---------------------------------------------------
# Not fit here. Copied from ../expansion-ml-comparison, where they were run on
# these fold files, and carried through the same collection and reporting code so
# the fusion grid has fixed points to be read against.
REFERENCE_METHODS = ["lgbm", "chemprop_st", "chemprop", "chemeleon"]
REFERENCE_LABELS = {
    "lgbm": "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    "chemeleon": "ChemProp + CheMeleon",
}

ALL_METHODS = GRID_METHODS + REFERENCE_METHODS
METHOD_LABELS = {m: method_label(m) for m in ALL_METHODS}

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
# LightGBM: Morgan count fingerprints for the reference arm, library defaults
# otherwise, matching how the reference `lgbm` method was fit.
FP_RADIUS = 2
FP_SIZE = 2048

# AttentiveFP, following the paper's architecture choice and the defaults of the
# reference implementation. `EXTRA_DIM` is filled in at fit time: it is the width
# of whatever fused feature block is concatenated to the graph readout, zero for
# the unimodal model.
ATTFP_HIDDEN = 200
ATTFP_LAYERS = 2
ATTFP_TIMESTEPS = 2
ATTFP_DROPOUT = 0.2
ATTFP_LR = 1e-3
ATTFP_WEIGHT_DECAY = 1e-5
ATTFP_EPOCHS = 150
ATTFP_PATIENCE = 25
ATTFP_BATCH = 64
# The graph embedding handed to the tabular learners is the readout AttentiveFP
# feeds its own output layer, so the GNN modality is the same vector the network
# itself predicts from.
GNN_EMBED_DIM = ATTFP_HIDDEN

# The SMILES branch: a character-level bidirectional GRU trained on the same
# supervised objective, its embedding the concatenated final hidden states.
BIGRU_HIDDEN = 128
BIGRU_EMBED = 64          # character embedding width
BIGRU_LAYERS = 1
BIGRU_DROPOUT = 0.2
BIGRU_LR = 1e-3
BIGRU_EPOCHS = 150
BIGRU_PATIENCE = 25
BIGRU_BATCH = 64
SMILES_EMBED_DIM = 2 * BIGRU_HIDDEN   # forward and backward final states

MOL2VEC_DIM = 300

# Late fusion needs each modality's prediction for the *training* molecules, and
# an in-sample prediction would be a memorised label. The meta-feature matrix is
# built from out-of-fold predictions instead, over an inner split of the fold's
# own training molecules, grouped by cluster exactly as the outer folds are.
INNER_SPLITS = 5

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

# The uncertainty side of the paper. Sigma is the spread of a molecule's five
# within-repeat predictions; the rest describe how well that spread tracks the
# error it is supposed to predict.
UNCERTAINTY_METRICS = ["sigma", "err_unc_corr", "ece", "miscalibration_area"]
UNCERTAINTY_HIGHER_IS_BETTER = {
    "sigma": None,              # neither: a magnitude, not a quality
    "err_unc_corr": True,
    "ece": False,
    "miscalibration_area": False,
}
UNCERTAINTY_LABELS = {
    "sigma": "Epistemic $\\sigma$",
    "err_unc_corr": "Corr($|e|$, $\\sigma$)",
    "ece": "ECE",
    "miscalibration_area": "Miscalibration area",
}
# Bins for the regression-style expected calibration error. Equal-frequency, and
# eight of them: the helper in the authors' uncertainty_analysis.py defaults to
# ten but is called with eight, which is also what their SI Section S6 states.
ECE_BINS = 8


# --- helpers -------------------------------------------------------------
def fold_input(group: str, repeat: int, fold: int) -> Path:
    """The chemprop training file for one group and one fold."""
    return FOLD_DIR / f"{group}_r{repeat}_f{fold}.csv"


def st_fold_input(endpoint: str, repeat: int, fold: int) -> Path:
    """The single-task training file for one endpoint and one fold."""
    return FOLD_DIR / f"st_{endpoint}_r{repeat}_f{fold}.csv"


def fold_embeddings(endpoint: str, repeat: int, fold: int) -> Path:
    """The supervised GNN and BiGRU embeddings for one endpoint and one fold.

    Holds the graph readout and the BiGRU final state for every molecule the fold
    touches, plus each encoder's own test-set prediction and the wall-clock time
    it took to fit. Written once by 16_encode_folds.py and read by all
    thirty-three configurations of that fold.
    """
    return EMBED_DIR / f"{endpoint}_r{repeat}_f{fold}.npz"


def pred_csv(method: str, name: str, repeat: int, fold: int) -> Path:
    """Tidy predictions for one method and fold."""
    return PRED_DIR / method / f"{name}_r{repeat}_f{fold}.csv"


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, RESULTS_DIR, LOG_DIR,
                 FIGURE_DIR, TABLE_DIR, EMBED_DIR, CONTROL_DIR, PAPER_GNN_DIR):
        path.mkdir(parents=True, exist_ok=True)


# --- hyperparameter search ----------------------------------------------
# Verbatim from Section S4 of the paper's Supporting Information, and identical
# to the space in `src/fusion_early.py`. `max_features="auto"` is the one
# substitution: scikit-learn removed it in 1.3, where it meant 1.0 for a
# regressor, so 1.0 takes its place.
LGBM_SEARCH_SPACE = {
    "boosting_type": ["gbdt"],
    "num_leaves": [5, 15, 30],
    "max_depth": [50, 100, 300, -1],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
    "subsample_for_bin": [50, 100, 200],
    "min_split_gain": [0.0],
    "min_child_weight": [0.001],
    "min_child_samples": [20],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}

RF_SEARCH_SPACE = {
    "n_estimators": [50, 100, 150, 200, 300],
    "max_depth": [None, 5, 10, 20, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1.0, "sqrt", 0.3, 0.5, 0.7],
    "bootstrap": [True, False],
}

SEARCH_SPACES = {"lgbm": LGBM_SEARCH_SPACE, "rf": RF_SEARCH_SPACE}

# Their protocol, unchanged: 60 sampled configurations scored by mean squared
# error over a 3-fold split, searched once and reused everywhere. The one
# adaptation is what the three folds are. They have no validation split, so their
# search runs over an unstratified `cv=3` of the training set; here the inner
# folds are grouped by chemical cluster, the same rule that builds the outer
# folds, so a chemotype cannot straddle the inner boundary either.
SEARCH_ITERATIONS = 60
SEARCH_CV = 3
SEARCH_SCORING = "neg_mean_squared_error"

# Where the tuned settings land: one JSON per data set, keyed by endpoint and
# method. Tuning is done on repeat 0, fold 0's fitting molecules and the result is
# reused by all 25 folds, so no fold's hyperparameters were chosen with sight of
# the molecules it is scored on.
HPARAM_JSON = DATA_DIR / "hyperparameters.json"
TUNE_REPEAT = 0
TUNE_FOLD = 0

# `attfp` is not searched. The released code hardcodes DeepChem's AttentiveFP
# defaults at a fixed 50 epochs and never runs a search for it; the space in the
# SI applies to the graph meta-learner, which the release does not implement. A
# 60-point search over a network that has to be retrained per fold would also cost
# more than the rest of this repository put together.
TUNED_LEARNERS = ["lgbm", "rf"]

# The leak-free control from `fusion.py`: the same late-fusion configurations with
# the meta-learner fit on the fold's held-out fifth instead of on the molecules
# the base learners already saw. Tabular meta-learners only -- the graph one would
# need another 900 networks to answer a question about stacking, not about graphs.
CONTROL_METHODS = [
    fusion_method(combo, "late", learner)
    for combo in COMBOS
    for learner in TABULAR_LEARNERS
]
CONTROL_DIR = RESULTS_DIR / "control"


# The other control: the GNN modality as the released extractor actually produces
# it. `extract_attentivefp_embeddings_strict_dgl` hooks the first nn.Linear in
# module order, which in DeepChem's AttentiveFP is the projection of raw atom
# features applied before any message passing, so the block it returns is a
# 30-wide mean of unlearned atom features rather than the 200-wide learned
# readout. The paper's own Table S3 lists the modality as 30 features. Rerunning
# the LightGBM configurations that name the GNN modality on that block measures
# what the difference is worth.
PAPER_GNN_METHODS = [unimodal_method("gnn", "lgbm")] + [
    fusion_method(combo, fusion, "lgbm")
    for combo in COMBOS for fusion in FUSIONS
]
PAPER_GNN_DIR = RESULTS_DIR / "paper_gnn"
