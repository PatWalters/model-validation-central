"""Shared configuration for the Causal-Chemprop comparison.

Causal-Chemprop (Natajaya, Attia, Burns and Doyle, J. Chem. Inf. Model. 2026,
doi:10.1021/acs.jcim.6c01429) puts a structural causal model on top of a
trained Chemprop. The 200-d hidden layer of the Chemprop feed-forward network
and the target are 201 nodes; DAGMA learns a linear DAG over them on an
"analog set" of training molecules whose Murcko scaffolds resemble the
held-out series; dowhy's GCM fits additive-noise mechanisms on each node. At
inference one molecule of the held-out series, the anchor, is observed with its
measured value. Its exogenous noise is abducted, the test molecule's embedding
is substituted on the nodes with large intrinsic causal contribution, and the
counterfactual target is the prediction. The paper reports better within-
series ranking than Chemprop on eleven kinase scaffold groups.

The method needs an anchor -- a measured molecule from the series being
predicted -- which no arm in the rest of this repository gets. So the
comparison has two halves.

  paper     the paper's eleven ABL1 / AURKA / SRC tasks, from the authors'
            release. Table 1 reproduced from their checkpoints, then Monroe +
            TabPFN 3, Atom-JEPA and LightGBM on the same splits and anchors.

  protocol  ExpansionRx and Biogen on the shared folds. Each held-out test
            cluster with at least seven measured molecules is a series; one
            seeded-random member is its anchor, revealed to every
            anchor-aware arm; the rest are scored. Causal-Chemprop runs the
            authors' code per fold and per series.

Every arm in the protocol half is scored on exactly the same molecules -- the
non-anchor members of the eligible series -- and every arm that is not
anchor-aware by construction also appears with the cheapest anchor-aware
correction there is: shift its predictions for a series by its own error on
that series' anchor. That is the paper's "direct residual-correction" baseline,
and it is what prices the causal machinery against the anchor it is given.

  imported, not re-run
    lgbm, chemprop_st, chemprop, chemeleon, monroe   expansion-ml-comparison
    atomjepa_ft                                       atom-jepa

  fitted here
    cc_chemprop      the authors' Chemprop recipe on the fold, ordinary
                     inference: the encoder Causal-Chemprop sits on
    causal_chemprop  the authors' SCM fitted on that encoder, per series
    monroe_anchor    Monroe + TabPFN 3 with every series anchor added to the
                     in-context training set

  derived, by 08_collect_metrics.py
    <arm>_offset     <arm> shifted per series by its residual on the anchor
"""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


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
        targets=["LogD", "LogS", "LOG_HLM", "LOG_MLM", "LOG_Caco_AB",
                 "LOG_Caco_Efflux", "LOG_MPPB", "LOG_MBPB", "LOG_MGMB"],
        groups={
            "physchem_binding": ["LogD", "LogS", "LOG_MPPB", "LOG_MBPB", "LOG_MGMB"],
            "metabolism": ["LOG_HLM", "LOG_MLM"],
            "permeability": ["LOG_Caco_AB", "LOG_Caco_Efflux"],
        },
    ),
    "biogen": DataSet(
        name="biogen",
        label="Biogen ADME",
        targets=["LOG_SOL", "LOG_HLM", "LOG_RLM", "LOG_MDR1_ER", "LOG_HPPB", "LOG_RPPB"],
        groups={
            "physchem_binding": ["LOG_SOL", "LOG_HPPB", "LOG_RPPB"],
            "metabolism": ["LOG_HLM", "LOG_RLM"],
            "permeability": ["LOG_MDR1_ER"],
        },
    ),
}

DATASET = os.environ.get("ADME_DATASET", "expansion")
if DATASET not in DATASETS:
    raise SystemExit(f"unknown ADME_DATASET {DATASET!r} -- choose from {', '.join(DATASETS)}")
ACTIVE = DATASETS[DATASET]

# --- paths ---------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

SOURCE_REPO = Path(
    os.environ.get("ADME_SOURCE_REPO", PROJECT_DIR.parent / "expansion-ml-comparison")
).expanduser()
# Fine-tuned Atom-JEPA's per-fold predictions, from that study's sweep.
ATOMJEPA_REPO = Path(
    os.environ.get("ATOMJEPA_REPO", PROJECT_DIR.parent / "atom-jepa")
).expanduser()

# Where 09_run_monroe.py in the source comparison cached Monroe's embedding.
MONROE_EMBED_REPO = Path(os.environ.get("MONROE_EMBED_REPO", SOURCE_REPO)).expanduser()

# The authors' checkout with their Zenodo record unpacked into data/ and
# output/. Never written to: 01_reproduce_paper.py works on copies.
UPSTREAM_HOME = Path(
    os.environ.get("CAUSAL_CHEMPROP_HOME", Path.home() / "CAUSAL_CHEMPROP_UPSTREAM")
).expanduser().resolve()
UPSTREAM_COPY = {
    "released": PROJECT_DIR / "work" / "upstream_released",
    "refit": PROJECT_DIR / "work" / "upstream_refit",
}

MONROE_HOME = Path(
    os.environ.get("MONROE_HOME", Path.home() / "software" / "monroe")
).expanduser().resolve()

DATA_DIR = PROJECT_DIR / "data" / DATASET
FOLD_DIR = PROJECT_DIR / "folds" / DATASET
PRED_DIR = PROJECT_DIR / "predictions" / DATASET
DIAG_DIR = PROJECT_DIR / "diagnostics" / DATASET
RESULTS_DIR = PROJECT_DIR / "results" / DATASET
LOG_DIR = PROJECT_DIR / "logs" / DATASET
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
# Per-fold Chemprop checkpoints for the SCM stage; regenerable, not tracked.
MODEL_DIR = PROJECT_DIR / "work" / "models" / DATASET

MASTER_CSV = DATA_DIR / "master.csv"
FOLD_CSV = FOLD_DIR / "fold_assignments.csv"
SERIES_CSV = DATA_DIR / "series.csv"          # 02_make_series.py
MONROE_NPZ = DATA_DIR / "monroe_embeddings.npz"

PREDICTIONS_PARQUET = RESULTS_DIR / "predictions_all.parquet"
FOLD_METRICS_CSV = RESULTS_DIR / "fold_metrics.csv"
SERIES_METRICS_CSV = RESULTS_DIR / "series_metrics.csv"
DIAGNOSTICS_CSV = RESULTS_DIR / "diagnostics.csv"
TIMING_CSV = RESULTS_DIR / "timing.csv"


def prepare_upstream_copies() -> None:
    """Two private copies of the release, so nothing upstream is ever written."""
    for dst in UPSTREAM_COPY.values():
        if dst.exists():
            continue
        shutil.copytree(UPSTREAM_HOME, dst, ignore=shutil.ignore_patterns(
            ".git", "__MACOSX", "*.zip", "tensorboard_logs"))


# --- the paper's own tasks ----------------------------------------------
PAPER_TASKS = ["ABL1_0", "ABL1_1", "ABL1_2", "ABL1_3",
               "AURKA_0", "AURKA_1", "AURKA_2", "AURKA_3",
               "SRC_0", "SRC_1", "SRC_2"]
PAPER_RELEASED_ARMS = ["causal_chemprop", "chemprop", "chemprop_retrained"]
# Table 1, Pearson r and Spearman rho, as printed.
PAPER_TABLE1 = {
    #            causal (r, rs)   chemprop (r, rs)   retrained (r, rs)
    "ABL1_0":  ((0.89, 0.76), (0.66, 0.71), (0.51, 0.55)),
    "ABL1_1":  ((0.59, 0.16), (-0.05, -0.29), (-0.16, -0.33)),
    "ABL1_2":  ((0.52, 0.38), (0.06, 0.20), (-0.04, 0.08)),
    "ABL1_3":  ((0.51, 0.77), (0.46, 0.64), (0.55, 0.80)),
    "AURKA_0": ((0.53, 0.54), (0.42, 0.40), (0.55, 0.53)),
    "AURKA_1": ((0.76, 0.90), (0.67, 0.86), (0.29, 0.31)),
    "AURKA_2": ((0.32, -0.22), (0.50, 0.36), (-0.20, -0.22)),
    "AURKA_3": ((0.75, 0.75), (0.57, 0.48), (0.63, 0.46)),
    "SRC_0":   ((0.58, 0.42), (0.57, 0.50), (0.61, 0.45)),
    "SRC_1":   ((0.73, 0.77), (-0.48, -0.46), (0.72, 0.67)),
    "SRC_2":   ((0.37, 0.41), (-0.04, -0.08), (0.14, 0.17)),
}

# --- data columns --------------------------------------------------------
SMILES_COL = "SMILES"
ID_COL = "Name"
SET_COL = "ds"
CLUSTER_COL = "cluster"

TARGET_COLS = list(ACTIVE.targets)
TARGET_GROUPS = {group: list(targets) for group, targets in ACTIVE.groups.items()}
GROUPS = list(TARGET_GROUPS)
GROUP_OF = dict(ACTIVE.group_of)

# --- series and anchors --------------------------------------------------
# A series is one held-out test cluster, for one endpoint, with at least this
# many measured molecules -- the paper's eligibility floor for a scaffold group
# (">=7-member"). One member is the anchor and the rest are scored.
MIN_SERIES = 7
# The paper's analog set: training molecules whose Murcko scaffold has ECFP4
# Tanimoto >= 0.25 to the series scaffold, by the authors' own
# find_similar_scaffolds_diverse_decorators. Here the series scaffold is the
# anchor's, which in the paper's tasks is the group scaffold by construction.
ANALOG_SCAFFOLD_SIM = 0.25
# Below roughly 90 analogs the paper finds the 201-node SCM under-determined
# and calls the group infeasible. The same floor, per fold.
MIN_ANALOGS = 90
ANCHOR_SEED = 0xA9C4

# --- methods -------------------------------------------------------------
LGBM_METHOD = "lgbm"
CHEMELEON_METHOD = "chemeleon"
MONROE_METHOD = "monroe"
ATOMJEPA_METHOD = "atomjepa_ft"
CC_CHEMPROP_METHOD = "cc_chemprop"
CAUSAL_METHOD = "causal_chemprop"
MONROE_ANCHOR_METHOD = "monroe_anchor"
OFFSET = "_offset"

BASELINE_IS_SINGLE_TASK = {
    LGBM_METHOD: True,
    "chemprop_st": True,
    "chemprop": False,
    CHEMELEON_METHOD: False,
    MONROE_METHOD: True,
}
BASELINE_METHODS = list(BASELINE_IS_SINGLE_TASK)

# Arms that see no anchor, each of which also gets an _offset twin.
UNANCHORED = [LGBM_METHOD, CHEMELEON_METHOD, MONROE_METHOD, ATOMJEPA_METHOD, CC_CHEMPROP_METHOD]

METHOD_LABELS = {
    LGBM_METHOD: "LightGBM + Morgan",
    "chemprop_st": "ChemProp single-task",
    "chemprop": "ChemProp multi-task",
    CHEMELEON_METHOD: "ChemProp + CheMeleon",
    MONROE_METHOD: "Monroe + TabPFN 3",
    ATOMJEPA_METHOD: "Atom-JEPA fine-tuned",
    CC_CHEMPROP_METHOD: "Chemprop (Causal-Chemprop's encoder)",
    CAUSAL_METHOD: "Causal-Chemprop",
    MONROE_ANCHOR_METHOD: "Monroe + TabPFN 3, anchors in context",
}
for _m in UNANCHORED:
    METHOD_LABELS[_m + OFFSET] = METHOD_LABELS[_m] + " + anchor offset"

# --- Causal-Chemprop -----------------------------------------------------
# The authors' Chemprop, as causal_utils.train builds it: bond message passing,
# depth 6, hidden 200, dropout 0.2, mean aggregation, batch norm, a 3-layer
# 200-unit FFN, MSE on standardised targets, Noam schedule 5e-5 / 5e-4 / 1e-5,
# batch 128, up to 200 epochs, early stopping on validation loss, patience 20.
CC_MPNN_HIDDEN = 200
CC_FFN_HIDDEN = 200
CC_FFN_LAYERS = 3
CC_DEPTH = 6
CC_DROPOUT = 0.2
CC_MAX_EPOCHS = 200
CC_PATIENCE = 20
CC_BATCH = 128
CC_LR = (5e-5, 5e-4, 1e-5)

# The SCM. The release hard-codes these per task; the paper says one fixed
# rule. Where the eleven tasks disagree the majority value is taken:
#   LAMBDA (DAGMA L1)  0.001 on 7 of 11 tasks, 0.002 on 4
#   W_THRESHOLD        0.3 on all 11
#   T (DAGMA stages)   5 on all 11
#   THRESHOLD          the interv/counterfactual cosine ratio above which the
#                      plain SCM prediction replaces the counterfactual:
#                      4 on 6 of 11, 2 on 5
# ICC threshold follows eq 10 of the paper, as test_counterfactual computes it.
SCM_LAMBDA = 0.001
SCM_W_THRESHOLD = 0.3
SCM_T = 5
SCM_SWITCH = 4

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
PRED_COLUMNS = ["method", "endpoint", "repeat", "fold", ID_COL, SMILES_COL, "y_true", "y_pred"]

METRICS = ["r2", "spearman", "mae"]
METRIC_HIGHER_IS_BETTER = {"r2": True, "spearman": True, "mae": False, "pearson_series": True,
                           "spearman_series": True}
METRIC_LABELS = {"r2": "$R^2$", "spearman": "Spearman $\\rho$", "mae": "MAE"}


def pred_csv(method: str, name: str, repeat: int, fold: int) -> Path:
    """Tidy predictions for one method and fold -- also the resume point."""
    return PRED_DIR / method / f"{name}_r{repeat}_f{fold}.csv"


def ensure_dirs() -> None:
    for path in (DATA_DIR, FOLD_DIR, PRED_DIR, DIAG_DIR, RESULTS_DIR, LOG_DIR,
                 FIGURE_DIR, TABLE_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
