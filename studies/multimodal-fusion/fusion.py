"""Assembling the design grid: modality blocks, the two fusion strategies, the
three learners.

Everything a configuration needs is derived here from one fold's cached blocks,
so `05_run_grid.py` is a loop rather than a pile of special cases.

Early fusion concatenates modality feature vectors and hands the result to one
predictor. Late fusion fits one predictor per modality, stacks their predictions
into a matrix as wide as the number of modalities, and hands *that* to a
meta-learner. Both follow `src/fusion_early.py` and `src/fusion_late.py` of
github.com/jwasswa2023/Multimodal_Fusion, including the two choices worth naming:
no feature scaling anywhere a tree ensemble is the predictor, and meta-features
built from base predictions on the molecules the base models were fit on.

That second one is stacking leakage, and it is theirs rather than a slip in
reading them: their base learners fit the whole training set and then predict it,
and there is no out-of-fold scheme anywhere in the release. It is reproduced as
written, and `holdout_meta=True` fits the same meta-learner on the fold's
held-out fifth instead, which is the identical procedure without the leak. The
report shows both.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

import config as cfg


# --- one fold's cached blocks -------------------------------------------
class FoldBlocks:
    """Every modality's features and base prediction, for one endpoint and fold.

    Rows are the molecules the fold touches, in master.csv order; `fit`, `val` and
    `test` index into that. Two of the four feature blocks come from step 2's
    per-molecule caches and two from step 3's per-fold encoders, but by the time
    they are here they are four matrices with the same number of rows.
    """

    def __init__(self, npz, masks, rdkit_all, mol2vec_all, y_all, graphs_all,
                 paper_gnn_block: bool = False):
        rows = npz["rows"]
        self.rows = rows
        position = {row: i for i, row in enumerate(rows)}
        self.fit = np.array([position[i] for i in np.flatnonzero(masks.fit)], dtype=int)
        self.val = np.array([position[i] for i in np.flatnonzero(masks.val)], dtype=int)
        self.test = np.array([position[i] for i in np.flatnonzero(masks.test)], dtype=int)

        gnn_key = "gnn_atom_mean" if paper_gnn_block else "gnn_embed"
        self.features = {
            "rdkit": rdkit_all[rows],
            "mol2vec": mol2vec_all[rows],
            "gnn": npz[gnn_key],
            "smiles": npz["smiles_embed"],
        }
        # The two modalities whose base prediction comes free with the encoder
        # step 3 trained. The tabular two get a base learner fit in
        # `base_predictions`.
        self.encoder_preds = {
            "gnn": npz["gnn_pred"].astype(float),
            "smiles": npz["smiles_pred"].astype(float),
        }
        # Only the graph learner needs these, and importing torch into 24 pool
        # workers that will never touch it is a gigabyte apiece for nothing.
        self.graphs = None if graphs_all is None else [graphs_all[i] for i in rows]
        self.y = y_all[rows]
        self.seconds = {
            "gnn": float(npz["gnn_seconds"]),
            "smiles": float(npz["smiles_seconds"]),
        }

    def block(self, modality: str) -> np.ndarray:
        return self.features[modality]

    @staticmethod
    def ordered(modalities: list[str]) -> list[str]:
        """Modalities in a fixed canonical order, whatever order they arrive in.

        Fixing it means a column index means the same thing in every
        configuration sharing a prefix, which is what makes the grouped SHAP
        blocks of step 8 comparable across the grid.
        """
        return [m for m in cfg.MODALITIES if m in modalities]

    def matrix(self, modalities: list[str], rows: np.ndarray) -> np.ndarray:
        """Early fusion: the modality blocks side by side."""
        return np.concatenate(
            [self.features[m][rows] for m in self.ordered(modalities)], axis=1
        )

    def block_slices(self, modalities: list[str]) -> dict[str, slice]:
        """Where each modality lives in the concatenated matrix."""
        out, start = {}, 0
        for m in self.ordered(modalities):
            width = self.features[m].shape[1]
            out[m] = slice(start, start + width)
            start += width
        return out


# --- learners ------------------------------------------------------------
def make_tabular(learner: str, params: dict, seed: int, threads: int = 1):
    """A LightGBM or random forest regressor at the tuned settings."""
    if learner == "lgbm":
        return LGBMRegressor(**{**params, "verbose": -1,
                                "random_state": seed, "n_jobs": threads})
    if learner == "rf":
        return RandomForestRegressor(**{**params, "random_state": seed, "n_jobs": threads})
    raise ValueError(f"{learner!r} is not a tabular learner")


GLOBAL_CLIP = 10.0


def standardize(fit_x: np.ndarray, *others: np.ndarray):
    """Zero mean, unit variance, from the fitting rows only, then clipped.

    Used for the global features handed to the graph learner and nowhere else. A
    tree ensemble does not need it and the released code never does it; a neural
    head fed raw RDKit descriptors, where MolWt sits beside a 0/1 flag, does.

    The clip is not cosmetic. Several RDKit descriptors are near-constant over a
    fold's fitting molecules and then take a wildly different value on one test
    molecule, which standardizes to something in the millions and drags the
    linear head off by ten orders of magnitude -- the predictions stay correctly
    *ordered*, so it shows up as a catastrophic R squared beside an unremarkable
    Spearman rho. Ten standard deviations is well outside anything the network
    saw and well inside what its head can represent.
    """
    mean = fit_x.mean(axis=0)
    sd = fit_x.std(axis=0)
    sd[sd < 1e-8] = 1.0
    return tuple(
        np.clip((x - mean) / sd, -GLOBAL_CLIP, GLOBAL_CLIP).astype(np.float32)
        for x in (fit_x, *others)
    )


# --- late fusion ---------------------------------------------------------
def base_predictions(blocks: FoldBlocks, modalities: list[str], params: dict,
                     seed: int, threads: int = 1) -> np.ndarray:
    """One column per modality, one row per molecule the fold touches.

    The GNN and SMILES columns are the encoders' own predictions, which step 3
    already has. The two tabular columns come from a LightGBM base learner fit on
    the fold's fitting molecules at the settings tuned for that modality alone,
    which is what `fusion_late.py` fits.

    Every row is filled, including the fitting rows themselves. Which rows the
    meta-learner is then *trained* on is the caller's decision, and it is the only
    thing separating the released procedure from the leak-free control.
    """
    columns = []
    for modality in blocks.ordered(modalities):
        if modality in blocks.encoder_preds:
            columns.append(blocks.encoder_preds[modality])
            continue
        x = blocks.features[modality]
        model = make_tabular("lgbm", params[modality], seed, threads)
        model.fit(x[blocks.fit], blocks.y[blocks.fit])
        columns.append(model.predict(x))
    return np.column_stack(columns)


# --- the whole grid, one configuration at a time -------------------------
def fit_predict(spec: dict, blocks: FoldBlocks, params: dict, seed: int,
                threads: int = 1, device=None, holdout_meta: bool = False) -> np.ndarray:
    """Test-set predictions for one configuration of one fold.

    `spec` is `cfg.GRID_SPEC[method]`. `params["model"]` is what step 4 tuned for
    this configuration; `params["base"]` holds the per-modality settings the late
    base learners need.
    """
    modalities, fusion, learner = spec["modalities"], spec["fusion"], spec["learner"]

    # The unimodal AttentiveFP is the encoder step 3 already trained. Refitting an
    # identically seeded network on identical molecules would only reproduce it.
    if fusion == "unimodal" and learner == "attfp":
        return blocks.encoder_preds["gnn"][blocks.test]

    if fusion == "unimodal":
        x = blocks.block(modalities[0])
        model = make_tabular(learner, params["model"], seed, threads)
        model.fit(x[blocks.fit], blocks.y[blocks.fit])
        return model.predict(x[blocks.test])

    if fusion == "early":
        if learner == "attfp":
            others = [m for m in modalities if m != "gnn"]
            globals_ = blocks.matrix(others, np.arange(len(blocks.y)))
            return _graph_learner(blocks, globals_, blocks.fit, blocks.val, seed, device)

        model = make_tabular(learner, params["model"], seed, threads)
        model.fit(blocks.matrix(modalities, blocks.fit), blocks.y[blocks.fit])
        return model.predict(blocks.matrix(modalities, blocks.test))

    # Late fusion. `meta` is filled for every row; the split below decides which
    # rows train the meta-learner and which validate it.
    meta = base_predictions(blocks, modalities, params["base"], seed, threads)
    train_rows, val_rows = (
        (blocks.val, blocks.fit) if holdout_meta else (blocks.fit, blocks.val)
    )

    if learner == "attfp":
        return _graph_learner(blocks, meta, train_rows, val_rows, seed, device)

    model = make_tabular(learner, params["model"], seed, threads)
    model.fit(meta[train_rows], blocks.y[train_rows])
    return model.predict(meta[blocks.test])


def _graph_learner(blocks: FoldBlocks, globals_: np.ndarray, train_rows: np.ndarray,
                   val_rows: np.ndarray, seed: int, device) -> np.ndarray:
    """AttentiveFP as the final predictor, with the other modalities as globals.

    The graph *is* the GNN modality here, so what is concatenated onto the readout
    is the rest of the design: the other modalities' feature blocks under early
    fusion, the stacked base predictions under late fusion. That is the
    `global_feat_size` entry in the paper's AttentiveFP search space, which their
    released code never implements.
    """
    import nets

    g_train, g_val, g_test = standardize(
        globals_[train_rows], globals_[val_rows], globals_[blocks.test]
    )
    model = nets.fit_graph_regressor(
        [blocks.graphs[i] for i in train_rows], blocks.y[train_rows],
        [blocks.graphs[i] for i in val_rows], blocks.y[val_rows],
        seed=seed, train_globals=g_train, val_globals=g_val, device=device,
    )
    return nets.graph_predict(
        model, [blocks.graphs[i] for i in blocks.test], globals_=g_test, device=device
    )
