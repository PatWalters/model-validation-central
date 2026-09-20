"""The master table, the fold assignments and the cached features, in one place.

Four arms fit models here and all four need the same three things: which rows of
master.csv one fold trains on, which it is scored on, and a feature matrix in
master.csv row order. In `studies/expansion-ml-comparison` each runner builds
those masks itself from the `ds` column and the fold file, and they agree because
the code is copied between them. There are six splitting schemes here rather than
one fixed holdout, so the masks are worth writing once and sharing: an arm that
trained on a different set of molecules than its neighbour would not be a
comparison at all.

Every arm therefore takes its rows from `FoldData.fold`, and writes its
predictions through `FoldData.write_predictions`, so the tidy schema and the
row selection are identical by construction rather than by inspection.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


@dataclass(frozen=True)
class FoldRows:
    """Row numbers into master.csv for one (target, split, fold)."""

    target: int
    split: str
    fold: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class FoldData:
    """master.csv, the fold assignments, and whichever features were asked for."""

    master: pd.DataFrame
    folds: pd.DataFrame
    ecfp_bits: np.ndarray | None = None
    morgan_counts: np.ndarray | None = None
    embeddings: np.ndarray | None = None
    embeddings_ok: np.ndarray | None = None

    # master.csv row number for every (target, idx) pair, built once.
    _row_of: dict = None

    @classmethod
    def load(cls, need: str | None = None) -> "FoldData":
        """Read the tables, and the one feature matrix `need` names.

        `need` is "ecfp" for k-NN, "morgan" for LightGBM, "monroe" for the
        in-context arm, and None for an arm that reads no cached features.
        """
        if not cfg.MASTER_CSV.exists():
            raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 00_prepare_data.py first")
        master = pd.read_csv(cfg.MASTER_CSV)
        folds = pd.read_csv(cfg.FOLD_CSV)

        data = cls(master=master, folds=folds)
        data._row_of = {
            (int(t), int(i)): row
            for row, (t, i) in enumerate(zip(master["target"], master["idx"]))
        }

        if need == "ecfp":
            data.ecfp_bits = _load_matrix(cfg.ECFP_BITS_NPY, len(master))
        elif need == "morgan":
            data.morgan_counts = _load_matrix(cfg.MORGAN_COUNTS_NPY, len(master))
        elif need == "monroe":
            data.embeddings, data.embeddings_ok = _load_embeddings(master)
        elif need is not None:
            raise ValueError(f"unknown feature set {need!r}")
        return data

    @property
    def y(self) -> np.ndarray:
        return self.master["y"].to_numpy(dtype=float)

    def fold(self, target: int, split: str, fold: int) -> FoldRows:
        """The three roles of one fold, as master.csv row numbers.

        Molecules RDKit could not parse would break the comparability of a fold
        -- one arm would silently see fewer of them -- so an affected fold stops
        the run instead. In practice every molecule in this collection parses.
        """
        g = self.folds[
            (self.folds["target"] == target)
            & (self.folds["split"] == split)
            & (self.folds["fold"] == fold)
        ]
        if g.empty:
            raise SystemExit(f"no fold assignment for tid_{target} {split} f{fold}")

        roles = {}
        for role in ("train", "val", "test"):
            idxs = g.loc[g["split_role"] == role, "idx"].to_numpy()
            rows = np.array([self._row_of[(target, int(i))] for i in idxs], dtype=int)
            bad = int((~self.master["ok"].to_numpy()[rows]).sum())
            if bad:
                raise SystemExit(
                    f"tid_{target} {split} f{fold} {role}: {bad} unparseable molecules"
                )
            roles[role] = np.sort(rows)
        return FoldRows(target, split, fold, roles["train"], roles["val"], roles["test"])

    def usable_embeddings(self, rows: np.ndarray, where: str) -> None:
        """Stop if any of `rows` has no Monroe embedding."""
        if self.embeddings_ok is None:
            raise SystemExit("embeddings not loaded")
        missing = int((~self.embeddings_ok[rows]).sum())
        if missing:
            raise SystemExit(
                f"{where}: {missing} molecules have no Monroe embedding, so this fold "
                "would not be comparable with the other arms"
            )

    def write_predictions(
        self,
        method: str,
        target: int,
        split: str,
        fold: int,
        rows: np.ndarray,
        pred: np.ndarray,
        extra: dict | None = None,
    ) -> Path:
        """One fold's test predictions, in the schema every arm shares.

        `extra` carries anything an arm wants to record about the fit itself --
        k for k-NN, the stopping epoch for a network -- as constant columns
        outside PRED_COLUMNS. 07_collect_metrics.py reads only PRED_COLUMNS, so
        these are for the record rather than for the statistics.
        """
        out = pd.DataFrame(
            {
                "method": method,
                "target": target,
                "split": split,
                "fold": fold,
                cfg.OUT_ID_COL: self.master[cfg.OUT_ID_COL].to_numpy()[rows],
                cfg.OUT_SMILES_COL: self.master[cfg.OUT_SMILES_COL].to_numpy()[rows],
                "y_true": self.y[rows],
                "y_pred": np.asarray(pred, dtype=float),
            },
            columns=cfg.PRED_COLUMNS,
        )
        if len(out) != len(rows):
            raise SystemExit(f"{method} tid_{target} {split} f{fold}: prediction length mismatch")
        if not np.isfinite(out["y_pred"]).all():
            raise SystemExit(f"{method} tid_{target} {split} f{fold}: non-finite predictions")
        for key, value in (extra or {}).items():
            out[key] = value

        path = cfg.pred_csv(method, target, split, fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
        return path


def _load_matrix(path: Path, n: int) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 00_prepare_data.py first")
    X = np.load(path)
    if len(X) != n:
        raise SystemExit(f"{path.name} has {len(X)} rows, master.csv has {n}")
    return X


def _load_embeddings(master: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if not cfg.MONROE_NPZ.exists():
        raise SystemExit(
            f"{cfg.MONROE_NPZ} not found -- run 04_run_monroe.py --embed first"
        )
    cached = np.load(cfg.MONROE_NPZ, allow_pickle=False)
    X, ok, names = cached["X"], cached["ok"], cached["names"]
    if len(X) != len(master) or not np.array_equal(
        names, master[cfg.OUT_ID_COL].to_numpy().astype(str)
    ):
        raise SystemExit(
            f"{cfg.MONROE_NPZ.name} does not line up with {cfg.MASTER_CSV.name} -- "
            "re-run with --embed"
        )
    return X, ok


def add_fold_arguments(parser) -> None:
    """The three selectors every runner takes, plus --force."""
    parser.add_argument("--target", nargs="+", type=int, default=cfg.TARGETS, choices=cfg.TARGETS)
    parser.add_argument("--split", nargs="+", default=cfg.SPLITS, choices=cfg.SPLITS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true",
                        help="refit folds that already have predictions")


def selected_folds(args) -> list[tuple[int, str, int]]:
    """Every (target, split, fold) a run was asked for, target-major."""
    return [
        (target, split, fold)
        for target in args.target
        for split in args.split
        for fold in args.fold
    ]
