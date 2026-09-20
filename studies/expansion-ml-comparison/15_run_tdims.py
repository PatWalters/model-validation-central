#!/usr/bin/env python
"""Step 15: the TDiMS arm of the comparison.

TDiMS (Hamada et al., Nat. Comput. Sci. 2026, doi:10.1038/s43588-026-01036-3) is
a molecular descriptor, not a model. For every molecule it enumerates pairs of
substructures -- heteroatoms, circular substructures from the Morgan fingerprint,
and optionally a fixed list of CEP ring fragments -- and stores a function of the
topological distance between the two members of each pair. A molecule becomes a
long, sparse vector over every substructure pair seen anywhere in the data set.
Nothing is pre-trained and nothing is trained downstream.

The point of this arm is a controlled comparison with `monroe35`. That arm is a
58.5 M-parameter graph transformer pre-trained on 81 M molecules, frozen, read by
TabPFN 3.5 in context. This arm is a hand-designed descriptor read by the *same*
TabPFN 3.5, through the same `fit_predict_tabpfn` at the same ensemble settings,
on the same folds, with the same fit and test masks and the same per-fold seed.
The only thing that differs between the two is what a molecule is turned into.

Choosing the descriptor configuration
-------------------------------------
The configuration is part of TDiMS, not a detail of it: the paper searches over
the Morgan radius, the distance transform f_dis, the duplicate-merge rule f_dup
and whether the CEP fragment list is included, and "attempts to perform some
initial training, and the most promising combination is passed to the estimator".
So this script does the same, and does it without touching the test set.

Each fold of this study holds out a fifth of the training molecules. LightGBM and
Monroe never use it -- neither has anything to early-stop -- so it is free here,
and it is where the configuration is chosen. Choosing with the real head over the
whole 24-point grid would cost more than the arm itself, so it happens in two
stages, both on that held-out fifth: a ridge regression screens all 24 on one
fold, and the four best per endpoint are then scored by the actual TabPFN 3.5
head over three folds. The winner predicts the test set for all 25 folds. Every
score both stages were made from is kept, in results/<dataset>/tdims_screen.csv
and results/<dataset>/tdims_config.csv.

Everything between the descriptor and the head is the authors' own pipeline from
experiments/run_nested_cv_experiment.py, in their order: zero out feature values
above the threshold, drop features that are zero throughout the fold's training
molecules, scale (only the f_dis = x configurations need it), and select features
with `SelectFromModel` around a LASSO at alpha=1e-4 with threshold="mean" and
scikit-learn's default iteration cap, which is what their experiment script uses
(see FS_MAX_ITER -- their two code paths disagree, and it matters). Each step that
has anything to fit is refitted inside every fold on that fold's training rows
alone, so no label from a validation or test molecule reaches it.

Their pipeline ends in a LASSO, ridge, elastic net or random forest chosen by an
inner grid search. Here it ends in TabPFN 3.5 instead, because that is the whole
point of the arm -- the estimator is held fixed at Monroe's and the representation
is what varies.

Three phases:

  --features  build the descriptor cache, one sparse matrix per configuration,
              in data/<dataset>/tdims/. Label-free, so it is done once and every
              later phase reads it.
  --select    the two-stage validation sweep described above
  (default)   for each endpoint x repeat x fold, select features on the fold's
              training rows and predict the fixed test set with TabPFN 3.5.

    python 15_run_tdims.py --features                  # ~25 min for 24 configs
    python 15_run_tdims.py --select                    # the validation sweep
    python 15_run_tdims.py                             # all 225 folds
    ADME_DATASET=biogen python 15_run_tdims.py --features

TDiMS itself comes from the authors' release (IBM/materials, models/tdims, Apache
2.0); point TDIMS_HOME at that checkout. The head needs the TabPFN 3.5
environment, the same one run_monroe35.sh uses, and TABPFN_TOKEN set.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import config as cfg

TDIMS_HOME = Path(
    os.environ.get("TDIMS_HOME", Path.home() / "software" / "tdims")
).expanduser().resolve()

# --- the configuration grid ---------------------------------------------
# The four axes the paper searches over. `atom_set` is not an axis: heteroatom
# substructures are always included, as they are in the paper.
#
# f_dis is applied to the mean topological distance of a pair as d ** f_dis, so
# -2 is the inverse square (Coulomb-like, near pairs dominate), -1 the inverse,
# and 1 the raw distance (far pairs dominate, which is the conjugation case).
# f_dup merges the values of a pair that occurs in several places.
RADII = (1, 2)
FUNC_DIS = {"dm2": -2, "dm1": -1, "dp1": 1}
FUNC_MERGE = {"max": max, "sum": sum}
# The CEP fragment list is the one the authors ship: 26 ring systems from the
# Harvard Clean Energy Project. It was drawn up for organic electronics, so how
# much of it a drug-like set matches is an open question -- which is why it is an
# axis here rather than a decision.
FRAGMENTS = {"nofrag": False, "cep": True}


def config_tag(radius: int, dis: str, merge: str, frag: str) -> str:
    return f"r{radius}_{dis}_{merge}_{frag}"


def all_configs() -> list[dict]:
    return [
        {"tag": config_tag(radius, dis, merge, frag), "radius": radius,
         "func_dis": FUNC_DIS[dis], "func_merge": FUNC_MERGE[merge],
         "fragment_set": FRAGMENTS[frag]}
        for radius in RADII
        for dis in FUNC_DIS
        for merge in FUNC_MERGE
        for frag in FRAGMENTS
    ]


CONFIGS = {spec["tag"]: spec for spec in all_configs()}

# The LASSO feature selector of the TDiMS paper, refitted per fold on that fold's
# training rows. Its settings are the authors' own -- with one ambiguity that has
# to be resolved, because their release contains two different answers.
#
# experiments/run_nested_cv_experiment.py, the pipeline behind the paper's
# numbers, builds the selector as `Lasso(random_state=0)` with alpha forced to
# 1e-4, so it inherits scikit-learn's `max_iter=1000`. The convenience wrapper
# `tdims_ext.get_representation_with_fs_selection` passes `max_iter=100000`.
#
# For f_dis = x^-2 and x^-1 the difference is nil: the selector converges in 256
# iterations and the two settings pick the identical feature set, Jaccard 1.0000.
# For f_dis = x it is not nil. Those features are raw bond counts, standardised
# but strongly collinear, and coordinate descent crawls: at 1000 iterations it
# stops unconverged with 233 features, at 100000 it converges after 7201 with
# 126, and the two sets overlap at Jaccard 0.27. It is also the difference
# between a tractable sweep and an intractable one, since the radius-2 raw-
# distance configurations are the slowest cells in the grid by two orders of
# magnitude.
#
# The experiment script wins, because it is the code that produced the published
# results. Set TDIMS_FS_MAX_ITER=100000 to run the other answer as a sensitivity
# check.
FS_ALPHA = 1e-4
FS_THRESHOLD = "mean"
FS_MAX_ITER = int(os.environ.get("TDIMS_FS_MAX_ITER", "1000"))
# A ceiling on what reaches TabPFN, so one endpoint cannot stall an unattended
# sweep. It is a guard, not a modelling choice: every time it binds is counted and
# printed, and on these data sets it never has.
MAX_FEATURES = 4000

# How many folds select features at once. Running selections in parallel does not
# make them finish sooner -- see `select_stream` -- so this is not a core count.
# It is how far ahead of the head the selector is allowed to work, and each fold
# in flight holds its own slice of the training rows, so it is the memory knob
# too. Two is enough to keep the GPU fed.
N_THREADS = int(os.environ.get("TDIMS_THREADS", "2"))


def cache_path(tag: str) -> Path:
    return cfg.TDIMS_DIR / f"{tag}.npz"


def config_csv() -> Path:
    return cfg.RESULTS_DIR / "tdims_config.csv"


# --- phase 1: the descriptor cache ---------------------------------------
def tdims_encode():
    """Import the authors' encoder from its checkout, failing loudly if absent."""
    src = TDIMS_HOME / "src"
    if not (src / "tdims" / "load.py").exists():
        raise SystemExit(
            f"no TDiMS source at {src}\n"
            "set TDIMS_HOME to a checkout of IBM/materials models/tdims"
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tdims.load import encode

    return encode


def build_features(df: pd.DataFrame, tags: list[str], force: bool) -> None:
    """Encode every molecule once per configuration and cache it sparse.

    The descriptor never sees a label, so this is done on the whole table at once:
    the set of substructure pairs a molecule is scored against is, by construction,
    the set found across the data set, exactly as in the paper. Which molecules are
    *fit* on is decided fold by fold later, from the same masks every other arm uses.
    """
    encode = tdims_encode()
    smiles = df[cfg.SMILES_COL].tolist()
    cfg.TDIMS_DIR.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        path = cache_path(tag)
        if path.exists() and not force:
            with np.load(path, allow_pickle=False) as cached:
                print(f"{tag:<22} cached  {tuple(cached['shape'])}")
            continue

        spec = CONFIGS[tag]
        start = time.time()
        X, keys = encode(
            smiles,
            radius=spec["radius"],
            func_dis=spec["func_dis"],
            func_merge=spec["func_merge"],
            fragment_set=spec["fragment_set"],
            atom_set=True,
            fingerprint_set=True,
        )
        X = sparse.csr_matrix(np.asarray(X, dtype=np.float32))

        # An all-zero row means RDKit could not read that SMILES. 01_make_folds.py
        # already drops those, so this is an assertion, not a fallback: a silent
        # zero vector would train on a molecule the other arms did not.
        empty = np.asarray(X.getnnz(axis=1)).ravel() == 0
        if empty.any():
            raise SystemExit(
                f"{tag}: {int(empty.sum())} molecules got an all-zero TDiMS vector, "
                f"first is {df[cfg.SMILES_COL].iloc[int(np.flatnonzero(empty)[0])]!r}"
            )

        np.savez_compressed(
            path,
            data=X.data, indices=X.indices, indptr=X.indptr, shape=np.asarray(X.shape),
            names=df[cfg.ID_COL].to_numpy().astype(str),
            keys=np.array(list(keys), dtype=str),
        )
        print(f"{tag:<22} {X.shape[0]} x {X.shape[1]:<7} nnz={X.nnz:>9,} "
              f"({time.time() - start:.0f}s)", flush=True)


def load_features(df: pd.DataFrame, tag: str) -> np.ndarray:
    path = cache_path(tag)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 15_run_tdims.py --features first")
    with np.load(path, allow_pickle=False) as cached:
        names = cached["names"]
        if len(names) != len(df) or not np.array_equal(
            names, df[cfg.ID_COL].to_numpy().astype(str)
        ):
            raise SystemExit(
                f"{path.name} does not line up with {cfg.MASTER_CSV.name} -- "
                "re-run with --features --force"
            )
        X = sparse.csr_matrix(
            (cached["data"], cached["indices"], cached["indptr"]),
            shape=tuple(cached["shape"]),
        )
    # Densified here, deliberately. The matrix is very sparse, so keeping it
    # sparse looks like the obvious choice -- but the LASSO selector below runs
    # on it once per endpoint and fold, and scikit-learn's sparse coordinate
    # descent is about 25x slower than its dense path on these shapes (110s
    # against 4s on Biogen at radius 2). The largest configuration here is
    # 7,608 x 144,220 float32, under 5 GB, so the trade is worth making once per
    # configuration rather than paying it back on every fit.
    X = np.asarray(X.todense(), dtype=np.float32)

    # `clip_gt1_to0`, the first step of the authors' own pipeline in
    # experiments/run_nested_cv_experiment.py: a feature value above the
    # threshold is set to zero, not clamped to it. Under f_dis = x^-2 or x^-1 a
    # value above 1 means the two substructures of the pair are less than one
    # bond apart on average -- they overlap -- and the pipeline drops those
    # rather than letting them dominate. Raw distances are all above 1, so the
    # f_dis = x configurations get a threshold of 1000 instead, which is the
    # authors' number. This is an elementwise, label-free transform, so it
    # belongs here with the descriptor rather than inside the fold.
    threshold = 1000.0 if CONFIGS[tag]["func_dis"] == 1 else 1.0
    X[X > threshold] = 0.0
    return X


# --- the fold itself -----------------------------------------------------
def fold_masks(df: pd.DataFrame, folds: pd.DataFrame, endpoint: str,
               repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit, validation and test rows for one fold.

    The fit and test masks are the two 02_run_lightgbm.py uses, so this arm trains
    on exactly the molecules every other arm trains on. The validation mask is the
    held-out fifth, which is used only to choose the descriptor configuration and
    never to fit or to score.
    """
    held_out = folds[folds["repeat"] == repeat].set_index(cfg.ID_COL)["fold"]
    fold_of = df[cfg.ID_COL].map(held_out).to_numpy()  # NaN for the test molecules

    measured = df[endpoint].notna().to_numpy()
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    fit = measured & ~is_test & (fold_of != fold)
    val = measured & ~is_test & (fold_of == fold)
    test = measured & is_test
    return fit, val, test


def select_features(X: np.ndarray, y: np.ndarray, fit: np.ndarray, tag: str):
    """The rest of the authors' preprocessing, fitted on this fold's rows only.

    Three steps of their pipeline, in their order, after the clip that
    `load_features` already applied:

      sparse_filter   drop every feature that is zero in all of this fold's
                      training molecules (their `min_samples=1`)
      scaler          passthrough for f_dis = x^-2 and x^-1, whose values are
                      already on a common scale; `StandardScaler(with_mean=False)`
                      for f_dis = x, whose values are raw bond counts
      select          `SelectFromModel` around a LASSO at alpha=1e-4, threshold
                      "mean", with their `SafeSelectFromModel` guarantee that at
                      least one feature survives

    Nothing here sees a label outside `fit`. Returns the selected column indices
    and the scale to divide them by, so the caller can slice the validation and
    test rows straight out of the full matrix without materialising it twice.
    """
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import Lasso

    X_fit = X[fit]
    kept = np.flatnonzero((X_fit != 0).any(axis=0))
    X_fit = X_fit[:, kept]

    scale = None
    if CONFIGS[tag]["func_dis"] == 1:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(with_mean=False).fit(X_fit)
        scale = scaler.scale_.astype(np.float32)
        X_fit = X_fit / scale

    selector = SelectFromModel(
        estimator=Lasso(random_state=0, alpha=FS_ALPHA, max_iter=FS_MAX_ITER, tol=1e-4),
        threshold=FS_THRESHOLD,
    ).fit(X_fit, y[fit])
    support = selector.get_support()
    coefs = np.abs(np.ravel(selector.estimator_.coef_))

    capped = False
    if support.sum() > MAX_FEATURES:
        # Keep the largest coefficients. Only a guard; see MAX_FEATURES.
        support = np.zeros_like(support)
        support[np.argsort(coefs)[::-1][:MAX_FEATURES]] = True
        capped = True
    if not support.any():
        # `SafeSelectFromModel`: never hand the estimator an empty matrix.
        support[int(np.argmax(coefs))] = True

    columns = kept[support]
    return columns, (None if scale is None else scale[support]), capped


def select_stream(X: np.ndarray, tag: str, items: list[tuple]):
    """Yield (key, selection) in order, selecting ahead of whoever consumes it.

    Feature selection, not the head, is what this arm costs: a LASSO over 145,000
    to 168,000 radius-2 features, refitted for every endpoint and every fold.
    Running several of those at once buys nothing -- measured at 1.05x on six
    folds whether the workers are threads or forked processes, which rules out
    the GIL and leaves memory bandwidth, since dense coordinate descent streams
    a matrix of about a gigabyte through the cache once per iteration.

    What is worth overlapping is selection against the head, because one saturates
    memory and the other sits on the GPU. So every fold is submitted up front and
    the results are handed back in order: while the caller has one fold's features
    in TabPFN, the pool is already selecting the next fold's.

    `items` is a list of (key, y, fit_mask). `N_THREADS` bounds how many folds are
    in flight and so bounds the memory, since each holds its own slice of the
    training rows. Nothing about the result depends on the scheduling --
    `select_features` reads `X` and writes nothing shared.
    """
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        futures = [
            (key, pool.submit(select_features, X, y, fit, tag))
            for key, y, fit in items
        ]
        for key, future in futures:
            yield key, future.result()


def fold_matrix(X: np.ndarray, rows: np.ndarray, columns: np.ndarray,
                scale: np.ndarray | None) -> np.ndarray:
    """The selected features of some rows, on the scale the selector was fitted on."""
    out = X[np.ix_(rows, columns)]
    return out if scale is None else out / scale


def make_head(output_type: str):
    """TabPFN 3.5, built exactly as the `monroe35` arm builds it."""
    monroe_home = Path(
        os.environ.get("MONROE_HOME", Path.home() / "software" / "monroe")
    ).expanduser().resolve()
    if str(monroe_home) not in sys.path:
        sys.path.insert(0, str(monroe_home))
    from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn

    specs = default_ensemble_specs()

    def predict(X_fit, y_fit, X_test, seed):
        return fit_predict_tabpfn(
            X_fit, y_fit, X_test, is_classification=False,
            ensemble_specs=specs, seed=seed, output_type=output_type,
        )

    return predict


def check_head() -> None:
    """Refuse to run unless the installed tabpfn is the one that loads TabPFN 3.5.

    The checkpoint is chosen by the library, so a run in the TabPFN 3 environment
    would quietly write 3's predictions under 3.5's name. Same check
    09_run_monroe.py makes.
    """
    import tabpfn
    from tabpfn.settings import settings

    default = settings.tabpfn.model_version.value
    if default != "v3.5":
        raise SystemExit(
            f"this arm is TabPFN 3.5, but tabpfn {tabpfn.__version__} would load "
            f"{default} by default -- run it in the monroe35 environment"
        )
    print(f"tabpfn {tabpfn.__version__} loads TabPFN {default}, "
          f"recorded as {cfg.TDIMS_METHOD}")


# --- phase 2: choosing the configuration ---------------------------------
# Choosing among 24 configurations with the real head, on several folds, for
# every endpoint, costs several times what fitting the arm itself costs. So the
# choice is made in two stages, both on the held-out fifth and neither anywhere
# near the test set:
#
#   screen   all 24 configurations, one fold, scored by a ridge regression on the
#            selected features. Ridge is one of the estimator families the TDiMS
#            paper itself optimises over, and it costs a fraction of a second once
#            the features are selected. Its job is only to drop the configurations
#            that are clearly not in contention.
#   select   the survivors, on more folds, scored by the actual TabPFN 3.5 head.
#            This is what picks the configuration that goes to the test set.
#
# The screen can only cost accuracy by discarding a configuration ridge ranks low
# and TabPFN would have ranked top, so the shortlist is deliberately wider than
# one and results/<dataset>/tdims_screen.csv keeps every score it was built from.
def screen_csv() -> Path:
    return cfg.RESULTS_DIR / "tdims_screen.csv"


def run_screen(df: pd.DataFrame, folds: pd.DataFrame, endpoints: list[str],
               tags: list[str], repeat: int, fold: int, force: bool) -> None:
    """Rank every configuration per endpoint with a cheap ridge on one fold."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score

    path = screen_csv()
    done = set()
    if path.exists() and not force:
        prior = pd.read_csv(path)
        done = set(map(tuple, prior[["endpoint", "config"]].to_numpy()))
        print(f"screen: resuming, {len(done)} of {len(endpoints) * len(tags)} present")
    elif path.exists():
        path.unlink()

    wrote_header = path.exists()
    alphas = np.logspace(-2, 3, 6)  # the ridge grid of the TDiMS paper
    for tag in tags:
        if all((endpoint, tag) in done for endpoint in endpoints):
            continue
        X = load_features(df, tag)
        start = time.time()
        pending = [e for e in endpoints if (e, tag) not in done]
        masks = {e: fold_masks(df, folds, e, repeat, fold) for e in pending}
        stream = select_stream(
            X, tag, [(e, df[e].to_numpy(), masks[e][0]) for e in pending]
        )
        for endpoint, (columns, scale, _) in stream:
            y = df[endpoint].to_numpy()
            fit, val, _ = masks[endpoint]
            model = RidgeCV(alphas=alphas).fit(fold_matrix(X, fit, columns, scale), y[fit])
            pred = model.predict(fold_matrix(X, val, columns, scale))
            pd.DataFrame([{
                "endpoint": endpoint, "config": tag, "repeat": repeat, "fold": fold,
                "n_features": int(len(columns)),
                "ridge_val_r2": float(r2_score(y[val], pred)),
            }]).to_csv(path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
        print(f"screen {tag:<22} {X.shape[1]:>7} features  "
              f"({time.time() - start:.0f}s)", flush=True)
        del X


def shortlist(endpoints: list[str], tags: list[str], k: int) -> dict[str, list[str]]:
    """The k best-screening configurations for each endpoint."""
    scores = pd.read_csv(screen_csv())
    scores = scores[scores["config"].isin(tags)]
    out = {}
    for endpoint in endpoints:
        sub = scores[scores["endpoint"] == endpoint]
        if sub.empty:
            raise SystemExit(f"no screen scores for {endpoint} -- re-run --select")
        out[endpoint] = list(sub.nlargest(k, "ridge_val_r2")["config"])
    return out


def run_selection(df: pd.DataFrame, folds: pd.DataFrame, endpoints: list[str],
                  short: dict[str, list[str]], repeats: list[int], sel_folds: list[int],
                  predict, force: bool) -> None:
    """Score the shortlisted configurations with the real head, and write the table.

    Resumable at the row level: the scores are appended to tdims_config.csv as
    they are computed and any (endpoint, config, repeat, fold) already in the file
    is skipped, so a sweep that dies halfway does not start over.
    """
    from sklearn.metrics import r2_score

    tags = sorted({tag for chosen in short.values() for tag in chosen})
    path = config_csv()
    done = set()
    if path.exists() and not force:
        prior = pd.read_csv(path)
        done = set(map(tuple, prior[["endpoint", "config", "repeat", "fold"]].to_numpy()))
        print(f"select: resuming, {len(done)} scores present")
    elif path.exists():
        path.unlink()

    wrote_header = path.exists()
    capped_total = 0
    for tag in tags:
        on_shortlist = [e for e in endpoints if tag in short[e]]
        if not on_shortlist:
            continue
        X = load_features(df, tag)
        start = time.time()
        todo = [
            (endpoint, repeat, fold)
            for endpoint in on_shortlist
            for repeat in repeats
            for fold in sel_folds
            if (endpoint, tag, repeat, fold) not in done
        ]
        masks = {key: fold_masks(df, folds, *key) for key in todo}
        stream = select_stream(
            X, tag, [(key, df[key[0]].to_numpy(), masks[key][0]) for key in todo]
        )
        for key, (columns, scale, capped) in stream:
            endpoint, repeat, fold = key
            y = df[endpoint].to_numpy()
            fit, val, _ = masks[key]
            capped_total += capped
            Xf = fold_matrix(X, fit, columns, scale)
            Xv = fold_matrix(X, val, columns, scale)
            pred = predict(Xf, y[fit], Xv, cfg.fold_seed(repeat, fold))
            row = pd.DataFrame([{
                "endpoint": endpoint, "config": tag, "repeat": repeat,
                "fold": fold, "n_features": int(len(columns)),
                "n_fit": int(fit.sum()), "n_val": int(val.sum()),
                "val_r2": float(r2_score(y[val], pred)),
            }])
            row.to_csv(path, mode="a", header=not wrote_header, index=False)
            wrote_header = True
        print(f"{tag:<22} {X.shape[1]:>7} features  ({time.time() - start:.0f}s)",
              flush=True)
        del X

    if capped_total:
        print(f"note: the {MAX_FEATURES}-feature cap bound on {capped_total} fits")
    report_selection()


def chosen_configs(endpoints: list[str]) -> dict[str, str]:
    """The best-scoring configuration per endpoint, from the validation table."""
    path = config_csv()
    if not path.exists():
        raise SystemExit(f"{path} not found -- run 15_run_tdims.py --select first")
    scores = pd.read_csv(path)
    means = scores.groupby(["endpoint", "config"])["val_r2"].mean()
    chosen = {}
    for endpoint in endpoints:
        if endpoint not in means.index.get_level_values("endpoint"):
            raise SystemExit(f"no validation scores for {endpoint} -- re-run --select")
        chosen[endpoint] = means[endpoint].idxmax()
    return chosen


def report_selection() -> None:
    scores = pd.read_csv(config_csv())
    means = scores.groupby(["endpoint", "config"])["val_r2"].mean().unstack()
    print("\nvalidation R^2 by configuration (mean over the selection folds):")
    print(means.round(3).to_string())
    print("\nchosen:")
    for endpoint, row in means.iterrows():
        spread = row.max() - row.min()
        print(f"  {endpoint:<17} {row.idxmax():<22} "
              f"val R2 {row.max():.3f}   (best - worst config: {spread:.3f})")


# --- phase 3: the folds --------------------------------------------------
def pred_path(endpoint: str, repeat: int, fold: int) -> Path:
    return cfg.PRED_DIR / cfg.TDIMS_METHOD / f"{endpoint}_r{repeat}_f{fold}.csv"


def run_folds(df, X, folds, tag, keys, predict) -> int:
    """Predict the test set for every fold of one configuration.

    The selector runs ahead of the head, so the next fold's features are being
    chosen on the CPU while the current fold's are in TabPFN on the GPU. Each
    fold's predictions are written as it finishes, which is also the resume
    point if the run is interrupted.
    """
    masks = {key: fold_masks(df, folds, *key) for key in keys}
    stream = select_stream(
        X, tag, [(key, df[key[0]].to_numpy(), masks[key][0]) for key in keys]
    )

    capped_total = 0
    start = time.time()
    for done, (key, (columns, scale, capped)) in enumerate(stream, 1):
        endpoint, repeat, fold = key
        y = df[endpoint].to_numpy()
        fit, _, test = masks[key]
        capped_total += capped
        Xf = fold_matrix(X, fit, columns, scale)
        Xt = fold_matrix(X, test, columns, scale)
        pred = predict(Xf, y[fit], Xt, cfg.fold_seed(repeat, fold))

        test_df = df.loc[test]
        pd.DataFrame(
            {
                "method": cfg.TDIMS_METHOD,
                "endpoint": endpoint,
                "repeat": repeat,
                "fold": fold,
                cfg.ID_COL: test_df[cfg.ID_COL].to_numpy(),
                cfg.SMILES_COL: test_df[cfg.SMILES_COL].to_numpy(),
                "y_true": y[test],
                "y_pred": np.asarray(pred, dtype=float),
            },
            columns=cfg.PRED_COLUMNS,
        ).to_csv(pred_path(*key), index=False)
        if done % 25 == 0 or done == len(keys):
            print(f"{tag:<22} {done:>3}/{len(keys)} folds  "
                  f"({time.time() - start:.0f}s)", flush=True)
    return capped_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--features", action="store_true",
                        help="build the descriptor cache and stop")
    parser.add_argument("--select", action="store_true",
                        help="choose the configuration on the held-out fifth and stop")
    parser.add_argument("--config", nargs="+", default=sorted(CONFIGS), choices=sorted(CONFIGS),
                        help="which descriptor configurations --features and --select cover")
    parser.add_argument("--shortlist", type=int, default=4,
                        help="configurations per endpoint the ridge screen passes to the "
                             "TabPFN stage (default: 4 of 24)")
    parser.add_argument("--screen-fold", type=int, default=0, choices=cfg.FOLDS,
                        help="the single fold the ridge screen runs on (repeat 0)")
    parser.add_argument("--select-repeat", nargs="+", type=int, default=[0], choices=cfg.REPEATS,
                        help="repeats the configuration is chosen on (default: 0)")
    parser.add_argument("--select-fold", nargs="+", type=int, default=[0, 1, 2], choices=cfg.FOLDS,
                        help="folds the configuration is chosen on (default: 0 1 2)")
    parser.add_argument("--endpoint", nargs="+", default=cfg.TARGET_COLS, choices=cfg.TARGET_COLS)
    parser.add_argument("--repeat", nargs="+", type=int, default=cfg.REPEATS, choices=cfg.REPEATS)
    parser.add_argument("--fold", nargs="+", type=int, default=cfg.FOLDS, choices=cfg.FOLDS)
    parser.add_argument("--force", action="store_true",
                        help="recompute what is already on disk")
    parser.add_argument("--output-type", default="mean", choices=["mean", "median"],
                        help="TabPFN point estimate (default: mean, as the monroe arms use)")
    args = parser.parse_args()

    if not cfg.MASTER_CSV.exists():
        raise SystemExit(f"{cfg.MASTER_CSV} not found -- run 01_make_folds.py first")
    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)

    if args.features:
        build_features(df, args.config, args.force)
        return

    check_head()
    folds = pd.read_csv(cfg.FOLD_CSV)
    predict = make_head(args.output_type)

    if args.select:
        run_screen(df, folds, args.endpoint, args.config, 0, args.screen_fold, args.force)
        short = shortlist(args.endpoint, args.config, args.shortlist)
        print("\nshortlist from the ridge screen:")
        for endpoint, tags in short.items():
            print(f"  {endpoint:<17} {', '.join(tags)}")
        print()
        run_selection(df, folds, args.endpoint, short, args.select_repeat,
                      args.select_fold, predict, args.force)
        return

    chosen = chosen_configs(args.endpoint)
    (cfg.PRED_DIR / cfg.TDIMS_METHOD).mkdir(parents=True, exist_ok=True)

    # Grouped by configuration, so each feature matrix is loaded once even when
    # several endpoints chose the same one.
    capped_total = 0
    for tag in sorted(set(chosen[e] for e in args.endpoint)):
        mine = [e for e in args.endpoint if chosen[e] == tag]
        keys = [
            (endpoint, repeat, fold)
            for endpoint in mine
            for repeat in args.repeat
            for fold in args.fold
            if args.force or not pred_path(endpoint, repeat, fold).exists()
        ]
        if not keys:
            continue
        X = load_features(df, tag)
        capped_total += run_folds(df, X, folds, tag, keys, predict)
        del X

    if capped_total:
        print(f"note: the {MAX_FEATURES}-feature cap bound on {capped_total} folds")


if __name__ == "__main__":
    main()
