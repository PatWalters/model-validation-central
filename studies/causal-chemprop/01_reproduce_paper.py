#!/usr/bin/env python
"""Step 1: reproduce the paper's Table 1 from the authors' own release.

Causal-Chemprop is evaluated in the paper on eleven kinase tasks: four ABL1,
four AURKA and three SRC Murcko-scaffold groups, each held out of training, each
with one fixed anchor molecule whose pIC50 is revealed at inference. The
authors' Zenodo record (10.5281/zenodo.18197047) ships the splits, the eleven
trained Chemprop checkpoints, the eleven retrained-on-the-anchor checkpoints and
the eleven fitted structural causal models; their notebook IC50.ipynb hard-codes
the per-task hyperparameters and intrinsic causal contributions in
causal_utils.py.

Nothing is reimplemented here. This script imports their causal_utils, calls
their `train`, `data_similarity`, `test_counterfactual` and `test_regular`
exactly as IC50.ipynb does, and reads the points back off the matplotlib axes
the same way the notebook's own summary cell does. The upstream code reads
data/ and output/ relative to the working directory, so it runs from inside
the upstream checkout.

Three released arms per task, as in Table 1:

  causal_chemprop      released Chemprop + released SCM + the hard-coded ICCs
  chemprop             released Chemprop, ordinary inference
  chemprop_retrained   released Chemprop retrained with the anchor added

and two refits that ask how much of Table 1 is the release rather than the
recipe:

  causal_refit_scm     the released Chemprop, the SCM and ICCs refitted with
                       the authors' fit_causal on the released analog set
  causal_refit_all     Chemprop retrained from scratch with the authors'
                       train(fit_mcmpnn=True), then the SCM refitted on it

Each task writes predictions/paper/<arm>/<task>.csv, the resume point.

    python 01_reproduce_paper.py                        # released arms only
    python 01_reproduce_paper.py --refit scm all        # and the two refits
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

import config as cfg

# The release was written under torch 2.5. torch >= 2.6 refuses its Lightning
# checkpoints unless weights_only is off; the SCMs are plain pickles that the
# authors' code unpickles regardless, so this is the same trust in the release.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


def run_arm(cu, i: int, arm: str):
    """One task, one arm, through the authors' own functions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if arm == "chemprop_retrained":
        model = cu.train(i=i, mpnn_hidden_size=200, fnn_hidden_size=200,
                         fit_mcmpnn=False, fit_causal=False, retrain=True)
        cu.test_regular(i, model, ax)
    elif arm == "chemprop":
        model = cu.train(i=i, mpnn_hidden_size=200, fnn_hidden_size=200,
                         fit_mcmpnn=False, fit_causal=False, retrain=False)
        cu.test_regular(i, model, ax)
    else:
        fit_mcmpnn = arm == "causal_refit_all"
        fit_causal = arm in ("causal_refit_scm", "causal_refit_all")
        model = cu.train(i=i, mpnn_hidden_size=200, fnn_hidden_size=200,
                         fit_mcmpnn=fit_mcmpnn, fit_causal=fit_causal, retrain=False)
        _, analog_similarity = cu.data_similarity(i)
        cu.test_counterfactual(i, model, analog_similarity, False, ax)
    offsets = ax.collections[0].get_offsets()
    plt.close(fig)
    # The ordinary arms' own prediction for the anchor, which the anchor-offset
    # correction needs. Causal-Chemprop is never asked to predict its anchor.
    anchor_pred = np.nan
    if arm in ("chemprop", "chemprop_retrained"):
        import cc_kit
        anchor_pred = float(cc_kit.predict_regular(model, [pd.read_csv(cu.TEST_FPATH[i])["SMI"].iloc[0]])[0])
    return offsets[:, 0], offsets[:, 1], getattr(model, "icc", None), anchor_pred


def dump_splits(cu) -> None:
    """The training/validation split train() makes, for arms fitted elsewhere.

    causal_utils.train splits the unique SMILES of the training file 80:20 with
    astartes at seed 8675309. 07_paper_atomjepa.py early-stops on the same
    validation molecules, so the two encoders are selected on the same data.
    """
    from astartes import train_test_split
    out = cfg.PROJECT_DIR / "work" / "paper_splits"
    out.mkdir(parents=True, exist_ok=True)
    for i, task in enumerate(cfg.PAPER_TASKS):
        df = pd.read_csv(cu.TRAINING_FPATH[i])
        tr, va = train_test_split(pd.unique(df["SMI"]), random_state=cu.RANDOM_SEED,
                                  train_size=0.8, test_size=0.2)
        df.assign(split=np.where(df["SMI"].isin(tr), "train", np.where(df["SMI"].isin(va), "val", "none"))
                  )[["SMI", "logS", "split"]].to_csv(out / f"{task}.csv", index=False)
        anchor = pd.read_csv(cu.TEST_FPATH[i]).iloc[[0]]
        anchor[["SMI", "logS"]].to_csv(out / f"{task}_anchor.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", nargs="+", type=int, default=list(range(len(cfg.PAPER_TASKS))))
    parser.add_argument("--refit", nargs="*", default=[], choices=["scm", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    arms = list(cfg.PAPER_RELEASED_ARMS) + [f"causal_refit_{r}" for r in args.refit]
    out_root = cfg.PROJECT_DIR / "predictions" / "paper"

    # The upstream code reads data/ and output/ relative to the working
    # directory, and fit_causal pickles the refitted SCM back into the
    # checkpoint's own directory -- on top of the released one. So the
    # released arms run in one copy of the release and the refits in another.
    # GCM's joblib fan-out (n_jobs=-1) is what makes one refit fast on an idle
    # machine and what oversubscribes a busy one; pinned to one, as the sweep is.
    import dowhy.gcm.config
    dowhy.gcm.config.set_default_n_jobs(int(os.environ.get("GCM_N_JOBS", "1")))
    cfg.prepare_upstream_copies()
    sys.path.insert(0, str(cfg.UPSTREAM_COPY["released"]))
    os.chdir(cfg.UPSTREAM_COPY["released"])
    import causal_utils as cu

    dump_splits(cu)
    timing = []
    for i in args.task:
        task = cfg.PAPER_TASKS[i]
        test = pd.read_csv(cu.TEST_FPATH[i])
        for arm in arms:
            out = out_root / arm / f"{task}.csv"
            if out.exists() and not args.force:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            os.chdir(cfg.UPSTREAM_COPY["refit" if arm.startswith("causal_refit") else "released"])
            start = time.time()
            y_true, y_pred, icc, anchor_pred = run_arm(cu, i, arm)
            seconds = time.time() - start
            # The anchor is row 0 of the counterfactual file and is never scored.
            scored = test.iloc[1:]
            if len(scored) != len(y_true) or not (abs(scored["logS"].to_numpy() - y_true) < 1e-9).all():
                raise SystemExit(f"{task} {arm}: the scored rows do not line up with {cu.TEST_FPATH[i]}")
            pd.DataFrame({
                "task": task, "method": arm, cfg.SMILES_COL: test["SMI"].to_numpy(),
                "role": ["anchor"] + ["scored"] * len(scored),
                "y_true": np.r_[test["logS"].iloc[0], y_true], "y_pred": np.r_[anchor_pred, y_pred],
            }).to_csv(out, index=False)
            timing.append({"task": task, "method": arm, "seconds": seconds,
                           "n_icc": None if icc is None else len(icc)})
            print(f"{task:8s} {arm:20s} n={len(y_true):3d} {seconds:6.1f}s", flush=True)

    if timing:
        path = cfg.PROJECT_DIR / "results" / "paper" / "timing.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(timing).to_csv(path, mode="a", header=not path.exists(), index=False)


if __name__ == "__main__":
    main()
