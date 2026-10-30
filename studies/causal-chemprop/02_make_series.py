#!/usr/bin/env python
"""Step 2: the series, their anchors, and which of them Causal-Chemprop can run on.

The paper's unit is a held-out scaffold group with one anchor whose value is
revealed. Here the unit is a held-out test cluster: the protocol's test set is
whole BitBIRCH-Lean clusters on Biogen and largely so on ExpansionRx, so a test
cluster is a chemical series that training never saw in full -- the setting the
paper describes. Murcko scaffolds would not do: Biogen's 1,058 test molecules
carry 932 distinct scaffolds and no endpoint has a single seven-member group.

For every endpoint, a series is a test cluster with at least MIN_SERIES
measured molecules. One member, drawn with a fixed seed from the measured ones,
is its anchor; every other member is scored. The draw depends on nothing but
the seed, the endpoint and the cluster, never on a label or a prediction, and
it is the same for every arm and every fold.

Causal-Chemprop needs an analog set to fit its SCM on: the training molecules
whose Murcko scaffold has ECFP4 Tanimoto >= 0.25 to the series scaffold
(`find_similar_scaffolds_diverse_decorators` in the authors' data_processing.py,
re-implemented here with the same fingerprint and the same comparison, and
checked against theirs on the first series of every endpoint). The paper calls a
group infeasible below about 90 analogs. A series is kept only if every one of
the 25 fold training sets gives it at least MIN_ANALOGS, so the scored
molecules are the same in every fold, for every arm.

Writes data/<ds>/series.csv: one row per (endpoint, series, molecule) with its
role, plus series-level columns -- size, whether the cluster also has training
members, the smallest analog set over the folds, and feasibility.

    python 02_make_series.py
    ADME_DATASET=biogen python 02_make_series.py
"""

import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import config as cfg

from rdkit import RDLogger

# The authors' fingerprint calls are the deprecated API; one warning per molecule.
RDLogger.DisableLog("rdApp.*")


def scaffold_smiles(smiles: str) -> str:
    """get_murcko_scaffold from the authors' data_processing.py."""
    sys.path.insert(0, str(cfg.UPSTREAM_HOME))
    from data_processing import get_murcko_scaffold
    return get_murcko_scaffold(Chem.MolFromSmiles(smiles))


def scaffold_fp(scaffold: str):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(scaffold), radius=2, nBits=2048)


def check_against_upstream(anchor_scaffold: str, train: pd.DataFrame, ours: np.ndarray) -> None:
    """Our vectorised analog search has to pick exactly the authors' molecules."""
    from data_processing import find_similar_scaffolds_diverse_decorators
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        theirs = find_similar_scaffolds_diverse_decorators(
            target_scaffold_smiles=anchor_scaffold,
            dataset_smiles=train[cfg.SMILES_COL].tolist(),
            dataset_ids=train[cfg.ID_COL].tolist(),
            scaffold_similarity_threshold=cfg.ANALOG_SCAFFOLD_SIM,
        )
    theirs = set(theirs["molecule_id"]) if len(theirs) else set()
    if theirs != set(train[cfg.ID_COL].to_numpy()[ours]):
        raise SystemExit(f"analog search disagrees with upstream for scaffold {anchor_scaffold}")


def main() -> None:
    cfg.ensure_dirs()
    df = pd.read_csv(cfg.MASTER_CSV)
    folds = pd.read_csv(cfg.FOLD_CSV)
    df["scaffold"] = [scaffold_smiles(s) for s in df[cfg.SMILES_COL]]
    fps = [scaffold_fp(s) for s in df["scaffold"]]
    is_test = (df[cfg.SET_COL] == "test").to_numpy()
    train_clusters = set(df.loc[~is_test, cfg.CLUSTER_COL])
    fold_of = {r: df[cfg.ID_COL].map(g.set_index(cfg.ID_COL)["fold"]).to_numpy()
               for r, g in folds.groupby("repeat")}

    rows, summary = [], []
    for e_idx, endpoint in enumerate(cfg.TARGET_COLS):
        measured = df[endpoint].notna().to_numpy()
        test = df[is_test & measured]
        sizes = test[cfg.CLUSTER_COL].value_counts()
        checked = False
        for cluster in sorted(sizes[sizes >= cfg.MIN_SERIES].index):
            members = test[test[cfg.CLUSTER_COL] == cluster]
            rng = np.random.default_rng([cfg.ANCHOR_SEED, e_idx, int(cluster)])
            anchor = members[cfg.ID_COL].to_numpy()[rng.integers(len(members))]
            a_row = df.index[df[cfg.ID_COL] == anchor][0]
            sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[a_row], fps))
            is_analog = sims >= cfg.ANALOG_SCAFFOLD_SIM
            n_analogs = []
            for r in cfg.REPEATS:
                for f in cfg.FOLDS:
                    fit = measured & ~is_test & (fold_of[r] != f)
                    n_analogs.append(int((is_analog & fit).sum()))
                    if not checked:
                        sub = df[fit].reset_index(drop=True)
                        check_against_upstream(df.at[a_row, "scaffold"], sub, is_analog[fit])
                        checked = True
            feasible = min(n_analogs) >= cfg.MIN_ANALOGS
            for _, m in members.iterrows():
                rows.append({
                    "endpoint": endpoint, "series": int(cluster), cfg.ID_COL: m[cfg.ID_COL],
                    "role": "anchor" if m[cfg.ID_COL] == anchor else "scored",
                })
            summary.append({
                "endpoint": endpoint, "series": int(cluster), "n": len(members),
                "anchor": anchor, "anchor_scaffold": df.at[a_row, "scaffold"],
                "novel": int(cluster) not in train_clusters,
                "min_analogs": min(n_analogs), "max_analogs": max(n_analogs),
                "feasible": feasible,
            })

    members = pd.DataFrame(rows)
    series = pd.DataFrame(summary)
    out = members.merge(series[["endpoint", "series", "n", "novel", "min_analogs", "feasible"]],
                        on=["endpoint", "series"])
    out.to_csv(cfg.SERIES_CSV, index=False)
    series.to_csv(cfg.DATA_DIR / "series_summary.csv", index=False)

    print(f"{cfg.ACTIVE.label}: series of >= {cfg.MIN_SERIES} measured test molecules, "
          f"feasible if every fold gives >= {cfg.MIN_ANALOGS} analogs\n")
    tab = series.groupby("endpoint").agg(
        series=("series", "size"), feasible=("feasible", "sum"),
        novel=("novel", "sum"),
        scored=("n", lambda n: int((n[series.loc[n.index, "feasible"]] - 1).sum())),
        median_analogs=("min_analogs", "median"),
    ).reindex(cfg.TARGET_COLS).fillna(0).astype(int)
    print(tab.to_string())
    print(f"\n{int(series['feasible'].sum())} feasible series, "
          f"{int(tab['scored'].sum())} scored molecules")


if __name__ == "__main__":
    main()
