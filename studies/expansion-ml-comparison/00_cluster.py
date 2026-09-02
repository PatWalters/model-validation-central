"""Regenerate the `cluster` column of expansion_log_scaled.csv from the SMILES.

The 651 clusters group the cross-validation folds. No molecule family straddles
the fit/validation boundary, so a fold measures generalisation to new chemistry
rather than memorisation of a series.

Clustering is BitBIRCH-Lean at its default settings, in `clustering.py`, which
the Biogen preparation script uses as well.

Running this script on the shipped CSV reproduces the stored labels exactly
(adjusted Rand index 1.0), so the column is a derived quantity, not a fixture.

    python 00_cluster.py --check      # verify the stored column
    python 00_cluster.py --write      # rewrite it
"""

import argparse

import pandas as pd

import config as cfg
from clustering import assign_clusters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the cluster column in place")
    parser.add_argument("--check", action="store_true", help="compare against the stored column")
    opts = parser.parse_args()
    if not (opts.write or opts.check):
        parser.error("choose --check or --write")

    df = pd.read_csv(cfg.RAW_CSV)
    ids = assign_clusters(df[cfg.SMILES_COL])
    sizes = pd.Series(ids).value_counts()
    print(f"{len(sizes)} clusters, largest {sizes.iloc[0]}, {int((sizes == 1).sum())} singletons")

    if opts.check:
        from sklearn.metrics import adjusted_rand_score

        # Compare the partitions, not the labels: ids are arbitrary names for groups.
        score = adjusted_rand_score(df[cfg.CLUSTER_COL], ids)
        print(f"adjusted Rand index against the stored column: {score:.6f}")
        if score < 1.0:
            raise SystemExit("clustering does not reproduce the stored labels")
        print("stored labels reproduced exactly")

    if opts.write:
        df[cfg.CLUSTER_COL] = ids
        df.to_csv(cfg.RAW_CSV, index=False)
        print(f"wrote {cfg.RAW_CSV}")


if __name__ == "__main__":
    main()
