#!/usr/bin/env python
"""Step 9: which of these molecules were in the PCBA pre-training set, and with what labels?

Atom-JEPA-Bio's second stage trains on PCBA as Monroe prepares it: the
1,328-assay table of the Graphium release (Zenodo record 8024997,
PCBA_1328_1564k.parquet), deduplicated by InChI, with the test splits of every
Polaris and MoleculeACE benchmark moved to validation, and the assays filtered
to the 1,089 with >= 100 actives, >= 100 inactives and >= 1,000 labels. Monroe
pre-trains on the same preparation, so whatever this finds, both arms share it.

Two things are counted for every molecule in master.csv:

  structure  an exact standard-InChI match to a PCBA row (Monroe's own dedup
             key, computed the same way, with datamol's to_inchi), and a looser
             match on the 14-character InChIKey connectivity block
  labels     for an exact match, how many PCBA assays label it, and whether any
             of those is an ADME-like assay: AID 1996 (aqueous solubility,
             Burnham/NCGC kinetic solubility), and the CYP450 inhibition panel

The Polaris hold-out cannot be reproduced without a Polaris login, and it moves
the Polaris test splits of the six Biogen (Fang et al.) benchmarks -- a
different split from ours -- out of PCBA training. So an exact match here is an
upper bound on what reached training, not a count of it.

    curl -L -o pcba_1328.zip https://zenodo.org/api/records/8024997/files/pcba_1328.zip/content
    unzip pcba_1328.zip
    python 09_check_pcba_overlap.py PCBA_1328_1564k.parquet      # both data sets

Writes results/<ds>/pcba_overlap.csv, one row per molecule, which 06_report.py
uses to split the test set into molecules PCBA has and has not seen.
"""

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config as cfg

# Assays that measure something close to an endpoint here.
ADME_AIDS = {
    1996: "aqueous solubility",
    1851: "CYP450 panel",
    410: "CYP1A2",
    883: "CYP2C9",
    884: "CYP3A4",
    891: "CYP2D6",
    899: "CYP2C19",
}


def _inchi(smiles: list[str]) -> list[tuple[str | None, str | None]]:
    import datamol as dm
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles:
        try:
            i = dm.to_inchi(s)
        except Exception:
            i = None
        out.append((i, Chem.InchiToInchiKey(i) if i else None))
    return out


def inchis(smiles: list[str], workers: int) -> tuple[list, list]:
    chunks = [smiles[i:i + 5000] for i in range(0, len(smiles), 5000)]
    with Pool(workers) as pool:
        res = [x for part in pool.map(_inchi, chunks) for x in part]
    return [r[0] for r in res], [r[1] for r in res]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pcba_parquet")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    table = pq.read_table(args.pcba_parquet)
    assay_cols = [c for c in table.column_names if c.startswith("assayID-")]
    smiles = table.column("SMILES").to_pylist()
    print(f"PCBA: {len(smiles):,} rows, {len(assay_cols)} assays; computing InChI ...", flush=True)
    p_inchi, p_key = inchis(smiles, args.workers)
    pcba = pd.DataFrame({"row": np.arange(len(smiles)), "inchi": p_inchi, "key": p_key})
    pcba = pcba.dropna(subset=["inchi"])
    skeletons = set(pcba["key"].str[:14])
    first_row = pcba.drop_duplicates("inchi").set_index("inchi")["row"]
    labels = table.select(assay_cols).to_pandas()
    n_labels = labels.notna().sum(axis=1).to_numpy()
    print(f"  {pcba['inchi'].nunique():,} unique InChI", flush=True)

    for name in cfg.DATASETS:
        master = pd.read_csv(cfg.PROJECT_DIR / "data" / name / "master.csv")
        m_inchi, m_key = inchis(master[cfg.SMILES_COL].tolist(), args.workers)
        out = master[[cfg.ID_COL, cfg.SMILES_COL, cfg.SET_COL]].copy()
        out["inchi_key"] = m_key
        rows = pd.Series(m_inchi).map(first_row)
        out["pcba_exact"] = rows.notna().to_numpy()
        out["pcba_skeleton"] = [k is not None and k[:14] in skeletons for k in m_key]
        out["pcba_n_labels"] = [int(n_labels[int(r)]) if pd.notna(r) else 0 for r in rows]
        for aid, what in ADME_AIDS.items():
            col = f"assayID-{aid}"
            vals = labels[col].to_numpy() if col in labels else None
            out[f"aid{aid}"] = [
                (vals[int(r)] if vals is not None and pd.notna(r) else np.nan) for r in rows
            ]
        path = cfg.PROJECT_DIR / "results" / name / "pcba_overlap.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)

        print(f"\n{cfg.DATASETS[name].label}: {len(out):,} molecules -> {path.name}")
        for part, g in out.groupby(cfg.SET_COL):
            n = len(g)
            print(f"  {part:<5} exact {g['pcba_exact'].sum():>5,} of {n:,} "
                  f"({100 * g['pcba_exact'].mean():.1f}%), skeleton {g['pcba_skeleton'].sum():,}, "
                  f"median assays labelled when matched "
                  f"{g.loc[g['pcba_exact'], 'pcba_n_labels'].median() if g['pcba_exact'].any() else 0:.0f}")
            for aid, what in ADME_AIDS.items():
                k = int(g[f"aid{aid}"].notna().sum())
                if k:
                    print(f"        AID {aid} ({what}): {k} labelled, "
                          f"{int((g[f'aid{aid}'] == 1).sum())} active")


if __name__ == "__main__":
    main()
