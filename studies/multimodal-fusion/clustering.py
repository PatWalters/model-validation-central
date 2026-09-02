"""BitBIRCH-Lean clustering, shared by the two data sets.

The clusters group the cross-validation folds, so that no molecule family
straddles the fit/validation boundary and a fold measures generalisation to new
chemistry rather than memorisation of a series.

Settings are the defaults from the bblean best-practices notebook at
https://github.com/mqcomplab/bblean. The threshold is not a free parameter: it
is estimated from the similarity distribution of whichever data set is being
clustered, so the Biogen compounds get their own threshold rather than one
borrowed from the ExpansionRx set.
"""

import numpy as np
import pandas as pd

N_FEATURES = 2048
FP_KIND = "ecfp4"
N_SAMPLES_FOR_STD = 50
STD_MULTIPLIER = 4
BRANCHING_FACTOR = 50
RECLUSTER_ITERATIONS = 5


def assign_clusters(smiles: pd.Series) -> np.ndarray:
    """Cluster ids for a series of SMILES, one per row, in input order."""
    import bblean
    import bblean.similarity as isim

    fps = bblean.fps_from_smiles(smiles, pack=True, n_features=N_FEATURES, kind=FP_KIND)

    average_sim = isim.jt_isim_packed(fps)
    sample = isim.jt_stratified_sampling(fps, n_samples=N_SAMPLES_FOR_STD)
    sim_matrix = isim.jt_sim_matrix_packed(fps[sample])
    sim_matrix = sim_matrix[~np.eye(sim_matrix.shape[0], dtype=bool)]  # drop self-similarity
    std = float(np.std(sim_matrix))

    tree = bblean.BitBirch(
        branching_factor=BRANCHING_FACTOR,
        threshold=average_sim + STD_MULTIPLIER * std,
        merge_criterion="diameter",
    )
    tree.fit(fps)
    tree.recluster_inplace(
        iterations=RECLUSTER_ITERATIONS,
        extra_threshold=std,
        shuffle=False,
        verbose=False,
    )
    return np.asarray(tree.get_assignments())
