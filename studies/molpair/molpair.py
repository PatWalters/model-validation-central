"""MolPAIR, reimplemented from Lee, Lee, Jeong and Yoo, arXiv 2609.38744.

The authors' code link does not resolve, so this follows the paper's Section 3
and Appendix A. Equation numbers below are the paper's. In outline, for one
training set and one set of queries:

  1. Global context (Eq. 2). A TabPFN 3 reads each molecule's frozen CheMeleon
     fingerprint and predicts the query's mean mu_q and a scale
     u_q = (Q0.9 - Q0.1) / 2.563.
  2. Out-of-fold errors (Eq. 3). Five folds inside the training set; each fold
     is predicted from the other four, giving mu-, u- and e = y - mu- for every
     training molecule without its own label in the context.
  3. Relation context (Eqs. 4-6). Each training molecule b is paired with up to
     eight anchors a from its own OOF fold, exact one-cut matched pairs first
     and then by Morgan Tanimoto. The row is
        z_ab = [dphi, |dphi|, s, same scaffold, MMP, mu_a, mu_b, mu_b - mu_a, u_a, u_b]
     on RDKit2D descriptors, and the target is d_ab = e_b - e_a. At most 8,192
     rows, subsampled deterministically. A second TabPFN 3 reads them.
  4. Query correction (Eqs. 7-10). Each query takes up to 32 anchors from the
     whole training set with Tanimoto >= 0.20, ranked the same way. The pair is
     predicted in both directions and antisymmetrised, d = (P(z_aq) - P(z_qa)) / 2,
     each anchor proposes c = e_a + d, the proposals are averaged with weights
     s^2 / u_R, and the average is scaled by the gate
     lambda = clip((max s - tau) / (1 - tau), 0, 1) before it is added to mu_q.
     A query with no accepted anchor keeps mu_q.

Nothing in here touches a test label: the queries contribute only their
features, and everything fitted -- imputation, feature selection, both TabPFN
contexts -- is fitted on the training molecules.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold


@dataclass
class Molecules:
    """Every per-molecule input, indexed by master.csv row."""

    chemeleon: np.ndarray       # (N, 2048) float32
    rdkit2d: np.ndarray         # (N, D) float64, NaN where a descriptor failed
    morgan: np.ndarray          # (N, bits) float32 0/1, for Tanimoto by matmul
    scaffold: np.ndarray        # (N,) str, "" for acyclic molecules
    mmp: sparse.csr_matrix      # (N, N) bool, symmetric

    @classmethod
    def load(cls, npz_path, n_molecules: int) -> "Molecules":
        f = np.load(npz_path, allow_pickle=False)
        bits = np.unpackbits(f["morgan"], axis=1).astype(np.float32)
        pairs = f["mmp_pairs"]
        ones = np.ones(len(pairs), dtype=bool)
        mmp = sparse.coo_matrix((ones, (pairs[:, 0], pairs[:, 1])),
                                shape=(n_molecules, n_molecules)).tocsr()
        return cls(f["chemeleon"], f["rdkit2d"], bits, f["scaffold"], (mmp + mmp.T).tocsr())

    def tanimoto(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        a, b = self.morgan[rows], self.morgan[cols]
        inter = a @ b.T
        union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
        return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    def is_mmp(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return self.mmp[rows][:, cols].toarray()


class Preprocessor:
    """Median imputation, constant-feature filter, variance selection (App. A.1).

    Fitted on the training molecules only and then applied to everything.
    """

    def __init__(self, max_features: int):
        self.max_features = max_features

    def fit(self, X: np.ndarray) -> "Preprocessor":
        self.median = np.nanmedian(X, axis=0)
        self.median = np.where(np.isfinite(self.median), self.median, 0.0)
        var = np.var(self._impute(X), axis=0)
        keep = np.flatnonzero(var > 0)
        # Highest variance first, ties broken by column order so it is stable.
        order = keep[np.lexsort((keep, -var[keep]))]
        self.columns = np.sort(order[: self.max_features])
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        bad = ~np.isfinite(X)
        X[bad] = np.take(self.median, np.nonzero(bad)[1])
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._impute(X)[:, self.columns]


def rank_anchors(score: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The top-k columns of each row by score, best first, and which are real.

    A score of -inf marks a candidate that is not allowed; rows with fewer than
    k allowed candidates come back with `valid` false in the remaining slots.
    """
    k = min(k, score.shape[1])
    part = np.argpartition(-score, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(score, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    idx = np.take_along_axis(part, order, axis=1)
    return idx, np.isfinite(np.take_along_axis(score, idx, axis=1))


class MolPAIR:
    """One fit/predict of the full method for one training set."""

    def __init__(self, mols: Molecules, settings: dict, seed: int, device: str = "auto"):
        self.mols = mols
        self.s = settings
        self.seed = seed
        self.device = device

    # --- the two TabPFNs -------------------------------------------------
    def _tabpfn(self, seed: int, cache: bool = False):
        """A TabPFN 3 regressor. `cache` keeps the encoded context between predict
        calls, which only the relation model needs: it predicts tens of thousands
        of pair rows in batches against one context. Uncached, every batch
        re-encodes the context; cached, the outputs agree to floating-point
        precision (0.006 at most on a unit-variance target in testing) at half the
        time."""
        from tabpfn import TabPFNRegressor

        return TabPFNRegressor(n_estimators=self.s["n_estimators"], random_state=seed,
                               device=self.device,
                               fit_mode="fit_with_cache" if cache else "fit_preprocessors")

    def _predict(self, model, X: np.ndarray, batch: int | None = None):
        """Predictive mean and the decile scale, (Q0.9 - Q0.1) / 2.563."""
        batch = batch or len(X)
        mu, u = [], []
        for start in range(0, len(X), batch):
            out = model.predict(X[start:start + batch], output_type="main")
            q = out["quantiles"]  # deciles 0.1 ... 0.9
            mu.append(np.asarray(out["mean"], dtype=np.float64))
            u.append((np.asarray(q[-1]) - np.asarray(q[0])) / self.s["quantile_to_sd"])
        return np.concatenate(mu), np.concatenate(u)

    # --- pair rows (Eq. 5) ------------------------------------------------
    def _pair_rows(self, R_a, R_b, s, scaf, mmp, mu_a, mu_b, u_a, u_b) -> np.ndarray:
        d = R_b - R_a
        aux = np.column_stack([s, scaf, mmp, mu_a, mu_b, mu_b - mu_a, u_a, u_b])
        return np.hstack([d, np.abs(d), aux]).astype(np.float32)

    def fit_predict(self, train: np.ndarray, y: np.ndarray, query: np.ndarray):
        """Predict `query` rows from `train` rows with labels `y`.

        Returns the global prediction, the corrected prediction and a frame of
        per-query diagnostics.
        """
        m, s = self.mols, self.s
        n = len(train)

        # 1. Global context and the query's global prediction.
        gpre = Preprocessor(s["global_max_features"]).fit(m.chemeleon[train])
        G_train, G_query = gpre.transform(m.chemeleon[train]), gpre.transform(m.chemeleon[query])
        glob = self._tabpfn(self.seed).fit(G_train, y)
        mu_q, u_q = self._predict(glob, G_query)

        # 2. Out-of-fold predictions inside the training set.
        oof_fold = np.empty(n, dtype=int)
        mu_oof, u_oof = np.empty(n), np.empty(n)
        kf = KFold(s["oof_folds"], shuffle=True, random_state=self.seed)
        for k, (fit_idx, held_idx) in enumerate(kf.split(G_train)):
            model = self._tabpfn(self.seed).fit(G_train[fit_idx], y[fit_idx])
            mu_oof[held_idx], u_oof[held_idx] = self._predict(model, G_train[held_idx])
            oof_fold[held_idx] = k
        e = y - mu_oof

        # 3. Relation context: up to eight same-fold anchors per training molecule.
        rpre = Preprocessor(s["relation_max_descriptors"]).fit(m.rdkit2d[train])
        R_train, R_query = rpre.transform(m.rdkit2d[train]), rpre.transform(m.rdkit2d[query])
        sim_tt = m.tanimoto(train, train)
        mmp_tt = m.is_mmp(train, train)
        scaf = m.scaffold
        a_list, b_list = [], []
        for k in range(s["oof_folds"]):
            members = np.flatnonzero(oof_fold == k)
            score = sim_tt[np.ix_(members, members)] + 2.0 * mmp_tt[np.ix_(members, members)]
            np.fill_diagonal(score, -np.inf)
            idx, ok = rank_anchors(score, s["train_anchors"])
            b_local = np.repeat(np.arange(len(members)), idx.shape[1])[ok.ravel()]
            a_list.append(members[idx.ravel()[ok.ravel()]])
            b_list.append(members[b_local])
        a, b = np.concatenate(a_list), np.concatenate(b_list)
        n_pairs_all = len(a)
        if len(a) > s["relation_context"]:
            keep = np.sort(np.random.default_rng(self.seed).choice(
                len(a), s["relation_context"], replace=False))
            a, b = a[keep], b[keep]
        same_scaf = (scaf[train[a]] == scaf[train[b]]) & (scaf[train[a]] != "")
        Z = self._pair_rows(R_train[a], R_train[b], sim_tt[a, b], same_scaf, mmp_tt[a, b],
                            mu_oof[a], mu_oof[b], u_oof[a], u_oof[b])
        rel = self._tabpfn(self.seed + s["relation_seed_offset"], cache=True).fit(Z, e[b] - e[a])

        # 4. Query anchors from the whole training set, then both directions.
        sim_qt = m.tanimoto(query, train)
        mmp_qt = m.is_mmp(query, train)
        score = np.where(sim_qt >= s["anchor_min_similarity"], sim_qt + 2.0 * mmp_qt, -np.inf)
        idx, ok = rank_anchors(score, s["query_anchors"])
        qi = np.repeat(np.arange(len(query)), idx.shape[1])[ok.ravel()]
        ai = idx.ravel()[ok.ravel()]
        s_qa = sim_qt[qi, ai]
        sc = ((scaf[query[qi]] == scaf[train[ai]]) & (scaf[train[ai]] != "")).astype(float)
        mm = mmp_qt[qi, ai]
        z_aq = self._pair_rows(R_train[ai], R_query[qi], s_qa, sc, mm,
                               mu_oof[ai], mu_q[qi], u_oof[ai], u_q[qi])
        z_qa = self._pair_rows(R_query[qi], R_train[ai], s_qa, sc, mm,
                               mu_q[qi], mu_oof[ai], u_q[qi], u_oof[ai])
        if len(qi):
            both, u_both = self._predict(rel, np.vstack([z_aq, z_qa]), s["relation_batch"])
            p_aq, p_qa = both[: len(qi)], both[len(qi):]
            d = (p_aq - p_qa) / 2                                   # Eq. 7
            u_r = (u_both[: len(qi)] + u_both[len(qi):]) / 2        # Eq. 8
        else:
            d = u_r = np.zeros(0)
        c = e[ai] + d
        w_raw = np.maximum(s_qa, 1e-4) ** 2 / np.maximum(u_r, 1e-4)  # Eq. 9

        frame = pd.DataFrame({"q": qi, "w": w_raw, "wc": w_raw * c, "s": s_qa})
        g = frame.groupby("q").agg(w=("w", "sum"), wc=("wc", "sum"), s_max=("s", "max"),
                                   n=("s", "size"))
        n_anchor = np.zeros(len(query), dtype=int)
        s_max = np.zeros(len(query))
        weighted = np.zeros(len(query))
        n_anchor[g.index] = g["n"].to_numpy()
        s_max[g.index] = g["s_max"].to_numpy()
        weighted[g.index] = (g["wc"] / g["w"]).to_numpy()
        tau = s["gate_tau"]
        lam = np.clip((s_max - tau) / (1 - tau), 0, 1)               # Eq. 10
        lam[n_anchor == 0] = 0.0
        correction = lam * weighted
        diag = pd.DataFrame({
            "mu": mu_q, "u": u_q, "n_anchors": n_anchor, "max_sim": s_max,
            "max_sim_all": sim_qt.max(axis=1), "gate": lam, "correction": correction,
        })
        self.info = {"n_train": n, "n_pairs_available": n_pairs_all, "n_relation": len(a),
                     "n_query_pairs": len(qi), "relation_features": Z.shape[1]}
        return mu_q, mu_q + correction, diag
