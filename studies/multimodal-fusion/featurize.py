"""The four modalities, as features.

Two of them are fixed properties of a molecule and are cached once per data set:
RDKit's descriptor block and a Mol2Vec embedding. The other two come out of
encoders trained on the fold's own training molecules, so what lives here is the
input those encoders consume -- a molecular graph and a character-indexed SMILES
string -- and the encoders themselves are in `nets.py`.

Everything follows `src/fusion_early.py` and `src/data_cleaning.py` of
github.com/jwasswa2023/Multimodal_Fusion (MIT), which is a set of Colab dumps
rather than an importable package, plus the featurization the repository loads
from a `.npz` it never commits the generator for. Where a choice had to be made
because the released material does not record one, the docstring says so.
"""

from __future__ import annotations

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

# --- RDKit descriptors ---------------------------------------------------
# The paper reports 208, which is `len(Descriptors.descList)` for the RDKit its
# DeepChem featurizer was pinned to. Taking the list from the installed RDKit
# rather than hard-coding 208 keeps this honest about which build produced it;
# the count is recorded in the cache and printed by 02_modality_cache.py.
#
# `is_normalized` is the one genuinely unresolved choice. The Multimodal_Fusion
# cleaning script constructs `RDKitDescriptors()`, which does not normalise; the
# same author's Physpropnet uses `RDKitDescriptors(is_normalized=True)`. The
# script that actually built their feature matrix was never committed, so this
# cannot be settled from the release. Raw values are used here, matching the
# repository this work is reimplementing, and a tree ensemble is invariant to the
# monotone per-descriptor transform the other choice would apply anyway.
DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]


def rdkit_descriptors(smiles: list[str]) -> tuple[np.ndarray, list[str]]:
    """The RDKit descriptor block, one row per molecule.

    Columns that are non-finite for any molecule in the collection are dropped
    rather than imputed: the released pipeline drops the *rows* instead, which is
    not available here because every method has to see the same molecules.
    """
    calc = Descriptors.CalcMolDescriptors
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"unparseable SMILES reached featurization: {smi!r}")
        values = calc(mol, missingVal=np.nan)
        rows.append([values[name] for name in DESCRIPTOR_NAMES])

    x = np.asarray(rows, dtype=np.float64)
    finite = np.isfinite(x).all(axis=0)
    names = [n for n, keep in zip(DESCRIPTOR_NAMES, finite) if keep]
    return x[:, finite].astype(np.float32), names


# --- Mol2Vec -------------------------------------------------------------
# The released repository loads `X_mol2vec` from an uncommitted .npz and neither
# it, the paper's SI, nor the prior study it defers to records how that block was
# built. What is recorded is the width, 300, which fixes the pre-trained model:
# the ZINC-20M skip-gram word2vec of Jaeger, Fulle and Turk, distributed as
# `model_300dim.pkl`. The rest is that model's documented usage -- radius 1,
# identifiers emitted alternately by radius then atom index, the unweighted sum
# of their vectors, and the model's own UNK vector for identifiers it never saw.
MOL2VEC_RADIUS = 1
MOL2VEC_UNSEEN = "UNK"


def mol2vec_sentence(mol: Chem.Mol, radius: int = MOL2VEC_RADIUS) -> list[str]:
    """The Morgan identifiers of a molecule, as `mol2alt_sentence` orders them.

    For each atom, the identifier at radius 0, then radius 1, and so on, walked
    atom by atom. Reimplemented here rather than imported because the `mol2vec`
    package is unmaintained and pins a gensim old enough to conflict with
    everything else in the environment.
    """
    info: dict = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)

    # bitInfo maps identifier -> ((atom index, radius), ...). Invert it so each
    # (atom, radius) cell holds the identifier that covers it.
    by_atom: dict[tuple[int, int], int] = {}
    for identifier, occurrences in info.items():
        for atom_idx, rad in occurrences:
            by_atom[(atom_idx, rad)] = identifier

    sentence = []
    for atom_idx in range(mol.GetNumAtoms()):
        for rad in range(radius + 1):
            identifier = by_atom.get((atom_idx, rad))
            if identifier is not None:
                sentence.append(str(identifier))
    return sentence


def mol2vec_embeddings(smiles: list[str], keyed_vectors) -> np.ndarray:
    """One 300-d vector per molecule: the sum over its Morgan identifiers."""
    dim = keyed_vectors.vector_size
    unseen_vec = (
        keyed_vectors[MOL2VEC_UNSEEN]
        if MOL2VEC_UNSEEN in keyed_vectors
        else np.zeros(dim, dtype=np.float32)
    )

    out = np.zeros((len(smiles), dim), dtype=np.float32)
    n_unseen = n_tokens = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        total = np.zeros(dim, dtype=np.float64)
        for token in mol2vec_sentence(mol):
            n_tokens += 1
            if token in keyed_vectors:
                total += keyed_vectors[token]
            else:
                n_unseen += 1
                total += unseen_vec
        out[i] = total
    share = n_unseen / max(n_tokens, 1)
    print(f"  mol2vec: {n_tokens:,} identifiers, {share:.2%} mapped to {MOL2VEC_UNSEEN}")
    return out


# --- molecular graphs ----------------------------------------------------
# The atom and bond features of DeepChem's `MolGraphConvFeaturizer(use_edges=True)`,
# which is what the released code hands AttentiveFP: a 30-wide atom vector and an
# 11-wide bond vector, no chirality and no partial charges.
ATOM_TYPES = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B"]
HYBRIDIZATIONS = ["SP", "SP2", "SP3"]
DEGREES = [0, 1, 2, 3, 4]
NUM_HS = [0, 1, 2, 3]
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOANY,
    Chem.BondStereo.STEREOZ,
    Chem.BondStereo.STEREOE,
]

ATOM_DIM = (len(ATOM_TYPES) + 1) + 1 + (len(HYBRIDIZATIONS) + 1) + 2 + 1 \
    + (len(DEGREES) + 1) + (len(NUM_HS) + 1)
BOND_DIM = len(BOND_TYPES) + 2 + (len(BOND_STEREO) + 1)


def _one_hot(value, choices) -> list[float]:
    """One-hot with a trailing bucket for anything outside `choices`."""
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else len(choices)] = 1.0
    return vec


def atom_features(mol: Chem.Mol) -> np.ndarray:
    donors, acceptors = hydrogen_bonding_sites(mol)
    rows = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        rows.append(
            _one_hot(atom.GetSymbol(), ATOM_TYPES)
            + [float(atom.GetFormalCharge())]
            + _one_hot(str(atom.GetHybridization()), HYBRIDIZATIONS)
            + [float(idx in donors), float(idx in acceptors)]
            + [float(atom.GetIsAromatic())]
            + _one_hot(atom.GetTotalDegree(), DEGREES)
            + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        )
    return np.asarray(rows, dtype=np.float32)


_HBOND_FACTORY = None


def hydrogen_bonding_sites(mol: Chem.Mol) -> tuple[set[int], set[int]]:
    """Donor and acceptor atom indices, from RDKit's own feature definitions."""
    global _HBOND_FACTORY
    if _HBOND_FACTORY is None:
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures
        import os

        _HBOND_FACTORY = ChemicalFeatures.BuildFeatureFactory(
            os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        )

    donors: set[int] = set()
    acceptors: set[int] = set()
    for feature in _HBOND_FACTORY.GetFeaturesForMol(mol):
        if feature.GetFamily() == "Donor":
            donors.update(feature.GetAtomIds())
        elif feature.GetFamily() == "Acceptor":
            acceptors.update(feature.GetAtomIds())
    return donors, acceptors


def bond_features(bond: Chem.Bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)[:-1]      # no unknown bucket: 4 wide
        + [float(bond.IsInRing()), float(bond.GetIsConjugated())]
        + _one_hot(bond.GetStereo(), BOND_STEREO)
    )


def mol_graph(smiles: str) -> dict[str, np.ndarray]:
    """One molecule as the arrays a PyTorch Geometric `Data` object needs.

    Bonds become two directed edges. A molecule with no bonds still gets valid
    empty edge arrays rather than failing, which matters because both collections
    contain a handful of single-atom fragments.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES reached graph featurization: {smiles!r}")

    x = atom_features(mol)
    src, dst, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        attrs += [feats, feats]

    edge_index = np.asarray([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64)
    edge_attr = np.asarray(attrs, dtype=np.float32) if attrs else np.zeros((0, BOND_DIM), np.float32)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}


# --- SMILES strings ------------------------------------------------------
# Character level, exactly as `build_smiles_vocab` in the released code: no
# atom-aware tokenizer, so a chlorine is the two tokens 'C' and 'l'. Kept as they
# wrote it, because the SMILES modality is theirs to define and a better
# tokenizer would be a different modality.
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
MAX_LEN_CAP = 200


def build_vocab(smiles_train: list[str], max_len_cap: int = MAX_LEN_CAP):
    """A vocabulary and sequence length, from the training SMILES only."""
    chars = sorted({ch for s in smiles_train for ch in s})
    itos = [PAD_TOKEN, UNK_TOKEN] + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    max_len = min(max_len_cap, max(len(s) for s in smiles_train))
    return stoi, itos, max_len


def encode_smiles(smiles: list[str], stoi: dict[str, int], max_len: int) -> np.ndarray:
    """Right-padded token indices, truncated at `max_len`."""
    pad, unk = stoi[PAD_TOKEN], stoi[UNK_TOKEN]
    out = np.full((len(smiles), max_len), pad, dtype=np.int64)
    for i, s in enumerate(smiles):
        ids = [stoi.get(ch, unk) for ch in s[:max_len]]
        out[i, : len(ids)] = ids
    return out
