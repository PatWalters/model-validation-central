"""The two supervised encoders, and the graph learner that reuses one of them.

Both encoders are trained on the fold's own training molecules against the fold's
own endpoint, which is what makes the GNN and SMILES modalities *supervised*
representations rather than fixed featurizations. Both also serve twice over: the
scalar they predict is the modality's base prediction for late fusion, and the
vector behind that scalar is the modality's feature block for early fusion.

Three deviations from github.com/jwasswa2023/Multimodal_Fusion, each deliberate.

1. AttentiveFP comes from PyTorch Geometric rather than DeepChem-on-DGL. The
   architecture is the same; DGL pins a torch old enough that it cannot share an
   environment with the rest of this pipeline.

2. The graph embedding is the 200-wide vector the output layer reads. The
   released `extract_attentivefp_embeddings_strict_dgl` hooks the first
   `nn.Linear` it finds by module order, which in DeepChem's AttentiveFP is
   `gnn.init_context.project_node[0]`, the projection applied to raw atom
   features *before any message passing*. What it captures is therefore a
   30-wide mean of unlearned atom features, and the paper's own Table S3 records
   the GNN modality as 30 features while calling it a learned graph
   representation. `--paper-gnn-block` reproduces that block so the difference
   can be measured rather than argued about.

3. Both encoders early-stop on the fold's held-out fifth. The released code
   trains for a fixed 50 and 20 epochs with no validation monitoring, which it
   can afford because it has no validation split; this pipeline has one, and
   every other method in the comparison uses it.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn.models import AttentiveFP

import config as cfg
from featurize import ATOM_DIM, BOND_DIM


def device_for(gpu: int | None = None) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}" if gpu is not None else "cuda")
    return torch.device("cpu")


# --- AttentiveFP ---------------------------------------------------------
class GraphRegressor(nn.Module):
    """AttentiveFP over a molecular graph, optionally with global features.

    With `global_dim = 0` this is the unimodal GNN: message passing, readout, a
    linear head. With `global_dim > 0` it is the paper's graph meta-learner, the
    fused block or the stacked base predictions concatenated onto the readout
    before the head -- the `global_feat_size` entry in their AttentiveFP search
    space, which the released code never implements.

    `embed()` returns the readout itself, which is the GNN modality's feature
    block wherever a tabular learner consumes it.
    """

    def __init__(
        self,
        global_dim: int = 0,
        hidden: int = cfg.ATTFP_HIDDEN,
        layers: int = cfg.ATTFP_LAYERS,
        timesteps: int = cfg.ATTFP_TIMESTEPS,
        dropout: float = cfg.ATTFP_DROPOUT,
    ):
        super().__init__()
        # out_channels == hidden, so the backbone's output is the graph vector the
        # head predicts from rather than a scalar.
        self.backbone = AttentiveFP(
            in_channels=ATOM_DIM,
            hidden_channels=hidden,
            out_channels=hidden,
            edge_dim=BOND_DIM,
            num_layers=layers,
            num_timesteps=timesteps,
            dropout=dropout,
        )
        self.global_dim = global_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden + global_dim, 1),
        )

    def embed(self, data) -> torch.Tensor:
        return self.backbone(data.x, data.edge_index, data.edge_attr, data.batch)

    def forward(self, data) -> torch.Tensor:
        h = self.embed(data)
        if self.global_dim:
            h = torch.cat([h, data.globals.view(h.size(0), self.global_dim)], dim=1)
        return self.head(h).squeeze(-1)


def as_tensors(graph: dict) -> dict:
    """A molecule's arrays as torch tensors, converted once and then shared.

    Converting from numpy is the expensive part of building a `Data` object, and
    a molecule's graph never changes: the same tensors are reused by every fold,
    every endpoint and every configuration in a process. Only the label and the
    global feature row differ, and those are scalars.
    """
    return {
        "x": torch.from_numpy(graph["x"]),
        "edge_index": torch.from_numpy(graph["edge_index"]),
        "edge_attr": torch.from_numpy(graph["edge_attr"]),
    }


def graph_dataset(
    graphs: list[dict],
    y: np.ndarray,
    globals_: np.ndarray | None = None,
) -> list[Data]:
    """PyG `Data` objects, one per molecule, in the order given.

    `graphs` may hold either the numpy arrays `featurize.mol_graph` returns or
    the shared tensors `as_tensors` makes of them; the second is what every
    caller in this pipeline passes, and it is roughly twenty times faster.
    """
    out = []
    for i, g in enumerate(graphs):
        tensors = g if torch.is_tensor(g["x"]) else as_tensors(g)
        data = Data(
            x=tensors["x"],
            edge_index=tensors["edge_index"],
            edge_attr=tensors["edge_attr"],
            y=torch.tensor([float(y[i])], dtype=torch.float32),
        )
        if globals_ is not None:
            data.globals = torch.as_tensor(
                globals_[i], dtype=torch.float32
            ).view(1, -1)
        out.append(data)
    return out


def _run_epoch(model, loader, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(train):
            pred = model(batch)
            loss = nn.functional.mse_loss(pred, batch.y.view(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total / max(n, 1)


def fit_graph_regressor(
    train_graphs, y_train, val_graphs, y_val, seed: int,
    train_globals=None, val_globals=None, device=None, verbose: bool = False,
) -> GraphRegressor:
    """Train AttentiveFP, keeping the weights with the best validation MSE."""
    device = device or device_for()
    torch.manual_seed(seed)

    global_dim = 0 if train_globals is None else train_globals.shape[1]
    model = GraphRegressor(global_dim=global_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.ATTFP_LR, weight_decay=cfg.ATTFP_WEIGHT_DECAY
    )

    train_loader = GeoLoader(
        graph_dataset(train_graphs, y_train, train_globals),
        batch_size=cfg.ATTFP_BATCH, shuffle=True,
    )
    val_loader = GeoLoader(
        graph_dataset(val_graphs, y_val, val_globals),
        batch_size=256, shuffle=False,
    )

    best, best_state, waited = float("inf"), None, 0
    for epoch in range(cfg.ATTFP_EPOCHS):
        _run_epoch(model, train_loader, device, optimizer)
        val_loss = _run_epoch(model, val_loader, device)
        if val_loss < best - 1e-6:
            best, waited = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg.ATTFP_PATIENCE:
                break
        if verbose and epoch % 10 == 0:
            print(f"    attfp epoch {epoch:3d}  val mse {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def graph_predict(model, graphs, globals_=None, device=None) -> np.ndarray:
    device = device or device_for()
    model.eval()
    loader = GeoLoader(
        graph_dataset(graphs, np.zeros(len(graphs)), globals_),
        batch_size=256, shuffle=False,
    )
    return np.concatenate([model(b.to(device)).cpu().numpy() for b in loader])


@torch.no_grad()
def graph_embed(model, graphs, device=None) -> np.ndarray:
    """The readout, one row per molecule, in the order given."""
    return graph_embed_and_predict(model, graphs, device)[0]


@torch.no_grad()
def graph_embed_and_predict(model, graphs, device=None) -> tuple[np.ndarray, np.ndarray]:
    """The readout and the prediction from a single forward pass.

    The two are wanted together every time -- the readout is the GNN modality's
    feature block and the prediction is its late-fusion base learner -- and a
    second pass over every molecule in the fold to recompute one from the other
    is the single most wasteful thing this pipeline could do 375 times.
    """
    if model.global_dim:
        raise ValueError("a graph learner with global features has no single-pass "
                         "prediction: its head needs those features too")
    device = device or device_for()
    model.eval()
    loader = GeoLoader(
        graph_dataset(graphs, np.zeros(len(graphs))), batch_size=256, shuffle=False
    )
    embeddings, predictions = [], []
    for batch in loader:
        batch = batch.to(device)
        h = model.embed(batch)
        embeddings.append(h.cpu().numpy())
        predictions.append(model.head(h).squeeze(-1).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(predictions)


def graph_embed_atom_mean(graphs) -> np.ndarray:
    """The block the released extractor actually produces: mean raw atom features.

    No model is involved, which is the point. Kept so the report can show what
    the paper's GNN modality contains rather than only asserting it.
    """
    rows = [
        g["x"].mean(axis=0).numpy() if torch.is_tensor(g["x"]) else g["x"].mean(axis=0)
        for g in graphs
    ]
    return np.asarray(rows, dtype=np.float32)


# --- SMILES BiGRU --------------------------------------------------------
class SmilesRegressor(nn.Module):
    """A character BiGRU over SMILES, predicting the endpoint.

    Verbatim from `SmilesEncoderRegressor` in the released code: a 64-wide
    character embedding, one bidirectional GRU layer of 128, dropout, and a linear
    head on the concatenated final forward and backward states. The 256-wide
    vector after the dropout is the SMILES modality's feature block.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, cfg.BIGRU_EMBED, padding_idx=pad_idx)
        self.gru = nn.GRU(
            input_size=cfg.BIGRU_EMBED,
            hidden_size=cfg.BIGRU_HIDDEN,
            num_layers=cfg.BIGRU_LAYERS,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(cfg.BIGRU_DROPOUT)
        self.fc = nn.Linear(cfg.SMILES_EMBED_DIM, 1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(self.embedding(x))
        return self.dropout(torch.cat([h[-2], h[-1]], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.embed(x)).squeeze(-1)


def _smiles_batches(ids: np.ndarray, y: np.ndarray, batch: int, shuffle: bool, rng=None):
    order = np.arange(len(ids))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch):
        sel = order[start : start + batch]
        yield torch.from_numpy(ids[sel]), torch.from_numpy(y[sel].astype(np.float32))


def fit_smiles_regressor(
    ids_train, y_train, ids_val, y_val, vocab_size: int, seed: int,
    pad_idx: int = 0, device=None,
) -> SmilesRegressor:
    """Train the BiGRU, keeping the weights with the best validation MSE."""
    device = device or device_for()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = SmilesRegressor(vocab_size, pad_idx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.BIGRU_LR)

    best, best_state, waited = float("inf"), None, 0
    for _ in range(cfg.BIGRU_EPOCHS):
        model.train()
        for xb, yb in _smiles_batches(ids_train, y_train, cfg.BIGRU_BATCH, True, rng):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in _smiles_batches(ids_val, y_val, 256, False):
                xb, yb = xb.to(device), yb.to(device)
                total += nn.functional.mse_loss(model(xb), yb, reduction="sum").item()
                n += len(yb)
        val_loss = total / max(n, 1)

        if val_loss < best - 1e-6:
            best, waited = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= cfg.BIGRU_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def smiles_predict(model, ids, device=None) -> np.ndarray:
    return smiles_embed_and_predict(model, ids, device)[1]


@torch.no_grad()
def smiles_embed_and_predict(model, ids, device=None) -> tuple[np.ndarray, np.ndarray]:
    """The 256-wide state and the prediction, from one pass. See `graph_embed_and_predict`."""
    device = device or device_for()
    model.eval()
    embeddings, predictions = [], []
    for xb, _ in _smiles_batches(ids, np.zeros(len(ids)), 256, False):
        h = model.embed(xb.to(device))
        embeddings.append(h.cpu().numpy())
        predictions.append(model.fc(h).squeeze(-1).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(predictions)


@torch.no_grad()
def smiles_embed(model, ids, device=None) -> np.ndarray:
    """The 256-wide state, in the order given.

    Extraction runs over an unshuffled sequence on purpose. The released
    `fusion_early.py` extracts its *training* embeddings from a loader built with
    `shuffle=True`, so that block comes back in a different order from the three
    it is concatenated with; `fusion_late.py` keeps a separate unshuffled loader
    and does not have the problem.
    """
    return smiles_embed_and_predict(model, ids, device)[0]
