"""Rung 4 — 2-layer GraphSAGE for directed edge classification.

Architecture (matches ``training/configs/rung4_sage_v1_calibrated.yaml``):

  Input  :  x [N, F_node],  edge_index [2, E],  edge_attr [E, F_edge]
            (per-run graph; F_node=2, F_edge=8 by default)

  Layer 1:  SAGEConv(F_node -> 128), ReLU, Dropout(0.3)
  Layer 2:  SAGEConv(128   -> 64 ), ReLU, Dropout(0.3)

  Edge head:  MLP( concat(h_src, h_dst, edge_attr) -> 64 -> 1 ) -> sigmoid
              produces per-directed-edge probabilities

  Loss:    weighted BCE with auto pos_weight (see training/train/losses.py)

Torch and torch_geometric are imported lazily so this file is import-safe
without them. Actual training happens inside the trainer Docker image.

Reference
---------
Cont, R. (2001). Empirical properties of asset returns: stylized facts
and statistical implications. Quantitative Finance, 1(2), 223-236.

Hamilton, W. L., Ying, R., Leskovec, J. (2017). Inductive Representation
Learning on Large Graphs. NeurIPS.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np

_log = logging.getLogger(__name__)


@dataclass
class GraphSAGEConfig:
    """Hyperparameter container - mirrors the model section of the YAML."""

    node_in_dim: int
    edge_in_dim: int
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.3
    aggregator: str = "mean"           # mean | max | sum
    head_hidden: int = 64
    head_dropout: float = 0.2


class EdgeGraphSAGEModel:
    """Sklearn-style ``fit/predict_proba`` wrapper around the SAGE module.

    The wrapper exposes:
      - ``fit(graphs, sample_weight=None)``       graphs is a list of
            ``training.features.pyg_builder.GraphArrays``; for each graph
            we forward, compute edge probabilities, and backprop weighted
            BCE.
      - ``predict_proba(graph)``                   returns numpy array of
            edge probabilities for one graph.

    The wrapper does its own minibatching (one graph at a time) which is
    appropriate for the graph sizes here (500 nodes per R01-R24 run).
    For larger graphs replace with the standard PyG NeighborLoader.
    """

    def __init__(
        self,
        config: GraphSAGEConfig,
        *,
        lr: float = 1e-3,
        epochs: int = 50,
        early_stopping_patience: int = 8,
        seed: int = 42,
        device: str = "auto",
        # ---- Phase G opt-in additions (back-compat with M3+) -------------
        # loss="bce" + val_recall_fn=None recovers the M3+ champion behavior
        # exactly. Phase G drivers pass loss="focal" and val_recall_fn=...
        # to switch to production-task-aware training.
        loss: str = "bce",
        focal_alpha: float = 0.85,
        focal_gamma: float = 2.0,
        val_recall_fn=None,
        weight_decay: float = 0.0,
    ) -> None:
        self.config = config
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.patience = int(early_stopping_patience)
        self.seed = int(seed)
        self.loss_name = str(loss)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)
        self.val_recall_fn = val_recall_fn
        self.weight_decay = float(weight_decay)
        # "auto" -> cuda if available, else cpu. Honour explicit
        # caller choice (cpu / cuda / cuda:0 / ...) otherwise.
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = str(device)
        self._module = None       # type: Any
        self._epoch_losses: list[float] = []
        self._best_val: float = float("inf")
        self._best_state: Optional[dict] = None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _build_module(self):
        try:
            import torch
            import torch.nn as nn
            from torch_geometric.nn import SAGEConv
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "EdgeGraphSAGEModel needs torch + torch_geometric. "
                "Run inside the trainer Docker image, or install:\n"
                "  pip install torch torch_geometric"
            ) from exc

        cfg = self.config

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                dims = [cfg.node_in_dim, *cfg.hidden_dims]
                self.convs = nn.ModuleList()
                for fan_in, fan_out in zip(dims[:-1], dims[1:]):
                    self.convs.append(SAGEConv(fan_in, fan_out, aggr=cfg.aggregator))
                self.dropout = nn.Dropout(cfg.dropout)
                head_in = cfg.hidden_dims[-1] * 2 + cfg.edge_in_dim
                self.edge_head = nn.Sequential(
                    nn.Linear(head_in, cfg.head_hidden),
                    nn.ReLU(),
                    nn.Dropout(cfg.head_dropout),
                    nn.Linear(cfg.head_hidden, 1),
                )

            def encode(self, x, edge_index):
                h = x
                for i, conv in enumerate(self.convs):
                    h = conv(h, edge_index)
                    h = torch.relu(h)
                    h = self.dropout(h)
                return h

            def forward(self, x, edge_index, edge_attr):
                h = self.encode(x, edge_index)
                src = edge_index[0]
                dst = edge_index[1]
                edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
                logits = self.edge_head(edge_repr).squeeze(-1)
                return logits

        torch.manual_seed(self.seed)
        self._module = _Module().to(self.device)
        return self._module

    @staticmethod
    def _to_tensors(graph_arrays, device: str):
        import torch
        x = torch.from_numpy(graph_arrays.x).float().to(device)
        edge_index = torch.from_numpy(graph_arrays.edge_index).long().to(device)
        edge_attr = torch.from_numpy(graph_arrays.edge_attr).float().to(device)
        y = torch.from_numpy(graph_arrays.y).float().to(device)
        return x, edge_index, edge_attr, y

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(
        self,
        train_graphs: Sequence,
        val_graphs: Optional[Sequence] = None,
    ) -> "EdgeGraphSAGEModel":
        """Train on a list of GraphArrays. Each "graph" is one run."""
        import torch
        from detect.train.losses import auto_pos_weight

        if not train_graphs:
            raise ValueError("fit: train_graphs is empty")

        module = self._build_module()
        optimizer = torch.optim.Adam(
            module.parameters(), lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Compute a single pos_weight from the pooled training labels.
        pooled = np.concatenate([g.y for g in train_graphs])
        pw = float(auto_pos_weight(pooled))
        _log.info("graphsage fit: %d graphs, pos_weight=%.2f, epochs=%d, loss=%s",
                  len(train_graphs), pw, self.epochs, self.loss_name)

        if self.loss_name == "focal":
            from detect.models.focal_loss import FocalLoss
            criterion = FocalLoss(alpha=self.focal_alpha,
                                  gamma=self.focal_gamma).to(self.device)
        else:
            criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pw, device=self.device)
            )
        self._epoch_losses = []
        self._best_val = float("inf")
        self._best_state = None
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            module.train()
            train_loss_sum = 0.0
            train_edge_sum = 0
            for g in train_graphs:
                if g.num_edges == 0:
                    continue
                x, ei, ea, y = self._to_tensors(g, self.device)
                optimizer.zero_grad()
                logits = module(x, ei, ea)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item()) * g.num_edges
                train_edge_sum += g.num_edges
            mean_train = train_loss_sum / max(train_edge_sum, 1)

            # Validation pass (if available) for early stopping.
            mean_val = float("nan")
            if val_graphs:
                module.eval()
                val_loss_sum = 0.0
                val_edge_sum = 0
                with torch.no_grad():
                    for g in val_graphs:
                        if g.num_edges == 0:
                            continue
                        x, ei, ea, y = self._to_tensors(g, self.device)
                        logits = module(x, ei, ea)
                        loss = criterion(logits, y)
                        val_loss_sum += float(loss.item()) * g.num_edges
                        val_edge_sum += g.num_edges
                mean_val = val_loss_sum / max(val_edge_sum, 1)
            self._epoch_losses.append({
                "epoch": epoch, "train_loss": mean_train, "val_loss": mean_val,
            })

            # Early-stopping signal.
            # Default: val_loss (lower-is-better). Phase G: val_recall_fn
            # (higher-is-better), evaluated against the production metric.
            stop_signal: float = float("nan")
            higher_is_better: bool = False
            if self.val_recall_fn is not None and val_graphs:
                module.eval()
                # The callback computes the production metric (e.g. mean
                # per-family trader recall) from the model on the val set.
                with torch.no_grad():
                    stop_signal = float(self.val_recall_fn(self, val_graphs))
                higher_is_better = True
            elif val_graphs and not np.isnan(mean_val):
                stop_signal = mean_val
                higher_is_better = False

            self._epoch_losses[-1]["stop_signal"] = stop_signal
            if not np.isnan(stop_signal):
                improved = (
                    (stop_signal > self._best_val + 1e-6)
                    if higher_is_better
                    else (stop_signal < self._best_val - 1e-6)
                )
                # First-epoch initialization (best starts at +inf for
                # lower-is-better, -inf for higher-is-better).
                if higher_is_better and self._best_state is None:
                    self._best_val = -float("inf")
                if improved:
                    self._best_val = stop_signal
                    self._best_state = {k: v.detach().clone()
                                        for k, v in module.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        _log.info("early stopping at epoch %d "
                                  "(no improvement on %s for %d epochs)",
                                  epoch,
                                  "val_recall" if higher_is_better else "val_loss",
                                  self.patience)
                        break

            _log.info(
                "epoch %3d  train_loss=%.4f  val_loss=%s",
                epoch, mean_train,
                f"{mean_val:.4f}" if not np.isnan(mean_val) else "n/a",
            )

        # Restore the best checkpoint if early stopping triggered.
        if self._best_state is not None:
            module.load_state_dict(self._best_state)
        return self

    def predict_proba(self, graph) -> np.ndarray:
        import torch
        if self._module is None:
            raise RuntimeError("predict_proba before fit")
        self._module.eval()
        if graph.num_edges == 0:
            return np.zeros(0, dtype=np.float32)
        x, ei, ea, _ = self._to_tensors(graph, self.device)
        with torch.no_grad():
            logits = self._module(x, ei, ea)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float32)

    @property
    def epoch_losses(self) -> list[dict]:
        return list(self._epoch_losses)
