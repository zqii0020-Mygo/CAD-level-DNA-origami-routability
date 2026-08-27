"""Multi-task training and evaluation for the routability surrogate."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from cadna.graph import NODE_TYPES

from .data import CLASS_NAMES, N_CLASSES, Normaliser, Split, load_graphs, make_split, to_pyg_list
from .metrics import (
    accuracy,
    balanced_accuracy,
    mae,
    macro_f1,
    multiclass_accuracy,
    roc_auc,
    spearman,
)
from .nets import GraphMLP, HeteroGNN

LOSS_WEIGHTS = {"routable": 1.0, "hamilton": 1.0, "failure": 1.0, "cost": 0.3}


@dataclass
class Config:
    name: str
    model: str                      # "mlp" | "gnn"
    include_precheck: bool
    hidden: int = 64
    layers: int = 3
    epochs: int = 120
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    note: str = ""


CONFIGS = [
    Config("mlp-counts", "mlp", include_precheck=False, hidden=128,
           note="graph_x only: can counting features alone predict routability?"),
    Config("mlp-rules", "mlp", include_precheck=True, hidden=128,
           note="graph_x + precheck_x: the exact-obstruction baseline to beat"),
    Config("gnn-structure", "gnn", include_precheck=False,
           note="message passing, no precheck block: does structure add anything?"),
    Config("gnn-rules", "gnn", include_precheck=True,
           note="message passing + precheck block: do they compose?"),
]


def _stack(loader_batch, key: str) -> torch.Tensor:
    return getattr(loader_batch, key)


def _losses(out: dict[str, torch.Tensor], batch, pos_weight: torch.Tensor) -> torch.Tensor:
    total = out["routable"].new_zeros(())

    y_r = _stack(batch, "y_routable")
    total = total + LOSS_WEIGHTS["routable"] * F.binary_cross_entropy_with_logits(
        out["routable"], y_r, pos_weight=pos_weight
    )

    y_h = _stack(batch, "y_hamilton")
    m = ~torch.isnan(y_h)
    if m.any():
        total = total + LOSS_WEIGHTS["hamilton"] * F.binary_cross_entropy_with_logits(
            out["hamilton"][m], y_h[m]
        )

    total = total + LOSS_WEIGHTS["failure"] * F.cross_entropy(out["failure"], batch.y_class)

    y_c = _stack(batch, "y_log_nodes_expanded")
    m = _stack(batch, "y_searched") > 0.5
    if m.any():
        total = total + LOSS_WEIGHTS["cost"] * F.mse_loss(out["cost"][m], y_c[m])
    return total


@torch.no_grad()
def predict(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    acc: dict[str, list[np.ndarray]] = {}

    def add(k: str, v: torch.Tensor) -> None:
        acc.setdefault(k, []).append(v.detach().cpu().numpy())

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        add("p_routable", torch.sigmoid(out["routable"]))
        add("p_hamilton", torch.sigmoid(out["hamilton"]))
        add("hat_class", out["failure"].argmax(dim=-1))
        add("p_cost", out["cost"])
        add("y_routable", batch.y_routable)
        add("y_hamilton", batch.y_hamilton)
        add("y_class", batch.y_class)
        add("y_cost", batch.y_log_nodes_expanded)
        add("searched", batch.y_searched)
        add("precheck_ok", batch.y_precheck_ok)
    return {k: np.concatenate(v) for k, v in acc.items()}


def score(pred: dict[str, np.ndarray]) -> dict[str, float]:
    cost_mask = pred["searched"] > 0.5
    y_cost = np.where(cost_mask, pred["y_cost"], np.nan)
    return {
        "routable_bacc": balanced_accuracy(pred["y_routable"], pred["p_routable"]),
        "routable_auc": roc_auc(pred["y_routable"], pred["p_routable"]),
        "hamilton_bacc": balanced_accuracy(pred["y_hamilton"], pred["p_hamilton"]),
        "hamilton_auc": roc_auc(pred["y_hamilton"], pred["p_hamilton"]),
        "failure_acc": multiclass_accuracy(pred["y_class"].astype(float), pred["hat_class"].astype(float)),
        "failure_f1": macro_f1(pred["y_class"].astype(float), pred["hat_class"].astype(float), N_CLASSES),
        "cost_rho": spearman(y_cost, pred["p_cost"]),
        "cost_mae": mae(y_cost, pred["p_cost"]),
    }


def _selection_metric(s: dict[str, float]) -> float:
    parts = [s["routable_bacc"], s["hamilton_bacc"], s["failure_f1"]]
    parts = [p for p in parts if not np.isnan(p)]
    return float(np.mean(parts)) if parts else 0.0


def run(cfg: Config, graphs, split: Split, seed: int = 0,
        device: str = "cpu", verbose: bool = False) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = to_pyg_list(graphs, include_precheck=cfg.include_precheck)
    tr = [data[i] for i in split.train]
    va = [data[i] for i in split.val]
    te = [data[i] for i in split.test]

    dl_tr = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True)
    dl_va = DataLoader(va, batch_size=256)
    dl_te = DataLoader(te, batch_size=256)

    graph_dim = int(data[0].graph_x.shape[1])
    if cfg.model == "mlp":
        model = GraphMLP(graph_dim, hidden=cfg.hidden)
    else:
        in_dims = {t: int(data[0][t].x.shape[1]) for t in NODE_TYPES}
        model = HeteroGNN(in_dims, graph_dim, hidden=cfg.hidden, layers=cfg.layers)
    model = model.to(device)

    y_tr = np.array([float(d.y_routable) for d in tr])
    n_pos = max(1.0, float((y_tr >= 0.5).sum()))
    pos_weight = torch.tensor(float((y_tr < 0.5).sum()) / n_pos, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best = (-1.0, None, -1)
    for epoch in range(cfg.epochs):
        model.train()
        for batch in dl_tr:
            batch = batch.to(device)
            opt.zero_grad()
            loss = _losses(model(batch), batch, pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        if epoch % 5 == 4 or epoch == cfg.epochs - 1:
            s = _selection_metric(score(predict(model, dl_va, device)))
            if s > best[0]:
                best = (s, copy.deepcopy(model.state_dict()), epoch)
            if verbose:
                print(f"    epoch {epoch + 1:>3}  val={s:.4f}  best={best[0]:.4f}")

    if best[1] is not None:
        model.load_state_dict(best[1])
    test_pred = predict(model, dl_te, device)
    return {
        "config": cfg.name,
        "seed": seed,
        "best_epoch": best[2] + 1,
        "val_selection": best[0],
        "test": score(test_pred),
        "n_params": sum(p.numel() for p in model.parameters()),
        "pred": test_pred,
    }


def reference_baselines(pred: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """What you get without learning anything.

    majority   always predict the training-majority class
    precheck   the pipeline's own cheap verdict: a design that fails a fatal
               precheck is not routable, anything else is assumed routable
    """
    y_r = pred["y_routable"]
    y_h = pred["y_hamilton"]
    y_c = pred["y_class"].astype(float)
    ok = pred["precheck_ok"]

    majority_r = float(y_r.mean() >= 0.5)
    majority_c = float(np.bincount(pred["y_class"].astype(int), minlength=N_CLASSES).argmax())
    return {
        "majority": {
            "routable_bacc": balanced_accuracy(y_r, np.full_like(y_r, majority_r)),
            "routable_auc": float("nan"),
            "hamilton_bacc": balanced_accuracy(y_h, np.full_like(y_h, float(np.nanmean(y_h) >= 0.5))),
            "hamilton_auc": float("nan"),
            "failure_acc": multiclass_accuracy(y_c, np.full_like(y_c, majority_c)),
            "failure_f1": macro_f1(y_c, np.full_like(y_c, majority_c), N_CLASSES),
            "cost_rho": float("nan"),
            "cost_mae": float("nan"),
        },
        "precheck-rule": {
            "routable_bacc": balanced_accuracy(y_r, ok),
            "routable_auc": roc_auc(y_r, ok),
            "hamilton_bacc": balanced_accuracy(y_h, ok),
            "hamilton_auc": roc_auc(y_h, ok),
            "failure_acc": float("nan"),
            "failure_f1": float("nan"),
            "cost_rho": float("nan"),
            "cost_mae": float("nan"),
        },
    }


def load_dataset(directory: str, split_seed: int = 0):
    graphs = load_graphs(directory)
    split = make_split(graphs, seed=split_seed)
    Normaliser.fit(graphs, split.train).apply(graphs)
    return graphs, split
