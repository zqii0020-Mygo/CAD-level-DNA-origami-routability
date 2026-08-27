"""Multi-task training and evaluation for the routability surrogate.

Two details that decide whether the numbers mean anything:

*Censoring.*  A design that hit the node budget has `nodes_expanded == budget`,
which is a lower bound on its true cost, not the cost.  Those designs are kept
out of the cost regression and out of the cost metrics; they enter the loss only
through a one-sided hinge that says "at least this much".

*Size.*  Cost grows with the number of cylinders, so the headline rank
correlation is easy to win by predicting size.  The model regresses the residual
against the training-fitted `CostBaseline`, `cost_rho` scores the reconstructed
absolute cost, and `cost_rho_within` scores it inside size bins, which is the
number the size-only reference cannot touch.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from cadna.graph import NODE_TYPES

from .data import (
    ClassMap,
    CostBaseline,
    Normaliser,
    Split,
    load_graphs,
    build_split,
    to_pyg_list,
)
from .metrics import (
    balanced_accuracy,
    class_f1s,
    mae,
    macro_f1,
    multiclass_accuracy,
    roc_auc,
    spearman,
    stratified_spearman,
)
from .nets import GraphMLP, HeteroGNN

LOSS_WEIGHTS = {"routable": 1.0, "hamilton": 1.0, "failure": 1.0, "cost": 0.3}


@dataclass
class DatasetContext:
    """What the dataset -- not the model -- decides: class slots and cost scale."""
    class_map: ClassMap
    cost_baseline: CostBaseline
    split_info: dict = field(default_factory=dict)


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

    # cost is predicted as a residual against the size baseline
    y_c = _stack(batch, "y_cost_resid")
    searched = _stack(batch, "y_searched") > 0.5
    censored = _stack(batch, "y_timeout") > 0.5
    m = searched & ~censored & ~torch.isnan(y_c)
    if m.any():
        total = total + LOSS_WEIGHTS["cost"] * F.mse_loss(out["cost"][m], y_c[m])
    c = censored & ~torch.isnan(y_c)
    if c.any():
        # right-censored: the true cost is at least the budget, so only
        # under-prediction is penalised
        total = total + LOSS_WEIGHTS["cost"] * F.relu(y_c[c] - out["cost"][c]).pow(2).mean()
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
        add("y_cost_resid", batch.y_cost_resid)
        add("cost_base", batch.y_cost_base)
        add("size", batch.y_size)
        add("searched", batch.y_searched)
        add("timeout", batch.y_timeout)
        add("precheck_ok", batch.y_precheck_ok)
    return {k: np.concatenate(v) for k, v in acc.items()}


def cost_view(pred: dict[str, np.ndarray], ctx: "DatasetContext | None" = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(true log cost, predicted log cost, size bin), censored entries masked out.

    The head predicts a residual, so the absolute prediction is the baseline
    plus the residual; timeouts are dropped because their target is a bound.
    """
    usable = (pred["searched"] > 0.5) & (pred["timeout"] < 0.5)
    y = np.where(usable, pred["y_cost"], np.nan)
    p = pred["cost_base"] + pred["p_cost"]
    bins = ctx.cost_baseline.bin_of(pred["size"]) if ctx else pred["size"]
    return y, p, bins


def score(pred: dict[str, np.ndarray], ctx: "DatasetContext | None" = None) -> dict[str, float]:
    n_classes = ctx.class_map.n if ctx else int(pred["y_class"].max()) + 1
    y_cost, p_cost, bins = cost_view(pred, ctx)
    return {
        "routable_bacc": balanced_accuracy(pred["y_routable"], pred["p_routable"]),
        "routable_auc": roc_auc(pred["y_routable"], pred["p_routable"]),
        "hamilton_bacc": balanced_accuracy(pred["y_hamilton"], pred["p_hamilton"]),
        "hamilton_auc": roc_auc(pred["y_hamilton"], pred["p_hamilton"]),
        "failure_acc": multiclass_accuracy(pred["y_class"].astype(float), pred["hat_class"].astype(float)),
        "failure_f1": macro_f1(pred["y_class"].astype(float), pred["hat_class"].astype(float), n_classes),
        "cost_rho": spearman(y_cost, p_cost),
        "cost_rho_within": stratified_spearman(y_cost, p_cost, bins),
        "cost_mae": mae(y_cost, p_cost),
    }


SLICES = ("all", "precheck_decided", "precheck_undecided")


def slice_masks(pred: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Split a prediction set by whether the precheck already decided the design.

    A fatal precheck is an *exact* obstruction, so those labels are free: any
    model that reproduces the rules gets them right, and the rules are a page of
    `precheck.py`.  The designs that reach the DFS are the ones whose label cost
    a search, and they are the only slice where a surrogate can be worth
    anything.  Reporting only the whole test set lets a model bank the free
    half, which is exactly the objection this slicing exists to answer.

    The mask uses `searched`, which is "the precheck was not fatal" -- a
    quantity the precheck computes in O(V + E) without any search, so slicing
    on it is something a deployment can do too.
    """
    searched = pred["searched"] > 0.5
    return {
        "all": np.ones(searched.shape, dtype=bool),
        "precheck_decided": ~searched,
        "precheck_undecided": searched,
    }


def _subset(pred: dict[str, np.ndarray], m: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[m] for k, v in pred.items()}


def score_sliced(pred: dict[str, np.ndarray], ctx: "DatasetContext | None" = None
                 ) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name, m in slice_masks(pred).items():
        s = score(_subset(pred, m), ctx)
        s["n"] = float(m.sum())
        out[name] = s
    return out


def per_class_f1(pred: dict[str, np.ndarray], ctx: "DatasetContext") -> dict[str, tuple[float, int]]:
    """Failure-class F1 by name -- the thin classes are invisible in the macro."""
    got = class_f1s(pred["y_class"].astype(float), pred["hat_class"].astype(float),
                    ctx.class_map.n)
    return {ctx.class_map.names[c]: v for c, v in got.items()}


def _selection_metric(s: dict[str, float]) -> float:
    parts = [s["routable_bacc"], s["hamilton_bacc"], s["failure_f1"]]
    parts = [p for p in parts if not np.isnan(p)]
    return float(np.mean(parts)) if parts else 0.0


def run(cfg: Config, graphs, split: Split, ctx: DatasetContext, seed: int = 0,
        device: str = "cpu", verbose: bool = False) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = to_pyg_list(graphs, include_precheck=cfg.include_precheck,
                       class_map=ctx.class_map, cost_baseline=ctx.cost_baseline)
    tr = [data[i] for i in split.train]
    va = [data[i] for i in split.val]
    te = [data[i] for i in split.test]

    dl_tr = DataLoader(tr, batch_size=cfg.batch_size, shuffle=True)
    dl_va = DataLoader(va, batch_size=256)
    dl_te = DataLoader(te, batch_size=256)

    graph_dim = int(data[0].graph_x.shape[1])
    n_classes = ctx.class_map.n
    if cfg.model == "mlp":
        model = GraphMLP(graph_dim, hidden=cfg.hidden, n_classes=n_classes)
    else:
        in_dims = {t: int(data[0][t].x.shape[1]) for t in NODE_TYPES}
        model = HeteroGNN(in_dims, graph_dim, hidden=cfg.hidden, layers=cfg.layers,
                          n_classes=n_classes)
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
            s = _selection_metric(score(predict(model, dl_va, device), ctx))
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
        "test": score(test_pred, ctx),
        "test_sliced": score_sliced(test_pred, ctx),
        "test_class_f1": per_class_f1(test_pred, ctx),
        "test_class_f1_sliced": {
            name: per_class_f1(_subset(test_pred, m), ctx)
            for name, m in slice_masks(test_pred).items()
        },
        "n_params": sum(p.numel() for p in model.parameters()),
        "pred": test_pred,
    }


def reference_baselines(pred: dict[str, np.ndarray],
                        ctx: DatasetContext | None = None) -> dict[str, dict[str, float]]:
    """What you get without learning anything.

    majority    always predict the training-majority class
    precheck    the pipeline's own cheap verdict: a design that fails a fatal
                precheck is not routable, anything else is assumed routable
    size-only   predict the search cost from the design size alone -- the
                reference the cost head has to beat.  It is constant inside a
                size bin, so `cost_rho_within` is undefined for it by
                construction: that is the point of the metric.
    """
    nan = float("nan")
    y_r = pred["y_routable"]
    y_h = pred["y_hamilton"]
    y_c = pred["y_class"].astype(float)
    ok = pred["precheck_ok"]
    n_classes = ctx.class_map.n if ctx else int(pred["y_class"].max()) + 1

    majority_r = float(y_r.mean() >= 0.5)
    majority_c = float(np.bincount(pred["y_class"].astype(int), minlength=n_classes).argmax())
    y_cost, _p, bins = cost_view(pred, ctx)
    size_only = pred["cost_base"]
    empty = {k: nan for k in (
        "routable_bacc", "routable_auc", "hamilton_bacc", "hamilton_auc",
        "failure_acc", "failure_f1", "cost_rho", "cost_rho_within", "cost_mae")}
    return {
        "majority": {
            **empty,
            "routable_bacc": balanced_accuracy(y_r, np.full_like(y_r, majority_r)),
            "hamilton_bacc": balanced_accuracy(y_h, np.full_like(y_h, float(np.nanmean(y_h) >= 0.5))),
            "failure_acc": multiclass_accuracy(y_c, np.full_like(y_c, majority_c)),
            "failure_f1": macro_f1(y_c, np.full_like(y_c, majority_c), n_classes),
        },
        "precheck-rule": {
            **empty,
            "routable_bacc": balanced_accuracy(y_r, ok),
            "routable_auc": roc_auc(y_r, ok),
            "hamilton_bacc": balanced_accuracy(y_h, ok),
            "hamilton_auc": roc_auc(y_h, ok),
        },
        "size-only": {
            **empty,
            "cost_rho": spearman(y_cost, size_only),
            "cost_rho_within": stratified_spearman(y_cost, size_only, bins),
            "cost_mae": mae(y_cost, size_only),
        },
    }


def reference_baselines_sliced(pred: dict[str, np.ndarray],
                               ctx: DatasetContext | None = None
                               ) -> dict[str, dict[str, dict[str, float]]]:
    """The reference rows, per slice.

    `precheck-rule` is the row to watch: on the undecided slice it calls every
    design routable, so its balanced accuracy collapses to 0.5.  That is the
    honest statement of what the rules are worth where they have not already
    answered, and the bar a surrogate actually has to clear.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, m in slice_masks(pred).items():
        sub = _subset(pred, m)
        out[name] = reference_baselines(sub, ctx)
        for row in out[name].values():
            row["n"] = float(m.sum())
    return out


def load_dataset(directory, split_seed: int = 0, split_spec: str = "random"):
    """Load, split, normalise, and fit what the dataset itself decides.

    `ClassMap` and `CostBaseline` are part of the dataset definition, not of a
    model, so every config in a run shares them; the cost baseline sees only the
    training split.  `split_spec` is `random`, `shape:<name>` or
    `size:<quantile>` -- the last two hold out a *region* of the design space,
    which is the difference between a surrogate and a curve fit.
    """
    graphs = load_graphs(directory)
    split, split_info = build_split(graphs, split_spec, seed=split_seed)
    Normaliser.fit(graphs, split.train).apply(graphs)
    ctx = DatasetContext(
        class_map=ClassMap.fit(graphs),
        cost_baseline=CostBaseline.fit(graphs, split.train),
        split_info=split_info,
    )
    return graphs, split, ctx
