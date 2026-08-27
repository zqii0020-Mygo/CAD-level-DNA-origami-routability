"""Loading, splitting and normalising the graph dataset.

The split is by design, deterministic in a seed, and stratified on the shape
family so every split sees all four routing regimes.  Feature statistics come
from the training split only.

Two things here are not bookkeeping but modelling decisions:

`ClassMap` gives the failure head one slot per class that actually occurs.
`export` never fires -- a route that fails `validate_route` is a router bug, not
a property of the design -- so a fixed seven-way head spends an output on a
class with no examples, which can only cost it precision elsewhere.

`CostBaseline` makes the search-cost target say something other than "this
design is big".  `nodes_expanded` grows with the number of cylinders, so a
predictor that has learned only the size gets a high rank correlation for free.
The baseline is a per-size-bin median fitted on the *training* split; the model
regresses the residual against it, and `cost_rho_within` scores the residual
inside size bins, where the size signal is gone.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from cadna.graph import NODE_TYPES, HeteroGraph
from cadna.precheck import FAILURE_CLASSES

# 0 == "no failure"; the FAILURE_CLASSES follow, so id = stored_id + 1
CLASS_NAMES = ("none",) + FAILURE_CLASSES
N_CLASSES = len(CLASS_NAMES)

COST_BINS = 8


@dataclass
class Split:
    train: list[int]
    val: list[int]
    test: list[int]

    def __repr__(self) -> str:
        return f"Split(train={len(self.train)}, val={len(self.val)}, test={len(self.test)})"


def load_graphs(directory: str | Path | Sequence[str | Path]) -> list[HeteroGraph]:
    """Load one graph directory, or several concatenated in the order given.

    Several is how an oversampled tail is added to an iid dataset without
    touching either manifest: the marker in each graph decides where it may
    land, not the directory it came from.
    """
    dirs = ([directory] if isinstance(directory, (str, Path))
            else list(directory))
    dirs = [d for part in dirs for d in str(part).split(",") if d]
    graphs: list[HeteroGraph] = []
    for d in dirs:
        paths = sorted(Path(d).glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"no .npz graphs in {d}")
        graphs += [HeteroGraph.load(p) for p in paths]
    return graphs


def make_split(graphs: list[HeteroGraph], seed: int = 0,
               fracs: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Split:
    """Shape-stratified random split, deterministic in `seed`.

    Designs drawn by rare-class oversampling (`meta["sampling"] != "iid"`) go
    into the training split only.  Validation and test have to stay a sample of
    the natural distribution: a test set topped up with hand-picked timeouts
    reports a class balance that no real design stream has.
    """
    by_shape: dict[str, list[int]] = {}
    extra_train: list[int] = []
    for i, g in enumerate(graphs):
        if str(g.meta.get("sampling", "iid")) != "iid":
            extra_train.append(i)
            continue
        by_shape.setdefault(str(g.meta.get("shape", "?")), []).append(i)

    train: list[int] = list(extra_train)
    val: list[int] = []
    test: list[int] = []
    rng = random.Random(seed)
    for shape in sorted(by_shape):
        idx = sorted(by_shape[shape])
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(fracs[0] * n))
        n_va = int(round(fracs[1] * n))
        train += idx[:n_tr]
        val += idx[n_tr:n_tr + n_va]
        test += idx[n_tr + n_va:]
    return Split(sorted(train), sorted(val), sorted(test))


def _stratified_pool_split(graphs: list[HeteroGraph], pool: list[int], seed: int,
                           val_frac: float) -> tuple[list[int], list[int]]:
    """Split an in-distribution pool into train / val, stratified by shape."""
    by_shape: dict[str, list[int]] = {}
    forced_train: list[int] = []
    for i in pool:
        g = graphs[i]
        if str(g.meta.get("sampling", "iid")) != "iid":
            forced_train.append(i)          # rare draws never go into validation
            continue
        by_shape.setdefault(str(g.meta.get("shape", "?")), []).append(i)

    rng = random.Random(seed)
    train, val = list(forced_train), []
    for shape in sorted(by_shape):
        idx = sorted(by_shape[shape])
        rng.shuffle(idx)
        n_va = int(round(val_frac * len(idx)))
        val += idx[:n_va]
        train += idx[n_va:]
    return sorted(train), sorted(val)


def make_ood_split(graphs: list[HeteroGraph], is_test, seed: int = 0,
                   val_frac: float = 0.176) -> tuple[Split, dict[str, Any]]:
    """Split where the test set is a *region* of the design space, not a sample.

    `is_test(graph)` selects the held-out region -- one shape family, or every
    design above a size threshold.  Three rules make it an honest extrapolation
    test rather than a relabelled random split:

    - validation comes from the training region, never the held-out one.  You
      do not get to tune on the distribution you are claiming to extrapolate to.
    - oversampled (`rare:`) designs can never be test data, since they are not
      iid.
    - a `rare:` design that falls *inside* the held-out region is dropped
      entirely rather than used for training: keeping it would show the model
      exactly the region the split exists to hide.

    Dropped designs are counted and reported, not silently discarded.
    """
    test: list[int] = []
    dropped: list[int] = []
    pool: list[int] = []
    for i, g in enumerate(graphs):
        iid = str(g.meta.get("sampling", "iid")) == "iid"
        if is_test(g):
            (test if iid else dropped).append(i)
        else:
            pool.append(i)
    train, val = _stratified_pool_split(graphs, pool, seed, val_frac)
    info = {
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_dropped_rare_in_test_region": len(dropped),
    }
    return Split(train, val, sorted(test)), info


def holdout_shape_split(graphs: list[HeteroGraph], shape: str, seed: int = 0
                        ) -> tuple[Split, dict[str, Any]]:
    """Train on three shape families, test on the fourth."""
    split, info = make_ood_split(graphs, lambda g: str(g.meta.get("shape")) == shape, seed)
    info.update({"kind": "holdout-shape", "held_out": shape})
    if not split.test:
        raise ValueError(f"no designs with shape {shape!r}")
    return split, info


def size_extrapolation_split(graphs: list[HeteroGraph], quantile: float = 0.8,
                             seed: int = 0) -> tuple[Split, dict[str, Any]]:
    """Train on the small designs, test on the large ones.

    The threshold is a quantile of the whole dataset's cylinder counts, so the
    test set is the top `1 - quantile` by size: designs strictly bigger than
    anything the model was trained on.
    """
    sizes = np.array([graph_size(g) for g in graphs], dtype=float)
    thr = float(np.quantile(sizes, quantile))
    split, info = make_ood_split(graphs, lambda g: graph_size(g) > thr, seed)
    train_sizes = [graph_size(graphs[i]) for i in split.train]
    test_sizes = [graph_size(graphs[i]) for i in split.test]
    info.update({
        "kind": "size-extrapolation", "quantile": quantile, "threshold_cylinders": thr,
        "train_size_max": max(train_sizes) if train_sizes else 0,
        "test_size_min": min(test_sizes) if test_sizes else 0,
        "test_size_max": max(test_sizes) if test_sizes else 0,
    })
    if not split.test:
        raise ValueError(f"quantile {quantile} leaves no designs above {thr}")
    return split, info


def build_split(graphs: list[HeteroGraph], spec: str = "random", seed: int = 0
                ) -> tuple[Split, dict[str, Any]]:
    """`random` | `shape:<name>` | `size:<quantile>`."""
    if spec in ("", "random"):
        split = make_split(graphs, seed=seed)
        return split, {"kind": "random", "seed": seed, "n_train": len(split.train),
                       "n_val": len(split.val), "n_test": len(split.test),
                       "n_dropped_rare_in_test_region": 0}
    kind, _, arg = spec.partition(":")
    if kind == "shape":
        return holdout_shape_split(graphs, arg, seed=seed)
    if kind == "size":
        return size_extrapolation_split(graphs, float(arg or 0.8), seed=seed)
    raise ValueError(f"unknown split spec {spec!r}; use random | shape:<name> | size:<quantile>")


@dataclass
class Normaliser:
    node_mean: dict[str, np.ndarray]
    node_std: dict[str, np.ndarray]
    graph_mean: np.ndarray
    graph_std: np.ndarray
    precheck_mean: np.ndarray
    precheck_std: np.ndarray

    @staticmethod
    def _stats(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        std[std < 1e-6] = 1.0            # constant column -> leave it alone
        return mean.astype(np.float32), std.astype(np.float32)

    @classmethod
    def fit(cls, graphs: list[HeteroGraph], train_idx: list[int]) -> "Normaliser":
        node_mean, node_std = {}, {}
        for t in NODE_TYPES:
            blocks = [graphs[i].x[t] for i in train_idx if graphs[i].x[t].shape[0]]
            if blocks:
                node_mean[t], node_std[t] = cls._stats(np.concatenate(blocks, axis=0))
            else:
                dim = graphs[train_idx[0]].x[t].shape[1]
                node_mean[t] = np.zeros(dim, np.float32)
                node_std[t] = np.ones(dim, np.float32)
        gm, gs = cls._stats(np.stack([graphs[i].graph_x for i in train_idx]))
        pm, ps = cls._stats(np.stack([graphs[i].precheck_x for i in train_idx]))
        return cls(node_mean, node_std, gm, gs, pm, ps)

    def apply(self, graphs: list[HeteroGraph]) -> None:
        """Standardise in place.  Idempotent only if called once -- call once."""
        for g in graphs:
            for t in NODE_TYPES:
                if g.x[t].shape[0]:
                    g.x[t] = ((g.x[t] - self.node_mean[t]) / self.node_std[t]).astype(np.float32)
            g.graph_x = ((g.graph_x - self.graph_mean) / self.graph_std).astype(np.float32)
            g.precheck_x = ((g.precheck_x - self.precheck_mean) / self.precheck_std).astype(np.float32)


@dataclass
class ClassMap:
    """The failure classes that actually occur, mapped onto contiguous slots.

    Presence of a class is a property of the dataset, not of an individual
    label, so this is fitted on all of it rather than on the training split.
    """

    names: list[str]
    to_slot: dict[int, int]

    @property
    def n(self) -> int:
        return len(self.names)

    def slot(self, vocab_id: int) -> int:
        return self.to_slot[int(vocab_id)]

    @classmethod
    def fit(cls, graphs: list[HeteroGraph]) -> "ClassMap":
        present = sorted({int(g.y["failure_class_id"]) + 1 for g in graphs})
        return cls(names=[CLASS_NAMES[i] for i in present],
                   to_slot={vid: k for k, vid in enumerate(present)})

    @property
    def dropped(self) -> list[str]:
        return [n for n in CLASS_NAMES if n not in self.names]


@dataclass
class CostBaseline:
    """log(nodes expanded) explained by design size alone: the number to beat.

    Fitted on the training split, on the designs that were actually searched to
    completion -- a design that hit the node budget is right-censored, so its
    count is a lower bound and would drag the median down.
    """

    edges: np.ndarray            # size bin edges, from training quantiles
    medians: np.ndarray          # median log-cost per bin
    overall: float

    def bin_of(self, sizes: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges, sizes, side="right") - 1,
                       0, len(self.medians) - 1)

    def predict(self, sizes: np.ndarray) -> np.ndarray:
        return self.medians[self.bin_of(np.asarray(sizes, dtype=float))]

    @classmethod
    def fit(cls, graphs: list[HeteroGraph], train_idx: list[int],
            n_bins: int = COST_BINS) -> "CostBaseline":
        sizes = np.array([graph_size(graphs[i]) for i in train_idx], dtype=float)
        cost = np.array([graphs[i].y.get("log_nodes_expanded", np.nan) for i in train_idx])
        usable = np.array([
            graphs[i].y.get("searched", 0.0) > 0.5 and graphs[i].y.get("timeout", 0.0) < 0.5
            for i in train_idx
        ])
        m = usable & np.isfinite(cost)
        if m.sum() < n_bins * 2:
            overall = float(np.median(cost[m])) if m.any() else 0.0
            return cls(np.array([-np.inf]), np.array([overall]), overall)

        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(sizes[m], qs))
        edges[0] = -np.inf
        medians = []
        overall = float(np.median(cost[m]))
        idx = np.clip(np.searchsorted(edges, sizes, side="right") - 1, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            sel = m & (idx == b)
            medians.append(float(np.median(cost[sel])) if sel.sum() >= 3 else overall)
        return cls(edges[:-1], np.array(medians), overall)


def graph_size(g: HeteroGraph) -> int:
    """Design size = cylinder count.  Read off the graph, so it survives scaling."""
    return g.num_nodes("cylinder")


def to_pyg_list(graphs: list[HeteroGraph], include_precheck: bool,
                class_map: "ClassMap | None" = None,
                cost_baseline: "CostBaseline | None" = None):
    """Convert once, up front: the graphs are small and fit in memory comfortably."""
    class_map = class_map or ClassMap.fit(graphs)
    out = []
    for g in graphs:
        d = g.to_pyg(include_precheck=include_precheck)
        # failure_class_id is -1 for "no failure"; shift so 0 == none, then map
        # onto the slots of the classes this dataset actually contains
        d.y_class = d.y_failure_class_id.new_tensor(
            [class_map.slot(int(d.y_failure_class_id) + 1)]
        ).long()
        size = float(graph_size(g))
        d.y_size = d.y_routable.new_tensor([size])
        base = float(cost_baseline.predict([size])[0]) if cost_baseline else 0.0
        d.y_cost_base = d.y_routable.new_tensor([base])
        d.y_cost_resid = d.y_log_nodes_expanded - d.y_cost_base
        out.append(d)
    return out


def target_matrix(graphs: list[HeteroGraph], key: str) -> np.ndarray:
    return np.array([float(g.y[key]) for g in graphs], dtype=np.float64)
