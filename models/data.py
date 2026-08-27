"""Loading, splitting and normalising the graph dataset.

The split is by design, deterministic in a seed, and stratified on the shape
family so every split sees all four routing regimes.  Feature statistics come
from the training split only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cadna.graph import NODE_TYPES, HeteroGraph
from cadna.precheck import FAILURE_CLASSES

# 0 == "no failure"; the seven FAILURE_CLASSES follow, so id = stored_id + 1
CLASS_NAMES = ("none",) + FAILURE_CLASSES
N_CLASSES = len(CLASS_NAMES)


@dataclass
class Split:
    train: list[int]
    val: list[int]
    test: list[int]

    def __repr__(self) -> str:
        return f"Split(train={len(self.train)}, val={len(self.val)}, test={len(self.test)})"


def load_graphs(directory: str | Path) -> list[HeteroGraph]:
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz graphs in {directory}")
    return [HeteroGraph.load(p) for p in paths]


def make_split(graphs: list[HeteroGraph], seed: int = 0,
               fracs: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Split:
    """Shape-stratified random split, deterministic in `seed`."""
    by_shape: dict[str, list[int]] = {}
    for i, g in enumerate(graphs):
        by_shape.setdefault(str(g.meta.get("shape", "?")), []).append(i)

    train: list[int] = []
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


def to_pyg_list(graphs: list[HeteroGraph], include_precheck: bool):
    """Convert once, up front: 1000 small graphs fit in memory comfortably."""
    out = []
    for g in graphs:
        d = g.to_pyg(include_precheck=include_precheck)
        # failure_class_id is -1 for "no failure"; shift so 0 == none
        d.y_class = (d.y_failure_class_id + 1).long()
        out.append(d)
    return out


def target_matrix(graphs: list[HeteroGraph], key: str) -> np.ndarray:
    return np.array([float(g.y[key]) for g in graphs], dtype=np.float64)
