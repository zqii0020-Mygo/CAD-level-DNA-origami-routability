"""Tests for the metrics, the split, the normaliser and the model shapes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadna import evaluate, generate, sample_params  # noqa: E402
from cadna.graph import NODE_TYPES, build_graph  # noqa: E402
from models.data import N_CLASSES, Normaliser, make_split, to_pyg_list  # noqa: E402
from models.metrics import (  # noqa: E402
    balanced_accuracy,
    mae,
    macro_f1,
    multiclass_accuracy,
    roc_auc,
    spearman,
)


def _tiny_dataset(n: int = 24):
    out = []
    for seed in range(n):
        d = generate(sample_params(seed))
        out.append(build_graph(d, evaluate(d, node_budget=2000, time_budget_s=0.5)))
    return out


# ---------------------------------------------------------------------- metrics
def test_roc_auc_matches_a_worked_example():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert abs(roc_auc(y, p) - 0.75) < 1e-9


def test_roc_auc_handles_ties_and_extremes():
    y = np.array([0.0, 1.0, 0.0, 1.0])
    assert abs(roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) - 0.5) < 1e-9   # all tied
    assert abs(roc_auc(y, np.array([0.0, 1.0, 0.0, 1.0])) - 1.0) < 1e-9   # perfect
    assert abs(roc_auc(y, np.array([1.0, 0.0, 1.0, 0.0])) - 0.0) < 1e-9   # inverted


def test_metrics_ignore_nan_targets():
    y = np.array([1.0, 0.0, np.nan, 1.0])
    p = np.array([0.9, 0.1, 0.9, 0.8])
    assert abs(roc_auc(y, p) - 1.0) < 1e-9
    assert abs(balanced_accuracy(y, p) - 1.0) < 1e-9
    assert abs(mae(y, p) - np.mean([0.1, 0.1, 0.2])) < 1e-9


def test_balanced_accuracy_punishes_the_majority_predictor():
    y = np.array([1.0] * 90 + [0.0] * 10)
    always_one = np.ones(100)
    assert multiclass_accuracy(y, always_one) == 0.9      # accuracy looks fine
    assert abs(balanced_accuracy(y, always_one) - 0.5) < 1e-9   # balanced does not


def test_spearman_is_rank_based():
    y = np.arange(10.0)
    assert abs(spearman(y, y**3) - 1.0) < 1e-9            # monotone, very nonlinear
    assert abs(spearman(y, -y) + 1.0) < 1e-9
    assert np.isnan(spearman(y, np.zeros(10)))            # no variance


def test_macro_f1_skips_absent_classes():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert abs(macro_f1(y, y, N_CLASSES) - 1.0) < 1e-9
    # predicting one class only: perfect recall on it, zero on the other
    got = macro_f1(y, np.zeros(4), N_CLASSES)
    assert abs(got - (2 * 0.5 * 1.0 / 1.5) / 2) < 1e-9


# ------------------------------------------------------------------------ split
def test_split_is_disjoint_complete_and_deterministic():
    graphs = _tiny_dataset(40)
    a = make_split(graphs, seed=0)
    b = make_split(graphs, seed=0)
    assert (a.train, a.val, a.test) == (b.train, b.val, b.test)

    everything = a.train + a.val + a.test
    assert sorted(everything) == list(range(len(graphs)))
    assert len(set(everything)) == len(graphs), "a design landed in two splits"
    assert make_split(graphs, seed=1).train != a.train


def test_split_is_stratified_by_shape():
    graphs = _tiny_dataset(60)
    split = make_split(graphs, seed=0)
    shapes = {str(g.meta["shape"]) for g in graphs}
    train_shapes = {str(graphs[i].meta["shape"]) for i in split.train}
    assert train_shapes == shapes, "training split is missing a routing regime"


# ------------------------------------------------------------------ normaliser
def test_normaliser_uses_training_statistics_only():
    graphs = _tiny_dataset(40)
    split = make_split(graphs, seed=0)

    before = Normaliser.fit(graphs, split.train)
    # corrupt a held-out design; training statistics must not move
    victim = split.test[0]
    graphs[victim].graph_x = graphs[victim].graph_x + 1000.0
    after = Normaliser.fit(graphs, split.train)
    assert np.allclose(before.graph_mean, after.graph_mean)
    assert np.allclose(before.graph_std, after.graph_std)


def test_normaliser_standardises_the_training_split():
    graphs = _tiny_dataset(40)
    split = make_split(graphs, seed=0)
    norm = Normaliser.fit(graphs, split.train)
    norm.apply(graphs)
    stack = np.stack([graphs[i].graph_x for i in split.train])
    varying = stack.std(axis=0) > 1e-3
    assert np.abs(stack[:, varying].mean(axis=0)).max() < 1e-4
    assert np.abs(stack[:, varying].std(axis=0) - 1.0).max() < 1e-3


# ---------------------------------------------------------------------- models
def test_models_produce_the_expected_head_shapes():
    import torch
    from torch_geometric.loader import DataLoader

    from models.nets import GraphMLP, HeteroGNN

    graphs = _tiny_dataset(12)
    for include_precheck in (False, True):
        data = to_pyg_list(graphs, include_precheck=include_precheck)
        batch = next(iter(DataLoader(data, batch_size=6)))
        graph_dim = int(data[0].graph_x.shape[1])
        in_dims = {t: int(data[0][t].x.shape[1]) for t in NODE_TYPES}

        for model in (GraphMLP(graph_dim, hidden=16),
                      HeteroGNN(in_dims, graph_dim, hidden=16, layers=2)):
            out = model(batch)
            assert out["routable"].shape == (6,)
            assert out["hamilton"].shape == (6,)
            assert out["cost"].shape == (6,)
            assert out["failure"].shape == (6, N_CLASSES)
            assert torch.isfinite(out["failure"]).all()


def test_precheck_block_changes_the_input_width():
    without = to_pyg_list(_tiny_dataset(4), include_precheck=False)
    with_ = to_pyg_list(_tiny_dataset(4), include_precheck=True)
    assert with_[0].graph_x.shape[1] == without[0].graph_x.shape[1] + 10


def test_class_ids_are_shifted_into_range():
    data = to_pyg_list(_tiny_dataset(40), include_precheck=False)
    ids = np.array([int(d.y_class) for d in data])
    assert ids.min() >= 0 and ids.max() < N_CLASSES


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
