"""Tests for the metrics, the split, the normaliser and the model shapes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadna import evaluate, generate, sample_params  # noqa: E402
from cadna.graph import NODE_TYPES, build_graph  # noqa: E402
from models.data import (  # noqa: E402
    CLASS_NAMES,
    N_CLASSES,
    ClassMap,
    CostBaseline,
    Normaliser,
    graph_size,
    make_split,
    to_pyg_list,
)
from models.provenance import dataset_provenance, git_state, provenance  # noqa: E402
from models.train import DatasetContext, score_sliced, slice_masks  # noqa: E402
from models.metrics import (  # noqa: E402
    balanced_accuracy,
    mae,
    macro_f1,
    multiclass_accuracy,
    roc_auc,
    spearman,
    stratified_spearman,
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


def test_oversampled_designs_stay_out_of_val_and_test():
    """Rare-class draws are not from the natural distribution -- train only."""
    graphs = _tiny_dataset(40)
    for g in graphs[::5]:
        g.meta["sampling"] = "rare:timeout"
    split = make_split(graphs, seed=0)
    marked = {i for i, g in enumerate(graphs) if g.meta.get("sampling", "iid") != "iid"}
    assert marked, "the fixture marked nothing"
    assert marked <= set(split.train)
    assert not (marked & set(split.val)) and not (marked & set(split.test))
    assert sorted(split.train + split.val + split.test) == list(range(len(graphs)))


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
    cmap = ClassMap.fit(graphs)
    for include_precheck in (False, True):
        data = to_pyg_list(graphs, include_precheck=include_precheck, class_map=cmap)
        batch = next(iter(DataLoader(data, batch_size=6)))
        graph_dim = int(data[0].graph_x.shape[1])
        in_dims = {t: int(data[0][t].x.shape[1]) for t in NODE_TYPES}

        for model in (GraphMLP(graph_dim, hidden=16, n_classes=cmap.n),
                      HeteroGNN(in_dims, graph_dim, hidden=16, layers=2, n_classes=cmap.n)):
            out = model(batch)
            assert out["routable"].shape == (6,)
            assert out["hamilton"].shape == (6,)
            assert out["cost"].shape == (6,)
            assert out["failure"].shape == (6, cmap.n)
            assert torch.isfinite(out["failure"]).all()


def test_precheck_block_changes_the_input_width():
    without = to_pyg_list(_tiny_dataset(4), include_precheck=False)
    with_ = to_pyg_list(_tiny_dataset(4), include_precheck=True)
    assert with_[0].graph_x.shape[1] == without[0].graph_x.shape[1] + 10


def test_class_ids_are_shifted_into_range():
    graphs = _tiny_dataset(40)
    cmap = ClassMap.fit(graphs)
    data = to_pyg_list(graphs, include_precheck=False, class_map=cmap)
    ids = np.array([int(d.y_class) for d in data])
    assert ids.min() >= 0 and ids.max() < cmap.n <= N_CLASSES


def test_class_map_only_keeps_classes_with_examples():
    """`export` never fires, so the failure head must not carry a slot for it."""
    graphs = _tiny_dataset(60)
    cmap = ClassMap.fit(graphs)
    present = {CLASS_NAMES[int(g.y["failure_class_id"]) + 1] for g in graphs}
    assert set(cmap.names) == present
    assert "export" not in cmap.names
    assert sorted(cmap.to_slot.values()) == list(range(cmap.n)), "slots are not contiguous"
    assert set(cmap.names) & set(cmap.dropped) == set()


# ------------------------------------------------------------------ cost target
def test_cost_baseline_is_fitted_on_the_training_split_only():
    graphs = _tiny_dataset(60)
    split = make_split(graphs, seed=0)
    before = CostBaseline.fit(graphs, split.train, n_bins=3)
    victim = split.test[0]
    graphs[victim].y["log_nodes_expanded"] = 1e6
    after = CostBaseline.fit(graphs, split.train, n_bins=3)
    assert np.allclose(before.medians, after.medians)


def test_cost_baseline_ignores_censored_designs():
    """A timed-out design reports the budget, which is a bound, not a cost."""
    graphs = _tiny_dataset(40)
    idx = list(range(len(graphs)))
    honest = CostBaseline.fit(graphs, idx, n_bins=3)
    for g in graphs:                      # pretend every design hit the budget
        g.y["timeout"] = 1.0
        g.y["log_nodes_expanded"] = 99.0
    censored = CostBaseline.fit(graphs, idx, n_bins=3)
    assert (honest.medians != 99.0).all()
    assert (censored.medians != 99.0).all(), "a censored value leaked into the baseline"


def test_size_stratified_correlation_cannot_be_won_by_size():
    graphs = _tiny_dataset(40)
    base = CostBaseline.fit(graphs, list(range(len(graphs))), n_bins=4)
    sizes = np.array([graph_size(g) for g in graphs], dtype=float)
    bins = base.bin_of(sizes)
    y = np.array([float(g.y["log_nodes_expanded"]) for g in graphs])
    size_pred = base.predict(sizes)
    within = stratified_spearman(y, size_pred, bins)
    assert np.isnan(within) or abs(within) < 1e-9, "the size baseline scored inside its own bin"


# ----------------------------------------------------------------------- slices
def _fake_pred(searched, routable, klass):
    n = len(searched)
    z = np.zeros(n)
    return {
        "searched": np.asarray(searched, dtype=float),
        "timeout": z.copy(),
        "y_routable": np.asarray(routable, dtype=float),
        "p_routable": np.asarray(routable, dtype=float),
        "y_hamilton": np.asarray(routable, dtype=float),
        "p_hamilton": np.asarray(routable, dtype=float),
        "y_class": np.asarray(klass, dtype=float),
        "hat_class": np.asarray(klass, dtype=float),
        "y_cost": z.copy(),
        "y_cost_resid": z.copy(),
        "cost_base": z.copy(),
        "p_cost": z.copy(),
        "size": np.arange(n, dtype=float),
        "precheck_ok": np.asarray(searched, dtype=float),
    }


def test_slices_partition_the_test_set():
    pred = _fake_pred([1, 0, 1, 0, 1], [1, 0, 0, 0, 1], [0, 1, 2, 1, 0])
    m = slice_masks(pred)
    assert set(m) == {"all", "precheck_decided", "precheck_undecided"}
    assert m["all"].all()
    assert not (m["precheck_decided"] & m["precheck_undecided"]).any()
    assert (m["precheck_decided"] | m["precheck_undecided"]).all()
    # the decided slice is exactly the designs the search never saw
    assert not (pred["searched"][m["precheck_decided"]] > 0.5).any()


def test_decided_slice_reports_no_binary_score():
    """Every precheck-decided design is unroutable, so discrimination is undefined.

    Reporting the surviving recall there would hand an all-negative predictor a
    1.000 for saying nothing.
    """
    pred = _fake_pred([1, 1, 0, 0], [1, 0, 0, 0], [0, 3, 1, 2])
    graphs = _tiny_dataset(12)
    ctx = DatasetContext(class_map=ClassMap.fit(graphs),
                         cost_baseline=CostBaseline.fit(graphs, list(range(12)), n_bins=2))
    sliced = score_sliced(pred, ctx)
    assert sliced["all"]["n"] == 4
    assert sliced["precheck_decided"]["n"] == 2
    assert sliced["precheck_undecided"]["n"] == 2
    assert np.isnan(sliced["precheck_decided"]["routable_bacc"])
    assert not np.isnan(sliced["precheck_undecided"]["routable_bacc"])
    # the failure class is still scoreable on the decided slice
    assert not np.isnan(sliced["precheck_decided"]["failure_acc"])


# ------------------------------------------------------------------ provenance
def test_provenance_identifies_code_and_data():
    block = provenance(["data/designs_v0_graphs"], run={"seeds": 1})
    assert set(block) >= {"created_utc", "git", "command", "environment", "datasets", "run"}
    assert block["git"]["commit"] is None or len(block["git"]["commit"]) == 40
    assert block["environment"]["python"]
    ds = block["datasets"][0]
    if ds["n_graphs"]:                      # skip when the dataset is not built
        assert ds["graphs_index_sha1"], "a dataset must be identified by content, not by name"
        assert ds["sampling"], "the iid / rare breakdown is part of what produced the table"


def test_git_state_reports_dirtiness():
    st = git_state()
    assert set(st) == {"commit", "branch", "describe", "dirty"}
    # None means git did not answer; it must never silently look clean
    assert st["dirty"] in (True, False, None)


def test_dataset_provenance_survives_a_missing_directory():
    ds = dataset_provenance("data/does_not_exist_graphs")
    assert ds["n_graphs"] == 0 and ds["graphs_index_sha1"] is None


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
