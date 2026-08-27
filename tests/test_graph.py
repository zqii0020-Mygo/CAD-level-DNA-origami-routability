"""Tests for the graph-extraction stage.  Runs under pytest or standalone.

The point of these is that the graph is a *lossless enough* view of the design:
whatever the router can see in the link graph, a GNN must be able to see in the
tensors -- in particular the port a crossover lands on, which is what carries
the parity invariant.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadna import (  # noqa: E402
    CADParams,
    build_link_graph,
    evaluate,
    generate,
    sample_params,
)
from cadna.graph import (  # noqa: E402
    NODE_TYPES,
    PRECHECK_FIELDS,
    R_ADJACENT,
    R_MATE,
    R_XO_ON,
    RELATIONS,
    TARGET_FIELDS,
    HeteroGraph,
    build_graph,
    targets_from_label,
)

SEEDS = range(80)


def _col(hg: HeteroGraph, ntype: str, name: str) -> np.ndarray:
    return hg.x[ntype][:, hg.node_fields[ntype].index(name)]


def _ecol(hg: HeteroGraph, rel, name: str) -> np.ndarray:
    return hg.edge_attr[rel][:, hg.edge_fields[rel].index(name)]


def test_node_and_edge_counts_match_the_design():
    for seed in SEEDS:
        d = generate(sample_params(seed))
        hg = build_graph(d)
        assert hg.num_nodes("feature") == len(d.features)
        assert hg.num_nodes("bundle") == len(d.bundles)
        assert hg.num_nodes("cylinder") == len(d.cylinders)
        assert hg.num_nodes("crossover") == len(d.crossovers)
        # symmetric relations are stored both ways, crossovers touch two helices
        assert hg.num_edges(R_ADJACENT) == 2 * len(d.adjacencies)
        assert hg.num_edges(R_MATE) == 2 * len(d.mates)
        assert hg.num_edges(R_XO_ON) == 2 * len(d.crossovers)


def test_schema_is_the_same_for_every_design():
    """One design must not have a wider feature matrix than another."""
    dims: set[tuple] = set()
    for seed in SEEDS:
        hg = build_graph(generate(sample_params(seed)))
        dims.add(tuple((t, hg.x[t].shape[1]) for t in NODE_TYPES))
        assert set(hg.edge_index) == set(RELATIONS)
    assert len(dims) == 1, f"{len(dims)} different node schemas"


def test_ports_survive_the_export():
    """The crossover nodes must reproduce the link graph's port structure.

    A `cylinder--adjacent--cylinder` edge alone says nothing about *which* helix
    ends get joined, so the parity invariant would be invisible.  The port lives
    on the crossover node (`port_a_is_lo`, `is_flip`) and on the incidence edge.
    """
    for seed in SEEDS:
        d = generate(sample_params(seed))
        if not d.crossovers:
            continue
        hg = build_graph(d)
        lg = build_link_graph(d)
        cyl_row = {c.id: i for i, c in enumerate(d.cylinders)}

        # every equal-port crossover link of the link graph is visible as a
        # flipping crossover node sitting on both its cylinders
        flip = _col(hg, "crossover", "is_flip")
        is_lo = _ecol(hg, R_XO_ON, "is_lo")
        src, dst = hg.edge_index[R_XO_ON]
        seen = set()
        for e in range(src.shape[0]):
            seen.add((int(src[e]), int(dst[e]), bool(is_lo[e])))
        for x in d.crossovers:
            i = x.id
            for cid in (x.cyl_a, x.cyl_b):
                assert (i, cyl_row[cid], True) in seen or (i, cyl_row[cid], False) in seen

        n_flip_links = sum(lk.flip and lk.kind == "crossover" for lk in lg.links)
        assert n_flip_links <= int(flip.sum()), "collapsed links exceed the crossover sites"


def test_port_degrees_agree_with_the_link_graph():
    for seed in SEEDS:
        d = generate(sample_params(seed))
        if not d.cylinders:
            continue
        hg = build_graph(d)
        lg = build_link_graph(d)
        lo = _col(hg, "cylinder", "port_deg_lo")
        hi = _col(hg, "cylinder", "port_deg_hi")
        for i, c in enumerate(d.cylinders):
            assert lo[i] == lg.port_degree(c.id, "lo")
            assert hi[i] == lg.port_degree(c.id, "hi")


def test_containment_edges_follow_the_hierarchy():
    for seed in SEEDS:
        d = generate(sample_params(seed))
        hg = build_graph(d)
        bun_row = {b.id: i for i, b in enumerate(d.bundles)}
        feat_row = {f.id: i for i, f in enumerate(d.features)}
        src, dst = hg.edge_index[("bundle", "contains", "cylinder")]
        for e, c in enumerate(d.cylinders):
            assert int(src[e]) == bun_row[c.bundle_id]
            assert int(dst[e]) == e
        src, dst = hg.edge_index[("feature", "contains", "bundle")]
        for e, b in enumerate(d.bundles):
            assert int(src[e]) == feat_row[b.feature_id]
            assert int(dst[e]) == e
        # reverse relations are the same edges, transposed
        for fwd, rev in ((("bundle", "contains", "cylinder"), ("cylinder", "in", "bundle")),
                         (("feature", "contains", "bundle"), ("bundle", "in", "feature"))):
            assert np.array_equal(hg.edge_index[fwd][::-1], hg.edge_index[rev])


def test_coordinates_are_normalised():
    for seed in SEEDS:
        d = generate(sample_params(seed))
        hg = build_graph(d)
        if not d.cylinders:
            continue
        for axis in ("cx", "cy", "cz"):
            v = _col(hg, "cylinder", axis)
            assert np.abs(v).max() <= 1.0 + 1e-5, f"seed {seed}: {axis} outside [-1, 1]"
        assert hg.meta["coord_scale"] > 0


def test_precheck_block_is_separable():
    """The obstruction detectors live in their own block, never in graph_x."""
    hg = build_graph(generate(sample_params(3)))
    assert list(hg.precheck_fields) == list(PRECHECK_FIELDS)
    assert not (set(hg.graph_fields) & set(PRECHECK_FIELDS))


def test_parity_blocked_flag_matches_the_invariant():
    """An odd ring of equal-port links is the classic odd-helix obstruction."""
    for n in (5, 7):
        p = CADParams(shape="polygon_ring", lattice="honeycomb", n_sides=n,
                      edge_bp=63, helices_per_edge=1)
        d = generate(p)
        hg = build_graph(d, evaluate(d))
        i = hg.precheck_fields.index("parity_blocked")
        n_noflip = hg.precheck_x[hg.precheck_fields.index("n_links_noflip")]
        odd = hg.num_nodes("cylinder") % 2 == 1
        assert hg.precheck_x[i] == float(n_noflip == 0 and odd)


def test_targets_carry_censoring_and_unknowns():
    row = {
        "design_id": "x", "routable": "False", "hamilton": "", "hamilton_path": "True",
        "failure_class": "timeout", "nodes_expanded": "200000", "timeout": "True",
        "staple_ok": "", "searched": "True",
    }
    y = targets_from_label(row)
    assert set(y) == set(TARGET_FIELDS)
    assert y["routable"] == 0.0 and y["hamilton_path"] == 1.0
    assert math.isnan(y["hamilton"]), "a timed-out cycle search is unknown, not False"
    assert math.isnan(y["staple_ok"])
    assert y["timeout"] == 1.0 and y["nodes_expanded"] == 200000.0
    assert abs(y["log_nodes_expanded"] - math.log1p(200000)) < 1e-9
    assert y["failure_class_id"] >= 0

    d = generate(sample_params(1))
    lab = evaluate(d)
    assert set(build_graph(d, lab).y) == set(TARGET_FIELDS)


def test_npz_roundtrip(tmp_path=None):
    out = Path(tmp_path) if tmp_path else Path(__file__).parent / "_tmp"
    out.mkdir(parents=True, exist_ok=True)
    for seed in (0, 5, 13):
        d = generate(sample_params(seed))
        hg = build_graph(d, evaluate(d))
        back = HeteroGraph.load(hg.save(out / f"{d.id}.npz"))
        back.validate()
        assert back.design_id == hg.design_id
        assert back.node_fields == hg.node_fields and back.edge_fields == hg.edge_fields
        assert back.graph_fields == hg.graph_fields
        # NaN marks an unknown target, so compare NaN-aware rather than by ==
        assert set(back.y) == set(hg.y)
        for k, v in hg.y.items():
            assert back.y[k] == v or (math.isnan(v) and math.isnan(back.y[k])), k
        for t in hg.x:
            assert np.array_equal(hg.x[t], back.x[t]), t
        for rel in hg.edge_index:
            assert np.array_equal(hg.edge_index[rel], back.edge_index[rel]), rel
        for rel in hg.edge_attr:
            assert np.array_equal(hg.edge_attr[rel], back.edge_attr[rel]), rel


def test_to_pyg_if_available():
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        print("  (skipped: torch_geometric not installed)")
        return
    d = generate(sample_params(2))
    hg = build_graph(d, evaluate(d))
    data = hg.to_pyg()
    for t in NODE_TYPES:
        assert data[t].num_nodes == hg.num_nodes(t)
    for rel in RELATIONS:
        assert data[rel].edge_index.shape[1] == hg.num_edges(rel)
    assert data.graph_x.shape[1] == hg.graph_x.size + hg.precheck_x.size
    assert data.validate()


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
