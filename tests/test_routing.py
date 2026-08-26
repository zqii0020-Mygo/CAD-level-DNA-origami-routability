"""Tests for the precheck / routing / label stage.  Runs under pytest or standalone."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadna import (  # noqa: E402
    CADParams,
    build_link_graph,
    evaluate,
    find_scaffold_path,
    find_scaffold_route,
    generate,
    precheck,
    sample_params,
)
from cadna.linkgraph import Link, LinkGraph, opposite  # noqa: E402
from cadna.precheck import FAILURE_CLASSES  # noqa: E402
from cadna.routing import validate_route  # noqa: E402

SEEDS = range(120)


def _design_level_links(design):
    """Every helix-end pair the *design* (not the link graph) actually connects."""
    cyl = {c.id: c for c in design.cylinders}
    ok = set()
    for x in design.crossovers:
        ca, cb = cyl[x.cyl_a], cyl[x.cyl_b]
        pa = "lo" if (x.bp_index - ca.bp_start) * 2 < ca.bp_len else "hi"
        pb = "lo" if (x.bp_index - cb.bp_start) * 2 < cb.bp_len else "hi"
        ok.add(frozenset({(ca.id, pa), (cb.id, pb)}))
    for m in design.mates:
        pa = "lo" if m.end_a == "start" else "hi"
        pb = "lo" if m.end_b == "start" else "hi"
        ok.add(frozenset({(m.cyl_a, pa), (m.cyl_b, pb)}))
    return ok


def test_repaired_honeycomb_block_routes():
    """The 6-ring a repaired 3x3 honeycomb block collapses to is Hamiltonian."""
    p = CADParams(shape="brick", lattice="honeycomb", n_rows=3, n_cols=3,
                  length_bp=63, crossover_end_margin_bp=0)
    lab = evaluate(generate(p))
    assert lab.stats["n_cylinders"] == 6
    assert lab.hamilton is True
    assert lab.export_ok is True


def _ring_link_graph(n: int) -> LinkGraph:
    """n cylinders in a ring, each neighbouring pair joined by a lo-lo and a hi-hi link.

    This is the idealised cross-section of a single bundle: every link is an
    equal-port (flipping) link.
    """
    lg = LinkGraph(n_cylinders=n)
    for i in range(n):
        j = (i + 1) % n
        for port in ("lo", "hi"):
            lk = Link(id=len(lg.links), cyl_a=i, port_a=port, cyl_b=j, port_b=port,
                      kind="crossover", source_id=-1)
            lg.links.append(lk)
            lg.adj.setdefault((i, port), []).append((j, port, lk.id))
            lg.adj.setdefault((j, port), []).append((i, port, lk.id))
    return lg


def test_parity_blocks_odd_rings():
    """The search must independently confirm the parity invariant."""
    for n in (4, 6, 8, 10):
        assert find_scaffold_route(_ring_link_graph(n)).routed, f"even ring {n} should route"
    for n in (5, 7, 9):
        res = find_scaffold_route(_ring_link_graph(n))
        assert res.status == "no_route", f"odd ring {n} should not close: {res.status}"


def test_parity_precheck_agrees_with_the_search():
    """Where the cheap parity rule fires, the DFS must reach the same verdict."""
    fired = 0
    for seed in range(300):
        d = generate(sample_params(seed))
        lg = build_link_graph(d)
        pc = precheck(d, lg)
        if pc.reason != "hamilton" or "parity" not in pc.detail:
            continue
        fired += 1
        assert lg.n_noflip == 0 and lg.n_cylinders % 2 == 1
        res = find_scaffold_route(lg, node_budget=100_000, time_budget_s=5.0)
        assert res.status == "no_route", f"{d.id}: parity said no, search said {res.status}"
    assert fired >= 1, "the sampler never produced a parity-blocked design"


def test_flat_sheet_has_a_path_but_no_cycle():
    """A single row of helices is a lattice path: it closes only via the scaffold loop."""
    p = CADParams(shape="plate", lattice="honeycomb", n_rows=1, n_cols=6,
                  length_bp=84, repair_cross_section=False, crossover_end_margin_bp=0)
    lg = build_link_graph(generate(p))
    assert find_scaffold_route(lg).status == "no_route"
    assert find_scaffold_path(lg).status == "routed"


def test_every_returned_route_is_valid():
    """Whatever the search returns must survive independent re-checking."""
    checked = 0
    for seed in SEEDS:
        d = generate(sample_params(seed))
        lg = build_link_graph(d)
        res = find_scaffold_route(lg, time_budget_s=1.0)
        if not res.routed:
            continue
        checked += 1
        ok, detail = validate_route(lg, res)
        assert ok, f"{d.id}: {detail}"

        # the route visits every cylinder exactly once
        assert sorted(s.cyl for s in res.route) == list(range(lg.n_cylinders))
        # and every step corresponds to a real crossover or mate of the design
        real = _design_level_links(d)
        for i, step in enumerate(res.route):
            prev = res.route[i - 1]
            pair = frozenset({(prev.cyl, opposite(prev.entry_port)),
                              (step.cyl, step.entry_port)})
            assert pair in real, f"{d.id} step {i}: {sorted(pair)} is not a real connection"
    assert checked >= 20, f"only {checked} routed designs to check"


def test_links_are_used_at_most_once():
    for seed in SEEDS:
        lg = build_link_graph(generate(sample_params(seed)))
        res = find_scaffold_route(lg, time_budget_s=1.0)
        if res.routed:
            ids = res.link_ids()
            assert len(ids) == len(set(ids)) == lg.n_cylinders


def test_a_cycle_implies_a_path():
    checked = 0
    for seed in SEEDS:
        lg = build_link_graph(generate(sample_params(seed)))
        if not find_scaffold_route(lg, time_budget_s=1.0).routed:
            continue
        res = find_scaffold_path(lg, time_budget_s=2.0)
        if res.timeout:
            continue        # undecided, not a contradiction
        checked += 1
        assert res.routed, "a Hamiltonian cycle contains a Hamiltonian path"
    assert checked >= 20, f"only {checked} decided cases"


def test_prune_does_not_change_the_answer():
    """The feasibility prune must only cut branches that cannot succeed."""
    for seed in SEEDS:
        lg = build_link_graph(generate(sample_params(seed)))
        a = find_scaffold_route(lg, node_budget=60_000, time_budget_s=1.0, prune=True)
        b = find_scaffold_route(lg, node_budget=60_000, time_budget_s=1.0, prune=False)
        if "timeout" in (a.status, b.status):
            continue
        assert a.routed == b.routed, f"seed {seed}: prune={a.status} noprune={b.status}"


def test_labels_are_consistent():
    for seed in SEEDS:
        lab = evaluate(generate(sample_params(seed)), time_budget_s=1.0)
        assert lab.failure_class is None or lab.failure_class in FAILURE_CLASSES
        assert lab.routable == (lab.failure_class is None)
        if lab.routable:
            assert lab.hamilton is True and lab.export_ok and lab.staple_ok
            assert lab.scaffold_ok
        if lab.hamilton is None:
            assert lab.failure_class == "timeout"
        if lab.hamilton is True and lab.hamilton_path is not None:
            assert lab.hamilton_path is True
        if not lab.searched:
            assert lab.nodes_expanded == 0


def test_evaluation_is_deterministic():
    # Node budget only: a wall-clock budget cuts the search at a different depth
    # each run, which is exactly the nondeterminism this test must not see.
    for seed in (0, 4, 19, 41):
        d = generate(sample_params(seed))
        a = evaluate(d, node_budget=20_000, time_budget_s=1e9).to_row()
        b = evaluate(d, node_budget=20_000, time_budget_s=1e9).to_row()
        for k in a:
            if k.endswith("elapsed_s"):
                continue  # wall clock
            assert a[k] == b[k], f"seed {seed} field {k}: {a[k]} != {b[k]}"


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
