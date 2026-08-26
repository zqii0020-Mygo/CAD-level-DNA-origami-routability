"""Sanity tests for the CAD generator.  Runs under pytest or standalone."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadna import CADParams, generate, load_design, sample_params, save_design  # noqa: E402
from cadna.io import design_to_dict  # noqa: E402
from cadna.lattice import HONEYCOMB, SQUARE, get_lattice  # noqa: E402


def test_crossover_offset_is_symmetric():
    for lat in (HONEYCOMB, SQUARE):
        for row in range(-3, 4):
            for col in range(-3, 4):
                for nb in lat.neighbors(row, col):
                    a, b = (row, col), nb
                    assert lat.crossover_offset(a, b) == lat.crossover_offset(b, a)


def test_offset_tables_span_the_period():
    hc = {HONEYCOMB.crossover_offset((0, 0), nb) for nb in HONEYCOMB.neighbors(0, 0)}
    assert hc == {0, 7, 14}
    sq = set()
    for row in (0, 1):
        for col in (0, 1):
            sq |= {SQUARE.crossover_offset((row, col), nb) for nb in SQUARE.neighbors(row, col)}
    assert sq == {0, 8, 16, 24}


def test_neighbors_are_at_the_interhelix_distance():
    for lat in (HONEYCOMB, SQUARE):
        for row in range(-2, 3):
            for col in range(-2, 3):
                xa, ya = lat.position(row, col)
                for nb in lat.neighbors(row, col):
                    xb, yb = lat.position(*nb)
                    d = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
                    assert abs(d - lat.interhelix) < 1e-9, (lat.name, (row, col), nb, d)


def test_honeycomb_block_adjacency_count():
    """3x3 honeycomb block: 6 horizontal + 3 vertical lattice contacts."""
    p = CADParams(shape="brick", lattice="honeycomb", n_rows=3, n_cols=3,
                  length_bp=63, fill=1.0, stagger_bp=0, crossover_end_margin_bp=0)
    d = generate(p)
    assert len(d.cylinders) == 9
    intra = [a for a in d.adjacencies if a.kind == "intra"]
    assert len(intra) == 9
    assert all(a.overlap_bp == 63 for a in intra)


def test_crossovers_respect_phase_and_bounds():
    for seed in range(60):
        d = generate(sample_params(seed))
        lat = get_lattice(d.lattice)
        cyl = {c.id: c for c in d.cylinders}
        margin = d.params["crossover_end_margin_bp"]
        for x in d.crossovers:
            ca, cb = cyl[x.cyl_a], cyl[x.cyl_b]
            assert ca.bp_start + margin <= x.bp_index < ca.bp_end - margin
            assert cb.bp_start + margin <= x.bp_index < cb.bp_end - margin
            if ca.bundle_id == cb.bundle_id:
                off = lat.crossover_offset((ca.row, ca.col), (cb.row, cb.col))
                assert (x.bp_index - off) % lat.step == 0
            assert x.dist_to_end_bp >= margin


def test_crossover_count_matches_the_period():
    """A full-length block has floor-many crossovers per adjacency, per period."""
    p = CADParams(shape="brick", lattice="honeycomb", n_rows=2, n_cols=2,
                  length_bp=21 * 4, fill=1.0, stagger_bp=0, crossover_end_margin_bp=0)
    d = generate(p)
    intra = [a for a in d.adjacencies if a.kind == "intra"]
    per_adj = {}
    for x in d.crossovers:
        per_adj[x.adjacency_id] = per_adj.get(x.adjacency_id, 0) + 1
    assert set(per_adj) == {a.id for a in intra}
    assert all(n == 4 for n in per_adj.values())


def test_mates_use_each_end_at_most_once():
    for seed in range(60):
        d = generate(sample_params(seed))
        seen = set()
        for m in d.mates:
            for e in ((m.cyl_a, m.end_a), (m.cyl_b, m.end_b)):
                assert e not in seen, f"end {e} mated twice in {d.id}"
                seen.add(e)
            assert d.cylinders[m.cyl_a].bundle_id != d.cylinders[m.cyl_b].bundle_id
        assert not (seen & set(d.unpaired_ends))


def test_wireframe_vertex_degree_parity():
    """Even helices per edge => every end finds a partner at its vertex."""
    for n_sides in (3, 5, 6):
        for h in (2, 4):
            p = CADParams(shape="polygon_ring", lattice="honeycomb",
                          n_sides=n_sides, helices_per_edge=h, edge_bp=63)
            d = generate(p)
            assert len(d.cylinders) == n_sides * h
            assert d.unpaired_ends == []
            assert len(d.mates) == n_sides * h


def test_polyhedron_shapes_build():
    expect = {"tetrahedron": 6, "octahedron": 12, "cube": 12}
    for name, n_edges in expect.items():
        p = CADParams(shape="polyhedron", polyhedron=name, lattice="honeycomb",
                      helices_per_edge=2, edge_bp=42)
        d = generate(p)
        assert len(d.features) == n_edges
        assert len(d.bundles) == n_edges
        assert len(d.cylinders) == n_edges * 2
        assert d.unpaired_ends == []


def test_generation_is_deterministic():
    for seed in (0, 5, 17, 33):
        a = design_to_dict(generate(sample_params(seed)))
        b = design_to_dict(generate(sample_params(seed)))
        assert a == b


def test_json_roundtrip(tmp_path=None):
    out = Path(tmp_path) if tmp_path else Path(__file__).parent / "_tmp"
    out.mkdir(parents=True, exist_ok=True)
    for seed in (0, 3, 11):
        d = generate(sample_params(seed))
        for name in ("rt.json", "rt.json.gz"):
            path = save_design(d, out / name)
            assert design_to_dict(load_design(path)) == design_to_dict(d)


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
