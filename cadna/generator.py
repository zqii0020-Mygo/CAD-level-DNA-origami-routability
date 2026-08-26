"""CAD generator: CADParams -> Design.

Four shape families cover the routing regimes we care about:

  brick        multi-layer block, one bundle, crossover-rich, no mates
  plate        1-2 layer sheet, long and thin, sparse crossover topology
  polygon_ring planar N-gon wireframe, one bundle per edge, mates at vertices
  polyhedron   3D wireframe (tetra / octa / cube), the DAEDALUS/ATHENA regime

Everything downstream (adjacency, crossovers, routing, graph export) is shape
agnostic and only sees Features / Bundles / Cylinders / Mates.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import deque

import networkx as nx
import numpy as np

from .adjacency import build_adjacency, build_crossovers
from .lattice import RISE_PER_BP, Lattice, Site, get_lattice
from .model import Bundle, Cylinder, Design, Feature, Mate
from .params import CADParams


# ------------------------------------------------------------------ geometry
def _frame(axis) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal frame (axis, u, v) with a deterministic choice of u."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, a)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return a, u, v


def _vec(a) -> tuple[float, float, float]:
    return (float(a[0]), float(a[1]), float(a[2]))


# ------------------------------------------------------------ cross sections
def _rect_cross_section(n_rows: int, n_cols: int) -> list[Site]:
    return [(r, c) for r in range(n_rows) for c in range(n_cols)]


def _largest_patch(lat: Lattice, sites: set[Site]) -> list[Site]:
    """The biggest lattice-connected component of a set of sites."""
    remaining = set(sites)
    best: list[Site] = []
    while remaining:
        seed = min(remaining)
        comp, q = {seed}, deque([seed])
        remaining.discard(seed)
        while q:
            cur = q.popleft()
            for nb in lat.neighbors(*cur):
                if nb in remaining:
                    remaining.discard(nb)
                    comp.add(nb)
                    q.append(nb)
        if len(comp) > len(best):
            best = sorted(comp)
    return best


def _repair_cross_section(lat: Lattice, sites: list[Site]) -> list[Site]:
    """Drop dangling helices, then keep the largest connected patch.

    A helix with fewer than two in-bundle neighbours cannot have a scaffold
    cycle pass through it, and a CAD tool would not let you place one.  This
    matters most in the honeycomb lattice, where the third neighbour alternates
    up/down with parity, so a plain rectangular (row, col) block grows dangling
    corners.  Only applied to bundles that have no mates -- a wireframe edge
    gets its second and third connections from its vertices instead.
    """
    cur = set(sites)
    while len(cur) > 2:
        dangling = {s for s in cur if sum(nb in cur for nb in lat.neighbors(*s)) < 2}
        if not dangling:
            break
        cur -= dangling
    patch = _largest_patch(lat, cur) if len(cur) >= 2 else []
    return patch if len(patch) >= 2 else _largest_patch(lat, set(sites))


def _compact_cross_section(lat: Lattice, k: int) -> list[Site]:
    """k lattice sites grown breadth-first from the origin (compact packing)."""
    seen = {(0, 0)}
    order: list[Site] = [(0, 0)]
    q = deque([(0, 0)])
    while len(order) < k and q:
        cur = q.popleft()
        for nb in sorted(lat.neighbors(*cur)):
            if nb in seen:
                continue
            seen.add(nb)
            order.append(nb)
            q.append(nb)
            if len(order) >= k:
                break
    return order[:k]


# ------------------------------------------------------------------- builder
class _Builder:
    def __init__(self, params: CADParams, lat: Lattice):
        self.p = params
        self.lat = lat
        self.rng = random.Random(params.seed * 7919 + 13)
        self.features: list[Feature] = []
        self.bundles: list[Bundle] = []
        self.cylinders: list[Cylinder] = []
        # vertex_id -> list of (cyl_id, "start" | "end")
        self.vertex_ends: dict[int, list[tuple[int, str]]] = {}

    def add_feature(self, kind: str, p0, p1, **meta) -> Feature:
        f = Feature(id=len(self.features), kind=kind, p0=_vec(p0), p1=_vec(p1), meta=meta)
        self.features.append(f)
        return f

    def add_bundle(
        self,
        feature: Feature,
        p0,
        axis,
        length_bp: int,
        cross_section: list[Site],
        v0: int | None = None,
        v1: int | None = None,
    ) -> Bundle:
        """Lay the cross-section helices along axis, with site (0,0) at bp 0 == p0."""
        a, u, v = _frame(axis)
        p0 = np.asarray(p0, dtype=float)
        b = Bundle(
            id=len(self.bundles),
            feature_id=feature.id,
            lattice=self.lat.name,
            origin=_vec(p0),
            axis=_vec(a),
            frame_u=_vec(u),
            frame_v=_vec(v),
            length_bp=length_bp,
            cross_section=[tuple(s) for s in cross_section],
        )
        stag = self.p.stagger_bp
        for (row, col) in cross_section:
            bp_start, bp_len = 0, length_bp
            if stag > 0:
                lo = self.rng.randint(0, stag)
                hi = self.rng.randint(0, stag)
                if length_bp - lo - hi >= 8:
                    bp_start, bp_len = lo, length_bp - lo - hi
            x, y = self.lat.position(row, col)
            base = p0 + x * u + y * v
            start = base + a * (bp_start * RISE_PER_BP)
            end = base + a * ((bp_start + bp_len) * RISE_PER_BP)
            c = Cylinder(
                id=len(self.cylinders),
                bundle_id=b.id,
                feature_id=feature.id,
                row=row,
                col=col,
                bp_start=bp_start,
                bp_len=bp_len,
                start_xyz=_vec(start),
                end_xyz=_vec(end),
                axis=_vec(a),
            )
            self.cylinders.append(c)
            b.cylinder_ids.append(c.id)
            if v0 is not None:
                self.vertex_ends.setdefault(v0, []).append((c.id, "start"))
            if v1 is not None:
                self.vertex_ends.setdefault(v1, []).append((c.id, "end"))
        self.bundles.append(b)
        feature.bundle_ids.append(b.id)
        return b

    # --------------------------------------------------------------- mating
    def build_mates(self) -> tuple[list[Mate], list[tuple[int, str]]]:
        """Pair the helix ends meeting at each vertex.

        A bundle must not fold back onto itself at a vertex, so the admissible
        pairs form a multipartite graph over the incident bundles.  Greedy
        nearest-first pairing gets stuck there (at a degree-3 vertex it can
        exhaust two bundles against each other and strand the third), so we take
        a maximum-cardinality matching and break ties towards short mates.
        Whatever it cannot match is a genuine CAD-level defect and is reported
        as an unpaired end.
        """
        mates: list[Mate] = []
        unpaired: list[tuple[int, str]] = []
        for vid, raw_ends in sorted(self.vertex_ends.items()):
            ends = sorted(set(raw_ends))
            pos, bundle_of = {}, {}
            for e in ends:
                c = self.cylinders[e[0]]
                pos[e] = np.asarray(c.start_xyz if e[1] == "start" else c.end_xyz)
                bundle_of[e] = c.bundle_id

            dist: dict[tuple, float] = {}
            for e1, e2 in itertools.combinations(ends, 2):
                if bundle_of[e1] == bundle_of[e2]:
                    continue
                dist[(e1, e2)] = float(np.linalg.norm(pos[e1] - pos[e2]))

            matched: set[tuple] = set()
            if dist:
                big = max(dist.values()) + 1.0
                g = nx.Graph()
                g.add_nodes_from(ends)
                for (e1, e2), d in sorted(dist.items()):
                    g.add_edge(e1, e2, weight=big - d)
                matched = {
                    tuple(sorted(pair))
                    for pair in nx.max_weight_matching(g, maxcardinality=True)
                }

            used: set[tuple[int, str]] = set()
            for e1, e2 in sorted(matched):
                used.add(e1)
                used.add(e2)
                mates.append(
                    Mate(
                        id=len(mates),
                        cyl_a=e1[0], end_a=e1[1],
                        cyl_b=e2[0], end_b=e2[1],
                        kind="vertex",
                        gap_nm=dist.get((e1, e2), dist.get((e2, e1), 0.0)),
                        vertex_id=vid,
                    )
                )
            unpaired.extend(e for e in ends if e not in used)
        return mates, unpaired


# -------------------------------------------------------------- shape makers
def _polyhedron_verts_edges(name: str):
    if name == "tetrahedron":
        verts = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        edges = list(itertools.combinations(range(4), 2))
    elif name == "octahedron":
        verts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        edges = [
            (i, j) for i, j in itertools.combinations(range(6), 2)
            if not np.allclose(np.array(verts[i]) + np.array(verts[j]), 0)
        ]
    elif name == "cube":
        verts = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        edges = [
            (i, j) for i, j in itertools.combinations(range(8), 2)
            if sum(a != b for a, b in zip(verts[i], verts[j])) == 1
        ]
    else:
        raise ValueError(f"unknown polyhedron {name!r}")
    return np.asarray(verts, dtype=float), edges


def _build_solid(bd: _Builder) -> None:
    p = bd.p
    sites = _rect_cross_section(p.n_rows, p.n_cols)
    if p.fill < 1.0:
        keep = [s for s in sites if bd.rng.random() < p.fill]
        sites = keep or sites[:1]
    if p.repair_cross_section and len(sites) > 2:
        # A single row of helices is a lattice path, so the repair would eat it
        # from both ends down to nothing.  Keep the repair only where it tidies
        # a cross-section rather than destroying it.
        fixed = _repair_cross_section(bd.lat, sites)
        if len(fixed) >= 3 and len(fixed) >= 0.6 * len(sites):
            sites = fixed
    axis = np.array([0.0, 0.0, 1.0])
    p0 = np.zeros(3)
    p1 = p0 + axis * (p.length_bp * RISE_PER_BP)
    f = bd.add_feature(p.shape, p0, p1, n_rows=p.n_rows, n_cols=p.n_cols, fill=p.fill)
    bd.add_bundle(f, p0, axis, p.length_bp, sites)


def _build_wireframe(bd: _Builder) -> None:
    p, lat = bd.p, bd.lat
    seg_nm = p.edge_bp * RISE_PER_BP
    edge_nm = seg_nm + 2.0 * p.vertex_inset_nm

    if p.shape == "polygon_ring":
        n = p.n_sides
        R = edge_nm / (2.0 * math.sin(math.pi / n))
        verts = np.array(
            [[R * math.cos(2 * math.pi * k / n), R * math.sin(2 * math.pi * k / n), 0.0]
             for k in range(n)]
        )
        edges = [(k, (k + 1) % n) for k in range(n)]
    else:
        verts, edges = _polyhedron_verts_edges(p.polyhedron)
        unit = float(np.linalg.norm(verts[edges[0][0]] - verts[edges[0][1]]))
        verts = verts * (edge_nm / unit)

    sites = _compact_cross_section(lat, p.helices_per_edge)
    for (i, j) in edges:
        a, b = verts[i], verts[j]
        d = b - a
        d = d / np.linalg.norm(d)
        p0 = a + d * p.vertex_inset_nm
        p1 = b - d * p.vertex_inset_nm
        f = bd.add_feature("edge", p0, p1, v0=int(i), v1=int(j))
        bd.add_bundle(f, p0, d, p.edge_bp, sites, v0=int(i), v1=int(j))


# ---------------------------------------------------------------------- API
def generate(params: CADParams) -> Design:
    """Turn one point of the CAD parameter space into a full design."""
    lat = get_lattice(params.lattice)
    bd = _Builder(params, lat)

    if params.shape in ("brick", "plate"):
        _build_solid(bd)
    elif params.shape in ("polygon_ring", "polyhedron"):
        _build_wireframe(bd)
    else:
        raise ValueError(f"unknown shape {params.shape!r}")

    mates, unpaired = bd.build_mates()

    tag = params.polyhedron if params.shape == "polyhedron" else params.shape
    design = Design(
        id=f"{params.shape}-{tag}-{params.lattice}-{params.seed:06d}",
        params=params.to_dict(),
        lattice=lat.name,
        features=bd.features,
        bundles=bd.bundles,
        cylinders=bd.cylinders,
        mates=mates,
        unpaired_ends=unpaired,
    )
    design.adjacencies = build_adjacency(design, lat)
    design.crossovers = build_crossovers(design, lat)
    return design


def generate_from_seed(seed: int, **kw) -> Design:
    from .params import sample_params

    return generate(sample_params(seed, **kw))
