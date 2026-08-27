"""Design -> heterogeneous graph: the surrogate's input tensor bundle.

The four tables of `model.py` become four node types, and the relations that
decide routability become the edges:

    feature  --contains-->  bundle  --contains-->  cylinder  <--on--  crossover
                                       cylinder  --adjacent-->  cylinder
                                       cylinder  --mate-->      cylinder

Three edge families, mirroring the three physical relations:

    adjacency    two helices are in contact (`cylinder --adjacent-- cylinder`)
                 plus the candidate crossover sites that realise that contact
                 (`crossover --on-- cylinder`, one node per site, carrying the
                 bp index and therefore the *port* it lands on)
    mate         a CAD vertex joins two helix ends, possibly lo-hi
    containment  the CAD hierarchy, feature > bundle > cylinder

Every asymmetric relation is stored together with its reverse, and the
symmetric ones (`adjacent`, `mate`) are stored in both directions, so message
passing works without a `ToUndirected` transform.

Feature blocks and leakage
--------------------------
Graph-level features come in two blocks, kept apart on purpose:

    graph_x     pure CAD/counting descriptors -- sizes, densities, bp budget
    precheck_x  the *exact obstruction detectors* of `precheck.py`: dead ports,
                component count, articulation points, bridges, minimum degree,
                and the parity of the flipping links

`precheck_x` decides every fatal precheck class on its own, so handing it to a
model makes those labels free.  It is exported because it is the baseline to
beat, and because the ablation (`include_precheck=False`) is the experiment
that shows whether the GNN learned anything past the rules.  Per-cylinder port
degrees stay in the ordinary node block: they are one-hop derivable from the
crossover nodes anyway, so hiding them would buy nothing.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .lattice import get_lattice
from .linkgraph import LinkGraph, build_link_graph, port_of
from .model import Design
from .params import LATTICES, SHAPES
from .precheck import FAILURE_CLASSES, link_graph_stats

Relation = tuple[str, str, str]

NODE_TYPES = ("feature", "bundle", "cylinder", "crossover")

R_ADJACENT = ("cylinder", "adjacent", "cylinder")
R_MATE = ("cylinder", "mate", "cylinder")
R_XO_ON = ("crossover", "on", "cylinder")
R_XO_HOSTS = ("cylinder", "hosts", "crossover")
R_BUNDLE_CYL = ("bundle", "contains", "cylinder")
R_CYL_BUNDLE = ("cylinder", "in", "bundle")
R_FEAT_BUNDLE = ("feature", "contains", "bundle")
R_BUNDLE_FEAT = ("bundle", "in", "feature")

RELATIONS: tuple[Relation, ...] = (
    R_ADJACENT, R_MATE, R_XO_ON, R_XO_HOSTS,
    R_BUNDLE_CYL, R_CYL_BUNDLE, R_FEAT_BUNDLE, R_BUNDLE_FEAT,
)

EDGE_FAMILY: dict[Relation, str] = {
    R_ADJACENT: "adjacency", R_XO_ON: "adjacency", R_XO_HOSTS: "adjacency",
    R_MATE: "mate",
    R_BUNDLE_CYL: "containment", R_CYL_BUNDLE: "containment",
    R_FEAT_BUNDLE: "containment", R_BUNDLE_FEAT: "containment",
}

FEATURE_KINDS = ("edge", "brick", "plate")   # the kinds `generator.py` emits
ADJ_KINDS = ("intra", "inter")
MATE_KINDS = ("vertex", "concat")

# Graph-level descriptors, split as described in the module docstring.
# Exact duplicates are left out rather than carried: `n_features` equals
# `n_bundles` (every shape family builds one bundle per feature), `n_links_mate`
# equals `n_mates` (mates never collapse into a shared link), and `port_deg_mean`
# is `link_density` by definition (2L / 2n).  The one-hot blocks below are
# rank-deficient in the usual way; that is left alone.
GRAPH_STAT_FIELDS = (
    "n_bundles", "n_cylinders", "n_candidate_crossovers",
    "n_mates", "n_links", "n_links_crossover", "link_density",
    "total_bp", "scaffold_len", "bp_over_scaffold",
    "port_deg_max", "cyl_deg_mean", "cyl_deg_max",
)
PRECHECK_FIELDS = (
    "n_dead_ports", "n_unpaired_ends", "n_components", "n_articulation",
    "n_bridges", "port_deg_min", "cyl_deg_min",
    "n_links_flip", "n_links_noflip", "parity_blocked",
)

TARGET_FIELDS = (
    "routable", "hamilton", "hamilton_path", "failure_class_id",
    "nodes_expanded", "log_nodes_expanded", "backtracks", "elapsed_s",
    "timeout", "searched", "precheck_ok",
    "staple_ok", "export_ok", "scaffold_ok", "max_staple_span_bp",
    "path_nodes_expanded", "path_timeout",
)


def _onehot(value: Any, vocab: Sequence[str]) -> list[float]:
    v = str(value)
    return [1.0 if v == k else 0.0 for k in vocab]


def _unit(v: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(float(x) ** 2 for x in v))
    if n < 1e-12:
        return [0.0, 0.0, 0.0]
    return [float(x) / n for x in v]


# --------------------------------------------------------------- the container
@dataclass
class HeteroGraph:
    """Framework-free hetero graph: numpy arrays plus the names of their columns."""

    design_id: str
    x: dict[str, np.ndarray] = field(default_factory=dict)
    node_fields: dict[str, list[str]] = field(default_factory=dict)
    edge_index: dict[Relation, np.ndarray] = field(default_factory=dict)
    edge_attr: dict[Relation, np.ndarray] = field(default_factory=dict)
    edge_fields: dict[Relation, list[str]] = field(default_factory=dict)
    graph_x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    graph_fields: list[str] = field(default_factory=list)
    precheck_x: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    precheck_fields: list[str] = field(default_factory=list)
    y: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def num_nodes(self, ntype: str) -> int:
        return int(self.x[ntype].shape[0]) if ntype in self.x else 0

    def num_edges(self, rel: Relation | None = None) -> int:
        if rel is not None:
            return int(self.edge_index[rel].shape[1]) if rel in self.edge_index else 0
        return sum(int(e.shape[1]) for e in self.edge_index.values())

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.design_id,
            "nodes": {t: self.num_nodes(t) for t in NODE_TYPES},
            "edges": {"/".join(r): self.num_edges(r) for r in self.edge_index},
            "node_dims": {t: int(a.shape[1]) for t, a in self.x.items()},
            "graph_dim": int(self.graph_x.size),
            "targets": dict(self.y),
        }

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        """Index bounds and column counts -- cheap, and it catches builder slips."""
        for t, a in self.x.items():
            assert a.ndim == 2, f"{t}.x is {a.ndim}-d"
            assert a.shape[1] == len(self.node_fields[t]), f"{t}: field/column mismatch"
            assert np.isfinite(a).all(), f"{t}.x has non-finite values"
        assert self.graph_x.size == len(self.graph_fields), "graph_x: field/column mismatch"
        assert self.precheck_x.size == len(self.precheck_fields), "precheck_x: field mismatch"
        assert np.isfinite(self.graph_x).all() and np.isfinite(self.precheck_x).all(), \
            "graph-level features have non-finite values"
        for rel, ei in self.edge_index.items():
            src, _, dst = rel
            assert ei.shape[0] == 2, f"{rel}: edge_index is not 2 x E"
            if ei.size:
                assert ei[0].max() < self.num_nodes(src), f"{rel}: src index out of range"
                assert ei[1].max() < self.num_nodes(dst), f"{rel}: dst index out of range"
                assert ei.min() >= 0, f"{rel}: negative index"
            if rel in self.edge_attr:
                ea = self.edge_attr[rel]
                assert ea.shape[0] == ei.shape[1], f"{rel}: attr/edge count mismatch"
                assert ea.shape[1] == len(self.edge_fields[rel]), f"{rel}: attr field mismatch"

    # ---------------------------------------------------------------- interop
    def to_pyg(self, include_precheck: bool = True):
        """Convert to a `torch_geometric.data.HeteroData` (PyG imported lazily)."""
        try:
            import torch
            from torch_geometric.data import HeteroData
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "to_pyg() needs torch_geometric: pip install torch_geometric"
            ) from exc

        data = HeteroData()
        data.design_id = self.design_id
        for t in NODE_TYPES:
            arr = self.x.get(t, np.zeros((0, len(self.node_fields.get(t, []))), np.float32))
            data[t].x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))
            data[t].num_nodes = int(arr.shape[0])
        for rel in RELATIONS:
            ei = self.edge_index.get(rel, np.zeros((2, 0), np.int64))
            data[rel].edge_index = torch.from_numpy(np.ascontiguousarray(ei, dtype=np.int64))
            if rel in self.edge_attr:
                data[rel].edge_attr = torch.from_numpy(
                    np.ascontiguousarray(self.edge_attr[rel], dtype=np.float32)
                )
        g = np.concatenate([self.graph_x, self.precheck_x]) if include_precheck else self.graph_x
        data.graph_x = torch.from_numpy(np.ascontiguousarray(g, dtype=np.float32)).view(1, -1)
        for k, v in self.y.items():
            setattr(data, f"y_{k}", torch.tensor([float(v)], dtype=torch.float32))
        return data

    # ------------------------------------------------------------------- disk
    def save(self, path: str | Path) -> Path:
        """One .npz per graph: arrays raw, everything else as one JSON blob."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for t, a in self.x.items():
            arrays[f"x::{t}"] = a
        for rel, ei in self.edge_index.items():
            arrays["ei::" + "|".join(rel)] = ei
        for rel, ea in self.edge_attr.items():
            arrays["ea::" + "|".join(rel)] = ea
        arrays["graph_x"] = self.graph_x
        arrays["precheck_x"] = self.precheck_x
        blob = json.dumps({
            "design_id": self.design_id,
            "node_fields": self.node_fields,
            "edge_fields": {"|".join(r): f for r, f in self.edge_fields.items()},
            "graph_fields": self.graph_fields,
            "precheck_fields": self.precheck_fields,
            "y": self.y,
            "meta": self.meta,
        })
        np.savez_compressed(path, meta_json=np.array(blob), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "HeteroGraph":
        with np.load(Path(path), allow_pickle=False) as z:
            o = json.loads(str(z["meta_json"]))
            hg = cls(
                design_id=o["design_id"],
                node_fields={k: list(v) for k, v in o["node_fields"].items()},
                edge_fields={tuple(k.split("|")): list(v) for k, v in o["edge_fields"].items()},
                graph_fields=list(o["graph_fields"]),
                precheck_fields=list(o["precheck_fields"]),
                y={k: float(v) for k, v in o["y"].items()},
                meta=o["meta"],
                graph_x=z["graph_x"],
                precheck_x=z["precheck_x"],
            )
            for key in z.files:
                if key.startswith("x::"):
                    hg.x[key[3:]] = z[key]
                elif key.startswith("ei::"):
                    hg.edge_index[tuple(key[4:].split("|"))] = z[key]
                elif key.startswith("ea::"):
                    hg.edge_attr[tuple(key[4:].split("|"))] = z[key]
        return hg


# ----------------------------------------------------------------- the builder
def _coord_frame(design: Design) -> tuple[np.ndarray, float]:
    """Centre and isotropic scale that put every helix endpoint inside [-1, 1]."""
    pts = [p for c in design.cylinders for p in (c.start_xyz, c.end_xyz)]
    if not pts:
        return np.zeros(3, dtype=np.float64), 1.0
    a = np.asarray(pts, dtype=np.float64)
    centre = (a.max(axis=0) + a.min(axis=0)) / 2.0
    scale = float(np.abs(a - centre).max())
    return centre, (scale if scale > 1e-9 else 1.0)


def _rows(fields: Sequence[str], rows: list[list[float]]) -> np.ndarray:
    for r in rows:
        assert len(r) == len(fields), f"row has {len(r)} values for {len(fields)} fields"
    if not rows:
        return np.zeros((0, len(fields)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def build_graph(
    design: Design,
    label: Any | None = None,
    lg: LinkGraph | None = None,
) -> HeteroGraph:
    """Design (+ an optional `DesignLabel` or labels.csv row) -> `HeteroGraph`."""
    lg = lg if lg is not None else build_link_graph(design)
    stats = link_graph_stats(design, lg)
    lat = get_lattice(design.lattice)
    step = float(lat.step)
    centre, scale = _coord_frame(design)

    cyl = {c.id: c for c in design.cylinders}
    bundle = {b.id: b for b in design.bundles}
    adj_by_id = {a.id: a for a in design.adjacencies}
    cyl_row = {c.id: i for i, c in enumerate(design.cylinders)}
    bun_row = {b.id: i for i, b in enumerate(design.bundles)}
    feat_row = {f.id: i for i, f in enumerate(design.features)}

    def nxyz(p: Sequence[float]) -> list[float]:
        return [float(v) for v in (np.asarray(p, dtype=np.float64) - centre) / scale]

    # ---- per-cylinder tallies shared by several node types
    n_adj_intra: dict[int, int] = defaultdict(int)
    n_adj_inter: dict[int, int] = defaultdict(int)
    for a in design.adjacencies:
        tgt = n_adj_intra if a.kind == "intra" else n_adj_inter
        tgt[a.cyl_a] += 1
        tgt[a.cyl_b] += 1

    n_xo_cyl: dict[int, int] = defaultdict(int)
    n_xo_bundle: dict[int, int] = defaultdict(int)
    xo_by_adj: dict[int, list] = defaultdict(list)
    xo_ports: dict[int, tuple[str, str]] = {}
    for x in design.crossovers:
        n_xo_cyl[x.cyl_a] += 1
        n_xo_cyl[x.cyl_b] += 1
        n_xo_bundle[cyl[x.cyl_a].bundle_id] += 1
        xo_by_adj[x.adjacency_id].append(x)
        xo_ports[x.id] = (port_of(cyl[x.cyl_a], x.bp_index), port_of(cyl[x.cyl_b], x.bp_index))

    n_mate_cyl: dict[int, int] = defaultdict(int)
    for m in design.mates:
        n_mate_cyl[m.cyl_a] += 1
        n_mate_cyl[m.cyl_b] += 1

    n_flip_cyl: dict[int, int] = defaultdict(int)
    n_noflip_cyl: dict[int, int] = defaultdict(int)
    for lk in lg.links:
        tgt = n_flip_cyl if lk.flip else n_noflip_cyl
        tgt[lk.cyl_a] += 1
        tgt[lk.cyl_b] += 1
    cyl_degree = dict(lg.cylinder_graph().degree())
    unpaired = {(int(c), str(e)) for c, e in design.unpaired_ends}

    n_cyl_bundle: dict[int, int] = defaultdict(int)
    n_cyl_feature: dict[int, int] = defaultdict(int)
    for c in design.cylinders:
        n_cyl_bundle[c.bundle_id] += 1
        n_cyl_feature[c.feature_id] += 1

    hg = HeteroGraph(design_id=design.id)

    # ---------------------------------------------------------------- features
    f_fields = [f"kind_{k}" for k in FEATURE_KINDS] + [
        "length_nm", "n_bundles", "n_cylinders",
        "cx", "cy", "cz", "dx", "dy", "dz",
        "meta_n_rows", "meta_n_cols", "meta_fill",
    ]
    f_rows = []
    for f in design.features:
        p0 = np.asarray(f.p0, dtype=np.float64)
        p1 = np.asarray(f.p1, dtype=np.float64)
        f_rows.append(
            _onehot(f.kind, FEATURE_KINDS)
            + [f.length_nm, float(len(f.bundle_ids)), float(n_cyl_feature[f.id])]
            + nxyz((p0 + p1) / 2.0)
            + _unit(p1 - p0)
            + [
                float(f.meta.get("n_rows", 0) or 0),
                float(f.meta.get("n_cols", 0) or 0),
                float(f.meta.get("fill", 0.0) or 0.0),
            ]
        )
    hg.x["feature"] = _rows(f_fields, f_rows)
    hg.node_fields["feature"] = f_fields

    # ----------------------------------------------------------------- bundles
    b_fields = [f"lattice_{k}" for k in LATTICES] + [
        "length_bp", "length_turns", "n_cylinders",
        "rows_extent", "cols_extent",
        "ax", "ay", "az", "ox", "oy", "oz",
        "n_crossovers", "mean_cyl_degree",
    ]
    b_rows = []
    for b in design.bundles:
        sites = list(b.cross_section) or [(0, 0)]
        rows = [s[0] for s in sites]
        cols = [s[1] for s in sites]
        nc = n_cyl_bundle[b.id]
        degs = [cyl_degree.get(c, 0) for c in b.cylinder_ids] or [0]
        b_rows.append(
            _onehot(b.lattice, LATTICES)
            + [
                float(b.length_bp), b.length_bp / step, float(nc),
                float(max(rows) - min(rows) + 1), float(max(cols) - min(cols) + 1),
            ]
            + _unit(b.axis)
            + nxyz(b.origin)
            + [float(n_xo_bundle[b.id]), float(np.mean(degs))]
        )
    hg.x["bundle"] = _rows(b_fields, b_rows)
    hg.node_fields["bundle"] = b_fields

    # --------------------------------------------------------------- cylinders
    c_fields = [
        "bp_len", "bp_len_turns", "bp_start", "bp_start_turns",
        "frac_of_bundle", "inset_hi_bp",
        "row", "col", "row_rel", "col_rel",
        "cx", "cy", "cz", "ax", "ay", "az",
        "n_adjacent_intra", "n_adjacent_inter", "n_crossovers", "n_mates",
        "port_deg_lo", "port_deg_hi", "cyl_degree",
        "n_links_flip", "n_links_noflip", "unpaired_lo", "unpaired_hi",
    ]
    c_rows = []
    for c in design.cylinders:
        b = bundle[c.bundle_id]
        sites = list(b.cross_section) or [(c.row, c.col)]
        row_c = float(np.mean([s[0] for s in sites]))
        col_c = float(np.mean([s[1] for s in sites]))
        p0 = np.asarray(c.start_xyz, dtype=np.float64)
        p1 = np.asarray(c.end_xyz, dtype=np.float64)
        c_rows.append([
            float(c.bp_len), c.bp_len / step, float(c.bp_start), c.bp_start / step,
            c.bp_len / max(1, b.length_bp),
            float(max(0, b.length_bp - c.bp_end)),
            float(c.row), float(c.col), c.row - row_c, c.col - col_c,
        ] + nxyz((p0 + p1) / 2.0) + _unit(c.axis) + [
            float(n_adj_intra[c.id]), float(n_adj_inter[c.id]),
            float(n_xo_cyl[c.id]), float(n_mate_cyl[c.id]),
            float(lg.port_degree(c.id, "lo")), float(lg.port_degree(c.id, "hi")),
            float(cyl_degree.get(c.id, 0)),
            float(n_flip_cyl[c.id]), float(n_noflip_cyl[c.id]),
            float((c.id, "start") in unpaired), float((c.id, "end") in unpaired),
        ])
    hg.x["cylinder"] = _rows(c_fields, c_rows)
    hg.node_fields["cylinder"] = c_fields

    # -------------------------------------------------------------- crossovers
    x_fields = [
        "bp_index", "bp_index_norm", "phase_frac", "dist_to_end_bp",
        "frac_along_a", "frac_along_b", "dist_end_a_bp", "dist_end_b_bp",
        "port_a_is_lo", "port_b_is_lo", "is_flip",
        "dist_nm", "overlap_bp", "dir_sin", "dir_cos",
    ] + [f"adj_kind_{k}" for k in ADJ_KINDS] + ["x", "y", "z"]
    x_rows: list[list[float]] = []
    xo_row: dict[int, int] = {}
    for x in design.crossovers:
        xo_row[x.id] = len(x_rows)
        ca, cb = cyl[x.cyl_a], cyl[x.cyl_b]
        pa, pb = xo_ports[x.id]
        a = adj_by_id.get(x.adjacency_id)
        ang = None if (a is None or a.dir_deg is None) else math.radians(a.dir_deg)
        da = x.bp_index - ca.bp_start if pa == "lo" else ca.bp_end - x.bp_index
        db = x.bp_index - cb.bp_start if pb == "lo" else cb.bp_end - x.bp_index
        bl = max(1, bundle[ca.bundle_id].length_bp)
        x_rows.append([
            float(x.bp_index), x.bp_index / bl, (x.bp_index % lat.step) / step,
            float(x.dist_to_end_bp),
            (x.bp_index - ca.bp_start) / max(1, ca.bp_len),
            (x.bp_index - cb.bp_start) / max(1, cb.bp_len),
            float(da), float(db),
            float(pa == "lo"), float(pb == "lo"), float(pa == pb),
            float(a.dist_nm) if a else 0.0, float(a.overlap_bp) if a else 0.0,
            0.0 if ang is None else math.sin(ang),
            0.0 if ang is None else math.cos(ang),
        ] + _onehot(a.kind if a else "intra", ADJ_KINDS) + nxyz(x.xyz))
    hg.x["crossover"] = _rows(x_fields, x_rows)
    hg.node_fields["crossover"] = x_fields

    # --------------------------------------------------------- adjacency edges
    a_fields = [f"kind_{k}" for k in ADJ_KINDS] + [
        "dist_nm", "overlap_bp", "dir_sin", "dir_cos",
        "n_crossovers", "has_lo_lo", "has_hi_hi", "has_lo_hi", "min_dist_to_end_bp",
    ]
    a_src: list[int] = []
    a_dst: list[int] = []
    a_rows: list[list[float]] = []
    for a in design.adjacencies:
        xs = xo_by_adj.get(a.id, [])
        ports = [xo_ports[x.id] for x in xs]
        ang = None if a.dir_deg is None else math.radians(a.dir_deg)
        row = _onehot(a.kind, ADJ_KINDS) + [
            float(a.dist_nm), float(a.overlap_bp),
            0.0 if ang is None else math.sin(ang),
            0.0 if ang is None else math.cos(ang),
            float(len(xs)),
            float(any(p == ("lo", "lo") for p in ports)),
            float(any(p == ("hi", "hi") for p in ports)),
            float(any(p[0] != p[1] for p in ports)),
            float(min((x.dist_to_end_bp for x in xs), default=0)),
        ]
        ia, ib = cyl_row[a.cyl_a], cyl_row[a.cyl_b]
        for s, d in ((ia, ib), (ib, ia)):      # symmetric relation, both ways
            a_src.append(s)
            a_dst.append(d)
            a_rows.append(row)
    hg.edge_index[R_ADJACENT] = np.asarray([a_src, a_dst], dtype=np.int64).reshape(2, -1)
    hg.edge_attr[R_ADJACENT] = _rows(a_fields, a_rows)
    hg.edge_fields[R_ADJACENT] = a_fields

    # -------------------------------------------------------------- mate edges
    m_fields = [f"kind_{k}" for k in MATE_KINDS] + [
        "gap_nm", "port_src_is_lo", "port_dst_is_lo", "is_flip",
    ]
    m_src: list[int] = []
    m_dst: list[int] = []
    m_rows: list[list[float]] = []
    for m in design.mates:
        pa = "lo" if m.end_a == "start" else "hi"
        pb = "lo" if m.end_b == "start" else "hi"
        base = _onehot(m.kind, MATE_KINDS)
        ia, ib = cyl_row[m.cyl_a], cyl_row[m.cyl_b]
        for s, d, ps, pd in ((ia, ib, pa, pb), (ib, ia, pb, pa)):
            m_src.append(s)
            m_dst.append(d)
            m_rows.append(base + [float(m.gap_nm), float(ps == "lo"), float(pd == "lo"),
                                  float(pa == pb)])
    hg.edge_index[R_MATE] = np.asarray([m_src, m_dst], dtype=np.int64).reshape(2, -1)
    hg.edge_attr[R_MATE] = _rows(m_fields, m_rows)
    hg.edge_fields[R_MATE] = m_fields

    # ----------------------------------------------------- crossover incidence
    xo_fields = ["is_lo", "is_side_a", "dist_to_end_bp", "frac_along"]
    xo_src: list[int] = []
    xo_dst: list[int] = []
    xo_rows: list[list[float]] = []
    for x in design.crossovers:
        pa, pb = xo_ports[x.id]
        for cid, port, side in ((x.cyl_a, pa, 1.0), (x.cyl_b, pb, 0.0)):
            c = cyl[cid]
            d_end = x.bp_index - c.bp_start if port == "lo" else c.bp_end - x.bp_index
            xo_src.append(xo_row[x.id])
            xo_dst.append(cyl_row[cid])
            xo_rows.append([float(port == "lo"), side, float(d_end),
                            (x.bp_index - c.bp_start) / max(1, c.bp_len)])
    ei = np.asarray([xo_src, xo_dst], dtype=np.int64).reshape(2, -1)
    attr = _rows(xo_fields, xo_rows)
    hg.edge_index[R_XO_ON] = ei
    hg.edge_attr[R_XO_ON] = attr
    hg.edge_fields[R_XO_ON] = xo_fields
    hg.edge_index[R_XO_HOSTS] = ei[::-1].copy()
    hg.edge_attr[R_XO_HOSTS] = attr.copy()
    hg.edge_fields[R_XO_HOSTS] = xo_fields

    # ------------------------------------------------------- containment edges
    bc = np.asarray(
        [[bun_row[c.bundle_id] for c in design.cylinders],
         [cyl_row[c.id] for c in design.cylinders]], dtype=np.int64
    ).reshape(2, -1)
    hg.edge_index[R_BUNDLE_CYL] = bc
    hg.edge_index[R_CYL_BUNDLE] = bc[::-1].copy()
    fb = np.asarray(
        [[feat_row[b.feature_id] for b in design.bundles],
         [bun_row[b.id] for b in design.bundles]], dtype=np.int64
    ).reshape(2, -1)
    hg.edge_index[R_FEAT_BUNDLE] = fb
    hg.edge_index[R_BUNDLE_FEAT] = fb[::-1].copy()

    # ------------------------------------------------------ graph-level blocks
    n = int(stats["n_cylinders"])
    parity_blocked = float(stats["n_links_noflip"] == 0 and n % 2 == 1)
    g_fields = list(GRAPH_STAT_FIELDS) + ["bp_slack", "log_scaffold_len"]
    g_fields += [f"shape_{s}" for s in SHAPES] + [f"lattice_{s}" for s in LATTICES]
    g_row = [float(stats[k]) for k in GRAPH_STAT_FIELDS]
    g_row += [
        float(stats["scaffold_len"] - stats["total_bp"]),
        math.log1p(max(0.0, float(stats["scaffold_len"]))),
    ]
    g_row += _onehot(design.params.get("shape"), SHAPES)
    g_row += _onehot(design.lattice, LATTICES)
    hg.graph_x = np.asarray(g_row, dtype=np.float32)
    hg.graph_fields = g_fields

    hg.precheck_x = np.asarray(
        [parity_blocked if k == "parity_blocked" else float(stats[k]) for k in PRECHECK_FIELDS],
        dtype=np.float32,
    )
    hg.precheck_fields = list(PRECHECK_FIELDS)

    hg.meta = {
        "shape": str(design.params.get("shape", "?")),
        "lattice": design.lattice,
        # "iid" or "rare:<class>" -- how this design was drawn (see gen_designs.py)
        "sampling": str(design.notes.get("sampling", "iid")),
        "coord_centre": [float(v) for v in centre],
        "coord_scale": float(scale),
        "lattice_step_bp": int(lat.step),
        "edge_family": {"|".join(r): f for r, f in EDGE_FAMILY.items()},
    }
    if label is not None:
        hg.y = targets_from_label(label)
    hg.validate()
    return hg


# --------------------------------------------------------------------- targets
def _to_float(v: Any) -> float:
    """CSV strings, python bools and None -> float.  Unknown becomes NaN."""
    if v is None or v == "" or v == "None":
        return float("nan")
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s in ("True", "true"):
            return 1.0
        if s in ("False", "false"):
            return 0.0
        try:
            return float(s)
        except ValueError:
            return float("nan")
    return float(v)


def targets_from_label(label: Any) -> dict[str, float]:
    """`DesignLabel` or a labels.csv row -> the target vector.

    `hamilton` / `hamilton_path` are NaN when the search timed out, and
    `timeout` marks the search-cost targets as right-censored: those designs hit
    the budget, so `nodes_expanded` is a lower bound, not the true cost.
    """
    get = label.get if isinstance(label, dict) else (lambda k, d=None: getattr(label, k, d))
    fc = get("failure_class", None)
    fc = None if fc in ("", "None") else fc
    nodes = _to_float(get("nodes_expanded", 0))
    y = {
        "routable": _to_float(get("routable", None)),
        "hamilton": _to_float(get("hamilton", None)),
        "hamilton_path": _to_float(get("hamilton_path", None)),
        "failure_class_id": float(FAILURE_CLASSES.index(fc)) if fc in FAILURE_CLASSES else -1.0,
        "nodes_expanded": nodes,
        "log_nodes_expanded": math.log1p(nodes) if nodes == nodes else float("nan"),
        "backtracks": _to_float(get("backtracks", 0)),
        "elapsed_s": _to_float(get("elapsed_s", 0.0)),
        "timeout": _to_float(get("timeout", False)),
        "searched": _to_float(get("searched", False)),
        "precheck_ok": _to_float(get("precheck_ok", False)),
        "staple_ok": _to_float(get("staple_ok", None)),
        "export_ok": _to_float(get("export_ok", None)),
        "scaffold_ok": _to_float(get("scaffold_ok", None)),
        "max_staple_span_bp": _to_float(get("max_staple_span_bp", 0)),
        "path_nodes_expanded": _to_float(get("path_nodes_expanded", 0)),
        "path_timeout": _to_float(get("path_timeout", False)),
    }
    return {k: y[k] for k in TARGET_FIELDS}
