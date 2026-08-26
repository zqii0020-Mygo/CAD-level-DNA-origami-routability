"""Data model for a CAD-level DNA origami design.

The hierarchy mirrors the heterogeneous graph the surrogate will consume:

    Feature   CAD primitive (an edge of a wireframe, a block, a plate)
      |
    Bundle    a group of parallel helices sharing one axis + lattice frame
      |
    Cylinder  one double helix, a bp interval along its bundle axis

plus the two relation types that make routing possible or impossible:

    CandidateCrossover  where two helices *may* be linked
    Mate                where two helix ends *must* be linked (CAD topology)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Vec3 = tuple[float, float, float]


@dataclass
class Feature:
    """A CAD-level primitive.  kind: 'edge' | 'block' | 'plate'."""
    id: int
    kind: str
    p0: Vec3
    p1: Vec3
    bundle_ids: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def length_nm(self) -> float:
        return sum((b - a) ** 2 for a, b in zip(self.p0, self.p1)) ** 0.5


@dataclass
class Bundle:
    """A set of parallel helices packed on `lattice`, sharing one axis frame."""
    id: int
    feature_id: int
    lattice: str
    origin: Vec3                 # 3D position of lattice site (0, 0) at bp 0
    axis: Vec3                   # unit vector, direction of increasing bp index
    frame_u: Vec3                # unit vector, cross-section +x
    frame_v: Vec3                # unit vector, cross-section +y
    length_bp: int
    cross_section: list[tuple[int, int]] = field(default_factory=list)
    cylinder_ids: list[int] = field(default_factory=list)


@dataclass
class Cylinder:
    """One double helix: lattice site + bp interval [bp_start, bp_start+bp_len)."""
    id: int
    bundle_id: int
    feature_id: int
    row: int
    col: int
    bp_start: int
    bp_len: int
    start_xyz: Vec3
    end_xyz: Vec3
    axis: Vec3

    @property
    def bp_end(self) -> int:
        return self.bp_start + self.bp_len


@dataclass
class Adjacency:
    """Two helices close enough to host crossovers.

    kind: 'intra' (same bundle, lattice neighbours)
          'inter' (different bundles, parallel and in contact)
    """
    id: int
    cyl_a: int
    cyl_b: int
    kind: str
    dist_nm: float
    overlap_bp: int
    dir_deg: float | None = None   # cross-section direction a -> b, intra only


@dataclass
class CandidateCrossover:
    """A geometrically allowed crossover site between two helices."""
    id: int
    cyl_a: int
    cyl_b: int
    adjacency_id: int
    bp_index: int                  # bundle-local bp coordinate
    xyz: Vec3
    dist_to_end_bp: int            # min distance to any of the four helix ends


@dataclass
class Mate:
    """A CAD-imposed end-to-end connection between two helices.

    Produced where features meet (wireframe vertices, concatenated blocks).
    end_a / end_b are 'start' or 'end'.
    """
    id: int
    cyl_a: int
    end_a: str
    cyl_b: int
    end_b: str
    kind: str                      # 'vertex' | 'concat'
    gap_nm: float
    vertex_id: int | None = None


@dataclass
class Design:
    id: str
    params: dict[str, Any]
    lattice: str
    features: list[Feature] = field(default_factory=list)
    bundles: list[Bundle] = field(default_factory=list)
    cylinders: list[Cylinder] = field(default_factory=list)
    adjacencies: list[Adjacency] = field(default_factory=list)
    crossovers: list[CandidateCrossover] = field(default_factory=list)
    mates: list[Mate] = field(default_factory=list)
    unpaired_ends: list[tuple[int, str]] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- convenience
    def cylinder(self, cid: int) -> Cylinder:
        return self.cylinders[cid]

    @property
    def total_bp(self) -> int:
        return sum(c.bp_len for c in self.cylinders)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "shape": self.params.get("shape"),
            "lattice": self.lattice,
            "n_features": len(self.features),
            "n_bundles": len(self.bundles),
            "n_cylinders": len(self.cylinders),
            "n_adjacency": len(self.adjacencies),
            "n_adjacency_inter": sum(a.kind == "inter" for a in self.adjacencies),
            "n_crossovers": len(self.crossovers),
            "n_mates": len(self.mates),
            "n_unpaired_ends": len(self.unpaired_ends),
            "total_bp": self.total_bp,
            "scaffold_len": self.params.get("scaffold_len"),
        }
