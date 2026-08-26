"""CAD parameter space and its sampler.

`CADParams` is the *only* input to the generator: one point in this space is
one design.  The sampler is deliberately allowed to produce degenerate or
over-constrained designs (odd helix counts, punched-out cross-sections,
scaffold far too short) -- those are the negative labels the surrogate has to
learn to predict.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

SHAPES = ("brick", "plate", "polygon_ring", "polyhedron")
LATTICES = ("honeycomb", "square")
POLYHEDRA = ("tetrahedron", "octahedron", "cube")

M13_LEN = 7249  # bp, standard M13mp18 scaffold


@dataclass
class CADParams:
    shape: str = "brick"
    lattice: str = "honeycomb"
    seed: int = 0

    # --- solid shapes (brick / plate) -------------------------------------
    n_rows: int = 2
    n_cols: int = 3
    length_bp: int = 84
    fill: float = 1.0          # fraction of lattice sites kept (<1 punches holes)
    stagger_bp: int = 0        # max random inset of each helix end
    repair_cross_section: bool = True   # drop dangling helices (see generator)

    # --- wireframe shapes (polygon_ring / polyhedron) ---------------------
    n_sides: int = 4           # polygon_ring only
    polyhedron: str = "tetrahedron"
    edge_bp: int = 63
    helices_per_edge: int = 2
    vertex_inset_nm: float = 1.5   # gap left at each vertex for the mate

    # --- global -----------------------------------------------------------
    scaffold_len: int = M13_LEN
    crossover_end_margin_bp: int = 7   # no crossover this close to a helix end
    inter_bundle_crossovers: bool = False

    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CADParams":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def sample_params(seed: int, shape: str | None = None, lattice: str | None = None) -> CADParams:
    """Draw one design point.  Deterministic in `seed`."""
    rng = random.Random(seed)
    shape = shape or rng.choice(SHAPES)
    lattice = lattice or rng.choice(LATTICES)
    step = 21 if lattice == "honeycomb" else 32

    p = CADParams(shape=shape, lattice=lattice, seed=seed)
    p.scaffold_len = rng.choice([M13_LEN, M13_LEN, M13_LEN, 2000, 3000, 5386])
    p.inter_bundle_crossovers = False

    if shape in ("brick", "plate"):
        if shape == "plate":
            p.n_rows = rng.choice([1, 2, 2, 3])
            p.n_cols = rng.randint(4, 12)
        else:
            p.n_rows = rng.randint(2, 5)
            p.n_cols = rng.randint(2, 6)
        # lengths snapped to the crossover period, plus off-period cases
        n_periods = rng.randint(2, 12)
        p.length_bp = n_periods * step + rng.choice([0, 0, 0, rng.randint(1, step - 1)])
        p.fill = rng.choice([1.0, 1.0, 1.0, 0.9, 0.75, 0.6])
        p.stagger_bp = rng.choice([0, 0, 0, step // 3, step])
        # a minority skip the repair, keeping the "dangling helix" class alive
        p.repair_cross_section = rng.random() < 0.85
    else:
        if shape == "polygon_ring":
            p.n_sides = rng.randint(3, 8)
        else:
            p.polyhedron = rng.choice(POLYHEDRA)
        p.edge_bp = rng.randint(2, 6) * step
        # odd helix counts are legal CAD but usually unroutable -> useful labels
        p.helices_per_edge = rng.choice([1, 2, 2, 2, 3, 4, 4, 6])
        p.vertex_inset_nm = rng.choice([1.0, 1.5, 2.0])

    p.crossover_end_margin_bp = rng.choice([0, step // 3, step // 3])
    return p
