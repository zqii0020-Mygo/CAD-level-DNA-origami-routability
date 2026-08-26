"""Lattice geometry and crossover phase rules (caDNAno-style).

Two conventions are supported, matching caDNAno / MagicDNA:

  honeycomb : 21 bp per 2 turns (10.5 bp/turn), 3 neighbours per helix,
              crossover phases spaced 7 bp apart.
  square    : 32 bp per 3 turns (10.67 bp/turn), 4 neighbours per helix,
              crossover phases spaced 8 bp apart.

Crossover rule
--------------
A crossover between two neighbouring helices occupies the *same* bp index on
both helices, so the phase offset must be a symmetric function of the two
lattice sites.  We derive it from the direction of the inter-helix vector:
the backbone of a duplex faces its neighbour only once per turn, and because
`step` bp is an integer number of turns the allowed indices are periodic with
period `step`.

  honeycomb: the 3 neighbour axes are 120 deg apart -> offsets {0, 7, 14}
  square:    the 2 neighbour axes are 90 deg apart, and the helix phase
             alternates with lattice parity -> offsets {0, 8, 16, 24}

`crossover_offset` is symmetric under swapping its two arguments, which is the
only hard requirement.  The concrete tables are intentionally isolated here so
they can later be replaced by the exact caDNAno lookup tables without touching
the rest of the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

RISE_PER_BP = 0.34  # nm, axial rise of B-form DNA
SQRT3 = math.sqrt(3.0)

Site = tuple[int, int]  # (row, col) lattice coordinate inside a bundle


@dataclass(frozen=True)
class Lattice:
    name: str
    step: int             # bp per crossover period (== integer number of turns)
    turns_per_step: int
    interhelix: float     # nm, centre-to-centre distance of touching helices

    @property
    def bp_per_turn(self) -> float:
        return self.step / self.turns_per_step

    @property
    def radius(self) -> float:
        return self.interhelix / 2.0

    @property
    def n_dirs(self) -> int:
        return 3 if self.name == "honeycomb" else 4

    # ---------------------------------------------------------------- geometry
    def position(self, row: int, col: int) -> tuple[float, float]:
        """2D cross-section position of a lattice site, in nm."""
        r = self.radius
        if self.name == "honeycomb":
            x = col * SQRT3 * r
            y = row * 3.0 * r + (r if (row + col) % 2 else 0.0)
        else:
            x = col * self.interhelix
            y = row * self.interhelix
        return x, y

    def neighbors(self, row: int, col: int) -> list[Site]:
        """Lattice sites in physical contact with (row, col)."""
        if self.name == "honeycomb":
            third = (row - 1, col) if (row + col) % 2 == 0 else (row + 1, col)
            return [(row, col - 1), (row, col + 1), third]
        return [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

    def are_neighbors(self, a: Site, b: Site) -> bool:
        return tuple(b) in self.neighbors(*a)

    def direction_deg(self, a: Site, b: Site) -> float:
        """Angle of the a -> b inter-helix vector in the cross-section plane."""
        xa, ya = self.position(*a)
        xb, yb = self.position(*b)
        return math.degrees(math.atan2(yb - ya, xb - xa)) % 360.0

    # --------------------------------------------------------------- crossovers
    def crossover_offset(self, a: Site, b: Site) -> int:
        """Symmetric bp phase offset for crossovers between sites a and b."""
        if self.name == "honeycomb":
            # 6 possible directions {30,90,150,210,270,330}; opposite directions
            # (k and k+3) must share an offset, hence `% 3`.
            k6 = int(round((self.direction_deg(a, b) - 30.0) / 60.0)) % 6
            return (self.step // 3) * (k6 % 3)
        (ra, ca), (rb, cb) = a, b
        if ra == rb:                                   # pair along the x axis
            return (self.step // 4) * (0 + 2 * (ra % 2))
        return (self.step // 4) * (1 + 2 * (ca % 2))   # pair along the y axis

    def crossover_indices(
        self, a: Site, b: Site, lo: int, hi: int, margin: int = 0
    ) -> Iterator[int]:
        """bp indices in [lo, hi) where a crossover between a and b may sit."""
        off = self.crossover_offset(a, b)
        start = lo + margin
        stop = hi - margin
        first = start + ((off - start) % self.step)
        i = first
        while i < stop:
            yield i
            i += self.step


HONEYCOMB = Lattice("honeycomb", step=21, turns_per_step=2, interhelix=2.25)
SQUARE = Lattice("square", step=32, turns_per_step=3, interhelix=2.60)

_LATTICES = {l.name: l for l in (HONEYCOMB, SQUARE)}


def get_lattice(name: str) -> Lattice:
    try:
        return _LATTICES[name]
    except KeyError:
        raise ValueError(f"unknown lattice {name!r}, expected one of {sorted(_LATTICES)}")
