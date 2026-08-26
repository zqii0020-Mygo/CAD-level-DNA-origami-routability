"""The port graph the scaffold router actually searches.

Routing model
-------------
The scaffold enters a helix at one end and leaves at the other, so every
cylinder has two **ports**, ``lo`` (low bp end) and ``hi`` (high bp end), and
traversing a cylinder always flips lo <-> hi.  Consecutive cylinders are joined
by a **link**, which is either

  crossover  a candidate crossover between two helices of the same bundle.
             Both helices share a bp frame, so a crossover near the low end of
             one is near the low end of the other: crossovers join *equal*
             ports (lo-lo or hi-hi).
  mate       a CAD vertex connection between helix ends of different bundles.
             Geometry decides the ports, so mates can join lo-hi.

A scaffold route is therefore a Hamiltonian cycle over cylinders that
alternates traversals and links.

Parity invariant
----------------
Entering at ``lo`` means exiting at ``hi``.  An equal-port link then makes the
next entry ``hi`` -- the entry port *flips*.  A lo-hi link leaves it unchanged.
Closing the cycle requires the entry port to come back to where it started, so
**a valid route must use an even number of equal-port links**.  In a design
whose links are all crossovers (a single bundle -- any brick or plate) every
link flips, so an odd cylinder count admits no route at all.  That is the
classic "odd number of helices" obstruction, and it falls straight out of the
model rather than being special-cased.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from .model import Cylinder, Design

PORTS = ("lo", "hi")
Port = tuple[int, str]  # (cylinder id, "lo" | "hi")


def opposite(port: str) -> str:
    return "hi" if port == "lo" else "lo"


def port_of(c: Cylinder, bp_index: int) -> str:
    """Which end of `c` a feature at `bp_index` belongs to."""
    return "lo" if (bp_index - c.bp_start) * 2 < c.bp_len else "hi"


def _end_port(end: str) -> str:
    return "lo" if end == "start" else "hi"


@dataclass
class Link:
    id: int
    cyl_a: int
    port_a: str
    cyl_b: int
    port_b: str
    kind: str                       # "crossover" | "mate"
    source_id: int                  # CandidateCrossover.id or Mate.id
    multiplicity: int = 1           # how many candidate sites collapse into this link
    bp_index: int | None = None     # crossover links only

    @property
    def flip(self) -> bool:
        """True if using this link flips the entry port (equal-port link)."""
        return self.port_a == self.port_b

    def other(self, port: Port) -> Port:
        if port == (self.cyl_a, self.port_a):
            return (self.cyl_b, self.port_b)
        return (self.cyl_a, self.port_a)


@dataclass
class LinkGraph:
    n_cylinders: int
    links: list[Link] = field(default_factory=list)
    adj: dict[Port, list[tuple[int, str, int]]] = field(default_factory=dict)

    def neighbors(self, cyl: int, port: str) -> list[tuple[int, str, int]]:
        """(other cylinder, other port, link id) reachable from this port."""
        return self.adj.get((cyl, port), [])

    def port_degree(self, cyl: int, port: str) -> int:
        return len(self.adj.get((cyl, port), ()))

    def dead_ports(self) -> list[Port]:
        return [
            (c, p)
            for c in range(self.n_cylinders)
            for p in PORTS
            if not self.adj.get((c, p))
        ]

    def cylinder_graph(self) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.n_cylinders))
        for lk in self.links:
            g.add_edge(lk.cyl_a, lk.cyl_b)
        return g

    @property
    def n_flip(self) -> int:
        return sum(lk.flip for lk in self.links)

    @property
    def n_noflip(self) -> int:
        return len(self.links) - self.n_flip


def build_link_graph(design: Design) -> LinkGraph:
    """Collapse candidate crossovers and mates into one link per port pair.

    Several crossovers can sit at the same end of the same helix pair; a router
    would take the most extreme one, so we keep the site closest to the ends and
    record how many collapsed into it as `multiplicity` (a useful feature: it is
    the slack the router has at that junction).
    """
    cyl = {c.id: c for c in design.cylinders}
    best: dict[tuple[Port, Port], tuple[float, str, int, int | None]] = {}
    mult: dict[tuple[Port, Port], int] = {}

    def offer(pa: Port, pb: Port, score: float, kind: str, sid: int, bp: int | None) -> None:
        key = (pa, pb) if pa <= pb else (pb, pa)
        mult[key] = mult.get(key, 0) + 1
        cur = best.get(key)
        # mates outrank crossovers; within a kind the site closest to the end wins
        rank = (0 if kind == "mate" else 1, score)
        if cur is None or rank < (0 if cur[1] == "mate" else 1, cur[0]):
            best[key] = (score, kind, sid, bp)

    for x in design.crossovers:
        ca, cb = cyl[x.cyl_a], cyl[x.cyl_b]
        pa, pb = port_of(ca, x.bp_index), port_of(cb, x.bp_index)
        da = x.bp_index - ca.bp_start if pa == "lo" else ca.bp_end - x.bp_index
        db = x.bp_index - cb.bp_start if pb == "lo" else cb.bp_end - x.bp_index
        offer((ca.id, pa), (cb.id, pb), float(da + db), "crossover", x.id, x.bp_index)

    for m in design.mates:
        pa, pb = _end_port(m.end_a), _end_port(m.end_b)
        offer((m.cyl_a, pa), (m.cyl_b, pb), m.gap_nm, "mate", m.id, None)

    lg = LinkGraph(n_cylinders=len(design.cylinders))
    for (pa, pb), (_score, kind, sid, bp) in sorted(best.items()):
        if pa[0] == pb[0]:
            continue  # a helix may not link to itself
        lk = Link(
            id=len(lg.links),
            cyl_a=pa[0], port_a=pa[1],
            cyl_b=pb[0], port_b=pb[1],
            kind=kind, source_id=sid,
            multiplicity=mult[(pa, pb)], bp_index=bp,
        )
        lg.links.append(lk)
        lg.adj.setdefault(pa, []).append((pb[0], pb[1], lk.id))
        lg.adj.setdefault(pb, []).append((pa[0], pa[1], lk.id))
    return lg
