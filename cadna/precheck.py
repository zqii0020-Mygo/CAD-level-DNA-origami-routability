"""Cheap necessary conditions on a design, before any search is attempted.

Everything here is O(V + E) or close to it, and every hard condition is an
*exact* obstruction -- if a check fires, no scaffold route exists, no search
needed.  This is both the first stage of the pipeline and the baseline the
learned surrogate has to beat: a model that only reproduces these rules has
learned nothing.

Exact obstructions used
-----------------------
dead port            a helix end with no crossover and no mate -- the scaffold
                     can neither arrive nor leave there
disconnected         a Hamiltonian cycle spans the graph, so it must be connected
degree < 2           every cylinder needs a distinct entry and exit neighbour
                     (from 3 cylinders up -- a pair joined by both a lo-lo and a
                     hi-hi link is a legal 2-cycle)
cut vertex           a Hamiltonian cycle is 2-connected, so a cut vertex rules
                     one out (only meaningful for 3+ cylinders)
parity               with only equal-port links the entry port flips every step,
                     so an odd cylinder count cannot close (see linkgraph.py)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .linkgraph import PORTS, LinkGraph, build_link_graph
from .model import Design

# The failure taxonomy the surrogate predicts.  `scaffold_length` is not in the
# original six: it is kept separate because it is frequent and physically
# distinct (the topology is fine, there is simply not enough scaffold).
FAILURE_CLASSES = (
    "geometry",
    "pairability",
    "hamilton",
    "timeout",
    "staple_routing",
    "export",
    "scaffold_length",
)


@dataclass
class PrecheckResult:
    ok: bool
    reason: str | None = None       # a FAILURE_CLASSES member
    detail: str = ""
    fatal: bool = False             # True => routing cannot possibly succeed
    stats: dict[str, Any] = field(default_factory=dict)


def _summarise(values, prefix: str) -> dict[str, Any]:
    values = list(values)
    if not values:
        return {f"{prefix}_min": 0, f"{prefix}_mean": 0.0, f"{prefix}_max": 0}
    return {
        f"{prefix}_min": min(values),
        f"{prefix}_mean": round(statistics.fmean(values), 3),
        f"{prefix}_max": max(values),
    }


def link_graph_stats(design: Design, lg: LinkGraph) -> dict[str, Any]:
    """Cheap structural features -- the precheck's own output, and GNN inputs."""
    g = lg.cylinder_graph()
    n = lg.n_cylinders
    port_deg = [lg.port_degree(c, p) for c in range(n) for p in PORTS]
    cyl_deg = [d for _, d in g.degree()] if n else []
    comps = nx.number_connected_components(g) if n else 0
    arts = list(nx.articulation_points(g)) if n >= 3 and comps == 1 else []
    bridges = list(nx.bridges(g)) if n >= 2 and comps == 1 else []

    total_bp = design.total_bp
    scaffold = int(design.params.get("scaffold_len", 0) or 0)
    stats: dict[str, Any] = {
        "n_features": len(design.features),
        "n_bundles": len(design.bundles),
        "n_cylinders": n,
        "n_candidate_crossovers": len(design.crossovers),
        "n_mates": len(design.mates),
        "n_unpaired_ends": len(design.unpaired_ends),
        "n_links": len(lg.links),
        "n_links_crossover": sum(lk.kind == "crossover" for lk in lg.links),
        "n_links_mate": sum(lk.kind == "mate" for lk in lg.links),
        "n_links_flip": lg.n_flip,
        "n_links_noflip": lg.n_noflip,
        "n_dead_ports": len(lg.dead_ports()),
        "n_components": comps,
        "n_articulation": len(arts),
        "n_bridges": len(bridges),
        "link_density": round(len(lg.links) / max(1, n), 3),
        "total_bp": total_bp,
        "scaffold_len": scaffold,
        "bp_over_scaffold": round(total_bp / scaffold, 4) if scaffold else 0.0,
    }
    stats.update(_summarise(port_deg, "port_deg"))
    stats.update(_summarise(cyl_deg, "cyl_deg"))
    return stats


def precheck(design: Design, lg: LinkGraph | None = None) -> PrecheckResult:
    lg = lg if lg is not None else build_link_graph(design)
    stats = link_graph_stats(design, lg)
    n = lg.n_cylinders

    def fail(reason: str, detail: str, fatal: bool = True) -> PrecheckResult:
        return PrecheckResult(ok=False, reason=reason, detail=detail, fatal=fatal, stats=stats)

    if n < 2:
        return fail("geometry", f"{n} cylinder(s): nothing to route")
    if not lg.links:
        return fail("pairability", "no crossover or mate links at all")
    if stats["n_dead_ports"]:
        dead = lg.dead_ports()[:4]
        return fail("pairability", f"{stats['n_dead_ports']} helix end(s) unreachable, e.g. {dead}")
    if stats["n_components"] > 1:
        return fail("geometry", f"cylinder graph splits into {stats['n_components']} components")
    # Two cylinders joined by both a lo-lo and a hi-hi link are a legal 2-cycle,
    # so the degree-2 requirement only bites from three cylinders up.
    if n >= 3 and stats["cyl_deg_min"] < 2:
        return fail("hamilton", "a cylinder has degree < 2, so no cycle can pass through it")
    if n >= 3 and stats["n_articulation"]:
        return fail("hamilton", f"{stats['n_articulation']} cut vertex/vertices: graph is not 2-connected")
    if lg.n_noflip == 0 and n % 2 == 1:
        return fail("hamilton", f"{n} cylinders with only equal-port links: parity forbids a closed route")

    # Non-fatal: the topology is routable, there just is not enough scaffold.
    if stats["scaffold_len"] and stats["total_bp"] > stats["scaffold_len"]:
        return PrecheckResult(
            ok=False, reason="scaffold_length", fatal=False, stats=stats,
            detail=f"needs {stats['total_bp']} bp, scaffold has {stats['scaffold_len']}",
        )
    return PrecheckResult(ok=True, stats=stats)
