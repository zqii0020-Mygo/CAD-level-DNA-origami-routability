"""Scaffold routing: search, validation, and the label record.

The router looks for a Hamiltonian cycle in the port graph of `linkgraph.py`:
enter a helix at one end, leave at the other, take a crossover or a mate to the
next helix, and come back to where you started having visited every helix once.

The search is a plain DFS with two standard accelerations -- a Warnsdorff-style
"fewest onward options first" ordering and a per-node feasibility prune -- under
an explicit node and time budget.  That budget is the point: how far the DFS
gets is the `search cost` label, and hitting the budget is the `timeout` class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .linkgraph import PORTS, Link, LinkGraph, build_link_graph, opposite
from .model import Design
from .precheck import PrecheckResult, precheck

DEFAULT_NODE_BUDGET = 200_000
DEFAULT_TIME_BUDGET_S = 5.0

# Longest stretch of a helix that may be left without an available staple
# crossover before staple routing is considered infeasible.
MAX_STAPLE_SPAN_BP = 49


class _BudgetExceeded(Exception):
    pass


@dataclass
class RouteStep:
    cyl: int
    entry_port: str
    link_id: int | None      # link used to enter this cylinder (None at the start)


@dataclass
class RoutingResult:
    status: str                                    # routed | no_route | timeout | skipped
    route: list[RouteStep] = field(default_factory=list)
    closing_link_id: int | None = None
    nodes_expanded: int = 0
    backtracks: int = 0
    max_depth: int = 0
    elapsed_s: float = 0.0

    @property
    def timeout(self) -> bool:
        return self.status == "timeout"

    @property
    def routed(self) -> bool:
        return self.status == "routed"

    def link_ids(self) -> list[int]:
        ids = [s.link_id for s in self.route if s.link_id is not None]
        if self.closing_link_id is not None:
            ids.append(self.closing_link_id)
        return ids


# ---------------------------------------------------------------------- search
def find_scaffold_route(
    lg: LinkGraph,
    node_budget: int = DEFAULT_NODE_BUDGET,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    prune: bool = True,
    start: int = 0,
) -> RoutingResult:
    """DFS for a Hamiltonian cycle over cylinder ports.

    The start cylinder is fixed, entering at `lo`: every cycle passes through
    every cylinder, and traversing a cycle backwards turns a `hi` entry into a
    `lo` entry, so this loses no solutions while halving the search.
    """
    n = lg.n_cylinders
    if n < 2:
        return RoutingResult(status="no_route")

    target = (start, "lo")
    visited = [False] * n
    visited[start] = True
    order: list[RouteStep] = [RouteStep(start, "lo", None)]
    closing: list[int] = []

    t0 = time.perf_counter()
    nodes = 0
    backtracks = 0
    max_depth = 1

    def feasible(cur: int, cur_exit: str) -> bool:
        """Every unvisited helix must still be able to be entered and left.

        A port whose only surviving option is the closing target, or the port we
        are about to leave from, is a port with no slack: the rest of the route
        uses each of those exactly once, so if two different ports depend on the
        same one the branch is already dead.  Counting them matters for the path
        search, where the virtual cylinder reaches every port and would
        otherwise make this prune vacuous.
        """
        need_target = 0
        need_cur = 0
        for u in range(n):
            if visited[u]:
                continue
            for p in PORTS:
                free = at_target = at_cur = False
                for (d, q, _li) in lg.neighbors(u, p):
                    if not visited[d]:
                        free = True
                        break
                    if (d, q) == target:
                        at_target = True
                    elif (d, q) == (cur, cur_exit):
                        at_cur = True
                if free or (at_target and at_cur):
                    continue        # has slack, or a choice of the two scarce ends
                if at_target:
                    need_target += 1
                elif at_cur:
                    need_cur += 1
                else:
                    return False
        return need_target <= 1 and need_cur <= 1

    def onward_options(d: int, q: str) -> int:
        exit_port = opposite(q)
        return sum(
            1 for (e, r, _li) in lg.neighbors(d, exit_port)
            if not visited[e] or (e, r) == target
        )

    def dfs(cur: int, entry: str) -> bool:
        nonlocal nodes, backtracks, max_depth
        nodes += 1
        if nodes >= node_budget:
            raise _BudgetExceeded
        if nodes % 1024 == 0 and time.perf_counter() - t0 > time_budget_s:
            raise _BudgetExceeded

        depth = len(order)
        if depth > max_depth:
            max_depth = depth
        cur_exit = opposite(entry)

        if depth == n:
            for (d, q, li) in lg.neighbors(cur, cur_exit):
                if (d, q) == target:
                    closing.append(li)
                    return True
            return False

        if prune and not feasible(cur, cur_exit):
            return False

        cands = [(d, q, li) for (d, q, li) in lg.neighbors(cur, cur_exit) if not visited[d]]
        cands.sort(key=lambda t: (onward_options(t[0], t[1]), t[0], t[1]))
        for (d, q, li) in cands:
            visited[d] = True
            order.append(RouteStep(d, q, li))
            if dfs(d, q):
                return True
            order.pop()
            visited[d] = False
            backtracks += 1
        return False

    try:
        found = dfs(start, "lo")
        status = "routed" if found else "no_route"
    except _BudgetExceeded:
        status = "timeout"
        found = False

    return RoutingResult(
        status=status,
        route=list(order) if found else [],
        closing_link_id=closing[0] if (found and closing) else None,
        nodes_expanded=nodes,
        backtracks=backtracks,
        max_depth=max_depth,
        elapsed_s=round(time.perf_counter() - t0, 6),
    )


def _with_virtual_cylinder(lg: LinkGraph) -> LinkGraph:
    """Augment the graph so a Hamiltonian *cycle* through it is a Hamiltonian path.

    A circular scaffold does not need a crossover to close: in a flat sheet the
    raster ends far apart and the scaffold loop simply spans the gap (this is how
    a Rothemund rectangle closes).  Modelling that as a free closure means adding
    one virtual cylinder whose two ports reach every real port; a cycle through
    it is exactly a Hamiltonian path over the real cylinders plus that free jump.
    """
    n = lg.n_cylinders
    aug = LinkGraph(n_cylinders=n + 1, links=list(lg.links))
    aug.adj = {k: list(v) for k, v in lg.adj.items()}
    for c in range(n):
        for p in PORTS:
            for vp in PORTS:
                lk = Link(id=len(aug.links), cyl_a=n, port_a=vp, cyl_b=c, port_b=p,
                          kind="virtual", source_id=-1)
                aug.links.append(lk)
                aug.adj.setdefault((n, vp), []).append((c, p, lk.id))
                aug.adj.setdefault((c, p), []).append((n, vp, lk.id))
    return aug


def find_scaffold_path(
    lg: LinkGraph,
    node_budget: int = DEFAULT_NODE_BUDGET,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
) -> RoutingResult:
    """Hamiltonian path search: a route that may close through the scaffold loop."""
    if lg.n_cylinders < 2:
        return RoutingResult(status="no_route")
    aug = _with_virtual_cylinder(lg)
    # The virtual cylinder is the last id, so start the DFS there.
    return find_scaffold_route(
        aug, node_budget=node_budget, time_budget_s=time_budget_s, start=lg.n_cylinders
    )


# ------------------------------------------------------------------ validation
def validate_route(lg: LinkGraph, res: RoutingResult) -> tuple[bool, str]:
    """The `export` gate: does the route the search returned actually hold up."""
    n = lg.n_cylinders
    route = res.route
    if len(route) != n:
        return False, f"route visits {len(route)} of {n} cylinders"
    if len({s.cyl for s in route}) != n:
        return False, "route revisits a cylinder"
    if res.closing_link_id is None:
        return False, "route does not close"

    link = {lk.id: lk for lk in lg.links}
    used: set[int] = set()
    for i, step in enumerate(route):
        prev = route[i - 1]
        li = step.link_id if i else res.closing_link_id
        if li is None:
            return False, f"step {i} has no link"
        if li in used:
            return False, f"link {li} used twice"
        used.add(li)
        lk = link[li]
        want = {(prev.cyl, opposite(prev.entry_port)), (step.cyl, step.entry_port)}
        have = {(lk.cyl_a, lk.port_a), (lk.cyl_b, lk.port_b)}
        if want != have:
            return False, f"step {i}: link {li} joins {sorted(have)} not {sorted(want)}"
    return True, ""


def check_staples(design: Design, lg: LinkGraph, res: RoutingResult) -> tuple[bool, str, int]:
    """Crossover sites the scaffold did not consume must still anchor staples.

    A helix stretch longer than MAX_STAPLE_SPAN_BP with no free crossover has
    nowhere to put a staple crossover, so its staples would be one long
    unsupported domain.
    """
    link = {lk.id: lk for lk in lg.links}
    consumed: set[int] = set()
    for li in res.link_ids():
        lk = link[li]
        if lk.kind == "crossover":
            consumed.add(lk.source_id)

    anchors: dict[int, list[int]] = {c.id: [] for c in design.cylinders}
    for x in design.crossovers:
        if x.id in consumed:
            continue
        anchors[x.cyl_a].append(x.bp_index)
        anchors[x.cyl_b].append(x.bp_index)

    worst = 0
    worst_cyl = -1
    for c in design.cylinders:
        pts = sorted({c.bp_start, c.bp_end, *anchors[c.id]})
        span = max((b - a) for a, b in zip(pts, pts[1:])) if len(pts) > 1 else c.bp_len
        if span > worst:
            worst, worst_cyl = span, c.id
    if worst > MAX_STAPLE_SPAN_BP:
        return False, f"cylinder {worst_cyl} has a {worst} bp stretch with no free crossover", worst
    return True, "", worst


def _add_path_label(
    lab: "DesignLabel", lg: LinkGraph, with_path: bool, node_budget: int, time_budget_s: float
) -> None:
    """Answer the relaxed question once the strict one has come back negative."""
    if not with_path or lg.n_cylinders < 2:
        return
    res = find_scaffold_path(lg, node_budget=node_budget, time_budget_s=time_budget_s)
    lab.path_nodes_expanded = res.nodes_expanded
    lab.path_elapsed_s = res.elapsed_s
    lab.path_timeout = res.timeout
    lab.hamilton_path = None if res.timeout else res.routed


# ----------------------------------------------------------------- label record
@dataclass
class DesignLabel:
    design_id: str
    shape: str
    lattice: str
    # targets
    routable: bool = False
    hamilton: bool | None = None            # None == unknown (search timed out)
    hamilton_path: bool | None = None       # cycle allowed to close via the scaffold loop
    failure_class: str | None = None
    nodes_expanded: int = 0
    backtracks: int = 0
    max_depth: int = 0
    elapsed_s: float = 0.0
    timeout: bool = False
    # the path search is costed separately so it cannot pollute the cycle-search label
    path_nodes_expanded: int = 0
    path_elapsed_s: float = 0.0
    path_timeout: bool = False
    # supporting detail (kept as separate flags so multi-label info survives)
    precheck_ok: bool = False
    precheck_reason: str | None = None
    detail: str = ""
    searched: bool = False
    n_route_links: int = 0
    n_route_crossovers: int = 0
    n_route_mates: int = 0
    staple_ok: bool | None = None
    max_staple_span_bp: int = 0
    export_ok: bool | None = None
    scaffold_ok: bool = True
    stats: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row = {k: v for k, v in self.__dict__.items() if k != "stats"}
        row.update(self.stats)
        return row


def evaluate(
    design: Design,
    node_budget: int = DEFAULT_NODE_BUDGET,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    with_path: bool = True,
) -> DesignLabel:
    """Run the whole precheck -> route -> staple -> export chain and label it.

    `routable` and `hamilton` are the strict cycle model.  `hamilton_path` is the
    same question allowing the circular scaffold to close itself once without a
    crossover, which is what a flat single-row sheet actually does.
    """
    lg = build_link_graph(design)
    pc: PrecheckResult = precheck(design, lg)

    lab = DesignLabel(
        design_id=design.id,
        shape=str(design.params.get("shape", "?")),
        lattice=design.lattice,
        precheck_ok=pc.ok,
        precheck_reason=pc.reason,
        detail=pc.detail,
        stats=pc.stats,
        scaffold_ok=pc.reason != "scaffold_length",
    )

    if pc.fatal:
        # Every fatal precheck is an exact obstruction, so no cycle exists.
        lab.routable = False
        lab.hamilton = False
        lab.failure_class = pc.reason
        _add_path_label(lab, lg, with_path, node_budget, time_budget_s)
        return lab

    res = find_scaffold_route(lg, node_budget=node_budget, time_budget_s=time_budget_s)
    lab.searched = True
    lab.nodes_expanded = res.nodes_expanded
    lab.backtracks = res.backtracks
    lab.max_depth = res.max_depth
    lab.elapsed_s = res.elapsed_s
    lab.timeout = res.timeout

    if res.timeout:
        lab.routable = False
        lab.hamilton = None
        lab.failure_class = "timeout"
        lab.detail = f"budget hit after {res.nodes_expanded} nodes"
        _add_path_label(lab, lg, with_path, node_budget, time_budget_s)
        return lab

    if not res.routed:
        lab.routable = False
        lab.hamilton = False
        lab.failure_class = "hamilton"
        lab.detail = f"search exhausted after {res.nodes_expanded} nodes, no cycle"
        _add_path_label(lab, lg, with_path, node_budget, time_budget_s)
        return lab

    lab.hamilton = True
    lab.hamilton_path = True   # a cycle contains a Hamiltonian path
    link = {lk.id: lk for lk in lg.links}
    used = res.link_ids()
    lab.n_route_links = len(used)
    lab.n_route_crossovers = sum(link[i].kind == "crossover" for i in used)
    lab.n_route_mates = sum(link[i].kind == "mate" for i in used)

    export_ok, export_detail = validate_route(lg, res)
    lab.export_ok = export_ok
    staple_ok, staple_detail, worst = check_staples(design, lg, res)
    lab.staple_ok = staple_ok
    lab.max_staple_span_bp = worst

    if not export_ok:
        lab.failure_class = "export"
        lab.detail = export_detail
    elif not lab.scaffold_ok:
        lab.failure_class = "scaffold_length"
        lab.detail = pc.detail
    elif not staple_ok:
        lab.failure_class = "staple_routing"
        lab.detail = staple_detail
    else:
        lab.routable = True
        lab.detail = ""
    return lab
