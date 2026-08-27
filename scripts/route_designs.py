"""Label a directory of designs: precheck -> route -> staple -> export.

    python scripts/route_designs.py --designs data/designs_v0

Reads every design the generator wrote and produces labels.csv next to them:
one row per design carrying all four target groups (routing feasibility,
Hamiltonian availability, failure class, search cost) plus the cheap structural
features the precheck computed along the way.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadna import evaluate, load_design  # noqa: E402
from cadna.routing import DEFAULT_NODE_BUDGET, DEFAULT_TIME_BUDGET_S  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", type=Path, default=Path("data/designs_v0"))
    ap.add_argument("--out", type=Path, default=None, help="default: <designs>/labels.csv")
    ap.add_argument("--node-budget", type=int, default=DEFAULT_NODE_BUDGET)
    ap.add_argument("--time-budget", type=float, default=DEFAULT_TIME_BUDGET_S)
    ap.add_argument("--no-path", action="store_true", help="skip the Hamiltonian path search")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = sorted(args.designs.glob("*.json.gz")) + sorted(args.designs.glob("*.json"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no designs found in {args.designs}", file=sys.stderr)
        return 1

    out = args.out or args.designs / "labels.csv"
    rows = []
    fc: Counter[str] = Counter()
    t0 = time.perf_counter()

    for i, path in enumerate(paths, 1):
        d = load_design(path)
        lab = evaluate(
            d,
            node_budget=args.node_budget,
            time_budget_s=args.time_budget,
            with_path=not args.no_path,
        )
        rows.append(lab.to_row())
        fc[str(lab.failure_class)] += 1
        if i % 100 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    fields = list(rows[0])
    for r in rows:  # stats keys can differ if a design is degenerate
        for k in r:
            if k not in fields:
                fields.append(k)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    routable = sum(bool(r["routable"]) for r in rows)
    ham = Counter(str(r["hamilton"]) for r in rows)
    hpath = Counter(str(r["hamilton_path"]) for r in rows)
    searched = [r for r in rows if r["searched"]]
    nodes = sorted(r["nodes_expanded"] for r in searched)

    print(f"\nwrote {n} labels -> {out}   ({time.perf_counter() - t0:.1f}s)")
    print(f"routable        : {routable}/{n} ({routable / n:.1%})")
    print(f"hamilton cycle  : {dict(ham)}")
    print(f"hamilton path   : {dict(hpath)}")
    print(f"failure classes : {dict(fc.most_common())}")
    # A timeout that stopped on the *time* budget rather than the node budget is
    # a label that depends on how busy the machine was, so it is not
    # reproducible.  In a quiet run this list is empty.
    wall_bound = [r for r in rows if r["timeout"] and r["nodes_expanded"] < args.node_budget]
    if wall_bound:
        print(f"WARNING: {len(wall_bound)} timeout label(s) hit the {args.time_budget}s wall "
              f"clock instead of the {args.node_budget}-node budget, so they depend on "
              f"machine load; e.g. {[r['design_id'] for r in wall_bound[:3]]}", file=sys.stderr)

    if nodes:
        q = lambda f: nodes[min(len(nodes) - 1, int(f * len(nodes)))]  # noqa: E731
        print(f"search reached  : {len(searched)}/{n}")
        print(f"nodes expanded  : p50={q(0.5)}  p90={q(0.9)}  p99={q(0.99)}  max={nodes[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
