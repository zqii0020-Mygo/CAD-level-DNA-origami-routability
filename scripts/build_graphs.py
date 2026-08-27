"""Turn a directory of designs (+ labels.csv) into hetero graphs on disk.

    python scripts/build_graphs.py --designs data/designs_v0

Writes one `.npz` per design into `<designs>/../graphs_v0` (override with
--out) plus a `graphs_index.csv` manifest carrying the node/edge counts and the
targets, so the training split can be chosen without opening any graph.

Designs whose label is missing from labels.csv are still exported, with an
empty target dict, unless --labelled-only is given.  With --relabel the labels
are recomputed here instead of being read from the CSV.
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
from cadna.graph import NODE_TYPES, RELATIONS, build_graph  # noqa: E402


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["design_id"]: row for row in csv.DictReader(fh)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", type=Path, default=Path("data/designs_v0"))
    ap.add_argument("--labels", type=Path, default=None, help="default: <designs>/labels.csv")
    ap.add_argument("--out", type=Path, default=None, help="default: <designs>_graphs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--relabel", action="store_true", help="re-run evaluate() instead of reading labels.csv")
    ap.add_argument("--labelled-only", action="store_true", help="skip designs with no label")
    args = ap.parse_args()

    paths = sorted(args.designs.glob("*.json.gz")) + sorted(args.designs.glob("*.json"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"no designs found in {args.designs}", file=sys.stderr)
        return 1

    labels = {} if args.relabel else read_labels(args.labels or args.designs / "labels.csv")
    if not labels and not args.relabel:
        print("warning: no labels.csv found -- graphs will carry no targets", file=sys.stderr)

    out = args.out or args.designs.with_name(args.designs.name + "_graphs")
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    missing = 0
    t0 = time.perf_counter()

    for i, path in enumerate(paths, 1):
        d = load_design(path)
        lab = evaluate(d) if args.relabel else labels.get(d.id)
        if lab is None:
            missing += 1
            if args.labelled_only:
                continue
        hg = build_graph(d, lab)
        gpath = hg.save(out / f"{d.id}.npz")
        row: dict[str, object] = {
            "design_id": d.id,
            "graph": gpath.name,
            "shape": hg.meta["shape"],
            "lattice": hg.meta["lattice"],
            "sampling": hg.meta.get("sampling", "iid"),
            **{f"n_{t}": hg.num_nodes(t) for t in NODE_TYPES},
            # full triple: `contains` and `in` name two relations each
            **{"n_" + "_".join(rel): hg.num_edges(rel) for rel in RELATIONS},
            "n_edges": hg.num_edges(),
        }
        row.update(hg.y)
        rows.append(row)
        if i % 100 == 0 or i == len(paths):
            print(f"  {i}/{len(paths)}  ({time.perf_counter() - t0:.1f}s)", flush=True)

    index = out / "graphs_index.csv"
    fields = list(rows[0])
    with index.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)

    total_nodes = sum(sum(int(r[f"n_{t}"]) for t in NODE_TYPES) for r in rows)
    shapes = Counter(str(r["shape"]) for r in rows)
    print(f"\nwrote {len(rows)} graphs -> {out}   ({time.perf_counter() - t0:.1f}s)")
    print(f"index           : {index}")
    print(f"shapes          : {dict(shapes.most_common())}")
    print(f"nodes total     : {total_nodes}   edges total: {sum(int(r['n_edges']) for r in rows)}")
    if missing:
        print(f"unlabelled      : {missing} design(s) had no row in labels.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
