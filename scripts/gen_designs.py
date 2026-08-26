"""Sample the CAD parameter space and dump a batch of designs.

    python scripts/gen_designs.py --n 500 --out data/designs_v0

Writes one <design_id>.json.gz per design plus index.csv, the table the routing
stage will later join its labels onto.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadna import generate, sample_params, save_design  # noqa: E402
from cadna.params import LATTICES, SHAPES  # noqa: E402

INDEX_FIELDS = [
    "design_id", "seed", "shape", "lattice", "path",
    "n_features", "n_bundles", "n_cylinders",
    "n_adjacency", "n_adjacency_inter", "n_crossovers",
    "n_mates", "n_unpaired_ends", "total_bp", "scaffold_len",
    "xover_per_cyl", "bp_over_scaffold",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of designs")
    ap.add_argument("--seed0", type=int, default=0, help="first seed")
    ap.add_argument("--out", type=Path, default=Path("data/designs_v0"))
    ap.add_argument("--shape", choices=SHAPES, default=None, help="force one shape")
    ap.add_argument("--lattice", choices=LATTICES, default=None, help="force one lattice")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    shapes: Counter[str] = Counter()
    failures = 0

    for k in range(args.n):
        seed = args.seed0 + k
        try:
            params = sample_params(seed, shape=args.shape, lattice=args.lattice)
            d = generate(params)
        except Exception as exc:  # a generator crash is a bug, not a label
            failures += 1
            print(f"[gen-error] seed={seed}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue

        path = save_design(d, args.out / f"{d.id}.json.gz")
        s = d.summary()
        shapes[s["shape"]] += 1
        rows.append({
            "design_id": d.id,
            "seed": seed,
            "shape": s["shape"],
            "lattice": s["lattice"],
            "path": path.name,
            "n_features": s["n_features"],
            "n_bundles": s["n_bundles"],
            "n_cylinders": s["n_cylinders"],
            "n_adjacency": s["n_adjacency"],
            "n_adjacency_inter": s["n_adjacency_inter"],
            "n_crossovers": s["n_crossovers"],
            "n_mates": s["n_mates"],
            "n_unpaired_ends": s["n_unpaired_ends"],
            "total_bp": s["total_bp"],
            "scaffold_len": s["scaffold_len"],
            "xover_per_cyl": round(s["n_crossovers"] / max(1, s["n_cylinders"]), 3),
            "bp_over_scaffold": round(s["total_bp"] / max(1, s["scaffold_len"]), 3),
        })

    index = args.out / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)

    if not args.quiet:
        print(f"wrote {len(rows)} designs -> {args.out}")
        print(f"index: {index}")
        print("shapes:", dict(sorted(shapes.items())))
        if rows:
            for key in ("n_cylinders", "n_crossovers", "n_mates", "bp_over_scaffold"):
                vals = sorted(r[key] for r in rows)
                lo, med, hi = vals[0], vals[len(vals) // 2], vals[-1]
                print(f"  {key:<17} min={lo}  median={med}  max={hi}")
            over = sum(r["bp_over_scaffold"] > 1.0 for r in rows)
            unp = sum(r["n_unpaired_ends"] > 0 for r in rows)
            print(f"  scaffold too short: {over}/{len(rows)}   unpaired ends: {unp}/{len(rows)}")
        if failures:
            print(f"generator errors: {failures}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
