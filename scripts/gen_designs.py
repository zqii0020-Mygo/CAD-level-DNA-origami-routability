"""Sample the CAD parameter space and dump a batch of designs.

    python scripts/gen_designs.py --n 500 --out data/designs_v0
    python scripts/gen_designs.py --n 10000 --out data/designs_v1 --oversample geometry,timeout

Writes one <design_id>.json.gz per design plus index.csv, the table the routing
stage will later join its labels onto.

Rare-class oversampling
-----------------------
Some failure classes are thin under the natural sampler -- `geometry` and
`timeout` are both well under 5% -- which leaves too few examples to learn or
to measure.  `--oversample` adds designs drawn by rejection sampling against a
*cheap probe*: the precheck for the classes it decides outright, and a
short-budget DFS as a proxy for `timeout`, since a design that blows a 20k-node
budget is far more likely to blow the real 200k one.

These designs are not from the natural distribution, so they are marked:
`notes["sampling"]` and the index's `sampling` column say `rare:<class>` instead
of `iid`.  The marker travels into the graph metadata, and `models.data`
keeps marked designs out of validation and test -- otherwise every headline
number would describe a distribution nobody deploys on.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadna import evaluate, generate, sample_params, save_design  # noqa: E402
from cadna.params import LATTICES, SHAPES  # noqa: E402
from cadna.precheck import FAILURE_CLASSES  # noqa: E402

# Seeds for the oversampled tail start here, so they can never collide with the
# iid seeds of the same run.
RARE_SEED0 = 10_000_000

INDEX_FIELDS = [
    "design_id", "seed", "sampling", "shape", "lattice", "path",
    "n_features", "n_bundles", "n_cylinders",
    "n_adjacency", "n_adjacency_inter", "n_crossovers",
    "n_mates", "n_unpaired_ends", "total_bp", "scaffold_len",
    "xover_per_cyl", "bp_over_scaffold",
]


def probe_class(design, node_budget: int, time_budget_s: float) -> str | None:
    """Cheap guess at the failure class, used only as a rejection-sampling filter."""
    lab = evaluate(design, node_budget=node_budget, time_budget_s=time_budget_s,
                   with_path=False)
    return lab.failure_class


def draw_rare(wanted: dict[str, int], seed0: int, max_tries: int,
              node_budget: int, time_budget_s: float, quiet: bool):
    """Rejection-sample designs whose probed class is one that is still short."""
    remaining = dict(wanted)
    tries = 0
    seed = seed0
    while tries < max_tries and any(v > 0 for v in remaining.values()):
        tries += 1
        seed += 1
        try:
            d = generate(sample_params(seed))
        except Exception:
            continue
        cls = probe_class(d, node_budget, time_budget_s)
        if cls is None or remaining.get(cls, 0) <= 0:
            continue
        remaining[cls] -= 1
        d.notes["sampling"] = f"rare:{cls}"
        yield seed, d
    if not quiet:
        short = {k: v for k, v in remaining.items() if v > 0}
        print(f"oversampling: {tries} draws"
              + (f", short of quota: {short}" if short else ", quotas met"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of designs")
    ap.add_argument("--seed0", type=int, default=0, help="first seed")
    ap.add_argument("--out", type=Path, default=Path("data/designs_v0"))
    ap.add_argument("--shape", choices=SHAPES, default=None, help="force one shape")
    ap.add_argument("--lattice", choices=LATTICES, default=None, help="force one lattice")
    ap.add_argument("--oversample", default=None,
                    help=f"comma-separated failure classes to top up, from {','.join(FAILURE_CLASSES)}")
    ap.add_argument("--oversample-n", type=int, default=300, help="extra designs per class")
    ap.add_argument("--oversample-tries", type=int, default=0,
                    help="draw budget for the rejection sampler (default: 60x the quota)")
    ap.add_argument("--probe-budget", type=int, default=20_000, help="probe DFS node budget")
    ap.add_argument("--probe-time", type=float, default=0.3, help="probe DFS time budget, s")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    rare_classes = [c.strip() for c in args.oversample.split(",")] if args.oversample else []
    for c in rare_classes:
        if c not in FAILURE_CLASSES:
            print(f"unknown failure class {c!r}; known: {FAILURE_CLASSES}", file=sys.stderr)
            return 2

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    shapes: Counter[str] = Counter()
    failures = 0

    def iid_designs():
        nonlocal failures
        for k in range(args.n):
            seed = args.seed0 + k
            try:
                params = sample_params(seed, shape=args.shape, lattice=args.lattice)
                yield seed, generate(params)
            except Exception as exc:  # a generator crash is a bug, not a label
                failures += 1
                print(f"[gen-error] seed={seed}: {type(exc).__name__}: {exc}", file=sys.stderr)

    def every_design():
        yield from iid_designs()
        if rare_classes:
            quota = {c: args.oversample_n for c in rare_classes}
            tries = args.oversample_tries or 60 * args.oversample_n * len(rare_classes)
            yield from draw_rare(quota, RARE_SEED0 + args.seed0, tries,
                                 args.probe_budget, args.probe_time, args.quiet)

    sampling_counts: Counter[str] = Counter()
    for seed, d in every_design():
        tag = str(d.notes.get("sampling", "iid"))
        sampling_counts[tag] += 1
        path = save_design(d, args.out / f"{d.id}.json.gz")
        s = d.summary()
        shapes[s["shape"]] += 1
        rows.append({
            "design_id": d.id,
            "seed": seed,
            "sampling": tag,
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
        if len(sampling_counts) > 1:
            print("sampling:", dict(sorted(sampling_counts.items())))
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
