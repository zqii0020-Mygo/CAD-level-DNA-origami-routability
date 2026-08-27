"""Train and compare the baseline surrogates.

    python scripts/train_baseline.py --graphs data/designs_v1_graphs --seeds 3
    python scripts/train_baseline.py \
        --graphs data/designs_v1_graphs,data/designs_v1_rare_graphs \
        --out results/baseline_v1_rare.json

Reports test metrics for each configuration next to three reference points --
the majority-class predictor, the pipeline's own precheck verdict, and a
size-only cost predictor -- on three slices of the test set:

    all                   every test design
    precheck-decided      a fatal precheck already answered: the label is free
    precheck-undecided    the label required the DFS

The third slice is the one that decides whether a surrogate is worth anything.
A model that has merely learned `precheck.py` scores well on `all` by banking
the free half; on `precheck-undecided` it has nothing left to bank.

The results JSON carries a provenance block (code revision, dataset digests,
environment) so the table can be traced back to what produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.provenance import provenance, run_start  # noqa: E402
from models.train import (  # noqa: E402
    CONFIGS,
    SLICES,
    load_dataset,
    reference_baselines_sliced,
    run,
)

METRIC_ORDER = [
    "routable_bacc", "routable_auc",
    "hamilton_bacc", "hamilton_auc",
    "failure_acc", "failure_f1",
    "cost_rho", "cost_rho_within", "cost_mae",
]

SLICE_TITLES = {
    "all": "all test designs",
    "precheck_decided": "precheck-decided -- a fatal exact obstruction, the label is free",
    "precheck_undecided": "precheck-undecided -- the label required the search  <<< the one that matters",
}


def fmt(v: float | None) -> str:
    return "  --  " if v is None or np.isnan(v) else f"{v:.3f}"


def mean_std(values) -> tuple[float, float]:
    a = np.array(list(values), dtype=float)
    if not a.size or np.isnan(a).all():
        return float("nan"), float("nan")
    return float(np.nanmean(a)), float(np.nanstd(a))


def print_metric_table(slice_name: str, n: int, refs: dict, summary: dict, configs) -> None:
    head = f"{'model':<16}" + "".join(f"{m:>16}" for m in METRIC_ORDER)
    print(f"\n=== {SLICE_TITLES[slice_name]}   (n = {n})")
    print(head)
    print("-" * len(head))
    for name, ref in refs.items():
        print(f"{name:<16}" + "".join(f"{fmt(ref.get(m)):>16}" for m in METRIC_ORDER))
    print("-" * len(head))
    for cfg in configs:
        row = f"{cfg.name:<16}"
        for m in METRIC_ORDER:
            mu, sd = summary[cfg.name][m]["mean"], summary[cfg.name][m]["std"]
            row += ("  --  ".rjust(16) if np.isnan(mu) else f"{mu:.3f}+-{sd:.2f}".rjust(16))
        print(row)


def print_class_table(slice_name: str, names, support, per_class, configs) -> None:
    if not names:
        return
    head = f"{'failure F1':<16}" + "".join(f"{n:>16}" for n in names)
    print(f"\n--- per-class failure F1, {slice_name}")
    print(head)
    print(f"{'test support':<16}" + "".join(f"{support.get(n, 0):>16}" for n in names))
    print("-" * len(head))
    for cfg in configs:
        print(f"{cfg.name:<16}" + "".join(fmt(per_class[cfg.name].get(n)).rjust(16) for n in names))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="data/designs_v1_graphs",
                    help="graph directory, or several comma-separated")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--split", default="random",
                    help="random | shape:<name> | size:<quantile> -- the last two hold out "
                         "a region of the design space instead of a random sample")
    ap.add_argument("--epochs", type=int, default=0, help="override config epochs")
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--out", type=Path, default=Path("results/baseline.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    graph_dirs = [d for d in str(args.graphs).split(",") if d]
    configs = list(CONFIGS)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        configs = [c for c in configs if c.name in wanted]
    if args.epochs:
        for c in configs:
            c.epochs = args.epochs

    # snapshot the code state now: a commit landing mid-run must not be recorded
    # as the code that produced this table
    start = run_start()
    t0 = time.perf_counter()
    results: dict[str, list[dict]] = {}
    refs = None
    ctx = None

    for seed in range(args.seeds):
        graphs, split, ctx = load_dataset(graph_dirs, split_seed=seed,
                                          split_spec=args.split)
        if seed == 0:
            print(f"{len(graphs)} graphs from {', '.join(graph_dirs)}   {split}")
            print(f"split: {ctx.split_info}")
            print(f"failure classes: {ctx.class_map.n} used ({', '.join(ctx.class_map.names)})"
                  + (f"; dropped as empty: {', '.join(ctx.class_map.dropped)}"
                     if ctx.class_map.dropped else ""))
        for cfg in configs:
            r = run(cfg, graphs, split, ctx, seed=seed, verbose=args.verbose)
            results.setdefault(cfg.name, []).append(r)
            if refs is None:
                refs = reference_baselines_sliced(r["pred"], ctx)
            und = r["test_sliced"]["precheck_undecided"]
            print(f"  seed {seed}  {cfg.name:<14} "
                  f"all: bacc={fmt(r['test']['routable_bacc'])} f1={fmt(r['test']['failure_f1'])}"
                  f"   undecided: bacc={fmt(und['routable_bacc'])} f1={fmt(und['failure_f1'])}"
                  f"   ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---------------------------------------------------------------- aggregate
    payload_slices: dict[str, dict] = {}
    for sl in SLICES:
        summary = {
            cfg.name: {
                m: dict(zip(("mean", "std"),
                            mean_std(r["test_sliced"][sl][m] for r in results[cfg.name])))
                for m in METRIC_ORDER
            }
            for cfg in configs
        }
        n = int(results[configs[0].name][0]["test_sliced"][sl]["n"])
        print_metric_table(sl, n, refs[sl], summary, configs)

        names = [c for c in ctx.class_map.names
                 if any(c in r["test_class_f1_sliced"][sl] for r in results[configs[0].name])]
        support = {}
        for name in names:
            got = [r["test_class_f1_sliced"][sl].get(name) for r in results[configs[0].name]]
            support[name] = int(np.mean([v[1] for v in got if v])) if any(got) else 0
        per_class = {
            cfg.name: {
                name: mean_std(v[0] for v in
                               (r["test_class_f1_sliced"][sl].get(name) for r in results[cfg.name])
                               if v)[0]
                for name in names
            }
            for cfg in configs
        }
        if sl != "precheck_decided":
            print_class_table(sl, names, support, per_class, configs)

        payload_slices[sl] = {
            "n": n,
            "reference": refs[sl],
            "results": summary,
            "class_f1": per_class,
            "class_support": support,
        }

    print()
    for cfg in configs:
        print(f"  {cfg.name:<14} {cfg.note}")

    # ------------------------------------------------------------------- write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": provenance(
            graph_dirs, **start,
            run={"seeds": args.seeds, "epochs": args.epochs or None,
                 "split_spec": args.split, "split_info": ctx.split_info,
                 "n_graphs": len(graphs), "split": {"train": len(split.train),
                                                    "val": len(split.val),
                                                    "test": len(split.test)}},
        ),
        "classes": {"used": ctx.class_map.names, "dropped": ctx.class_map.dropped},
        "configs": {c.name: {"model": c.model, "include_precheck": c.include_precheck,
                             "hidden": c.hidden, "layers": c.layers, "epochs": c.epochs,
                             "note": c.note} for c in configs},
        "n_params": {c.name: results[c.name][0]["n_params"] for c in configs},
        "slices": payload_slices,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    git = payload["provenance"]["git"]
    print(f"\nwrote {args.out}   ({time.perf_counter() - t0:.0f}s)")
    print(f"code: {git['describe']}"
          + ("   WARNING: uncommitted changes in the tree" if git["dirty"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
