"""Train and compare the baseline surrogates.

    python scripts/train_baseline.py --graphs data/designs_v0_graphs --seeds 3

Reports test metrics for each configuration next to two reference points: the
majority-class predictor and the pipeline's own precheck verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.train import CONFIGS, load_dataset, reference_baselines, run  # noqa: E402

METRIC_ORDER = [
    "routable_bacc", "routable_auc",
    "hamilton_bacc", "hamilton_auc",
    "failure_acc", "failure_f1",
    "cost_rho", "cost_rho_within", "cost_mae",
]


def fmt(v: float) -> str:
    return "  --  " if v is None or np.isnan(v) else f"{v:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", type=Path, default=Path("data/designs_v0_graphs"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=0, help="override config epochs")
    ap.add_argument("--only", default=None, help="comma-separated config names")
    ap.add_argument("--out", type=Path, default=Path("data/baseline_results.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    configs = list(CONFIGS)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        configs = [c for c in configs if c.name in wanted]
    if args.epochs:
        for c in configs:
            c.epochs = args.epochs

    t0 = time.perf_counter()
    results: dict[str, list[dict]] = {}
    refs = None

    ctx = None
    for seed in range(args.seeds):
        graphs, split, ctx = load_dataset(str(args.graphs), split_seed=seed)
        if seed == 0:
            print(f"{len(graphs)} graphs   {split}")
            print(f"failure classes: {ctx.class_map.n} used "
                  f"({', '.join(ctx.class_map.names)})"
                  + (f"; dropped as empty: {', '.join(ctx.class_map.dropped)}"
                     if ctx.class_map.dropped else ""))
        for cfg in configs:
            r = run(cfg, graphs, split, ctx, seed=seed, verbose=args.verbose)
            results.setdefault(cfg.name, []).append(r)
            if refs is None:
                refs = reference_baselines(r["pred"], ctx)
            print(f"  seed {seed}  {cfg.name:<14} "
                  + "  ".join(f"{k.split('_')[-1]}={fmt(r['test'][k])}" for k in METRIC_ORDER[:6])
                  + f"   ({time.perf_counter() - t0:.0f}s)", flush=True)

    head = f"{'model':<16}" + "".join(f"{m:>16}" for m in METRIC_ORDER)
    print("\n" + head)
    print("-" * len(head))
    for name, ref in (refs or {}).items():
        print(f"{name:<16}" + "".join(f"{fmt(ref[m]):>16}" for m in METRIC_ORDER))
    print("-" * len(head))
    summary = {}
    for cfg in configs:
        runs = results[cfg.name]
        row = f"{cfg.name:<16}"
        summary[cfg.name] = {}
        for m in METRIC_ORDER:
            vals = np.array([r["test"][m] for r in runs], dtype=float)
            mu, sd = float(np.nanmean(vals)), float(np.nanstd(vals))
            summary[cfg.name][m] = {"mean": mu, "std": sd}
            row += f"{fmt(mu)}+-{sd:.2f}".rjust(16)
        print(row)
    print("-" * len(head))
    for cfg in configs:
        print(f"  {cfg.name:<14} {cfg.note}")

    # per-class F1: the thin classes are what the macro average hides
    names = list(ctx.class_map.names) if ctx else []
    support = {}
    for name in names:
        vals = [r["test_class_f1"].get(name) for r in results[configs[0].name]]
        support[name] = int(np.mean([v[1] for v in vals if v])) if any(vals) else 0
    chead = f"{'failure F1':<16}" + "".join(f"{n:>16}" for n in names)
    print("\n" + chead)
    print(f"{'test support':<16}" + "".join(f"{support[n]:>16}" for n in names))
    print("-" * len(chead))
    per_class = {}
    for cfg in configs:
        row = f"{cfg.name:<16}"
        per_class[cfg.name] = {}
        for n in names:
            vals = [r["test_class_f1"].get(n) for r in results[cfg.name]]
            vals = [v[0] for v in vals if v]
            mu = float(np.mean(vals)) if vals else float("nan")
            per_class[cfg.name][n] = mu
            row += fmt(mu).rjust(16)
        print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_graphs": len(graphs),
        "classes": {"used": ctx.class_map.names, "dropped": ctx.class_map.dropped} if ctx else {},
        "reference": refs,
        "configs": {c.name: {"model": c.model, "include_precheck": c.include_precheck,
                             "hidden": c.hidden, "layers": c.layers, "epochs": c.epochs,
                             "note": c.note} for c in configs},
        "results": summary,
        "class_f1": per_class,
        "class_support": support,
        "n_params": {c.name: results[c.name][0]["n_params"] for c in configs},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}   ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
