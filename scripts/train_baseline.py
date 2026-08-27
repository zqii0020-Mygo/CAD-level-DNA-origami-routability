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
    "cost_rho", "cost_mae",
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

    for seed in range(args.seeds):
        graphs, split = load_dataset(str(args.graphs), split_seed=seed)
        if seed == 0:
            print(f"{len(graphs)} graphs   {split}")
        for cfg in configs:
            r = run(cfg, graphs, split, seed=seed, verbose=args.verbose)
            results.setdefault(cfg.name, []).append(r)
            if refs is None:
                refs = reference_baselines(r["pred"])
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "reference": refs,
        "configs": {c.name: {"model": c.model, "include_precheck": c.include_precheck,
                             "hidden": c.hidden, "layers": c.layers, "epochs": c.epochs,
                             "note": c.note} for c in configs},
        "results": summary,
        "n_params": {c.name: results[c.name][0]["n_params"] for c in configs},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}   ({time.perf_counter() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
