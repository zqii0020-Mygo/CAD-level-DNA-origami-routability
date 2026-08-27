"""Where a results file came from: code revision, data, environment.

A metrics table is only evidence if you can say which code and which data
produced it.  Every results JSON carries this block so `git log` can answer
that question months later, and so a table produced from an uncommitted working
tree is visibly marked as such (`git_dirty`).

The dataset digests are the strong part: a graph directory is identified by the
SHA-1 of its manifest, and the design directory it came from by the SHA-1 of
its `labels.csv`.  Those change if a single label changes, which a directory
name does not.
"""

from __future__ import annotations

import csv
import hashlib
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


def _run(*args: str) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_state() -> dict[str, Any]:
    """Commit, branch, and whether the tree had uncommitted changes."""
    head = _run("git", "rev-parse", "HEAD")
    status = _run("git", "status", "--porcelain")
    return {
        "commit": head,
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _run("git", "describe", "--always", "--dirty"),
        # None means git did not answer, which is not the same as "clean"
        "dirty": None if status is None else bool(status),
    }


def file_digest(path: str | Path, n_bytes: int = 1 << 20) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(n_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_range(index_csv: Path) -> dict[str, Any]:
    """Design seed span, read from a generator index."""
    try:
        with index_csv.open(newline="", encoding="utf-8") as fh:
            seeds = [int(r["seed"]) for r in csv.DictReader(fh) if r.get("seed", "").strip()]
    except (OSError, ValueError, KeyError):
        return {}
    return {"seed_min": min(seeds), "seed_max": max(seeds), "n_seeds": len(seeds)} if seeds else {}


def dataset_provenance(graphs_dir: str | Path) -> dict[str, Any]:
    """Identify one graph directory and the design batch behind it."""
    graphs_dir = Path(graphs_dir)
    manifest = graphs_dir / "graphs_index.csv"
    info: dict[str, Any] = {
        "graphs_dir": graphs_dir.as_posix(),
        "n_graphs": len(list(graphs_dir.glob("*.npz"))),
        "graphs_index_sha1": file_digest(manifest),
    }
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        info["sampling"] = dict(Counter(r.get("sampling", "iid") for r in rows))
        info["shapes"] = dict(Counter(r.get("shape", "?") for r in rows))

    # `<name>_graphs` is written by build_graphs.py from `<name>`
    name = graphs_dir.name
    designs = graphs_dir.with_name(name[:-7]) if name.endswith("_graphs") else None
    if designs is not None and designs.is_dir():
        info["designs_dir"] = designs.as_posix()
        info["labels_sha1"] = file_digest(designs / "labels.csv")
        info.update(_seed_range(designs / "index.csv"))
    return info


def environment() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for mod in ("numpy", "networkx", "torch", "torch_geometric"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:                      # not installed / no __version__
            versions[mod] = None
    return versions


def provenance(graph_dirs: Sequence[str | Path], **extra: Any) -> dict[str, Any]:
    """The full block: when, which code, which data, which command, plus `extra`."""
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "environment": environment(),
        "datasets": [dataset_provenance(d) for d in graph_dirs],
        **extra,
    }
