"""JSON (de)serialisation for designs.

The on-disk format is a flat record of the five node/edge tables, which maps
one-to-one onto the heterogeneous graph the surrogate will consume.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .model import (
    Adjacency,
    Bundle,
    CandidateCrossover,
    Cylinder,
    Design,
    Feature,
    Mate,
)

FORMAT_VERSION = 1


def design_to_dict(d: Design) -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "id": d.id,
        "lattice": d.lattice,
        "params": d.params,
        "features": [asdict(x) for x in d.features],
        "bundles": [asdict(x) for x in d.bundles],
        "cylinders": [asdict(x) for x in d.cylinders],
        "adjacencies": [asdict(x) for x in d.adjacencies],
        "crossovers": [asdict(x) for x in d.crossovers],
        "mates": [asdict(x) for x in d.mates],
        "unpaired_ends": [list(e) for e in d.unpaired_ends],
        "notes": d.notes,
        "summary": d.summary(),
    }


def design_from_dict(o: dict[str, Any]) -> Design:
    d = Design(id=o["id"], params=o["params"], lattice=o["lattice"])
    d.features = [Feature(**x) for x in o["features"]]
    for f in d.features:
        f.p0, f.p1 = tuple(f.p0), tuple(f.p1)
    d.bundles = [Bundle(**x) for x in o["bundles"]]
    for b in d.bundles:
        b.cross_section = [tuple(s) for s in b.cross_section]
        b.origin, b.axis = tuple(b.origin), tuple(b.axis)
        b.frame_u, b.frame_v = tuple(b.frame_u), tuple(b.frame_v)
    d.cylinders = [Cylinder(**x) for x in o["cylinders"]]
    for c in d.cylinders:
        c.start_xyz, c.end_xyz, c.axis = tuple(c.start_xyz), tuple(c.end_xyz), tuple(c.axis)
    d.adjacencies = [Adjacency(**x) for x in o["adjacencies"]]
    d.crossovers = [CandidateCrossover(**x) for x in o["crossovers"]]
    for x in d.crossovers:
        x.xyz = tuple(x.xyz)
    d.mates = [Mate(**x) for x in o["mates"]]
    d.unpaired_ends = [(int(a), str(b)) for a, b in o.get("unpaired_ends", [])]
    d.notes = o.get("notes", {})
    return d


def save_design(d: Design, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(design_to_dict(d), separators=(",", ":"))
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(blob)
    else:
        path.write_text(blob, encoding="utf-8")
    return path


def load_design(path: str | Path) -> Design:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return design_from_dict(json.load(fh))
    return design_from_dict(json.loads(path.read_text(encoding="utf-8")))
