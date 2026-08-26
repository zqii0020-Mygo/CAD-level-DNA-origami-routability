"""Cylinder adjacency and candidate-crossover enumeration.

Adjacency has two flavours:

  intra   two helices of the same bundle sitting on neighbouring lattice sites
          and overlapping in bp.  Their bp coordinates are shared, so crossover
          positions are exactly the lattice phase rule of `lattice.py`.

  inter   two helices of different bundles that end up parallel and in contact.
          They have no shared bp frame, so we record the contact (it is a real
          feature of the design graph) but do not enumerate crossovers for it
          unless `params.inter_bundle_crossovers` is set, in which case the
          overlap is phased off the closest approach point.
"""

from __future__ import annotations

import itertools

import numpy as np

from .lattice import RISE_PER_BP, Lattice
from .model import Adjacency, CandidateCrossover, Cylinder, Design

PARALLEL_COS = 0.98      # |cos angle| above which two helices count as parallel
CONTACT_SLACK = 1.30     # multiple of the lattice inter-helix distance


def _segment_distance(p0, p1, q0, q1) -> float:
    """Shortest distance between two 3D segments."""
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a, b, c = u @ u, u @ v, v @ v
    d, e = u @ w, v @ w
    den = a * c - b * b
    if den < 1e-12:
        sc = 0.0
        tc = (e / c) if c > 1e-12 else 0.0
    else:
        sc = (b * e - c * d) / den
        tc = (a * e - b * d) / den
    sc = min(1.0, max(0.0, sc))
    tc = min(1.0, max(0.0, tc))
    return float(np.linalg.norm(w + sc * u - tc * v))


def _bp_overlap(a: Cylinder, b: Cylinder) -> int:
    return max(0, min(a.bp_end, b.bp_end) - max(a.bp_start, b.bp_start))


def build_adjacency(design: Design, lat: Lattice) -> list[Adjacency]:
    out: list[Adjacency] = []
    by_bundle: dict[int, list[Cylinder]] = {}
    for c in design.cylinders:
        by_bundle.setdefault(c.bundle_id, []).append(c)

    # ---- intra-bundle: lattice neighbours with a bp overlap
    for bid, cyls in sorted(by_bundle.items()):
        site_to_cyl = {(c.row, c.col): c for c in cyls}
        for site, ca in sorted(site_to_cyl.items()):
            for nb in lat.neighbors(*site):
                cb = site_to_cyl.get(nb)
                if cb is None or cb.id <= ca.id:
                    continue
                ov = _bp_overlap(ca, cb)
                if ov <= 0:
                    continue
                out.append(
                    Adjacency(
                        id=len(out), cyl_a=ca.id, cyl_b=cb.id, kind="intra",
                        dist_nm=lat.interhelix, overlap_bp=ov,
                        dir_deg=lat.direction_deg(site, nb),
                    )
                )

    # ---- inter-bundle: parallel helices in physical contact
    cut = lat.interhelix * CONTACT_SLACK
    for ca, cb in itertools.combinations(design.cylinders, 2):
        if ca.bundle_id == cb.bundle_id:
            continue
        aa = np.asarray(ca.axis)
        ab = np.asarray(cb.axis)
        if abs(float(aa @ ab)) < PARALLEL_COS:
            continue
        p0, p1 = np.asarray(ca.start_xyz), np.asarray(ca.end_xyz)
        q0, q1 = np.asarray(cb.start_xyz), np.asarray(cb.end_xyz)
        dist = _segment_distance(p0, p1, q0, q1)
        if dist > cut:
            continue
        # overlap length measured along ca's axis
        ta = sorted((0.0, float((p1 - p0) @ aa)))
        tb = sorted((float((q0 - p0) @ aa), float((q1 - p0) @ aa)))
        ov_nm = max(0.0, min(ta[1], tb[1]) - max(ta[0], tb[0]))
        ov_bp = int(ov_nm / RISE_PER_BP)
        if ov_bp <= 0:
            continue
        out.append(
            Adjacency(
                id=len(out), cyl_a=ca.id, cyl_b=cb.id, kind="inter",
                dist_nm=dist, overlap_bp=ov_bp, dir_deg=None,
            )
        )
    return out


def build_crossovers(design: Design, lat: Lattice) -> list[CandidateCrossover]:
    """Enumerate every geometrically allowed crossover site of the design."""
    margin = int(design.params.get("crossover_end_margin_bp", 0))
    allow_inter = bool(design.params.get("inter_bundle_crossovers", False))
    cyl = {c.id: c for c in design.cylinders}
    bundles = {b.id: b for b in design.bundles}
    out: list[CandidateCrossover] = []

    for adj in design.adjacencies:
        ca, cb = cyl[adj.cyl_a], cyl[adj.cyl_b]
        if adj.kind == "inter" and not allow_inter:
            continue
        if adj.kind == "inter":
            # No shared bp frame: phase the overlap off cb's start, period `step`.
            lo = max(ca.bp_start, 0)
            hi = lo + adj.overlap_bp
            idx = range(lo + margin, hi - margin, lat.step)
        else:
            lo = max(ca.bp_start, cb.bp_start)
            hi = min(ca.bp_end, cb.bp_end)
            idx = lat.crossover_indices((ca.row, ca.col), (cb.row, cb.col), lo, hi, margin)

        b = bundles[ca.bundle_id]
        origin = np.asarray(b.origin)
        axis = np.asarray(b.axis)
        u, v = np.asarray(b.frame_u), np.asarray(b.frame_v)
        xa, ya = lat.position(ca.row, ca.col)
        xb, yb = lat.position(cb.row, cb.col)
        mid_u, mid_v = (xa + xb) / 2.0, (ya + yb) / 2.0

        for i in idx:
            d2end = min(
                i - ca.bp_start, ca.bp_end - i,
                i - cb.bp_start, cb.bp_end - i,
            )
            pos = origin + mid_u * u + mid_v * v + axis * (i * RISE_PER_BP)
            out.append(
                CandidateCrossover(
                    id=len(out), cyl_a=ca.id, cyl_b=cb.id, adjacency_id=adj.id,
                    bp_index=int(i),
                    xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
                    dist_to_end_bp=int(d2end),
                )
            )
    return out
