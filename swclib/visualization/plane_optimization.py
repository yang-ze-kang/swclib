"""Deterministic local plane search with an exact visible-center constraint."""

import math

import numpy as np

from .hemisphere import plane_parameters


def balance_score(counts):
    """Below a difference of five, crossings take priority over balance."""
    difference = abs(counts["left_only"] - counts["right_only"])
    if difference < 5:
        return 0, counts["cross"], difference
    return 1, difference, counts["cross"]


def optimize_plane(parameters, center, max_angle_deg, evaluate):
    """Search a spherical cap, then refine its four best sampled directions.

    ``evaluate`` receives a batch of plain plane parameters and returns count
    dictionaries in the same order. Each search level can therefore read
    neurons once through the bounded reader pool, rather than once per angle.
    This is a finite local search, not a proof of a global optimum.
    """
    original = plane_parameters(parameters)
    try:
        center = tuple(float(v) for v in center)
        limit = float(max_angle_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("中心必须是三个有限 XYZ 数值，最大调整角度须在 0–90° 内") from exc
    if len(center) != 3 or not all(math.isfinite(v) for v in center):
        raise ValueError("中心必须是三个有限 XYZ 数值")
    if not math.isfinite(limit) or not 0 <= limit <= 90:
        raise ValueError("最大调整角度须在 0–90° 内")

    normal = np.asarray(original["normal_xyz"])
    reference = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, reference)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    candidates, seen = [], set()

    def add(x, y):
        radius = math.hypot(x, y)
        if radius > 1 + 1e-12:
            return
        if limit == 0:
            x = y = radius = 0.0
        key = (round(x, 12), round(y, 12))
        if key in seen:
            return
        seen.add(key)
        if radius == 0:
            a, e = original["azimuth_deg"], original["elevation_deg"]
        else:
            angle = math.radians(min(radius, 1) * limit)
            direction = math.cos(angle) * normal + math.sin(angle) * (x*u + y*v) / radius
            a = math.degrees(math.atan2(direction[1], direction[0])) % 360
            e = math.degrees(math.atan2(direction[2], math.hypot(direction[0], direction[1])))
        p = dict(azimuth_deg=a, elevation_deg=e, offset=0)
        n = plane_parameters(p)["normal_xyz"]
        p["offset"] = math.fsum(n_i*c_i for n_i, c_i in zip(n, center))
        candidates.append(dict(x=x, y=y, plane=p, angle=min(radius, 1)*limit))

    add(0, 0)  # Always retain the original orientation passing through center.
    if limit:
        for i in range(-6, 7):
            for j in range(-6, 7):
                add(i/6, j/6)
        for i in range(48):
            angle = 2*math.pi*i/48
            add(math.cos(angle), math.sin(angle))

    old_plane = {key: original[key] for key in ("azimuth_deg", "elevation_deg", "offset")}
    first = evaluate([old_plane] + [c["plane"] for c in candidates])
    before = first[0]
    for candidate, counts in zip(candidates, first[1:]):
        candidate["counts"] = counts

    def score(candidate):
        return (*balance_score(candidate["counts"]), candidate["angle"])

    if limit:
        step = 1/6
        for _ in range(3):
            step /= 4
            seeds = sorted(candidates, key=score)[:4]
            start = len(candidates)
            for seed in seeds:
                for i in range(-2, 3):
                    for j in range(-2, 3):
                        add(seed["x"] + i*step, seed["y"] + j*step)
            new = candidates[start:]
            if not new:
                break
            for candidate, counts in zip(new, evaluate([c["plane"] for c in new])):
                candidate["counts"] = counts

    ranked = sorted(candidates, key=score)
    selected = []
    separation = math.radians(min(2, limit/3))
    for candidate in ranked:
        n = np.asarray(plane_parameters(candidate["plane"])["normal_xyz"])
        if any(math.acos(float(np.clip(n @ other["normal"], -1, 1))) < separation for other in selected):
            continue
        selected.append(dict(candidate=candidate, normal=n))
        if len(selected) == 3:
            break
    suggestions = [dict(plane=plane_parameters(item["candidate"]["plane"]),
                        counts=item["candidate"]["counts"], angle_change_deg=item["candidate"]["angle"])
                   for item in selected]
    best = ranked[0]
    return dict(plane=plane_parameters(best["plane"]), counts=best["counts"],
                before_counts=before, centered_counts=candidates[0]["counts"],
                center_xyz=list(center), max_angle_deg=limit,
                angle_change_deg=best["angle"], evaluations=len(candidates)+1,
                balance_threshold=5, suggestions=suggestions)
