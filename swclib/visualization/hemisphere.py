"""Global-coordinate hemisphere statistics for a user-positioned plane."""

import math
from pathlib import Path

import numpy as np

from swclib.data.swc import Swc


DESCRIPTION = ("Per-neuron hemisphere stats for an arbitrary 3D plane. "
               "Positive signed side along the plane normal is labeled as right hemisphere, "
               "negative signed side as left hemisphere. right_length/left_length follow the same rule.")


def plane_parameters(value):
    if not isinstance(value, dict) or set(value) != {"azimuth_deg", "elevation_deg", "offset"}:
        raise ValueError("平面需要 azimuth_deg、elevation_deg 和 offset 三个参数")
    a, e, d = (float(value[key]) for key in ("azimuth_deg", "elevation_deg", "offset"))
    if not all(math.isfinite(v) for v in (a, e, d)) or not -90 <= e <= 90:
        raise ValueError("平面参数必须有限，仰角范围为 -90° 到 90°")
    a %= 360
    azimuth, elevation = math.radians(a), math.radians(e)
    normal = [math.cos(elevation)*math.cos(azimuth), math.cos(elevation)*math.sin(azimuth), math.sin(elevation)]
    return dict(type="arbitrary_3d_plane", azimuth_deg=a, elevation_deg=e,
                normal_xyz=normal, offset=d, point_xyz=[v*d for v in normal])


def _read_geometry(path):
    swc = Swc(str(path))
    if not swc.nodes:
        raise ValueError(f"{Path(path).name}：SWC 没有节点")
    coords = {}
    for nid, node in swc.nodes.items():
        xyz = [float(node[key]) for key in ("x", "y", "z")]
        if not all(math.isfinite(v) for v in xyz):
            raise ValueError(f"{Path(path).name}：节点 {nid} 坐标不是有限数值")
        coords[nid] = xyz
    return swc, coords


def _valid_edges(swc, coords, path):
    for nid, node in swc.nodes.items():
        parent = node["parent"]
        if parent == -1:
            continue
        if parent not in coords or parent == nid:
            raise ValueError(f"{Path(path).name}：节点 {nid} 的父节点无效")
        yield nid, parent


def neuron_statistics(path, parameters, *, include_soma=True):
    """Split each edge at its exact plane intersection, without resampling."""
    plane = plane_parameters(parameters)
    normal, offset = plane["normal_xyz"], plane["offset"]
    swc, coords = _read_geometry(path)
    distances = {nid: math.fsum(n*x for n, x in zip(normal, xyz)) - offset
                 for nid, xyz in coords.items()}
    soma = None
    if include_soma:
        candidates = [nid for nid, node in swc.nodes.items() if node["type"] == 1] or swc.get_roots(return_coords=False)
        if len(candidates) != 1:
            raise ValueError(f"{Path(path).name}：存在 {len(candidates)} 个胞体候选，无法为单个 soma 字段确定唯一位置")
        nid = candidates[0]
        x, y, z = coords[nid]
        soma = dict(x=x, y=y, z=z, side="right" if distances[nid] >= 0 else "left", signed_distance=distances[nid])
    left, right = [], []
    for nid, parent in _valid_edges(swc, coords, path):
        length = math.dist(coords[nid], coords[parent])
        a, b = distances[nid], distances[parent]
        if a >= 0 and b >= 0:
            right.append(length)
        elif a <= 0 and b <= 0:
            left.append(length)
        else:
            denominator = abs(a)+abs(b)
            right.append(length*max(a,b)/denominator)
            left.append(length*abs(min(a,b))/denominator)
    left_length, right_length = math.fsum(left), math.fsum(right)
    # The plane itself belongs to the non-negative (right) side. For a neuron
    # without nonzero edges, use its node positions for the hemisphere label.
    sides = ({"left"} if left_length > 0 else set()) | ({"right"} if right_length > 0 else set())
    if not sides:
        sides = {"right" if value >= 0 else "left" for value in distances.values()}
    return dict(swc_name=Path(path).name,
                soma=soma,
                right_length=right_length, left_length=left_length, total_length=right_length+left_length,
                hemisphere="cross" if len(sides) > 1 else next(iter(sides)))


def _plane_distances(points, normals, offsets):
    """Vectorized distances with the reference's fsum at uncertain boundaries."""
    x = points[:, 0, None] * normals[:, 0]
    y = points[:, 1, None] * normals[:, 1]
    z = points[:, 2, None] * normals[:, 2]
    distances = x + y + z - offsets
    error = 8 * np.finfo(float).eps * (np.abs(x) + np.abs(y) + np.abs(z) + np.abs(offsets))
    rows, columns = np.nonzero(np.abs(distances) < error)
    for row, column in zip(rows, columns):
        distances[row, column] = math.fsum(
            float(n) * float(v) for n, v in zip(normals[column], points[row])
        ) - offsets[column]
    return distances


def neuron_plane_classes(path, parameters):
    """Classify many planes after one read, returning left=1/right=2/cross=3.

    A positive-length edge contributes left length if either endpoint is
    negative, and right length if either is positive or both are on the plane.
    This preserves neuron_statistics' length classification without computing
    every intersection length. Zero-length neurons use the same node fallback.
    Both edge and plane batches are bounded; source arrays stay in one worker.
    """
    planes = [plane_parameters(value) for value in parameters]
    if not planes:
        return np.empty(0, dtype=np.uint8)
    points, edges = _geometry_arrays(path)
    return _geometry_plane_classes(points, edges, planes)


def _geometry_arrays(path, swc=None):
    if swc is None:
        swc = Swc(str(path))
    if not swc.nodes:
        raise ValueError(f"{Path(path).name}：SWC 没有节点")
    points = swc.get_coords()
    if not np.isfinite(points).all():
        raise ValueError(f"{Path(path).name}：节点坐标不是有限数值")
    indices = {nid: i for i, nid in enumerate(swc.nodes)}
    edges = np.asarray([(indices[nid], indices[parent])
                        for nid, parent in _valid_edges(swc, swc.nodes, path)], dtype=np.int64).reshape(-1, 2)
    if len(edges):
        edges = edges[(points[edges[:, 0]] != points[edges[:, 1]]).any(axis=1)]
    return points, edges


def _geometry_plane_classes(points, edges, planes):
    classes = np.zeros(len(planes), dtype=np.uint8)
    for first in range(0, len(planes), 32):
        batch = planes[first:first + 32]
        normals = np.asarray([plane["normal_xyz"] for plane in batch])
        offsets = np.asarray([plane["offset"] for plane in batch])
        left, right = np.zeros(len(batch), dtype=bool), np.zeros(len(batch), dtype=bool)
        if len(edges):
            for start in range(0, len(edges), 2048):
                chunk = edges[start:start + 2048]
                a = _plane_distances(points[chunk[:, 0]], normals, offsets)
                b = _plane_distances(points[chunk[:, 1]], normals, offsets)
                left |= ((a < 0) | (b < 0)).any(axis=0)
                right |= ((a > 0) | (b > 0) | ((a == 0) & (b == 0))).any(axis=0)
                if (left & right).all():
                    break
        else:
            for start in range(0, len(points), 2048):
                distances = _plane_distances(points[start:start + 2048], normals, offsets)
                left |= (distances < 0).any(axis=0)
                right |= (distances >= 0).any(axis=0)
        classes[first:first + len(batch)] = left.astype(np.uint8) + 2 * right.astype(np.uint8)
    return classes


def neuron_plane_summary(path, *, swc=None):
    """Bounded geometry for exact broad-phase classification, never an estimate.

    Small neurons retain their geometry. Larger neurons retain a bounding box
    and a verified convex support envelope, or bounded groups of boxes if the
    geometry is degenerate. Ambiguous candidates require a full geometry read.
    """
    points, edges = _geometry_arrays(path, swc)
    return _summarize_geometry(points, edges)


def _summarize_geometry(points, edges, detail=64):
    metadata = dict(node_count=len(points), source_bounds=np.array([points.min(axis=0), points.max(axis=0)]))
    if len(points) <= 512:
        return dict(points=points, edges=edges, **metadata)
    active = points[np.unique(edges)] if len(edges) else points
    bounds = np.array([active.min(axis=0), active.max(axis=0)])
    envelope = _support_envelope(active)
    if envelope is not None:
        witnesses, error = envelope
        return dict(bounds=bounds, witnesses=witnesses, support_error=error,
                    has_edges=bool(len(edges)), **metadata)
    blocks = np.array_split(active, min(detail, len(active)))
    boxes = np.array([[block.min(axis=0), block.max(axis=0)] for block in blocks])
    extremes = np.concatenate([block[np.r_[block.argmin(axis=0), block.argmax(axis=0)]] for block in blocks])
    witnesses = np.unique(np.concatenate([extremes, active[np.linspace(0, len(active)-1, min(128, len(active)), dtype=int)]]), axis=0)
    return dict(bounds=bounds, boxes=boxes, witnesses=witnesses, detail=detail, has_edges=bool(len(edges)), **metadata)


def _support_envelope(points):
    """Bound every projection by <=512 vertices plus a verified error radius.

    Do not rely on Qhull's floating-point containment alone. For every source
    point verify a barycentric reconstruction from four retained vertices.
    Reconstruction residual + negative-weight mass * diameter bounds how far
    ANY unit-normal projection may lie outside the retained vertex range.
    Unverifiable or degenerate hulls fall back to enclosing boxes.
    """
    if len(points) <= 512:
        return points, 0.0
    from scipy.spatial import ConvexHull, Delaunay, QhullError
    origin = (points.min(axis=0) + points.max(axis=0)) / 2
    local = points - origin
    try:
        hull = ConvexHull(local)
        if len(hull.vertices) > 512:
            return None
        support = local[hull.vertices]
        tetrahedra = Delaunay(support)
    except QhullError:
        return None
    diameter = np.linalg.norm(np.ptp(local, axis=0))
    radius = 0.0
    for start in range(0, len(local), 4096):
        chunk = local[start:start+4096]
        simplex = tetrahedra.find_simplex(chunk, tol=1e-9)
        if (simplex < 0).any():
            return None
        transform = tetrahedra.transform[simplex]
        weights = np.einsum('nij,nj->ni', transform[:, :3], chunk-transform[:, 3])
        weights = np.column_stack([weights, 1-weights.sum(axis=1)])
        vertices = support[tetrahedra.simplices[simplex]]
        reconstructed = (weights[:, :, None]*vertices).sum(axis=1)
        residual = np.linalg.norm(chunk-reconstructed, axis=1)
        bound = residual + np.maximum(-weights, 0).sum(axis=1)*diameter
        if not np.isfinite(bound).all():
            return None
        radius = max(radius, float(bound.max()))
    # Cover centering, barycentric summation and projection roundoff as well.
    radius += 64*np.finfo(float).eps*max(1, float(np.abs(points).max()), float(diameter))
    return points[hull.vertices], radius


def neuron_refined_classes(path, parameters, detail):
    """Use a necessary exact read to tighten the summary for subsequent angles."""
    points, edges = _geometry_arrays(path)
    classes = _geometry_plane_classes(points, edges, [plane_parameters(p) for p in parameters])
    return classes, _summarize_geometry(points, edges, detail)


def summary_plane_classes(summary, parameters, *, prepared=None):
    """Return exact classes where proven, zero where a full read is required."""
    if prepared is None:
        planes = [plane_parameters(value) for value in parameters]
        normals = np.asarray([plane["normal_xyz"] for plane in planes])
        offsets = np.asarray([plane["offset"] for plane in planes])
    else:
        planes, normals, offsets = prepared
    if "points" in summary:
        return _geometry_plane_classes(summary["points"], summary["edges"], planes)
    bounds = summary["bounds"]
    a, b = bounds[0] * normals, bounds[1] * normals
    lo, hi = np.minimum(a, b).sum(axis=1) - offsets, np.maximum(a, b).sum(axis=1) - offsets
    error = 8 * np.finfo(float).eps * (np.abs(a).sum(axis=1) + np.abs(b).sum(axis=1) + np.abs(offsets))
    classes = np.zeros(len(planes), dtype=np.uint8)
    classes[lo > error] = 2
    classes[hi < -error] = 1
    # Only unresolved planes need witness projections, in bounded batches.
    unresolved = np.flatnonzero(classes == 0)
    for start in range(0, len(unresolved), 32):
        indices = unresolved[start:start+32]
        distances = _plane_distances(summary["witnesses"], normals[indices], offsets[indices])
        left = (distances < 0).any(axis=0)
        right = (distances > 0 if summary["has_edges"] else distances >= 0).any(axis=0)
        classes[indices[left & right]] = 3
        if "support_error" in summary:
            margin = summary["support_error"] + error[indices]
            classes[indices[distances.min(axis=0) > margin]] = 2
            classes[indices[distances.max(axis=0) < -margin]] = 1
            continue
        indices = indices[classes[indices] == 0]
        if not len(indices):
            continue
        boxes = summary["boxes"]
        a = boxes[:, 0, None, :] * normals[indices]
        b = boxes[:, 1, None, :] * normals[indices]
        lo = np.minimum(a, b).sum(axis=2) - offsets[indices]
        hi = np.maximum(a, b).sum(axis=2) - offsets[indices]
        error = 8 * np.finfo(float).eps * (np.abs(a).sum(axis=2) + np.abs(b).sum(axis=2) + np.abs(offsets[indices]))
        classes[indices[(lo > error).all(axis=0)]] = 2
        classes[indices[(hi < -error).all(axis=0)]] = 1
    return classes


def hemisphere_counts(neurons):
    """Mutually exclusive classes plus each side's total including crossings."""
    counts = dict(left_only=0, right_only=0, cross=0)
    keys = {"left": "left_only", "right": "right_only", "cross": "cross"}
    for neuron in neurons:
        counts[keys[neuron["hemisphere"]]] += 1
    return _count_totals(**counts)


def _count_totals(left_only, right_only, cross):
    return dict(left_only=left_only, right_only=right_only, cross=cross,
                total=left_only+right_only+cross, left=left_only+cross, right=right_only+cross)


def candidate_counts(classes):
    """Aggregate the small per-neuron class arrays returned by workers."""
    classes = np.asarray(classes, dtype=np.uint8)
    left = (classes == 1).sum(axis=0)
    right = (classes == 2).sum(axis=0)
    cross = (classes == 3).sum(axis=0)
    return [_count_totals(int(a), int(b), int(c)) for a, b, c in zip(left, right, cross)]
