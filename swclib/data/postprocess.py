"""SWC forest post-processing: overlap-based branch de-duplication and
short-branch pruning.

These passes operate on a neuron forest and are exposed two ways:

* **Array level** (``_post_merge_forest_by_overlap``,
  ``_post_prune_short_terminal_branches``) — operate on the
  ``(points, parents, node_types)`` triple, where ``points`` is an ``(N, 3)``
  ``xyz`` float array, ``parents`` is an ``(N,)`` int array of **0-based parent
  indices** (root = ``-1``), and ``node_types`` is an ``(N,)`` int array. These
  keep the exact signatures/stats expected by existing array-based pipelines.

* **``Swc`` level** (``merge_overlapping_branches``,
  ``prune_short_terminal_branches``) — operate on a :class:`swclib.data.swc.Swc`
  and return ``(Swc, stats)``. Internally they convert to arrays via
  ``swc_to_arrays`` / ``swc_from_arrays``.

Overlap detection reuses :class:`swclib.data.swc_fiber.SwcFiber` fiber-overlap
metrics. Coordinates are native ``(x, y, z)`` throughout this module.
"""
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Fiber + graph helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _FiberPointNode:
    coord: np.ndarray


def _make_overlap_fiber(points_xyz: np.ndarray) -> Any:
    from swclib.data.swc_fiber import SwcFiber

    fiber = SwcFiber()
    for point in points_xyz:
        fiber.append(_FiberPointNode(coord=np.asarray(point, dtype=np.float32)))
    return fiber


def _forest_connected_components(parents: np.ndarray) -> List[List[int]]:
    num_nodes = int(len(parents))
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    for child, parent in enumerate(parents.tolist()):
        parent = int(parent)
        if parent < 0 or parent >= num_nodes:
            continue
        adjacency[int(child)].append(parent)
        adjacency[parent].append(int(child))

    components: List[List[int]] = []
    visited = np.zeros((num_nodes,), dtype=bool)
    for start in range(num_nodes):
        if bool(visited[start]):
            continue
        queue: "deque[int]" = deque([start])
        visited[start] = True
        component: List[int] = []
        while queue:
            node_idx = int(queue.popleft())
            component.append(node_idx)
            for neighbor in adjacency[node_idx]:
                neighbor = int(neighbor)
                if bool(visited[neighbor]):
                    continue
                visited[neighbor] = True
                queue.append(neighbor)
        components.append(sorted(component))
    return components


def _neighbors_from_parents(parents: np.ndarray) -> List[List[int]]:
    num_nodes = int(len(parents))
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    for child, parent in enumerate(parents.tolist()):
        parent = int(parent)
        if parent < 0 or parent >= num_nodes:
            continue
        neighbors[int(child)].append(parent)
        neighbors[parent].append(int(child))
    return neighbors


def _component_path_indices(
    parents: np.ndarray,
    component_indices: Sequence[int],
) -> List[List[int]]:
    component_set = {int(idx) for idx in component_indices}
    if not component_set:
        return []

    children: Dict[int, List[int]] = {idx: [] for idx in component_set}
    for idx in component_set:
        parent = int(parents[idx])
        if parent in component_set:
            children[parent].append(idx)

    roots = [
        idx
        for idx in sorted(component_set)
        if int(parents[idx]) not in component_set
    ]
    if not roots:
        roots = [min(component_set)]

    paths: List[List[int]] = []

    def dfs(node_idx: int, path: List[int]) -> None:
        child_indices = sorted(children.get(int(node_idx), []))
        if not child_indices:
            if len(path) >= 2:
                paths.append(list(path))
            return
        for child_idx in child_indices:
            dfs(int(child_idx), path + [int(child_idx)])

    for root in roots:
        dfs(int(root), [int(root)])
    return paths


# ---------------------------------------------------------------------------
# Overlap-based matching / merging primitives
# ---------------------------------------------------------------------------
def _component_overlap_fibers(
    points: np.ndarray,
    parents: np.ndarray,
    component_indices: Sequence[int],
) -> List[Any]:
    fibers = []
    for path in _component_path_indices(parents, component_indices):
        fibers.append(_make_overlap_fiber(points[np.asarray(path, dtype=np.int64)]))
    return fibers


def _best_overlap_match(
    source_fibers: Sequence[Any],
    target_fibers: Sequence[Any],
    dist_sample: float,
    dist_threshold: float,
    min_overlap_length: float,
    min_overlap_ratio: float,
    min_iou: float,
) -> Optional[Dict[str, float]]:
    matches = _accepted_overlap_matches(
        source_fibers=source_fibers,
        target_fibers=target_fibers,
        dist_sample=dist_sample,
        dist_threshold=dist_threshold,
        min_overlap_length=min_overlap_length,
        min_overlap_ratio=min_overlap_ratio,
        min_iou=min_iou,
    )
    return matches[0] if matches else None


def _accepted_overlap_matches(
    source_fibers: Sequence[Any],
    target_fibers: Sequence[Any],
    dist_sample: float,
    dist_threshold: float,
    min_overlap_length: float,
    min_overlap_ratio: float,
    min_iou: float,
) -> List[Dict[str, float]]:
    matches: List[Dict[str, float]] = []
    for source_fiber_index, source_fiber in enumerate(source_fibers):
        source_length = max(float(source_fiber.length), 1e-7)
        for target_fiber_index, target_fiber in enumerate(target_fibers):
            target_length = max(float(target_fiber.length), 1e-7)
            overlap_length = float(
                source_fiber.get_overlap_length_with(
                    target_fiber,
                    dist_sample=float(dist_sample),
                    dist_threshold=float(dist_threshold),
                )
            )
            if overlap_length <= 0.0:
                continue
            iou = float(
                source_fiber.cal_iou(
                    target_fiber,
                    dist_sample=float(dist_sample),
                    dist_threshold=float(dist_threshold),
                    min_iou_thres=0.0,
                )
            )
            source_ratio = min(overlap_length, source_length) / source_length
            target_ratio = min(overlap_length, target_length) / target_length
            accepted = (
                overlap_length >= float(min_overlap_length)
                and (
                    iou >= float(min_iou)
                    or source_ratio >= float(min_overlap_ratio)
                    or target_ratio >= float(min_overlap_ratio)
                )
            )
            score = max(iou, source_ratio, target_ratio) + overlap_length * 1e-6
            if accepted:
                matches.append(
                    {
                        "source_fiber_index": float(source_fiber_index),
                        "target_fiber_index": float(target_fiber_index),
                        "overlap_length": float(overlap_length),
                        "source_overlap_ratio": float(source_ratio),
                        "target_overlap_ratio": float(target_ratio),
                        "iou": float(iou),
                        "source_length": float(source_length),
                        "target_length": float(target_length),
                        "score": float(score),
                    }
                )
    matches.sort(key=lambda item: float(item["score"]), reverse=True)
    return matches


def _overlap_node_mapping_from_match(
    source_points: np.ndarray,
    source_path: Sequence[int],
    target_points: np.ndarray,
    target_path: Sequence[int],
    target_fiber: Any,
    dist_sample: float,
    dist_threshold: float,
) -> Dict[int, int]:
    if len(source_path) == 0 or len(target_path) == 0:
        return {}

    from scipy.spatial import cKDTree

    source_path_np = np.asarray(source_path, dtype=np.int64)
    target_path_np = np.asarray(target_path, dtype=np.int64)
    source_coords = source_points[source_path_np]
    target_coords = target_points[target_path_np]
    if source_coords.shape[0] == 0 or target_coords.shape[0] == 0:
        return {}

    target_resampled, target_resampled_tree = target_fiber.cache_resample_by_distance(
        float(dist_sample)
    )
    if target_resampled.shape[0] == 0:
        return {}

    mapped_source_indices = set()
    node_dists, _ = target_resampled_tree.query(source_coords)
    node_radius = max(float(dist_threshold), float(dist_sample) * 0.75)
    for path_pos, dist in enumerate(np.asarray(node_dists).tolist()):
        if float(dist) <= node_radius:
            mapped_source_indices.add(int(source_path_np[path_pos]))

    if source_coords.shape[0] >= 2:
        midpoints = (source_coords[:-1] + source_coords[1:]) * 0.5
        midpoint_dists, _ = target_resampled_tree.query(midpoints)
        for edge_pos, dist in enumerate(np.asarray(midpoint_dists).tolist()):
            if float(dist) <= float(dist_threshold):
                left = int(source_path_np[edge_pos])
                right = int(source_path_np[edge_pos + 1])
                if float(node_dists[edge_pos]) <= node_radius:
                    mapped_source_indices.add(left)
                if float(node_dists[edge_pos + 1]) <= node_radius:
                    mapped_source_indices.add(right)

    if not mapped_source_indices:
        return {}

    target_tree = cKDTree(target_coords)
    mapping: Dict[int, int] = {}
    for source_idx in sorted(mapped_source_indices):
        _, target_pos = target_tree.query(source_points[int(source_idx)])
        mapping[int(source_idx)] = int(target_path_np[int(target_pos)])
    return mapping


def _find_overlap_merge_targets(
    source_points: np.ndarray,
    source_parents: np.ndarray,
    target_points: np.ndarray,
    target_parents: np.ndarray,
    dist_sample: float,
    dist_threshold: float,
    min_overlap_length: float,
    min_overlap_ratio: float,
    min_iou: float,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    if source_points.shape[0] == 0 or target_points.shape[0] == 0:
        return {}, {"matches": 0, "mapped_nodes": 0}

    source_indices = list(range(int(source_points.shape[0])))
    target_indices = list(range(int(target_points.shape[0])))
    source_paths = _component_path_indices(source_parents, source_indices)
    target_paths = _component_path_indices(target_parents, target_indices)
    source_fibers = [
        _make_overlap_fiber(source_points[np.asarray(path, dtype=np.int64)])
        for path in source_paths
    ]
    target_fibers = [
        _make_overlap_fiber(target_points[np.asarray(path, dtype=np.int64)])
        for path in target_paths
    ]
    matches = _accepted_overlap_matches(
        source_fibers=source_fibers,
        target_fibers=target_fibers,
        dist_sample=dist_sample,
        dist_threshold=dist_threshold,
        min_overlap_length=min_overlap_length,
        min_overlap_ratio=min_overlap_ratio,
        min_iou=min_iou,
    )
    mapping: Dict[int, int] = {}
    for match in matches:
        source_fiber_index = int(match["source_fiber_index"])
        target_fiber_index = int(match["target_fiber_index"])
        if source_fiber_index >= len(source_paths) or target_fiber_index >= len(target_paths):
            continue
        match_mapping = _overlap_node_mapping_from_match(
            source_points=source_points,
            source_path=source_paths[source_fiber_index],
            target_points=target_points,
            target_path=target_paths[target_fiber_index],
            target_fiber=target_fibers[target_fiber_index],
            dist_sample=dist_sample,
            dist_threshold=dist_threshold,
        )
        for source_idx, target_idx in match_mapping.items():
            mapping.setdefault(int(source_idx), int(target_idx))

    best_match = matches[0] if matches else None
    return mapping, {
        "matches": int(len(matches)),
        "mapped_nodes": int(len(mapping)),
        "best_match": best_match,
    }


def _copy_component_to_forest_lists(
    source_points: np.ndarray,
    source_parents: np.ndarray,
    source_node_types: np.ndarray,
    component_indices: Sequence[int],
    out_points: List[np.ndarray],
    out_parents: List[int],
    out_node_types: List[int],
) -> Tuple[Dict[int, int], List[int]]:
    component_set = {int(idx) for idx in component_indices}
    pending = set(component_set)
    local_to_out: Dict[int, int] = {}
    new_indices: List[int] = []

    while pending:
        progressed = False
        for node_idx in sorted(list(pending)):
            parent = int(source_parents[node_idx])
            if parent in component_set and parent not in local_to_out:
                continue
            out_idx = len(out_points)
            out_points.append(source_points[node_idx].astype(np.float32, copy=True))
            out_parents.append(int(local_to_out.get(parent, -1)))
            out_node_types.append(int(source_node_types[node_idx]))
            local_to_out[int(node_idx)] = int(out_idx)
            new_indices.append(int(out_idx))
            pending.remove(int(node_idx))
            progressed = True
        if not progressed:
            node_idx = int(min(pending))
            out_idx = len(out_points)
            out_points.append(source_points[node_idx].astype(np.float32, copy=True))
            out_parents.append(-1)
            out_node_types.append(int(source_node_types[node_idx]))
            local_to_out[node_idx] = int(out_idx)
            new_indices.append(int(out_idx))
            pending.remove(node_idx)

    return local_to_out, new_indices


def _nearest_existing_node_mapping(
    source_points: np.ndarray,
    component_indices: Sequence[int],
    target_points: np.ndarray,
    radius: float,
) -> Dict[int, int]:
    if radius <= 0.0 or len(component_indices) == 0 or target_points.shape[0] == 0:
        return {}

    radius2 = float(radius) * float(radius)
    mapping: Dict[int, int] = {}
    for node_idx in component_indices:
        node_idx = int(node_idx)
        delta = target_points - source_points[node_idx][None]
        dist2 = np.einsum("ij,ij->i", delta, delta)
        nearest_idx = int(np.argmin(dist2))
        if float(dist2[nearest_idx]) <= radius2:
            mapping[node_idx] = nearest_idx
    return mapping


def _merge_component_to_forest_lists(
    source_points: np.ndarray,
    source_parents: np.ndarray,
    source_node_types: np.ndarray,
    component_indices: Sequence[int],
    mapped_nodes: Dict[int, int],
    out_points: List[np.ndarray],
    out_parents: List[int],
    out_node_types: List[int],
) -> Tuple[Dict[int, int], List[int]]:
    component_set = {int(idx) for idx in component_indices}
    adjacency: Dict[int, List[int]] = {idx: [] for idx in component_set}
    for node_idx in component_set:
        parent = int(source_parents[node_idx])
        if parent in component_set:
            adjacency[node_idx].append(parent)
            adjacency[parent].append(node_idx)

    local_to_out = {int(k): int(v) for k, v in mapped_nodes.items()}
    visited = set(local_to_out.keys())
    queue: "deque[int]" = deque(sorted(visited))
    new_indices: List[int] = []

    while queue:
        node_idx = int(queue.popleft())
        parent_out_idx = int(local_to_out[node_idx])
        for neighbor in sorted(adjacency.get(node_idx, [])):
            neighbor = int(neighbor)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            out_idx = len(out_points)
            out_points.append(source_points[neighbor].astype(np.float32, copy=True))
            out_parents.append(parent_out_idx)
            out_node_types.append(int(source_node_types[neighbor]))
            local_to_out[neighbor] = int(out_idx)
            new_indices.append(int(out_idx))
            queue.append(neighbor)

    missing = [idx for idx in sorted(component_set) if idx not in local_to_out]
    if missing:
        copied_map, copied_indices = _copy_component_to_forest_lists(
            source_points,
            source_parents,
            source_node_types,
            missing,
            out_points,
            out_parents,
            out_node_types,
        )
        local_to_out.update(copied_map)
        new_indices.extend(copied_indices)

    return local_to_out, new_indices


# ---------------------------------------------------------------------------
# Array-level post-processing ops
# ---------------------------------------------------------------------------
def _post_merge_forest_by_overlap(
    points: np.ndarray,
    parents: np.ndarray,
    node_types: np.ndarray,
    dist_sample: float,
    dist_threshold: float,
    min_overlap_length: float,
    min_overlap_ratio: float,
    min_iou: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    components = _forest_connected_components(parents)
    out_points: List[np.ndarray] = []
    out_parents: List[int] = []
    out_node_types: List[int] = []
    target_fibers: List[Any] = []
    stats: Dict[str, Any] = {
        "enabled": True,
        "input_nodes": int(points.shape[0]),
        "input_components": int(len(components)),
        "merged_components": 0,
        "component_logs": [],
        "dist_sample": float(dist_sample),
        "dist_threshold": float(dist_threshold),
        "min_overlap_length": float(min_overlap_length),
        "min_overlap_ratio": float(min_overlap_ratio),
        "min_iou": float(min_iou),
    }

    for component_order, component_indices in enumerate(components):
        source_fibers = _component_overlap_fibers(points, parents, component_indices)
        match = None
        if target_fibers and source_fibers:
            match = _best_overlap_match(
                source_fibers=source_fibers,
                target_fibers=target_fibers,
                dist_sample=dist_sample,
                dist_threshold=dist_threshold,
                min_overlap_length=min_overlap_length,
                min_overlap_ratio=min_overlap_ratio,
                min_iou=min_iou,
            )

        if match is None:
            _, new_indices = _copy_component_to_forest_lists(
                points,
                parents,
                node_types,
                component_indices,
                out_points,
                out_parents,
                out_node_types,
            )
            stats["component_logs"].append(
                {
                    "component_order": int(component_order),
                    "action": "copy",
                    "input_nodes": int(len(component_indices)),
                    "new_nodes": int(len(new_indices)),
                }
            )
        else:
            target_points = np.stack(out_points, axis=0).astype(np.float32)
            mapped_nodes = _nearest_existing_node_mapping(
                source_points=points,
                component_indices=component_indices,
                target_points=target_points,
                radius=dist_threshold,
            )
            if not mapped_nodes:
                component_array = np.asarray(component_indices, dtype=np.int64)
                source_subset = points[component_array]
                delta = source_subset[:, None, :] - target_points[None]
                dist2 = np.einsum("ijk,ijk->ij", delta, delta)
                flat_idx = int(np.argmin(dist2))
                source_pos, target_idx = np.unravel_index(flat_idx, dist2.shape)
                mapped_nodes[int(component_array[int(source_pos)])] = int(target_idx)

            _, new_indices = _merge_component_to_forest_lists(
                points,
                parents,
                node_types,
                component_indices,
                mapped_nodes,
                out_points,
                out_parents,
                out_node_types,
            )
            stats["merged_components"] = int(stats["merged_components"]) + 1
            component_log = {
                "component_order": int(component_order),
                "action": "overlap_merge",
                "input_nodes": int(len(component_indices)),
                "mapped_nodes": int(len(mapped_nodes)),
                "new_nodes": int(len(new_indices)),
            }
            component_log.update(match)
            stats["component_logs"].append(component_log)

        if len(out_points) > 0:
            out_points_np = np.stack(out_points, axis=0).astype(np.float32)
            out_parents_np = np.asarray(out_parents, dtype=np.int64)
            target_fibers = _component_overlap_fibers(
                out_points_np,
                out_parents_np,
                list(range(len(out_points))),
            )

    if len(out_points) == 0:
        stats["output_nodes"] = 0
        stats["output_components"] = 0
        return points, parents, node_types, stats

    out_points_np = np.stack(out_points, axis=0).astype(np.float32)
    out_parents_np = np.asarray(out_parents, dtype=np.int64)
    out_node_types_np = np.asarray(out_node_types, dtype=np.int64)
    stats["output_nodes"] = int(out_points_np.shape[0])
    stats["output_components"] = int(len(_forest_connected_components(out_parents_np)))
    return out_points_np, out_parents_np, out_node_types_np, stats


def _compact_forest_by_keep_mask(
    points: np.ndarray,
    parents: np.ndarray,
    node_types: np.ndarray,
    keep_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    keep_mask = np.asarray(keep_mask, dtype=bool)
    kept_indices = np.flatnonzero(keep_mask)
    old_to_new = {int(old_idx): int(new_idx) for new_idx, old_idx in enumerate(kept_indices)}

    out_points = points[kept_indices].astype(np.float32, copy=True)
    out_node_types = node_types[kept_indices].astype(np.int64, copy=True)
    out_parents: List[int] = []
    for old_idx in kept_indices.tolist():
        parent = int(parents[int(old_idx)])
        out_parents.append(int(old_to_new.get(parent, -1)))
    return out_points, np.asarray(out_parents, dtype=np.int64), out_node_types


def _post_prune_short_terminal_branches(
    points: np.ndarray,
    parents: np.ndarray,
    node_types: np.ndarray,
    min_length: float,
    max_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    current_points = np.asarray(points, dtype=np.float32)
    current_parents = np.asarray(parents, dtype=np.int64)
    current_node_types = np.asarray(node_types, dtype=np.int64)
    min_length = float(min_length)
    max_iterations = max(1, int(max_iterations))
    stats: Dict[str, Any] = {
        "enabled": True,
        "input_nodes": int(current_points.shape[0]),
        "input_components": int(len(_forest_connected_components(current_parents))),
        "min_length": float(min_length),
        "max_iterations": int(max_iterations),
        "removed_nodes": 0,
        "removed_branches": 0,
        "iteration_logs": [],
    }
    if current_points.shape[0] < 2 or min_length <= 0.0:
        stats["output_nodes"] = int(current_points.shape[0])
        stats["output_components"] = int(len(_forest_connected_components(current_parents)))
        return current_points, current_parents, current_node_types, stats

    for iteration in range(max_iterations):
        if current_points.shape[0] < 2:
            break
        neighbors = _neighbors_from_parents(current_parents)
        degrees = np.asarray([len(item) for item in neighbors], dtype=np.int64)
        roots = {int(idx) for idx in np.flatnonzero(current_parents < 0).tolist()}
        components = _forest_connected_components(current_parents)
        component_sizes: Dict[int, int] = {}
        for component in components:
            for node_idx in component:
                component_sizes[int(node_idx)] = int(len(component))

        remove_nodes: set = set()
        branch_logs: List[Dict[str, Any]] = []
        endpoints = [int(idx) for idx in np.flatnonzero(degrees <= 1).tolist()]
        for leaf in endpoints:
            if leaf in remove_nodes or int(degrees[leaf]) != 1:
                continue

            path_nodes = [int(leaf)]
            prev_idx = int(leaf)
            current_idx = int(neighbors[leaf][0])
            branch_length = float(
                np.linalg.norm(current_points[prev_idx] - current_points[current_idx])
            )

            while int(degrees[current_idx]) == 2 and current_idx not in roots:
                path_nodes.append(int(current_idx))
                next_candidates = [
                    int(idx) for idx in neighbors[current_idx] if int(idx) != prev_idx
                ]
                if not next_candidates:
                    break
                next_idx = int(next_candidates[0])
                branch_length += float(
                    np.linalg.norm(current_points[current_idx] - current_points[next_idx])
                )
                prev_idx = int(current_idx)
                current_idx = int(next_idx)

            anchor_idx = int(current_idx)
            if anchor_idx in path_nodes:
                continue
            if anchor_idx not in roots and int(degrees[anchor_idx]) <= 1:
                continue
            if len(path_nodes) + 1 >= int(component_sizes.get(leaf, current_points.shape[0])):
                continue
            if branch_length >= min_length:
                continue
            if any(int(node_idx) in remove_nodes for node_idx in path_nodes):
                continue

            remove_nodes.update(int(node_idx) for node_idx in path_nodes)
            branch_logs.append(
                {
                    "leaf_index": int(leaf),
                    "anchor_index": int(anchor_idx),
                    "length": float(branch_length),
                    "removed_node_indices": [int(idx) for idx in path_nodes],
                }
            )

        if not remove_nodes:
            break

        keep_mask = np.ones((int(current_points.shape[0]),), dtype=bool)
        keep_mask[np.asarray(sorted(remove_nodes), dtype=np.int64)] = False
        current_points, current_parents, current_node_types = _compact_forest_by_keep_mask(
            current_points,
            current_parents,
            current_node_types,
            keep_mask,
        )
        stats["removed_nodes"] = int(stats["removed_nodes"]) + int(len(remove_nodes))
        stats["removed_branches"] = int(stats["removed_branches"]) + int(len(branch_logs))
        stats["iteration_logs"].append(
            {
                "iteration": int(iteration + 1),
                "removed_nodes": int(len(remove_nodes)),
                "removed_branches": int(len(branch_logs)),
                "branches": branch_logs,
            }
        )

    stats["output_nodes"] = int(current_points.shape[0])
    stats["output_components"] = int(len(_forest_connected_components(current_parents)))
    return current_points, current_parents, current_node_types, stats


# ---------------------------------------------------------------------------
# Swc <-> array adapters
# ---------------------------------------------------------------------------
def swc_to_arrays(swc: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a :class:`swclib.data.swc.Swc` to ``(points_xyz, parents, node_types)``.

    ``points_xyz`` is ``(N, 3)`` float32 in native ``(x, y, z)`` order, ``parents``
    is an ``(N,)`` int64 array of **0-based** parent indices (root = ``-1``), and
    ``node_types`` is an ``(N,)`` int64 array. Node ids are mapped to their
    insertion order, so the conversion round-trips with :func:`swc_from_arrays`.
    """
    node_ids = list(swc.nodes.keys())
    id_to_index = {int(node_id): int(idx) for idx, node_id in enumerate(node_ids)}
    num_nodes = len(node_ids)
    points = np.zeros((num_nodes, 3), dtype=np.float32)
    parents = np.full((num_nodes,), -1, dtype=np.int64)
    node_types = np.zeros((num_nodes,), dtype=np.int64)
    for idx, node_id in enumerate(node_ids):
        node = swc.nodes[node_id]
        points[idx, 0] = float(node["x"])
        points[idx, 1] = float(node["y"])
        points[idx, 2] = float(node["z"])
        node_types[idx] = int(node.get("type", 0))
        parent_id = int(node["parent"])
        if parent_id in id_to_index:
            parents[idx] = int(id_to_index[parent_id])
    return points, parents, node_types


def swc_from_arrays(
    points_xyz: np.ndarray,
    parents: np.ndarray,
    node_types: Optional[np.ndarray] = None,
    radius: float = 1.0,
) -> Any:
    """Build a :class:`swclib.data.swc.Swc` from ``(points_xyz, parents, node_types)``.

    ``points_xyz`` is ``(N, 3)`` in native ``(x, y, z)`` order, ``parents`` holds
    **0-based** parent indices (root = ``-1``). Node ids are assigned ``1..N`` in
    array order so the result round-trips with :func:`swc_to_arrays`.
    """
    from swclib.data.swc import Swc

    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    num_nodes = int(points_xyz.shape[0])
    if node_types is None:
        node_types = np.full((num_nodes,), 3, dtype=np.int64)
    else:
        node_types = np.asarray(node_types, dtype=np.int64)

    swc = Swc()
    for idx in range(num_nodes):
        node_id = idx + 1
        parent_idx = int(parents[idx])
        parent_id = parent_idx + 1 if parent_idx >= 0 else -1
        swc.nodes[node_id] = {
            "id": node_id,
            "type": int(node_types[idx]),
            "x": float(points_xyz[idx, 0]),
            "y": float(points_xyz[idx, 1]),
            "z": float(points_xyz[idx, 2]),
            "radius": float(radius),
            "parent": int(parent_id),
        }
        swc.edges.append((node_id, int(parent_id)))
    swc._refresh_bound_box()
    return swc


# ---------------------------------------------------------------------------
# Public Swc-level ops
# ---------------------------------------------------------------------------
def merge_overlapping_branches(
    swc: Any,
    dist_sample: float,
    dist_threshold: float,
    min_overlap_length: float,
    min_overlap_ratio: float,
    min_iou: float,
) -> Tuple[Any, Dict[str, Any]]:
    """Overlap-based branch de-duplication on a :class:`Swc`.

    Folds forest components that overlap an already-kept component into it,
    de-duplicating redundant branches. Returns ``(Swc, stats)``.
    """
    points, parents, node_types = swc_to_arrays(swc)
    points, parents, node_types, stats = _post_merge_forest_by_overlap(
        points,
        parents,
        node_types,
        dist_sample=dist_sample,
        dist_threshold=dist_threshold,
        min_overlap_length=min_overlap_length,
        min_overlap_ratio=min_overlap_ratio,
        min_iou=min_iou,
    )
    return swc_from_arrays(points, parents, node_types), stats


def prune_short_terminal_branches(
    swc: Any,
    min_length: float,
    max_iterations: int,
) -> Tuple[Any, Dict[str, Any]]:
    """Iteratively prune short terminal branches from a :class:`Swc`.

    Returns ``(Swc, stats)``.
    """
    points, parents, node_types = swc_to_arrays(swc)
    points, parents, node_types, stats = _post_prune_short_terminal_branches(
        points,
        parents,
        node_types,
        min_length=min_length,
        max_iterations=max_iterations,
    )
    return swc_from_arrays(points, parents, node_types), stats
