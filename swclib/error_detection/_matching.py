"""Directed, spatially constrained SWC path matching and auditable error events.

This is a specified operational metric, not FiberMetric FP/FN relabeling.
The spatial grid preserves every original vertex and parent-child edge.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from swclib.data.swc_fiber import resample_nodes_by_distance
from swclib.data.swc_forest import SwcForest
from swclib.metrics._fiber_matching import _load_resampled_forest
from collections import OrderedDict


@dataclass(frozen=True)
class Parameters:
    distance: float
    min_error_length: float
    sample_step: float = 0.5

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.distance, self.min_error_length, self.sample_step)):
            raise ValueError("All metric parameters must be finite and positive")
        if self.sample_step > min(self.distance, self.min_error_length):
            raise ValueError("sample_step must not exceed either error threshold")


class DenseTree:
    """Exact input polylines, subdivided with swclib; original nodes retained."""

    def __init__(self, swc, step):
        forest = _load_resampled_forest(swc)
        xyz, parents, original_ids = [], [], []
        original_to_dense = {}
        for node in forest.get_preorder_nodes():
            coord = np.asarray(list(node.coord), dtype=float)
            if node.is_root():
                original_to_dense[node.nid] = len(xyz)
                xyz.append(coord)
                parents.append(-1)
                original_ids.append(node.nid)
                continue
            parent = original_to_dense[node.parent.nid]
            start = xyz[parent]
            if np.linalg.norm(coord - start) < 1e-12:
                samples = coord[None, :]
            else:
                samples = resample_nodes_by_distance(
                    np.asarray([start, coord], dtype=np.float64), distance=step)[1:]
            for point in samples:
                idx = len(xyz)
                xyz.append(point)
                parents.append(parent)
                original_ids.append(node.nid)
                parent = idx
            original_to_dense[node.nid] = parent
        self.xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        self.parent = np.asarray(parents, dtype=np.int64)
        self.original_id = np.asarray(original_ids, dtype=np.int64)
        self.children = [[] for _ in parents]
        self.edge_length = np.zeros(len(parents))
        self.depth = np.zeros(len(parents))
        self.segment_anchor = np.full(len(parents), -1, dtype=np.int64)
        self.segment_start = np.arange(len(parents), dtype=np.int64)
        self.roots = []
        for i, parent in enumerate(parents):
            if parent < 0:
                self.roots.append(i)
            else:
                self.children[parent].append(i)
                self.edge_length[i] = np.linalg.norm(self.xyz[i] - self.xyz[parent])
                self.depth[i] = self.depth[parent] + self.edge_length[i]
        for i, parent in enumerate(parents):
            if parent >= 0:
                self.segment_anchor[i] = (parent if len(self.children[parent]) != 1
                                          or self.parent[parent] < 0
                                          else self.segment_anchor[parent])
                self.segment_start[i] = (i if len(self.children[parent]) != 1
                                         or self.parent[parent] < 0
                                         else self.segment_start[parent])
        self.original_nodes = len(swc.nodes)
        self.subtree_end = np.arange(len(parents), dtype=np.int64) + 1
        for i in range(len(parents) - 1, -1, -1):
            if parents[i] >= 0:
                self.subtree_end[parents[i]] = max(self.subtree_end[parents[i]], self.subtree_end[i])

    def __len__(self):
        return len(self.parent)

    def ancestor(self, first, second):
        return first <= second < self.subtree_end[first]


def _maximal_states(tree, states):
    ordered = sorted(set(states))
    return tuple(node for j, node in enumerate(ordered)
                 if j + 1 == len(ordered) or not tree.ancestor(node, ordered[j + 1]))


def _update_peaks(tree, previous, current):
    compatible = [old for old in previous if any(
        tree.ancestor(old, new) or tree.ancestor(new, old) for new in current)]
    return _maximal_states(tree, (*compatible, *current))


def _advance(previous, candidates, children):
    """Monotone discrete path matching: stay, advance one edge, then close.

    All horizontal GT advances must remain in the current prediction's ball.
    The one-edge diagonal permits both discretized curves to advance together.
    Keeping all reachable candidates avoids nearest-neighbor path switching.
    """
    if not previous or not candidates:
        return ()
    allowed = set(candidates)
    reached = set()
    for old in previous:
        if old in allowed:
            reached.add(old)
        reached.update(child for child in children[old] if child in allowed)
    stack = list(reached)
    while stack:
        old = stack.pop()
        for child in children[old]:
            if child in allowed and child not in reached:
                reached.add(child)
                stack.append(child)
    return tuple(sorted(reached))


def _new_event(origin, anchor, frontier, gt, pred):
    return {
        "kind": "false_junction",
        "pred_start_index": int(origin),
        "pred_anchor_index": int(anchor),
        "pred_start_node_id": int(pred.original_id[origin]),
        "location_xyz": pred.xyz[origin].tolist(),
        "anchor_xyz": pred.xyz[anchor].tolist(),
        "gt_frontier_indices": list(map(int, frontier)),
        "gt_frontier_xyz": gt.xyz[list(frontier)].tolist() if frontier else [],
        "max_evidence_length_voxel": 0.0,
        "outside_gt_seen": False,
        "suppressed_break_id": None,
    }


class _BallCandidates:
    """Bound query memory by chunk instead of retaining all neighborhood lists."""
    def __init__(self, gold, points, radius, chunk=2048):
        self.tree, self.points, self.radius = cKDTree(gold), points, radius
        self.chunk, self.cache = chunk, OrderedDict()

    def __getitem__(self, index):
        index=int(index)
        for (start,end),values in self.cache.items():
            if start<=index<end:
                return values[index-start]
        start=index;end=min(len(self.points),start+self.chunk)
        counts=self.tree.query_ball_point(self.points[start:end],self.radius,return_length=True)
        cumulative=np.cumsum(counts)
        # Dense somas/crossings shrink the query chunk rather than allocating
        # millions of Python neighbor lists in one call. Nothing is truncated.
        size=max(1,int(np.searchsorted(cumulative,262144,side='right')))
        end=min(end,start+size)
        values=self.tree.query_ball_point(self.points[start:end],self.radius)
        self.cache[start,end]=values
        if len(self.cache)>2:
            self.cache.popitem(last=False)
        return values[0]


def _prediction_events(gt, pred, params, *, return_states=False, confirm_topology=False):
    n = len(pred)
    candidates = _BallCandidates(gt.xyz, pred.xyz, params.distance + 1e-10)
    coverage = np.zeros(len(gt), dtype=bool)
    states = [()] * n
    peaks = [()] * n
    # Pending tuple: origin, anchor, frozen reachable GT frontier, length, outside.
    pending = [None] * n
    active = [None] * n
    events = {}
    root_offsets = []
    reset_nodes = []
    for i in range(n):
        parent = int(pred.parent[i])
        current = candidates[i]
        if parent < 0:
            root_distance = float(np.linalg.norm(gt.xyz[gt.roots] - pred.xyz[i], axis=1).min())
            root_offsets.append(root_distance)
            seeds = tuple(root for root in gt.roots if root in current)
            states[i] = _advance(seeds, current, gt.children) if seeds else tuple(sorted(current))
            if states[i]:
                coverage[list(states[i])] = True
                peaks[i] = _maximal_states(gt, states[i])
            else:
                pending[i] = (i, i, (), 0.0, True)
            continue

        # A confirmed excursion is one event until it re-enters the GT tube.
        # Independent later transitions can then create additional events.
        if active[parent] is not None:
            origin = active[parent]
            event = events[origin]
            event["max_evidence_length_voxel"] = max(
                event["max_evidence_length_voxel"], float(pred.depth[i] - pred.depth[origin]))
            if not current:
                active[i] = origin
                event["outside_gt_seen"] = True
                continue
            event.setdefault("pred_reentry_indices", []).append(i)
            states[i] = tuple(sorted(current))
            reset_nodes.append(i)
            coverage[list(states[i])] = True
            peaks[i] = _maximal_states(gt, states[i])
            continue

        previous_pending = pending[parent]
        previous = previous_pending[2] if previous_pending else states[parent]
        reached = _advance(previous, current, gt.children)
        if reached:
            states[i] = reached
            coverage[list(reached)] = True
            peaks[i] = _update_peaks(gt, peaks[parent], reached)
            continue

        if previous_pending:
            origin, anchor, frontier, length, outside = previous_pending
            length += float(pred.edge_length[i])
            outside = outside or not bool(current)
        else:
            origin, anchor, frontier, length, outside = i, parent, previous, 0.0, not bool(current)
        topology_switch = bool(confirm_topology and current and frontier and
            not any(gt.ancestor(a, b) or gt.ancestor(b, a)
                    for a in frontier for b in current))
        if length + 1e-9 < params.min_error_length and not topology_switch:
            pending[i] = (origin, anchor, frontier, length, outside)
            peaks[i] = peaks[parent]
            continue

        event = events.setdefault(origin, _new_event(origin, anchor, frontier, gt, pred))
        event.setdefault("gt_attribution_frontier_indices", list(map(int, peaks[anchor])))
        event.setdefault("pred_confirmation_index", i)
        event["short_topology_evidence"] = topology_switch
        event.setdefault("direction_reversal", bool(current and frontier and
            all(any(gt.ancestor(target, source) for source in frontier) for target in current)
            and max(gt.depth[list(current)]) < max(gt.depth[list(frontier)])))
        event["max_evidence_length_voxel"] = max(event["max_evidence_length_voxel"], length)
        event["outside_gt_seen"] = bool(event["outside_gt_seen"] or outside)
        if current:
            # Reset after recording the discontinuity. This lets later events
            # and correctly traced suffixes be evaluated independently.
            event.setdefault("pred_reentry_indices", []).append(i)
            states[i] = tuple(sorted(current))
            reset_nodes.append(i)
            coverage[list(states[i])] = True
            peaks[i] = _maximal_states(gt, states[i])
        else:
            active[i] = origin

    result = []
    for event_id, origin in enumerate(sorted(events)):
        event = events[origin]
        event["event_id"] = event_id
        event["subtype"] = "outside_gt_subtree" if event["outside_gt_seen"] else "within_gt_topology_switch"
        result.append(event)
    result = _merge_reversals(result, gt, pred, candidates, params)
    terminal_matches = []
    for i in range(n):
        if pred.children[i]:
            continue
        choices = states[i] or (pending[i][2] if pending[i] else ())
        if choices:
            # A single predicted terminal cannot explain two missing branches.
            best = min(choices, key=lambda g: (float(np.linalg.norm(gt.xyz[g] - pred.xyz[i])), g))
            terminal_matches.append(int(best))
    output = (result, coverage, root_offsets, terminal_matches)
    return (*output, states, reset_nodes) if return_states else output


def _merge_reversals(events, gt, pred, candidates, params):
    """One continuous backward traversal is one event, not one per reset.

    Merge only along the same prediction ancestry and same GT lineage, with
    continuous geometric support and no recovered forward progress > delta.
    A subsequent reversal after real forward tracing remains a separate event.
    """
    result = []
    for event in events:
        merged = False
        if event["direction_reversal"] and not event["outside_gt_seen"]:
            for previous in reversed(result):
                if not previous["direction_reversal"] or previous["outside_gt_seen"]:
                    continue
                targets = previous.get("merged_confirmation_indices", [previous["pred_confirmation_index"]])
                targets = [node for node in targets if pred.ancestor(node, event["pred_start_index"])]
                if not targets:
                    continue
                stop = max(targets)
                path, node = [], event["pred_start_index"]
                while node != stop:
                    path.append(node)
                    node = int(pred.parent[node])
                path.append(stop)
                path.reverse()
                mapped = []
                for node in path:
                    if not candidates[node]:
                        break
                    mapped.append(min(candidates[node], key=lambda g: (
                        float(np.linalg.norm(gt.xyz[g] - pred.xyz[node])), g)))
                if len(mapped) != len(path):
                    continue
                lowest = gt.depth[mapped[0]]
                continuous = True
                for a, b in zip(mapped, mapped[1:]):
                    if not (gt.ancestor(a, b) or gt.ancestor(b, a)):
                        continuous = False
                        break
                    lowest = min(lowest, gt.depth[b])
                    if gt.depth[b] - lowest > params.distance + 1e-9:
                        continuous = False
                        break
                if not continuous:
                    continue
                previous.setdefault("merged_raw_event_origins", [previous["pred_start_index"]]).append(event["pred_start_index"])
                previous.setdefault("merged_confirmation_indices", [previous["pred_confirmation_index"]]).append(event["pred_confirmation_index"])
                previous["max_evidence_length_voxel"] = max(previous["max_evidence_length_voxel"],
                    float(pred.depth[event["pred_start_index"]] - pred.depth[previous["pred_start_index"]]
                          + event["max_evidence_length_voxel"]))
                merged = True
                break
        if not merged:
            result.append(event)
    for event_id, event in enumerate(result):
        event["event_id"] = event_id
    return result


def _missing_components(gt, coverage, params, terminal_matches):
    events = []
    terminal_depths = {}
    for node in terminal_matches:
        segment = int(gt.segment_start[node])
        terminal_depths[segment] = max(terminal_depths.get(segment, -np.inf), gt.depth[node])
    for start in range(len(gt)):
        parent = int(gt.parent[start])
        if coverage[start] or (parent >= 0 and not coverage[parent]):
            continue
        stack = [start]
        members = []
        terminal_depth = gt.depth[start]
        reconnected = False
        while stack:
            node = stack.pop()
            members.append(node)
            terminal_depth = max(terminal_depth, gt.depth[node])
            for child in gt.children[node]:
                if coverage[child]:
                    reconnected = True
                else:
                    stack.append(child)
        length = float(terminal_depth - gt.depth[start])
        if length + 1e-9 < params.min_error_length:
            continue
        anchor = int(gt.segment_anchor[start])
        has_branch_origin = anchor >= 0 and len(gt.children[anchor]) > 1
        terminal_on_segment = (terminal_depths.get(int(gt.segment_start[start]), -np.inf)
                               - gt.depth[anchor] > params.distance + 1e-9) if has_branch_origin else False
        terminal_on_segment = bool(terminal_on_segment)
        missed_branch = has_branch_origin and not terminal_on_segment
        events.append({
            "event_id": len(events), "kind": "break_error",
            "subtype": "missed_branch" if missed_branch else "premature_termination",
            "gt_start_index": start, "gt_parent_index": parent,
            "gt_start_node_id": int(gt.original_id[start]),
            "location_xyz": gt.xyz[start].tolist(),
            "anchor_xyz": gt.xyz[parent if parent >= 0 else start].tolist(),
            "max_missing_path_length_voxel": length,
            "missing_node_count": len(members),
            "_missing_dense_indices": sorted(members),
            "reconnected_downstream": reconnected,
            "predicted_terminal_on_gt_segment": terminal_on_segment,
            "suppressed_by_false_junction": None,
        })
    return events


def _junction_precedence(junctions, breaks, gt, coverage, params):
    """Assign at most one adjacent missing component to each junction.

    Only a missing component following the furthest-reached GT frontier through
    at most one spatial tolerance of covered GT is eligible. The last ball may
    shrink when a prediction turns perpendicular to GT; earlier balls can have
    already covered that short distance. A distal missing branch is not suppressed
    merely because it belongs to the same subtree. Hungarian matching handles
    multiple errors at a shared frontier without suppressing an entire subtree.
    """
    if not junctions or not breaks:
        return
    cost = np.full((len(junctions), len(breaks)), 1e9)
    for i, event in enumerate(junctions):
        frontier = set(event["gt_attribution_frontier_indices"])
        queue = [(node, 0.0) for node in frontier]
        nearby = dict(queue)
        while queue:
            node, distance = queue.pop()
            for child in gt.children[node]:
                next_distance = distance + gt.edge_length[child]
                if (coverage[child] and next_distance <= params.distance + 1e-9
                        and next_distance < nearby.get(child, float("inf"))):
                    nearby[child] = next_distance
                    queue.append((child, next_distance))
        eligible = []
        for j, missing in enumerate(breaks):
            if missing["gt_parent_index"] not in nearby:
                continue
            anchor = np.asarray(event["anchor_xyz"])
            cost[i, j] = float(np.linalg.norm(gt.xyz[missing["gt_start_index"]] - anchor))
            eligible.append(missing["event_id"])
        event["eligible_break_ids"] = eligible
        event["ambiguous_break_attribution"] = len(eligible) > 1
    for i, j in zip(*linear_sum_assignment(cost)):
        if cost[i, j] >= 1e8:
            continue
        junctions[i]["suppressed_break_id"] = breaks[j]["event_id"]
        breaks[j]["suppressed_by_false_junction"] = junctions[i]["event_id"]


def evaluate(gt_swc, pred_swc, params, gt_dense=None, pred_dense=None):
    gt = gt_dense if gt_dense is not None else DenseTree(gt_swc, params.sample_step)
    pred = pred_dense if pred_dense is not None else DenseTree(pred_swc, params.sample_step)
    if not len(gt):
        raise ValueError("GT must contain at least one node")
    empty_prediction = not len(pred)
    zero_length_prediction = bool(len(pred) and pred.edge_length.sum() <= 1e-12)
    # A seed without a traced path is an immediate stop even on a short GT.
    # Keep a nonempty point-only GT self-comparison valid (there is no path to trace).
    if empty_prediction or (zero_length_prediction and gt.edge_length.sum() > 1e-12):
        root_offsets = [float(np.linalg.norm(gt.xyz[gt.roots] - pred.xyz[root], axis=1).min())
                        for root in pred.roots]
        return {"false_junction_events": [], "break_events": [{
                    "event_id": 0, "kind": "break_error", "subtype": "premature_termination",
                    "empty_prediction": empty_prediction,
                    "zero_length_prediction": zero_length_prediction,
                    "immediate_termination": True,
                    "location_xyz": gt.xyz[gt.roots[0]].tolist(),
                    "max_missing_path_length_voxel": float(gt.depth.max()),
                    "suppressed_by_false_junction": None}],
                "suppressed_break_events": [], "false_junction_count": 0,
                "wrong_connection_count": 0, "extra_branch_count": 0,
                "terminal_overextension_count": 0,
                "reversal_diagnostics": [], "reversal_count": 0,
                "break_error_count": 1, "premature_termination_count": 1,
                "missed_branch_count": 0, "has_false_junction": False,
                "has_break_error": True, "empty_prediction": empty_prediction,
                "zero_length_prediction": zero_length_prediction, "immediate_termination": True,
                "gt_nodes": gt.original_nodes, "pred_nodes": pred.original_nodes,
                "gt_dense_nodes": len(gt), "pred_dense_nodes": len(pred),
                "root_offsets_voxel": root_offsets}
    transitions, covered, root_offsets, terminal_matches, states, reset_nodes = _prediction_events(
        gt, pred, params, return_states=True)
    junctions, reversals = [], []
    for event in transitions:
        if event["direction_reversal"] and not event["outside_gt_seen"]:
            event["kind"] = "direction_reversal_diagnostic"
            event["subtype"] = "pure_reversal"
            event["event_id"] = len(reversals)
            reversals.append(event)
        else:
            event["event_id"] = len(junctions)
            junctions.append(event)
    from ._subtypes import classify_junctions
    classify_junctions(gt, pred, junctions, states, reset_nodes, params)
    missing = _missing_components(gt, covered, params, terminal_matches)
    # Pure reversal is diagnostic only: it must not suppress missing GT.
    _junction_precedence(junctions, missing, gt, covered, params)
    breaks = [event for event in missing if event["suppressed_by_false_junction"] is None]
    suppressed = [event for event in missing if event["suppressed_by_false_junction"] is not None]
    return {"false_junction_events": junctions, "break_events": breaks,
            "reversal_diagnostics": reversals, "reversal_count": len(reversals),
            "suppressed_break_events": suppressed,
            "false_junction_count": len(junctions), "break_error_count": len(breaks),
            "wrong_connection_count": sum(e["error_type"] == "wrong_connection" for e in junctions),
            "extra_branch_count": sum(e["error_type"] == "extra_branch" for e in junctions),
            "terminal_overextension_count": sum(e["error_type"] == "terminal_overextension" for e in junctions),
            "premature_termination_count": sum(e["subtype"] == "premature_termination" for e in breaks),
            "missed_branch_count": sum(e["subtype"] == "missed_branch" for e in breaks),
            "has_false_junction": bool(junctions), "has_break_error": bool(breaks),
            "empty_prediction": False, "zero_length_prediction": zero_length_prediction,
            "immediate_termination": False, "gt_nodes": gt.original_nodes,
            "pred_nodes": pred.original_nodes,
            "gt_dense_nodes": len(gt), "pred_dense_nodes": len(pred),
            "root_offsets_voxel": root_offsets,
            "covered_gt_dense_nodes": int(covered.sum())}
