"""Public, read-only SWC error detection with original-node localization."""
from collections import Counter
from os import PathLike
from pathlib import Path
import numbers

import numpy as np

from swclib.data.swc import Swc
from ._matching import DenseTree, Parameters, evaluate

LEGACY_ERROR_TYPES = ("wrong_connection", "extra_branch", "terminal_overextension",
               "premature_termination", "missed_branch")


def _checked_swc(value, name):
    """Validate around Swc's existing parser; never repair or re-root inputs."""
    if isinstance(value, (str, PathLike)):
        path = Path(value)
        swc = Swc(str(path))
        rows = [line.split() for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
        if len(rows) != len(swc.edges):
            raise ValueError(f"{name}: SWC reader skipped malformed data rows")
        if len(rows) != len(swc.nodes):
            raise ValueError(f"{name}: duplicate SWC node IDs")
        # Swc.open casts IDs to int. Reject fractional IDs instead of silently
        # evaluating a different topology after truncation.
        if any(not float(row[k]).is_integer() for row in rows for k in (0, 1, 6)):
            raise ValueError(f"{name}: node ID, type and parent must be integers")
    elif isinstance(value, Swc):
        swc = value
    else:
        raise TypeError(f"{name} must be a Swc object or SWC file path")
    children = {nid: [] for nid in swc.nodes}
    roots = []
    for nid, node in swc.nodes.items():
        if (not isinstance(nid, numbers.Integral) or isinstance(nid, bool) or nid == -1
                or node.get('id', nid) != nid):
            raise ValueError(f"{name}: invalid or inconsistent node ID {nid}")
        if not np.isfinite([node[k] for k in ('x', 'y', 'z', 'radius')]).all():
            raise ValueError(f"{name}: nonfinite coordinate/radius at node {nid}")
        parent = node['parent']
        if not isinstance(parent, numbers.Integral) or isinstance(parent, bool):
            raise ValueError(f"{name}: parent must be an integer at node {nid}")
        if parent == -1:
            roots.append(nid)
        elif parent in children:
            children[parent].append(nid)
        else:
            raise ValueError(f"{name}: missing parent {parent} at node {nid}")
    stack, visited = list(roots), 0
    while stack:
        visited += 1
        stack.extend(children[stack.pop()])
    if visited != len(swc.nodes):
        raise ValueError(f"{name}: parent-pointer cycle")
    return swc


def _xyz(swc, nid):
    return np.asarray([swc.nodes[nid][axis] for axis in ('x', 'y', 'z')], dtype=float)


def _position(tree, swc, index):
    """An interpolated location belongs to an original directed SWC edge."""
    index = int(index)
    nid = int(tree.original_id[index])
    parent = int(swc.nodes[nid]['parent'])
    point, end = tree.xyz[index], _xyz(swc, nid)
    is_vertex = np.linalg.norm(point-end) <= 1e-9
    fraction = None
    if parent != -1:
        start = _xyz(swc, parent)
        delta = end-start
        square = float(delta@delta)
        fraction = float(np.clip((point-start)@delta/square, 0, 1)) if square else 1.0
    return dict(xyz=point.tolist(), dense_index=index,
                node_id=nid if is_vertex else None,
                edge_parent_id=parent if parent != -1 else None,
                edge_child_id=nid if parent != -1 else None,
                edge_fraction=fraction)


def _junction_geometry(event, tree, swc):
    origin = event['pred_start_index']
    confirmation = event['pred_confirmation_index']
    event['source'] = 'pred'
    event['start'] = _position(tree, swc, origin)
    event['anchor'] = _position(tree, swc, event['pred_anchor_index'])
    event['confirmation'] = _position(tree, swc, confirmation)
    # Includes descendants after a re-entry: this is the affected prediction
    # subtree, not a claim that each descendant is itself an error event.
    ids = list(dict.fromkeys(map(int, tree.original_id[origin:tree.subtree_end[origin]])))
    event['affected_subtree'] = dict(source='pred', root_node_id=ids[0], node_ids=ids,
                                    start=event['start'])
    path, node = [], confirmation
    while node != origin:
        path.append(node)
        node = int(tree.parent[node])
    path.append(origin)
    path.reverse()
    event['confirmation_path_xyz'] = tree.xyz[path].tolist()
    event['confirmation_path_node_ids'] = list(dict.fromkeys(map(int, tree.original_id[path])))
    event['root_to_start_length'] = float(tree.depth[origin])


def _onset_review(event, tree, swc):
    """Expose upstream geometry for review without inventing an exact onset.

    Walk the event's own ancestry to the nearest earlier prediction fork/root
    or correspondence reset. This is a structural review boundary, not proof
    that the boundary is correct or that deviation cannot begin farther back.
    Preserve original edge IDs and clip an interpolated reset/end exactly.
    """
    origin = int(event['pred_start_index'])
    reset = int(event.get('classification_last_reset_index', -1))
    path, node = [origin], origin
    reason = 'prediction_root'
    while tree.parent[node] >= 0:
        node = int(tree.parent[node])
        path.append(node)
        if node == reset:
            reason = 'gt_reentry'
            break
        if tree.parent[node] < 0:
            break
        if len(tree.children[node]) > 1:
            reason = 'prediction_fork'
            break
    path.reverse()
    segments = []
    for a, b in zip(path, path[1:]):
        nid = int(tree.original_id[b])
        if tree.edge_length[b] <= 1e-12:
            continue
        if segments and segments[-1]['edge_child_id'] == nid:
            segment = segments[-1]
            segment['end_xyz'] = tree.xyz[b].tolist()
            segment['length_voxel'] += float(tree.edge_length[b])
        else:
            segments.append(dict(edge_parent_id=int(swc.nodes[nid]['parent']), edge_child_id=nid,
                start_xyz=tree.xyz[a].tolist(), end_xyz=tree.xyz[b].tolist(),
                length_voxel=float(tree.edge_length[b])))
    for segment in segments:
        a = _xyz(swc, segment['edge_parent_id'])
        delta = _xyz(swc, segment['edge_child_id'])-a
        square = float(delta@delta)
        for key in ('start', 'end'):
            segment[key+'_fraction'] = float(np.clip((np.asarray(segment[key+'_xyz'])-a)@delta/square, 0, 1))
    points = ([segments[0]['start_xyz']] + [s['end_xyz'] for s in segments]
              if segments else [event['start']['xyz']])
    return dict(status='uncertain' if segments else 'no_upstream_path',
        method='upstream_prediction_segment', exact_onset=None, probability=None,
        boundary_kind=reason, boundary_is_error_onset=False,
        earlier_onset_not_excluded=True,
        start=_position(tree, swc, path[0]), end=event['start'],
        length_voxel=float(tree.depth[origin]-tree.depth[path[0]]),
        path_xyz=points, segments=segments)


def detect_errors_legacy(pred, gt, *, distance_tolerance=3.0, min_error_length=3.0,
                  sample_step=0.5):
    """Detect five tracing error types without modifying either input.

    Args:
        pred, gt: ``Swc`` objects or paths, in the same XYZ coordinate system.
        distance_tolerance: Spatial matching tolerance (default 3).
        min_error_length: Sustained failure/missing arc threshold (default 3).
        sample_step: Maximum subdivision spacing (default 0.5).

    Returns:
        JSON-serializable dict. ``counts`` counts all events; ``first_fj_counts``
        counts FJ entries with no earlier FJ on their prediction ancestry, plus
        the unchanged break events. ``events`` lists FJ and break records.
        Each record has ``error_type``, ``source``, ``start`` and ``anchor``.
        FJ additionally has ``is_first_fj``, ``confirmation``, a confirmation
        path and ``affected_subtree.node_ids`` using original prediction IDs.
        ``start`` is the first failed matching sample of a sustained event,
        not its later confirmation point or necessarily the geometric turn.
        ``onset_review`` supplies an uncertain upstream review interval and
        original edges; its boundary is not an asserted earliest error point.

    A same-location missing GT component attributed to FJ is recorded under
    ``suppressed_break_events`` instead of counted twice. Pure backtracking is
    diagnostic only. Empty/zero-length predictions produce one immediate break
    (a nonempty point-only GT identity has no missing path). Lengths use input
    coordinate units, i.e. voxels when inputs use voxel XYZ. See README.md.
    """
    params = Parameters(float(distance_tolerance), float(min_error_length), float(sample_step))
    pred_swc, gt_swc = _checked_swc(pred, 'pred'), _checked_swc(gt, 'gt')
    gold, prediction = DenseTree(gt_swc, params.sample_step), DenseTree(pred_swc, params.sample_step)
    result = evaluate(gt_swc, pred_swc, params, gt_dense=gold, pred_dense=prediction)
    first_end = -1
    for event in result['false_junction_events']:
        origin = event['pred_start_index']
        event['is_first_fj'] = origin >= first_end
        if event['is_first_fj']:
            first_end = int(prediction.subtree_end[origin])
        event['id'] = f"fj-{event['event_id']}"
        _junction_geometry(event, prediction, pred_swc)
        event['start_role'] = 'matching_failure'
        event['onset_review'] = _onset_review(event, prediction, pred_swc)
    for group in ('break_events', 'suppressed_break_events'):
        for event in result[group]:
            start = event.get('gt_start_index', gold.roots[0])
            parent = int(gold.parent[start])
            event.update(id=f"break-{event['event_id']}", error_type=event['subtype'], source='gt',
                         start=_position(gold, gt_swc, start),
                         anchor=_position(gold, gt_swc, parent if parent >= 0 else start))
            event['anchor_xyz'] = event['anchor']['xyz']
            # Exact missing geometry is provided by the matching pass. Do not
            # highlight covered distal descendants as missing.
            members = event.pop('_missing_dense_indices', None)
            if members is None:  # Immediate termination: entire GT is missing.
                members = range(len(gold))
            event['missing_segments_xyz'] = [
                [gold.xyz[gold.parent[i]].tolist(), gold.xyz[i].tolist()]
                for i in members if gold.parent[i] >= 0]
    for event in result['reversal_diagnostics']:
        event['id'] = f"reversal-{event['event_id']}"
        _junction_geometry(event, prediction, pred_swc)
    result['events'] = result['false_junction_events'] + result['break_events']
    result['counts'] = _counts(result['events'])
    result['first_fj_counts'] = _counts([event for event in result['events']
                                       if event.get('is_first_fj', True)])
    result['schema_version'] = 1
    result['parameters'] = dict(distance_tolerance=params.distance,
                                min_error_length=params.min_error_length,
                                sample_step=params.sample_step, coordinate_units='input_xyz_units')
    return result


def _counts(events):
    counts = Counter(event['error_type'] for event in events)
    output = {key: counts[key] for key in LEGACY_ERROR_TYPES}
    output['false_junction'] = sum(counts[key] for key in LEGACY_ERROR_TYPES[:3])
    output['break_error'] = sum(counts[key] for key in LEGACY_ERROR_TYPES[3:])
    output['total'] = len(events)
    return output


def detect_errors(pred, gt, *, distance_tolerance=3.0, min_error_length=3.0,
                  sample_step=0.5, coverage_threshold=0.9, short_length=None,
                  other_gts=None, fiber_metric_options=None, max_fiber_pairs=4000000,
                  strategy="fiber"):
    """Detect fiber-referenced errors; raw FiberMetric results remain unchanged.

    See README.md for confirmed/candidate/diagnostic counts and onset intervals.
    ``strategy='legacy'`` is only for reproducing the earlier operational metric.
    """
    if strategy == "legacy":
        return detect_errors_legacy(pred, gt, distance_tolerance=distance_tolerance,
                                   min_error_length=min_error_length, sample_step=sample_step)
    if strategy != "fiber":
        raise ValueError("strategy must be 'fiber' or 'legacy'")
    from ._fiber_detection import detect
    return detect(pred, gt, distance_tolerance=distance_tolerance,
                  min_error_length=min_error_length, sample_step=sample_step,
                  coverage_threshold=coverage_threshold, short_length=short_length,
                  other_gts=other_gts, fiber_metric_options=fiber_metric_options,
                  max_fiber_pairs=max_fiber_pairs)


from ._fiber_detection import FJ_TYPES, BREAK_TYPES, CANDIDATE_TYPES, DIAGNOSTIC_TYPES
ERROR_TYPES = FJ_TYPES + BREAK_TYPES + CANDIDATE_TYPES + DIAGNOSTIC_TYPES
