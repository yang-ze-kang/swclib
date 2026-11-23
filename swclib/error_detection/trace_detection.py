"""Trace final retained v2 edges back to the prediction that created them.

Node identities and directed edge coordinates from the recorder establish
provenance. No nearest-prediction guess is used for FJ candidate attribution.
"""
from collections import Counter, ChainMap
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from swclib.data.swc import Swc
from .detection import _checked_swc, detect_errors
from ._matching import DenseTree, Parameters

_MAX_LINE = 16 * 1024 * 1024


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    return value


def _point(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Trace coordinates must be finite XYZ triples')
    return point.tolist()


def _probability(value):
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Recorded candidate probability must be finite and in [0, 1]')
    return value


def _swc(nodes):
    swc = Swc()
    for nid, (parent, x, y, z) in nodes.items():
        swc.nodes[nid] = dict(id=nid, parent=parent, x=x, y=y, z=z, type=3, radius=1.)
        swc.edges.append((nid, parent))
    return _checked_swc(swc, 'reconstructed prediction')


def _geometry(swc):
    return [dict(id=int(nid), parent=int(node['parent']),
                 xyz=[float(node[k]) for k in ('x', 'y', 'z')])
            for nid, node in swc.nodes.items()]


class _RecordedTrace:
    def __init__(self, path):
        path = Path(path).expanduser().resolve(strict=True)
        self.path = path/'trace_steps.jsonl' if path.is_dir() else path
        self.folder = self.path.parent
        manifest = self.folder/'trace_replay.json'
        self.metadata = json.loads(manifest.read_text()) if manifest.is_file() else {}
        self.nodes, self.owners, self.steps, self.offsets = {}, {}, [], []
        self.end = None
        self.partial = False
        watched=[self.path, self.folder/'trace_initial.npz']+([manifest] if manifest.is_file() else [])
        self.file_versions={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in watched}
        self.removed_count = 0
        self.added_count = 0
        with self.path.open('rb') as stream:
            line = self._record(stream)
            if not line or line.get('event') != 'header' or line.get('version') != 2 or line.get('mode') != 'direct':
                raise ValueError('Error provenance currently requires a direct v2 trace_steps.jsonl recording')
            if line.get('coordinate_frame') != 'global_xyz_voxels':
                raise ValueError('Trace coordinates must use global_xyz_voxels')
            self.header = line
            initial = self.folder/'trace_initial.npz'
            with np.load(initial, allow_pickle=False) as arrays:
                points, parents = arrays['points_xyz'], arrays['parents']
            if (points.shape != (len(parents), 3) or not np.isfinite(points).all()
                    or not np.issubdtype(parents.dtype, np.integer)
                    or np.any(parents < -1) or np.any(parents >= len(points))):
                raise ValueError('Invalid trace_initial.npz')
            self.nodes = {i: [int(parent), *point.tolist()]
                          for i, (point, parent) in enumerate(zip(points, parents))}
            _swc(self.nodes)  # Reject initial cycles even if later deleted.
            self.initial_ids = set(self.nodes)
            for i in self.nodes:
                self.owners[i] = dict(status='initial_tree', recorded_node_id=i)
            line_number, previous_step = 1, -1
            while True:
                offset = stream.tell()
                record = self._record(stream)
                if record is None:
                    break
                line_number += 1
                if self.end is not None:
                    raise ValueError('Unexpected records after trace end')
                if record.get('event') == 'end':
                    self.end = record
                    continue
                if record.get('event') != 'prediction':
                    raise ValueError('Unexpected event in trace recording')
                step = _integer(record.get('step'), 'prediction step')
                if step <= previous_step:
                    raise ValueError('Prediction steps must be strictly increasing')
                previous_step = step
                index = len(self.steps)
                seed = _point(record['seed_xyz'])
                root = _integer(record['root'], 'prediction root')
                if root not in self.nodes or self.nodes[root][1:] != seed:
                    raise ValueError(f'Step {step}: root/seed does not match recorded state')
                reference = dict(prediction_step=step, prediction_index=index, log_line=line_number,
                                 log_path=str(self.path), seed_xyz=seed, root_node_id=root)
                self.steps.append(reference)
                self.offsets.append(offset)
                removed = [_integer(n, 'removed node') for n in record['removed_nodes']]
                if len(set(removed)) != len(removed) or any(n not in self.nodes for n in removed):
                    raise ValueError(f'Step {step}: invalid removed node IDs')
                for nid in removed:
                    self.nodes.pop(nid)
                    self.owners.pop(nid)
                self.removed_count += len(removed)
                added = {}
                for row in record['added_nodes']:
                    if len(row) != 5:
                        raise ValueError('added_nodes must contain [id, parent, x, y, z]')
                    nid, parent = (_integer(row[0], 'added node'), _integer(row[1], 'parent'))
                    if nid < 0 or nid in added or nid in self.nodes:
                        raise ValueError(f'Step {step}: duplicate active added node {nid}')
                    added[nid] = [parent, *_point(row[2:])]
                combined = ChainMap(added, self.nodes)
                edge_candidates = {}
                candidate_ids = set()
                for candidate in record['candidates']:
                    candidate_id = _integer(candidate['index'], 'candidate index')
                    if candidate_id in candidate_ids:
                        raise ValueError(f'Step {step}: duplicate candidate index')
                    candidate_ids.add(candidate_id)
                    owner = dict(candidate_index=candidate_id,
                                 probability=_probability(candidate.get('probability')),
                                 stop_reason=candidate.get('stop_reason'))
                    for a, b in candidate['actual_edges_xyz']:
                        key = tuple(_point(a)+_point(b))
                        if owner not in edge_candidates.setdefault(key, []):
                            edge_candidates[key].append(owner)
                for nid, data in added.items():
                    parent = data[0]
                    if parent == -1:
                        matches = []
                    elif parent not in combined:
                        raise ValueError(f'Step {step}: missing parent {parent}')
                    else:
                        matches = edge_candidates.get(tuple(combined[parent][1:]+data[1:]), [])
                    status = 'matched' if len(matches) == 1 else 'multiple_candidates' if matches else 'candidate_unrecorded'
                    self.owners[nid] = dict(status=status, recorded_node_id=nid, **reference,
                        candidates=matches, probability=matches[0]['probability'] if len(matches)==1 else None,
                        edge_xyz=[combined[parent][1:], data[1:]] if parent != -1 else None)
                self.nodes.update(added)
                self.added_count += len(added)
        self.pred = _swc(self.nodes)
        self.seed_tree = cKDTree([r['seed_xyz'] for r in self.steps]) if self.steps else None

    def check_unchanged(self):
        if any((p.stat().st_size,p.stat().st_mtime_ns)!=version for p,version in self.file_versions.items()):
            raise ValueError('Trace inputs changed during detection; retry after recording completes')

    def _record(self, stream):
        line = stream.readline(_MAX_LINE+1)
        if len(line) > _MAX_LINE:
            raise ValueError('Trace line exceeds 16 MiB')
        if not line:
            return None
        if not line.endswith(b'\n'):
            self.partial = True
            return None
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError('Trace records must be JSON objects')
        return record

    def source(self, position):
        nid = position['edge_child_id']
        if nid is None:
            nid = position['node_id']
        return deepcopy(self.owners.get(nid, dict(status='unrecorded', recorded_node_id=nid)))

    def details(self, index):
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[index])
            record = self._record(stream)
        if record is None or record.get('step') != self.steps[index]['prediction_step']:
            raise ValueError('Trace file changed during detection; retry after recording completes')
        return dict(**self.steps[index],
                    candidates=[dict(candidate_index=c['index'], probability=_probability(c.get('probability')),
                                     stop_reason=c.get('stop_reason'), actual_edge_count=len(c['actual_edges_xyz']),
                                     points_xyz=c['points_xyz'], actual_edges_xyz=c['actual_edges_xyz'])
                                for c in record['candidates']],
                    candidate_scores=[_probability(p) for p in record.get('scores', [])],
                    threshold_candidates=record.get('threshold_candidates', []),
                    root_survives=self.steps[index]['root_node_id'] in self.nodes)


def _ground_truth(trace, value):
    source = value if value is not None else trace.metadata.get('ground_truth')
    if source is None:
        raise ValueError('Provide gt or ground_truth in trace_replay.json')
    if isinstance(source, (str, Path)):
        source = Path(source).expanduser()
        if not source.is_absolute():
            source = trace.folder/source
    gt = deepcopy(_checked_swc(source, 'gt'))
    scale = np.asarray(trace.metadata.get('swc_to_voxel_scale_xyz', [1, 1, 1]), dtype=float)
    offset = np.asarray(trace.metadata.get('swc_to_voxel_offset_xyz', [0, 0, 0]), dtype=float)
    if (scale.shape != (3,) or offset.shape != (3,) or not np.isfinite([scale, offset]).all()
            or np.any(scale <= 0)):
        raise ValueError('Invalid SWC-to-voxel scale/offset in trace_replay.json')
    # Match Replay: supplied/manifest GT is in SWC coordinates; trace points
    # already use global voxel coordinates and must not be transformed twice.
    gt.rescale(scale).add_offset(offset)
    gt._refresh_bound_box()
    return gt, str(source.resolve()) if isinstance(source, Path) else None, scale.tolist(), offset.tolist()


def detect_trace_errors(trace_path, gt=None, *, distance_tolerance=3.0,
                        min_error_length=3.0, sample_step=0.5, include_geometry=False, **detection_options):
    """Detect retained-tree errors and attach recorded prediction provenance.

    ``trace_path`` is a direct v2 trace_steps.jsonl file or its directory;
    its sibling trace_initial.npz is required. ``gt`` is a Swc/path, defaulting
    to the manifest ground_truth. GT receives the manifest SWC-to-voxel transform.
    FJ start/anchor/confirmation each reference their creating prediction and
    exact candidate probabilities. Shared candidate ownership remains explicit.
    Break associations are related prediction records, not causal probabilities.
    All detect_errors fields remain available; counts refer to retained raw
    tracing geometry, before exported-SWC rounding/compaction/resampling.
    """
    params = Parameters(float(distance_tolerance), float(min_error_length), float(sample_step))
    distance_tolerance, min_error_length, sample_step = params.distance, params.min_error_length, params.sample_step
    trace = _RecordedTrace(trace_path)
    gold, gt_path, scale, offset = _ground_truth(trace, gt)
    if detection_options.get('other_gts') is not None:
        detection_options['other_gts'] = {label:_ground_truth(trace,value)[0]
                                         for label,value in detection_options['other_gts'].items()}
    result = detect_errors(trace.pred, gold, distance_tolerance=distance_tolerance,
                           min_error_length=min_error_length, sample_step=sample_step, **detection_options)
    details = {}
    for event in result['events']:
        if event.get('source') != 'pred' or event['kind'] == 'break_error':
            continue
        origin = {key: trace.source(event[key]) for key in ('start', 'anchor', 'confirmation') if key in event}
        origin['status'] = origin['start']['status']
        origin['confirmation_prediction_indices'] = sorted(set(
            trace.owners[n]['prediction_index'] for n in event.get('confirmation_path_node_ids', [])
            if 'prediction_index' in trace.owners[n]))
        event['trace_origin'] = origin
        for index in origin['confirmation_prediction_indices']:
            details.setdefault(index, None)
        for key in ('start', 'anchor', 'confirmation'):
            if key in origin and 'prediction_index' in origin[key]:
                index = origin[key]['prediction_index']
                details.setdefault(index, None)
        # Each interval edge has an exact creation record where available.
        # Their probabilities describe candidates, not the likelihood that an
        # upstream edge is the earliest cause of this FJ.
        review = event.get('onset_review', {'segments': []})
        indices = set()
        for segment in review['segments']:
            source = trace.source(dict(edge_child_id=segment['edge_child_id']))
            segment['trace_origin'] = source
            if 'prediction_index' in source:
                indices.add(source['prediction_index'])
                details.setdefault(source['prediction_index'], None)
        review['prediction_indices'] = sorted(indices)
        review['provenance_counts'] = dict(Counter(s['trace_origin']['status'] for s in review['segments']))
        estimate = event.get('onset_estimate', {})
        if estimate.get('estimated_onset'):
            source = trace.source(estimate['estimated_onset'])
            origin['estimated_onset'] = source
            if 'prediction_index' in source:
                details.setdefault(source['prediction_index'], None)
    dense_gt = DenseTree(gold, sample_step) if result.get('schema_version', 1) == 1 else None
    for event in result['break_events'] + [e for e in result.get('candidate_events',[]) if e.get('source')=='gt']:
        points = [event['anchor']['xyz']]
        points += [p['xyz'] for p in event.get('related_pred_positions', [])]
        points.append(event['start']['xyz'])
        # error_type is shared by both schemas; new GT-side candidates do
        # not carry the legacy matcher's subtype field.
        if dense_gt is not None and event['error_type'] == 'missed_branch':
            anchor = int(dense_gt.segment_anchor[event['gt_start_index']])
            if anchor >= 0:
                points.append(dense_gt.xyz[anchor].tolist())
        related = set()
        if trace.seed_tree is not None:
            for point in points:
                related.update(trace.seed_tree.query_ball_point(point, distance_tolerance+1e-9))
        event['trace_origin'] = dict(status='related_predictions_only' if related else 'unresolved',
            probability=None, related_prediction_indices=sorted(related),
            association_rule='recorded seed within distance_tolerance of GT matching anchor or missed-branch fork',
            association_points_xyz=points)
        for index in related:
            details.setdefault(index, None)
    for index in details:
        details[index] = trace.details(index)
    result['trace_predictions'] = {str(index): value for index, value in sorted(details.items())}
    result['trace'] = dict(path=str(trace.path), initial_path=str(trace.folder/'trace_initial.npz'),
        version=2, mode='direct', scope='last_retained_raw_tree',
        complete=bool(trace.end and trace.end.get('status') == 'complete' and not trace.partial),
        end=trace.end, partial_last_line=trace.partial, prediction_count=len(trace.steps),
        initial_node_count=len(trace.initial_ids), added_node_count=trace.added_count,
        removed_node_count=trace.removed_count, retained_node_count=len(trace.nodes),
        retained_initial_node_count=sum(o['status']=='initial_tree' for o in trace.owners.values()),
        gt_path=gt_path, swc_to_voxel_scale_xyz=scale, swc_to_voxel_offset_xyz=offset,
        score_threshold=trace.header.get('strategies', {}).get('score', {}).get('threshold'),
        fj_provenance_counts=dict(Counter(e['trace_origin']['status'] for e in result['false_junction_events'])))
    if include_geometry:
        result['trace_geometry'] = dict(pred_nodes=_geometry(trace.pred), gt_nodes=_geometry(gold))
    trace.check_unchanged()
    return result
