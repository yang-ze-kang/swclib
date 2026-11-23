"""Read-only adapter between the detector and the viewer (one pair per request)."""
from pathlib import Path

from swclib.error_detection import detect_errors, detect_trace_errors
from swclib.error_detection.detection import _checked_swc


def detect_pair(pred_path=None, gt_path=None, distance_tolerance=3.0, min_error_length=3.0,
                sample_step=0.5, trace_path=None, coverage_threshold=0.9, short_length=None,
                other_gts=None, fiber_metric_options=None, max_fiber_pairs=4000000):
    options=dict(coverage_threshold=coverage_threshold, short_length=short_length,
                 other_gts=other_gts, fiber_metric_options=fiber_metric_options,
                 max_fiber_pairs=max_fiber_pairs)
    if trace_path is not None:
        if pred_path:
            raise ValueError('pred_path 与 trace_path 只能提供一个')
        if not isinstance(trace_path, str) or not trace_path.strip():
            raise ValueError('请输入 trace_steps.jsonl 路径或其目录')
        result = detect_trace_errors(trace_path, gt_path or None, distance_tolerance=distance_tolerance,
            min_error_length=min_error_length, sample_step=sample_step, include_geometry=True, **options)
        geometry = result.pop('trace_geometry')
        return dict(result=result, pred=dict(path=result['trace']['path'], nodes=geometry['pred_nodes']),
                    gt=dict(path=result['trace']['gt_path'], nodes=geometry['gt_nodes']))
    sources = []
    for name, value in [('pred', pred_path), ('gt', gt_path)]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{name}: 请输入一个 SWC 文件路径')
        path = Path(value).expanduser().resolve(strict=True)
        if not path.is_file() or path.suffix.lower() != '.swc':
            raise ValueError(f'{name}: 需要一个 .swc 文件，不能使用目录')
        sources.append((path, _checked_swc(path, name)))
    pred, gt = sources
    result = detect_errors(pred[1], gt[1], distance_tolerance=distance_tolerance,
                           min_error_length=min_error_length, sample_step=sample_step, **options)
    # Original nodes are transferred once; each event identifies its descendant
    # nodes by ID. The browser clips only the entry edge at the event location.
    def morphology(source):
        path, swc = source
        return dict(path=str(path), nodes=[dict(id=int(nid), parent=int(node['parent']),
            xyz=[float(node[axis]) for axis in ('x', 'y', 'z')])
            for nid, node in swc.nodes.items()])
    return dict(result=result, pred=morphology(pred), gt=morphology(gt))
