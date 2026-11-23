"""Unmodified FiberMetric scores and length-weighted, shared-trunk references."""
from copy import deepcopy
import numpy as np

from swclib.metrics.fiber_metric import FiberMetric
from swclib.metrics._fiber_matching import _load_resampled_forest
from ._matching import _BallCandidates


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


class FiberReference:
    def __init__(self, gold_swc, pred_swc, gold, pred, radius, threshold,
                 options=None, max_pairs=4000000):
        options = dict(options or {})
        # Operate on explicit forests: original geometry, identity coordinates.
        # These options would change geometry behind the detector's back.
        if options.get('align_roots', False) or tuple(options.get('scale', (1,1,1))) != (1,1,1):
            raise ValueError('FiberMetric alignment/scaling must be applied to both inputs before detection')
        if options.get('resample_step') is not None:
            raise ValueError('Pass already resampled SWCs; detection uses FiberMetric forest mode (resample_step=None)')
        if options.get('dist_threshold', radius) != radius:
            raise ValueError('FiberMetric dist_threshold must equal distance_tolerance')
        options.update(dist_threshold=radius, resample_step=None)
        options.setdefault('query_workers', 1)
        metric = FiberMetric(**options)
        ng = sum(not gold.children[i] and gold.depth[i] >= metric.min_fiber_length
                 for i in range(len(gold)))
        npred = sum(not pred.children[i] and pred.depth[i] >= metric.min_fiber_length
                    for i in range(len(pred)))
        if max_pairs is not None and (isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1):
            raise ValueError('max_fiber_pairs must be a positive integer or None')
        if max_pairs is not None and ng*npred > max_pairs:
            raise ValueError(f'Exact FiberMetric needs {ng} x {npred} fiber pairs; '
                             f'exceeds max_fiber_pairs={max_pairs}. Increase the explicit memory budget; '
                             'scores are never silently approximated or omitted.')
        raw = metric.run(_load_resampled_forest(deepcopy(gold_swc)),
                         _load_resampled_forest(deepcopy(pred_swc)), return_fibers=True)
        self.gold_fibers, self.pred_fibers = raw['gold_fibers'] or [], raw['pred_fibers'] or []
        self.raw = plain({k:v for k,v in raw.items() if k not in ('gold_fibers','pred_fibers')})
        self.parameters = plain(vars(metric))
        self.leaves = {}
        self.catalog = {}
        for label, tree, fibers in [('gt',gold,self.gold_fibers),('pred',pred,self.pred_fibers)]:
            original = {int(nid):i for i,nid in enumerate(tree.original_id)}
            leaves = sorted((original[f[-1].nid], i) for i,f in enumerate(fibers))
            self.leaves[label] = (np.asarray([n for n,_ in leaves], dtype=int),
                                  np.asarray([f for _,f in leaves], dtype=int))
            self.catalog[label] = [dict(fiber_id=i, root_node_id=int(f[0].nid),
                                       tip_node_id=int(f[-1].nid), length=float(f.length))
                                   for i,f in enumerate(fibers)]
        matched_p = {int(m[1]) for m in (raw['matches'] or [])}
        matched_g = {int(m[0]) for m in (raw['matches'] or [])}
        self.fp = [i for i in range(len(self.pred_fibers)) if i not in matched_p]
        self.fn = [i for i in range(len(self.gold_fibers)) if i not in matched_g]
        self.references = self._references(gold, pred, radius, threshold)

    def ids(self, label, tree, index):
        nodes, ids = self.leaves[label]
        a,b = np.searchsorted(nodes, [index, tree.subtree_end[index]])
        return sorted(map(int, ids[a:b]))

    def _references(self, gold, pred, radius, threshold):
        """Range additions over GT leaf intervals; geometry queried once per edge.

        Every GT sample belongs to a contiguous interval of descendant leaf
        fibers. Union these intervals before adding length, so overlapping
        GT nodes never count the same predicted length twice. DFS push/pop
        reuses each prediction trunk for all of its terminal fibers.
        """
        nodes, fiber_ids = self.leaves['gt']
        count = len(nodes)
        if not count or not len(pred):
            return []
        lo = np.searchsorted(nodes, np.arange(len(gold)))
        hi = np.searchsorted(nodes, gold.subtree_end)
        mid = pred.xyz.copy()
        valid = pred.parent >= 0
        mid[valid] = (pred.xyz[valid]+pred.xyz[pred.parent[valid]])*.5
        candidates = _BallCandidates(gold.xyz, mid, radius)
        diff = np.zeros(count+1)
        leaves = dict(zip(*self.leaves['pred']))
        stack, result = [], []
        for i in range(len(pred)):
            while stack and stack[-1][0] != pred.parent[i]:
                _, changes = stack.pop()
                for a,b,length in changes:
                    diff[a] -= length; diff[b] += length
            intervals = sorted(set((int(lo[g]),int(hi[g])) for g in candidates[i] if hi[g]>lo[g]))
            merged = []
            for a,b in intervals:
                if merged and a <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a,b])
            changes = [(a,b,float(pred.edge_length[i])) for a,b in merged if pred.edge_length[i]>0]
            for a,b,length in changes:
                diff[a] += length; diff[b] -= length
            stack.append((i,changes))
            if i not in leaves:
                continue
            length = float(pred.depth[i])
            covered = np.clip(np.cumsum(diff[:-1]), 0, length)
            best = float(covered.max(initial=0))
            choices = np.flatnonzero((covered >= best*threshold-1e-9) & (covered > 1e-9))
            ranked = sorted((dict(gt_fiber_id=int(fiber_ids[k]), covered_length=float(covered[k]),
                                  coverage_fraction=float(covered[k]/length) if length else 0.)
                             for k in choices), key=lambda c:(-c['covered_length'],c['gt_fiber_id']))
            result.append(dict(pred_fiber_id=int(leaves[i]), metric_label='FP' if int(leaves[i]) in self.fp else 'TP',
                               length=length, status='unique' if len(ranked)==1 else 'ambiguous' if ranked else 'unsupported',
                               candidates=ranked, best_coverage_fraction=best/length if length else 0.))
        return sorted(result, key=lambda item:item['pred_fiber_id'])
