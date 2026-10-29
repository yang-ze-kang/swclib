"""Internal, exact geometry preparation and batching for FiberMetric."""

from collections import deque
from copy import copy
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from swclib.data.swc_forest import SwcForest
from swclib.utils.points import point_pair_distance


# Query buffers stay bounded independently of the configurable deduplication
# batch. A longer source fiber stays intact to preserve its np.sum order.
_QUERY_POINT_LIMIT = 262144


def _load_resampled_forest(swc):
    """Use the normal loader, attaching children before their parents.

    Otherwise anytree checks an already attached ancestor chain for every
    node. Validate the topology first, and preserve root and sibling order.
    Unusual or invalid records retain the original loader's behavior.
    """
    children, roots = {nid: [] for nid in swc.nodes}, []
    for nid, record in swc.nodes.items():
        if int(record.get("id", nid)) != nid:
            return SwcForest(swc)
        parent = int(record["parent"])
        if parent == -1:
            roots.append(nid)
        elif parent not in children:
            return SwcForest(swc)
        else:
            children[parent].append(nid)

    depths = {nid: 0 for nid in roots}
    pending = deque(roots)
    while pending:
        nid = pending.popleft()
        for child in children[nid]:
            depths[child] = depths[nid] + 1
            pending.append(child)
    if len(depths) != len(swc.nodes):
        return SwcForest(swc)

    ordered = copy(swc)
    ordered.nodes = dict(
        sorted(swc.nodes.items(), key=lambda item: -depths[item[0]])
    )
    return SwcForest(ordered)


@dataclass
class _Samples:
    midpoints: np.ndarray
    lengths: np.ndarray
    tree: object


def _samples(fiber, distance, cache):
    key = id(fiber)
    if key not in cache:
        # Reuse the existing sampling convention and tree cache verbatim.
        coords, tree = fiber.cache_resample_by_distance(distance)
        cache[key] = _Samples(
            (coords[:-1] + coords[1:]) * 0.5,
            np.linalg.norm(coords[1:] - coords[:-1], axis=1),
            tree,
        )
    return cache[key]


def _within_distance(tree, points, threshold, *, query_workers=1, parallel_min_points=65536):
    """The original query-then-<= mask, with a conservative search bound."""
    upper = np.inf
    if np.isfinite(threshold) and threshold >= 0:
        # cKDTree's upper bound is strict. Also avoid its squared bound
        # underflowing to zero for zero or extremely small thresholds.
        upper = np.nextafter(
            max(float(threshold), np.sqrt(np.finfo(float).tiny)), np.inf
        )
    hit = np.empty(len(points), dtype=bool)
    for start in range(0, len(points), _QUERY_POINT_LIMIT):
        end = start + _QUERY_POINT_LIMIT
        chunk = points[start:end]
        workers = query_workers if len(chunk) >= parallel_min_points else 1
        distances, _ = tree.query(chunk, distance_upper_bound=upper, workers=workers)
        hit[start:end] = distances <= threshold
    return hit


def _source_batches(fibers, active, distance, cache, batch_point_limit):
    batch, point_count = [], 0
    for index in np.flatnonzero(active):
        data = _samples(fibers[index], distance, cache)
        size = len(data.midpoints)
        if batch and point_count + size > batch_point_limit:
            yield batch
            batch, point_count = [], 0
        batch.append((index, data))
        point_count += size
    if batch:
        yield batch


def _directed_overlaps(
    source, target, eligible, distance, threshold, cache, verbose,
    *, batch_point_limit, query_workers, parallel_min_points,
):
    overlap = np.zeros(eligible.shape)
    query_options = dict(query_workers=query_workers, parallel_min_points=parallel_min_points)
    with tqdm(
        total=int(eligible.sum()), desc="Calculating fiber overlaps", disable=not verbose
    ) as progress:
        for batch in _source_batches(source, eligible.any(axis=1), distance, cache, batch_point_limit):
            indices = [index for index, _ in batch]
            active_targets = np.flatnonzero(eligible[indices].any(axis=0))
            if len(batch) == 1:
                index, data = batch[0]
                for target_index in active_targets:
                    tree = _samples(target[target_index], distance, cache).tree
                    hit = _within_distance(tree, data.midpoints, threshold, **query_options)
                    overlap[index, target_index] = np.sum(data.lengths[hit])
                    progress.update(1)
                continue

            sizes = [len(data.midpoints) for _, data in batch]
            offsets = np.cumsum([0] + sizes)
            points, inverse = np.unique(
                np.concatenate([data.midpoints for _, data in batch]),
                axis=0,
                return_inverse=True,
            )
            maps = [inverse[offsets[k]:offsets[k + 1]] for k in range(len(batch))]
            selected = np.zeros(len(points), dtype=bool)
            hit = np.zeros(len(points), dtype=bool)
            for target_index in active_targets:
                active = np.flatnonzero(eligible[indices, target_index])
                selected.fill(False)
                for k in active:
                    selected[maps[k]] = True
                selected_indices = np.flatnonzero(selected)
                tree = _samples(target[target_index], distance, cache).tree
                hit[selected_indices] = _within_distance(
                    tree, points[selected_indices], threshold, **query_options
                )
                for k in active:
                    index, data = batch[k]
                    # Do not use sparse matrix sums or reorder segments:
                    # threshold decisions and Hungarian ties need the exact
                    # same floating-point IoUs as the scalar implementation.
                    overlap[index, target_index] = np.sum(data.lengths[hit[maps[k]]])
                progress.update(len(active))
    return overlap


def _fiber_iou_matrix(gold, pred, metric, skip_center_dist, verbose=False):
    lengths_gold = [fiber.length for fiber in gold]
    lengths_pred = [fiber.length for fiber in pred]
    centers_gold, centers_pred = {}, {}
    ious = np.zeros((len(gold), len(pred)))
    eligible = np.zeros(ious.shape, dtype=bool)
    minimum_ratio = min(0.5, metric.iou_threshold)
    check_centers = not np.isposinf(skip_center_dist)

    for i, gold_fiber in enumerate(gold):
        for j, pred_fiber in enumerate(pred):
            first, second = lengths_gold[i], lengths_pred[j]
            if (
                first / (second + metric.eps) < minimum_ratio
                or second / (first + metric.eps) < minimum_ratio
            ):
                # These nonzero scores participate in the global assignment.
                ious[i, j] = min(first, second) / max(first, second)
                continue
            if metric.with_direction:
                max_length = max(first, second)
                if (
                    gold_fiber[0].distance(pred_fiber[0]) > 0.5 * max_length
                    or gold_fiber[-1].distance(pred_fiber[-1]) > 0.5 * max_length
                ):
                    ious[i, j] = 0.01
                    continue
            if check_centers:
                if i not in centers_gold:
                    centers_gold[i] = gold_fiber.center
                if j not in centers_pred:
                    centers_pred[j] = pred_fiber.center
                if point_pair_distance(centers_gold[i], centers_pred[j]) > skip_center_dist:
                    ious[i, j] = 0.01
                    continue
            if len(gold_fiber) <= 1 or len(pred_fiber) <= 1:
                ious[i, j] = gold_fiber.cal_iou(
                    pred_fiber, dist_sample=metric.dist_sample,
                    dist_threshold=metric.dist_threshold,
                )
                continue
            # Preserve the separate length check inside SwcFiber.cal_iou.
            if first / second < 0.5 or second / first < 0.5:
                continue
            eligible[i, j] = True

    if not eligible.any():
        return ious
    # Everything here is local to this run, after root alignment and scaling.
    cache = {}
    options = dict(
        batch_point_limit=metric.batch_point_limit,
        query_workers=metric.query_workers,
        parallel_min_points=metric.parallel_min_points,
    )
    forward = _directed_overlaps(
        gold, pred, eligible, metric.dist_sample, metric.dist_threshold, cache, verbose, **options
    )
    reverse = _directed_overlaps(
        pred, gold, eligible.T, metric.dist_sample, metric.dist_threshold, cache, verbose, **options
    )
    for i, j in np.argwhere(eligible):
        overlap = (forward[i, j] + reverse[j, i]) / 2.0
        union = lengths_gold[i] + lengths_pred[j] - overlap
        ious[i, j] = overlap / (union + 1e-7)  # SwcFiber.cal_iou's epsilon
    return ious
