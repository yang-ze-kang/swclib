"""Regression comparisons against the original scalar FiberMetric geometry."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial import cKDTree

from swclib.data.swc import Swc
from swclib.data.swc_forest import SwcForest
from swclib.data.swc_node import SwcNode
from swclib.data.euclidean_point import EuclideanPoint3D
import swclib.data.swc_fiber as fiber_module
import swclib.metrics.fiber_metric as metric_module
import swclib.metrics._fiber_matching as matching
from swclib.metrics.fiber_metric import FiberMetric
from swclib.utils.points import point_pair_distance


def _scalar_resample(nodes, distance=2, eps=1e-8):
    """The original scalar sampler, including dtype and endpoint handling."""
    nodes = np.asarray(nodes)
    diffs = np.diff(nodes, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    result = []
    for t in np.arange(0, cum_len[-1], distance):
        index = np.searchsorted(cum_len, t)
        if index == 0:
            result.append(nodes[0])
        else:
            ratio = (t - cum_len[index - 1]) / seg_lens[index - 1]
            result.append(nodes[index - 1] + ratio * diffs[index - 1])
    if len(result) == 0 or np.linalg.norm(result[-1] - nodes[-1]) > eps:
        result.append(nodes[-1])
    return np.array(result)


def _scalar_ious(gold, pred, metric, skip_center_dist, verbose=False):
    """Keep the pre-optimization pair loop as an independent test oracle."""
    lengths_gold = [f.length for f in gold]
    lengths_pred = [f.length for f in pred]
    ious = np.zeros((len(gold), len(pred)))
    for i, first in enumerate(gold):
        for j, second in enumerate(pred):
            a, b = lengths_gold[i], lengths_pred[j]
            if (a / (b + metric.eps) < min(0.5, metric.iou_threshold)
                    or b / (a + metric.eps) < min(0.5, metric.iou_threshold)):
                ious[i, j] = min(a, b) / max(a, b)
                continue
            if metric.with_direction:
                if (first[0].distance(second[0]) > 0.5 * max(a, b)
                        or first[-1].distance(second[-1]) > 0.5 * max(a, b)):
                    ious[i, j] = 0.01
                    continue
            if point_pair_distance(first.center, second.center) > skip_center_dist:
                ious[i, j] = 0.01
                continue
            ious[i, j] = first.cal_iou(
                second, dist_sample=metric.dist_sample, dist_threshold=metric.dist_threshold
            )
    return ious


def _tree(seed=0, offset=(0., 0., 0.)):
    rng = np.random.default_rng(seed)
    swc = Swc()
    rows = [(-1, 1, [0., 0., 0.]), (1, 3, [3., 0., 0.]), (2, 3, [7., 0., 0.])]
    for branch in range(6):
        parent = 3 if branch < 4 else 2
        for step in range(1, 5):
            xyz = np.array([7. + step * (1. + branch), branch * step * .8, step * .3])
            xyz += rng.normal(0, .2, 3)
            rows.append((parent, (2, 3, 4)[branch % 3], xyz))
            parent = len(rows)
    for nid, (parent, kind, xyz) in enumerate(rows, 1):
        xyz = np.asarray(xyz) + offset
        swc.nodes[nid] = dict(id=nid, parent=parent, type=kind,
                              x=xyz[0], y=xyz[1], z=xyz[2], radius=1.)
        swc.edges.append((nid, parent))
    return swc


def _write(path, swc):
    path.write_text("".join(
        f"{n['id']} {n['type']} {n['x']:.17g} {n['y']:.17g} {n['z']:.17g} 1 {n['parent']}\n"
        for n in swc.nodes.values()
    ))
    return path


def _fiber_signature(fibers):
    if fibers is None:
        return None
    return [[(n.nid, n.ntype, tuple(n.coord), n.root_length) for n in f.nodes] for f in fibers]


def _assert_results_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if key == "ious" and actual[key] is not None:
            np.testing.assert_array_equal(actual[key], expected[key])
        elif key in ("gold_fibers", "pred_fibers"):
            assert _fiber_signature(actual[key]) == _fiber_signature(expected[key])
        else:
            assert actual[key] == expected[key], key


def _reference_run(monkeypatch, metric, *args, **kwargs):
    with monkeypatch.context() as context:
        context.setattr(metric_module, "_fiber_iou_matrix", _scalar_ious)
        context.setattr(metric_module, "_load_resampled_forest", SwcForest)
        context.setattr(fiber_module, "resample_nodes_by_distance", _scalar_resample)
        return metric.run(*args, **kwargs)


@pytest.mark.parametrize("seed", [0, 5, 21])
@pytest.mark.parametrize("options,center_limit", [
    ({}, np.inf),
    ({}, 0.),
    ({"resample_step": None}, 100.),
    ({"resample_step": .75, "dist_sample": .7}, 6.),
    ({"scale": (.35, 1.2, 2.1)}, np.inf),
    ({"align_roots": True}, np.inf),
    ({"with_direction": True, "use_category": True}, 10.),
    ({"only_from_soma": True, "min_fiber_length": 12.}, np.inf),
    ({"iou_threshold": .3, "eps": .2}, np.inf),
    ({"dist_threshold": 0.}, np.inf),
    ({"dist_threshold": np.inf}, np.inf),
])
def test_complete_results_match_scalar_reference(tmp_path, monkeypatch, seed, options, center_limit):
    gold = _write(tmp_path / "gold.swc", _tree(seed))
    pred = _write(tmp_path / "pred.swc", _tree(seed + 1, (1., -.25, .5)))
    expected = _reference_run(monkeypatch, FiberMetric(**options), gold, pred,
                              skip_center_dist=center_limit, return_fibers=True)
    actual = FiberMetric(**options).run(gold, pred, skip_center_dist=center_limit, return_fibers=True)
    _assert_results_equal(actual, expected)


@pytest.mark.parametrize("point_limit", [7, 100, 100000])
def test_batch_boundaries_and_candidate_subsets_preserve_reductions(point_limit):
    gold = SwcForest(_tree(9)).get_fibers()
    pred = SwcForest(_tree(10, (3., 1., 0.))).get_fibers()
    metric = FiberMetric(dist_sample=.7, dist_threshold=2.5, batch_point_limit=point_limit)
    expected = _scalar_ious(gold, pred, metric, 9.)
    actual = matching._fiber_iou_matrix(gold, pred, metric, 9.)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("threshold", [0., 1e-160, 1e-120, .125, 5., 1e150, np.inf, -1., np.nan])
def test_bounded_query_preserves_inclusive_distance_boundary(threshold):
    rng = np.random.default_rng(239)
    radius = threshold if np.isfinite(threshold) and threshold > 0 else 1.
    directions = rng.normal(size=(300, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    points = np.concatenate([
        directions * np.nextafter(radius, 0.), directions * radius,
        directions * np.nextafter(radius, np.inf), np.zeros((3, 3)),
    ])
    tree = cKDTree(np.zeros((1, 3)))
    distances, _ = tree.query(points)
    expected = distances <= threshold
    np.testing.assert_array_equal(matching._within_distance(tree, points, threshold), expected)


def test_iou_threshold_boundary_and_ties_are_unchanged(tmp_path, monkeypatch):
    gold = _write(tmp_path / "gold.swc", _tree(12))
    pred = _write(tmp_path / "pred.swc", _tree(13))
    reference = _reference_run(monkeypatch, FiberMetric(), gold, pred,
                               skip_center_dist=np.inf, return_fibers=True)
    score = float(reference["ious"].max())
    for threshold in (np.nextafter(score, 0.), score, np.nextafter(score, np.inf)):
        options = dict(iou_threshold=threshold)
        expected = _reference_run(monkeypatch, FiberMetric(**options), gold, pred,
                                  skip_center_dist=np.inf, return_fibers=True)
        actual = FiberMetric(**options).run(gold, pred, skip_center_dist=np.inf, return_fibers=True)
        _assert_results_equal(actual, expected)


def test_identical_branches_keep_hungarian_tie_selection(tmp_path, monkeypatch):
    path = tmp_path / "ties.swc"
    path.write_text(
        "1 1 0 0 0 1 -1\n2 3 5 0 0 1 1\n"
        "3 3 10 0 0 1 2\n4 3 10 0 0 1 2\n"
    )
    expected = _reference_run(monkeypatch, FiberMetric(), path, path,
                              skip_center_dist=np.inf, return_fibers=True)
    actual = FiberMetric().run(path, path, skip_center_dist=np.inf, return_fibers=True)
    assert actual["ious"].shape == (2, 2)
    assert len(np.unique(actual["ious"])) == 1
    _assert_results_equal(actual, expected)


def test_repeated_runs_do_not_cache_mutable_forest_geometry(monkeypatch):
    expected_forests = (SwcForest(_tree()), SwcForest(_tree(1)))
    actual_forests = (SwcForest(_tree()), SwcForest(_tree(1)))
    metric = FiberMetric(scale=(.7, 1.2, 1.1), min_fiber_length=0.)
    for _ in range(2):
        expected = _reference_run(monkeypatch, metric, *expected_forests,
                                  skip_center_dist=np.inf, return_fibers=True)
        actual = metric.run(*actual_forests, skip_center_dist=np.inf, return_fibers=True)
        _assert_results_equal(actual, expected)


@pytest.mark.parametrize("empty_gold,empty_pred", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("return_fibers", [False, True])
def test_empty_forests_keep_existing_result_structure(monkeypatch, empty_gold, empty_pred, return_fibers):
    def inputs():
        return (SwcForest() if empty_gold else SwcForest(_tree()),
                SwcForest() if empty_pred else SwcForest(_tree(1)))
    expected = _reference_run(monkeypatch, FiberMetric(), *inputs(), return_fibers=return_fibers)
    actual = FiberMetric().run(*inputs(), return_fibers=return_fibers)
    _assert_results_equal(actual, expected)


def _forest_signature(forest):
    return [(n.nid, n.parent.nid if n.parent else -1,
             tuple(c.nid for c in n.children), n.root_length, tuple(n.coord))
            for n in forest.get_node_list()]


@pytest.mark.parametrize("order", ["original", "reverse", "shuffle"])
def test_faster_loading_preserves_order_coordinates_and_root_lengths(order):
    swc = _tree(4)
    # Include another root to exercise root ordering as well as sibling order.
    swc.nodes[100] = dict(id=100, parent=-1, type=1, x=50., y=0., z=0., radius=1.)
    swc.nodes[101] = dict(id=101, parent=100, type=3, x=60., y=0., z=0., radius=1.)
    items = list(swc.nodes.items())
    if order == "reverse":
        items.reverse()
    elif order == "shuffle":
        np.random.default_rng(5).shuffle(items)
    swc.nodes = dict(items)
    original = deepcopy(swc.nodes)
    expected = SwcForest(swc)
    actual = matching._load_resampled_forest(swc)
    assert _forest_signature(actual) == _forest_signature(expected)
    assert _fiber_signature(actual.get_fibers()) == _fiber_signature(expected.get_fibers())
    assert list(swc.nodes.items()) == list(original.items())


@pytest.mark.parametrize("invalid", ["cycle", "self_parent", "orphan", "duplicate_id"])
def test_invalid_topology_keeps_original_loader_errors(invalid):
    swc = _tree()
    if invalid == "cycle":
        swc.nodes[1]["parent"] = 3
    elif invalid == "self_parent":
        swc.nodes[3]["parent"] = 3
    elif invalid == "orphan":
        swc.nodes[3]["parent"] = 999
    else:
        swc.nodes[3]["id"] = 2
    with pytest.raises(Exception) as original:
        SwcForest(swc)
    with pytest.raises(type(original.value)):
        matching._load_resampled_forest(swc)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int64, np.longdouble])
@pytest.mark.parametrize("step", [.125, .7, 1., 2., 1000.])
def test_resampling_preserves_scalar_values_and_dtype(dtype, step):
    rng = np.random.default_rng(87)
    curve = rng.integers(0, 4, size=(32, 3)).cumsum(axis=0).astype(dtype)
    for points in (curve, np.repeat(curve, 2, axis=0), curve[:1], np.zeros((4, 3), dtype=dtype)):
        expected = _scalar_resample(points, step)
        actual = fiber_module.resample_nodes_by_distance(points, step)
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
        if dtype == np.float64:
            assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("eps", [0., 1e-8])
def test_resampling_preserves_endpoint_rounding_and_large_coordinates(eps):
    for end in (np.nextafter(2., 0.), 2., np.nextafter(2., np.inf)):
        for offset in (0., 20000.):
            points = np.array([[-0., 0., 0.], [1., .5, 0.], [end, 0., 0.]]) + offset
            expected = _scalar_resample(points, .5, eps)
            actual = fiber_module.resample_nodes_by_distance(points, .5, eps)
            assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("first,second", [
    ([], []), ([], [0]), ([0], [0]), ([0], [1]),
    ([0, 1, 2], [0, 1, 2]), ([0, 1, 2], [2, 1, 0]),
    ([0, 1, 2], [0, 3, 2]), ([0, 1, 2], [0, 1]),
    ([0, 1, 2, 0], [0, 2, 1, 0]),
])
def test_fiber_equality_preserves_forward_and_reverse_matching(first, second):
    nodes = [SwcNode(nid=i, coord=[i, 0., 0.]) for i in range(4)]
    a, b = fiber_module.SwcFiber(), fiber_module.SwcFiber()
    a.nodes = [nodes[i] for i in first]
    b.nodes = [nodes[i] for i in second]
    expected = a.nodes == b.nodes or a.nodes[::-1] == b.nodes
    assert (a == b) == expected


def test_fiber_coordinates_are_copies_and_follow_geometry_changes():
    fiber = fiber_module.SwcFiber()
    assert fiber.coords.shape == (0,)
    fiber.nodes = [SwcNode(nid=1, coord=[1., 2., 3.]), SwcNode(nid=2, coord=[4., 5., 6.])]
    expected = np.array([node.coord for node in fiber.nodes])
    np.testing.assert_array_equal(fiber.coords, expected)
    coords = fiber.coords
    coords[0, 0] = 999.
    np.testing.assert_array_equal(fiber.coords, expected)
    fiber.nodes[0].coord[0] = 12.
    assert fiber.coords[0, 0] == 12.


def test_fiber_coordinates_preserve_nonstandard_point_protocols():
    class ShiftedPoint(EuclideanPoint3D):
        def __getitem__(self, index):
            return super().__getitem__(index) + 10.
    fiber = fiber_module.SwcFiber()
    fiber.nodes = [SwcNode(nid=i, coord=[i, 0., 0.]) for i in range(3)]
    fiber.nodes[0].coord = [1., 2., 3.]
    fiber.nodes[1].coord = np.array([4., 5., 6.])
    fiber.nodes[2].coord = ShiftedPoint([7., 8., 9.])
    np.testing.assert_array_equal(fiber.coords, np.array([node.coord for node in fiber.nodes]))


@pytest.mark.parametrize("workers,minimum,expected_workers", [
    (4, 5, [4, 4, 1]), (1, 5, [1, 1, 1]),
    (2, 0, [2, 2, 2]), (-1, 5, [-1, -1, 1]), (4, 6, [1, 1, 1]),
])
def test_parallel_queries_respect_each_chunk_size(monkeypatch, workers, minimum, expected_workers):
    monkeypatch.setattr(matching, "_QUERY_POINT_LIMIT", 5)
    tree = cKDTree(np.array([[0., 0., 0.], [3., 0., 0.]]))
    calls = []
    class RecordingTree:
        def query(self, points, *, workers, **kwargs):
            calls.append((len(points), workers))
            return tree.query(points, **kwargs)
    points = np.column_stack([np.arange(12), np.zeros((12, 2))])
    actual = matching._within_distance(RecordingTree(), points, 1.,
                                      query_workers=workers, parallel_min_points=minimum)
    np.testing.assert_array_equal(actual, tree.query(points)[0] <= 1.)
    assert calls == list(zip([5, 5, 2], expected_workers))
    matching._within_distance(RecordingTree(), points[:0], 1., query_workers=workers)
    assert len(calls) == 3


@pytest.mark.parametrize("options", [
    dict(query_workers=1, parallel_min_points=0, batch_point_limit=7),
    dict(query_workers=2, parallel_min_points=0, batch_point_limit=7),
    dict(query_workers=2, parallel_min_points=50, batch_point_limit=100),
])
def test_query_and_batch_options_preserve_complete_results(tmp_path, monkeypatch, options):
    gold = _write(tmp_path / "gold.swc", _tree(22))
    pred = _write(tmp_path / "pred.swc", _tree(23, (1., -.5, .25)))
    expected = _reference_run(monkeypatch, FiberMetric(), gold, pred,
                              skip_center_dist=np.inf, return_fibers=True)
    actual = FiberMetric(**options).run(gold, pred, skip_center_dist=np.inf, return_fibers=True)
    _assert_results_equal(actual, expected)


@pytest.mark.parametrize("name,value", [
    ("query_workers", 0), ("query_workers", -2), ("query_workers", 2.5),
    ("query_workers", True), ("parallel_min_points", -1), ("parallel_min_points", 1.5),
    ("parallel_min_points", False), ("batch_point_limit", 0), ("batch_point_limit", -1),
    ("batch_point_limit", None), ("batch_point_limit", "100"), ("batch_point_limit", True),
])
def test_invalid_performance_options_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        FiberMetric(**{name: value})


def test_performance_options_accept_numpy_integers():
    metric = FiberMetric(query_workers=np.int64(2), parallel_min_points=np.int64(0),
                         batch_point_limit=np.int64(100))
    assert (metric.query_workers, metric.parallel_min_points, metric.batch_point_limit) == (2, 0, 100)
