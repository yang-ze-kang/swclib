import numpy as np

from swclib.data.postprocess import _post_merge_overlapping_paths


def _merge_paths(points, parents):
    points = np.asarray(points, dtype=np.float32)
    parents = np.asarray(parents, dtype=np.int64)
    node_types = np.full((len(points),), 3, dtype=np.int64)
    return _post_merge_overlapping_paths(
        points,
        parents,
        node_types,
        dist_sample=0.25,
        dist_threshold=0.35,
        min_overlap_length=1.0,
        min_overlap_ratio=0.2,
        min_iou=0.1,
    )


def _topology_counts(parents):
    child_counts = np.zeros((len(parents),), dtype=np.int64)
    for parent in parents:
        if int(parent) >= 0:
            child_counts[int(parent)] += 1
    return int(np.sum(child_counts >= 2)), int(np.sum(child_counts == 0))


def test_overlap_merge_preserves_divergent_tail_after_shared_prefix():
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (3.0, 0.1, 0.0),
        (3.0, 1.0, 0.0),
        (3.0, 2.0, 0.0),
        (3.0, 3.0, 0.0),
    ]
    parents = [-1, 0, 1, 2, 3, 4, 2, 6, 7, 8]

    out_points, out_parents, _, stats = _merge_paths(points, parents)

    branches, leaves = _topology_counts(out_parents)
    assert out_points.shape[0] == 9
    assert branches == 1
    assert leaves == 2
    assert np.any(np.all(np.isclose(out_points, (3.0, 3.0, 0.0)), axis=1))
    assert stats["path_logs"][1]["action"] == (
        "merge_overlapping_prefix_preserve_tail"
    )
    assert stats["path_logs"][1]["new_nodes"] == 3


def test_overlap_merge_still_deletes_fully_duplicate_branch():
    points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (3.0, 0.05, 0.0),
        (4.0, 0.05, 0.0),
        (5.0, 0.05, 0.0),
    ]
    parents = [-1, 0, 1, 2, 3, 4, 2, 6, 7]

    out_points, out_parents, _, stats = _merge_paths(points, parents)

    branches, leaves = _topology_counts(out_parents)
    assert out_points.shape[0] == 6
    assert branches == 0
    assert leaves == 1
    assert stats["path_logs"][1]["action"] == "delete_overlapping_branch"
    assert stats["path_logs"][1]["new_nodes"] == 0
