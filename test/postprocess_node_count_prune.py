import numpy as np

from swclib.data.postprocess import (
    _post_prune_terminal_branches_by_node_count,
    prune_terminal_branches_by_node_count,
)
from swclib.data.swc import Swc


def _make_demo_swc():
    swc = Swc()
    records = [
        (1, -1, (0.0, 0.0, 0.0)),
        (2, 1, (1.0, 0.0, 0.0)),
        (3, 2, (2.0, 0.0, 0.0)),
        (4, 3, (3.0, 0.0, 0.0)),
        (5, 3, (2.0, 1.0, 0.0)),
        (6, 3, (2.0, -1.0, 0.0)),
        (7, 6, (2.0, -2.0, 0.0)),
        (8, 4, (4.0, 0.0, 0.0)),
    ]
    for node_id, parent_id, coord in records:
        swc.nodes[node_id] = {
            "id": node_id,
            "type": 3,
            "x": coord[0],
            "y": coord[1],
            "z": coord[2],
            "radius": 1.0,
            "parent": parent_id,
        }
        swc.edges.append((node_id, parent_id))
    swc._refresh_bound_box()
    return swc


def test_array_prune_terminal_branches_by_node_count():
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, -1.0, 0.0],
            [2.0, -2.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    parents = np.asarray([-1, 0, 1, 2, 2, 2, 5, 3], dtype=np.int64)
    node_types = np.full((8,), 3, dtype=np.int64)

    out_points, out_parents, out_types, stats = _post_prune_terminal_branches_by_node_count(
        points,
        parents,
        node_types,
        min_node_count=2,
        max_iterations=1,
    )

    assert out_points.shape[0] == 7
    assert out_parents.tolist() == [-1, 0, 1, 2, 2, 4, 3]
    assert out_types.tolist() == [3, 3, 3, 3, 3, 3, 3]
    assert stats["removed_nodes"] == 1
    assert stats["removed_branches"] == 1
    assert stats["iteration_logs"][0]["branches"][0]["node_count"] == 1


def test_swc_prune_terminal_branches_by_node_count():
    swc = _make_demo_swc()

    pruned, stats = prune_terminal_branches_by_node_count(
        swc,
        min_node_count=2,
        max_iterations=1,
    )

    coords = {(node["x"], node["y"], node["z"]) for node in pruned.nodes.values()}
    assert (2.0, 1.0, 0.0) not in coords
    assert (2.0, -1.0, 0.0) in coords
    assert (2.0, -2.0, 0.0) in coords
    assert (4.0, 0.0, 0.0) in coords
    assert len(pruned.nodes) == 7
    assert stats["removed_nodes"] == 1


if __name__ == "__main__":
    test_array_prune_terminal_branches_by_node_count()
    test_swc_prune_terminal_branches_by_node_count()
