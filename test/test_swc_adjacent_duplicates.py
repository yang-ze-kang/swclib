"""Contract duplicate edges without joining crossing branches or losing roots."""
from copy import deepcopy

import numpy as np
import pytest

from swclib.data.postprocess import swc_from_arrays, swc_to_arrays


def test_coincident_separate_branches_are_preserved():
    swc = swc_from_arrays([[0, 0, 0], [1, 0, 0], [1, 1, 0],
                           [0, 1, 0], [1, 1, 0], [2, 1, 0]], [-1, 0, 1, 0, 3, 4])
    before = deepcopy(swc.nodes)
    swc.remove_duplicate_nodes(only_adjacent=True)
    assert swc.nodes == before
    swc.remove_duplicate_nodes()  # default API still performs global deduplication
    assert len(swc.nodes) == 5


def test_zero_edge_chain_keeps_ancestor_despite_reversed_node_order():
    swc = swc_from_arrays([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]],
                          [-1, 0, 1, 2, 1])
    swc.nodes = dict(reversed(list(swc.nodes.items())))
    original = deepcopy(swc.nodes)
    result = swc.remove_duplicate_nodes(only_adjacent=True, in_place=False, reindex=True)
    assert swc.nodes == original
    points, parents, _ = swc_to_arrays(result)
    root, = np.flatnonzero(parents < 0)
    assert len(points) == 3
    np.testing.assert_array_equal(points[root], [0, 0, 0])
    assert sum(parents == root) == 2


def test_adjacent_dedup_honors_rounding_and_radius():
    swc = swc_from_arrays([[0, 0, 0], [.00001, 0, 0], [1, 0, 0]], [-1, 0, 1])
    assert len(swc.remove_duplicate_nodes(only_adjacent=True, in_place=False).nodes) == 3
    assert len(swc.remove_duplicate_nodes(only_adjacent=True, round_ndigits=3, in_place=False).nodes) == 2
    swc.nodes[2]['radius'] = 2
    assert len(swc.remove_duplicate_nodes(only_adjacent=True, round_ndigits=3,
                                          use_radius=True, in_place=False).nodes) == 3


def test_adjacent_parent_cycle_is_rejected():
    swc = swc_from_arrays([[0, 0, 0], [0, 0, 0]], [-1, 0])
    ids = list(swc.nodes)
    swc.nodes[ids[0]]['parent'] = ids[1]
    with pytest.raises(ValueError, match='Cycle'):
        swc.remove_duplicate_nodes(only_adjacent=True)
