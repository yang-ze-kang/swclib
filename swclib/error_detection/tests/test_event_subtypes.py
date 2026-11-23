"""Behavioral controls for event roles, independent GT paths and grouping."""
from test_topology_metric import evaluate, PARAMS, swc, line, y_tree, crossing_trees


def roles(gold, prediction):
    result = evaluate(gold, prediction, PARAMS)
    assert result['false_junction_count'] == sum(result[k + '_count'] for k in
                                               ('wrong_connection', 'extra_branch', 'terminal_overextension'))
    return sorted(e['error_type'] for e in result['false_junction_events'])


def test_single_path_wrong_turn_replaces_continuation():
    assert roles(line(), swc([(0, 0, 0), (20, 0, 0), (20, 20, 0)])) == ['wrong_connection']


def test_extra_path_with_correct_continuation_retained():
    pred = swc([(0, 0, 0), (20, 0, 0), (40, 0, 0), (20, 20, 0)], [-1, 0, 1, 1])
    assert roles(line(), pred) == ['extra_branch']


def test_other_gt_branch_does_not_substitute_for_lost_branch():
    pred = swc([(0, 0, 0), (20, 0, 0), (40, 20, 0), (20, 0, 20)], [-1, 0, 1, 1])
    assert roles(y_tree(), pred) == ['wrong_connection']


def test_frontier_at_gt_fork_requires_both_correct_gt_children():
    gold = swc([(0, 0, 0), (20, 0, 0), (60, 0, 0), (20, 40, 0)], [-1, 0, 1, 1])
    pred = swc([(0, 0, 0), (17, 0, 0), (60, 0, 0), (17, 0, -20)], [-1, 0, 1, 1])
    assert roles(gold, pred) == ['wrong_connection']


def test_all_correct_gt_branches_plus_spurious_branch():
    pred = swc([(0, 0, 0), (20, 0, 0), (40, 20, 0), (40, -20, 0), (20, 0, 20)],
               [-1, 0, 1, 1, 1])
    assert roles(y_tree(), pred) == ['extra_branch']


def test_wrong_turn_after_fork_is_not_extra_due_to_other_correct_gt_route():
    pred = swc([(0, 0, 0), (20, 0, 0), (30, 10, 0), (30, 10, 20), (40, -20, 0)],
               [-1, 0, 1, 2, 1])
    assert roles(y_tree(), pred) == ['wrong_connection']


def test_extension_beyond_gt_leaf_is_separate():
    assert roles(line(), line(50)) == ['terminal_overextension']


def test_extension_of_one_correct_gt_branch_is_not_an_extra_branch():
    pred = swc([(0, 0, 0), (20, 0, 0), (50, 30, 0), (40, -20, 0)], [-1, 0, 1, 1])
    assert roles(y_tree(), pred) == ['terminal_overextension']


def test_short_correct_route_ending_at_gt_leaf_can_preserve_it():
    pred = swc([(0, 0, 0), (20, 0, 0), (22, 0, 0), (20, 20, 0)], [-1, 0, 1, 1])
    assert roles(line(22), pred) == ['extra_branch']


def test_sibling_reconnecting_only_after_an_error_is_not_correct_continuation():
    assert roles(crossing_trees(), crossing_trees(swapped=True)) == ['wrong_connection'] * 2


def test_later_wrong_turn_after_reconnection_is_not_a_second_extra_branch():
    pred = swc([(0, 0, 0), (20, 0, 0), (100, 0, 0), (20, 20, 0),
                (40, 20, 0), (40, 0, 0), (60, 0, 0), (60, 20, 0)],
               [-1, 0, 1, 1, 3, 4, 5, 6])
    assert roles(line(100), pred) == ['extra_branch', 'wrong_connection']


def test_two_overextensions_do_not_circularly_certify_each_other():
    pred = swc([(0, 0, 0), (40, 0, 0), (50, 0, 0), (40, 20, 0)], [-1, 0, 1, 1])
    assert roles(line(), pred) == ['terminal_overextension'] * 2


def test_zero_length_sibling_does_not_prove_preserved_continuation():
    pred = swc([(0, 0, 0), (40, 0, 0), (40, 0, 0), (40, 20, 0)], [-1, 0, 1, 1])
    assert roles(line(), pred) == ['terminal_overextension']


def test_fork_after_shared_excursion_cannot_use_downstream_reconnection_as_support():
    pred = swc([(0, 0, 0), (20, 0, 0), (20, 4, 0), (60, 4, 0),
                (60, 0, 0), (100, 0, 0), (21, 2, 0), (21, 20, 0)],
               [-1, 0, 1, 2, 3, 4, 2, 6])
    assert 'extra_branch' not in roles(line(100), pred)

