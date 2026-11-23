"""Behavioral fixtures: known topology errors, not implementation snapshots."""
import pytest
from swclib.data.swc import Swc

from swclib.error_detection._matching import Parameters
from swclib.error_detection.detection import detect_errors_legacy as detect_errors


def evaluate(gt, pred, params):
    return detect_errors(pred, gt, distance_tolerance=params.distance,
                         min_error_length=params.min_error_length, sample_step=params.sample_step)


PARAMS = Parameters(distance=3.0, min_error_length=3.0, sample_step=0.5)


def swc(points, parents=None):
    result = Swc()
    parents = parents if parents is not None else [-1] + list(range(len(points) - 1))
    for index, (coord, parent) in enumerate(zip(points, parents)):
        nid = index + 1
        pid = parent + 1 if parent >= 0 else -1
        result.nodes[nid] = dict(id=nid, type=3, x=coord[0], y=coord[1], z=coord[2],
                                 radius=1.0, parent=pid)
        result.edges.append((nid, pid))
    return result


def line(end=40):
    return swc([(0, 0, 0), (end, 0, 0)])


def y_tree():
    return swc([(0, 0, 0), (20, 0, 0), (40, 20, 0), (40, -20, 0)], [-1, 0, 1, 1])


def counts(result):
    return result["false_junction_count"], result["break_error_count"]


def test_identical_straight_and_branched_trees():
    for gold in (line(), y_tree()):
        assert counts(evaluate(gold, gold, PARAMS)) == (0, 0)


def test_resampling_and_node_ids_do_not_change_a_correct_path():
    pred = swc([(x, 0, 0) for x in (0, 0.3, 2.7, 11.1, 29.8, 40)])
    pred.nodes = dict(reversed(list(pred.nodes.items())))
    assert counts(evaluate(line(), pred, PARAMS)) == (0, 0)


def test_translation_within_tolerance_is_not_an_error():
    pred = swc([(0, 2.5, 0), (40, 2.5, 0)])
    assert counts(evaluate(line(), pred, PARAMS)) == (0, 0)


def test_premature_termination_is_one_event():
    result = evaluate(line(), line(20), PARAMS)
    assert counts(result) == (0, 1)
    assert result["premature_termination_count"] == 1


def test_empty_prediction_remains_a_break_sample():
    result = evaluate(y_tree(), Swc(), PARAMS)
    assert counts(result) == (0, 1)
    assert result["empty_prediction"]


@pytest.mark.parametrize("step", [0.5, 0.1])
@pytest.mark.parametrize("gt_end", [2.0, 5.5, 6.2, 40.0])
def test_seed_only_prediction_is_immediate_stop_independent_of_gt_length(gt_end, step):
    result = evaluate(line(gt_end), swc([(0, 0, 0)]), Parameters(3, 3, step))
    assert counts(result) == (0, 1)
    assert result["immediate_termination"] and result["zero_length_prediction"]
    assert not result["empty_prediction"]
    assert result["premature_termination_count"] == 1


def test_duplicate_coordinate_edges_without_tracing_are_immediate_stop():
    pred = swc([(0, 0, 0)] * 3)
    result = evaluate(line(5.5), pred, PARAMS)
    assert counts(result) == (0, 1)
    assert result["zero_length_prediction"]


def test_seed_only_before_branch_is_one_immediate_stop():
    gold = swc([(0, 0, 0), (20, 20, 0), (20, -20, 0)], [-1, 0, 0])
    result = evaluate(gold, swc([(0, 0, 0)]), PARAMS)
    assert counts(result) == (0, 1)
    assert result["premature_termination_count"] == 1
    assert result["missed_branch_count"] == 0


def test_positive_length_trace_keeps_normal_missing_length_threshold():
    result = evaluate(line(5.5), line(0.5), PARAMS)
    assert counts(result) == (0, 0)
    assert not result["immediate_termination"]


def test_point_only_gt_identity_has_no_path_to_miss():
    gold = swc([(0, 0, 0)])
    assert counts(evaluate(gold, gold, PARAMS)) == (0, 0)


def test_missing_branch():
    pred = swc([(0, 0, 0), (20, 0, 0), (40, 20, 0)])
    result = evaluate(y_tree(), pred, PARAMS)
    assert counts(result) == (0, 1)
    assert result["missed_branch_count"] == 1


def test_shallow_missing_branch_is_not_a_premature_terminal():
    gold = swc([(0, 0, 0), (20, 0, 0), (60, 0, 0), (60, 10, 0)], [-1, 0, 1, 1])
    result = evaluate(gold, line(60), PARAMS)
    assert counts(result) == (0, 1)
    assert result["missed_branch_count"] == 1


def test_actual_termination_partway_along_a_child_branch():
    pred = swc([(0, 0, 0), (20, 0, 0), (30, 10, 0), (40, -20, 0)], [-1, 0, 1, 1])
    result = evaluate(y_tree(), pred, PARAMS)
    assert counts(result) == (0, 1)
    assert result["premature_termination_count"] == 1


def test_early_stop_before_branch_does_not_count_all_descendants():
    result = evaluate(y_tree(), line(8), PARAMS)
    assert counts(result) == (0, 1)
    assert result["premature_termination_count"] == 1


def test_sustained_off_gt_path_has_junction_precedence():
    pred = swc([(0, 0, 0), (20, 0, 0), (20, 20, 0)])
    result = evaluate(line(), pred, PARAMS)
    assert counts(result) == (1, 0)
    assert len(result["suppressed_break_events"]) == 1


def test_independent_junction_and_break_in_different_branches():
    gold = swc([(0, 0, 0), (20, 0, 0), (60, 30, 0), (60, -30, 0)], [-1, 0, 1, 1])
    pred = swc([(0, 0, 0), (20, 0, 0), (36, 12, 0), (36, 12, 20), (36, -12, 0)],
               [-1, 0, 1, 2, 1])
    result = evaluate(gold, pred, PARAMS)
    assert counts(result) == (1, 1)


def test_additional_wrong_branch_does_not_hide_a_correct_gt():
    gold = line()
    pred = swc([(0, 0, 0), (20, 0, 0), (40, 0, 0), (20, 20, 0)], [-1, 0, 1, 1])
    assert counts(evaluate(gold, pred, PARAMS)) == (1, 0)


def test_short_off_gt_tip_below_persistence_threshold():
    pred = swc([(0, 0, 0), (40, 0, 0), (40, 4, 0)])
    assert counts(evaluate(line(), pred, PARAMS)) == (0, 0)


def test_two_separate_excursions_are_two_events():
    pred = swc([(0, 0, 0), (10, 0, 0), (10, 10, 0), (20, 10, 0), (20, 0, 0),
                (30, 0, 0), (30, 10, 0), (40, 10, 0), (40, 0, 0), (50, 0, 0)])
    result = evaluate(line(50), pred, PARAMS)
    assert counts(result) == (2, 0)


def test_break_after_reconnecting_is_independent():
    pred = swc([(0, 0, 0), (10, 0, 0), (10, 10, 0), (20, 10, 0), (20, 0, 0), (30, 0, 0)])
    assert counts(evaluate(line(50), pred, PARAMS)) == (1, 1)


def crossing_trees(swapped=False):
    # Both routes cross at (40,0), but those coincident vertices are distinct.
    points = [(0, 0, 0), (10, 0, 0), (20, 10, 0), (30, 10, 0), (40, 0, 0),
              (60, 10, 0), (20, -10, 0), (30, -10, 0), (40, 0, 0), (60, -10, 0)]
    parents = [-1, 0, 1, 2, 3, 4, 1, 6, 7, 8]
    if swapped:
        parents[5], parents[9] = 8, 4
    return swc(points, parents)


def test_geometrically_correct_crossing_is_not_a_junction():
    gold = crossing_trees()
    assert counts(evaluate(gold, gold, PARAMS)) == (0, 0)


def test_wrong_parent_at_crossing_is_detected_despite_full_spatial_coverage():
    result = evaluate(crossing_trees(), crossing_trees(swapped=True), PARAMS)
    assert counts(result) == (2, 0)
    assert all(e["subtype"] == "within_gt_topology_switch" for e in result["false_junction_events"])


def test_shared_wrong_trunk_is_one_event_even_if_it_branches():
    pred = swc([(0, 0, 0), (20, 0, 0), (20, 10, 0), (30, 20, 0), (10, 20, 0)],
               [-1, 0, 1, 2, 2])
    assert counts(evaluate(line(), pred, PARAMS)) == (1, 0)


def test_root_gap_is_a_break_not_a_path_switch():
    pred = swc([(10, 0, 0), (40, 0, 0)])
    assert counts(evaluate(line(), pred, PARAMS)) == (0, 1)


def test_continuous_reversal_is_one_diagnostic_and_missing_gt_is_break():
    pred = swc([(0, 0, 0), (25, 0, 0), (5, 0, 0)])
    result = evaluate(line(60), pred, PARAMS)
    assert counts(result) == (0, 1)
    assert result["reversal_count"] == 1
    assert result["reversal_diagnostics"][0]["direction_reversal"]
    assert not result["suppressed_break_events"]


def test_separate_reversals_after_forward_tracing_remain_separate():
    pred = swc([(0, 0, 0), (30, 0, 0), (10, 0, 0), (30, 0, 0), (10, 0, 0)])
    result = evaluate(line(60), pred, PARAMS)
    assert counts(result) == (0, 1)
    assert result["reversal_count"] == 2


def test_reversal_after_complete_trace_does_not_invent_a_break():
    pred = swc([(0, 0, 0), (40, 0, 0), (5, 0, 0)])
    result = evaluate(line(), pred, PARAMS)
    assert counts(result) == (0, 0)
    assert result["reversal_count"] == 1


def test_reversal_followed_by_off_gt_excursion_keeps_both_types():
    pred = swc([(0, 0, 0), (25, 0, 0), (5, 0, 0), (5, 20, 0)])
    result = evaluate(line(60), pred, PARAMS)
    assert counts(result) == (1, 1)
    assert result["reversal_count"] == 1


def test_off_gt_detour_returning_to_ancestor_is_still_junction():
    pred = swc([(0, 0, 0), (25, 0, 0), (25, 10, 0), (5, 10, 0), (5, 0, 0)])
    result = evaluate(line(60), pred, PARAMS)
    assert counts(result) == (1, 0)
    assert result["reversal_count"] == 0


def test_reject_empty_gt():
    with pytest.raises(ValueError, match="GT must"):
        evaluate(Swc(), line(), PARAMS)
