"""Public contract, analytical locations and original SWC subtree identity."""
import copy
import json

import numpy as np
import pytest

from swclib.error_detection.detection import detect_errors_legacy as detect_errors
from swclib.data.swc import Swc
from test_topology_metric import line, swc, y_tree


@pytest.mark.parametrize('step', [0.5, 0.25, 0.1])
def test_start_confirmation_and_sparse_original_edge_are_distinct(step):
    pred = swc([(0, 0, 0), (20, 0, 0), (20, 20, 0)])
    event = detect_errors(pred, line(), sample_step=step)['false_junction_events'][0]
    # Analytic tube boundary y=3, then the first failed grid point, then 3 arc.
    assert np.allclose(event['start']['xyz'], [20, 3+step, 0])
    assert np.allclose(event['anchor']['xyz'], [20, 3, 0])
    assert np.allclose(event['confirmation']['xyz'], [20, 6+step, 0])
    assert event['start']['node_id'] is None
    assert event['start']['edge_parent_id'] == 2
    assert event['start']['edge_child_id'] == 3
    assert event['start']['edge_fraction'] == pytest.approx((3+step)/20)
    assert event['affected_subtree']['node_ids'] == [3]
    assert event['confirmation_path_xyz'][0] == event['start']['xyz']
    assert event['confirmation_path_xyz'][-1] == event['confirmation']['xyz']


def test_all_events_mark_first_and_subtree_keeps_reconnected_descendants():
    pred = swc([(0, 0, 0), (20, 0, 0), (100, 0, 0), (20, 20, 0),
                (40, 20, 0), (40, 0, 0), (60, 0, 0), (60, 20, 0)],
               [-1, 0, 1, 1, 3, 4, 5, 6])
    result = detect_errors(pred, line(100))
    a, b = result['false_junction_events']
    assert a['is_first_fj'] and not b['is_first_fj']
    assert a['affected_subtree']['node_ids'] == [4, 5, 6, 7, 8]
    assert b['affected_subtree']['node_ids'] == [8]
    assert result['counts']['false_junction'] == 2
    assert result['first_fj_counts']['false_junction'] == 1
    assert result['counts']['break_error'] == result['first_fj_counts']['break_error']
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_independent_sibling_errors_are_both_first():
    pred = swc([(0, 0, 0), (20, 0, 0), (20, 20, 0), (20, -20, 0)], [-1, 0, 1, 1])
    result = detect_errors(pred, line())
    assert len(result['false_junction_events']) == 2
    assert all(e['is_first_fj'] for e in result['false_junction_events'])
    assert [e['affected_subtree']['node_ids'] for e in result['false_junction_events']] == [[3], [4]]


def test_break_start_is_on_gt_and_empty_prediction_has_seed_location():
    event = detect_errors(line(20), line())['break_events'][0]
    assert event['source'] == 'gt' and event['error_type'] == 'premature_termination'
    assert event['start']['xyz'] == [23.5, 0, 0]
    assert event['missing_segments_xyz'][-1][1] == [40, 0, 0]
    empty = detect_errors(Swc(), y_tree())['break_events'][0]
    assert empty['start']['xyz'] == [0, 0, 0] and empty['start']['node_id'] == 1
    assert empty['start']['edge_parent_id'] is None and empty['immediate_termination']


def test_objects_not_mutated_and_original_nonsequential_ids_preserved(tmp_path):
    pred = tmp_path/'pred.swc'
    pred.write_text('91 3 20 20 0 1 45\n7 1 0 0 0 1 -1\n45 3 20 0 0 1 7\n')
    obj, gold = Swc(str(pred)), line()
    before = copy.deepcopy((obj.__dict__, gold.__dict__))
    result = detect_errors(obj, gold)
    assert (obj.__dict__, gold.__dict__) == before
    event = result['false_junction_events'][0]
    assert event['start']['edge_parent_id'] == 45 and event['start']['edge_child_id'] == 91
    assert event['affected_subtree']['node_ids'] == [91]
    assert detect_errors(pred, gold) == result


@pytest.mark.parametrize('text', [
    '1 1 0 0 0 1 -1\n1 3 20 0 0 1 -1\n',
    '1 1 0 0 0 1 2\n2 3 20 0 0 1 1\n',
    '1 1 0 0 0 1 99\n', '1.2 1 0 0 0 1 -1\n',
    '1 1 nan 0 0 1 -1\n', '1 1 0 0 0 1 -1\nmalformed\n',
])
def test_malformed_files_are_rejected_without_repair(tmp_path, text):
    path = tmp_path/'bad.swc'; path.write_text(text)
    with pytest.raises(ValueError):
        detect_errors(path, line())
    assert path.read_text() == text


@pytest.mark.parametrize('kwargs', [dict(distance_tolerance=0), dict(min_error_length=float('nan')),
                                  dict(sample_step=4), dict(sample_step=-1)])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        detect_errors(line(), line(), **kwargs)


def test_empty_gt_is_rejected():
    with pytest.raises(ValueError, match='GT'):
        detect_errors(line(), Swc())
