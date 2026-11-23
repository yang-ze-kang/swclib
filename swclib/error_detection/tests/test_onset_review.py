"""Review intervals are context, not new events or asserted causal onsets."""
import json

import numpy as np
import pytest

from swclib.error_detection.detection import detect_errors_legacy as detect_errors
from swclib.error_detection import detect_trace_errors as _detect_trace_errors
from functools import partial
detect_trace_errors = partial(_detect_trace_errors, strategy="legacy")
from test_topology_metric import line, swc
from test_trace_detection import fixture, prediction


def test_sparse_edge_retains_earlier_turn_without_changing_threshold_start():
    result = detect_errors(swc([(0,0,0), (20,0,0), (20,20,0)]), line())
    event = result['false_junction_events'][0]
    review = event['onset_review']
    assert event['start_role'] == 'matching_failure'
    assert event['start']['xyz'] == [20,3.5,0]
    assert review['path_xyz'] == [[0,0,0], [20,0,0], [20,3.5,0]]
    assert review['length_voxel'] == pytest.approx(23.5)
    assert review['boundary_kind'] == 'prediction_root'
    assert review['exact_onset'] is None and review['probability'] is None
    assert not review['boundary_is_error_onset'] and review['earlier_onset_not_excluded']
    assert review['segments'][-1]['end_fraction'] == pytest.approx(3.5/20)
    assert result['counts']['false_junction'] == 1
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_fork_review_follows_only_the_affected_child():
    pred = swc([(0,0,0), (20,0,0), (40,0,0), (20,20,0)], [-1,0,1,1])
    event = detect_errors(pred, line())['false_junction_events'][0]
    review = event['onset_review']
    assert review['boundary_kind'] == 'prediction_fork'
    assert review['start']['xyz'] == [20,0,0]
    assert [s['edge_child_id'] for s in review['segments']] == [4]
    assert event['error_type'] == 'extra_branch'


def test_later_event_stops_at_previous_reentry_even_inside_original_edge():
    pred = swc([(0,0,0), (20,0,0), (100,0,0), (20,20,0),
                (40,20,0), (40,0,0), (60,0,0), (60,20,0)], [-1,0,1,1,3,4,5,6])
    event = detect_errors(pred, line(100))['false_junction_events'][1]
    review = event['onset_review']
    assert review['boundary_kind'] == 'gt_reentry'
    assert review['start']['xyz'] == [40,3,0]
    assert [s['edge_child_id'] for s in review['segments']] == [6,7,8]
    assert review['segments'][0]['start_fraction'] == pytest.approx(.85)
    assert review['end'] == event['start']
    assert sum(s['length_voxel'] for s in review['segments']) == pytest.approx(review['length_voxel'])


def test_root_already_outside_has_no_invented_upstream_interval():
    result = detect_errors(swc([(0,10,0), (20,10,0)]), line())
    review = result['false_junction_events'][0]['onset_review']
    assert review['status'] == 'no_upstream_path'
    assert review['segments'] == [] and review['length_voxel'] == 0
    assert review['exact_onset'] is None


def test_zero_length_vertices_do_not_create_fake_probability_edges():
    pred = swc([(0,0,0), (20,0,0), (20,0,0), (20,20,0)])
    review = detect_errors(pred, line())['false_junction_events'][0]['onset_review']
    assert all(s['length_voxel'] > 0 for s in review['segments'])
    assert all(np.isfinite(s['start_fraction']) and np.isfinite(s['end_fraction']) for s in review['segments'])


def test_interval_includes_prior_prediction_that_started_deviation(tmp_path):
    records = [prediction(1,1,[20,0,0],[[2,1,20,2,0]],[[[20,0,0],[20,2,0]]],.91),
               prediction(2,2,[20,2,0],[[3,2,20,12,0]],[[[20,2,0],[20,12,0]]],.73)]
    result = detect_trace_errors(fixture(tmp_path, records))
    event = result['false_junction_events'][0]
    assert event['trace_origin']['start']['prediction_step'] == 2
    review = event['onset_review']
    assert review['prediction_indices'] == [0,1]
    sources = [s['trace_origin'] for s in review['segments']]
    assert sources[0]['status'] == 'initial_tree'
    assert [(s['prediction_step'], s['probability']) for s in sources[1:]] == [(1,.91),(2,.73)]
    assert review['exact_onset'] is None and review['probability'] is None
    assert result['trace_predictions']['0']['candidates'][0]['probability'] == .91


def test_deleted_predecessor_is_not_returned_as_review_source(tmp_path):
    records = [prediction(1,1,[20,0,0],[[2,1,20,2,0]],[[[20,0,0],[20,2,0]]],.99),
               prediction(2,1,[20,0,0],[[3,1,20,2,0]],[[[20,0,0],[20,2,0]]],.61,removed=[2]),
               prediction(3,3,[20,2,0],[[4,3,20,12,0]],[[[20,2,0],[20,12,0]]],.72)]
    event = detect_trace_errors(fixture(tmp_path, records))['false_junction_events'][0]
    assert event['onset_review']['prediction_indices'] == [1,2]
    assert event['trace_origin']['start']['prediction_step'] == 3


def test_shared_upstream_edge_keeps_both_candidate_probabilities(tmp_path):
    edge = [[[20,0,0],[20,2,0]]]
    candidates = [dict(index=i, probability=p, points_xyz=[[20,2,0]], actual_edges_xyz=edge,
                       stop_reason='advanced') for i,p in [(3,.91),(5,.81)]]
    records = [prediction(1,1,[20,0,0],[[2,1,20,2,0]],edge,candidates=candidates),
               prediction(2,2,[20,2,0],[[3,2,20,12,0]],[[[20,2,0],[20,12,0]]],.73)]
    review = detect_trace_errors(fixture(tmp_path, records))['false_junction_events'][0]['onset_review']
    source = next(s['trace_origin'] for s in review['segments'] if s['edge_child_id'] == 2)
    assert source['status'] == 'multiple_candidates' and source['probability'] is None
    assert [c['probability'] for c in source['candidates']] == [.91,.81]
