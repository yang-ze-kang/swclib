"""Analytical topology cases for the new public default (not legacy snapshots)."""
import copy
import json
import numpy as np
import pytest

from swclib.error_detection import detect_errors, detect_trace_errors
from swclib.metrics.fiber_metric import FiberMetric
from swclib.metrics._fiber_matching import _load_resampled_forest
from test_topology_metric import swc, line, y_tree
from test_trace_detection import fixture, prediction


def parallel_branches(separation=10):
    return swc([(0,0,0),(20,0,0),(60,0,0),(20,separation,0),(60,separation,0)],[-1,0,1,1,3])


def switch(separation=10):
    return swc([(0,0,0),(35,0,0),(35,separation,0),(60,separation,0)])


def assert_serializable(result):
    assert json.loads(json.dumps(result,allow_nan=False))==result


def test_original_fiber_scores_and_inputs_are_unchanged():
    gt=parallel_branches();pred=switch()
    before=copy.deepcopy((gt.__dict__,pred.__dict__))
    expected=FiberMetric(dist_threshold=3,resample_step=None,query_workers=1).run(
        _load_resampled_forest(copy.deepcopy(gt)),_load_resampled_forest(copy.deepcopy(pred)),return_fibers=True)
    result=detect_errors(pred,gt)
    for key in ('TP','FP','FN','precision','recall','f1','num_gt','num_pred','matches','FN_fiber_ids'):
        assert json.loads(json.dumps(expected[key]))==result['fiber_metric'][key]
    np.testing.assert_array_equal(expected['ious'],result['fiber_metric']['ious'])
    assert (gt.__dict__,pred.__dict__)==before
    assert_serializable(result)


def test_local_gt_detour_is_not_fj_or_break_even_when_fiber_is_fp():
    gt=swc([(0,0,0),(20,0,0),(30,-8,0),(40,0,0),(60,0,0)])
    result=detect_errors(line(60),gt)
    assert result['fiber_metric']['FP']==1
    assert result['counts']['false_junction']==result['counts']['break_error']==0
    assert result['diagnostic_counts']['geometric_deviation']==1
    assert result['fiber_explanations'][0]['status']=='explained'


@pytest.mark.parametrize('length',[20,25,30])
def test_incomplete_correct_trace_locates_actual_predicted_end(length):
    result=detect_errors(line(length),line(60))
    event=result['break_events'][0]
    assert event['error_type']=='premature_termination'
    assert event['start']['xyz']==[length,0,0]
    assert event['source']=='pred'
    assert result['counts']['false_junction']==0


def test_legal_child_continuation_is_only_a_missed_branch():
    result=detect_errors(swc([(0,0,0),(20,0,0),(40,20,0)]),y_tree())
    assert result['counts']['false_junction']==0
    assert result['counts']['missed_branch']==1
    assert result['break_events'][0]['start']['xyz']==[20,0,0]


@pytest.mark.parametrize('points',[[],[(0,0,0)],[(0,0,0)]*3])
def test_no_prediction_arc_is_premature_termination_not_disconnection(points):
    result=detect_errors(swc(points),line(100))
    assert result['immediate_termination']
    assert result['counts']['premature_termination']==1
    assert result['counts']['disconnection']==0
    assert result['counts']['break_error']==1
    assert result['break_events'][0]['gt_interval']['ends']==[]


@pytest.mark.parametrize('separation',[4,10])
def test_short_real_switch_is_not_filtered_as_short_geometry(separation):
    result=detect_errors(switch(separation),parallel_branches(separation))
    assert result['counts']['false_junction']==1
    event=result['false_junction_events'][0]
    assert event['evidence']['incompatible_gt_lineage']
    assert event['error_type']=='wrong_connection'
    if separation==4:
        assert event['short_topology_evidence']


def test_rejoining_after_a_true_wrong_connection_does_not_erase_first_error():
    pred=swc([(0,0,0),(35,0,0),(35,10,0),(45,10,0),(45,0,0),(60,0,0)])
    result=detect_errors(pred,parallel_branches())
    assert result['counts']['false_junction']==2
    assert [e['is_first_fj'] for e in result['false_junction_events']]==[True,False]
    assert result['first_fj_counts']['false_junction']==1


def test_unsupported_extra_branch_is_candidate_and_combined_count_is_explicit():
    pred=swc([(0,0,0),(20,0,0),(40,0,0),(20,20,0)],[-1,0,1,1])
    result=detect_errors(pred,line())
    assert result['counts']['false_junction']==0
    assert result['candidate_counts']['false_branch_candidate']==1
    assert result['counts_with_candidates']['false_junction_including_candidates']==1


def test_unannotated_departure_cannot_be_called_cross_neuron_merge():
    pred=swc([(0,0,0),(20,0,0),(20,20,0)])
    without=detect_errors(pred,line())
    other=swc([(0,15,0),(30,15,0),(30,30,0)])
    # Follow the neighboring annotated neuron for a sustained length.
    pred=swc([(0,0,0),(20,0,0),(20,15,0),(30,15,0),(30,30,0)])
    with_other=detect_errors(pred,line(),other_gts={'neuron-B':other})
    assert without['counts']['cross_neuron_merge']==0
    assert with_other['counts']['cross_neuron_merge']==1
    assert with_other['false_junction_events'][0]['external_neuron_candidates'][0]['neuron_id']=='neuron-B'


def test_wholly_unsupported_component_is_fragment_candidate_not_merge():
    pred=swc([(0,20,0),(40,20,0)])
    result=detect_errors(pred,line(),other_gts={'B':pred})
    assert result['counts']['cross_neuron_merge']==0
    assert result['candidate_counts']['false_fragment_candidate']==1


@pytest.mark.parametrize('gap',[1,10])
def test_disconnected_distal_geometry_does_not_establish_parent_connection(gap):
    pred=swc([(0,0,0),(20,0,0),(20+gap,0,0),(60,0,0)],[-1,0,-1,2])
    result=detect_errors(pred,line(60))
    assert result['counts']['disconnection']==1
    event=result['break_events'][0]
    assert event['gt_interval']['ends']
    assert any(p['node_id']==3 for p in event['related_pred_positions'])


def test_duplicate_distal_fragment_does_not_create_break():
    pred=swc([(0,0,0),(60,0,0),(20,0,0),(50,0,0)],[-1,0,-1,2])
    result=detect_errors(pred,line(60))
    assert result['counts']['break_error']==0
    assert result['diagnostic_counts']['duplicate_tracing']==1


def test_length_reference_does_not_weight_dense_node_clusters_more():
    gt=parallel_branches()
    sparse=switch()
    dense=swc([(0,0,0)]+[(35,i/20,0) for i in range(201)]+[(60,10,0)])
    a=detect_errors(sparse,gt)['fiber_references'][0]['candidates']
    b=detect_errors(dense,gt)['fiber_references'][0]['candidates']
    assert [x['gt_fiber_id'] for x in a]==[x['gt_fiber_id'] for x in b]
    np.testing.assert_allclose([x['coverage_fraction'] for x in a],[x['coverage_fraction'] for x in b],atol=.015)


def test_equally_supported_fibers_keep_reference_ambiguity():
    gt=swc([(0,0,0),(20,0,0),(60,1,0),(60,-1,0)],[-1,0,1,1])
    result=detect_errors(line(60),gt)
    reference=result['fiber_references'][0]
    assert reference['status']=='ambiguous' and len(reference['candidates'])==2


def test_onset_estimate_precedes_threshold_and_owns_outgoing_edge():
    pred=swc([(0,0,0),(20,0,0),(20,20,0)])
    event=detect_errors(pred,line())['candidate_events'][0]
    estimate=event['onset_estimate']
    assert event['start']['xyz']==[20,3.5,0]
    assert estimate['status']=='estimated'
    assert estimate['estimated_onset']['xyz']==[20,0,0]
    assert estimate['estimated_onset']['edge_parent_id']==2
    assert estimate['estimated_onset']['edge_child_id']==3
    assert estimate['estimated_onset']['edge_fraction']==0
    assert estimate['distance_to_reference']<3
    assert estimate['exact_onset'] is None and not estimate['calibrated']


def test_trace_default_uses_new_engine_and_candidates_keep_probabilities(tmp_path):
    path=fixture(tmp_path)
    result=detect_trace_errors(path)
    assert result['schema_version']==2 and result['counts']['false_junction']==0
    event=result['candidate_events'][0]
    assert event['trace_origin']['start']['probability']==.82
    assert event['trace_origin']['confirmation']['prediction_step']==2
    assert event['trace_origin']['estimated_onset']['prediction_step']==1
    assert_serializable(result)


def test_trace_detour_matches_swc_mode(tmp_path):
    path=fixture(tmp_path,[prediction(1,1,[20,0,0],[[2,1,60,0,0]],[[[20,0,0],[60,0,0]]])])
    gt=swc([(0,0,0),(20,0,0),(30,-8,0),(40,0,0),(60,0,0)])
    result=detect_trace_errors(path,gt)
    assert result['counts']['total']==0
    assert result['diagnostic_counts']['geometric_deviation']==1


def test_exact_metric_memory_budget_fails_explicitly_instead_of_approximating():
    with pytest.raises(ValueError,match='fiber pairs'):
        detect_errors(y_tree(),y_tree(),max_fiber_pairs=1)


def test_empty_prediction_and_point_only_gt_contract():
    assert detect_errors(swc([]),line())['counts']['premature_termination']==1
    point=swc([(0,0,0)])
    assert detect_errors(point,point)['counts']['total']==0


def test_overlapping_distal_fragment_still_requires_connection():
    pred=swc([(0,0,0),(20,0,0),(19,0,0),(60,0,0)],[-1,0,-1,2])
    result=detect_errors(pred,line(60))
    assert result['counts']['disconnection']==1
    assert result['diagnostic_counts']['duplicate_tracing']==0


def test_duplicate_rooted_fiber_is_explained_without_changing_fp():
    pred=swc([(0,0,0),(40,0,0),(0,0,0),(40,0,0)],[-1,0,-1,2])
    result=detect_errors(pred,line())
    assert result['fiber_metric']['FP']==1
    assert result['counts']['total']==0
    assert result['diagnostic_counts']['duplicate_tracing']==1


def test_shared_trunk_is_one_event_and_lists_all_affected_fibers():
    # The unsupported edge is before a new prediction fork.
    pred=swc([(0,0,0),(20,0,0),(20,20,0),(25,25,0),(15,25,0)],[-1,0,1,2,2])
    result=detect_errors(pred,line(60))
    departures=[e for e in result['candidate_events'] if e['error_type']=='unsupported_path_candidate']
    assert len(departures)==1
    assert len(departures[0]['pred_fiber_ids'])==2


def test_ambiguous_fn_is_visible_and_not_silently_treated_as_correct():
    gt=swc([(0,0,0),(20,0,0),(60,1,0),(60,-1,0)],[-1,0,1,1])
    result=detect_errors(line(60),gt)
    assert result['fiber_metric']['FN']==1
    assert result['candidate_counts']['ambiguous_missing_branch']==1


def test_trace_ambiguous_fn_has_related_predictions_without_legacy_subtype(tmp_path):
    path=fixture(tmp_path,[prediction(1,1,[20,0,0],[[2,1,60,0,0]],
                                     [[[20,0,0],[60,0,0]]])])
    gt=swc([(0,0,0),(20,0,0),(60,1,0),(60,-1,0)],[-1,0,1,1])
    result=detect_trace_errors(path,gt)
    assert result['counts']['total']==0
    assert result['candidate_counts']['ambiguous_missing_branch']==1
    event=result['candidate_events'][0]
    assert 'subtype' not in event
    assert event['status']=='undetermined' and event['source']=='gt'
    assert event['start']['xyz']==[20,0,0]
    origin=event['trace_origin']
    assert origin['status']=='related_predictions_only'
    assert origin['related_prediction_indices']==[0] and origin['probability'] is None
    assert result['trace_predictions']['0']['candidate_scores']==[.2,.4]
    assert_serializable(result)


@pytest.mark.parametrize('strategy',['fiber','legacy'])
def test_trace_missed_branch_retains_related_fork_prediction(tmp_path,strategy):
    path=fixture(tmp_path,[prediction(1,1,[20,0,0],[[2,1,40,20,0]],
                                     [[[20,0,0],[40,20,0]]])])
    result=detect_trace_errors(path,y_tree(),strategy=strategy)
    assert result['counts']['missed_branch']==1
    origin=result['break_events'][0]['trace_origin']
    assert origin['status']=='related_predictions_only'
    assert origin['related_prediction_indices']==[0] and origin['probability'] is None


def test_fast_continuation_queries_agree_with_reference_on_control_trees():
    from swclib.error_detection._matching import DenseTree, Parameters, _prediction_events
    from swclib.error_detection._subtypes import classify_junctions as reference
    from swclib.error_detection._fast_subtypes import classify_junctions as fast
    from test_topology_metric import crossing_trees
    cases=[(line(),swc([(0,0,0),(20,0,0),(40,0,0),(20,20,0)],[-1,0,1,1])),
           (y_tree(),swc([(0,0,0),(20,0,0),(40,20,0),(20,0,20)],[-1,0,1,1])),
           (crossing_trees(),crossing_trees(swapped=True))]
    params=Parameters(3,3,.5)
    for gt,prediction in cases:
        g,p=DenseTree(gt,.5),DenseTree(prediction,.5)
        events,_,_,_,states,resets=_prediction_events(g,p,params,return_states=True)
        left,right=copy.deepcopy(events),copy.deepcopy(events)
        reference(g,p,left,states,resets,params);fast(g,p,right,states,resets,params)
        assert [e['error_type'] for e in left]==[e['error_type'] for e in right]
