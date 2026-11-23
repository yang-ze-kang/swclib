"""Independent recording fixtures: provenance, deletion, ambiguity and GT transforms."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from swclib.data.swc import Swc
from swclib.error_detection import detect_trace_errors as _detect_trace_errors
from functools import partial
detect_trace_errors = partial(_detect_trace_errors, strategy="legacy")
from swclib.error_detection import trace_detection as module


def prediction(step, root, seed, added, edges, probability=.8, removed=(), candidates=None):
    if candidates is None:
        candidates=[dict(index=7,probability=probability,points_xyz=[e[1] for e in edges],
                         actual_edges_xyz=edges,stop_reason='advanced')]
    return dict(event='prediction',step=step,root=root,seed_xyz=seed,added_nodes=added,
                removed_nodes=list(removed),candidates=candidates,scores=[.2,.4],
                threshold_candidates=[],forward_steps=5,window_shape_xyz=[64,64,32])


def fixture(folder, records=None, initial=None, manifest=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    initial=initial or ([[0,0,0],[20,0,0]],[-1,0])
    np.savez(folder/'trace_initial.npz',points_xyz=initial[0],parents=initial[1])
    gt=folder/'gt.swc';gt.write_text('1 1 0 0 0 1 -1\n2 3 40 0 0 1 1\n')
    metadata=dict(ground_truth='gt.swc',swc_to_voxel_scale_xyz=[1,1,1],swc_to_voxel_offset_xyz=[0,0,0])
    metadata.update(manifest or {})
    (folder/'trace_replay.json').write_text(json.dumps(metadata))
    if records is None:
        records=[prediction(1,1,[20,0,0],[[2,1,20,4,0]],[[[20,0,0],[20,4,0]]],.82),
                 prediction(2,2,[20,4,0],[[3,2,20,12,0]],[[[20,4,0],[20,12,0]]],.93)]
    header=dict(event='header',version=2,mode='direct',coordinate_frame='global_xyz_voxels',seed_xyz=[0,0,0])
    (folder/'trace_steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in [header,*records,dict(event='end',status='complete')]))
    return folder/'trace_steps.jsonl'


def test_first_failed_edge_and_later_confirmation_have_distinct_sources(tmp_path):
    path=fixture(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.iterdir()}
    result=detect_trace_errors(path,include_geometry=True)
    event=result['false_junction_events'][0]
    assert event['start']['xyz']==[20,3.5,0] and event['confirmation']['xyz']==[20,6.5,0]
    source=event['trace_origin']
    assert source['start']['prediction_step']==1 and source['start']['probability']==.82
    assert source['start']['prediction_index']==0 and source['start']['log_line']==2
    assert source['confirmation']['prediction_step']==2 and source['confirmation']['probability']==.93
    assert source['confirmation_prediction_indices']==[0,1]
    assert event['affected_subtree']['node_ids']==[2,3]
    assert result['trace']['retained_node_count']==4
    assert before=={p:p.read_bytes() for p in tmp_path.iterdir()}
    assert json.loads(json.dumps(result,allow_nan=False))==result


def test_removed_error_is_not_counted_and_reused_geometry_uses_new_owner(tmp_path):
    first=prediction(1,1,[20,0,0],[[2,1,20,12,0]],[[[20,0,0],[20,12,0]]],.99)
    correct=prediction(2,1,[20,0,0],[[3,1,40,0,0]],[[[20,0,0],[40,0,0]]],.7,removed=[2])
    assert detect_trace_errors(fixture(tmp_path/'correct',[first,correct]))['counts']['total']==0
    repeated=prediction(2,1,[20,0,0],[[3,1,20,12,0]],[[[20,0,0],[20,12,0]]],.65,removed=[2])
    result=detect_trace_errors(fixture(tmp_path/'repeated',[first,repeated]))
    source=result['false_junction_events'][0]['trace_origin']['start']
    assert source['prediction_step']==2 and source['probability']==.65 and source['recorded_node_id']==3


def test_shared_prefix_keeps_all_candidates_without_choosing_probability(tmp_path):
    edges=[[[20,0,0],[20,12,0]]]
    candidates=[dict(index=i,probability=p,points_xyz=[[20,12,0]],actual_edges_xyz=edges,stop_reason='advanced') for i,p in [(3,.9),(5,.7)]]
    record=prediction(10,1,[20,0,0],[[2,1,20,12,0]],edges,candidates=candidates)
    result=detect_trace_errors(fixture(tmp_path,[record]))
    source=result['false_junction_events'][0]['trace_origin']['start']
    assert source['status']=='multiple_candidates' and source['probability'] is None
    assert source['prediction_step']==10 and source['prediction_index']==0
    assert [(c['candidate_index'],c['probability']) for c in source['candidates']]==[(3,.9),(5,.7)]


def test_initial_tree_errors_are_not_assigned_to_new_predictions(tmp_path):
    result=detect_trace_errors(fixture(tmp_path,[],([[0,0,0],[20,0,0],[20,12,0]],[-1,0,1])))
    source=result['false_junction_events'][0]['trace_origin']['start']
    assert source['status']=='initial_tree' and 'prediction_step' not in source
    assert result['trace']['prediction_count']==0


def test_break_has_related_stop_records_but_no_causal_probability(tmp_path):
    record=prediction(1,1,[20,0,0],[],[],candidates=[])
    result=detect_trace_errors(fixture(tmp_path,[record]))
    source=result['break_events'][0]['trace_origin']
    assert source['status']=='related_predictions_only' and source['probability'] is None
    assert source['related_prediction_indices']==[0]
    assert result['trace_predictions']['0']['candidate_scores']==[.2,.4]
    assert result['trace_predictions']['0']['candidates']==[]


def test_distant_records_are_not_assigned_to_a_break(tmp_path):
    record=prediction(1,0,[0,0,0],[],[],candidates=[])
    result=detect_trace_errors(fixture(tmp_path,[record]))
    assert result['break_events'][0]['trace_origin']['status']=='unresolved'


def test_gt_transform_applies_once_and_does_not_mutate_swc(tmp_path):
    path=fixture(tmp_path,[],manifest=dict(swc_to_voxel_scale_xyz=[2,2,2]))
    gt=Swc();gt.nodes={0:dict(id=0,parent=-1,x=0.,y=0.,z=0.,radius=1,type=1),
        1:dict(id=1,parent=0,x=10.,y=0.,z=0.,radius=1,type=3)};gt.edges=[(0,-1),(1,0)]
    before=copy.deepcopy(gt.__dict__)
    result=detect_trace_errors(path,gt,include_geometry=True)
    assert result['counts']['total']==0
    assert result['trace_geometry']['gt_nodes'][1]['xyz']==[20.,0.,0.]
    assert gt.__dict__==before


def test_partial_record_is_marked_incomplete_and_not_invented(tmp_path):
    path=fixture(tmp_path)
    path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n{"event":')
    result=detect_trace_errors(path)
    assert result['trace']['prediction_count']==2 and not result['trace']['complete']
    assert result['trace']['partial_last_line']


def test_missing_candidate_edge_is_marked_unrecorded_not_nearest_guessed(tmp_path):
    record=prediction(1,1,[20,0,0],[[2,1,20,12,0]],[[[20,0,0],[20,12,0]]])
    record['candidates'][0]['actual_edges_xyz']=[[[21,0,0],[21,12,0]]]
    result=detect_trace_errors(fixture(tmp_path,[record]))
    source=result['false_junction_events'][0]['trace_origin']['start']
    assert source['status']=='candidate_unrecorded' and source['probability'] is None
    assert source['prediction_step']==1


@pytest.mark.parametrize('bad', ['root','parent','version','cycle','initial_missing'])
def test_invalid_recordings_fail_explicitly(tmp_path,bad):
    path=fixture(tmp_path)
    records=[json.loads(line) for line in path.read_text().splitlines()]
    if bad=='root':records[1]['root']=999
    if bad=='parent':records[1]['added_nodes'][0][1]=999
    if bad=='version':records[0]['version']=1
    if bad=='cycle':np.savez(tmp_path/'trace_initial.npz',points_xyz=[[0,0,0],[20,0,0]],parents=[1,0])
    if bad=='initial_missing':(tmp_path/'trace_initial.npz').unlink()
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises((ValueError,FileNotFoundError)):
        detect_trace_errors(path)


def test_file_changed_during_detection_is_rejected(tmp_path,monkeypatch):
    path=fixture(tmp_path);original=module.detect_errors
    def changed(*args,**kwargs):
        result=original(*args,**kwargs)
        path.write_text(path.read_text()+'\n')
        return result
    monkeypatch.setattr(module,'detect_errors',changed)
    with pytest.raises(ValueError,match='changed'):
        detect_trace_errors(path)
