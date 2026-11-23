"""Trace mode HTTP, saved probabilities and input exclusivity."""
import json

from swclib.error_detection.tests.test_trace_detection import fixture, prediction
from test_server import service


def test_trace_detection_endpoint(service,tmp_path):
    _,_,request,_=service
    path=fixture(tmp_path)
    before={p:p.read_bytes() for p in tmp_path.iterdir()}
    status,raw=request('/api/errors/detect',dict(trace_path=str(path)))
    assert status==200,raw
    pair=json.loads(raw);event=pair['result']['candidate_events'][0]
    assert event['trace_origin']['start']['probability']==.82
    assert event['trace_origin']['confirmation']['prediction_step']==2
    assert pair['pred']['nodes'][2]['id']==2
    assert 'trace_geometry' not in pair['result']
    assert request('/api/errors/detect',dict(trace_path=str(path),pred_path=str(tmp_path/'gt.swc')))[0]==400
    assert request('/api/errors/detect',dict(trace_path=''))[0]==400
    assert request('/api/errors/detect',dict(trace_path=str(path),out_path=str(path)))[0]==400
    assert before=={p:p.read_bytes() for p in tmp_path.iterdir()}


def test_trace_detection_endpoint_keeps_ambiguous_gt_candidate(service,tmp_path):
    _,_,request,_=service
    path=fixture(tmp_path,[prediction(1,1,[20,0,0],[[2,1,60,0,0]],
                                     [[[20,0,0],[60,0,0]]])])
    (tmp_path/'gt.swc').write_text('1 1 0 0 0 1 -1\n2 3 20 0 0 1 1\n'
                                 '3 3 60 1 0 1 2\n4 3 60 -1 0 1 2\n')
    before={p:p.read_bytes() for p in tmp_path.iterdir()}
    status,raw=request('/api/errors/detect',dict(trace_path=str(path)))
    assert status==200,raw
    result=json.loads(raw)['result']
    assert result['schema_version']==2 and result['counts']['total']==0
    assert result['fiber_metric']['FN']==1
    assert result['candidate_counts']['ambiguous_missing_branch']==1
    event=result['candidate_events'][0]
    assert event['error_type']=='ambiguous_missing_branch'
    assert event['trace_origin']['related_prediction_indices']==[0]
    assert event['trace_origin']['probability'] is None
    assert before=={p:p.read_bytes() for p in tmp_path.iterdir()}
