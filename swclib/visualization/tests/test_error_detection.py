"""Detector service returns original geometry and remains read-only over HTTP."""
import json
import urllib.request

from swclib.visualization.error_detection import detect_pair
from test_server import service


def pair(tmp_path):
    pred, gt = tmp_path/'pred.swc', tmp_path/'gt.swc'
    pred.write_text('1 1 0 0 0 1 -1\n2 3 20 0 0 1 1\n3 3 20 20 0 1 2\n')
    gt.write_text('1 1 0 0 0 1 -1\n2 3 40 0 0 1 1\n')
    return dict(pred_path=str(pred), gt_path=str(gt))


def test_adapter_keeps_sparse_original_ids_and_xyz(tmp_path):
    args=pair(tmp_path); result=detect_pair(**args)
    assert result['pred']['nodes'][2] == dict(id=3,parent=2,xyz=[20,20,0])
    assert result['result']['counts']['wrong_connection'] == 0
    assert result['result']['candidate_counts']['unsupported_path_candidate'] == 1
    assert result['result']['candidate_events'][0]['affected_subtree']['node_ids'] == [3]


def test_detect_http_input_validation_lock_and_readonly(service,tmp_path):
    server,_,request,url=service
    args=pair(tmp_path)
    before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    status,raw=request('/api/errors/detect',args)
    assert status==200,raw
    result=json.loads(raw)['result']
    assert result['counts']['false_junction']==0 and result['counts']['break_error']==0
    assert result['candidate_counts']['unsupported_path_candidate']==1
    assert request('/api/errors/detect',dict(args,sample_step=4))[0]==400
    assert request('/api/errors/detect',dict(args,pred_path=str(tmp_path)))[0]==400
    assert request('/api/errors/detect',dict(args,out_path=args['pred_path']))[0]==400
    assert request('/api/errors/detect',args,headers={'X-Viewer-Token':'invalid'})[0]==403
    server.error_detection_lock.acquire()
    try: assert request('/api/errors/detect',args)[0]==409
    finally: server.error_detection_lock.release()
    assert before=={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    for name in ['error-detection.js','error-renderer.js','plugin-navigation.js']:
        with urllib.request.urlopen(url+'/'+name) as response:
            assert response.status==200 and response.headers['Content-Type'].startswith('text/javascript')
