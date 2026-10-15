"""HTTP contract and kernel-guard checks. Requires permission to bind loopback."""
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request

import pytest

from swclib.visualization import server as module


@pytest.fixture
def service():
    server=module.create_server(port=0,swc_workers=2)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}'
    config=json.load(urllib.request.urlopen(url+'/api/config'))
    def request(route,body,headers=None):
        request_headers={'Content-Type':'application/json','X-Viewer-Token':config['token']}
        if headers:request_headers.update(headers)
        req=urllib.request.Request(url+route,data=json.dumps(body).encode(),headers=request_headers)
        try:
            with urllib.request.urlopen(req) as response:return response.status,response.read()
        except urllib.error.HTTPError as error:return error.code,error.read()
    yield server,config,request,url
    server.shutdown();server.server_close();thread.join(timeout=5)


def test_remove_contract_and_source_integrity(service,tmp_path):
    server,config,request,url=service
    assert config['read_only'] and config['filesystem_protection']=='seccomp'
    assert config['limits']['max_edge']==300
    source=tmp_path/'neuron.swc';source.write_text('1 1 1 2 3 1 -1\n')
    before=(source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode)
    status,body=request('/api/add',{'kind':'swc','path':str(source)});assert status==200
    key=json.loads(body)['entries'][0]['id']
    assert request('/api/swc',{'id':key})[0]==200
    assert request('/api/swc',{'id':key,'out_path':str(source)})[0]==400
    assert request('/api/remove',{'ids':[key],'path':str(source)})[0]==400
    assert request('/api/delete',{'path':str(source)})[0]==404
    assert request('/api/remove',{'ids':[key]},headers={'Origin':'http://other.invalid'})[0]==403
    assert request('/api/remove',{'ids':[key]},headers={'X-Viewer-Token':'invalid'})[0]==403
    assert json.loads(request('/api/remove',{'ids':[key]})[1])['removed']==[key]
    assert request('/api/swc',{'id':key})[0]==400
    assert before==(source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode)


def test_handler_cannot_overwrite_source_even_if_reader_regresses(service,tmp_path,monkeypatch):
    server,_,request,_=service
    source=tmp_path/'neuron.swc';source.write_text('1 1 1 2 3 1 -1\n')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    key=json.loads(request('/api/add',{'kind':'swc','path':str(source)})[1])['entries'][0]['id']
    def erroneous_reader(path,crop):
        with open(path,'w') as stream:stream.write('corrupted')
        raise AssertionError('write unexpectedly succeeded')
    monkeypatch.setattr(server.swc_readers,'read',erroneous_reader)
    assert request('/api/swc',{'id':key})[0]==400
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest


def test_failed_guard_refuses_request(service,monkeypatch):
    server,_,request,_=service
    def fail():raise RuntimeError('test filter installation failure')
    monkeypatch.setattr(server.readonly,'install',fail)
    assert request('/api/add',{'kind':'swc','path':'/does/not/matter'})[0]==503
    assert server.catalog.list()==[]


def test_region_http_contract_and_fixed_coordinates(service,tmp_path):
    import numpy as np
    import tifffile
    _,_,request,_=service
    tifffile.imwrite(tmp_path/'z_0.tif',np.arange(100,dtype='u1').reshape(10,10))
    status,body=request('/api/add',{'path':str(tmp_path),'kind':'brain','options':{'mode':'slice'}})
    assert status==200
    source=json.loads(body)['entries'][0]
    args={'id':source['id'],'start':[1,1,0],'end':[4,4,1]}
    status,body=request('/api/region',args);assert status==200
    block=json.loads(body)['entry'];assert block['kind']=='brain-region'
    assert json.loads(request('/api/region',args)[1])['entry']['id']==block['id']
    assert request('/api/region',dict(args,out_path=str(tmp_path/'should-not-exist.tif')))[0]==400
    assert request('/api/volume',{'id':block['id'],'kind':'brain-region','start':[0,0,0]})[0]==400
    assert request('/api/remove',{'ids':[source['id']]})[0]==200
    assert request('/api/volume',{'id':block['id'],'kind':'brain-region'})[0]==200
    assert len(list(tmp_path.iterdir()))==1


def test_soma_candidates_http_is_readonly_bounded_and_survives_bad_file(service,tmp_path,monkeypatch):
    server,_,request,_=service
    source=tmp_path/'neuron.swc';source.write_text('1 3 1 2 3 1 -1\n2 1 4 5 6 1 1\n')
    broken=tmp_path/'broken.swc';broken.write_text('1 1 invalid 2 3 1 -1\n')
    before=source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode
    entries=json.loads(request('/api/add',{'path':str(tmp_path),'kind':'swc'})[1])['entries']
    ids={e['name']:e['id'] for e in entries}
    args={'id':ids['neuron.swc'],'rule':'type-first'}
    status,body=request('/api/somas',args)
    assert status==200 and json.loads(body)['candidates']==[{'node_id':2,'center':[4,5,6],'method':'type=1'}]
    assert request('/api/somas',dict(args,out_path=str(source)))[0]==400
    assert request('/api/somas',dict(args,rule='unknown'))[0]==400
    assert request('/api/somas',dict(args,id=ids['broken.swc']))[0]==400
    assert request('/api/somas',args)[0]==200
    assert before==(source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode)
    with monkeypatch.context() as patch:
        patch.setattr(server.swc_readers,'acquire',lambda path:None)
        assert request('/api/somas',args)[0]==409
    def bad_reader(path,rule):
        with open(path,'w') as stream:stream.write('corrupted')
    monkeypatch.setattr(server.swc_readers,'somas',bad_reader)
    assert request('/api/somas',args)[0]==400
    assert before==(source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode)


def test_hemisphere_export_uses_full_sources_and_protects_existing_files(service,tmp_path):
    server,_,request,_=service
    source=tmp_path/'neuron.swc';source.write_text('1 1 -2 0 0 1 -1\n2 3 3 0 0 1 1\n')
    before=source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode
    key=json.loads(request('/api/add',{'path':str(source),'kind':'swc'})[1])['entries'][0]['id']
    args={'ids':[key],'plane':{'azimuth_deg':0,'elevation_deg':0,'offset':0},'directory':str(tmp_path)}
    assert request('/api/hemisphere/export',dict(args,filename='neuron.swc'))[0]==400
    assert request('/api/hemisphere/export',dict(args,crop={'start':[0,0,0],'end':[1,1,1]}))[0]==400
    status,body=request('/api/hemisphere/export',args);assert status==200,body
    output=tmp_path/'brain_hemisphere.json';result=json.loads(output.read_text())
    assert set(result)=={'description','plane','neurons'}
    assert result['neurons'][0]['right_length']==3 and result['neurons'][0]['left_length']==2
    assert result['neurons'][0]['soma']['x']==-2
    snapshot=output.read_bytes(),output.stat().st_mtime_ns
    status,body=request('/api/hemisphere/export',args)
    assert status==409 and json.loads(body)['code']=='export_exists'
    assert json.loads(body)['path']==str(output)
    assert snapshot==(output.read_bytes(),output.stat().st_mtime_ns)
    changed=dict(args,plane=dict(args['plane'],offset=1),overwrite=True)
    assert request('/api/hemisphere/export',changed)[0]==200
    assert json.loads(output.read_text())['plane']['offset']==1
    custom=dict(args);custom['path']=str(tmp_path/'custom.json');del custom['directory']
    assert request('/api/hemisphere/export',custom)[0]==200
    assert (tmp_path/'custom.json').is_file()
    assert request('/api/hemisphere/export',dict(custom,overwrite='false'))[0]==400
    assert request('/api/hemisphere/export',dict(custom,path=str(source),overwrite=True))[0]==400
    assert request('/api/hemisphere/export',dict(custom,directory=str(tmp_path)))[0]==400
    assert before==(source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode)
    assert server.swc_readers.active==0 and server.swc_readers.reserved==0


def test_bad_neuron_aborts_entire_export_without_output(service,tmp_path):
    server,_,request,_=service
    (tmp_path/'a.swc').write_text('1 1 0 0 0 1 -1\n')
    (tmp_path/'b.swc').write_text('1 1 0 0 0 1 -1\n2 1 1 0 0 1 1\n')
    ids=[e['id'] for e in json.loads(request('/api/add',{'path':str(tmp_path),'kind':'swc'})[1])['entries']]
    status,body=request('/api/hemisphere/export',{'ids':ids,'plane':{'azimuth_deg':0,'elevation_deg':0,'offset':0},'directory':str(tmp_path)})
    assert status==400 and '2 个胞体候选' in json.loads(body)['error']
    assert not (tmp_path/'brain_hemisphere.json').exists()
    assert server.swc_readers.active==0 and server.swc_readers.reserved==0


def test_hemisphere_counts_are_readonly_and_use_requested_complete_neurons(service,tmp_path):
    server,_,request,_=service
    for name,source in {
        'left':'1 1 -3 0 0 1 -1\n2 3 -1 0 0 1 1\n',
        'right':'1 1 1 0 0 1 -1\n2 3 3 0 0 1 1\n',
        'cross':'1 1 -1 0 0 1 -1\n2 1 1 0 0 1 1\n',
    }.items():(tmp_path/f'{name}.swc').write_text(source)
    entries=json.loads(request('/api/add',{'path':str(tmp_path),'kind':'swc'})[1])['entries']
    ids={e['name']:e['id'] for e in entries}
    args=dict(ids=[ids['left.swc'],ids['cross.swc'],ids['cross.swc']],plane=dict(azimuth_deg=0,elevation_deg=0,offset=0))
    snapshot={p:p.read_bytes() for p in tmp_path.iterdir()}
    status,body=request('/api/hemisphere/count',args);assert status==200,body
    assert json.loads(body)['counts']==dict(total=2,left_only=1,right_only=0,cross=1,left=2,right=1)
    assert {p:p.read_bytes() for p in tmp_path.iterdir()}==snapshot
    assert request('/api/hemisphere/count',dict(args,crop={}))[0]==400
    assert request('/api/hemisphere/count',dict(args,ids=[]))[0]==400
    assert request('/api/hemisphere/count',dict(args,plane={}))[0]==400
    assert request('/api/hemisphere/count',dict(args,path=str(tmp_path)))[0]==400
    assert server.swc_readers.active==0 and server.swc_readers.reserved==0


def test_parallel_http_reads_are_bounded_and_release_on_error(service,tmp_path,monkeypatch):
    server,config,request,_=service
    assert config['swc_workers']==2
    for i in range(3):(tmp_path/f'{i}.swc').write_text('1 1 1 2 3 1 -1\n')
    ids=[e['id'] for e in json.loads(request('/api/add',{'kind':'swc','path':str(tmp_path)})[1])['entries']]
    original=server.swc_readers.read
    gate=threading.Event();entered=threading.Barrier(3)
    def hold(path,crop):
        entered.wait(timeout=10)
        assert gate.wait(timeout=10)
        return original(path,crop)
    monkeypatch.setattr(server.swc_readers,'read',hold)
    with ThreadPoolExecutor(2) as clients:
        futures=[clients.submit(request,'/api/swc',{'id':key}) for key in ids[:2]]
        try:
            entered.wait(timeout=10)
            assert server.swc_readers.active==2
            assert request('/api/swc',{'id':ids[2]})[0]==409
            assert server.read_lock.acquire(blocking=False),'SWC blocked the image queue'
            server.read_lock.release()
        finally:gate.set()
        assert all(f.result()[0]==200 for f in futures)
    monkeypatch.setattr(server.swc_readers,'read',original)
    assert request('/api/swc',{'id':ids[2],'crop':{'start':[3,3,3],'end':[0,0,0]}})[0]==400
    released=threading.Event();release=server.swc_readers.release
    def track_release(cost):
        release(cost)
        released.set()
    monkeypatch.setattr(server.swc_readers,'release',track_release)
    assert request('/api/swc',{'id':ids[2]})[0]==200
    assert released.wait(timeout=5)
    assert server.swc_readers.active==0 and server.swc_readers.reserved==0


def test_optimize_http_constraints_cache_and_cleanup(service, tmp_path):
    import math
    server, _, request, _ = service
    for i, side in enumerate((-1, 1)):
        (tmp_path/f'{i}.swc').write_text(f'1 1 -.2 {10*side} -.5 1 -1\n2 1 .2 {10*side} .5 1 1\n')
    ids = [e['id'] for e in json.loads(request('/api/add', dict(path=str(tmp_path),kind='swc'))[1])['entries']]
    args = dict(ids=ids, plane=dict(azimuth_deg=0,elevation_deg=0,offset=0), center=[0,0,0])
    before = {p:p.read_bytes() for p in tmp_path.iterdir()}
    status, body = request('/api/hemisphere/optimize', args)
    assert status == 200, body
    result = json.loads(body)
    assert result['max_angle_deg'] == 10 and result['counts']['cross'] == 0
    assert len(result['suggestions']) == 3 and 0 < result['angle_change_deg'] <= 10
    assert math.fsum(n*c for n,c in zip(result['plane']['normal_xyz'],args['center'])) == result['plane']['offset']
    assert server.swc_readers._summary_bytes > 0
    cached = list(server.swc_readers._summary_cache.values())
    assert request('/api/hemisphere/optimize', args)[0] == 200
    assert all(a[1] is b[1] for a,b in zip(cached,server.swc_readers._summary_cache.values()))
    for invalid in [dict(center=[0,0]), dict(max_angle_deg=91), dict(ids=[]), dict(crop={})]:
        assert request('/api/hemisphere/optimize', dict(args, **invalid))[0] == 400
    centered = json.loads(request('/api/hemisphere/optimize', dict(args, center=[5,6,7],max_angle_deg=0))[1])
    assert centered['plane']['offset'] == 5 and centered['angle_change_deg'] == 0
    assert {p:p.read_bytes() for p in tmp_path.iterdir()} == before
    with server.export_lock:
        assert request('/api/hemisphere/optimize', args)[0] == 409
    # Changed sources invalidate summaries; failed searches release all slots.
    (tmp_path/'0.swc').write_text('1 3 0 0 0 1 99\n')
    assert request('/api/hemisphere/optimize', args)[0] == 400
    assert server.swc_readers.active == 0 and server.swc_readers.reserved == 0
    assert not server.export_lock.locked()
    (tmp_path/'0.swc').write_text('1 3 -1 0 0 1 -1\n')
    assert request('/api/hemisphere/optimize', args)[0] == 200
