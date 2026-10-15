"""Behavior tests: topology, XYZ crops, lazy I/O and decode-before-budget guards."""
import json
import struct

import numpy as np
import pytest
import tifffile

from swclib.visualization import data


def unpack(parts):
    raw = b''.join(parts)
    length = struct.unpack_from('<I', raw)[0]
    return json.loads(raw[4:4 + length]), raw[4 + length:]


@pytest.fixture
def neuron(tmp_path):
    path = tmp_path / 'neuron.swc'
    path.write_text('1 1 10 20 30 1 -1\n2 3 11 20 30 1 1\n3 3 12 20 30 1 2\n4 3 13 21 30 1 3\n5 3 13 19 30 1 3\n')
    return path


def test_topology_colors_and_edges(neuron):
    meta, raw = unpack(data.swc_packet(neuron))
    assert meta['count'] == 5 and meta['edge_count'] == 4
    colors = np.frombuffer(raw, 'u1', count=20, offset=60).reshape(-1, 4)
    assert colors.tolist() == [[79,151,255,255],[249,87,103,255],[63,216,147,255],[255,205,71,255],[255,205,71,255]]
    edges = np.frombuffer(raw, '<u4', offset=80).reshape(-1, 2)
    assert edges.tolist() == [[1,0],[2,1],[3,2],[4,2]]


def test_crop_exclusive_end_rebases_and_promotes_cut_root(neuron):
    meta, raw = unpack(data.swc_packet(neuron, {'start':[11,19,29], 'end':[13,22,31]}))
    assert meta['count'] == 2 and meta['edge_count'] == 1
    assert meta['bounds'] == [[0,1,1],[1,1,1]]
    assert meta['roots'] == 1
    assert np.frombuffer(raw,'<f4',count=6).tolist() == [0,1,1,1,1,1]
    assert list(raw[24:28]) == [79,151,255,255]


def test_empty_swc_and_empty_crop(neuron, tmp_path):
    for path, crop in [(neuron,{'start':[0,0,0],'end':[1,1,1]}),(tmp_path/'empty.swc',None)]:
        if crop is None: path.write_text('# empty\n')
        meta, raw = unpack(data.swc_packet(path, crop))
        assert meta['count'] == 0 and meta['bounds'] is None and not raw


def test_brain_regions_are_lazy_independent_and_deduplicated(tmp_path,monkeypatch):
    from swclib.whole_brain import WholeBrainReader
    volume=np.arange(4*6*8,dtype='uint16').reshape(4,6,8)
    brain=tmp_path/'brain01';brain.mkdir()
    for z,plane in enumerate(volume):tifffile.imwrite(brain/f'z_{z}.tif',plane)
    before={p:p.read_bytes() for p in brain.iterdir()}
    catalog=data.Catalog();source=catalog.add(brain,'brain',options={'mode':'slice'})[0]
    with monkeypatch.context() as patch:
        patch.setattr(WholeBrainReader,'read_region',lambda *a,**k:pytest.fail('registration decoded pixels'))
        a=catalog.add_region(source['id'],[1,2,0],[5,5,3],data.Limits())
        b=catalog.add_region(source['id'],[2,0,1],[4,3,4],data.Limits())
        assert catalog.add_region(source['id'],[1,2,0],[5,5,3],data.Limits())['id']==a['id']
        assert data.describe_image(a,data.Limits())['shape']==[4,3,3]
    assert a['id']!=b['id'] and a['name']=='brain01_[1,2,0]-[5,5,3]'
    assert len(catalog.list())==3
    catalog.remove([source['id']])
    for entry,start,end in [(a,[1,2,0],[5,5,3]),(b,[2,0,1],[4,3,4])]:
        meta,raw=unpack(data.volume_packet(entry,{},data.Limits()))
        assert meta['origin']==start and meta['shape']==[e-s for s,e in zip(start,end)]
        expected=volume[start[2]:end[2],start[1]:end[1],start[0]:end[0]].astype('f4')
        expected=(expected-expected.min())/(expected.max()-expected.min())
        np.testing.assert_allclose(np.frombuffer(raw,'<f4').reshape(expected.shape),expected)
    for key in ['start','end','origin']:
        with pytest.raises(ValueError,match='固定'):data.volume_packet(a,{key:[0,0,0]},data.Limits())
    catalog.remove([a['id']]);assert catalog.get(b['id'],'brain-region')==b
    assert before=={p:p.read_bytes() for p in brain.iterdir()}


def test_brain_region_registration_validates_before_recording(tmp_path,monkeypatch):
    from swclib.whole_brain import WholeBrainReader
    tifffile.imwrite(tmp_path/'z_0.tif',np.zeros((20,20),dtype='uint8'))
    catalog=data.Catalog();source=catalog.add(tmp_path,'brain',options={'mode':'slice'})[0]
    monkeypatch.setattr(WholeBrainReader,'read_region',lambda *a,**k:pytest.fail('invalid ROI decoded pixels'))
    for start,end in [([0,0,0],[301,1,1]),([0,0,0],[10,10,2]),([-1,0,0],[10,10,1]),([0,0,0],[0,1,1]),([.5,0,0],[10,10,1])]:
        with pytest.raises(ValueError):catalog.add_region(source['id'],start,end,data.Limits(),{'volume_mib':4096})
    assert catalog.list()==[source]


def test_catalog_does_not_decode_and_deduplicates(tmp_path, monkeypatch):
    for name in ['a.tif','b.tiff','c.npy','d.nrrd','ignored.txt']:
        (tmp_path/name).write_bytes(b'not an image')
    monkeypatch.setattr(tifffile,'imread',lambda *a,**k:pytest.fail('eager pixel read'))
    catalog=data.Catalog()
    assert len(catalog.add(tmp_path,'image')) == 4
    assert len(catalog.add(tmp_path,'image')) == 4
    assert len(catalog.list()) == 4
    assert all(set(e)=={'id','kind','path','name','options'} for e in catalog.list())


def test_small_image_preserves_low_intensity_and_zyx(tmp_path):
    volume=np.arange(60,dtype=np.uint16).reshape(3,4,5)
    volume[-1,-1,-1]=65535
    path=tmp_path/'volume.tif';tifffile.imwrite(path,volume,photometric='minisblack')
    entry=data.Catalog().add(path,'image')[0]
    meta,raw=unpack(data.volume_packet(entry,{'origin':[100,200,300]},data.Limits()))
    result=np.frombuffer(raw,'<f4').reshape(volume.shape)
    assert meta['shape']==[5,4,3] and meta['origin']==[100,200,300]
    assert meta['display_dtype']=='float32'
    np.testing.assert_allclose(result,volume.astype('f4')/65535)
    assert result[0,0,1]>0 and result[0,0,1]<1/255


def test_oversized_image_rejected_before_pixels(tmp_path,monkeypatch):
    path=tmp_path/'volume.tif';tifffile.imwrite(path,np.zeros((32,128,128),dtype='u1'),photometric='minisblack')
    entry=data.Catalog().add(path,'image')[0]
    monkeypatch.setattr(tifffile,'imread',lambda *a,**k:pytest.fail('decoded before guard'))
    with pytest.raises(ValueError,match='超过当前显示上限'):
        data.volume_packet(entry,{},data.Limits(1))


def test_low_memory_decode_scratch_and_client_limits(monkeypatch):
    monkeypatch.setattr(data,'available_memory',lambda:100*data.MIB)
    limits=data.Limits(64)
    assert limits.describe(scratch=30*data.MIB)['max_voxels']==0
    assert limits.describe(client={'volume_mib':1,'max_texture_size':64})['max_voxels']==data.MIB//4
    with pytest.raises(ValueError):limits.check((1,1,65),'uint8',client={'max_texture_size':64})


def test_whole_brain_crop_xyz_and_preflight(tmp_path,monkeypatch):
    volume=np.arange(5*7*9,dtype='u2').reshape(5,7,9)
    for z,plane in enumerate(volume):tifffile.imwrite(tmp_path/f'slice_{z}.tif',plane)
    entry=data.Catalog().add(tmp_path,'brain',options={'mode':'slice','pattern':r'slice_(\d+)\.tif'})[0]
    meta,raw=unpack(data.volume_packet(entry,{'start':[3,2,1],'end':[8,6,5]},data.Limits()))
    assert meta['shape']==[5,4,4] and meta['origin']==[3,2,1]
    expected=volume[1:5,2:6,3:8].astype('f4');expected=(expected-expected.min())/(expected.max()-expected.min())
    np.testing.assert_allclose(np.frombuffer(raw,'<f4').reshape(4,4,5),expected)
    from swclib.whole_brain import WholeBrainReader
    monkeypatch.setattr(WholeBrainReader,'read_region',lambda *a,**k:pytest.fail('decoded before guard'))
    with pytest.raises(ValueError,match='超过当前显示上限'):
        data.volume_packet(entry,{'start':[0,0,0],'end':[10000]*3},data.Limits())


@pytest.mark.parametrize('start,end',[([0,0],[1,1,1]),([0,0,float('nan')],[1,1,1]),([0,0,0],[0,1,1]),([.5,0,0],[1,1,1])])
def test_invalid_integer_regions(start,end):
    with pytest.raises(ValueError):data.region(start,end,integer=True)


def test_fnt_cross_cube_crop_only_decodes_intersections(tmp_path,monkeypatch):
    import nrrd
    volume=np.arange(4*5*6,dtype='u2').reshape(4,5,6)
    (tmp_path/'catalog').write_text('[CH00]\npattern=z<Z>y<Y>x<X>.nrrd\nsize=6 5 4\ncubesize=3 3 2\n')
    for z in range(0,4,2):
        for y in range(0,5,3):
            for x in range(0,6,3):
                nrrd.write(str(tmp_path/f'z{z}y{y}x{x}.nrrd'),volume[z:z+2,y:y+3,x:x+3],index_order='C')
    calls=[];read=nrrd.read
    def tracked(path,**kwargs):
        calls.append(Path(path).name)
        return read(path,**kwargs)
    from pathlib import Path
    monkeypatch.setattr(nrrd,'read',tracked)
    entry=data.Catalog().add(tmp_path,'brain',options={'mode':'fnt'})[0]
    data.describe_image(entry,data.Limits())
    assert not calls
    meta,raw=unpack(data.volume_packet(entry,{'start':[3,3,2],'end':[6,5,4]},data.Limits()))
    assert calls==['z2y3x3.nrrd']
    assert meta['shape']==[3,2,2] and meta['origin']==[3,3,2]
    expected=volume[2:4,3:5,3:6].astype('f4');expected=(expected-expected.min())/(expected.max()-expected.min())
    np.testing.assert_allclose(np.frombuffer(raw,'<f4').reshape(2,2,3),expected)


@pytest.mark.parametrize('suffix',['.npy','.nrrd'])
def test_scalar_image_formats(tmp_path,suffix):
    import nrrd
    volume=np.arange(24,dtype='f4').reshape(2,3,4)
    path=tmp_path/('volume'+suffix)
    if suffix=='.npy':np.save(path,volume)
    else:nrrd.write(str(path),volume,index_order='C')
    entry=data.Catalog().add(path,'image')[0]
    meta,raw=unpack(data.volume_packet(entry,{},data.Limits()))
    np.testing.assert_allclose(np.frombuffer(raw,'<f4').reshape(volume.shape),volume/23)


def test_constant_image_and_invalid_color_tiff(tmp_path):
    path=tmp_path/'constant.tif';tifffile.imwrite(path,np.ones((8,9),dtype='u1')*42)
    entry=data.Catalog().add(path,'image')[0]
    meta,raw=unpack(data.volume_packet(entry,{},data.Limits()))
    assert meta['shape']==[9,8,1] and np.frombuffer(raw,'<f4').min()==1
    tifffile.imwrite(path,np.zeros((8,9,3),dtype='u1'),photometric='rgb')
    with pytest.raises(ValueError,match='灰度 TIFF'):data.image_info(path)


def test_maximum_image_is_300_per_axis_even_with_large_budgets(monkeypatch):
    monkeypatch.setattr(data,'available_memory',lambda:16*1024**3)
    limits=data.Limits()
    assert limits.check((300,300,300),'uint16')['max_voxels']==27_000_000
    assert limits.describe()['suggested_cube']==300
    for shape in [(301,1,1),(1,301,1),(1,1,301),(301,299,299)]:
        with pytest.raises(ValueError,match='超过当前显示上限'):
            data.Limits(4096).check(shape,'uint16',client={'volume_mib':4096,'max_texture_size':4096})


def test_catalog_removal_never_changes_source_files(neuron,tmp_path):
    import hashlib
    path=tmp_path/'image.npy';np.save(path,np.zeros((3,4,5),dtype='u1'))
    paths=[neuron,path]
    before={p:(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns,p.stat().st_mode) for p in paths}
    catalog=data.Catalog();a=catalog.add(neuron,'swc')[0];b=catalog.add(path,'image')[0]
    assert catalog.remove([a['id'],a['id'],'missing'])==[a['id']]
    assert catalog.remove([b['id']])==[b['id']]
    assert catalog.list()==[] and catalog.sources=={}
    assert catalog.add(neuron,'swc')[0]['id']!=a['id']
    for p in paths:assert before[p]==(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns,p.stat().st_mode)
    with pytest.raises(ValueError):catalog.remove(str(neuron))


def test_kernel_readonly_guard_blocks_python_and_native_mutations(tmp_path):
    import subprocess
    import sys
    source=tmp_path/'source.txt';source.write_text('original bytes')
    script=r'''
import ctypes, os, sys
from pathlib import Path
from swclib.visualization.readonly import ReadOnlyGuard
source=Path(sys.argv[1]);libc=ctypes.CDLL(None,use_errno=True)
guard=ReadOnlyGuard();guard.install()
assert source.read_text()=='original bytes'
operations=[lambda:source.write_text('overwrite'),lambda:source.open('a'),
            lambda:os.open(source,os.O_RDWR),lambda:os.truncate(source,0),
            lambda:source.unlink(),lambda:source.rename(source.with_suffix('.renamed')),
            lambda:source.chmod(0o444),lambda:os.utime(source,None),
            lambda:os.link(source,source.with_suffix('.link')),
            lambda:os.symlink(source,source.with_suffix('.symlink')),
            lambda:source.with_suffix('.new').write_bytes(b'new')]
for operation in operations:
    try:operation()
    except PermissionError:pass
    else:raise AssertionError('filesystem mutation escaped guard')
assert libc.open(os.fsencode(source),os.O_WRONLY|os.O_TRUNC)==-1
assert ctypes.get_errno()==13
assert libc.unlink(os.fsencode(source))==-1
assert source.read_text()=='original bytes'
print('read-only checks passed')
'''
    result=subprocess.run([sys.executable,'-c',script,str(source)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'passed' in result.stdout and source.read_text()=='original bytes'
    assert len(list(tmp_path.iterdir()))==1
