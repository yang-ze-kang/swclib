"""Real process overlap, source integrity, binary parity and memory admission."""
import errno
import os
import threading
import time

import pytest

from swclib.visualization import loading
from swclib.visualization.data import MIB, swc_packet


def guarded_probe(path):
    start=time.monotonic()
    try:
        with open(path,'w') as stream:stream.write('corrupted')
    except OSError as error:
        denied=error.errno
    else:
        denied=None
    time.sleep(.3)
    return os.getpid(),start,time.monotonic(),denied


@pytest.fixture(scope='module')
def readers():
    result=loading.SWCReaders(2)
    yield result
    result.close()


def test_processes_overlap_and_both_enforce_readonly(readers,tmp_path):
    source=tmp_path/'source.swc';source.write_text('1 1 1 2 3 1 -1\n')
    before=source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode
    futures=[readers.pool.submit(guarded_probe,str(source)) for _ in range(2)]
    a,b=[future.result(timeout=10) for future in futures]
    assert a[0]!=b[0] and max(a[1],b[1])<min(a[2],b[2])
    assert a[3]==b[3]==errno.EACCES
    assert before==(source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode)


def test_parallel_packets_match_existing_reader_exactly(readers,tmp_path):
    source=tmp_path/'source.swc'
    source.write_text('1 1 10 20 30 1 -1\n2 3 11 20 30 1 1\n3 3 12 20 30 1 2\n4 3 13 21 30 1 3\n5 3 13 19 30 1 3\n')
    crops=[None,{'start':[11,19,29],'end':[13,22,31]}, {'start':[0,0,0],'end':[1,1,1]}]
    futures=[readers.pool.submit(loading._read_packet,str(source),crop) for crop in crops]
    for future,crop in zip(futures,crops):
        assert future.result(timeout=10)==b''.join(swc_packet(source,crop))


def test_memory_admission_and_worker_configuration(tmp_path,monkeypatch):
    reader=loading.SWCReaders.__new__(loading.SWCReaders)
    reader.workers=2;reader.active=reader.reserved=0;reader.lock=threading.Lock()
    source=tmp_path/'source.swc';source.write_text('1 1 1 2 3 1 -1\n')
    monkeypatch.setattr(loading,'available_memory',lambda:100*MIB)
    first=reader.acquire(source);second=reader.acquire(source)
    assert first and second and reader.acquire(source) is None
    reader.release(first);reader.release(second)
    monkeypatch.setattr(loading,'available_memory',lambda:20*MIB)
    with pytest.raises(ValueError,match='内存'):reader.acquire(source)
    assert reader.active==reader.reserved==0
    assert loading.worker_count()==1
    assert loading.worker_count(8)==8
    for value in [0,9,-1,1.5]:
        with pytest.raises(ValueError):loading.worker_count(value)


def test_display_prepares_full_summary_without_changing_packet(readers, tmp_path, monkeypatch):
    from swclib.visualization.hemisphere import hemisphere_counts
    source = tmp_path/'full.swc'
    source.write_text('1 1 -2 0 0 1 -1\n2 3 2 0 0 1 1\n3 3 3 0 0 1 2\n')
    crop = dict(start=[0,-1,-1],end=[4,1,1])
    assert readers.read(str(source), crop) == [b''.join(swc_packet(source,crop))]
    entries = [dict(path=str(source))]
    with monkeypatch.context() as patch:
        def no_read(*args, **kwargs):
            pytest.fail('Display summary was not reused')
        patch.setattr(readers,'_neuron_batch',no_read)
        summaries = readers.plane_summaries(entries)
        assert summaries[0]['node_count'] == 3
        plane = dict(azimuth_deg=0,elevation_deg=0,offset=0)
        result = readers.plane_counts(entries,[plane],summaries)[0]
    assert result['cross'] == 1
    assert result == hemisphere_counts(readers.hemispheres(entries,plane,include_soma=False))
    # Existing display behavior still allows missing parents.
    source.write_text('1 3 1 2 3 1 99\n')
    assert readers.read(str(source),None) == [b''.join(swc_packet(source))]
    with pytest.raises(ValueError,match='父节点'):
        readers.plane_summaries(entries)


def test_uncertain_planes_use_exact_fallback_and_refine_summary(readers,tmp_path):
    from swclib.visualization.hemisphere import hemisphere_counts
    source = tmp_path/'touch.swc'
    source.write_text(''.join(f'{i+1} 3 {i} 0 0 1 {i if i else -1}\n' for i in range(700)))
    entries = [dict(path=str(source))]
    summaries = readers.plane_summaries(entries)
    assert summaries[0]['detail'] == 64
    plane = dict(azimuth_deg=0,elevation_deg=0,offset=699)
    result = readers.plane_counts(entries,[plane],summaries)[0]
    assert summaries[0]['detail'] == 256
    assert result['left_only'] == 1 and result['cross'] == 0
    assert result == hemisphere_counts(readers.hemispheres(entries,plane,include_soma=False))
    assert readers.active == readers.reserved == 0


def test_summary_cache_evicts_within_memory_budget(readers,tmp_path,monkeypatch):
    import numpy as np
    readers._summary_cache.clear();readers._summary_bytes=0
    monkeypatch.setattr(loading,'available_memory',lambda:100000)
    readers._remember_summary('a',(1,),dict(points=np.zeros((10,3))))
    readers._remember_summary('b',(2,),dict(points=np.zeros((10,3))))
    assert readers._summary_bytes <= 2000
    assert readers._cached_summary('a',(1,)) is None
    assert readers._cached_summary('b',(2,)) is not None
    assert readers._cached_summary('b',(3,)) is None
    assert readers._summary_bytes == 0


def test_convex_summary_runs_inside_protected_reader(readers,tmp_path):
    import numpy as np
    from swclib.visualization.hemisphere import hemisphere_counts
    source=tmp_path/'cloud.swc'
    xyz=np.random.default_rng(13).normal(size=(2000,3)).cumsum(axis=0)
    source.write_text(''.join(f'{i+1} 3 {x} {y} {z} 1 {i if i else -1}\n' for i,(x,y,z) in enumerate(xyz)))
    snapshot=source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode
    entries=[dict(path=str(source))]
    summaries=readers.plane_summaries(entries)
    assert 'support_error' in summaries[0] and len(summaries[0]['witnesses']) <= 512
    plane=dict(azimuth_deg=27,elevation_deg=13,offset=-4)
    assert readers.plane_counts(entries,[plane],summaries)[0] == hemisphere_counts(readers.hemispheres(entries,plane,include_soma=False))
    assert snapshot==(source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode)
    assert list(tmp_path.iterdir())==[source]
