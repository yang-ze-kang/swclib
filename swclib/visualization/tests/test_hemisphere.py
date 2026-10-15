import json
import math
from concurrent.futures import ThreadPoolExecutor

import pytest

from swclib.visualization.exporting import ExportExistsError, ExportWriter, write_report
from swclib.visualization.hemisphere import DESCRIPTION, hemisphere_counts, neuron_statistics, plane_parameters


def plane(a=0,e=0,d=0):
    return {'azimuth_deg':a,'elevation_deg':e,'offset':d}


def test_reference_plane_matches_supplied_schema():
    result=plane_parameters(plane(124,6,4177.036228665835))
    assert result['normal_xyz']==pytest.approx([-.5561295862362149,.8244960179889524,.10452846326765347])
    assert result['point_xyz']==pytest.approx([-2322.97342954161,3443.9497375305723,436.61917799575446])
    assert math.sqrt(sum(v*v for v in result['normal_xyz']))==pytest.approx(1)
    assert plane_parameters(plane(-90))['azimuth_deg']==270
    for value in [plane(e=91),plane(d=float('nan')),dict(plane(),normal_xyz=[1,0,0])]:
        with pytest.raises(ValueError):plane_parameters(value)


def test_exact_segment_split_and_original_coordinates(tmp_path):
    path=tmp_path/'cross.swc';path.write_text('1 1 8 10 20 1 -1\n2 3 14 10 20 1 1\n3 3 14 13 24 1 2\n')
    result=neuron_statistics(path,plane(d=10))
    assert result=={'swc_name':'cross.swc','soma':{'x':8,'y':10,'z':20,'side':'left','signed_distance':-2},
                    'left_length':2,'right_length':9,'total_length':11,'hemisphere':'cross'}
    # Flipping n and offset swaps the length labels, preserving geometry.
    reverse=neuron_statistics(path,plane(180,d=-10))
    assert reverse['right_length']==pytest.approx(2)
    assert reverse['left_length']==pytest.approx(9)


def test_fallback_unique_root_and_plane_boundary(tmp_path):
    path=tmp_path/'root.swc';path.write_text('1 3 0 0 0 1 -1\n2 3 0 3 4 1 1\n')
    result=neuron_statistics(path,plane())
    assert result['soma']['side']=='right' and result['right_length']==5 and result['left_length']==0
    path.write_text('1 3 -2 0 0 1 -1\n')
    result=neuron_statistics(path,plane());assert result['hemisphere']=='left' and result['total_length']==0


def test_counts_use_export_length_classes_including_zero_length_fallback(tmp_path):
    sources=[
        ('1 1 -3 0 0 1 -1\n2 3 -1 0 0 1 1\n','left'),
        ('1 1 1 0 0 1 -1\n2 3 3 0 0 1 1\n','right'),
        ('1 1 -1 0 0 1 -1\n2 3 2 0 0 1 1\n','cross'),
        ('1 3 0 0 0 1 -1\n','right'),
        # A left-hand segment ending on the plane still has left-only length.
        ('1 1 -1 0 0 1 -1\n2 3 0 0 0 1 1\n','left'),
        # Isolated right-hand nodes do not override a positive left length.
        ('1 1 -3 0 0 1 -1\n2 3 -1 0 0 1 1\n3 3 1 0 0 1 -1\n','left'),
    ]
    neurons=[]
    for i,(source,expected) in enumerate(sources):
        path=tmp_path/f'{i}.swc';path.write_text(source)
        exported=neuron_statistics(path,plane())
        counted=neuron_statistics(path,plane(),include_soma=False)
        assert exported['hemisphere']==counted['hemisphere']==expected
        neurons.append(counted)
    assert hemisphere_counts(neurons)==dict(total=6,left_only=3,right_only=2,cross=1,left=4,right=3)
    assert hemisphere_counts([])==dict(total=0,left_only=0,right_only=0,cross=0,left=0,right=0)


def test_counting_does_not_require_unique_soma(tmp_path):
    path=tmp_path/'ambiguous.swc';path.write_text('1 1 -1 0 0 1 -1\n2 1 1 0 0 1 1\n')
    assert neuron_statistics(path,plane(),include_soma=False)['hemisphere']=='cross'
    with pytest.raises(ValueError,match='胞体候选'):neuron_statistics(path,plane())


@pytest.mark.parametrize('source',[
    '1 1 0 0 0 1 -1\n2 1 1 0 0 1 1\n',
    '1 3 0 0 0 1 -1\n2 3 1 0 0 1 -1\n',
    '1 3 0 0 0 1 7\n', '# empty\n',
    '1 1 0 0 0 1 -1\n2 3 1 0 0 1 99\n',
    '1 1 nan 0 0 1 -1\n',
])
def test_ambiguous_missing_and_malformed_sources_rejected(tmp_path,source):
    path=tmp_path/'invalid.swc';path.write_text(source)
    with pytest.raises(ValueError):neuron_statistics(path,plane())


def test_export_never_overwrites_and_publishes_complete_json(tmp_path):
    report={'description':DESCRIPTION,'plane':plane_parameters(plane()),'neurons':[]}
    output=tmp_path/'output';output.mkdir()
    path=write_report(str(output),report)
    assert json.loads((output/'brain_hemisphere.json').read_text())==report
    assert str(output/'brain_hemisphere.json')==path
    with pytest.raises(ValueError,match='存在'):write_report(str(output),report)
    assert sorted(p.name for p in output.iterdir())==['brain_hemisphere.json']
    linkdir=tmp_path/'link';linkdir.mkdir();(linkdir/'brain_hemisphere.json').symlink_to(path)
    with pytest.raises(ValueError,match='存在'):write_report(str(linkdir),report)
    assert json.loads((output/'brain_hemisphere.json').read_text())==report
    broken=tmp_path/'broken';broken.mkdir();(broken/'brain_hemisphere.json').symlink_to(tmp_path/'absent')
    with pytest.raises(ValueError,match='存在'):write_report(str(broken),report)
    with pytest.raises(FileNotFoundError):write_report(str(tmp_path/'missing'),report)


def test_concurrent_exports_have_one_winner(tmp_path):
    report={'description':DESCRIPTION,'plane':plane_parameters(plane()),'neurons':[]}
    def save(_):
        try:return write_report(str(tmp_path),report)
        except ValueError:return None
    with ThreadPoolExecutor(2) as clients:results=list(clients.map(save,range(2)))
    assert sum(value is not None for value in results)==1
    assert len(list(tmp_path.iterdir()))==1


def test_custom_export_path_requires_explicit_overwrite(tmp_path):
    target=tmp_path/'custom.json'
    old={'neurons': ['old']};new={'neurons': ['new']}
    assert write_report(str(target),old)==str(target)
    before=target.read_bytes(),target.stat().st_mtime_ns
    with pytest.raises(ExportExistsError) as error:
        write_report(str(target),new)
    assert error.value.path==str(target)
    assert (target.read_bytes(),target.stat().st_mtime_ns)==before
    write_report(str(target),new,overwrite=True)
    assert json.loads(target.read_text())==new
    assert sorted(p.name for p in tmp_path.iterdir())==['custom.json']
    # Failed serialization must leave the previous complete report intact.
    with pytest.raises(ValueError):write_report(str(target),{'invalid':float('nan')},overwrite=True)
    assert json.loads(target.read_text())==new
    with pytest.raises(ValueError):write_report(str(target),new,overwrite='false')
    source=tmp_path/'source.swc';source.write_text('original')
    with pytest.raises(ValueError):write_report(str(source),new,overwrite=True)
    assert source.read_text()=='original'


@pytest.mark.parametrize('kind',['symlink','hardlink'])
def test_confirmed_export_replaces_link_without_modifying_source(tmp_path,kind):
    import os
    source=tmp_path/'original.swc';source.write_text('source data')
    target=tmp_path/'report.json'
    if kind=='symlink':target.symlink_to(source)
    else:os.link(source,target)
    before=source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode
    with pytest.raises(ExportExistsError):write_report(str(target),{'new':True})
    write_report(str(target),{'new':True},overwrite=True)
    assert json.loads(target.read_text())=={'new':True}
    assert not target.is_symlink()
    assert (source.read_bytes(),source.stat().st_mtime_ns,source.stat().st_mode)==before


def test_export_writer_is_prestarted_before_request_guard(tmp_path):
    import threading
    from swclib.visualization.readonly import ReadOnlyGuard
    writer=ExportWriter();errors=[]
    source=tmp_path/'source.swc';source.write_text('original')
    def request():
        try:
            ReadOnlyGuard().install()
            with pytest.raises(PermissionError):source.write_text('changed')
            writer.write(str(tmp_path),{'plane':plane_parameters(plane()),'neurons':[]})
            with pytest.raises(PermissionError):source.unlink()
        except BaseException as exc:errors.append(exc)
    thread=threading.Thread(target=request);thread.start();thread.join(10);writer.close()
    assert not thread.is_alive() and not errors
    assert source.read_text()=='original'
