"""Soma candidates retain source coordinates and never infer fitted centers."""
import pytest

from swclib.visualization.plugins import soma_candidates


def test_prefers_all_type_one_nodes_including_non_roots(tmp_path):
    path=tmp_path/'neuron.swc'
    path.write_text('1 3 10 20 30 1 -1\n2 1 11.25 20 30 2 1\n3 1 12 21 30 2 2\n4 3 40 50 60 1 -1\n')
    result=soma_candidates(path,'type-first')
    assert result['node_count']==4
    assert result['candidates']==[
        {'node_id':2,'center':[11.25,20,30],'method':'type=1'},
        {'node_id':3,'center':[12,21,30],'method':'type=1'}]


def test_roots_fallback_retains_multiple_candidates(tmp_path):
    path=tmp_path/'roots.swc'
    path.write_text('7 3 5 6 7 1 -1\n2 2 8 9 10 1 -1\n8 3 4 5 6 1 99\n')
    result=soma_candidates(path,'type-first')
    assert [c['node_id'] for c in result['candidates']]==[2,7]
    assert all(c['method']=='parent=-1' for c in result['candidates'])
    assert soma_candidates(path,'type-only')['candidates']==[]


def test_missing_malformed_and_nonfinite_candidates(tmp_path):
    path=tmp_path/'neuron.swc'
    path.write_text('# no nodes\n')
    assert soma_candidates(path,'type-first')=={'node_count':0,'candidates':[]}
    with pytest.raises(ValueError):soma_candidates(path,'unknown')
    path.write_text('1 1 nan 2 3 1 -1\n')
    with pytest.raises(ValueError,match='有限'):soma_candidates(path,'type-first')
    path.write_text('1 1 invalid 2 3 1 -1\n')
    with pytest.raises(ValueError):soma_candidates(path,'type-first')
