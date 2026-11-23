"""Replay uses recorded states, including removals, rejected stitches and incomplete writes."""
import json

import numpy as np
import pytest

from swclib.visualization.trace_replay import Replay, ReplayService


def fixture(root):
    root.mkdir(exist_ok=True)
    source = root / 'trace_steps.jsonl'
    records = [dict(event='header',version=1,coordinate_frame='global_xyz_voxels',
                    seed_xyz=[10,20,30],history_xyz=[[8,20,30],[10,20,30]],iteration=4),
               dict(event='prediction',seed_xyz=[10,20,30],added_nodes=[[1,0,12,20,30]],
                    removed_nodes=[],candidates=[],forward_steps=5),
               dict(event='prediction',seed_xyz=[12,20,30],added_nodes=[],removed_nodes=[1],
                    candidates=[],forward_steps=5),
               dict(event='end',status='complete')]
    source.write_text(''.join(json.dumps(record)+'\n' for record in records))
    return source


def test_rejected_stitch_restores_global_state_and_seek_is_repeatable(tmp_path):
    fixture(tmp_path)
    for stage in ['before','after']:
        np.savez_compressed(tmp_path/f'trace_tree_{stage}.npz',points_xyz=[[8,20,30],[10,20,30]],parents=[-1,0])
    (tmp_path/'trace_state.json').write_text(json.dumps(dict(accepted=False,rolled_back=True)))
    replay = Replay(tmp_path)
    assert not replay.warnings
    before = replay.frame(0)
    assert len(before['traced_edges_xyz']) == 2
    after = replay.frame(1)
    assert after['outcome']['rolled_back']
    assert after['traced_edges_xyz'] == [[[8,20,30],[10,20,30]]]
    assert replay.frame(0) == before
    assert replay.frame(1) == after


def test_partial_record_does_not_invent_completed_steps(tmp_path):
    source = fixture(tmp_path)
    source.write_text('\n'.join(source.read_text().splitlines()[:-1])+'\n{"event":')
    replay = Replay(tmp_path)
    assert len(replay.frames) == 2
    assert any('未完成' in message for message in replay.warnings)
    assert any('局部历史' in message for message in replay.warnings)
    assert len(replay.frame(1)['traced_edges_xyz']) == 1


def test_gt_uses_swc_parser_and_declared_coordinate_transform(tmp_path):
    fixture(tmp_path)
    gt = tmp_path/'gt.swc'
    gt.write_text('1 1 1 2 3 1 -1\n2 3 2 4 6 1 1\n')
    (tmp_path/'trace_replay.json').write_text(json.dumps(dict(ground_truth=str(gt),
        swc_to_voxel_scale_xyz=[2,3,4],swc_to_voxel_offset_xyz=[10,20,30])))
    replay = Replay(tmp_path)
    assert replay.gt == [[[12,26,42],[14,32,54]]]


def test_replaced_session_and_invalid_index_are_rejected(tmp_path):
    fixture(tmp_path)
    service = ReplayService()
    first = service.open(tmp_path)
    second = service.open(tmp_path)
    with pytest.raises(ValueError,match='替换'):
        service.frame(first['id'],0)
    for index in [True, -1, 4, 1.5]:
        with pytest.raises(ValueError,match='超出'):
            service.frame(second['id'],index)
