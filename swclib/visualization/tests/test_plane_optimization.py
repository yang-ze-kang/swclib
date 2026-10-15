"""Plane-search classification agrees with the full export's length rule."""

import math

import numpy as np
import pytest

from swclib.visualization.hemisphere import (candidate_counts, hemisphere_counts, neuron_plane_classes,
    neuron_plane_summary, neuron_statistics, plane_parameters, summary_plane_classes)
from swclib.visualization.plane_optimization import balance_score, optimize_plane


def plane(a=0, e=0, d=0):
    return dict(azimuth_deg=a, elevation_deg=e, offset=d)


@pytest.mark.parametrize('source', [
    '1 1 -1 0 0 1 -1\n2 3 0 0 0 1 1\n',
    '1 1 0 0 0 1 -1\n2 3 0 1 0 1 1\n',
    '1 1 -1 0 0 1 -1\n2 3 1 0 0 1 1\n',
    '1 1 -3 0 0 1 -1\n2 3 -1 0 0 1 1\n3 3 1 0 0 1 -1\n',
    '1 3 -1 0 0 1 -1\n2 3 1 0 0 1 -1\n',
    '1 3 0 0 0 1 -1\n2 3 0 0 0 1 1\n',
    '1 1 -2 1 0 1 -1\n2 3 -1 1 0 1 1\n3 3 0 0 0 1 -1\n4 3 0 0 1 1 3\n',
])
def test_batched_classes_match_export_at_boundaries_and_isolated_nodes(tmp_path, source):
    path = tmp_path / 'neuron.swc'
    path.write_text(source)
    planes = [plane(a, e, d) for a in (0, 60, 180, 270) for e in (0, 90) for d in (-1, 0, 1)]
    labels = {'left': 1, 'right': 2, 'cross': 3}
    expected = [labels[neuron_statistics(path, p, include_soma=False)['hemisphere']] for p in planes]
    np.testing.assert_array_equal(neuron_plane_classes(path, planes), expected)


def test_batched_classification_reads_once_and_handles_multiple_edge_chunks(tmp_path, monkeypatch):
    import swclib.visualization.hemisphere as module
    path = tmp_path / 'long.swc'
    path.write_text(''.join(f'{i+1} 3 {1000000+i*.01} {i*.02} {i%7} 1 {i if i else -1}\n' for i in range(4100)))
    planes = [plane(d=1000000+i) for i in range(50)]
    labels = {'left': 1, 'right': 2, 'cross': 3}
    expected = [labels[neuron_statistics(path, p, include_soma=False)['hemisphere']] for p in planes]
    original = module.Swc
    calls = []
    def tracked(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(module, 'Swc', tracked)
    np.testing.assert_array_equal(neuron_plane_classes(path, planes), expected)
    assert len(calls) == 1


def test_precise_plane_boundary_matches_export_fsum(tmp_path):
    from swclib.visualization.hemisphere import plane_parameters
    path = tmp_path / 'point.swc'
    xyz = [1234567.89, -9876543.21, 2345678.91]
    path.write_text('1 3 ' + ' '.join(map(str, xyz)) + ' 1 -1\n')
    planes = []
    for a in range(0, 360, 7):
        p = plane(a, 23)
        p['offset'] = math.fsum(n*x for n,x in zip(plane_parameters(p)['normal_xyz'], xyz))
        planes.append(p)
    np.testing.assert_array_equal(neuron_plane_classes(path, planes), np.full(len(planes), 2))


@pytest.mark.parametrize('source', ['# empty\n', '1 3 nan 0 0 1 -1\n', '1 3 0 0 0 1 99\n'])
def test_candidate_classification_rejects_bad_sources(tmp_path, source):
    path = tmp_path / 'bad.swc'
    path.write_text(source)
    with pytest.raises(ValueError):
        neuron_plane_classes(path, [plane()])


def test_search_reduces_crossings_and_all_suggestions_obey_constraints(tmp_path):
    paths = []
    for side in (-1, 1):
        path = tmp_path / f'{side}.swc'
        path.write_text(f'1 1 99.8 {200+side*10} 299.5 1 -1\n2 1 100.2 {200+side*10} 300.5 1 1\n')
        paths.append(path)
    batches = []
    def evaluate(planes):
        batches.append(len(planes))
        return candidate_counts([neuron_plane_classes(path, planes) for path in paths])
    original, center = plane(d=100), [100, 200, 300]
    result = optimize_plane(original, center, 10, evaluate)
    assert result['before_counts']['cross'] == 2
    assert result['counts']['cross'] == 0
    assert result['counts']['left_only'] == result['counts']['right_only'] == 1
    assert len(batches) <= 4 and sum(batches) == result['evaluations']
    assert len(result['suggestions']) == 3
    assert result['plane'] == result['suggestions'][0]['plane']
    for candidate in result['suggestions']:
        p = candidate['plane']
        simple = {key: p[key] for key in original}
        assert math.fsum(a*b for a,b in zip(p['normal_xyz'], center)) == p['offset']
        angle = math.degrees(math.acos(np.clip(p['normal_xyz'][0], -1, 1)))
        assert angle <= 10 + 1e-10
        assert angle == pytest.approx(candidate['angle_change_deg'], abs=1e-10)
        assert hemisphere_counts([neuron_statistics(path, simple, include_soma=False) for path in paths]) == candidate['counts']
    assert optimize_plane(original, center, 10, evaluate) == result


def test_balance_threshold_and_crossing_priority():
    def score(left, right, cross):
        return balance_score(dict(left_only=left, right_only=right, cross=cross))
    assert score(10, 14, 2) < score(10, 10, 6)  # below 5: fewer crossings win
    assert score(10, 14, 20) < score(10, 15, 0)  # five does not meet the threshold
    assert score(10, 15, 20) < score(10, 16, 0)  # outside it: smaller difference wins
    assert score(10, 10, 2) < score(10, 14, 2)  # same crossings: smaller difference


@pytest.mark.parametrize('a,e,limit', [(359, 0, 10), (45, 90, 10), (170, -90, 90), (20, 25, 0)])
def test_center_is_mandatory_and_ties_keep_orientation(a, e, limit):
    center = [17, -83, 90]
    counts = dict(left_only=8, right_only=8, cross=2, total=18, left=10, right=10)
    def evaluate(planes):
        n0 = np.asarray(plane_parameters(plane(a, e))['normal_xyz'])
        for p in planes[1:]:
            n = plane_parameters(p)['normal_xyz']
            assert math.fsum(v*c for v,c in zip(n, center)) == p['offset']
            assert math.degrees(math.acos(np.clip(n0 @ n, -1, 1))) <= limit + 1e-6
        return [counts] * len(planes)
    result = optimize_plane(plane(a, e, -999), center, limit, evaluate)
    assert result['angle_change_deg'] == 0
    assert result['plane']['offset'] != -999
    assert result['plane']['azimuth_deg'] == a and result['plane']['elevation_deg'] == e
    if limit == 0:
        assert result['evaluations'] == 2 and len(result['suggestions']) == 1


@pytest.mark.parametrize('center,limit', [(None,10), ([0,0],10), ([0,0,float('nan')],10), ([0,0,0],-1), ([0,0,0],91), ([0,0,0],float('inf'))])
def test_invalid_constraints_rejected_before_reading(center, limit):
    def unread(planes):
        pytest.fail('Invalid request triggered source reading')
    with pytest.raises(ValueError):
        optimize_plane(plane(), center, limit, unread)


def test_bounded_summary_never_guesses_uncertain_classes(tmp_path):
    path = tmp_path/'long.swc'
    rng = np.random.default_rng(37)
    xyz = rng.normal(size=(1500, 3)).cumsum(axis=0)
    path.write_text(''.join(f'{i+1} 3 {x} {y} {z} 1 {i if i else -1}\n' for i,(x,y,z) in enumerate(xyz)))
    summary = neuron_plane_summary(path)
    assert 'points' not in summary and len(summary['witnesses']) <= 512
    assert summary['node_count'] == 1500
    planes = [plane(a, e, d) for a,e,d in zip(rng.uniform(0,360,200), rng.uniform(-90,90,200), rng.uniform(-70,70,200))]
    # Include exact touching planes: these must request the full edge test.
    planes.append(plane(d=xyz[:,0].max()))
    classes = summary_plane_classes(summary, planes)
    expected = neuron_plane_classes(path, planes)
    assert (classes != 0).sum() > 150
    np.testing.assert_array_equal(classes[classes != 0], expected[classes != 0])
    assert classes[-1] == 0


def test_large_summary_ignores_isolated_nodes_when_edges_have_length(tmp_path):
    path = tmp_path/'islands.swc'
    path.write_text('1 3 -3 0 0 1 -1\n2 3 -1 0 0 1 1\n' +
                    ''.join(f'{i} 3 1 0 0 1 -1\n' for i in range(3, 800)))
    summary = neuron_plane_summary(path)
    np.testing.assert_array_equal(summary_plane_classes(summary, [plane()]), [1])


def test_small_summary_preserves_exact_edges_and_boundary_fallback(tmp_path):
    path = tmp_path/'boundary.swc'
    path.write_text('1 3 -1 0 0 1 -1\n2 3 0 0 0 1 1\n3 3 0 1 0 1 2\n')
    summary = neuron_plane_summary(path)
    assert 'points' in summary
    np.testing.assert_array_equal(summary_plane_classes(summary, [plane(), plane(d=1)]), [3, 1])


@pytest.mark.parametrize('origin,scale', [([0,0,0],[1,1,1]), ([1e10,-1e10,2e10],[100,20,3]), ([0,0,0],[1,1,1e-8])])
def test_verified_support_envelope_bounds_every_projection(origin, scale):
    from swclib.visualization.hemisphere import _support_envelope
    rng = np.random.default_rng(134)
    points = rng.normal(size=(3000,3))*scale+origin
    envelope = _support_envelope(points)
    assert envelope is not None
    support, error = envelope
    assert len(support) <= 512 and 0 <= error < np.ptp(points,axis=0).max()
    normals = rng.normal(size=(200,3));normals /= np.linalg.norm(normals,axis=1)[:,None]
    full, compact = points @ normals.T, support @ normals.T
    assert (full.max(axis=0) <= compact.max(axis=0)+error).all()
    assert (full.min(axis=0) >= compact.min(axis=0)-error).all()
