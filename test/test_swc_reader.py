"""SWC region queries preserve topology and only read relevant neurons."""

from pathlib import Path
import weakref

import numpy as np
import pytest

from swclib.data.swc import Swc
from swclib.whole_brain.swc_reader import SwcReader


def _write_swc(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# id type x y z radius parent\n" + "".join(
        " ".join(map(str, row)) + "\n" for row in rows
    ))
    return path


def test_single_file_region_preserves_topology_and_local_coordinates(tmp_path):
    path = _write_swc(tmp_path / "neuron.swc", [
        (90, 1, 0, 2, 3, 2, -1),
        (30, 3, 1, 2, 3, 0.5, 90),
        (10, 2, 2, 3, 4, 0.75, 30),
        (70, 3, 3, 4, 5, 1, 10),
        (50, 3, 2, 3, 4, 1.5, 999),
    ])
    reader = SwcReader(path, rescale=(2, 3, 4))
    reader.add_offset((10, -2, 5))
    start, end = (12, 4, 17), (16, 20, 30)
    np.testing.assert_array_equal(
        reader.query_cube_mask((*start, *end)), [False, True, True, False, True]
    )
    assert reader.check_cube_nonempty((*start, *end))
    result = reader.read_region(start, end)
    assert result.nodes == {
        1: dict(id=1, type=3, x=0.0, y=0.0, z=0.0, radius=0.5, parent=-1),
        2: dict(id=2, type=2, x=2.0, y=3.0, z=4.0, radius=0.75, parent=1),
        3: dict(id=3, type=3, x=2.0, y=3.0, z=4.0, radius=1.5, parent=-1),
    }
    assert result.edges == [(1, -1), (2, 1), (3, -1)]
    assert result.bound_box == [0, 0, 0, 2, 3, 4]
    assert reader.id2idx == {90: 0, 30: 1, 10: 2, 70: 3, 50: 4}
    assert reader.ids.dtype == reader.parents.dtype == np.int64
    assert reader.xs.dtype == np.float32

    out_path = tmp_path / "output" / "crop.swc"
    overridden = reader.read_region(start, end, out_path=out_path, out_r=2.5)
    assert all(node["radius"] == 2.5 for node in overridden.nodes.values())
    assert Swc(out_path).nodes == overridden.nodes
    np.testing.assert_array_equal(reader.rs, [2, 0.5, 0.75, 1, 1.5])


def test_single_file_empty_region_and_half_open_boundary(tmp_path):
    path = _write_swc(tmp_path / "neuron.swc", [(1, 1, 1, 2, 3, 1, -1)])
    reader = SwcReader(path)
    assert reader.check_cube_nonempty((1, 2, 3, 2, 3, 4))
    assert not reader.check_cube_nonempty((0, 0, 0, 1, 2, 3))
    out_path = tmp_path / "empty.swc"
    result = reader.read_region((0, 0, 0), (1, 2, 3), out_path=out_path)
    assert not result.nodes
    assert not result.edges
    assert out_path.read_text() == ""


def test_empty_single_file(tmp_path):
    path = _write_swc(tmp_path / "empty.swc", [])
    reader = SwcReader(path)
    assert reader.query_cube_mask((0, 0, 0, 1, 1, 1)).size == 0
    assert not reader.check_cube_nonempty((0, 0, 0, 1, 1, 1))
    assert not reader.read_region((0, 0, 0), (1, 1, 1)).nodes


def test_directory_merge_keeps_neuron_parent_ids_separate(tmp_path):
    directory = tmp_path / "neurons"
    _write_swc(directory / "a.swc", [
        (10, 1, 1, 1, 1, 2, -1),
        (30, 3, 2, 1, 1, 0.5, 10),
        (50, 3, 3, 1, 1, 0.75, 30),
    ])
    _write_swc(directory / "b.swc", [
        (10, 1, 1, 2, 1, 3, -1),
        (30, 2, 2, 2, 1, 1.5, 10),
        # ID 50 exists only in a.swc, so it must not become this node's parent.
        (90, 2, 3, 2, 1, 1.75, 50),
    ])
    _write_swc(directory / "far.swc", [(10, 1, 100, 100, 100, 1, -1)])
    reader = SwcReader(directory)
    out_path = tmp_path / "combined.swc"
    result = reader.read_region((2, 1, 1), (4, 3, 2), out_path=out_path)
    assert result.nodes == {
        1: dict(id=1, type=3, x=0.0, y=0.0, z=0.0, radius=0.5, parent=-1),
        2: dict(id=2, type=3, x=1.0, y=0.0, z=0.0, radius=0.75, parent=1),
        3: dict(id=3, type=2, x=0.0, y=1.0, z=0.0, radius=1.5, parent=-1),
        4: dict(id=4, type=2, x=1.0, y=1.0, z=0.0, radius=1.75, parent=-1),
    }
    assert Swc(out_path).nodes == result.nodes
    assert result.bound_box == [0, 0, 0, 1, 1, 0]
    overridden = reader.read_region((2, 1, 1), (4, 3, 2), out_r=0)
    assert all(node["radius"] == 0 for node in overridden.nodes.values())


def test_directory_bbox_overlap_does_not_imply_nonempty(tmp_path):
    # The segment crosses the cube, but the existing reader selects nodes.
    _write_swc(tmp_path / "crossing.swc", [
        (1, 1, 0, 0, 0, 1, -1), (2, 3, 10, 10, 10, 1, 1),
    ])
    reader = SwcReader(tmp_path)
    assert not reader.check_cube_nonempty((4, 4, 4, 6, 6, 6))
    assert not reader.read_region((4, 4, 4), (6, 6, 6)).nodes
    assert reader.check_cube_nonempty((10, 10, 10, 11, 11, 11))
    assert not reader.check_cube_nonempty((1, 1, 1, 10, 10, 10))


def test_directory_queries_load_only_candidates_and_release_arrays(tmp_path, monkeypatch):
    _write_swc(tmp_path / "a.swc", [(10, 1, 1, 1, 1, 1, -1)])
    _write_swc(tmp_path / "b.swc", [
        (10, 1, 100, 100, 100, 1, -1), (20, 3, 101, 101, 101, 1, 10),
    ])
    _write_swc(tmp_path / "c.swc", [(10, 3, 2, 2, 2, 1, -1)])
    calls, arrays = [], []
    original = SwcReader.load_swc

    def tracked(self, path, rescale):
        # The previous neuron's arrays must be released before loading another.
        assert all(ref() is None for ref in arrays)
        calls.append(Path(path).name)
        result = original(self, path, rescale)
        arrays.extend(weakref.ref(column) for column in result[:7])
        return result

    monkeypatch.setattr(SwcReader, "load_swc", tracked)
    reader = SwcReader(tmp_path)
    assert calls == []  # Indexing never loads complete neuron arrays.
    assert not any(name in reader.__dict__ for name in SwcReader._COLUMNS)

    # Empty space requires no file reads at all.
    assert not reader.read_region((20, 20, 20), (30, 30, 30)).nodes
    assert not reader.check_cube_nonempty((20, 20, 20, 30, 30, 30))
    assert not reader.query_cube_mask((20, 20, 20, 30, 30, 30)).any()
    assert calls == []

    cube = (0, 0, 0, 3, 3, 3)
    np.testing.assert_array_equal(reader.query_cube_mask(cube), [True, False, False, True])
    assert calls == ["a.swc", "c.swc"]
    calls.clear()
    assert reader.check_cube_nonempty(cube)
    assert calls == ["a.swc"]  # Short circuit at the first actual hit.
    calls.clear()
    assert len(reader.read_region(cube[:3], cube[3:]).nodes) == 2
    assert calls == ["a.swc", "c.swc"]
    assert all(ref() is None for ref in arrays)
    assert not any(name in reader.__dict__ for name in SwcReader._COLUMNS)


def test_directory_global_arrays_align_with_mask_and_keep_parent_links(tmp_path, monkeypatch):
    # Creation order differs from the deterministic filename order.
    _write_swc(tmp_path / "b.swc", [
        (90, 3, 4, 2, 3, 0.5, 10),  # parent can follow the child in a file
        (10, 1, 3, 2, 3, 1, -1),
        (30, 3, 5, 2, 3, 1, 50),  # ID 50 exists only in the other neuron
    ])
    _write_swc(tmp_path / "a.swc", [
        (10, 1, 1, 2, 3, 2, -1), (50, 2, 2, 3, 4, 3, 10),
    ])
    reader = SwcReader(tmp_path)
    reader.add_offset((10, -2, 5))
    cube = (12, 0, 8, 16, 2, 10)
    mask = reader.query_cube_mask(cube)
    before = reader.read_region(cube[:3], cube[3:])
    np.testing.assert_array_equal(mask, [False, True, True, True, True])

    np.testing.assert_array_equal(reader.xs, [11, 12, 14, 13, 15])
    np.testing.assert_array_equal(reader.ys, [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(reader.zs, [8, 9, 8, 8, 8])
    np.testing.assert_array_equal(reader.ids, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(reader.parents, [-1, 1, 4, -1, -1])
    np.testing.assert_array_equal(reader.types, [1, 2, 3, 1, 3])
    np.testing.assert_array_equal(reader.rs, [2, 3, 0.5, 1, 1])
    assert reader.id2idx == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    assert [p.name for p in reader.swc_paths] == ["a.swc", "b.swc"]

    def unexpected_load(*args, **kwargs):
        pytest.fail("Materialized arrays should be reused")

    monkeypatch.setattr(SwcReader, "load_swc", unexpected_load)
    np.testing.assert_array_equal(reader.query_cube_mask(cube), mask)
    after = reader.read_region(cube[:3], cube[3:])
    assert before.nodes == after.nodes
    assert before.edges == after.edges
    assert before.bound_box == after.bound_box
    assert reader.xs is reader.xs

    # In-memory edits and subsequent offsets remain visible to all queries.
    reader.xs[0] = -10
    reader.add_offset((1, 0, 0))
    assert reader.check_cube_nonempty((-9, 0, 8, -8, 1, 9))
    result = reader.read_region((-9, 0, 8), (-8, 1, 9))
    assert len(result.nodes) == 1
    assert result.nodes[1]["x"] == 0


@pytest.mark.parametrize("rescale", [
    (0.1, 0.3, 1.7), (-2, 0, 0.5),
    np.array([0.1, 0.3, 1.7]), (np.float64(0.1), 0.3, 1.7),
])
def test_directory_scaling_offsets_and_float_boundaries_match_single_file(tmp_path, monkeypatch, rescale):
    path = _write_swc(tmp_path / "neuron.swc", [
        (9, 1, 0.1, -2.3, 3.7, 1, -1),
        (1, 3, -100.3, 20.7, -3.1, 1, 9),
        (4, 3, 12345.6, 0.7, 30.2, 1, 1),
        (8, 3, 10.3, 4.2, -10.8, 1, 4),
        (7, 3, -12.2, -6.1, 40.1, 1, 8),
    ])
    # Exercise multiple complete index chunks and one partial final chunk.
    monkeypatch.setattr(SwcReader, "_INDEX_CHUNK_SIZE", 2)
    directory = SwcReader(tmp_path, rescale=rescale)
    single = SwcReader(path, rescale=rescale)
    np.testing.assert_array_equal(
        single.xs, np.array([0.1, -100.3, 12345.6, 10.3, -12.2], dtype=np.float32) * rescale[0]
    )
    for offset in ((1e6, 2.9, -0.3), (-1e6, -4.6, 0.9), (0.01, 0.2, 1.2)):
        directory.add_offset(offset)
        single.add_offset(offset)
    for x, y, z in zip(single.xs, single.ys, single.zs):
        # Query at and just above each float32 point to catch index/point-mask
        # precision differences, including the upper bounding-box face.
        for epsilon in (0, 1e-10):
            start = (float(x) + epsilon, float(y) + epsilon, float(z) + epsilon)
            end = tuple(v + 1 for v in start)
            cube = (*start, *end)
            np.testing.assert_array_equal(directory.query_cube_mask(cube), single.query_cube_mask(cube))
            assert directory.check_cube_nonempty(cube) == single.check_cube_nonempty(cube)
            assert directory.read_region(start, end).nodes == single.read_region(start, end).nodes
    for column in ("xs", "ys", "zs"):
        np.testing.assert_array_equal(getattr(directory, column), getattr(single, column))
        assert getattr(directory, column).dtype == getattr(single, column).dtype


def test_directory_discovery_empty_files_and_parser_alignment(tmp_path):
    _write_swc(tmp_path / "b.SWC", [(2, 1, 2, 2, 2, 1, -1)])
    (tmp_path / "a.swc").write_text(
        "  # indented comment\n\n1 2 3\n1 1 1 1 1 1 -1\n"
    )
    _write_swc(tmp_path / "empty.swc", [])
    _write_swc(tmp_path / "ignored.txt", [(1, 1, 0, 0, 0, 1, -1)])
    _write_swc(tmp_path / "nested" / "a.swc", [(1, 1, 3, 3, 3, 1, -1)])
    reader = SwcReader(tmp_path)
    assert [p.name for p in reader.swc_paths] == ["a.swc", "b.SWC", "empty.swc"]
    np.testing.assert_array_equal(reader.query_cube_mask((0, 0, 0, 4, 4, 4)), [True, True])
    recursive = SwcReader(tmp_path, recursive=True)
    np.testing.assert_array_equal(recursive.xs, [1, 2, 3])


def test_empty_directory_and_changed_neuron(tmp_path):
    reader = SwcReader(tmp_path)
    assert reader.query_cube_mask((0, 0, 0, 1, 1, 1)).shape == (0,)
    assert not reader.check_cube_nonempty((0, 0, 0, 1, 1, 1))
    assert not reader.read_region((0, 0, 0), (1, 1, 1)).nodes
    assert reader.xs.size == 0
    assert reader.id2idx == {}
    with pytest.raises(FileNotFoundError):
        SwcReader(tmp_path / "missing.swc")

    path = _write_swc(tmp_path / "changed.swc", [(1, 1, 0, 0, 0, 1, -1)])
    reader = SwcReader(tmp_path)
    _write_swc(path, [])
    with pytest.raises(ValueError, match="changed after indexing"):
        reader.read_region((0, 0, 0), (1, 1, 1))
