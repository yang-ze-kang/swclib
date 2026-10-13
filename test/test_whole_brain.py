"""End-to-end voxel/axis checks against independently written test volumes."""

import gzip
from itertools import product

import nrrd
import numpy as np
import pytest
import tifffile

from swclib.whole_brain import FNTReader, WBTReader, WholeBrainReader


def _write_nrrd(path, data, encoding="gzip", endian="little"):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "NRRD0004\n"
        "type: ushort\n"
        "dimension: 3\n"
        f"sizes: {' '.join(map(str, data.shape[::-1]))}\n"
        f"encoding: {encoding}\n"
        f"endian: {endian}\n\n"
    )
    raw = data.astype("<u2" if endian == "little" else ">u2").tobytes()
    path.write_bytes(header.encode() + (gzip.compress(raw) if encoding == "gzip" else raw))


@pytest.fixture
def volume():
    z, y, x = np.indices((5, 7, 9))
    return (100 * z + 10 * y + x).astype(np.uint16)


def _write_fnt(root, volume, encoding="gzip", endian="little"):
    root.mkdir(exist_ok=True)
    (root / "catalog").write_text(
        "[CH00DS]\nlocation=ch00ds.nrrd\n"
        "[CH00]\npattern=ch00/z<08Z>/y<08Y>.x<08X>.nrrd\n"
        "size=9 7 5\ncubesize=5 4 3\norigin=1 2 3\n"
        "direction=0.35 0 0 0 0.35 0 0 0 1\n"
    )
    for z, y, x in product(range(0, 5, 3), range(0, 7, 4), range(0, 9, 5)):
        _write_nrrd(root / f"ch00/z{z:08d}/y{y:08d}.x{x:08d}.nrrd",
                    volume[z:z + 3, y:y + 4, x:x + 5], encoding, endian)
    return root


@pytest.fixture
def fnt_dir(tmp_path, volume):
    return _write_fnt(tmp_path / "fnt", volume)


@pytest.mark.parametrize("backend,workers", [("thread", 1), ("thread", 3), ("process", 2)])
@pytest.mark.parametrize("encoding,endian", [("gzip", "little"), ("raw", "big")])
def test_fnt_reads_volume_and_cross_cube_crop(tmp_path, volume, backend, workers, encoding, endian):
    root = _write_fnt(tmp_path, volume, encoding, endian)
    reader = FNTReader(root)
    assert reader.get_dimensions() == volume.shape
    assert reader.get_bbox() == ((0, 0, 0), (9, 7, 5))
    assert reader.cube_shape == (3, 4, 5)
    assert reader.cubesize == (5, 4, 3)
    assert reader.origin == (1, 2, 3)
    assert reader.spacing == pytest.approx((0.35, 0.35, 1))
    options = dict(num_workers=workers, parallel_backend=backend)
    np.testing.assert_array_equal(reader.read_region((0, 0, 0), volume.shape, **options), volume)
    np.testing.assert_array_equal(reader.read_region((1, 2, 3), (5, 6, 8), **options), volume[1:5, 2:6, 3:8])
    np.testing.assert_array_equal(reader.read_region((4, 6, 8), (5, 7, 9), **options), volume[4:, 6:, 8:])


def test_fnt_only_decodes_intersecting_cubes(fnt_dir, volume, monkeypatch):
    import swclib.whole_brain.fntreader as module

    calls = []
    original = module.nrrd.read

    def tracked(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(module.nrrd, "read", tracked)
    reader = FNTReader(fnt_dir)
    assert not calls
    crop = reader.read_region((3, 4, 5), (5, 7, 9), num_workers=1)
    np.testing.assert_array_equal(crop, volume[3:, 4:, 5:])
    assert calls == [str(fnt_dir / "ch00/z00000003/y00000004.x00000005.nrrd")]


def test_fnt_channel_selection(fnt_dir, volume):
    with (fnt_dir / "catalog").open("a") as stream:
        stream.write("[CH01]\npattern=second/z<Z>/y<Y>.x<X>.nrrd\nsize=9 7 5\ncubesize=9 7 5\n")
    _write_nrrd(fnt_dir / "second/z0/y0.x0.nrrd", volume + 1000)
    reader = FNTReader(fnt_dir, channel="CH01")
    assert reader.channels == ("ch00", "ch01")
    np.testing.assert_array_equal(reader.read_region((0, 0, 0), volume.shape), volume + 1000)
    with pytest.raises(ValueError, match="Unknown cube channel"):
        FNTReader(fnt_dir, channel="ch00ds")


def test_fnt_missing_and_malformed_cube(fnt_dir, volume):
    reader = FNTReader(fnt_dir)
    path = fnt_dir / "ch00/z00000003/y00000004.x00000005.nrrd"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        reader.read_region((3, 4, 5), (5, 7, 9))
    _write_nrrd(path, volume[:1, :1, :1])
    with pytest.raises(ValueError, match="cube shape"):
        reader.read_region((3, 4, 5), (5, 7, 9))
    path.write_bytes(b"not a nrrd")
    with pytest.raises(nrrd.NRRDError, match="Invalid NRRD"):
        reader.read_region((3, 4, 5), (5, 7, 9))


def test_fnt_full_sized_edge_cube(fnt_dir, volume):
    data = np.zeros((3, 4, 5), dtype=np.uint16)
    data[:2, :3, :4] = volume[3:, 4:, 5:]
    _write_nrrd(fnt_dir / "ch00/z00000003/y00000004.x00000005.nrrd", data)
    reader = FNTReader(fnt_dir)
    np.testing.assert_array_equal(reader.read_region((3, 4, 5), (5, 7, 9)), volume[3:, 4:, 5:])


@pytest.fixture(params=["fnt", "slice"])
def reader(request, tmp_path, volume):
    if request.param == "fnt":
        return WholeBrainReader(_write_fnt(tmp_path, volume), mode="fnt")
    for z, data in enumerate(volume):
        directory = tmp_path / ("a" if z % 2 else "b")
        directory.mkdir(exist_ok=True)
        tifffile.imwrite(directory / f"slice_{z:03d}.tiff", data)
    return WholeBrainReader(tmp_path, mode="slice", slice_name_pattern=r"slice_(\d+)\.tiff", slice_ext="tiff")


def test_unified_reader_and_padding(reader, volume):
    assert reader.shape == volume.shape
    assert reader.get_dimensions() == volume.shape
    assert reader.get_bbox() == ((0, 0, 0), (9, 7, 5))
    assert np.dtype(reader.dtype) == volume.dtype
    np.testing.assert_array_equal(reader.read_region((1, 2, 3), (5, 7, 9)), volume[1:, 2:, 3:])
    expected = np.zeros((4, 4, 4), dtype=volume.dtype)
    expected[:2, :2, :2] = volume[3:, 5:, 7:]
    np.testing.assert_array_equal(reader.read_region((3, 5, 7), (7, 9, 11), padding="right"), expected)
    assert not reader.read_region((8, 9, 10), (10, 11, 12), padding="right").any()


@pytest.mark.parametrize("start,end,options", [
    ((-1, 0, 0), (2, 2, 2), {}),
    ((0, 0, 0), (6, 7, 9), {}),
    ((1, 1, 1), (1, 2, 2), {}),
    ((0.5, 0, 0), (2, 2, 2), {}),
    ((0, 0), (2, 2, 2), {}),
    ((0, 0, 0), (2, 2, 2), {"padding": "both"}),
    ((0, 0, 0), (2, 2, 2), {"num_workers": 0}),
    ((0, 0, 0), (2, 2, 2), {"parallel_backend": "unknown"}),
])
def test_invalid_regions(reader, start, end, options):
    with pytest.raises(ValueError):
        reader.read_region(start, end, **options)


@pytest.mark.parametrize("mode", ["raster", "tiff"])
@pytest.mark.parametrize("backend", ["thread", "process"])
def test_slices_nonzero_start_and_last_slice(tmp_path, volume, mode, backend):
    for z in range(2, 5):
        tifffile.imwrite(tmp_path / f"slice_{z}.tif", volume[z])
    reader = WBTReader(tmp_path, r"slice_(\d+)\.tif")
    lower, upper = reader.get_bbox()
    assert lower == (0, 0, 2)
    assert upper == (9, 7, 5)
    np.testing.assert_array_equal(
        reader.read_region(lower[::-1], upper[::-1], mode=mode, parallel_backend=backend, num_workers=2), volume[2:]
    )
    expected = np.zeros((3, 4, 4), dtype=volume.dtype)
    expected[:1, :2, :2] = volume[4:, 5:, 7:]
    np.testing.assert_array_equal(
        reader.read_region((4, 5, 7), (7, 9, 11), mode=mode, padding="right", num_workers=1), expected
    )
    with pytest.raises(ValueError, match="Invalid Z"):
        reader.read_region((0, 0, 0), (3, 2, 2))


def test_slice_directory_validation(tmp_path, volume):
    with pytest.raises(ValueError, match="no matching"):
        WBTReader(tmp_path, r"slice_(\d+)\.tif")
    tifffile.imwrite(tmp_path / "slice_0.tif", volume[0])
    tifffile.imwrite(tmp_path / "slice_2.tif", volume[2])
    with pytest.raises(ValueError, match="continuous"):
        WBTReader(tmp_path, r"slice_(\d+)\.tif")
    tifffile.imwrite(tmp_path / "slice_00.tif", volume[0])
    with pytest.raises(ValueError, match="unique"):
        WBTReader(tmp_path, r"slice_(\d+)\.tif")


def test_unified_invalid_mode(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        WholeBrainReader(tmp_path, mode="unknown")


@pytest.mark.parametrize("axis_order", ["ZYX", "zyx", "zYx", "XYZ", "xyz", "xYz"])
@pytest.mark.parametrize("kind", ["fnt", "raster", "tiff"])
@pytest.mark.parametrize("unified", [False, True])
def test_axis_order(tmp_path, volume, axis_order, kind, unified):
    if kind == "fnt":
        root = _write_fnt(tmp_path, volume)
        reader = (WholeBrainReader(root, mode="fnt", axis_order=axis_order) if unified
                  else FNTReader(root, axis_order=axis_order))
        options = {}
        start_z = 0
    else:
        # A nonzero first slice verifies that XYZ validation still checks Z.
        for z in range(2, 5):
            tifffile.imwrite(tmp_path / f"slice_{z}.tif", volume[z])
        options = {"slice_name_pattern": r"slice_(\d+)\.tif", "axis_order": axis_order}
        reader = (WholeBrainReader(tmp_path, mode="slice", **options) if unified
                  else WBTReader(tmp_path, **options))
        options = {"mode": kind}
        start_z = 2

    xyz = axis_order.upper() == "XYZ"

    def coords(zyx):
        return zyx[::-1] if xyz else zyx

    def array(zyx):
        return zyx.transpose(2, 1, 0) if xyz else zyx

    assert reader.axis_order == axis_order.upper()
    assert reader.shape == reader.get_dimensions() == coords(volume.shape)
    assert reader.get_bbox() == ((0, 0, start_z), (9, 7, 5))
    # Unequal extents detect accidental axis swaps even when cubes are crossed.
    actual = reader.read_region(coords((2, 2, 3)), coords((5, 6, 9)), num_workers=2, **options)
    np.testing.assert_array_equal(actual, array(volume[2:5, 2:6, 3:9]))
    assert actual.shape == coords((3, 4, 6))

    expected = np.zeros((4, 5, 6), dtype=volume.dtype)
    expected[:2, :2, :2] = volume[3:, 5:, 7:]
    actual = reader.read_region(coords((3, 5, 7)), coords((7, 10, 13)),
                                padding="right", num_workers=1, **options)
    np.testing.assert_array_equal(actual, array(expected))
    actual = reader.read_region(coords((8, 9, 10)), coords((10, 12, 14)),
                                padding="right", num_workers=1, **options)
    assert actual.shape == coords((2, 3, 4))
    assert not actual.any()
    with pytest.raises(ValueError, match="Invalid X"):
        reader.read_region(coords((2, 2, 8)), coords((3, 3, 10)), **options)
    if start_z:
        with pytest.raises(ValueError, match="Invalid Z"):
            reader.read_region(coords((1, 0, 0)), coords((3, 2, 2)), **options)


@pytest.mark.parametrize("axis_order", ["", "YXZ", "ZY", None, 123])
def test_invalid_axis_order(tmp_path, axis_order):
    # Rejected before any attempt to access image files.
    for factory in (
        lambda: FNTReader(tmp_path, axis_order=axis_order),
        lambda: WBTReader(tmp_path, r"(\d+)", axis_order=axis_order),
        lambda: WholeBrainReader(tmp_path, mode="fnt", axis_order=axis_order),
        lambda: WholeBrainReader(tmp_path, mode="slice", slice_name_pattern=r"(\d+)", axis_order=axis_order),
    ):
        with pytest.raises(ValueError, match="axis_order"):
            factory()
