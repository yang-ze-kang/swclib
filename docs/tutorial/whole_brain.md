# Whole-Brain Image I/O

Whole-brain datasets can be stored as 2D TIFF slices or FNT NRRD cubes.
`WholeBrainReader` provides one interface for both layouts and reads the
requested sub-region in parallel.

## Unified reader

```python
from swclib.whole_brain import WholeBrainReader

reader = WholeBrainReader(
    "/mnt/mcgovern/GZClab_Facility/MADM-Cux2-T31/fnt/cubes",
    mode="fnt",
    channel="ch00",
)
print(reader.get_dimensions())  # (11600, 25698, 33509), ZYX
print(reader.get_bbox())        # ((0, 0, 0), (33509, 25698, 11600)), XYZ
print(reader.spacing)           # (0.35, 0.35, 1.0), XYZ

block = reader.read_region(
    start=(5700, 12000, 16000),  # ZYX, inclusive
    end=(5800, 12300, 16300),    # ZYX, exclusive
    num_workers=8,
)
assert block.shape == (100, 300, 300)

# For a TIFF slice dataset, replace only the reader setup:
reader = WholeBrainReader(
    "/data/brain/slices",
    mode="slice",
    slice_name_pattern=r"slice_(\d+)\.tif",
)
block = reader.read_region(start=(100, 200, 300), end=(120, 240, 360))
```

### Choosing axis order

All three readers accept `axis_order="ZYX"` (the default) or `axis_order="XYZ"`
at construction. Values are case-insensitive, including `"zyx"` and `"xYz"`.
The selected order applies to `read_region` input coordinates, returned NumPy
array axes, `shape` and `get_dimensions()`.

```python
reader = WholeBrainReader(
    "/mnt/mcgovern/GZClab_Facility/MADM-Cux2-T31/fnt/cubes",
    mode="fnt",
    channel="ch00",
    axis_order="xyz",
)
print(reader.get_dimensions())  # (33509, 25698, 11600), XYZ
image = reader.read_region(
    start=(16000, 12000, 5700),  # XYZ
    end=(16300, 12200, 5800),
    num_workers=8,
)
assert image.shape == (300, 200, 100)  # X, Y, Z

# Save as a conventional ZYX TIFF stack for Fiji/ImageJ:
import tifffile
tifffile.imwrite(
    "region.tif",
    image.transpose(2, 1, 0),
    photometric="minisblack",
    metadata={"axes": "ZYX"},
)
```

`get_bbox()` always returns XYZ corners, including in the default ZYX mode.
Catalog `size`, `cubesize` and physical metadata stay in XYZ; `cube_shape`
stays in ZYX. In XYZ mode, returned arrays are transposed views, avoiding an
extra volume copy.

## FNT cubes

You can also use `from swclib.whole_brain import FNTReader` and construct
`FNTReader("/path/to/cubes", channel="ch00")` directly. The input directory
contains the `catalog` and channel directories:

```text
cubes/
  catalog
  ch00/
    z00000000/
      y00000000.x00000000.nrrd
      y00000000.x00000256.nrrd
    z00000096/
      ...
```

The catalog describes full image size, cube size and filename patterns.
Filenames encode cube **voxel origins**, not cube indices. The reader resolves
only the cubes intersecting the region and handles smaller boundary cubes.
It does not scan the full directory tree or load the downsampled `ch00ds.nrrd`.
Missing or malformed cubes raise errors so that incomplete data is visible.

`origin`, `direction` and `spacing` expose physical metadata from the catalog;
`read_region` and `get_bbox` still use voxel coordinates. NRRD decoding requires
`pynrrd`, included in the package dependencies.

Both formats support `padding="right"` to fill beyond the positive volume
boundaries with zeros. Without padding, out-of-bounds reads raise `ValueError`.
The returned shape always equals `end - start`. All upper bounds are exclusive;
in particular, `WBTReader.get_bbox()` now includes the last slice in this convention.

---

## `WBTReader`

`WBTReader` recursively reads a directory of 2D TIFF slices and exposes a 3D volume interface. Reads are parallelized with a thread or process pool.

### Setup

```python
from swclib.whole_brain.tifreader import WBTReader

reader = WBTReader(
    slice_dir="/data/brain/xy_slices/",
    slice_name_pattern=r"slice_(\d+)\.tif",   # regex on file basename; group 1 = Z index
    slice_ext=".tif",                         # optional; default is ".tif"
)
```

`slice_dir` is searched recursively, including all subdirectories. `slice_ext`
filters files by extension before sorting; pass values such as `.tiff` or
`tiff` to read other TIFF extensions. `slice_name_pattern` must contain exactly
one capture group matching the Z index of each slice file name.

### Inspect volume dimensions

```python
depth, height, width = reader.get_dimensions()
print(f"Volume: {depth} x {height} x {width}")

(x1, y1, z1), (x2, y2, z2) = reader.get_bbox()
```

### Read a 3D sub-region

```python
volume = reader.read_region(
    start=(z1, y1, x1),        # inclusive start corner (z, y, x)
    end=(z2, y2, x2),          # exclusive end corner (z, y, x)
    mode="tiff",               # backend: "tiff" or "raster"
    num_workers=16,            # parallel worker count
    parallel_backend="thread", # "thread" or "process"
    padding=None,              # or "right" for zero-fill beyond positive bounds
)
# returns numpy array of shape (z2-z1, y2-y1, x2-x1)
```

### Read modes

| Mode | Library | Best for |
|------|---------|---------|
| `"tiff"` | tifffile | Standard TIFF files |
| `"raster"` | rasterio | GeoTIFF or large tiled TIFF with spatial metadata |

### Parallel backends

| Backend | Use when |
|---------|---------|
| `"thread"` | I/O-bound reads (most common case) |
| `"process"` | CPU-bound decoding (e.g., heavy compression) |

### Typical workflow

```python
from swclib.whole_brain.tifreader import WBTReader
from swclib.image.mask2swc import Mask2Swc

reader = WBTReader(
    slice_dir="/data/brain/slices/",
    slice_name_pattern=r"img_z(\d{4})\.tif",
    slice_ext=".tif",
)

# Read a local region around a neuron of interest
block = reader.read_region(
    start=(2000, 3000, 100),
    end=(2512, 3512, 200),
    num_workers=32,
)

# Reconstruct SWC from local block
converter = Mask2Swc(voxel_size=(0.5, 0.5, 1.0))
converter.run(block, "local_neuron.swc")
```

---

## `SwcReader` — SWC file loading for whole-brain coordinates

For whole-brain SWC files with large coordinate values, use `SwcReader` from the same module:

```python
from swclib.whole_brain.swc_reader import SwcReader

reader = SwcReader("whole_brain_neuron.swc")
swc = reader.read_region(
    start=(1000, 2000, 300),
    end=(1500, 2500, 400),
    out_path="local_neuron.swc",
)
# returns a swclib.data.swc.Swc object in local coordinates
```
