# Whole-Brain Image I/O

Whole-brain imaging datasets are typically stored as thousands of 2D TIFF slices rather than a single 3D volume. Loading the entire stack into memory is impractical. `WBTReader` addresses this by reading only the requested sub-region in parallel.

---

## `WBTReader`

`WBTReader` reads a directory of 2D TIFF slices and exposes a 3D volume interface. Reads are parallelized with a thread or process pool.

### Setup

```python
from swclib.whole_brain.tifreader import WBTReader

reader = WBTReader(
    slice_dir="/data/brain/xy_slices/",
    slice_name_pattern=r"slice_(\d+)\.tif",   # regex with one capture group = Z index
)
```

`slice_name_pattern` must contain exactly one capture group matching the Z index of each slice file.

### Inspect volume dimensions

```python
depth, height, width = reader.get_dimensions()
print(f"Volume: {depth} x {height} x {width}")

(x1, y1, z1), (x2, y2, z2) = reader.get_bbox()
```

### Read a 3D sub-region

```python
volume = reader.read_region(
    start=(x1, y1, z1),        # inclusive start corner (x, y, z)
    end=(x2, y2, z2),          # exclusive end corner (x, y, z)
    mode="tiff",               # backend: "tiff" or "raster"
    num_workers=16,            # parallel worker count
    parallel_backend="thread", # "thread" or "process"
    padding=None,              # optional (px, py, pz) padding
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

nodes = SwcReader.read("whole_brain_neuron.swc")
# returns numpy array of shape (N, 7): [id, type, x, y, z, radius, parent]
```