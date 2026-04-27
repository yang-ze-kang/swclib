# API Reference — Whole Brain

## `swclib.whole_brain.tifreader.WBTReader`

Parallel reader for large 3D image stacks stored as 2D TIFF slices.

### Constructor

```python
WBTReader(slice_dir, slice_name_pattern)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `slice_dir` | `str` | Directory containing 2D TIFF slice files |
| `slice_name_pattern` | `str` | Regex with one capture group matching the Z index |

**Example pattern:** `r"slice_(\d+)\.tif"` matches `slice_0042.tif` with Z index `42`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `slice_dir` | `str` | Slice directory |
| `slice_name_pattern` | `str` | Regex pattern |
| `start_z` | `int` | Minimum Z index found |
| `end_z` | `int` | Maximum Z index found |
| `height` | `int` | Slice height (Y dimension) |
| `width` | `int` | Slice width (X dimension) |
| `depth` | `int` | Number of slices (Z dimension) |
| `dtype` | `dtype` | NumPy dtype of slice data |

### Methods

#### `get_dimensions() → tuple`
Returns `(depth, height, width)`.

#### `get_bbox() → tuple`
Returns `((x1, y1, z1), (x2, y2, z2))` bounding box.

#### `read_region(start, end, mode='tiff', num_workers=32, parallel_backend='thread', padding=None) → ndarray`

Read a 3D sub-region from the slice stack.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `tuple[int, int, int]` | — | Inclusive start corner `(x, y, z)` |
| `end` | `tuple[int, int, int]` | — | Exclusive end corner `(x, y, z)` |
| `mode` | `str` | `'tiff'` | Backend: `'tiff'` (tifffile) or `'raster'` (rasterio) |
| `num_workers` | `int` | `32` | Number of parallel workers |
| `parallel_backend` | `str` | `'thread'` | `'thread'` or `'process'` |
| `padding` | `tuple` or `None` | `None` | Optional padding `(px, py, pz)` added to each side |

Returns: `ndarray` of shape `(z2-z1, y2-y1, x2-x1)`.

---

## `swclib.whole_brain.swc_reader.SwcReader`

Minimal SWC reader for large whole-brain coordinate files.

### Static method

#### `SwcReader.read(path) → ndarray`

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | SWC file path |

Returns: `ndarray` of shape `(N, 7)` with columns `[id, type, x, y, z, radius, parent]`.