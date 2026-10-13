# API Reference — Whole Brain

## `swclib.whole_brain.WholeBrainReader`

Unified entry point for FNT NRRD cubes and 2D TIFF slices:

```python
WholeBrainReader(data_dir, mode="slice", axis_order="ZYX", **kwargs)
```

Use `mode="fnt"` with optional `channel="ch00"`, or `mode="slice"` with
`slice_name_pattern` and optional `slice_ext=".tif"`. Constructor options are
forwarded to the corresponding reader. `get_dimensions()`, `get_bbox()` and
`read_region(start, end, **kwargs)` share the conventions below. Backend
attributes, including `shape`, `dtype`, `width`, `height` and `depth`, are exposed
on the unified reader.

* `axis_order` accepts **ZYX** (default) or **XYZ**, case-insensitively.
  Region `start`/`end` coordinates and returned array axes follow this order.
* `get_dimensions()` and `shape` return `(depth, height, width)` for ZYX or
  `(width, height, depth)` for XYZ. `axis_order` exposes the normalized uppercase value.
* `get_bbox()` returns **XYZ** voxel corners with an **exclusive upper corner**.
* `padding=None` rejects out-of-bounds regions; `padding="right"` zero-fills
  beyond positive boundaries. Negative coordinates are rejected.
* `num_workers=1` reads serially; `parallel_backend="thread"` or `"process"`
  selects the executor. Pending reads are limited to `num_workers`.

The constructor's `mode` selects the data format. The `mode` passed to
`read_region` selects the decoding backend (`"nrrd"` for FNT, `"raster"` or
`"tiff"` for slices); normally omit it to use the backend default.

## `swclib.whole_brain.FNTReader`

```python
FNTReader(fnt_dir, channel="ch00", axis_order="ZYX")
reader.read_region(start, end, mode="nrrd", num_workers=8,
                   parallel_backend="thread", padding=None)
```

`fnt_dir` is the export directory containing `catalog`, not the `ch00` directory
or a `.fnt` tracing file. Channel names are case-insensitive. Only catalog
sections with a `pattern` are cube channels; downsampled `location` sections
such as `CH00DS` are not selected.

The reader parses `pattern`, `size`, `cubesize`, `origin` and `direction` from
the catalog and reads one cube header to determine the dtype. It constructs
filenames directly from the requested voxel coordinates, with no recursive
file scan. NRRD decoding uses `pynrrd`, including gzip and raw encodings and
byte-order conversion. Gzip requires decoding each intersecting cube in full.
Only intersecting cubes are read, then cropped and assembled internally in ZYX
order. In XYZ mode, the returned array is a transposed view without an extra copy.
Edge cubes may be truncated to the volume boundary or padded to full cube size.
Missing files, malformed cubes and inconsistent dtypes raise errors.
The cube at voxel origin is required for initialization.

| Attribute | Description |
|-----------|-------------|
| `channel`, `channels` | Selected channel and available cube channels |
| `size` | Full volume dimensions in XYZ order |
| `cubesize`, `cube_shape` | Nominal cube dimensions in XYZ and ZYX order |
| `origin` | Physical XYZ origin from the catalog, default `(0, 0, 0)` |
| `direction` | 3×3 matrix; each row is the physical vector of an X, Y or Z voxel step |
| `spacing` | Length of each direction vector, in XYZ order |
| `dtype` | NumPy dtype of the image data |

Metadata does not resample or transform reads: region coordinates and the
bounding box always refer to the voxel grid.
`axis_order` does not change `get_bbox()`, `size`, `cubesize`, `cube_shape` or
physical metadata; their fixed conventions are listed above.

---

## `swclib.whole_brain.tifreader.WBTReader`

Parallel reader for large 3D image stacks stored as 2D TIFF slices. The reader
recursively scans the slice directory and all subdirectories.

### Constructor

```python
WBTReader(slice_dir, slice_name_pattern, slice_ext=".tif", axis_order="ZYX")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `slice_dir` | `str` | Directory containing 2D TIFF slice files; searched recursively |
| `slice_name_pattern` | `str` | Regex with one capture group matching the Z index in each file basename |
| `slice_ext` | `str` | File extension filter. Defaults to `.tif`; values without a leading dot are accepted |
| `axis_order` | `str` | `"ZYX"` (default) or `"XYZ"`, case-insensitive; controls region coordinates, array axes and dimensions |

**Example pattern:** `r"slice_(\d+)\.tif"` matches `slice_0042.tif` with Z index `42`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `slice_dir` | `str` | Slice directory |
| `start_z` | `int` | Minimum Z index found |
| `end_z` | `int` | Maximum Z index found |
| `height` | `int` | Slice height (Y dimension) |
| `width` | `int` | Slice width (X dimension) |
| `depth` | `int` | Exclusive maximum Z index (`end_z + 1`); includes leading unavailable indices when `start_z > 0` |
| `dtype` | `dtype` | NumPy dtype of slice data |

### Methods

#### `get_dimensions() → tuple`
Returns `(depth, height, width)` for ZYX or `(width, height, depth)` for XYZ.

#### `get_bbox() → tuple`
Returns `((0, 0, start_z), (width, height, depth))` with an exclusive upper
corner. The Z upper bound is now `end_z + 1`, correcting the previous omission
of the last slice when using the bbox to read a region.

#### `read_region(start, end, mode='raster', num_workers=32, parallel_backend='thread', padding=None) → ndarray`

Read a 3D sub-region from the slice stack.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `tuple[int, int, int]` | — | Inclusive start corner in `axis_order` |
| `end` | `tuple[int, int, int]` | — | Exclusive end corner in `axis_order` |
| `mode` | `str` | `'raster'` | Backend: `'raster'` (rasterio) or `'tiff'` (tifffile) |
| `num_workers` | `int` | `32` | Number of parallel workers |
| `parallel_backend` | `str` | `'thread'` | `'thread'` or `'process'` |
| `padding` | `str` or `None` | `None` | `"right"` zero-fills beyond positive volume boundaries; `None` rejects out-of-bounds reads |

Returns: `ndarray` with axes in `axis_order` and shape equal to `end - start`.

---

## `swclib.whole_brain.swc_reader.SwcReader`

Minimal SWC reader for large whole-brain coordinate files.

### Constructor

```python
SwcReader(path, rescale=(1.0, 1.0, 1.0))
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | SWC file path |
| `rescale` | `tuple[float, float, float]` | Coordinate scale applied while loading `(sx, sy, sz)` |

### Methods

#### `read_region(start, end, out_path=None, out_r=None) → Swc`

Read nodes inside a whole-brain coordinate cube and return a local-coordinate
`Swc` object. Nodes are reindexed to `1..N`; nodes whose parent falls outside
the region become roots.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `tuple[float, float, float]` | — | Inclusive start corner `(x, y, z)` |
| `end` | `tuple[float, float, float]` | — | Exclusive end corner `(x, y, z)` |
| `out_path` | `str` or `None` | `None` | Optional path to save the returned SWC |
| `out_r` | `float` or `None` | `None` | Optional radius override for returned and saved nodes |

Returns: `Swc` object containing the cropped region. Empty regions return an
empty `Swc`.
