# API Reference — Image

## `swclib.image.swc2mask.Swc2Mask`

Converts an SWC neuron to a 3D binary mask volume.

### Constructor

```python
Swc2Mask(shape=(300, 300, 300), scale=(1, 1, 1), radius=1, method='line')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `tuple[int, int, int]` | `(300,300,300)` | Output volume shape `(Z, Y, X)` |
| `scale` | `tuple[float, float, float]` | `(1,1,1)` | Voxel size `(sx, sy, sz)` |
| `radius` | `float` | `1` | Neurite rendering radius |
| `method` | `str` | `'line'` | Rendering method: `'line'` or `'sphere_cone'` |

### Methods

#### `run(swc, out_file=None) → ndarray`

| Parameter | Type | Description |
|-----------|------|-------------|
| `swc` | `str` or `Swc` | SWC file path or object |
| `out_file` | `str` or `None` | If given, saves result as TIFF |

Returns: `ndarray` of shape `(Z, Y, X)`, dtype `uint8`.

---

### Module-level functions

#### `swc_to_mask_line(swc, shape, scale, max_radius) → ndarray`

Distance-transform–based line rendering.

| Parameter | Type | Description |
|-----------|------|-------------|
| `swc` | `Swc` | Source SWC object |
| `shape` | `tuple` | Output shape `(Z, Y, X)` |
| `scale` | `tuple` | Voxel size |
| `max_radius` | `float` | Cutoff radius |

---

#### `swc_to_mask_sphere_cone(swc, shape, foreground_value, r_scale) → ndarray`

Sphere + truncated cone rendering.

| Parameter | Type | Description |
|-----------|------|-------------|
| `swc` | `Swc` | Source SWC object |
| `shape` | `tuple` | Output shape `(Z, Y, X)` |
| `foreground_value` | `int` | Voxel value for foreground |
| `r_scale` | `float` | Radius scale factor |

---

## `swclib.image.mask2swc.Mask2Swc`

Converts a 3D binary mask to an SWC reconstruction.

### Constructor

```python
Mask2Swc(
    voxel_size=(1.0, 1.0, 1.0/0.35),
    thres_soma_degree=4,
    thres_segment_merge_short_dis=3.0,
    thres_segment_merge_long_dis=20.0,
    thres_segment_merge_angle=30.0,
    connectivity=26,
    thres_fiber_min_len=20.0,
    thres_branch_min_len=15.0,
    smooth_window_size=7,
    smooth_ploy=2,
    node_sample_distance=2.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voxel_size` | `tuple` | `(1,1,1/0.35)` | Physical voxel size `(sx, sy, sz)` in µm |
| `thres_soma_degree` | `int` | `4` | Degree threshold used by soma handling |
| `thres_segment_merge_short_dis` | `float` | `3.0` | Short endpoint distance threshold for merging |
| `thres_segment_merge_long_dis` | `float` | `20.0` | Long endpoint distance threshold for merging |
| `thres_segment_merge_angle` | `float` | `30.0` | Max angle (degrees) for fiber merging |
| `connectivity` | `int` | `26` | Neighbor connectivity for skeleton: `6` or `26` |
| `thres_fiber_min_len` | `float` | `20.0` | Min length (µm) to keep an isolated fiber |
| `thres_branch_min_len` | `float` | `15.0` | Min length (µm) to keep a side branch |
| `smooth_window_size` | `int` | `7` | Savitzky-Golay smoothing window |
| `smooth_ploy` | `int` | `2` | Savitzky-Golay polynomial order. The parameter name follows the current source spelling |
| `node_sample_distance` | `float` | `2.0` | Output node spacing after resampling |

### Methods

#### `run(mask, swc_path, soma_path=None, radius=0.1, verbos=False) → str`

Runs the full conversion pipeline and writes the SWC file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `ndarray` | 3D binary mask, shape `(Z, Y, X)` |
| `swc_path` | `str` | Output SWC file path |
| `soma_path` | `str` or `None` | Optional soma annotation file |
| `radius` | `float` | Default node radius in output SWC |
| `verbos` | `bool` | Print pipeline statistics. The parameter name follows the current source spelling |

Returns: Output SWC file path (str).

---

#### `build_graph_from_skeleton(skel, connectivity, voxel_size) → nx.Graph`

Builds a NetworkX graph from a skeletonized binary mask.

#### `remove_isolated_nodes(G) → (nx.Graph, list)`

Removes degree-0 nodes.

#### `merge_fibers(G, voxel_size, ...) → (nx.Graph, int)`

Merges nearby near-collinear fiber endpoints. Returns `(graph, num_merged)`.

#### `remove_short_fibers(G, voxel_size, thres_min_len) → (nx.Graph, int)`

Removes fibers shorter than the threshold.

#### `remove_short_branchs(G, voxel_size, thres_min_len) → (nx.Graph, int)`

Prunes side branches shorter than the threshold.

#### `refine_fibers(G, smooth_window_size, smooth_ploy, node_sample_distance, voxel_size) → nx.Graph`

Applies Savitzky-Golay smoothing and uniform resampling to all fibers.
