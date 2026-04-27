# SWC ↔ Mask Conversion

swclib supports bidirectional conversion between SWC neuron structures and 3D binary mask images (volumetric arrays stored as TIFF files).

---

## SWC → 3D Mask (`Swc2Mask`)

`Swc2Mask` renders an SWC neuron into a 3D binary (or distance-transformed) mask volume.

### Basic usage

```python
from swclib.image.swc2mask import Swc2Mask

converter = Swc2Mask(
    shape=(300, 300, 300),  # output volume shape (Z, Y, X)
    scale=(1, 1, 1),        # voxel size (sx, sy, sz)
    radius=1,               # neurite radius in voxels
    method="line",          # rendering method: "line" or "sphere_cone"
)

mask = converter.run("neuron.swc", out_file="neuron_mask.tif")
# mask: numpy array of shape (Z, Y, X), dtype uint8
```

### Rendering methods

| Method | Description |
|--------|-------------|
| `"line"` | Fast distance-transform–based rendering. Suitable for thin structures. |
| `"sphere_cone"` | Renders each node as a sphere and each edge as a truncated cone. More accurate for thick structures. |

### Anisotropic voxels

If your image has different voxel sizes in each axis (common in confocal microscopy), pass the voxel size via `scale`:

```python
# XY resolution: 0.5 µm/vox, Z resolution: 1.0 µm/vox
converter = Swc2Mask(
    shape=(200, 512, 512),
    scale=(0.5, 0.5, 1.0),
    radius=1,
)
```

### Low-level functions

For fine-grained control you can call the rendering functions directly:

```python
from swclib.image.swc2mask import swc_to_mask_line, swc_to_mask_sphere_cone
from swclib.data.swc import Swc

swc = Swc("neuron.swc")

# Line method
mask = swc_to_mask_line(swc, shape=(300,300,300), scale=(1,1,1), max_radius=3)

# Sphere+cone method
mask = swc_to_mask_sphere_cone(swc, shape=(300,300,300), foreground_value=1, r_scale=1.0)
```

---

## 3D Mask → SWC (`Mask2Swc`)

`Mask2Swc` converts a binary 3D mask to an SWC reconstruction through a pipeline of skeletonization, graph building, and post-processing.

### Pipeline overview

```
Binary mask
    │
    ▼  skeletonize (scikit-image)
Skeleton
    │
    ▼  build_graph_from_skeleton
NetworkX Graph
    │
    ├─ remove_isolated_nodes
    ├─ merge_fibers           (merge nearby parallel branches)
    ├─ remove_short_fibers    (prune noise)
    ├─ remove_short_branchs   (prune short side branches)
    └─ refine_fibers          (smooth + resample)
    │
    ▼
SWC File
```

### Basic usage

```python
from swclib.image.mask2swc import Mask2Swc
import tifffile

mask = tifffile.imread("neuron_mask.tif")  # binary uint8 array

converter = Mask2Swc(
    voxel_size=(1.0, 1.0, 1.0),    # physical size of each voxel (µm)
    thres_fiber_min_len=20.0,       # remove isolated fibers shorter than this
    thres_branch_min_len=15.0,      # prune side branches shorter than this
    node_sample_distance=2.0,       # output node spacing
)

converter.run(mask, "output.swc", soma_path="soma.txt", radius=0.1)
```

### With soma annotation

If you have a soma annotation file, pass it via `soma_path` to improve root assignment:

```python
converter.run(mask, "output.swc", soma_path="soma_annotation.txt")
```

The soma file format matches the output of `SwcSoma.save_somas_to_file`.

### Configuration parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `voxel_size` | `(1,1,1)` | Physical voxel dimensions (sx, sy, sz) in µm |
| `connectivity` | `26` | Neighbor connectivity for skeletonization (`6` or `26`) |
| `thres_fiber_min_len` | `20.0` | Minimum length to keep an isolated fiber (µm) |
| `thres_branch_min_len` | `15.0` | Minimum length to keep a side branch (µm) |
| `thres_segment_merge_angle` | — | Max angle (°) between fibers to be merged |
| `thres_segment_merge_dist` | — | Max endpoint distance for fiber merging (µm) |
| `smooth_window_size` | — | Savitzky-Golay smoothing window size |
| `smooth_poly` | — | Savitzky-Golay polynomial order |
| `node_sample_distance` | `2.0` | Output node spacing after resampling |

### Verbose mode

```python
converter.run(mask, "output.swc", verbose=True)
# Prints step-by-step statistics for each pipeline stage
```

---

## Soma mask generation

`SwcSoma` provides utilities to render detected somas to a volumetric mask:

```python
from swclib.data.swc_soma import read_soma_from_file, create_soma_mask

somas = read_soma_from_file("soma_annotation.txt")
soma_mask = create_soma_mask(somas, volume_shape=(300, 300, 300), out_path="soma_mask.tif")
```