# Quick Start

This page walks through the most common use cases in a few lines of code.

!!! note "Coordinate conventions"
    SWC node coordinates are stored as `(x, y, z)`. Image volumes are NumPy
    arrays with shape `(Z, Y, X)`. APIs that read 3D image regions use
    `(z, y, x)` for volume corners unless otherwise noted.

## Load and inspect an SWC file

```python
from swclib.data.swc import Swc

swc = Swc("neuron.swc")

print(f"Nodes:        {len(swc.nodes)}")
print(f"Total length: {swc.length:.2f} µm")
print(f"Bounding box: {swc.bound_box}")
```

## Scale to physical units

SWC files often store coordinates in voxel indices. Use `rescale` to convert to physical units (e.g., micrometers):

```python
# voxel size: 0.5 µm in XY, 0.35 µm in Z
swc.rescale((0.5, 0.5, 0.35))
```

## Resample to uniform node spacing

```python
swc.resample(min_distance=2.0)   # one node every 2 µm
swc.save_to_swc("neuron_resampled.swc")
```

## Crop a local SWC region

```python
local = swc.read_region(
    start=(100, 100, 50),   # (x, y, z), inclusive
    end=(200, 220, 90),     # (x, y, z), exclusive
    out_path="neuron_crop.swc",
)
```

Nodes inside the cube are reindexed. If a kept node's parent lies outside the
cube, that node becomes a root in the returned SWC.

## Convert SWC to 3D binary mask

```python
from swclib.image.swc2mask import Swc2Mask

converter = Swc2Mask(
    shape=(300, 300, 300),  # output volume shape: (Z, Y, X)
    scale=(1, 1, 1),
    radius=1,
    method="line",          # or "sphere_cone"
)
mask = converter.run("neuron.swc", out_file="neuron_mask.tif")
```

## Convert 3D mask back to SWC

```python
from swclib.image.mask2swc import Mask2Swc
import tifffile

mask = tifffile.imread("neuron_mask.tif")
converter = Mask2Swc(
    voxel_size=(1.0, 1.0, 1.0),
    thres_fiber_min_len=20.0,
    thres_branch_min_len=15.0,
    node_sample_distance=2.0,
)
converter.run(mask, "neuron_reconstructed.swc", radius=0.1)
```

## Compare two reconstructions

```python
from swclib.metrics.fiber_metric import FiberMetric

metric = FiberMetric(iou_threshold=0.8, dist_threshold=5.0)
result = metric.run("gold.swc", "predicted.swc")

print(f"Precision: {result['precision']:.3f}")
print(f"Recall:    {result['recall']:.3f}")
print(f"F1 score:  {result['f1']:.3f}")
```

## Batch evaluation

```python
from swclib.metrics.manager import MetricManager

manager = MetricManager(metric_names=["ssd", "length", "keypoints", "fiber"], scale=(1, 1, 1))

manager.update_metric_params({
    "fiber": {"iou_threshold": 0.8, "dist_threshold": 5.0},
    "keypoints": {"threshold_dis": 5.0},
})

pairs = [("gold1.swc", "pred1.swc"), ("gold2.swc", "pred2.swc")]
for gold, pred in pairs:
    manager.add_data(gold, pred)

summary = manager.collect(save_path="results.json")
```

---

Continue reading the [Tutorial](tutorial/data.md) for detailed explanations.
