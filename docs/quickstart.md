# Quick Start

This page walks through the most common use cases in a few lines of code.

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

## Convert SWC to 3D binary mask

```python
from swclib.image.swc2mask import Swc2Mask

converter = Swc2Mask(shape=(300, 300, 300), scale=(1, 1, 1), radius=1)
mask = converter.run("neuron.swc", out_file="neuron_mask.tif")
```

## Convert 3D mask back to SWC

```python
from swclib.image.mask2swc import Mask2Swc
import tifffile

mask = tifffile.imread("neuron_mask.tif")
converter = Mask2Swc(voxel_size=(1.0, 1.0, 1.0))
converter.run(mask, "neuron_reconstructed.swc")
```

## Compare two reconstructions

```python
from swclib.metrics.fiber_metric import FiberMetric

metric = FiberMetric(iou_threshold=0.8, dist_threshold=5.0)
result = metric.run("gold.swc", "predicted.swc")

print(f"Precision: {result['precision']:.3f}")
print(f"Recall:    {result['recall']:.3f}")
print(f"F1 score:  {result['f1_score']:.3f}")
```

## Batch evaluation

```python
from swclib.metrics.manager import MetricManager

manager = MetricManager(
    metric_names=["ssd", "length", "keypoints", "fiber"],
    scale=(1, 1, 1),
)

pairs = [("gold1.swc", "pred1.swc"), ("gold2.swc", "pred2.swc")]
for gold, pred in pairs:
    manager.add_data(gold, pred)

summary = manager.collect(save_path="results.json")
```

---

Continue reading the [Tutorial](tutorial/data.md) for detailed explanations.