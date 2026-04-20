# swclib

A Python library for processing, converting, and evaluating neuron morphology data in [SWC format](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Docs/swcFormat.html).

SWC is a standard file format in computational neuroscience for representing 3D reconstructions of neuronal structures (soma, axons, dendrites) as a tree of labeled points.

## Features

- **Data structures** — tree representation of SWC files with efficient spatial queries
- **Format conversion** — bidirectional conversion between SWC and 3D binary mask images
- **Morphology metrics** — spatial distance (SSD), length, keypoint (branch/leaf), and fiber-level metrics for comparing reconstructions
- **Whole-brain I/O** — parallel reader for large tiled TIFF image stacks

## Installation

```bash
pip install swclib
```

Or install from source:

```bash
git clone https://github.com/yang-ze-kang/swclib.git
cd swclib
pip install -e .
```

**Dependencies:** numpy, scipy, networkx, anytree, miniball, rtree, rasterio, tifffile, scikit-image, opencv-python, tqdm

## Quick Start

```python
from swclib.data.swc import Swc

# Load an SWC file
swc = Swc("neuron.swc")
print(f"Total length: {swc.length:.2f}")

# Scale to physical units (e.g., voxel size = 0.5 µm in XY, 0.35 µm in Z)
swc.rescale((0.5, 0.5, 0.35))

# Resample to uniform node spacing
swc.resample(min_distance=2.0)

# Save
swc.save_to_swc("neuron_resampled.swc")
```

## Module Overview

```
swclib/
├── data/         # Core SWC data structures
├── image/        # SWC ↔ binary mask conversion
├── metrics/      # Reconstruction quality metrics
├── whole_brain/  # Large-scale tiled TIFF I/O
├── geometry/     # 3D geometric primitives
└── utils/        # Shared helpers
```

## Usage

### Loading SWC Files

**`Swc`** — lightweight dict-based representation, good for file I/O and simple operations:

```python
from swclib.data.swc import Swc

swc = Swc("neuron.swc")

# Coordinate operations
swc.rescale((0.5, 0.5, 0.35))       # apply voxel size
swc.add_offset((100, 100, 50))       # translate
swc.resample(min_distance=2.0)       # uniform resampling
swc.remove_duplicate_nodes()

# Properties
coords = swc.get_coords()            # ndarray (N, 3)
roots  = swc.get_roots()             # root nodes (parent = -1)
length = swc.length                  # total path length (float)

# Merge multiple SWCs
from swclib.data.swc import merge_swcs
merged = merge_swcs([swc1, swc2], offsets=[(0,0,0), (500,0,0)])

swc.save_to_swc("output.swc")
```

**`SwcForest`** — object-oriented tree representation, handles both single-root and multi-root SWC files:

```python
from swclib.data.swc_forest import SwcForest

forest = SwcForest("neuron.swc")

# Tree structure
print(forest.size())                     # node count
branch_nodes = forest.get_branch_nodes()
leaf_nodes   = forest.get_leaf_nodes()

# Spatial queries
nearest = forest.get_nearest_node([x, y, z])
nearest_k = forest.get_nearest_node([x, y, z], topk=5)

# Fiber extraction (root-to-leaf paths)
fibers = forest.get_fibers()
fibers_from_soma = forest.get_fibers(only_from_soma=True)
fibers_long = forest.get_fibers(min_length=20.0)

# Soma detection
somas = forest.get_somas()

forest.save_to_file("output.swc")
```

### SWC ↔ 3D Mask Conversion

**SWC → mask:**

```python
from swclib.image.swc2mask import Swc2Mask

converter = Swc2Mask(
    shape=(300, 300, 300),  # output volume (Z, Y, X)
    scale=(1, 1, 1),        # voxel size
    radius=1,
    method="line"           # or "sphere_cone"
)
mask = converter.run("neuron.swc", out_file="mask.tif")
```

**Mask → SWC:**

```python
from swclib.image.mask2swc import Mask2Swc
import tifffile

mask = tifffile.imread("mask.tif")

converter = Mask2Swc(
    voxel_size=(1.0, 1.0, 1.0),
    thres_fiber_min_len=20.0,
    thres_branch_min_len=15.0,
    node_sample_distance=2.0,
)
converter.run(mask, "output.swc", soma_path="soma.txt")
```

### Reconstruction Metrics

All metrics share the same interface: `metric.run(gold_swc, pred_swc)` returns a dict.

**Spatial distance (SSD):**

```python
from swclib.metrics.ssd_metric import SSDMetric

metric = SSDMetric(min_distance=2.0, scale=(1, 1, 1))
result = metric.run("gold.swc", "pred.swc")
# result keys: sd, sd_gt2pred, sd_pred2gt, ssd, ssd_gt2pred, ssd_pred2gt, ssd_percent
print(result["sd"])
```

**Length metric** — edge-level precision/recall:

```python
from swclib.metrics.length_metric import LengthMetric

metric = LengthMetric(radius_threshold=5.0, scale=(1, 1, 1))
result = metric.run("gold.swc", "pred.swc")
# result keys: precision, recall, f1_score, TP, FP, FN, num_gt, num_pred
print(result["f1_score"])
```

**Keypoint metric** — branch/leaf point detection:

```python
from swclib.metrics.keypoint_metric import KeypointMetric

metric = KeypointMetric(
    keypoint_types=["branch", "leaf"],
    threshold_dis=5.0,
    scale=(1, 1, 1),
    use_category=True,     # separate branch vs leaf stats
)
result = metric.run("gold.swc", "pred.swc")
print(result["branch"]["f1"])
print(result["leaf"]["f1"])
```

**Fiber metric** — path-level (root-to-leaf) matching, most comprehensive:

```python
from swclib.metrics.fiber_metric import FiberMetric

metric = FiberMetric(
    iou_threshold=0.8,
    dist_threshold=5.0,
    scale=(1, 1, 1),
    only_from_soma=True,   # only fibers starting at soma (type=1)
    use_category=True,     # separate axon vs dendrite stats
    min_fiber_length=20.0,
)
result = metric.run("gold.swc", "pred.swc")
# result keys: precision, recall, f1_score, TP, FP, FN,
#              iou_matched, iou_all, ious, matches, FN_fiber_ids
print(result["f1_score"])
```

**Batch evaluation with MetricManager:**

```python
from swclib.metrics.manager import MetricManager

manager = MetricManager(
    metric_names=["ssd", "length", "keypoints", "fiber"],
    collect_method="micro",   # aggregate TP/FP/FN across all pairs
    scale=(1, 1, 1),
)

for gold, pred in zip(gold_files, pred_files):
    manager.add_data(gold, pred)

summary = manager.collect(save_path="results.json")
```

### Whole-Brain Image Reader

Efficient parallel reader for large tiled TIFF stacks:

```python
from swclib.whole_brain.tifreader import WBTReader

reader = WBTReader(
    slice_dir="/data/brain/slices/",
    slice_name_pattern=r"slice_(\d+)\.tif",
)

depth, height, width = reader.get_dimensions()

# Read a 3D sub-region (start and end are (x, y, z) tuples)
volume = reader.read_region(
    start=(1000, 1000, 100),
    end=(1500, 1500, 200),
    mode="tiff",
    num_workers=16,
)
```

## SWC Format Reference

Each line in an SWC file represents one node:

```
id  type  x  y  z  radius  parent_id
```

Node types:

| Value | Meaning |
|-------|---------|
| 1 | Soma |
| 2 | Axon |
| 3 | Basal dendrite |
| 4 | Apical dendrite |
| 5 | Other |

Root nodes have `parent_id = -1`.

## License

[MIT License](LICENSE)