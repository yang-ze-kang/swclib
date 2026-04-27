# Reconstruction Metrics

swclib provides four complementary metrics for comparing a predicted neuron reconstruction against a gold standard. All metrics share the same interface:

```python
result = metric.run(gold_swc_path, pred_swc_path)
```

`gold_swc_path` and `pred_swc_path` are paths to SWC files. The return value is always a plain dict.

---

## Overview

| Metric | Class | What it measures |
|--------|-------|-----------------|
| SSD | `SSDMetric` | Mean spatial distance between node sets |
| Length | `LengthMetric` | Edge-level path coverage (precision/recall) |
| Keypoint | `KeypointMetric` | Detection of branch and leaf points |
| Fiber | `FiberMetric` | Path-level (root-to-leaf) matching |

---

## SSD Metric

The **Spatial Distance (SSD) metric** resamples both trees to uniform node spacing, then computes the mean distance between each node and its nearest neighbor in the other tree.

### Usage

```python
from swclib.metrics.ssd_metric import SSDMetric

metric = SSDMetric(
    min_distance=2.0,       # resampling step (µm)
    scale=(1, 1, 1),        # coordinate scaling
)
result = metric.run("gold.swc", "pred.swc")
```

### Output

| Key | Description |
|-----|-------------|
| `sd` | Mean bidirectional spatial distance |
| `sd_gt2pred` | Mean distance from gold nodes to nearest predicted node |
| `sd_pred2gt` | Mean distance from predicted nodes to nearest gold node |
| `ssd` | Mean of distances exceeding `min_distance` |
| `ssd_gt2pred` | SSD in gold→pred direction |
| `ssd_pred2gt` | SSD in pred→gold direction |
| `ssd_percent` | Fraction of nodes with distance > `min_distance` |

---

## Length Metric

The **Length metric** compares reconstructions at the *edge* level. Each edge (segment between two consecutive nodes) in the gold standard is matched against edges in the predicted tree within a spatial radius.

### Usage

```python
from swclib.metrics.length_metric import LengthMetric

metric = LengthMetric(
    radius_threshold=5.0,   # matching radius (µm)
    length_threshold=2.0,   # minimum edge length to evaluate (µm)
    scale=(1, 1, 1),
    resample_step=1.0,      # resampling step for edge comparison
)
result = metric.run("gold.swc", "pred.swc")
```

### Output

| Key | Description |
|-----|-------------|
| `precision` | Fraction of predicted length that matches gold |
| `recall` | Fraction of gold length that is covered by prediction |
| `f1_score` | Harmonic mean of precision and recall |
| `TP` | Matched edge length |
| `FP` | Unmatched predicted edge length |
| `FN` | Unmatched gold edge length |
| `num_gt` | Total gold edge count |
| `num_pred` | Total predicted edge count |

---

## Keypoint Metric

The **Keypoint metric** evaluates detection of topologically significant points: branch nodes (degree ≥ 3) and leaf nodes (degree 1).

### Usage

```python
from swclib.metrics.keypoint_metric import KeypointMetric

metric = KeypointMetric(
    keypoint_types=["branch", "leaf"],
    threshold_dis=5.0,      # matching distance threshold (µm)
    scale=(1, 1, 1),
    use_category=True,      # report branch and leaf stats separately
)
result = metric.run("gold.swc", "pred.swc")
```

### Output (combined mode, `use_category=False`)

| Key | Description |
|-----|-------------|
| `precision` | Keypoint detection precision |
| `recall` | Keypoint detection recall |
| `f1` | F1 score |
| `TP` / `FP` / `FN` | Matched / extra / missed keypoints |

### Output (per-category mode, `use_category=True`)

The dict contains nested sub-dicts for each keypoint type:

```python
result["branch"]["precision"]
result["branch"]["recall"]
result["branch"]["f1"]

result["leaf"]["precision"]
result["leaf"]["recall"]
result["leaf"]["f1"]
```

---

## Fiber Metric

The **Fiber metric** is the most comprehensive metric. It extracts all root-to-leaf paths (fibers) from both trees and matches them using Intersection-over-Union (IoU).

A fiber is counted as a **True Positive (TP)** if its IoU with a matched gold fiber exceeds `iou_threshold`.

### Usage

```python
from swclib.metrics.fiber_metric import FiberMetric

metric = FiberMetric(
    iou_threshold=0.8,          # minimum IoU for a match to count as TP
    dist_threshold=5.0,         # spatial tolerance (µm) for IoU calculation
    dist_sample=1.0,            # resampling step for IoU calculation
    scale=(1, 1, 1),
    only_from_soma=True,        # only evaluate fibers starting at type=1 nodes
    min_fiber_length=20.0,      # ignore fibers shorter than this (µm)
    align_roots=True,           # align tree roots before comparison
    align_roots_threshold=20.0, # max distance to match roots
    with_direction=False,       # consider fiber direction
    use_category=True,          # separate axon (type=2) and dendrite (type=3/4) stats
)
result = metric.run("gold.swc", "pred.swc")
```

### Output

| Key | Description |
|-----|-------------|
| `precision` | Fraction of predicted fibers that match a gold fiber |
| `recall` | Fraction of gold fibers that are matched by a prediction |
| `f1_score` | Harmonic mean |
| `TP` / `FP` / `FN` | Fiber-level counts |
| `num_gt` | Number of gold fibers |
| `num_pred` | Number of predicted fibers |
| `iou_matched` | Mean IoU among matched fiber pairs |
| `iou_all` | Mean IoU across all gold fibers (unmatched = 0) |
| `ious` | Per-fiber IoU values |
| `matches` | List of matched (gold_fiber, pred_fiber) pairs |
| `FN_fiber_ids` | Node IDs of unmatched gold fibers |

With `use_category=True`, per-category stats appear under `result["axon"]` and `result["dendrite"]`.

With `return_fibers=True`, the actual `SwcFiber` objects are returned under `result["gold_fibers"]` and `result["pred_fibers"]`.

### Example: inspect unmatched fibers

```python
result = metric.run("gold.swc", "pred.swc", return_fibers=True)

for i, fiber in enumerate(result["gold_fibers"]):
    iou = result["ious"][i]
    if iou < metric.iou_threshold:
        print(f"Unmatched gold fiber, length={fiber.length:.1f}, IoU={iou:.2f}")
```

---

## Batch evaluation with MetricManager

`MetricManager` runs multiple metrics on a dataset of reconstruction pairs and aggregates results.

```python
from swclib.metrics.manager import MetricManager

manager = MetricManager(
    metric_names=["ssd", "length", "keypoints", "fiber"],
    collect_method="micro",  # aggregate TP/FP/FN across all pairs ("micro")
    scale=(1, 1, 1),
)

# Optionally override per-metric parameters
manager.update_metric_params({
    "fiber": {"iou_threshold": 0.8, "dist_threshold": 5.0},
    "keypoints": {"threshold_dis": 5.0},
})

# Add reconstruction pairs
for gold_path, pred_path in zip(gold_files, pred_files):
    manager.add_data(gold_path, pred_path)

# Aggregate and save
summary = manager.collect(save_path="results.json")
print(summary)
```

### Collect methods

| Method | Description |
|--------|-------------|
| `"micro"` | Sum all TP/FP/FN across the dataset, then compute final P/R/F1 |
| `"macro"` | Compute P/R/F1 per pair, then average |

Micro averaging is typically preferred for imbalanced datasets (neurons with very different fiber counts).

---

## Choosing the right metric

| Scenario | Recommended metric |
|----------|--------------------|
| Quick sanity check | SSD |
| Evaluate path tracing completeness | Length |
| Evaluate topology accuracy | Keypoint |
| Comprehensive morphology evaluation | Fiber |
| Full benchmark pipeline | MetricManager with all four |