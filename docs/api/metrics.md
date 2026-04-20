# API Reference — Metrics

All metrics share the same base interface:

```python
result = metric.run(gold_swc_path, pred_swc_path)
```

Returns a `dict`.

---

## `swclib.metrics.ssd_metric.SSDMetric`

Spatial distance metric.

### Constructor

```python
SSDMetric(min_distance=2.0, scale=(1, 1, 1))
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_distance` | `float` | `2.0` | Resampling step and SSD threshold (µm) |
| `scale` | `tuple` | `(1,1,1)` | Coordinate scaling `(sx, sy, sz)` |

### `run(gold, pred) → dict`

| Parameter | Type | Description |
|-----------|------|-------------|
| `gold` | `str` | Gold standard SWC path |
| `pred` | `str` | Predicted SWC path |

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `sd` | `float` | Mean bidirectional spatial distance |
| `sd_gt2pred` | `float` | Mean distance gold → pred |
| `sd_pred2gt` | `float` | Mean distance pred → gold |
| `ssd` | `float` | Mean of distances exceeding `min_distance` |
| `ssd_gt2pred` | `float` | SSD in gold→pred direction |
| `ssd_pred2gt` | `float` | SSD in pred→gold direction |
| `ssd_percent` | `float` | Fraction of nodes with distance > `min_distance` |

---

## `swclib.metrics.length_metric.LengthMetric`

Edge-level path coverage metric.

### Constructor

```python
LengthMetric(
    radius_threshold=5.0,
    length_threshold=2.0,
    scale=(1, 1, 1),
    resample_step=1.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius_threshold` | `float` | `5.0` | Spatial tolerance for edge matching (µm) |
| `length_threshold` | `float` | `2.0` | Minimum edge length to evaluate (µm) |
| `scale` | `tuple` | `(1,1,1)` | Coordinate scaling |
| `resample_step` | `float` | `1.0` | Resampling step for edge comparison |

### `run(gold, pred) → dict`

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `precision` | `float` | Predicted length that matches gold |
| `recall` | `float` | Gold length covered by prediction |
| `f1_score` | `float` | Harmonic mean of precision and recall |
| `TP` | `int` | Matched edge count |
| `FP` | `int` | Unmatched predicted edge count |
| `FN` | `int` | Unmatched gold edge count |
| `num_gt` | `int` | Total gold edges |
| `num_pred` | `int` | Total predicted edges |

---

## `swclib.metrics.keypoint_metric.KeypointMetric`

Branch and leaf node detection metric.

### Constructor

```python
KeypointMetric(
    keypoint_types=["branch", "leaf"],
    threshold_dis=5.0,
    scale=(1, 1, 1),
    use_category=False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keypoint_types` | `list[str]` | `["branch", "leaf"]` | Which keypoints to evaluate |
| `threshold_dis` | `float` | `5.0` | Matching distance threshold (µm) |
| `scale` | `tuple` | `(1,1,1)` | Coordinate scaling |
| `use_category` | `bool` | `False` | Report per-type stats separately |

### `run(gold, pred) → dict`

**Combined mode (`use_category=False`):**

| Key | Type | Description |
|-----|------|-------------|
| `precision` | `float` | Detection precision |
| `recall` | `float` | Detection recall |
| `f1` | `float` | F1 score |
| `TP` / `FP` / `FN` | `int` | Matched / extra / missed keypoints |
| `num_gt` | `int` | Total gold keypoints |
| `num_pred` | `int` | Total predicted keypoints |

**Per-category mode (`use_category=True`):**

Result contains nested dicts `result["branch"]` and `result["leaf"]`, each with the same keys above.

---

## `swclib.metrics.fiber_metric.FiberMetric`

Fiber-level (root-to-leaf path) matching metric.

### Constructor

```python
FiberMetric(
    iou_threshold=0.8,
    dist_threshold=5.0,
    dist_sample=1.0,
    align_roots=True,
    align_roots_threshold=20.0,
    scale=(1, 1, 1),
    resample_step=2.0,
    only_from_soma=False,
    with_direction=False,
    use_category=False,
    min_fiber_length=0.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iou_threshold` | `float` | `0.8` | Minimum IoU for a match to count as TP |
| `dist_threshold` | `float` | `5.0` | Spatial tolerance for IoU calculation (µm) |
| `dist_sample` | `float` | `1.0` | Resampling step for IoU calculation |
| `align_roots` | `bool` | `True` | Align tree roots before comparison |
| `align_roots_threshold` | `float` | `20.0` | Max distance to match roots |
| `scale` | `tuple` | `(1,1,1)` | Coordinate scaling |
| `resample_step` | `float` | `2.0` | Tree resampling step |
| `only_from_soma` | `bool` | `False` | Only evaluate fibers from type=1 nodes |
| `with_direction` | `bool` | `False` | Consider fiber direction in matching |
| `use_category` | `bool` | `False` | Report axon / dendrite stats separately |
| `min_fiber_length` | `float` | `0.0` | Ignore fibers shorter than this (µm) |

### `run(gold, pred, skip_center_dist=100, return_fibers=False, verbose=False) → dict`

| Parameter | Type | Description |
|-----------|------|-------------|
| `gold` | `str` | Gold standard SWC path |
| `pred` | `str` | Predicted SWC path |
| `skip_center_dist` | `float` | Skip pairs whose soma centers are farther than this |
| `return_fibers` | `bool` | Include `SwcFiber` objects in result |
| `verbose` | `bool` | Print matching progress |

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `precision` | `float` | Fiber precision |
| `recall` | `float` | Fiber recall |
| `f1_score` | `float` | Harmonic mean |
| `TP` / `FP` / `FN` | `int` | Matched / extra / missed fibers |
| `num_gt` | `int` | Total gold fibers |
| `num_pred` | `int` | Total predicted fibers |
| `iou_matched` | `float` | Mean IoU of matched pairs |
| `iou_all` | `float` | Mean IoU across all gold fibers |
| `ious` | `list[float]` | Per-gold-fiber IoU values |
| `matches` | `list` | Matched `(gold_fiber, pred_fiber)` pairs |
| `FN_fiber_ids` | `list[list[int]]` | Node IDs of unmatched gold fibers |
| `gold_fibers` | `list[SwcFiber]` or `None` | Returned when `return_fibers=True` |
| `pred_fibers` | `list[SwcFiber]` or `None` | Returned when `return_fibers=True` |

With `use_category=True`, nested dicts `result["axon"]` and `result["dendrite"]` contain per-type stats.

---

## `swclib.metrics.manager.MetricManager`

Batch evaluation manager.

### Constructor

```python
MetricManager(
    metric_names=["ssd", "length", "keypoints", "fiber"],
    collect_method="micro",
    scale=(1, 1, 1),
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric_names` | `list[str]` | All four | Metrics to compute |
| `collect_method` | `str` | `"micro"` | `"micro"` (aggregate TP/FP/FN) or `"macro"` (average P/R/F1) |
| `scale` | `tuple` | `(1,1,1)` | Coordinate scaling applied to all metrics |

### Methods

#### `update_metric_params(metric_params)`
Override default parameters for individual metrics.

```python
manager.update_metric_params({
    "fiber": {"iou_threshold": 0.9, "dist_threshold": 3.0},
    "keypoints": {"threshold_dis": 3.0},
})
```

#### `add_data(swc_gt, swc_pred)`
Compute metrics for one gold/pred pair and store results.

#### `collect(save_path=None) → dict`
Aggregate stored results into a summary dict. Optionally save to JSON.