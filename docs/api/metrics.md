# API Reference — Metrics

All metrics share the same base interface:

```python
result = metric.run(gold_swc_path, pred_swc_path)
```

Returns a `dict`.

Both file paths and in-memory `SwcForest` objects are accepted by most metrics.
When a metric accepts `scale`, it multiplies SWC coordinates by `(sx, sy, sz)`
before comparison.

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
    radius_threshold=2.0,
    length_threshold=0.2,
    scale=(1, 1, 1),
    resample_step=2.0,
    debug=False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `radius_threshold` | `float` | `2.0` | Spatial tolerance for edge matching (µm) |
| `length_threshold` | `float` | `0.2` | Minimum edge overlap length to count as matched (µm) |
| `scale` | `tuple` | `(1,1,1)` | Coordinate scaling |
| `resample_step` | `float` | `2.0` | Resampling step before edge comparison; use `None` to skip |
| `debug` | `bool` | `False` | Print intermediate matching information |

### `run(gold, pred) → dict`

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `precision` | `float` | Predicted length that matches gold |
| `recall` | `float` | Gold length covered by prediction |
| `f1` | `float` | Harmonic mean of precision and recall |
| `TP` | `float` | Matched gold length |
| `FP` | `float` | Predicted length not matched by gold |
| `FN` | `float` | Gold length not matched by prediction |

---

## `swclib.metrics.keypoint_metric.KeypointMetric`

Branch and leaf node detection metric.

### Constructor

```python
KeypointMetric(
    keypoint_types=["branch", "leaf"],
    threshold_dis=5.0,
    scale=(0.35, 0.35, 1),
    use_category=False,
    mode="block",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `keypoint_types` | `list[str]` | `["branch", "leaf"]` | Which keypoints to evaluate |
| `threshold_dis` | `float` | `5.0` | Matching distance threshold (µm) |
| `scale` | `tuple` | `(0.35,0.35,1)` | Coordinate scaling |
| `use_category` | `bool` | `False` | Report per-type stats separately |
| `mode` | `str` | `"block"` | `"block"` counts isolated roots as leaves; `"whole"` does not |

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

Result contains flat per-category fields such as `branch_precision`,
`branch_recall`, `branch_f1`, `leaf_precision`, `leaf_recall`, and `leaf_f1`.

---

## `swclib.metrics.fiber_metric.FiberMetric`

Fiber-level (root-to-leaf path) matching metric.

### Constructor

```python
FiberMetric(
    iou_threshold=0.8,
    dist_threshold=5.0,
    dist_sample=1.0,
    align_roots=False,
    align_roots_thredhold=20.0,
    scale=(1, 1, 1),
    resample_step=2.0,
    only_from_soma=False,
    with_direction=False,
    use_category=False,
    min_fiber_length=5.0,
    eps=1e-6,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `iou_threshold` | `float` | `0.8` | Minimum IoU for a match to count as TP |
| `dist_threshold` | `float` | `5.0` | Spatial tolerance for IoU calculation (µm) |
| `dist_sample` | `float` | `1.0` | Resampling step for IoU calculation |
| `align_roots` | `bool` | `False` | Align predicted roots to gold roots before comparison |
| `align_roots_thredhold` | `float` | `20.0` | Max distance to match roots. The parameter name intentionally follows the current source spelling |
| `scale` | `tuple` | `(1,1,1)` | Coordinate scaling |
| `resample_step` | `float` | `2.0` | Tree resampling step; use `None` to skip |
| `only_from_soma` | `bool` | `False` | Only evaluate fibers from type=1 nodes |
| `with_direction` | `bool` | `False` | Consider fiber direction in matching |
| `use_category` | `bool` | `False` | Report axon / dendrite stats separately |
| `min_fiber_length` | `float` | `5.0` | Ignore fibers shorter than this (µm) |
| `eps` | `float` | `1e-6` | Numerical guard for length-ratio checks |

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
| `f1` | `float` | Harmonic mean |
| `TP` / `FP` / `FN` | `int` | Matched / extra / missed fibers |
| `num_gt` | `int` | Total gold fibers |
| `num_pred` | `int` | Total predicted fibers |
| `iou_matched` | `float` | Mean IoU of matched pairs |
| `iou_all` | `float` | Mean IoU across all gold fibers |
| `ious` | `list[float]` | Per-gold-fiber IoU values |
| `matches` | `list` | Matched `(gold_fiber, pred_fiber)` pairs |
| `FN_fiber_ids` | `list[list[int]]` or `None` | Node IDs of unmatched gold fibers when fiber details are requested |
| `gold_fibers` | `list[SwcFiber]` or `None` | Returned when `return_fibers=True` |
| `pred_fibers` | `list[SwcFiber]` or `None` | Returned when `return_fibers=True` |

With `use_category=True`, the result includes flat axon and dendrite fields such
as `axon_precision`, `axon_recall`, `axon_f1`, `dendrite_precision`,
`dendrite_recall`, and `dendrite_f1`.

---

## `swclib.metrics.point_metric.PointMetric`

Point-level bidirectional nearest-neighbor metric over all resampled nodes.

### Constructor

```python
PointMetric(
    dist_threshold=4,
    scale=(1, 1, 1),
    resample_step=2.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dist_threshold` | `float` | `4` | Nearest-neighbor match threshold |
| `scale` | `tuple` | `(1,1,1)` | Coordinate scaling |
| `resample_step` | `float` or `None` | `2.0` | Resampling step before comparison |

### `run(gold, pred, return_points=False) → dict`

**Returns:**

| Key | Type | Description |
|-----|------|-------------|
| `precision` / `recall` / `f1` | `float` | Point-level scores |
| `MES` | `float` | Morphology error score-style overlap ratio |
| `TP` / `FP` / `FN` | `int` | Matched / extra / missed points |
| `S_G`, `S_hit_pred`, `S_miss`, `S_extra` | `int` | Raw count terms used by MES |

When `return_points=True`, the result also includes nearest-neighbor distances,
indices, matched pairs, missed/extra point IDs, and the coordinate arrays.

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
| `metric_names` | `list[str]` | All four | Metrics to compute. Supported names: `"ssd"`, `"point"`, `"length"`, `"keypoints"`, `"fiber"` |
| `collect_method` | `str` | `"micro"` | Current implementation supports `"micro"` |
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

!!! warning
    `MetricManager.collect()` assumes non-SSD metrics expose top-level `TP`,
    `FP`, and `FN`. Metrics configured with per-category output may need
    custom aggregation if you want category-specific summaries.
