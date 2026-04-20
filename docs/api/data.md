# API Reference — Data

## `swclib.data.swc.Swc`

Lightweight dict-based representation of an SWC file.

### Constructor

```python
Swc(file_name=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_name` | `str` or `None` | Path to SWC file. If `None`, creates empty instance. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `nodes` | `dict[int, dict]` | Node dict keyed by node ID |
| `edges` | `list[tuple]` | List of `(node_id, parent_id)` |
| `bound_box` | `list[float]` | `[min_x, min_y, min_z, max_x, max_y, max_z]` |
| `file_name` | `str` | Source file path |
| `length` | `float` | *Property.* Total path length. |

### Methods

---

#### `open(file_name)`
Parse and load an SWC text file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_name` | `str` | Path to SWC file |

---

#### `rescale(scale)`
Scale all node coordinates by `(sx, sy, sz)`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `scale` | `tuple[float, float, float]` | Scale factors `(sx, sy, sz)` |

---

#### `add_offset(offset)`
Translate all node coordinates.

| Parameter | Type | Description |
|-----------|------|-------------|
| `offset` | `tuple[float, float, float]` | Translation `(ox, oy, oz)` |

---

#### `resample(min_distance=2.0, in_place=True)`
Upsample or downsample the tree so consecutive nodes are approximately `min_distance` apart.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_distance` | `float` | `2.0` | Target spacing |
| `in_place` | `bool` | `True` | Modify in place or return new `Swc` |

---

#### `remove_duplicate_nodes(use_radius=False, round_ndigits=None, reindex=False, in_place=True)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_radius` | `bool` | `False` | Include radius in duplicate check |
| `round_ndigits` | `int` or `None` | `None` | Round coordinates before comparison |
| `reindex` | `bool` | `False` | Reassign node IDs after removal |
| `in_place` | `bool` | `True` | Modify in place or return new `Swc` |

---

#### `get_coords() → ndarray`
Returns all node coordinates as an array of shape `(N, 3)`.

---

#### `get_roots(return_coords=True)`
Returns nodes with `parent_id == -1`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `return_coords` | `bool` | `True` | If `True`, returns `ndarray`; else returns list of node dicts |

---

#### `get_density(dis=10.0, p=0.3, exclude_hops=1) → float`
Estimates local spatial density of the neuron.

---

#### `check_min_distance_between_non_adjacent_nodes(threshold=5.0) → tuple`
Returns `(min_distance, pairs)` where `pairs` is a list of node ID pairs closer than `threshold`.

---

#### `save_to_swc(out_path, sort_by_id=True, write_header=True, float_fmt='.6f', mkdir=True, reindex=False, radius=None) → str`
Write the SWC to disk. Returns the output path.

---

### Module-level functions

#### `merge_swcs(swcs, offsets=None, keep_file_name=True) → Swc`

| Parameter | Type | Description |
|-----------|------|-------------|
| `swcs` | `list[Swc]` | List of Swc objects to merge |
| `offsets` | `list[tuple]` or `None` | Per-file translation offsets |
| `keep_file_name` | `bool` | Preserve file names in merged result |

---

## `swclib.data.swc_forest.SwcForest`

Object-oriented tree representation. Handles both single-root and multi-root (disconnected) SWC files.

### Constructor

```python
SwcForest(swc=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `swc` | `str`, `Swc`, or `None` | File path, `Swc` object, or empty |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `roots` | `list[SwcNode]` | List of root nodes |
| `id_set` | `set[int]` | All node IDs |
| `scale` | `tuple` | Applied scaling |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `size()` | `int` | Number of nodes |
| `length(force_update=False)` | `float` | Total path length |
| `get_edge_num()` | `int` | Number of edges |
| `contains(nid)` | `bool` | Node ID membership test |
| `get_node_list(update=False)` | `list[SwcNode]` | All nodes (cached) |
| `get_node_by_nid(nid)` | `SwcNode` | ID → node lookup |
| `get_branch_nodes()` | `list[SwcNode]` | Nodes with ≥ 2 children |
| `get_leaf_nodes()` | `list[SwcNode]` | Nodes with 0 children |
| `get_roots(return_coords=True)` | `list[SwcNode]` or `ndarray` | Root nodes |
| `get_somas()` | `list[SwcSoma]` | Soma regions |
| `get_fibers(only_from_soma=False, min_length=0.0)` | `list[SwcFiber]` | All root-to-leaf paths |
| `get_fibers_by_roi(roi)` | `list[SwcFiber]` | Fibers passing through ROI |
| `get_fiber_by_leaf(leaf, roi=None)` | `SwcFiber` | Path to a specific leaf |
| `get_nearest_node(coord, subtree_root=None, topk=1)` | `SwcNode` or `list` | KD-tree nearest neighbor |
| `get_copy()` | `SwcForest` | Deep copy |
| `rescale(scale)` | — | Scale coordinates |
| `relocation(offset)` | — | Translate coordinates |
| `sort_node_list(key='default')` | — | Reorder/reindex nodes |
| `get_lca_preprocess(node_num)` | — | Initialize LCA data structure |
| `get_lca(u, v)` | `int` | Lowest Common Ancestor of nodes `u` and `v` |
| `save_to_file(path)` | — | Export to SWC file |

---

## `swclib.data.swc_node.SwcNode`

A single node in the tree.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `nid` | `int` | Node ID |
| `ntype` | `int` | Node type (1–5) |
| `coord` | `EuclideanPoint3D` | 3D coordinates |
| `radius` | `float` | Node radius |
| `parent` | `SwcNode` or `None` | Parent node |
| `root_length` | `float` | Accumulated path length from root |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `is_root()` | `bool` | True if node has no valid parent |
| `is_regular()` | `bool` | True if `nid >= 0` |
| `distance(target, mode='3D')` | `float` | Distance to another node |
| `parent_distance()` | `float` | Distance to parent |
| `get_subtree_node_list()` | `list[SwcNode]` | All descendants |
| `get_subtree_length(force_update=False)` | `float` | Subtree path length |
| `get_subtree_leafs(roi=None)` | `list[SwcNode]` | Terminal nodes |
| `get_subtree_fibers(roi=None, with_root=False)` | `list[SwcFiber]` | Paths to all leaves |
| `get_fiber_by_leaf(roi=None)` | `SwcFiber` | Path from root to this leaf |
| `get_rerooted_tree(nid_start=1)` | `SwcNode` | Reroot tree at this node |
| `to_swc_str(pid=None, scale=(1,1,1))` | `str` | Format as SWC line |

### Module-level function

```python
nodes2coords(nodes) → ndarray  # shape (N, 3)
```

---

## `swclib.data.swc_fiber.SwcFiber`

A root-to-leaf path.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `nodes` | `list[SwcNode]` | Ordered nodes from root to leaf |
| `length` | `float` | *Property.* Total path length |
| `coords` | `ndarray` | *Property.* Coordinates, shape `(N, 3)` |
| `center` | `ndarray` | *Property.* Mean coordinate |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `append(node)` | — | Add node to end |
| `pop()` | — | Remove last node |
| `reverse()` | — | Reverse node order |
| `copy()` | `SwcFiber` | Shallow copy |
| `get_nearest_node(point, return_dist=False)` | `SwcNode` or `(SwcNode, float)` | Nearest node on fiber |
| `cal_iou(fiber, dist_sample=1.0, dist_threshold=3.0)` | `float` | IoU with another fiber |
| `get_overlap_length_with(fiber, dist_sample=1.0, dist_threshold=3.0)` | `float` | Overlap length |
| `is_sub_fiber_of(fiber, ...)` | `bool` or `float` | Whether this fiber is a sub-path |
| `to_str_list(scale=(1,1,1))` | `str` | Format as SWC lines |

---

## `swclib.data.swc_soma.SwcSoma`

Dataclass representing a soma region.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `center` | `SwcNode` | Soma center node |
| `anno_fibers` | `list[list[SwcNode]]` | Fibers radiating from soma |
| `scale` | `tuple` | Scaling factors |

### Module-level functions

```python
read_soma_from_file(path) → list[SwcSoma]
save_somas_to_file(somas, path, scale=(1,1,1))
create_soma_mask(somas, volume_shape, out_path=None) → ndarray
```