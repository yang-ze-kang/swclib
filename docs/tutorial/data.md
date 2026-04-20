# Loading and Manipulating SWC Data

swclib provides two levels of data representation, each suited for different use cases:

| Class | Location | Best for |
|-------|----------|---------|
| `Swc` | `swclib.data.swc` | File I/O, coordinate transforms, simple queries |
| `SwcForest` | `swclib.data.swc_forest` | Tree algorithms, fiber extraction, LCA queries |

---

## `Swc` — Lightweight dict-based representation

`Swc` stores nodes and edges as plain dicts. It is the fastest way to read, transform, and write SWC files.

### Loading

```python
from swclib.data.swc import Swc

swc = Swc("path/to/neuron.swc")
```

Each node is accessible via `swc.nodes[node_id]` as a dict with keys:
`id`, `type`, `x`, `y`, `z`, `radius`, `parent`.

### Coordinate transforms

```python
# Scale to physical units (voxel size in µm)
swc.rescale((0.5, 0.5, 0.35))

# Translate by offset
swc.add_offset((100.0, 200.0, 50.0))
```

### Resampling

`resample` inserts or removes nodes so that consecutive nodes are approximately `min_distance` apart.

```python
swc.resample(min_distance=2.0)
```

### Removing duplicate nodes

```python
# Duplicates based on XYZ coordinates only
swc.remove_duplicate_nodes()

# Also consider radius when detecting duplicates
swc.remove_duplicate_nodes(use_radius=True)

# Round coordinates before comparison
swc.remove_duplicate_nodes(round_ndigits=2)
```

### Querying nodes and structure

```python
# All node coordinates as (N, 3) array
coords = swc.get_coords()

# Root nodes (parent_id == -1)
roots = swc.get_roots(return_coords=True)   # returns ndarray
roots = swc.get_roots(return_coords=False)  # returns list of node dicts

# Total path length
print(swc.length)

# Bounding box: [min_x, min_y, min_z, max_x, max_y, max_z]
print(swc.bound_box)
```

### Spatial density

```python
# Fraction of volume within 'dis' µm that contains neuron nodes
density = swc.get_density(dis=10.0, p=0.3, exclude_hops=1)
```

### Detecting crossings

```python
# Find pairs of non-adjacent nodes that are closer than 'threshold'
min_dist, pairs = swc.check_min_distance_between_non_adjacent_nodes(threshold=5.0)
```

### Merging multiple SWC files

```python
from swclib.data.swc import merge_swcs

swc1 = Swc("cell1.swc")
swc2 = Swc("cell2.swc")

# Merge without offset
merged = merge_swcs([swc1, swc2])

# Merge with per-file translations
merged = merge_swcs([swc1, swc2], offsets=[(0, 0, 0), (500, 0, 0)])
merged.save_to_swc("merged.swc")
```

### Saving

```python
swc.save_to_swc(
    "output.swc",
    sort_by_id=True,
    write_header=True,
    float_fmt=".6f",
    mkdir=True,
    reindex=False,
)
```

---

## `SwcForest` — Object-oriented tree representation

`SwcForest` wraps SWC data in a proper tree structure using `anytree`. It handles both single-root and multi-root (disconnected) SWC files, making it the standard representation for topology-aware operations.

### Loading

```python
from swclib.data.swc_forest import SwcForest

# From file path
forest = SwcForest("neuron.swc")

# From an existing Swc object
forest = SwcForest(swc)
```

### Basic properties

```python
print(forest.size())           # number of nodes
print(forest.length())         # total path length
print(forest.get_edge_num())   # number of edges
print(len(forest.roots))       # number of root nodes
```

### Accessing nodes

```python
# All nodes (cached list)
nodes = forest.get_node_list()

# Dict lookup by node ID
node = forest.get_node_by_nid(42)

# Branch nodes (degree > 1) and leaf nodes (degree 0)
branches = forest.get_branch_nodes()
leaves   = forest.get_leaf_nodes()
```

### Nearest-neighbor queries

```python
# Nearest node to a 3D point
nearest = forest.get_nearest_node([x, y, z])

# Top-k nearest
results = forest.get_nearest_node([x, y, z], topk=5)
# returns list of (SwcNode, distance)
```

### Soma detection

Nodes with `type == 1` define the soma. `get_somas` extracts connected soma regions:

```python
somas = forest.get_somas()
for soma in somas:
    print(soma.center.coord)
```

### Fiber extraction

A *fiber* is a path from a root (or soma) to a leaf node.

```python
# All fibers
fibers = forest.get_fibers()

# Only fibers that originate at a soma node (type=1)
fibers = forest.get_fibers(only_from_soma=True)

# Filter by minimum length
fibers = forest.get_fibers(min_length=20.0)

# Fibers that pass through a region of interest
fibers = forest.get_fibers_by_roi(roi)

# Single fiber to a specific leaf node
fiber = forest.get_fiber_by_leaf(leaves[0])
```

### LCA (Lowest Common Ancestor)

For distance calculations between arbitrary nodes:

```python
forest.get_lca_preprocess(node_num=forest.size())
ancestor_id = forest.get_lca(node_u, node_v)
```

### Saving

```python
forest.save_to_file("output.swc")
```

---

## `SwcNode` — Individual node

Every node in `SwcForest` is a `SwcNode` instance:

```python
node = forest.get_node_list()[0]

print(node.nid)            # node ID
print(node.ntype)          # 1=soma, 2=axon, 3=basal, 4=apical, 5=other
print(node.coord)          # EuclideanPoint3D
print(node.radius)
print(node.root_length)    # accumulated path length from root

# Distance to another node
d = node.distance(other_node, mode="3D")   # or "2D" for XY only

# Topology checks
node.is_root()
node.is_leaf()

# Sub-tree operations
descendants = node.get_subtree_node_list()
subtree_len = node.get_subtree_length()
leaves = node.get_subtree_leafs()
fibers = node.get_subtree_fibers()
```

---

## `SwcFiber` — A single root-to-leaf path

```python
fiber = fibers[0]

print(fiber.length)           # path length (float)
print(fiber.coords)           # ndarray (N, 3)
print(fiber.center)           # mean coordinate

# Nearest node on this fiber to an external point
nearest = fiber.get_nearest_node([x, y, z], return_dist=False)

# Compare two fibers
iou    = fiber.cal_iou(other_fiber, dist_threshold=3.0)
is_sub = fiber.is_sub_fiber_of(other_fiber)
overlap = fiber.get_overlap_length_with(other_fiber, dist_threshold=3.0)
```