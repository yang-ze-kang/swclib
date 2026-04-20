# Installation

## Requirements

- Python >= 3.8
- pip or conda

## Install via pip

```bash
pip install swclib
```

## Install from source

```bash
git clone https://github.com/yang-ze-kang/swclib.git
cd swclib
pip install -e .
```

## Dependencies

swclib requires the following packages, which are installed automatically:

| Package | Purpose |
|---------|---------|
| `numpy` | Array operations |
| `scipy` | Spatial indexing (KDTree), signal processing |
| `networkx` | Graph-based skeletonization |
| `anytree` | Tree data structure |
| `miniball` | Minimum bounding sphere |
| `rtree` | Spatial index for edge matching |
| `rasterio` | Tiled TIFF I/O |
| `tifffile` | TIFF image reading/writing |
| `scikit-image` | Image skeletonization and filtering |
| `opencv-python` | Image rasterization |
| `tqdm` | Progress bars |

## Verify installation

```python
import swclib
print("swclib installed successfully")
```