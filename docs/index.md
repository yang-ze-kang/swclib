# swclib

**swclib** is a Python library for processing, converting, and evaluating neuron morphology data in [SWC format](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Docs/swcFormat.html).

SWC is the standard file format in computational neuroscience for representing 3D reconstructions of neuronal structures. Each neuron is described as a tree of labeled 3D points, encoding the soma, axons, and dendrites along with their spatial coordinates and radii.

---

## What swclib provides

| Module | Description |
|--------|-------------|
| `swclib.data` | Load, manipulate, and traverse SWC trees and forests |
| `swclib.image` | Bidirectional conversion between SWC and 3D binary mask images |
| `swclib.metrics` | Quantitative comparison of neuron reconstructions |
| `swclib.whole_brain` | Parallel reader for large tiled TIFF image stacks |

---

## Typical workflow

```
Raw Image Volume
      │
      ▼
  Segmentation / Reconstruction Algorithm
      │
      ▼
   SWC File  ──────────────────────────────────────────────┐
      │                                                     │
      ▼                                                     ▼
swclib.image                                         swclib.metrics
(SWC ↔ Mask)                                    (compare with gold standard)
      │                                                     │
      ▼                                                     ▼
 3D Mask (.tif)                                    Precision / Recall / F1
```

---

## SWC format overview

Each line in an SWC file encodes one node:

```
id   type   x   y   z   radius   parent_id
```

Standard node types:

| Type | Meaning |
|------|---------|
| 1 | Soma |
| 2 | Axon |
| 3 | Basal dendrite |
| 4 | Apical dendrite |
| 5 | Undefined / other |

Root nodes use `parent_id = -1`. A single SWC file may contain one tree (single soma) or multiple disconnected trees (forest).

---

## Getting started

See [Installation](install.md) and [Quick Start](quickstart.md).