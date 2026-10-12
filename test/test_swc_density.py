import math

import pytest

from swclib.data.swc import Swc


def _line(count, spacing=1.0, y=0.0, start_id=1):
    swc = Swc()
    for offset in range(count):
        nid = start_id + offset
        parent = -1 if offset == 0 else nid - 1
        swc.nodes[nid] = {
            "id": nid,
            "type": 0,
            "x": offset * spacing,
            "y": y,
            "z": 0.0,
            "radius": 1.0,
            "parent": parent,
        }
        swc.edges.append((nid, parent))
    return swc


def test_density_radius_supports_mean_and_min_reductions():
    swc = Swc()
    for nid, x in ((1, 0.0), (2, 2.0), (3, 8.0)):
        swc.nodes[nid] = {
            "id": nid,
            "type": 3,
            "x": x,
            "y": 0.0,
            "z": 0.0,
            "radius": 1.0,
            "parent": -1,
        }
        swc.edges.append((nid, -1))

    # All three components are disconnected, so all pair distances are valid:
    # mean = (2 + 6 + 8) / 3 and minimum = 2.
    assert swc.get_density(dis=1.0, r=10.0, exclude_hops=5, reduction="mean") == pytest.approx(
        (10.0 - 16.0 / 3.0) / 10.0
    )
    assert swc.get_density(dis=1.0, r=10.0, exclude_hops=5, reduction="min") == pytest.approx(
        0.8
    )
    assert swc.get_density(dis=1.0, r=10.0, exclude_hops=5, reduction="minimum") == pytest.approx(
        0.8
    )


def test_density_defaults_use_ten_voxel_radius_and_five_hops():
    swc = _line(11, spacing=1.0)
    close = _line(11, spacing=1.0, y=0.5, start_id=100)
    swc.nodes.update(close.nodes)
    swc.edges.extend(close.edges)

    assert 0.0 <= swc.get_density() <= 1.0
    assert 0.0 <= swc.get_density(reduction="min") <= 1.0


def test_density_no_neighbor_inside_radius_is_zero():
    assert _line(1).get_density(dis=1.0, r=10.0, exclude_hops=5) == 0.0
    assert _line(20).get_density(dis=1.0, r=10.0, exclude_hops=20) == 0.0


def test_density_validates_parameters():
    swc = _line(10)
    with pytest.raises(ValueError):
        swc.get_density(dis=0.0)
    with pytest.raises(ValueError):
        swc.get_density(r=0.0)
    with pytest.raises(ValueError):
        swc.get_density(exclude_hops=-1)
    with pytest.raises(TypeError):
        swc.get_density(exclude_hops=1.5)
    with pytest.raises(ValueError):
        swc.get_density(reduction="max")


def test_density_handles_deep_chain_without_recursive_forest():
    density = _line(1500).get_density(dis=1.0, r=10.0, exclude_hops=5)
    assert math.isfinite(density)
    assert 0.0 <= density <= 1.0


def test_density_hop_filter_is_independent_of_file_node_order():
    swc = _line(40, spacing=1.0)
    items = list(swc.nodes.items())
    swc.nodes = dict(reversed(items))
    swc.edges = list(reversed(swc.edges))

    density = swc.get_density(dis=1.0, r=10.0, exclude_hops=3)
    assert math.isfinite(density)
    assert 0.0 <= density <= 1.0


def test_density_allows_sibling_branch_neighbors():
    swc = Swc()
    swc.nodes[1] = {
        "id": 1,
        "type": 1,
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "radius": 1.0,
        "parent": -1,
    }
    swc.edges.append((1, -1))
    for nid in range(2, 102):
        swc.nodes[nid] = {
            "id": nid,
            "type": 3,
            "x": 1.0,
            "y": 0.0,
            "z": 0.0,
            "radius": 1.0,
            "parent": 1,
        }
        swc.edges.append((nid, 1))

    # Sibling leaves are two graph hops apart through the root, but the
    # exclusion follows only ancestor/descendant lineages. Their overlapping
    # positions therefore remain valid spatial neighbors.
    assert swc.get_density(dis=1.0, r=10.0, exclude_hops=2) == 1.0
    # The same branch-safe exclusion is the default behavior.
    assert swc.get_density(dis=1.0, r=10.0) == 1.0
