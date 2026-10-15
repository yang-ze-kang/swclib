"""Read-only data adapters for the viewer's built-in plugins."""

import math


def soma_candidates(path, rule):
    """Return annotated positions, without fitting, editing or loading images.

    SwcForest.get_somas fits a sphere to specially annotated straight fibers;
    this viewer needs the original SWC node positions instead. Reuse Swc's
    parser and root API and keep the candidate rule explicit.
    """
    from swclib.data.swc import Swc

    if rule not in ("type-first", "roots", "type-only"):
        raise ValueError("不支持的胞体候选规则")
    swc = Swc(str(path))
    ids = [nid for nid, node in swc.nodes.items() if node["type"] == 1] if rule != "roots" else []
    method = "type=1"
    if rule == "roots" or (rule == "type-first" and not ids):
        ids = swc.get_roots(return_coords=False)
        method = "parent=-1"
    candidates = []
    for nid in sorted(ids):
        node = swc.nodes[nid]
        center = [float(node[axis]) for axis in ("x", "y", "z")]
        if not all(math.isfinite(v) for v in center):
            raise ValueError(f"候选节点 {nid} 的坐标不是有限数值")
        candidates.append(dict(node_id=nid, center=center, method=method))
    return dict(candidates=candidates, node_count=len(swc.nodes))
