import numpy as np
from scipy.spatial import cKDTree
import os
from copy import deepcopy
import math
from datetime import datetime, timezone

from swclib.data.swc_node import nodes2coords
from swclib.utils.points import cal_segment_length


def merge_swcs(swcs, *, offsets=None, keep_file_name=True):
    """
    Merge a list of Swc objects into a single Swc.

    Args:
        swcs: list[Swc]
        offsets: None or list[tuple[float,float,float]]
                 If provided, offsets[i] will be added to xyz of swcs[i].
        keep_file_name: if True, keep merged.file_name = None (or join names yourself).

    Returns:
        Swc: merged swc with re-indexed node ids and fixed parent links.
    """
    merged = Swc(file_name=None)

    if offsets is None:
        offsets = [(0.0, 0.0, 0.0)] * len(swcs)
    if len(offsets) != len(swcs):
        raise ValueError(f"offsets length {len(offsets)} != swcs length {len(swcs)}")
    
    if len(swcs) == 0:
        return merged
    
    if len(swcs) == 1:
        # Just copy the single SWC (with offset if needed)
        swc = deepcopy(swcs[0])
        if offsets[0] != (0.0, 0.0, 0.0):
            swc.add_offset(offsets[0])
        return swc
    
    # ---- helpers: adapt these to your node structure ----
    def get_id(node):
        # node can be dict: node["id"]
        # or object: node.id
        return node["id"] if isinstance(node, dict) else node.id

    def set_id(node, new_id):
        if isinstance(node, dict):
            node["id"] = new_id
        else:
            node.id = new_id

    def get_parent(node):
        # parent id: -1 for root
        return node["parent"] if isinstance(node, dict) else node.parent

    def set_parent(node, new_parent):
        if isinstance(node, dict):
            node["parent"] = new_parent
        else:
            node.parent = new_parent

    def get_xyz(node):
        # return x,y,z
        if isinstance(node, dict):
            return float(node["x"]), float(node["y"]), float(node["z"])
        return float(node.x), float(node.y), float(node.z)

    def set_xyz(node, x, y, z):
        if isinstance(node, dict):
            node["x"], node["y"], node["z"] = x, y, z
        else:
            node.x, node.y, node.z = x, y, z

    # -----------------------------------------------------

    next_id = 1
    all_edges = []

    # bbox init
    minx = miny = minz = math.inf
    maxx = maxy = maxz = -math.inf

    for swc, (ox, oy, oz) in zip(swcs, offsets):
        if swc is None:
            continue

        # old_id -> new_id
        id_map = {}

        # 1) copy nodes with new ids
        # assume swc.nodes is dict[id] = node
        for old_id in sorted(swc.nodes.keys()):
            node = deepcopy(swc.nodes[old_id])

            new_id = next_id
            next_id += 1
            id_map[old_id] = new_id
            set_id(node, new_id)

            # apply offset
            x, y, z = get_xyz(node)
            x, y, z = x + ox, y + oy, z + oz
            set_xyz(node, x, y, z)

            # bbox update
            minx, miny, minz = min(minx, x), min(miny, y), min(minz, z)
            maxx, maxy, maxz = max(maxx, x), max(maxy, y), max(maxz, z)

            merged.nodes[new_id] = node

        # 2) fix parent links
        for old_id in sorted(swc.nodes.keys()):
            node_new = merged.nodes[id_map[old_id]]
            old_parent = get_parent(swc.nodes[old_id])

            if old_parent is None or int(old_parent) < 0:
                set_parent(node_new, -1)
            else:
                # if parent not in nodes (broken swc), treat as root
                if old_parent not in id_map:
                    raise NotImplementedError("broken SWC with missing parent")
                    set_parent(node_new, -1)
                else:
                    set_parent(node_new, id_map[old_parent])

        # 3) rebuild edges (or map existing edges)
        # safer: rebuild from parent pointers
        for old_id in sorted(swc.nodes.keys()):
            old_parent = get_parent(swc.nodes[old_id])
            if old_parent is None or int(old_parent) < 0:
                continue
            if old_parent not in id_map:
                raise NotImplementedError("broken SWC with missing parent")
                continue
            all_edges.append((id_map[old_id], id_map[old_parent]))

    merged.edges = all_edges

    # finalize bbox
    if minx is math.inf:
        merged.bound_box = [0, 0, 0, 0, 0, 0]
    else:
        merged.bound_box = [minx, miny, minz, maxx, maxy, maxz]

    if keep_file_name:
        merged.file_name = None

    return merged


class Swc(object):

    def __init__(self, file_name=None):
        self.nodes = {}
        self.edges = []
        self.bound_box = [np.inf, np.inf, np.inf, 0, 0, 0]  # x0,y0,z0,x1,y1,z1
        self.file_name = file_name
        if file_name is not None:
            self.open(file_name)

    def open(self, file_name):
        with open(file_name) as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue
                if len(line.split()) != 7:
                    continue
                id, ntype, x, y, z, r, pid = map(float, line.split())
                id = int(id)
                ntype = int(ntype)
                pid = int(pid)
                self.nodes[id] = {
                    "id": id,
                    "type": ntype,
                    "x": x,
                    "y": y,
                    "z": z,
                    "radius": r,
                    "parent": pid,
                }
                self.edges.append((id, pid))
                if x < self.bound_box[0]:
                    self.bound_box[0] = x
                if x > self.bound_box[3]:
                    self.bound_box[3] = x
                if y < self.bound_box[1]:
                    self.bound_box[1] = y
                if y > self.bound_box[4]:
                    self.bound_box[4] = y
                if z < self.bound_box[2]:
                    self.bound_box[2] = z
                if z > self.bound_box[5]:
                    self.bound_box[5] = z

    def get_coords(self):
        return np.array(
            [[node["x"], node["y"], node["z"]] for node in self.nodes.values()]
        )

    def get_roots(self, return_coords=True):
        roots = []
        for nid, node in self.nodes.items():
            pid = int(node["parent"])
            if pid == -1:
                roots.append(nid)
        if return_coords:
            return [
                [self.nodes[nid]["x"], self.nodes[nid]["y"], self.nodes[nid]["z"]]
                for nid in roots
            ]
        return roots

    def rescale(self, scale):
        """
        Rescale the coordinates of all nodes by the given scale factor.
        """
        for nid, node in self.nodes.items():
            node["x"] *= scale[0]
            node["y"] *= scale[1]
            node["z"] *= scale[2]
        self.bound_box = [
            self.bound_box[0] * scale[0],
            self.bound_box[1] * scale[1],
            self.bound_box[2] * scale[2],
            self.bound_box[3] * scale[0],
            self.bound_box[4] * scale[1],
            self.bound_box[5] * scale[2],
        ]
        return self

    def add_offset(self, offset):
        """
        Add an offset to the coordinates of all nodes.
        """
        for nid, node in self.nodes.items():
            node["x"] += offset[0]
            node["y"] += offset[1]
            node["z"] += offset[2]
        self.bound_box = [
            self.bound_box[0] + offset[0],
            self.bound_box[1] + offset[1],
            self.bound_box[2] + offset[2],
            self.bound_box[3] + offset[0],
            self.bound_box[4] + offset[1],
            self.bound_box[5] + offset[2],
        ]
        return self

    def resample(self, min_distance=2.0, in_place=True):
        """
        Resample the SWC along each polyline segment (between "key nodes") using
        approximately uniform spacing = min_distance.

        - Upsample: insert nodes if spacing is too large
        - Downsample: remove redundant degree-2 nodes if spacing is too small
        - Topology is preserved by protecting key nodes: root, branch nodes, leaves
        """
        if len(self.nodes) == 0:
            return self

        # -------- Build parent/children adjacency --------
        parent = {}
        children = {nid: [] for nid in self.nodes.keys()}
        roots = []
        for nid, node in self.nodes.items():
            pid = int(node["parent"])
            parent[nid] = pid
            if pid == -1:
                roots.append(nid)
            else:
                children[pid].append(nid)
        assert len(roots) > 0

        # A "key node" is where we must keep topology stable:
        # - root (no valid parent)
        # - branch (children count != 1)
        # - leaf (children count == 0)
        # Note: leaf is already included by children count != 1, but keep explicit for clarity.
        key_nodes = set()
        for nid in self.nodes.keys():
            cdeg = len(children.get(nid, []))
            pid = parent.get(nid, -1)
            if pid == -1 or pid not in self.nodes:
                key_nodes.add(nid)  # root
            if cdeg == 0:
                key_nodes.add(nid)  # leaf
            if cdeg != 1:
                key_nodes.add(nid)  # branch or leaf

        next_id = max(self.nodes.keys()) + 1

        # -------- Helpers --------
        def _node_xyzr(nid):
            n = self.nodes[nid]
            return (
                np.array([float(n["x"]), float(n["y"]), float(n["z"])], dtype=float),
                float(n.get("radius", 1.0)),
            )

        def _resample_polyline(node_ids):
            """
            Given a chain of node_ids [start ... end] with start and end being key nodes,
            resample along arc-length with step=min_distance.
            Returns a list of tuples: (new_id, xyz, radius, type, old_id_or_None)
            where old_id_or_None is the original node id if we reused it (start/end),
            otherwise None for inserted nodes.
            """
            nonlocal next_id

            # Collect points and radii along the chain
            pts = []
            rs = []
            types = []
            for nid in node_ids:
                p, r = _node_xyzr(nid)
                pts.append(p)
                rs.append(r)
                types.append(int(self.nodes[nid].get("type", 0)))
            pts = np.stack(pts, axis=0)
            rs = np.array(rs, dtype=float)

            # Cumulative arc-length
            seg = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            total = float(cum[-1])

            # If chain is degenerate
            if total <= 1e-12:
                start_id = node_ids[0]
                end_id = node_ids[-1]
                out = []
                out.append((start_id, pts[0], rs[0], types[0], start_id))
                if end_id != start_id:
                    out.append((end_id, pts[-1], rs[-1], types[-1], end_id))
                return out

            # Sample positions: 0, min_distance, 2*min_distance, ..., total
            # Always include the end.
            k = int(np.floor(total / float(min_distance)))
            s_positions = [0.0]
            for i in range(1, k + 1):
                s_positions.append(i * float(min_distance))
            if total - s_positions[-1] > 1e-6:
                s_positions.append(total)
            else:
                # Numerical snap: ensure last is exactly total
                s_positions[-1] = total

            # Linear interpolation along cum-length
            out = []
            start_old = node_ids[0]
            end_old = node_ids[-1]

            for j, s in enumerate(s_positions):
                # Find right interval in cum
                idx = int(np.searchsorted(cum, s, side="right") - 1)
                idx = max(0, min(idx, len(cum) - 2))

                s0, s1 = float(cum[idx]), float(cum[idx + 1])
                if s1 - s0 <= 1e-12:
                    t = 0.0
                else:
                    t = (s - s0) / (s1 - s0)

                p = pts[idx] + t * (pts[idx + 1] - pts[idx])
                r = float(rs[idx] + t * (rs[idx + 1] - rs[idx]))

                if j == 0:
                    # Reuse start node id
                    out.append(
                        (
                            start_old,
                            p,
                            r,
                            int(self.nodes[start_old].get("type", 0)),
                            start_old,
                        )
                    )
                elif j == len(s_positions) - 1:
                    # Reuse end node id
                    out.append(
                        (
                            end_old,
                            p,
                            r,
                            int(self.nodes[end_old].get("type", 0)),
                            end_old,
                        )
                    )
                else:
                    # Insert new node
                    nid_new = next_id
                    next_id += 1
                    # out.append((nid_new, p, r, 5, None))  # type=5 for inserted node
                    out.append((nid_new, p, r, int(types[idx + 1]), None))

            return out

        # -------- Rebuild nodes/edges by traversing segments --------
        new_nodes = {}
        new_edges = []

        # We'll traverse from each root, and for every child, build a chain until the next key node.
        stack = list(roots)
        visited_key = set()

        def _ensure_node(nid, xyz, radius, ntype):
            # Create or overwrite node in new_nodes (overwriting is fine for reused key nodes)
            new_nodes[nid] = {
                "id": int(nid),
                "type": int(ntype),
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "radius": float(radius),
                "parent": -1,  # will be set when connecting
            }

        # Initialize roots in new_nodes (keep their original coordinates/radius)
        for r in stack:
            p, rr = _node_xyzr(r)
            _ensure_node(r, p, rr, int(self.nodes[r].get("type", 0)))
            new_nodes[r]["parent"] = -1
            new_edges.append((r, -1))

        # DFS over key nodes; for each outgoing child, resample that segment
        key_queue = stack[:]
        while key_queue:
            start = key_queue.pop()
            if start in visited_key:
                continue
            visited_key.add(start)

            for ch in children.get(start, []):
                # Build chain: start -> ... -> end_key
                chain = [start]
                cur = ch
                while True:
                    chain.append(cur)
                    if cur in key_nodes:
                        end_key = cur
                        break
                    # degree-2 interior node must have exactly one child in this direction
                    nxts = children.get(cur, [])
                    if len(nxts) != 1:
                        # Safety: treat as key if something weird happens
                        end_key = cur
                        key_nodes.add(cur)
                        break
                    cur = nxts[0]

                # Resample this polyline segment
                sampled = _resample_polyline(chain)

                # Create nodes + connect parents along sampled list
                prev_id = None
                for idx_s, (nid_s, xyz_s, r_s, t_s, old_id) in enumerate(sampled):
                    if idx_s == 0:
                        # first is the start key node: parent stays as already set (root or will be set by its own segment)
                        # But if it wasn't initialized (non-root key), keep original parent if it exists later; for now leave as is.
                        pass
                    else:
                        # set parent to prev_id
                        _ensure_node(nid_s, xyz_s, r_s, t_s)
                        new_nodes[nid_s]["parent"] = int(prev_id)
                        new_edges.append((nid_s, int(prev_id)))

                    prev_id = nid_s

                # Push the end key node for further traversal
                end_id = sampled[-1][0]
                if end_id in key_nodes:
                    key_queue.append(end_id)

        xs = [n["x"] for n in new_nodes.values()]
        ys = [n["y"] for n in new_nodes.values()]
        zs = [n["z"] for n in new_nodes.values()]
        bound_box = [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]

        if in_place:
            self.nodes = new_nodes
            self.edges = new_edges
            self.bound_box = bound_box
            return self
        else:
            new_swc = Swc()
            new_swc.nodes = new_nodes
            new_swc.edges = new_edges
            new_swc.bound_box = bound_box
            return new_swc

    def get_father_path(self, nid, step=5):
        path = []
        current_id = nid
        while current_id != -1 and current_id in self.nodes:
            path.append(current_id)
            if len(path) >= step:
                break
            current_id = self.nodes[current_id]["parent"]
        return path

    def check_min_distance_between_non_adjacent_nodes(self, threshold=5.0):
        """
        Check the minimum distance between non-adjacent nodes in the SWC structure.
        Returns the minimum distance found and the list of node ID pairs that are within the threshold.
        """
        n = len(self.nodes)
        if n < 3:
            return np.inf, []
        id2idx = {idx: node_id for idx, node_id in enumerate(self.nodes.keys())}
        coords = [
            [node["x"], node["y"], node["z"]] for node_id, node in self.nodes.items()
        ]
        tree = cKDTree(coords)

        min_dist = np.inf
        matched_pair = []
        for i in range(n):
            dists, idxs = tree.query(coords[i], k=n, distance_upper_bound=threshold)
            for dist, idx in zip(dists[1:], idxs[1:]):
                if dist == np.inf:
                    break
                nid1 = id2idx[i]
                nid2 = id2idx[idx]
                if (nid1, nid2) in self.edges or (nid2, nid1) in self.edges:
                    continue
                if (nid1, nid2) in matched_pair or (nid2, nid1) in matched_pair:
                    continue
                p1 = self.get_father_path(nid1, step=10)
                p2 = self.get_father_path(nid2, step=10)
                if set(p1) & set(p2):
                    continue

                matched_pair.append((nid1, nid2))
                if dist < min_dist:
                    min_dist = dist
        return min_dist, matched_pair

    def get_density(self, dis=10.0, p=0.3, exclude_hops=1):
        """
        Compute swc density.

        Parameters
        ----------
        dis : float
            Resampling distance (also controls point spacing after resample).
        in_place : bool
            Whether resampling modifies self in-place.
        exclude_hops : int
            Topological exclusion radius in hops.
            - 1 means exclude only direct neighbors (parent/child).
            - 2 means also exclude neighbors within 2 hops, etc.
        Returns: float, density value
        -------

        """
        from swclib.data.swc_forest import SwcForest

        swc = self.resample(min_distance=dis, in_place=False)

        swc_tree = SwcForest(swc)
        components = swc_tree.get_components()

        coords, node_ids = [], []
        for comp in components:
            if len(comp) <= 2 + exclude_hops:
                continue
            coord = nodes2coords(comp)
            coords.append(coord)
            node_ids.extend([node.nid for node in comp])
        if len(coords) == 0:
            return 0.0
        coords = np.vstack(coords)
        id2idx = {nid: i for i, nid in enumerate(node_ids)}

        n = len(coords)
        if n <= 2 + exclude_hops:
            return 0.0

        tree = cKDTree(coords)

        # -------- Build adjacency (undirected) for hop-based exclusion --------
        # edges are stored as (id, pid); ignore pid = -1 or missing.
        adj = {nid: set() for nid in node_ids}
        for nid, pid in swc.edges:
            if nid not in node_ids:
                continue
            if pid == -1:
                continue
            if nid not in swc.nodes or pid not in swc.nodes:
                continue
            adj[nid].add(pid)
            adj[pid].add(nid)

        def _excluded_set(start_nid, hops):
            """
            Nodes to exclude from nearest-neighbor search: within <=hops topological steps.
            hops=1 -> {parent/child + self}
            """
            if hops <= 0:
                return {start_nid}
            seen = {start_nid}
            frontier = {start_nid}
            for _ in range(hops):
                new_frontier = set()
                for u in frontier:
                    for v in adj.get(u, ()):
                        if v not in seen:
                            seen.add(v)
                            new_frontier.add(v)
                frontier = new_frontier
                if not frontier:
                    break
            return seen

        # Precompute excluded sets if you want speed; for hop=1 it's cheap anyway.
        # For large n and larger exclude_hops, you may want to cache or limit.
        excluded_cache = {}
        if exclude_hops <= 1:
            # Fast path: exclude only direct neighbors + self
            for nid in node_ids:
                s = {nid}
                s |= adj.get(nid, set())
                excluded_cache[nid] = s
        else:
            for nid in node_ids:
                excluded_cache[nid] = _excluded_set(nid, exclude_hops)

        # -------- For each node, find nearest valid (non-excluded) neighbor --------
        dlist = []

        # Start with a small k and expand if needed
        k0 = min(16, n)
        for nid in node_ids:
            idx0 = id2idx[nid]
            excluded = excluded_cache[nid]
            if len(excluded) == n:
                # all nodes are excluded (rare: tiny component)
                continue

            k = k0
            found = None

            while True:
                # query k nearest (includes itself at rank 0)
                dists, idxs = tree.query(coords[idx0], k=k)

                # Make them iterable even when k==1
                if np.isscalar(dists):
                    dists = np.array([dists])
                    idxs = np.array([idxs])

                # Scan candidates in increasing distance
                for dist, j in zip(dists, idxs):
                    cand_nid = node_ids[int(j)]
                    if cand_nid in excluded:
                        continue
                    found = float(dist)
                    break

                if found is not None:
                    break

                if k >= n:
                    break  # no valid match exists (rare: tiny component)
                k = min(n, k * 2)
            assert found is not None
            dlist.append(found)

        dlist = np.sort(dlist)
        dlist = dlist[: -int(len(dlist) * p)]
        mean_dist = float(np.mean(dlist))
        density = np.clip(1 - mean_dist / (dis * exclude_hops * 2), 0.0, 1.0)
        return float(density)

    def remove_duplicate_nodes(
        self,
        use_radius: bool = False,
        round_ndigits: int = None,
        reindex: bool = False,
        in_place: bool = True,
    ):
        """
        Remove duplicate nodes that share the same position.

        Parameters
        ----------
        use_radius : bool
            If True, use (x, y, z, radius) as duplicate key.
            Otherwise use only (x, y, z).
        round_ndigits : int | None
            If not None, round coordinates (and radius if enabled) before comparison.
            Useful when coordinates differ only by tiny floating-point noise.
        reindex : bool
            Whether to reassign node ids to 1..N after deduplication.
        in_place : bool
            Whether to modify self in place.

        Returns
        -------
        Swc
            Deduplicated SWC object.
        """
        if len(self.nodes) == 0:
            return self if in_place else Swc()

        def _make_key(node):
            x = float(node["x"])
            y = float(node["y"])
            z = float(node["z"])
            r = float(node["radius"])

            if round_ndigits is not None:
                x = round(x, round_ndigits)
                y = round(y, round_ndigits)
                z = round(z, round_ndigits)
                r = round(r, round_ndigits)

            if use_radius:
                return (x, y, z, r)
            return (x, y, z)

        # Keep the first node for each duplicate key
        key_to_keep_id = {}
        old_id_to_keep_id = {}
        kept_ids = []

        for nid, node in self.nodes.items():
            key = _make_key(node)
            if key not in key_to_keep_id:
                key_to_keep_id[key] = nid
                kept_ids.append(nid)
            old_id_to_keep_id[nid] = key_to_keep_id[key]

        # Rebuild kept nodes and redirect parent pointers
        new_nodes = {}
        for old_id in kept_ids:
            node = self.nodes[old_id]
            old_parent = int(node["parent"])

            if old_parent == -1:
                new_parent = -1
            else:
                # If parent exists, redirect it to the kept node
                new_parent = old_id_to_keep_id.get(old_parent, -1)

            # Avoid self-loop after deduplication
            if new_parent == old_id:
                new_parent = -1

            new_nodes[old_id] = {
                "id": old_id,
                "type": int(node["type"]),
                "x": float(node["x"]),
                "y": float(node["y"]),
                "z": float(node["z"]),
                "radius": float(node["radius"]),
                "parent": int(new_parent),
            }

        # Optional: reindex node ids to compact 1..N
        if reindex:
            old_to_new = {
                old_id: i + 1 for i, old_id in enumerate(sorted(new_nodes.keys()))
            }
            reindexed_nodes = {}

            for old_id in sorted(new_nodes.keys()):
                node = new_nodes[old_id]
                new_id = old_to_new[old_id]
                old_parent = int(node["parent"])
                new_parent = -1 if old_parent == -1 else old_to_new[old_parent]

                reindexed_nodes[new_id] = {
                    "id": new_id,
                    "type": int(node["type"]),
                    "x": float(node["x"]),
                    "y": float(node["y"]),
                    "z": float(node["z"]),
                    "radius": float(node["radius"]),
                    "parent": int(new_parent),
                }

            new_nodes = reindexed_nodes

        # Rebuild edges
        new_edges = []
        for nid, node in new_nodes.items():
            new_edges.append((nid, int(node["parent"])))

        # Recompute bounding box
        if len(new_nodes) > 0:
            xs = [node["x"] for node in new_nodes.values()]
            ys = [node["y"] for node in new_nodes.values()]
            zs = [node["z"] for node in new_nodes.values()]
            new_bound_box = [
                min(xs),
                min(ys),
                min(zs),
                max(xs),
                max(ys),
                max(zs),
            ]
        else:
            new_bound_box = [np.inf, np.inf, np.inf, 0, 0, 0]

        if in_place:
            self.nodes = new_nodes
            self.edges = new_edges
            self.bound_box = new_bound_box
            return self
        else:
            new_swc = Swc()
            new_swc.nodes = new_nodes
            new_swc.edges = new_edges
            new_swc.bound_box = new_bound_box
            return new_swc

    def save_to_swc(
        self,
        out_path: str,
        *,
        sort_by_id: bool = True,
        write_header: bool = True,
        float_fmt: str = ".6f",
        mkdir: bool = True,
        reindex: bool = False,
    ) -> str:
        """
        Export current SWC to a .swc text file.

        Parameters
        ----------
        out_path : str
            Output SWC path.
        sort_by_id : bool
            Whether to sort nodes by node id before writing.
        write_header : bool
            Whether to write SWC header lines.
        float_fmt : str
            Format for floats, e.g. ".3f", ".6f".
        mkdir : bool
            Create parent directory if missing.
        reindex : bool
            Whether to reassign node ids to 1..N when saving.

        Returns
        -------
        str
            The output path.
        """
        if mkdir:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        node_ids = list(self.nodes.keys())
        if sort_by_id:
            node_ids.sort()

        if reindex:
            old_to_new = {old_id: i + 1 for i, old_id in enumerate(node_ids)}
        else:
            old_to_new = {old_id: old_id for old_id in node_ids}

        with open(out_path, "w", encoding="utf-8") as f:
            if write_header:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                f.write("# Exported by Swc.save_to_swc\n")
                f.write(f"# Export time (UTC): {ts}\n")
                f.write("# id type x y z radius parent\n")

            for old_id in node_ids:
                n = self.nodes[old_id]
                nid = old_to_new[old_id]
                ntype_i = int(n["type"])
                x = float(n["x"])
                y = float(n["y"])
                z = float(n["z"])
                r = float(n["radius"])

                old_parent = int(n["parent"])
                if old_parent == -1:
                    pid = -1
                else:
                    pid = old_to_new[old_parent]

                f.write(
                    f"{nid} {ntype_i} "
                    f"{format(x, float_fmt)} {format(y, float_fmt)} {format(z, float_fmt)} "
                    f"{format(r, float_fmt)} {pid}\n"
                )

        return out_path
