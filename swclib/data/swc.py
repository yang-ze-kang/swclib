import numpy as np
from scipy.spatial import cKDTree
import os
from copy import deepcopy
import math
from datetime import datetime, timezone

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
    
    @property
    def length(self):
        total_length = 0.0
        for nid, pid in self.edges:
            if pid == -1:
                continue
            if nid not in self.nodes or pid not in self.nodes:
                continue
            n1 = self.nodes[nid]
            n2 = self.nodes[pid]
            p1 = np.array([n1["x"], n1["y"], n1["z"]], dtype=float)
            p2 = np.array([n2["x"], n2["y"], n2["z"]], dtype=float)
            seg_len = np.linalg.norm(p1 - p2)
            total_length += seg_len
        return total_length

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

    def _refresh_bound_box(self):
        if len(self.nodes) == 0:
            self.bound_box = [np.inf, np.inf, np.inf, 0, 0, 0]
            return self.bound_box

        xs = [node["x"] for node in self.nodes.values()]
        ys = [node["y"] for node in self.nodes.values()]
        zs = [node["z"] for node in self.nodes.values()]
        self.bound_box = [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]
        return self.bound_box

    def read_region(
        self,
        start,
        end,
        out_path: str = None,
        out_r=None,
    ):
        """
        Read nodes in a cube and return them as a local-coordinate Swc object.

        Nodes are reindexed to 1..N. If a kept node's parent is outside the
        region, that node becomes a root in the returned SWC.
        """
        x0, y0, z0 = start
        x1, y1, z1 = end

        kept_nodes = [
            (old_id, node)
            for old_id, node in self.nodes.items()
            if x0 <= node["x"] < x1
            and y0 <= node["y"] < y1
            and z0 <= node["z"] < z1
        ]

        swc = Swc()
        if len(kept_nodes) == 0:
            if out_path is not None:
                swc.save_to_swc(out_path, write_header=False)
            return swc

        oldid2newid = {
            int(old_id): new_id
            for new_id, (old_id, _) in enumerate(kept_nodes, start=1)
        }

        for old_id, old_node in kept_nodes:
            old_id = int(old_id)
            new_id = oldid2newid[old_id]
            old_parent = int(old_node["parent"])
            new_parent = oldid2newid.get(old_parent, -1)
            radius = out_r if out_r is not None else old_node["radius"]

            swc.nodes[new_id] = {
                "id": new_id,
                "type": int(old_node["type"]),
                "x": float(old_node["x"] - x0),
                "y": float(old_node["y"] - y0),
                "z": float(old_node["z"] - z0),
                "radius": float(radius),
                "parent": new_parent,
            }
            swc.edges.append((new_id, new_parent))

        swc._refresh_bound_box()

        if out_path is not None:
            swc.save_to_swc(out_path, write_header=False)
        return swc

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

    def rotate_around_z_axis_line(self, xy_coord, angle, angle_in_degrees=True):
        """
        Rotate all nodes around the line parallel to the z-axis passing through (x, y).

        Args:
            x: x coordinate of the rotation axis.
            y: y coordinate of the rotation axis.
            angle: rotation angle.
            angle_in_degrees: if True, interpret angle in degrees; otherwise in radians.
        """
        if angle_in_degrees:
            angle = math.radians(angle)

        x, y = xy_coord
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        for node in self.nodes.values():
            dx = node["x"] - x
            dy = node["y"] - y
            node["x"] = x + dx * cos_a - dy * sin_a
            node["y"] = y + dx * sin_a + dy * cos_a

        self._refresh_bound_box()
        return self

    def resample(self, min_distance=2.0, in_place=True, round_mode=False):
        """
        Resample the SWC along each polyline segment (between "key nodes") using
        approximately uniform spacing = min_distance.

        - Upsample: insert nodes if spacing is too large
        - Downsample: remove redundant degree-2 nodes if spacing is too small
        - Topology is preserved by protecting key nodes: root, branch nodes, leaves
        - round_mode=True: move each segment's end key node to the nearest
          multiple of min_distance. A tail shorter than min_distance / 2 is
          dropped; otherwise the segment is extended to the next multiple.
        """
        if len(self.nodes) == 0:
            return self
        min_distance = float(min_distance)
        if min_distance <= 0:
            raise ValueError("min_distance must be positive")

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

        # -------- Rebuild nodes/edges by traversing segments --------
        new_nodes = {}
        new_edges = []

        # -------- Helpers --------
        def _node_xyzr(nid, use_new=False):
            n = new_nodes[nid] if use_new and nid in new_nodes else self.nodes[nid]
            return (
                np.array([float(n["x"]), float(n["y"]), float(n["z"])], dtype=float),
                float(n.get("radius", 1.0)),
            )

        def _node_type(nid, use_new=False):
            n = new_nodes[nid] if use_new and nid in new_nodes else self.nodes[nid]
            return int(n.get("type", 0))

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
            for idx_n, nid in enumerate(node_ids):
                # A branch node may have been moved by its parent segment. Reuse that
                # updated position as the start of every child segment.
                use_new = idx_n == 0
                p, r = _node_xyzr(nid, use_new=use_new)
                pts.append(p)
                rs.append(r)
                types.append(_node_type(nid, use_new=use_new))
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

            # Sample positions along arc length.
            if round_mode:
                k = int(np.floor(total / min_distance))
                remainder = total - k * min_distance
                if remainder + 1e-6 >= min_distance * 0.5:
                    k += 1
                if k == 0:
                    # Preserve the end key node even when a very short segment is
                    # rounded down to zero length.
                    s_positions = [0.0, 0.0]
                else:
                    s_positions = [i * min_distance for i in range(k + 1)]
            else:
                # 0, min_distance, 2*min_distance, ..., total. Always include the end.
                k = int(np.floor(total / min_distance))
                s_positions = [0.0]
                for i in range(1, k + 1):
                    s_positions.append(i * min_distance)
                if total - s_positions[-1] > 1e-6:
                    s_positions.append(total)
                else:
                    # Numerical snap: ensure last is exactly total
                    s_positions[-1] = total

            def _interpolate_at(s):
                if s > total + 1e-6:
                    for idx_seg in range(len(seg) - 1, -1, -1):
                        if seg[idx_seg] > 1e-12:
                            direction = (pts[idx_seg + 1] - pts[idx_seg]) / seg[idx_seg]
                            return (
                                pts[-1] + (s - total) * direction,
                                float(rs[-1]),
                                idx_seg,
                            )
                    return pts[-1], float(rs[-1]), max(0, len(seg) - 1)

                s = min(s, total)

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
                return p, r, idx

            # Linear interpolation along cum-length
            out = []
            start_old = node_ids[0]
            end_old = node_ids[-1]

            for j, s in enumerate(s_positions):
                p, r, idx = _interpolate_at(float(s))

                if j == 0:
                    # Reuse start node id
                    out.append(
                        (
                            start_old,
                            p,
                            r,
                            _node_type(start_old, use_new=True),
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
                            _node_type(end_old),
                            end_old,
                        )
                    )
                else:
                    # Insert new node
                    nid_new = next_id
                    next_id += 1
                    # out.append((nid_new, p, r, 5, None))  # type=5 for inserted node
                    insert_type = int(types[min(idx + 1, len(types) - 1)])
                    out.append((nid_new, p, r, insert_type, None))

            return out

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

    def get_density(self, dis=2.0, r=10.0, exclude_hops=5, reduction="mean"):
        """Estimate spatial density from points inside a fixed radius.

        The SWC is resampled at ``dis``. For every node, all Euclidean
        neighbors within ``r`` are collected after removing the node itself
        and ancestor/descendant nodes at most ``exclude_hops`` edges away in
        the SWC graph.  The exclusion does not cross a branch through a
        parent, so sibling subtrees remain eligible spatial neighbors.
        The retained distances are pooled across query nodes and reduced by
        ``reduction``. The result is normalized as
        ``clip((r - statistic) / r, 0, 1)``.

        ``reduction="mean"`` measures the typical non-topological spacing;
        ``reduction="min"`` reports the closest spatial crossing or overlap.
        Nodes from disconnected components are valid spatial neighbors.

        Parameters
        ----------
        dis : float
            Positive, finite resampling distance in the SWC coordinate unit.
        r : float
            Positive, finite Euclidean search radius. The default ``10.0``
            is intended for a 10-voxel neighborhood.
        exclude_hops : int
            Number of ancestor/descendant hops to exclude, including the
            query node. ``0`` excludes only the node itself; ``1`` excludes
            its direct parent and children as well. A path cannot cross a
            parent to reach a sibling branch.
        reduction : {"mean", "min"}
            Statistic used over all retained neighbor distances. ``minimum``
            is accepted as an alias for ``min``.

        Returns
        -------
        float
            A value in ``[0, 1]``. Returns ``0.0`` when no retained neighbor
            lies within the search radius.
        """
        from numbers import Integral

        dis = float(dis)
        r = float(r)
        if not np.isfinite(dis) or dis <= 0:
            raise ValueError("dis must be a positive finite number")
        if not np.isfinite(r) or r <= 0:
            raise ValueError("r must be a positive finite number")
        if not isinstance(exclude_hops, Integral) or isinstance(exclude_hops, bool):
            raise TypeError("exclude_hops must be an integer")
        exclude_hops = int(exclude_hops)
        if exclude_hops < 0:
            raise ValueError("exclude_hops must be non-negative")
        reduction = str(reduction).strip().lower()
        if reduction == "minimum":
            reduction = "min"
        if reduction not in {"mean", "min"}:
            raise ValueError("reduction must be 'mean' or 'min'")

        if self.nodes:
            raw_coords = np.asarray(
                [
                    [node["x"], node["y"], node["z"]]
                    for node in self.nodes.values()
                ],
                dtype=float,
            )
            if not np.isfinite(raw_coords).all():
                raise ValueError("SWC coordinates must be finite")

        swc = self.resample(min_distance=dis, in_place=False)
        if not swc.nodes:
            return 0.0

        node_ids = list(swc.nodes.keys())
        id2idx = {nid: idx for idx, nid in enumerate(node_ids)}
        coords = np.asarray(
            [
                [node["x"], node["y"], node["z"]]
                for node in (swc.nodes[nid] for nid in node_ids)
            ],
            dtype=float,
        )
        if not np.isfinite(coords).all():
            raise ValueError("SWC coordinates must be finite")

        n = len(node_ids)
        if n < 2:
            return 0.0

        # Build one indexed adjacency graph from parent fields.  This keeps
        # the topological exclusion logic consistent with the SWC itself and
        # avoids constructing a recursive anytree forest for every query.
        adj = [set() for _ in range(n)]
        parent_indices = None
        if exclude_hops <= 1:
            # The common path only needs direct parent/child adjacency.
            for idx, nid in enumerate(node_ids):
                pid = int(swc.nodes[nid]["parent"])
                parent_idx = id2idx.get(pid)
                if parent_idx is None:
                    continue
                adj[idx].add(parent_idx)
                adj[parent_idx].add(idx)
        else:
            parent_indices = np.full(n, -1, dtype=np.int64)
            for idx, nid in enumerate(node_ids):
                pid = int(swc.nodes[nid]["parent"])
                parent_idx = id2idx.get(pid)
                if parent_idx is None:
                    continue
                parent_indices[idx] = parent_idx
                adj[idx].add(parent_idx)
                adj[parent_idx].add(idx)

        # For h > 1, checking lineage membership by walking parent pointers
        # for every candidate can be quadratic on a deep tree.  Precompute
        # depths, component labels, and an ancestor jump table instead.  A
        # lineage-distance query then costs O(log N) and uses O(N log N)
        # memory, independent of the requested hop radius.
        component = depth = None
        up = None
        if exclude_hops > 1:
            component = np.full(n, -1, dtype=np.int64)
            depth = np.zeros(n, dtype=np.int64)
            children = [[] for _ in range(n)]
            for child, parent_idx in enumerate(parent_indices):
                if parent_idx >= 0:
                    children[int(parent_idx)].append(child)

            # Traverse from roots through directed children so ``depth``
            # agrees with the parent pointers.  Node order in an SWC file is
            # not required to be parent-before-child.
            component_id = 0
            for root in np.flatnonzero(parent_indices < 0):
                root = int(root)
                if component[root] >= 0:
                    continue
                component[root] = component_id
                stack = [root]
                while stack:
                    current = stack.pop()
                    for child in children[current]:
                        if component[child] >= 0:
                            continue
                        component[child] = component_id
                        depth[child] = depth[current] + 1
                        stack.append(child)
                component_id += 1

            # A valid SWC forest has at least one root per component.  Keep a
            # defensive fallback for malformed input so the function still
            # returns a finite result instead of looping forever.
            for start in range(n):
                if component[start] >= 0:
                    continue
                component[start] = component_id
                stack = [start]
                while stack:
                    current = stack.pop()
                    for neighbor in adj[current]:
                        if component[neighbor] >= 0:
                            continue
                        component[neighbor] = component_id
                        depth[neighbor] = depth[current] + 1
                        stack.append(neighbor)
                component_id += 1

            max_depth = int(depth.max()) if n else 0
            levels = max(1, max_depth.bit_length() + 1)
            # A root is its own ancestor.  This avoids sentinel indexing in
            # the jump table and keeps disconnected roots independent via the
            # component check below.
            up = np.empty((levels, n), dtype=np.int64)
            up[0] = np.where(parent_indices >= 0, parent_indices, np.arange(n))
            for level in range(1, levels):
                up[level] = up[level - 1][up[level - 1]]

        def _lineage_distance(first, second):
            """Return ancestor/descendant distance, or infinity otherwise."""
            if component[first] != component[second]:
                return np.inf

            if depth[first] <= depth[second]:
                ancestor, descendant = first, second
            else:
                ancestor, descendant = second, first

            diff = int(depth[descendant] - depth[ancestor])
            level = 0
            lifted = descendant
            remaining = diff
            while remaining:
                if remaining & 1:
                    lifted = int(up[level, lifted])
                remaining >>= 1
                level += 1
            return diff if lifted == ancestor else np.inf

        tree = cKDTree(coords)
        distances = []
        for idx in range(n):
            if exclude_hops == 0:
                def is_excluded(candidate_idx):
                    return candidate_idx == idx
            elif exclude_hops == 1:
                excluded = adj[idx]

                def is_excluded(candidate_idx):
                    return candidate_idx == idx or candidate_idx in excluded
            else:
                def is_excluded(candidate_idx):
                    return _lineage_distance(idx, candidate_idx) <= exclude_hops

            # Query the complete radius rather than only the nearest point;
            # this makes the mean statistic represent the requested spatial
            # neighborhood and lets ``min`` expose the closest crossing.
            for candidate_idx in tree.query_ball_point(coords[idx], r):
                candidate_idx = int(candidate_idx)
                if is_excluded(candidate_idx):
                    continue
                distance = float(np.linalg.norm(coords[idx] - coords[candidate_idx]))
                if distance <= r + 1e-12:
                    distances.append(distance)

        if not distances:
            return 0.0

        distances = np.asarray(distances, dtype=float)
        statistic = float(
            np.mean(distances) if reduction == "mean" else np.min(distances)
        )
        return float(np.clip((r - statistic) / r, 0.0, 1.0))

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
        radius: float = None,
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

        if reindex:
            from swclib.data.swc_forest import SwcForest
            tree = SwcForest(self)
            nodes = tree.get_preorder_nodes()
            node_ids = [node.nid for node in nodes]
            old2new = {old_id: i + 1 for i, old_id in enumerate(node_ids)}
            node_ids = sorted(node_ids, key=lambda nid: old2new[nid])
        else:
            node_ids = list(self.nodes.keys())
            if sort_by_id:
                node_ids.sort()
            old2new = {old_id: old_id for old_id in node_ids}

        with open(out_path, "w", encoding="utf-8") as f:
            if write_header:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                f.write("# Exported by Swc.save_to_swc\n")
                f.write(f"# Export time (UTC): {ts}\n")
                f.write("# id type x y z radius parent\n")

            for old_id in node_ids:
                n = self.nodes[old_id]
                nid = old2new[old_id]
                ntype_i = int(n["type"])
                x = float(n["x"])
                y = float(n["y"])
                z = float(n["z"])
                r = float(radius) if radius is not None else float(n["radius"])

                old_parent = int(n["parent"])
                if old_parent == -1:
                    pid = -1
                else:
                    pid = old2new[old_parent]

                f.write(
                    f"{nid} {ntype_i} "
                    f"{format(x, float_fmt)} {format(y, float_fmt)} {format(z, float_fmt)} "
                    f"{format(r, float_fmt)} {pid}\n"
                )

        return out_path
