import numpy as np
from scipy.spatial import cKDTree

class Swc(object):

    def __init__(self, file_name):
        self.nodes = {}
        self.edges = []
        self.bound_box = [0, 0, 0, 0, 0, 0]  # x0,y0,z0,x1,y1,z1
        self.open(file_name)

    def open(self, file_name):
        with open(file_name) as f:
            for line in f.readlines():
                if line.startswith("#"):
                    continue
                if len(line.split())!=7:
                    continue
                id, type, x, y, z, r, pid = map(float, line.split())
                id = int(id)
                typ = int(type)
                pid = int(pid)
                self.nodes[id] = {
                    "id": id,
                    "type": type,
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
        return np.array([[node["x"], node["y"], node["z"]] for node in self.nodes.values()])

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

    def resample(self, min_distance=2.0):
        """
        Densify edges so that the distance between adjacent nodes (node-parent chain)
        is <= min_distance by inserting intermediate nodes on long edges.
        """
        if len(self.nodes) == 0:
            return self

        next_id = max(self.nodes.keys()) + 1

        new_nodes = {}
        new_edges = []
        for nid, node in self.nodes.items():
            new_nodes[nid] = dict(node)

        for nid in sorted(self.nodes.keys()):
            node = new_nodes[nid]
            pid = int(node["parent"])

            if pid == -1 or pid not in new_nodes:
                new_edges.append((nid, pid))
                continue

            parent = new_nodes[pid]
            p = np.array([parent["x"], parent["y"], parent["z"]], dtype=float)
            c = np.array([node["x"], node["y"], node["z"]], dtype=float)
            v = c - p
            dist = float(np.linalg.norm(v))
            if dist <= min_distance or dist == 0.0:
                new_edges.append((nid, pid))
                continue

            m = int(np.ceil(dist / float(min_distance)))
            num_insert = max(0, m - 1)
            chain_ids = []
            for k in range(1, num_insert + 1):
                t = k / float(m)
                xyz = p + t * v
                r = float(parent.get("radius", 1.0)) + t * (float(node.get("radius", 1.0)) - float(parent.get("radius", 1.0)))

                new_nodes[next_id] = {
                    "id": next_id,
                    "type": 5, # 5 indicates an inserted node
                    "x": float(xyz[0]),
                    "y": float(xyz[1]),
                    "z": float(xyz[2]),
                    "radius": r,
                    "parent": None,  # fill later
                }
                chain_ids.append(next_id)
                next_id += 1

            # pid -> chain[0] -> ... -> chain[-1] -> nid
            if chain_ids:
                new_nodes[chain_ids[0]]["parent"] = pid
                new_edges.append((chain_ids[0], pid))
                for a, b in zip(chain_ids[1:], chain_ids[:-1]):
                    new_nodes[a]["parent"] = b
                    new_edges.append((a, b))
                last = chain_ids[-1]
                node["parent"] = last
                new_edges.append((nid, last))
            else:
                new_edges.append((nid, pid))
        self.nodes = new_nodes
        self.edges = new_edges
        xs = [n["x"] for n in self.nodes.values()]
        ys = [n["y"] for n in self.nodes.values()]
        zs = [n["z"] for n in self.nodes.values()]
        self.bound_box = [min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)]
        return self
        


    def get_father_path(self, nid, step=5):
        path = []
        current_id = nid
        while current_id != -1 and current_id in self.nodes:
            path.append(current_id)
            if len(path) >= step:
                break
            current_id = self.nodes[current_id]['parent']
        return path

    def check_min_distance_between_non_adjacent_nodes(self, threshold=5.0):
        n = len(self.nodes)
        if n < 3:
            return np.inf, []
        id2idx = {idx:node_id for idx, node_id in enumerate(self.nodes.keys())}
        coords = [[node['x'], node['y'], node['z']] for node_id, node in self.nodes.items()]
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
                

