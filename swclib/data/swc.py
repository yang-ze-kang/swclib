import numpy as np
from scipy.spatial import cKDTree

class Swc(object):

    def __init__(self, file_name):
        self.nodes = {}
        self.edges = []
        self.bound_box = [0, 0, 0, 0, 0, 0]  # x0,y0,z0,x1,y1,z1
        self.rows = self.open(file_name)

    def open(self, file_name):
        with open(file_name) as f:
            for line in f.readlines():
                if line.startswith("#"):
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
                

