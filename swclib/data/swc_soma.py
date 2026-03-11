from dataclasses import dataclass, field
from typing import List
import numpy as np
from tifffile import tifffile
from scipy.spatial import cKDTree
import networkx as nx

from swclib.data.swc_node import SwcNode
from swclib.utils.nx import nx_clear_invalid_edges, nx_swc_to_grpah, nx_graph_to_swc

@dataclass
class SwcSoma:
    center: "SwcNode" = field(default_factory=lambda: SwcNode(0,0,0))
    annoed_center: List = field(default_factory=list)
    anno_fibers: List = field(default_factory=list)
    scale: List = field(default_factory=list)

    def rescale(self, scale):
        assert len(scale)==len(self.center[:])
        for i in range(len(scale)):
            self.center[i] = self.center[i] * scale[i]
            if len(self.annoed_center[:])>0:
                self.annoed_center[i] = self.annoed_center[i] * scale[i]
                for j in range(len(self.anno_fibers)):
                    for k in range(len(self.anno_fibers[j])):
                        self.anno_fibers[j][k][i] = self.anno_fibers[j][k][i] * scale[i]

def read_soma_from_file(path):
    somas = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#soma"):
                parts = line = line.replace('#soma', '').strip().split()
                somas.append(SwcSoma(
                    center=SwcNode(ntype=1, coord=[float(parts[0]), float(parts[1]), float(parts[2])], radius=float(parts[3])),
                    scale=[float(parts[4]), float(parts[5]), float(parts[6])]
                ))
            else:
                continue
    return somas

def save_somas_to_file(somas: List["SwcSoma"], path, scale=(1.0, 1.0, 1.0)):
    with open(path, 'w') as f:
        swc_counter = 0
        for soma in somas:
            f.write(f"#soma {soma.center[0]*scale[0]} {soma.center[1]*scale[1]} {soma.center[2]*scale[2]} {soma.center.radius} {soma.scale[0]} {soma.scale[1]} {soma.scale[2]}\n")
        for soma in somas:
            root = swc_counter
            f.write(f"{root} 1 {soma.annoed_center[0]*scale[0]:.13e} {soma.annoed_center[1]*scale[1]:.13e} {soma.annoed_center[2]*scale[2]:.13e} {0.1:.13e} -1\n")
            swc_counter += 1
            for fiber in soma.anno_fibers:
                pre = root
                for node in fiber[1:]:
                    f.write(f"{swc_counter} {node.ntype} {node[0]*scale[0]:.13e} {node[1]*scale[1]:.13e} {node[2]*scale[2]:.13e} {0.1:.13e} {pre}\n")
                    pre = swc_counter
                    swc_counter += 1


def create_soma_mask(somas, volume_shape=(300, 300, 300), out_path=None):
    """
    somas: soma nodes with attributes (x, y, z, r) - after tree.rescale()
    volume_shape: (Z, Y, X)
    voxel_size: (vz, vy, vx) physical size of one voxel AFTER rescale
    """
    Y, X, Z = volume_shape

    mask = np.zeros((Z, Y, X), dtype=np.uint8)

    # voxel center coordinates
    zz, yy, xx = np.ogrid[:Z, :Y, :X]
    for soma in somas:
        sy, sx, sz = soma.scale
        cx, cy, cz, r = soma.center[0]/sx, soma.center[1]/sy, soma.center[2]/sz, soma.center.radius
        
        # bounding box 加速
        z_min = max(0, int(cz - r) - 2)
        z_max = min(Z, int(cz + r) + 2)
        y_min = max(0, int(cy - r) - 2)
        y_max = min(Y, int(cy + r) + 2)
        x_min = max(0, int(cx - r) - 2)
        x_max = min(X, int(cx + r) + 2)

        zz_local = zz[z_min:z_max]
        yy_local = yy[:,y_min:y_max]
        xx_local = xx[:,:,x_min:x_max]

        dist2 = (((xx_local - cx)*sx)**2 +
                 ((yy_local - cy)*sy)**2 +
                 ((zz_local - cz)*sz)**2)

        mask[z_min:z_max, y_min:y_max, x_min:x_max][dist2 <= r*r] = 1
    
    if out_path is not None:
        tifffile.imwrite(out_path, mask)

    return mask

def nx_refine_with_soma_annotation(G: nx.Graph, soma_path: str, distance=3, scale=(1.0, 1.0, 1.0), min_distance=2.0):
    somas = read_soma_from_file(soma_path)
    if len(somas)>0:
        nodes = list(G.nodes())
        node_coords = [G.nodes[node]["coord"] for node in nodes]
        tree = cKDTree(node_coords)
        for soma in somas:
            soma.rescale(scale)
            ids = tree.query_ball_point(soma.center[:], r=soma.center.radius)
            for id in ids:
                if nodes[id] not in G:
                    print("soma has overlap")
                    continue
                G.remove_node(nodes[id])
            new_id = max(G.nodes()) + 1
            G.add_node(new_id, coord=soma.center, ntype=1)
            ids2 = tree.query_ball_point(soma.center[:], r=soma.center.radius+distance)
            for id in list(set(ids2) - set(ids)):
                if nodes[id] not in G:
                    continue
                if G.degree[nodes[id]]!=1:
                    continue
                u = new_id
                v = nodes[id]
                p0 = np.array(G.nodes[u]["coord"][:], dtype=float)
                p1 = np.array(G.nodes[v]["coord"][:], dtype=float)
                d = float(np.linalg.norm(p1 - p0))
                if d <= min_distance:
                    G.add_edge(u, v)
                    continue
                n_seg = int(np.ceil(d / min_distance))
                n_mid = n_seg - 1
                prev = u
                for k in range(1, n_mid + 1):
                    t = k / n_seg
                    pk = (1 - t) * p0 + t * p1
                    new_id2 = max(G.nodes()) + 1
                    G.add_node(new_id2, coord=pk, ntype=0)
                    G.add_edge(prev, new_id2)
                    prev = new_id2
                G.add_edge(prev, v)
    G = nx_clear_invalid_edges(G)
    return G

def refine_with_soma(swc_path, soma_path, out_path, scale=(1.0, 1.0, 1/0.35)):
    G = nx_swc_to_grpah(swc_path, scale=scale)
    G = nx_refine_with_soma_annotation(G, soma_path, scale=scale)
    nx_graph_to_swc(G, scale=(1/scale[0],1/scale[1],1/scale[2]),swc_path=out_path)