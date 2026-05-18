import numpy as np
from skimage.morphology import skeletonize
import networkx as nx
from dataclasses import dataclass
from math import acos, degrees
from scipy.signal import savgol_filter

from swclib.data.swc_fiber import resample_nodes_by_distance
from swclib.data.swc_soma import nx_refine_with_soma_annotation
from swclib.utils.nx import nx_clear_invalid_edges, nx_graph_to_swc


@dataclass
class Segment:
    id: int
    node_ids: list
    node_coords: list
    length: float
    end_nodes: tuple  # (node_u, node_v)
    end_dirs: tuple  # directions



def euclidean_dist(p, q, voxel_size):
    diff = (np.array(p) - np.array(q)) * np.array(voxel_size)
    return float(np.linalg.norm(diff))


def safe_unit_vector(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return np.zeros_like(v)
    return v / n


def angle_between(v1, v2):
    v1 = safe_unit_vector(v1)
    v2 = safe_unit_vector(v2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return degrees(acos(dot))

@dataclass
class Mask2Swc:

    voxel_size=(1.0, 1.0, 1.0/0.35)
    thres_soma_degree=4
    thres_segment_merge_short_dis=3.0
    thres_segment_merge_long_dis=20.0
    thres_segment_merge_angle=30.0
    thres_fiber_min_len=20.0
    thres_branch_min_len=15.0
    connectivity=26
    smooth_window_size=7
    smooth_ploy=2
    node_sample_distance=2

    def run(self, mask, swc_path, soma_path=None, radius=0.1, verbos=False):

        # 1. skeletonize
        skel = skeletonize(mask)

        # 2. build graph
        G, self.max_node_id = self.build_graph_from_skeleton(skel, connectivity=self.connectivity, voxel_size=self.voxel_size)

        # 3. remove isolated nodes
        G, isolated = self.remove_isolated_nodes(G)
        if verbos:
            print(
                f"Remove isolated nodes: {len(isolated)} isolated nodes has been removed."
            )

        # 4. merge fibers
        G, merged_num = self.merge_fibers(
            G,
            voxel_size=self.voxel_size,
            thres_segment_merge_short_dis=self.thres_segment_merge_short_dis,
            thres_segment_merge_long_dis=self.thres_segment_merge_long_dis,
            thres_segment_merge_angle=self.thres_segment_merge_angle,
        )
        if verbos:
            print(f"Merge fibers: {merged_num} fibers has been merged.")

        # 5. remove short fibers
        G, removed_num = self.remove_short_fibers(
            G,
            voxel_size=self.voxel_size,
            thres_min_len=self.thres_fiber_min_len,
        )
        if verbos:
            print(f"Remove short fiber: {removed_num} fibers have been removed.")

        # 5. remove short branchs
        G, removed_num = self.remove_short_branchs(
            G,
            voxel_size=self.voxel_size,
            thres_min_len=self.thres_branch_min_len,
        )
        if verbos:
            print(f"Remove short branch: {removed_num} branchs have been removed.")

        # smooth swc fibers
        G = self.refine_fibers(G, self.smooth_window_size, self.smooth_ploy, self.node_sample_distance, self.voxel_size)
        if soma_path is not None:
            G = nx_refine_with_soma_annotation(G, soma_path, scale=self.voxel_size)
        swc_lines = nx_graph_to_swc(G, scale=(1/self.voxel_size[0], 1/self.voxel_size[1], 1/self.voxel_size[2]), swc_path=swc_path, radius=radius)
        return swc_lines

    def build_graph_from_skeleton(
        self, skel: np.ndarray, connectivity: int = 26, voxel_size=(1.0, 1.0, 1.0)
    ) -> nx.Graph:
        """
        Params:
            skel: (0/1)
        Returns:
            network G, attr 'coord' = (z,y,x) or (y,x)
        """
        nodes = np.argwhere(skel > 0)
        G = nx.Graph()

        # coord -> node_id
        coord_to_id = {}
        for nid, coord in enumerate(nodes):
            coord = tuple(coord.tolist())
            coord_to_id[coord] = nid
            swc_node = tuple([coord[2]*voxel_size[0], coord[1]*voxel_size[1], coord[0]*voxel_size[2]])
            G.add_node(nid, coord=swc_node, ntype=0)

        # neighbourhood offsets
        if skel.ndim == 2:
            # 8 neighbourhood
            offsets = [
                (dy, dx)
                for dy in [-1, 0, 1]
                for dx in [-1, 0, 1]
                if not (dy == 0 and dx == 0)
            ]
        else:
            # 3D: 6 / 26 neighbourhood
            if connectivity == 6:
                offsets = [
                    (dz, dy, dx)
                    for dz, dy, dx in [
                        (1, 0, 0),
                        (-1, 0, 0),
                        (0, 1, 0),
                        (0, -1, 0),
                        (0, 0, 1),
                        (0, 0, -1),
                    ]
                ]
            elif connectivity == 26:
                offsets = [
                    (dz, dy, dx)
                    for dz in [-1, 0, 1]
                    for dy in [-1, 0, 1]
                    for dx in [-1, 0, 1]
                    if not (dz == 0 and dy == 0 and dx == 0)
                ]
            else:
                raise NotImplementedError

        for i, coord in enumerate(nodes):
            for shift in offsets:
                neighbor_coord = tuple(coord + shift)
                if neighbor_coord in coord_to_id:
                    j = coord_to_id[neighbor_coord]
                    if not G.has_edge(i, j):
                        weight = np.linalg.norm(coord - neighbor_coord)
                        G.add_edge(i, j, weight=weight)
        G = nx.minimum_spanning_tree(G, weight="weight")
        return G, len(nodes)

    def remove_isolated_nodes(self, G: nx.Graph):
        isolated = [n for n, d in G.degree() if d == 0]
        G.remove_nodes_from(isolated)
        return G, isolated

    def extract_segments(self, G: nx.Graph, voxel_size=(1, 1, 1)) -> list:
        segments = []
        visited_edges = set()

        def add_segment_from_path(path_nodes, sid):

            length = 0.0
            coords = [G.nodes[n]["coord"] for n in path_nodes]
            for i in range(len(coords) - 1):
                length += euclidean_dist(coords[i], coords[i + 1], voxel_size)

            # 端点
            u, v = path_nodes[0], path_nodes[-1]
            # 端点方向: 用各自靠里的一个点估计
            if len(path_nodes) >= 2:
                dir_u = np.array(G.nodes[path_nodes[1]]["coord"]) - np.array(
                    G.nodes[path_nodes[0]]["coord"]
                )
                dir_v = np.array(G.nodes[path_nodes[-2]]["coord"]) - np.array(
                    G.nodes[path_nodes[-1]]["coord"]
                )
            else:
                dir_u = dir_v = np.zeros(len(coords[0]))

            seg = Segment(
                id=sid,
                node_ids=path_nodes,
                node_coords=[G.nodes[node]["coord"] for node in path_nodes],
                length=length,
                end_nodes=(u, v),
                end_dirs=(dir_u, dir_v),
            )
            return seg

        # 所有关键点
        key_nodes = [n for n, d in G.degree() if d != 2]

        # 1) 从关键点出发，找所有关键点-关键点路径
        sid = 0
        for s in key_nodes:
            for t in G.neighbors(s):
                edge = tuple(sorted((s, t)))
                if edge in visited_edges:
                    continue

                # 从 s -> t 延展
                path = [s, t]
                prev = s
                curr = t

                while G.degree(curr) == 2:
                    neigh = [x for x in G.neighbors(curr) if x != prev]
                    if len(neigh) != 1:
                        break
                    nxt = neigh[0]
                    path.append(nxt)
                    prev, curr = curr, nxt

                # 标记边已访问
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i + 1])))
                    visited_edges.add(e)
                segments.append(add_segment_from_path(path, sid))
                sid += 1

        # assert len(visited_edges) == G.number_of_edges()
        return segments

    def merge_fibers(
        self,
        G: nx.Graph,
        voxel_size=(1, 1, 1),
        thres_segment_merge_short_dis=2.0,
        thres_segment_merge_long_dis=10.0,
        thres_segment_merge_angle=45.0,
    ):
        """
        思路：对长度 < thres_segment_len 的 segment，
        若其某个端点与另一个 segment 端点距离 < thres_segment_dis 且角度 < thres_segment_connect_angle，
        则在这两个端点之间增加一条边，实现“连接/合并”。
        然后返回更新后的图（G 直接修改）。
        """
        segments = self.extract_segments(G, voxel_size)
        # 预先把所有 segment 的端点坐标、方向缓存下来
        seg_info = []
        for seg in segments:
            u, v = seg.end_nodes
            coord_u = np.array(G.nodes[u]["coord"])
            coord_v = np.array(G.nodes[v]["coord"])
            dir_u, dir_v = seg.end_dirs
            seg_info.append(
                {
                    "seg": seg,
                    "u": u,
                    "v": v,
                    "coord_u": coord_u,
                    "coord_v": coord_v,
                    "dir_u": dir_u,
                    "dir_v": dir_v,
                    "length": seg.length,
                }
            )
        # 对每对 segment 尝试合并
        merged_num = 0
        seg_info = sorted(seg_info, key=lambda item: item["length"])
        for s_info in seg_info:
            seg_s = s_info["seg"]
            for t_info in seg_info:
                seg_t = t_info["seg"]
                if seg_t.id == seg_s.id:
                    continue

                # 遍历四种端点组合: (s.u, t.u), (s.u, t.v), (s.v, t.u), (s.v, t.v)
                pairs = [("u", "u"), ("u", "v"), ("v", "u"), ("v", "v")]
                merged = False
                for side_s, side_t in pairs:
                    node_s = s_info[side_s]
                    node_t = t_info[side_t]
                    coord_s = s_info[f"coord_{side_s}"]
                    coord_t = t_info[f"coord_{side_t}"]
                    dir_s = s_info[f"dir_{side_s}"]
                    dir_t = t_info[f"dir_{side_t}"]

                    # 距离判断
                    dist = euclidean_dist(coord_s, coord_t, voxel_size)
                    if dist<=thres_segment_merge_short_dis:
                        if not G.has_edge(node_s, node_t):
                            merged_num += 1
                            G.add_edge(node_s, node_t)
                            merged = True
                            break
                    elif dist<=thres_segment_merge_long_dis:
                        ang = angle_between(dir_s, -dir_t)
                        if ang <= thres_segment_merge_angle:
                            if not G.has_edge(node_s, node_t):
                                merged_num += 1
                                G.add_edge(node_s, node_t)
                                merged = True
                                break
                if merged:
                    break
        return G, merged_num

    def remove_short_fibers(self, G: nx.Graph, voxel_size=(1, 1, 1), thres_min_len=5.0):
        removed_num = 0
        segments = self.extract_segments(G, voxel_size=voxel_size)
        for seg in segments:
            if seg.length >= thres_min_len:
                continue
            nodes = seg.node_ids
            if G.degree(nodes[0]) == 1 and G.degree(nodes[-1]) == 1:
                removed_num += 1
                for n in nodes:
                    if n in G:
                        G.remove_node(n)
        G = nx_clear_invalid_edges(G)
        return G, removed_num

    def remove_short_branchs(
        self, G: nx.Graph, voxel_size=(1, 1, 1), thres_min_len=5.0
    ):
        removed_num = 0
        segments = self.extract_segments(G, voxel_size=voxel_size)
        for seg in segments:
            if seg.length >= thres_min_len:
                continue
            nodes = seg.node_ids
            if G.degree(nodes[0]) > 2 and G.degree(nodes[-1]) == 1:
                removed_num += 1
                for n in nodes[1:]:
                    if n in G:
                        G.remove_node(n)
            elif G.degree(nodes[0]) == 1 and G.degree(nodes[-1]) > 2:
                removed_num += 1
                for n in nodes[::-1][1:]:
                    if n in G:
                        G.remove_node(n)
        G = nx_clear_invalid_edges(G)
        return G, removed_num

    def refine_fibers(
        self,
        G: nx.Graph,
        smooth_window_size=7,
        smooth_ploy=2,
        node_sample_distance=2,
        voxel_size=(1, 1, 1),
    ):
        segments = self.extract_segments(G, voxel_size=voxel_size)
        segments = sorted(segments, key=lambda x: x.length)
        for segment in segments:
            if segment.length > 3:
                node_coords = segment.node_coords
                if len(node_coords) > smooth_window_size:
                    node_coords = savgol_filter(node_coords, window_length=smooth_window_size, polyorder=smooth_ploy, axis=0)
                node_coords = resample_nodes_by_distance(
                    node_coords, node_sample_distance
                )
                node_ids = segment.node_ids
                for node_id in node_ids[1:-1]:
                    assert node_id in G
                    G.remove_node(node_id)
                pre_node_id = segment.node_ids[0]
                for i in range(1, len(node_coords)-1):
                    self.max_node_id+=1
                    node_id = self.max_node_id
                    G.add_node(node_id, coord=node_coords[i], ntype=0)
                    G.add_edge(pre_node_id, node_id)
                    pre_node_id = node_id
                G.add_edge(pre_node_id, segment.node_ids[-1])
        G = nx_clear_invalid_edges(G)
        return G