from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


def resample_nodes_by_distance(nodes, distance=2):
    nodes = np.asarray(nodes)
    diffs = np.diff(nodes, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    target_lens = np.arange(0, total_len, distance)
    new_nodes = []
    for t in target_lens:
        idx = np.searchsorted(cum_len, t)
        if idx == 0:
            new_nodes.append(nodes[0])
        else:
            ratio = (t - cum_len[idx - 1]) / seg_lens[idx - 1]
            p = nodes[idx - 1] + ratio * diffs[idx - 1]
            new_nodes.append(p)
    new_nodes.append(nodes[-1])
    return np.array(new_nodes)


class SwcFiber:

    def __init__(self):
        self.nodes = []

        self.resampled_coords = None
        self.last_sampled_dist = -1
        self.cktree = None

    def __len__(self):
        return len(self.nodes)
    
    def __eq__(self, fiber):
        if self.nodes==fiber.nodes:
            return True
        elif self.nodes[::-1]==fiber.nodes:
            return True
        return False
    
    def __getitem__(self, idx):
        return self.nodes[idx]

    @property
    def coords(self):
        return np.array([node.coord for node in self.nodes])

    @property
    def length(self):
        points = self.coords
        diffs = points[1:] - points[:-1]
        seg_lengths = np.linalg.norm(diffs, axis=1)
        return float(seg_lengths.sum())
    
    @property
    def center(self):
        return self.coords.mean(0)

    def append(self, node):
        self.nodes.append(node)

    def reverse(self):
        self.nodes = self.nodes[::-1]
        return self

    def copy(self):
        new_fiber = SwcFiber()
        new_fiber.nodes = self.nodes[:]
        return new_fiber
    
    def get_nearest_node(self, point, return_dist=False):
        coords = self.coords
        tree = self.cahce_cKDTree(coords)
        dist, idx = tree.query(point)
        if return_dist:
            return self.nodes[idx], dist
        else:
            return self.nodes[idx]

    def cahce_resample_coords_by_distance(self, dist_sample):
        if self.resampled_coords is not None and self.last_sampled_dist == dist_sample:
            return self.resampled_coords
        self.resampled_coords = resample_nodes_by_distance(self.coords, dist_sample)
        self.last_sampled_dist = dist_sample
        return self.resampled_coords

    def cahce_cKDTree(self, coords):
        if self.cktree is None:
            self.cktree = cKDTree(coords)
        return self.cktree

    def cal_iou(
        self,
        fiber: "SwcFiber",
        dist_sample=1.0,
        dist_threshold=3.0,
        min_iou_thres=0.5,
        eps=1e-7,
    ):
        l1 = self.length
        l2 = fiber.length
        if l1 / l2 < min_iou_thres or l2 / l1 < min_iou_thres:
            return 0.0
        coords1 = self.coords
        coords2 = fiber.coords
        coords1 = self.cahce_resample_coords_by_distance(dist_sample)
        coords2 = resample_nodes_by_distance(coords2, dist_sample)
        l1 = (coords1.shape[0] - 1) * dist_sample
        l2 = (coords2.shape[0] - 1) * dist_sample
        tree1 = self.cahce_cKDTree(coords1)
        tree2 = cKDTree(coords2)
        # fiber1->fiber2
        midpoints = (coords1[:-1] + coords1[1:]) * 0.5
        dists, _ = tree2.query(midpoints)
        mask = dists <= dist_threshold
        overlap1 = np.sum(mask) * dist_sample
        # fiber2->fiber1
        midpoints = (coords2[:-1] + coords2[1:]) * 0.5
        dists, _ = tree1.query(midpoints)
        mask = dists <= dist_threshold
        overlap2 = np.sum(mask) * dist_sample
        loverlap = (overlap1 + overlap2) / 2.0
        lunion = l1 + l2 - loverlap
        return loverlap / (lunion + eps)
    
    def is_sub_fiber_of(self, fiber: "SwcFiber", dist_sample=1.0, dist_threshold=3.0, same_fiber_iou_thres=0.9):
        l1 = self.length
        l2 = fiber.length
        if l1 > l2  * 1.5:
            return False
        coords1 = self.coords
        coords2 = fiber.coords
        coords1 = self.cahce_resample_coords_by_distance(dist_sample)
        coords2 = resample_nodes_by_distance(coords2, dist_sample)
        l1 = (coords1.shape[0] - 1) * dist_sample
        l2 = (coords2.shape[0] - 1) * dist_sample
        tree1 = self.cahce_cKDTree(coords1)
        tree2 = cKDTree(coords2)
        # fiber1->fiber2
        midpoints = (coords1[:-1] + coords1[1:]) * 0.5
        dists, _ = tree2.query(midpoints)
        mask = dists <= dist_threshold
        overlap1 = np.sum(mask) * dist_sample
        # fiber2->fiber1
        midpoints = (coords2[:-1] + coords2[1:]) * 0.5
        dists, _ = tree1.query(midpoints)
        mask = dists <= dist_threshold
        overlap2 = np.sum(mask) * dist_sample
        loverlap = (overlap1 + overlap2) / 2.0
        if loverlap / (l1 + 1e-7) > same_fiber_iou_thres:
            return True
        return False
    
    def to_str_list(self, scale=(1.0, 1.0, 1.0)):
        swc_str = []
        for node in self.nodes:
            if node.is_virtual():
                continue
            swc_str.append(node.to_swc_str(scale=scale))
        return "".join(swc_str)
