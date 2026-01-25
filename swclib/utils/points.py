import numpy as np

def cal_tree_point_angle(p1,p2,p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    v1 = p2 - p1
    v2 = p3 - p2
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    cos_angle = np.dot(v1, v2)
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
    return angle

def point_pair_distance(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))


def sample_points_from_point_pair(p0, p1, step=1):
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    L = np.linalg.norm(p1 - p0)
    if L < 1e-12:
        return np.array([p0])
    n_steps = int(L // step)
    ts = np.linspace(0, n_steps * step / L, n_steps + 1)
    points = p0 + np.outer(ts, (p1 - p0))
    return points

def cal_segment_length(points):
    points = np.asarray(points)
    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.sum(seg_lens)