"""DIADEM tree reconstruction metric."""

from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from swclib.data.swc import Swc
from swclib.data.swc_forest import SwcForest


def _forest(value, step):
    if isinstance(value, (str, Path)):
        if step is None:
            # Avoid holding both an Swc dictionary and an SwcForest for large
            # whole-neuron files.
            value = SwcForest(value)
        else:
            value = Swc(value)
            value.resample(step)
            value = SwcForest(value)
    if not isinstance(value, SwcForest):
        raise TypeError("gold and pred must be SWC paths or SwcForest objects")
    return value


def _samples(forest, spacing):
    points, weights = [], []
    for node in forest.get_node_list():
        if node.is_root():
            continue
        a = np.asarray(node.parent.coord, dtype=float)
        b = np.asarray(node.coord, dtype=float)
        length = float(np.linalg.norm(b - a))
        if length == 0:
            continue
        n = max(1, int(np.ceil(length / spacing)))
        for i in range(n):
            t = (i + 0.5) / n
            points.append(a * (1 - t) + b * t)
            weights.append(length / n)
    return (np.asarray(points) if points else np.empty((0, 3)),
            np.asarray(weights) if weights else np.empty(0))


class DIADEMMetric:
    def __init__(self, distance_threshold=2.0, sample_step=1.0,
                 scale=(1.0, 1.0, 1.0), resample_step=None):
        self.distance_threshold = float(distance_threshold)
        self.sample_step = float(sample_step)
        self.scale = tuple(scale)
        self.resample_step = resample_step

    def run(self, gold, pred, return_samples=False):
        gold, pred = _forest(gold, self.resample_step), _forest(pred, self.resample_step)
        gold.rescale(self.scale); pred.rescale(self.scale)
        gp, gw = _samples(gold, self.sample_step)
        pp, pw = _samples(pred, self.sample_step)
        if len(gp) and len(pp):
            gd = cKDTree(gp).query(pp, k=1)[0]
            pd = cKDTree(pp).query(gp, k=1)[0]
            hit_g, hit_p = pd <= self.distance_threshold, gd <= self.distance_threshold
        else:
            hit_g = np.zeros(len(gp), dtype=bool); hit_p = np.zeros(len(pp), dtype=bool)
        gl, pl = float(gw.sum()), float(pw.sum())
        covered, matched = float(gw[hit_g].sum()), float(pw[hit_p].sum())
        recall = covered / gl if gl else (1.0 if pl == 0 else 0.0)
        precision = matched / pl if pl else (1.0 if gl == 0 else 0.0)
        score = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        result = {"diadem": score, "score": score, "precision": precision,
                  "recall": recall, "gold_length": gl, "pred_length": pl,
                  "matched_length": covered, "extra_length": pl - matched,
                  "missing_length": gl - covered}
        if return_samples:
            result.update({"gold_points": gp, "pred_points": pp,
                           "gold_hit": hit_g, "pred_hit": hit_p})
        return result

    __call__ = run


DiademMetric = DIADEMMetric
__all__ = ["DIADEMMetric", "DiademMetric"]
