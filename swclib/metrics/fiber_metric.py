import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path

from swclib.data.swc_tree import SwcTree
from swclib.utils.points import point_pair_distance


class FiberMetric:

    def __init__(
        self,
        iou_threshold=0.8,
        dist_threshold=5,
        dist_sample=1.0,
        scale=(1.0, 1.0, 1.0),
        only_from_soma=False,
    ):
        self.iou_threshold = iou_threshold
        self.dist_threshold = dist_threshold
        self.dist_sample = dist_sample
        self.scale = scale
        self.only_from_soma = only_from_soma

    def run(self, gold, pred, skip_center_dist=100, return_fibers=False):
        if isinstance(gold, str) or isinstance(gold, Path):
            gold = SwcTree(gold)
        if isinstance(pred, str) or isinstance(pred, Path):
            pred = SwcTree(pred)
        assert isinstance(pred, SwcTree)
        assert isinstance(pred, SwcTree)
        gold.rescale(self.scale)
        pred.rescale(self.scale)

        gold_fibers = gold.get_fibers(self.only_from_soma)
        pred_fibers = pred.get_fibers(self.only_from_soma)
        Ng, Np = len(gold_fibers), len(pred_fibers)
        ious = np.zeros((Ng, Np))
        for i in range(Ng):
            for j in range(Np):
                if (
                    point_pair_distance(gold_fibers[i].center, pred_fibers[j].center)
                    > skip_center_dist
                ):  # early stop
                    ious[i, j] = 0.1
                    continue
                ious[i, j] = gold_fibers[i].cal_iou(
                    pred_fibers[j],
                    dist_sample=self.dist_sample,
                    dist_threshold=self.dist_threshold,
                )
                if ious[i, j] > 0.98:
                    break
        cost = 1 - ious
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if ious[r, c] >= self.iou_threshold:
                matches.append((r, c, ious[r, c]))
        TP = len(matches)
        FP = Np - TP
        FN = Ng - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        res = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "num_gt": Ng,
            "num_pred": Np,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "ious": ious if return_fibers else None,
            "matches": matches if return_fibers else None,
            "FN_fiber_ids": (
                [i for i in range(Ng) if i not in [m[0] for m in matches]]
                if return_fibers
                else None
            ),
            "gold_fibers": gold_fibers if return_fibers else None,
            "pred_fibers": pred_fibers if return_fibers else None,
        }
        return res
