import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path

from swclib.data.swc_node import nodes2coords
from swclib.data.swc_forest import SwcForest


class KeypointMetric:

    def __init__(
        self, keypoint_types=["branch", "leaf"], threshold_dis=5, scale=(0.35, 0.35, 1), use_category=False
    ):
        self.keypoint_types = keypoint_types
        self.threshold_dis = threshold_dis
        self.scale = scale
        self.use_category = use_category

    def _calculate(self, gold_coords, pred_coords):
        Ng = len(gold_coords)
        Np = len(pred_coords)
        if Np == 0:
            TP = 0
        else:
            cost = np.linalg.norm(
                pred_coords[:, None, :] - gold_coords[None, :, :], axis=2
            )
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_dist = cost[row_ind, col_ind]
            valid = matched_dist < self.threshold_dis
            TP = valid.sum()
        FP = Np - TP
        FN = Ng - TP

        precision = TP / Np if Np > 0 else 0
        recall = TP / Ng if Ng > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0
        )
        return Ng, Np, precision, recall, f1, TP, FP, FN

    def run(self, gold, pred):
        if isinstance(gold, str) or isinstance(gold, Path):
            gold = SwcForest(gold)
        if isinstance(pred, str) or isinstance(pred, Path):
            pred = SwcForest(pred)
        assert isinstance(pred, SwcForest)
        assert isinstance(pred, SwcForest)
        gold.rescale(self.scale)
        pred.rescale(self.scale)

        gold_keypoints, pred_keypoints = [], []
        if "branch" in self.keypoint_types:
            gold_keypoints.extend(gold.get_branch_nodes())
            pred_keypoints.extend(pred.get_branch_nodes())
        if "leaf" in self.keypoint_types:
            gold_keypoints.extend(gold.get_leaf_nodes())
            pred_keypoints.extend(pred.get_leaf_nodes())
        gold_keypoints = nodes2coords(gold_keypoints)
        pred_keypoints = nodes2coords(pred_keypoints)

        Ng, Np, precision, recall, f1, TP, FP, FN = self._calculate(gold_keypoints, pred_keypoints)
        res = {
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "num_gt": Ng,
            "num_pred": Np,
        }
        if self.use_category:
            gold_branchs = gold.get_branch_nodes()
            pred_branchs = pred.get_branch_nodes()
            goldp = nodes2coords(gold_branchs)
            predp = nodes2coords(pred_branchs)
            Ng, Np, precision, recall, f1, TP, FP, FN = self._calculate(goldp, predp)
            res.update({
                "branch_recall": recall,
                "branch_precision": precision,
                "branch_f1": f1,
                "branch_TP": TP,
                "branch_FP": FP,
                "branch_FN": FN,
                "branch_num_gt": Ng,
                "branch_num_pred": Np,
            })
            gold_leafs = gold.get_leaf_nodes()
            pred_leafs = pred.get_leaf_nodes()
            goldp = nodes2coords(gold_leafs)
            predp = nodes2coords(pred_leafs)
            Ng, Np, precision, recall, f1, TP, FP, FN = self._calculate(goldp, predp)
            res.update({
                "leaf_recall": recall,
                "leaf_precision": precision,
                "leaf_f1": f1,
                "leaf_TP": TP,
                "leaf_FP": FP,
                "leaf_FN": FN,
                "leaf_num_gt": Ng,
                "leaf_num_pred": Np,
            })

        return res


if __name__ == "__main__":
    metric = KeypointMetric(threshold_dis=5, scale=(0.35, 0.35, 1))
    path1 = "/gpfs-flash/hulab/yangzekang/neuron/neuron-seg/output_r0.1.swc"
    path2 = "/gpfs-flash/hulab/yangzekang/neuron/data/guolab/etv133_block_swc_yzk/annotation_idx_3_6539_1105_7627.swc"
    res = metric.run(path2, path1)
    print(res)
