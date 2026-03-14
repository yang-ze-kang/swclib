import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path

from swclib.data.swc_node import nodes2coords
from swclib.data.swc_forest import SwcForest


class KeypointMetric:

    def __init__(
        self, keypoint_types=["branch", "leaf"], threshold_dis=5, scale=(0.35, 0.35, 1)
    ):
        self.keypoint_types = keypoint_types
        self.threshold_dis = threshold_dis
        self.scale = scale

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

        Ng = len(gold_keypoints)
        Np = len(pred_keypoints)
        
        if Np == 0:
            TP = 0
        else:
            cost = np.linalg.norm(
                pred_keypoints[:, None, :] - gold_keypoints[None, :, :], axis=2
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
        res = {
            "recall": recall,
            "precision": precision,
            "f1_score": f1,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "num_gt": Ng,
            "num_pred": Np,
        }
        return res


if __name__ == "__main__":
    metric = KeypointMetric(threshold_dis=5, scale=(0.35, 0.35, 1))
    path1 = "/gpfs-flash/hulab/yangzekang/neuron/neuron-seg/output_r0.1.swc"
    path2 = "/gpfs-flash/hulab/yangzekang/neuron/data/guolab/etv133_block_swc_yzk/annotation_idx_3_6539_1105_7627.swc"
    res = metric.run(path2, path1)
    print(res)
