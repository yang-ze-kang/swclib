import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path

from swclib.data.swc import Swc
from swclib.data.swc_forest import SwcForest
from swclib.data.swc_node import nodes2coords


class PointMetric:

    def __init__(
        self,
        dist_threshold=4,
        scale=(1.0, 1.0, 1.0),
        resample_step=2.0,
    ):
        self.dist_threshold = dist_threshold
        self.scale = scale
        self.resample_step = resample_step

    def _load_swc_forest(self, swc):
        if isinstance(swc, str) or isinstance(swc, Path):
            if self.resample_step is not None:
                swc = Swc(swc)
                swc.resample(self.resample_step)
            swc = SwcForest(swc)
        return swc

    def _get_coords(self, swc_forest):
        coords = nodes2coords(swc_forest.get_node_list())
        if coords.size == 0:
            return np.empty((0, 3), dtype=float)
        return coords.astype(float)

    def _calculate(self, gold_coords, pred_coords, return_points=False):
        Ng = len(gold_coords)
        Np = len(pred_coords)

        if Ng == 0 or Np == 0:
            TP = 0
            FP = Np
            FN = Ng
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            mes = 0.0
            res = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "f1_score": f1,
                "MES": mes,
                "num_gt": Ng,
                "num_pred": Np,
                "TP": TP,
                "TP_pred": TP,
                "TP_gold": TP,
                "FP": FP,
                "FN": FN,
                "S_G": Ng,
                "S_hit_pred": TP,
                "S_miss": FN,
                "S_extra": FP,
            }
            if return_points:
                res.update({
                    "distances": None,
                    "matches": None,
                    "FN_point_ids": list(range(Ng)),
                    "FP_point_ids": list(range(Np)),
                    "gold_coords": gold_coords,
                    "pred_coords": pred_coords,
                })
            return res

        gold_tree = cKDTree(gold_coords)
        pred_tree = cKDTree(pred_coords)

        pred_to_gold_dist, pred_to_gold_idx = gold_tree.query(pred_coords)
        gold_to_pred_dist, gold_to_pred_idx = pred_tree.query(gold_coords)

        pred_matched = pred_to_gold_dist < self.dist_threshold
        gold_matched = gold_to_pred_dist < self.dist_threshold

        S_G = Ng
        S_miss = int((~gold_matched).sum())
        S_extra = int((~pred_matched).sum())

        TP_pred = int(pred_matched.sum())
        TP_gold = int(gold_matched.sum())
        TP = TP_pred
        FP = S_extra
        FN = S_miss

        precision = TP_pred / Np if Np > 0 else 0.0
        recall = TP_gold / S_G if S_G > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        mes = (
            (S_G - S_miss) / (S_G + S_extra)
            if (S_G + S_extra) > 0
            else 0.0
        )

        res = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "MES": mes,
            "num_gt": Ng,
            "num_pred": Np,
            "TP": TP,
            "TP_pred": TP_pred,
            "TP_gold": TP_gold,
            "FP": FP,
            "FN": FN,
            "S_G": S_G,
            "S_hit_pred": TP_pred,
            "S_miss": S_miss,
            "S_extra": S_extra,
        }
        if return_points:
            matches = [
                (int(gt_idx), int(pred_idx), float(dist))
                for pred_idx, (gt_idx, dist, is_matched) in enumerate(
                    zip(pred_to_gold_idx, pred_to_gold_dist, pred_matched)
                )
                if is_matched
            ]
            res.update({
                "pred_to_gold_dist": pred_to_gold_dist,
                "pred_to_gold_idx": pred_to_gold_idx,
                "gold_to_pred_dist": gold_to_pred_dist,
                "gold_to_pred_idx": gold_to_pred_idx,
                "matches": matches,
                "FN_point_ids": [i for i in range(Ng) if not gold_matched[i]],
                "FP_point_ids": [i for i in range(Np) if not pred_matched[i]],
                "gold_coords": gold_coords,
                "pred_coords": pred_coords,
            })
        return res

    def run(self, gold, pred, return_points=False):
        gold = self._load_swc_forest(gold)
        pred = self._load_swc_forest(pred)
        assert isinstance(gold, SwcForest)
        assert isinstance(pred, SwcForest)

        gold.rescale(self.scale)
        pred.rescale(self.scale)

        gold_coords = self._get_coords(gold)
        pred_coords = self._get_coords(pred)
        return self._calculate(gold_coords, pred_coords, return_points=return_points)
