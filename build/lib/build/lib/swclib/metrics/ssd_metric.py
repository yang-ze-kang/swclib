import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree

from swclib.data.swc import Swc


class SSDMetric:

    def __init__(
        self, min_distance=2.0, scale=(1.0, 1.0, 1.0)
    ):
        self.min_distance = min_distance
        self.scale = scale

    def run(self, gold, pred):
        if isinstance(gold, str) or isinstance(gold, Path):
            gold = Swc(gold)
        if isinstance(pred, str) or isinstance(pred, Path):
            pred = Swc(pred)
        assert isinstance(pred, Swc)
        assert isinstance(gold, Swc)
        gold.rescale(self.scale)
        pred.rescale(self.scale)

        gold.resample(self.min_distance)
        pred.resample(self.min_distance)
        gt_coords = gold.get_coords()
        pred_coords = pred.get_coords()

        if len(pred_coords)==0:
            return {
                "sd": None,
                "sd_gt2pred": None,
                "sd_pred2gt": None,
                "ssd": None,
                "ssd_gt2pred": None,
                "ssd_pred2gt": None,
            }

        # pred->gold
        tree_gt = cKDTree(gt_coords)
        dists_gt2pred, _ = tree_gt.query(pred_coords, k=1)
        sd_pred2gt = np.mean(dists_gt2pred)
        mask1 = dists_gt2pred > self.min_distance
        if mask1.sum() == 0:
            ssd_pred2gt = 0
        else:
            ssd_pred2gt = dists_gt2pred[mask1].mean()

        # pred->gold
        tree_pred = cKDTree(pred_coords)
        dists_pred2gt, _ = tree_pred.query(gt_coords, k=1)
        sd_gt2pred = np.mean(dists_pred2gt)
        mask2 = dists_pred2gt > self.min_distance
        if mask2.sum() == 0:
            ssd_gt2pred = 0
        else:
            ssd_gt2pred = dists_pred2gt[mask2].mean()
        
        # final
        sd = (sd_gt2pred + sd_pred2gt) / 2
        ssd = (ssd_gt2pred + ssd_pred2gt) / 2
        ssd_percent = (mask1.sum() + mask2.sum()) / (len(pred_coords) + len(gt_coords))

        res = {
            "sd": sd,
            "sd_gt2pred": sd_gt2pred,
            "sd_pred2gt": sd_pred2gt,
            "ssd": ssd,
            "ssd_gt2pred": ssd_gt2pred,
            "ssd_pred2gt": ssd_pred2gt,
            "ssd_percent": ssd_percent,
        }
        return res
