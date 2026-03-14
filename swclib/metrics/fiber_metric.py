import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
from tqdm import tqdm

from swclib.data.swc import Swc
from swclib.data.swc_forest import SwcForest
from swclib.utils.points import point_pair_distance


class FiberMetric:

    def __init__(
        self,
        iou_threshold=0.8,
        dist_threshold=5,
        dist_sample=1.0,
        align_roots=False,
        align_roots_thredhold=20.0,
        scale=(1.0, 1.0, 1.0),
        resample_step=2.0,
        only_from_soma=False,
        with_direction=False,
        eps=1e-6,
    ):
        self.iou_threshold = iou_threshold
        self.dist_threshold = dist_threshold
        self.dist_sample = dist_sample
        self.align_roots = align_roots
        self.align_roots_thredhold = align_roots_thredhold
        self.scale = scale
        self.resample_step = resample_step
        self.only_from_soma = only_from_soma
        self.with_direction = with_direction
        self.eps = eps

    def run(self, gold, pred, skip_center_dist=100, return_fibers=False, verbose=False):
        if isinstance(gold, str) or isinstance(gold, Path):
            if self.resample_step!=None:
                gold = Swc(gold)
                gold.resample(self.resample_step)
            gold = SwcForest(gold)
        if isinstance(pred, str) or isinstance(pred, Path):
            if self.resample_step!=None:
                pred = Swc(pred)
                pred.resample(self.resample_step)
            pred = SwcForest(pred)
        assert isinstance(pred, SwcForest)
        assert isinstance(gold, SwcForest)
        # align roots if needed
        if self.align_roots:
            roots = gold.get_roots(return_coords=True)
            pred = pred.align_roots(
                roots, align_roots_thredhold=self.align_roots_thredhold
            )

        # rescale to physical units
        gold.rescale(self.scale)
        pred.rescale(self.scale)

        gold_fibers = gold.get_fibers(self.only_from_soma)
        pred_fibers = pred.get_fibers(self.only_from_soma)
        Ng, Np = len(gold_fibers), len(pred_fibers)
        fiber_length_gt = [f.length for f in gold_fibers]
        fiber_length_pred = [f.length for f in pred_fibers]
        ious = np.zeros((Ng, Np))
        for i in tqdm(range(Ng), desc="Calculating fiber IoUs", disable=not verbose):
            for j in range(Np):
                if (
                    fiber_length_gt[i] / (fiber_length_pred[j] + self.eps) < 0.5
                    and fiber_length_pred[j] / (fiber_length_gt[i] + self.eps) < 0.5
                ): # early stop by max length
                    ious[i, j] = min(fiber_length_gt[i], fiber_length_pred[j]) / max(fiber_length_gt[i], fiber_length_pred[j])
                    continue
                if self.with_direction: # early stop by endpoints distance
                    max_length = max(fiber_length_gt[i], fiber_length_pred[j])
                    if gold_fibers[i][0].distance(pred_fibers[j][0]) > 0.5 * max_length or \
                        gold_fibers[i][-1].distance(pred_fibers[j][-1]) > 0.5 * max_length:
                        ious[i, j] = 0.01
                        continue
                if (
                    point_pair_distance(gold_fibers[i].center, pred_fibers[j].center)
                    > skip_center_dist
                ):  # early stop by center distance
                    ious[i, j] = 0.01
                    continue
                ious[i, j] = gold_fibers[i].cal_iou(
                    pred_fibers[j],
                    dist_sample=self.dist_sample,
                    dist_threshold=self.dist_threshold,
                )
                # if ious[i, j] > 0.98:
                #     break
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
        # Subdivide by fiber types
        ftypes = [fiber[-1].ntype for fiber in gold_fibers]
        tp_a, tp_d = 0, 0
        toatl_d = sum([1 for t in ftypes if t == 3 or t == 4])
        toatl_a = sum([1 for t in ftypes if t == 2])
        for i,j,iou in matches:
            if ftypes[i] == 2:
                tp_a += 1
            elif ftypes[i] == 3 or ftypes[i] == 4:
                tp_d += 1
        recall_a = tp_a / (toatl_a + self.eps)
        recall_d = tp_d / (toatl_d + self.eps)
        mean_iou = np.mean(np.max(ious, axis=0))
        res = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "num_gt": Ng,
            "num_pred": Np,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "axon_recall": recall_a,
            "dendrite_recall": recall_d,
            "mean_iou": mean_iou,
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
