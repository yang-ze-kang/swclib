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
        use_category=False,
        min_fiber_length=5.0,
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
        self.use_category = use_category
        self.min_fiber_length = min_fiber_length
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

        # gt information
        gold_fibers = gold.get_fibers(self.only_from_soma, min_length=self.min_fiber_length)
        Ng = len(gold_fibers)

        # if no predicted fibers
        if pred.size() <= 1:
            res = {
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "num_gt": Ng,
                "num_pred": 0,
                "TP": 0,
                "FP": 0,
                "FN": Ng,
                "iou_matched": 0.0,
                "iou_all": 0.0,
                "ious": None,
                "matches": None,
                "FN_fiber_ids": [list(range(Ng))] if return_fibers else None,
                "gold_fibers": gold_fibers if return_fibers else None,
                "pred_fibers": None,
            }
            if self.use_category:
                res.update({
                    "axon_precision": 0.0,
                    "axon_recall": 0.0,
                    "axon_f1": 0.0,
                    "axon_TP": 0,
                    "axon_FP": 0,
                    "axon_FN": len([i for i in range(Ng) if gold_fibers[i][-1].ntype == 2]),
                    "iou_matched_axon": 0.0,
                    "iou_all_axon": 0.0,
                    "dendrite_precision": 0.0,
                    "dendrite_recall": 0.0,
                    "dendrite_f1": 0.0,
                    "dendrite_TP": 0,
                    "dendrite_FP": 0,
                    "dendrite_FN": len([i for i in range(Ng) if gold_fibers[i][-1].ntype == 3 or gold_fibers[i][-1].ntype == 4]),
                    "iou_matched_dendrite": 0.0,
                    "iou_all_dendrite": 0.0,
                })
            return res

        # align roots if needed
        if self.align_roots:
            roots = gold.get_roots(return_coords=True)
            pred = pred.align_roots(
                roots, align_roots_thredhold=self.align_roots_thredhold
            )

        # rescale to physical units
        gold.rescale(self.scale)
        pred.rescale(self.scale)

        
        pred_fibers = pred.get_fibers(self.only_from_soma, min_length=self.min_fiber_length)
        Np = len(pred_fibers)
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
        cost = 1 - ious
        row_ind, col_ind = linear_sum_assignment(cost)
        matches,ious_matched = [], []
        for r, c in zip(row_ind, col_ind):
            ious_matched.append(ious[r, c])
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
            "iou_matched": sum(ious_matched) / len(ious_matched) if len(ious_matched) > 0 else 0.0,
            "iou_all": sum(ious_matched) / max(Ng, Np) if max(Ng, Np) > 0 else 0.0,
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
        # Subdivide by fiber types
        if self.use_category:
            assert self.with_direction == True, "use_category requires with_direction to be True"
            pred_axon_ids = [j for j in range(Np) if pred_fibers[j][-1].ntype == 2]
            pred_dendrite_ids = [j for j in range(Np) if pred_fibers[j][-1].ntype == 3 or pred_fibers[j][-1].ntype == 4]
            gt_axon_ids = [i for i in range(Ng) if gold_fibers[i][-1].ntype == 2]
            gt_dendrite_ids = [i for i in range(Ng) if gold_fibers[i][-1].ntype == 3 or gold_fibers[i][-1].ntype == 4]
            axon_ious = ious[np.ix_(gt_axon_ids, pred_axon_ids)]
            dendrite_ious = ious[np.ix_(gt_dendrite_ids, pred_dendrite_ids)]
            # axon
            cost = 1 - axon_ious
            row_ind, col_ind = linear_sum_assignment(cost)
            matches, iou_matched_axon = [], []
            for r, c in zip(row_ind, col_ind):
                iou_matched_axon.append(axon_ious[r, c])
                if axon_ious[r, c] >= self.iou_threshold:
                    matches.append((r, c, axon_ious[r, c]))
            axon_TP = len(matches)
            axon_FP = len(pred_axon_ids) - axon_TP
            axon_FN = len(gt_axon_ids) - axon_TP
            axon_precision = axon_TP / (axon_TP + axon_FP) if (axon_TP + axon_FP) > 0 else 0.0
            axon_recall = axon_TP / (axon_TP + axon_FN) if (axon_TP + axon_FN) > 0 else 0.0
            axon_f1 = 2 * axon_precision * axon_recall / (axon_precision + axon_recall + self.eps)

            # dendrite
            cost = 1 - dendrite_ious
            row_ind, col_ind = linear_sum_assignment(cost)
            matches, iou_matched_dendrite = [], []
            for r, c in zip(row_ind, col_ind):
                iou_matched_dendrite.append(dendrite_ious[r, c])
                if dendrite_ious[r, c] >= self.iou_threshold:
                    matches.append((r, c, dendrite_ious[r, c]))
            dendrite_TP = len(matches)
            dendrite_FP = len(pred_dendrite_ids) - dendrite_TP
            dendrite_FN = len(gt_dendrite_ids) - dendrite_TP
            dendrite_precision = dendrite_TP / (dendrite_TP + dendrite_FP) if (dendrite_TP + dendrite_FP) > 0 else 0.0
            dendrite_recall = dendrite_TP / (dendrite_TP + dendrite_FN) if (dendrite_TP + dendrite_FN) > 0 else 0.0
            dendrite_f1 = 2 * dendrite_precision * dendrite_recall / (dendrite_precision + dendrite_recall + self.eps)
            res.update({
                "axon_precision": axon_precision,
                "axon_recall": axon_recall,
                "axon_f1": axon_f1,
                "axon_TP": axon_TP,
                "axon_FP": axon_FP,
                "axon_FN": axon_FN,
                "iou_matched_axon": sum(iou_matched_axon) / len(iou_matched_axon) if len(iou_matched_axon) > 0 else 0.0,
                "iou_all_axon": sum(iou_matched_axon) / max(len(gt_axon_ids), len(pred_axon_ids)) if max(len(gt_axon_ids), len(pred_axon_ids)) > 0 else 0.0,
                "dendrite_precision": dendrite_precision,
                "dendrite_recall": dendrite_recall,
                "dendrite_f1": dendrite_f1,
                "dendrite_TP": dendrite_TP,
                "dendrite_FP": dendrite_FP,
                "dendrite_FN": dendrite_FN,
                "iou_matched_dendrite": sum(iou_matched_dendrite) / len(iou_matched_dendrite) if len(iou_matched_dendrite) > 0 else 0.0,
                "iou_all_dendrite": sum(iou_matched_dendrite) / max(len(gt_dendrite_ids), len(pred_dendrite_ids)) if max(len(gt_dendrite_ids), len(pred_dendrite_ids)) > 0 else 0.0,
            })
        return res
