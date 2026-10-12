from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LocalBendResult:
    """Local bend measurements evaluated on a uniformly sampled fiber.

    All coordinates and distances are in the physical units produced by
    multiplying the input coordinates by ``scale``. Values outside the valid
    two-sided window are ``NaN`` and identified by ``valid_mask``.
    """

    coords: np.ndarray
    arc_lengths: np.ndarray
    bend_scores: np.ndarray
    confidence: np.ndarray
    angles: np.ndarray
    mean_curvature: np.ndarray
    valid_mask: np.ndarray


def _empty_bend_result(coords: np.ndarray) -> LocalBendResult:
    size = len(coords)
    values = np.full(size, np.nan, dtype=float)
    if size == 0:
        arc_lengths = np.empty(0, dtype=float)
    else:
        arc_lengths = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))]
        )
    return LocalBendResult(
        coords=coords,
        arc_lengths=arc_lengths,
        bend_scores=values.copy(),
        confidence=values.copy(),
        angles=values.copy(),
        mean_curvature=values.copy(),
        valid_mask=np.zeros(size, dtype=bool),
    )


def _fit_pca_direction(
    points: np.ndarray,
    robust: bool,
    robust_iterations: int,
    huber_delta: float,
    eps: float,
) -> Tuple[np.ndarray, float]:
    """Fit an oriented line direction and return its line-likeness."""
    weights = np.ones(len(points), dtype=float)
    eigenvalues = None
    direction = None

    iteration_count = robust_iterations if robust else 1
    for iteration in range(iteration_count):
        weight_sum = weights.sum()
        if weight_sum <= eps:
            break

        center = np.sum(points * weights[:, None], axis=0) / weight_sum
        centered = points - center
        covariance = (centered * weights[:, None]).T @ centered / weight_sum
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        direction = eigenvectors[:, -1]

        if not robust or iteration == iteration_count - 1:
            break

        projection = centered @ direction
        residuals = np.linalg.norm(
            centered - projection[:, None] * direction[None, :], axis=1
        )
        residual_median = np.median(residuals)
        mad = np.median(np.abs(residuals - residual_median))
        robust_scale = max(1.4826 * mad, residual_median, eps)
        cutoff = huber_delta * robust_scale
        weights = np.ones_like(residuals)
        outliers = residuals > cutoff
        weights[outliers] = cutoff / np.maximum(residuals[outliers], eps)

    if direction is None or eigenvalues is None or eigenvalues[-1] <= eps:
        return np.full(points.shape[1], np.nan), np.nan

    reference = points[-1] - points[0]
    if np.linalg.norm(reference) <= eps:
        return np.full(points.shape[1], np.nan), np.nan
    if np.dot(direction, reference) < 0:
        direction = -direction

    confidence = (eigenvalues[-1] - eigenvalues[-2]) / max(
        eigenvalues[-1], eps
    )
    return direction, float(np.clip(confidence, 0.0, 1.0))


def local_bend_scores(
    coords,
    window: float = 5.0,
    sample_step: float = 1.0,
    scale: Sequence[float] = (1.0, 1.0, 1.0),
    min_points: int = 4,
    robust: bool = True,
    robust_iterations: int = 3,
    huber_delta: float = 1.5,
    eps: float = 1e-8,
) -> LocalBendResult:
    """Measure local bending along an ordered SWC fiber.

    The fiber is first converted to physical coordinates and uniformly
    resampled. At each valid sample, separate PCA lines are fitted to the left
    and right half-windows. Their consistently oriented directions ``t_left``
    and ``t_right`` define the bounded bend score

    ``(1 - dot(t_left, t_right)) / 2``.

    The score is zero for a straight fiber, 0.5 for a 90-degree turn, and one
    for a 180-degree reversal. The optional iteratively reweighted PCA reduces
    the influence of transverse coordinate outliers. ``mean_curvature`` is the
    direction-change angle divided by ``window`` and has inverse-distance
    units. Endpoints without a complete two-sided window remain invalid.

    Args:
        coords: Ordered coordinates with shape ``(N, 3)``.
        window: Physical arc length used on each side of a sample.
        sample_step: Physical spacing for uniform resampling.
        scale: Per-axis input-to-physical scale in ``(x, y, z)`` order.
        min_points: Minimum number of resampled points in each half-window,
            including the center sample.
        robust: Use Huber-style iteratively reweighted PCA when true.
        robust_iterations: Number of weighted PCA iterations.
        huber_delta: Huber residual cutoff in robust-scale units.
        eps: Numerical tolerance.

    Returns:
        A :class:`LocalBendResult` aligned with the resampled coordinates.
    """
    if not np.isfinite(window) or window <= 0:
        raise ValueError("window must be finite and positive")
    if not np.isfinite(sample_step) or sample_step <= 0:
        raise ValueError("sample_step must be finite and positive")
    if not isinstance(min_points, (int, np.integer)) or min_points < 2:
        raise ValueError("min_points must be an integer of at least 2")
    if (
        not isinstance(robust_iterations, (int, np.integer))
        or robust_iterations < 1
    ):
        raise ValueError("robust_iterations must be a positive integer")
    if not np.isfinite(huber_delta) or huber_delta <= 0:
        raise ValueError("huber_delta must be finite and positive")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if window + eps < (min_points - 1) * sample_step:
        raise ValueError(
            "window is too short for min_points at the requested sample_step"
        )

    points = np.asarray(coords, dtype=float)
    if points.size == 0:
        points = np.empty((0, 3), dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3)")
    if not np.all(np.isfinite(points)):
        raise ValueError("coords must contain only finite values")

    scale_array = np.asarray(scale, dtype=float)
    if scale_array.shape != (3,):
        raise ValueError("scale must contain three values in (x, y, z) order")
    if not np.all(np.isfinite(scale_array)) or np.any(scale_array <= 0):
        raise ValueError("scale values must be finite and positive")

    physical_points = points * scale_array[None, :]
    if len(physical_points) > 1:
        keep = np.concatenate(
            [[True], np.linalg.norm(np.diff(physical_points, axis=0), axis=1) > eps]
        )
        physical_points = physical_points[keep]

    if len(physical_points) < 2:
        return _empty_bend_result(physical_points)

    sampled = resample_nodes_by_distance(
        physical_points, distance=sample_step, eps=eps
    )
    result = _empty_bend_result(sampled)
    if len(sampled) < 2 or result.arc_lengths[-1] + eps < 2 * window:
        return result

    bend_scores = result.bend_scores.copy()
    confidence = result.confidence.copy()
    angles = result.angles.copy()
    mean_curvature = result.mean_curvature.copy()
    valid_mask = result.valid_mask.copy()
    arc_lengths = result.arc_lengths
    total_length = arc_lengths[-1]

    for index, center_length in enumerate(arc_lengths):
        if center_length + eps < window or total_length - center_length + eps < window:
            continue

        left_start = np.searchsorted(
            arc_lengths, center_length - window - eps, side="left"
        )
        right_stop = np.searchsorted(
            arc_lengths, center_length + window + eps, side="right"
        )
        left_points = sampled[left_start : index + 1]
        right_points = sampled[index:right_stop]
        if len(left_points) < min_points or len(right_points) < min_points:
            continue

        left_direction, left_confidence = _fit_pca_direction(
            left_points, robust, robust_iterations, huber_delta, eps
        )
        right_direction, right_confidence = _fit_pca_direction(
            right_points, robust, robust_iterations, huber_delta, eps
        )
        if not np.all(np.isfinite(left_direction)) or not np.all(
            np.isfinite(right_direction)
        ):
            continue

        cosine = float(
            np.clip(np.dot(left_direction, right_direction), -1.0, 1.0)
        )
        sine = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
        angle = float(np.arctan2(sine, cosine))

        bend_scores[index] = np.clip((1.0 - cosine) * 0.5, 0.0, 1.0)
        confidence[index] = min(left_confidence, right_confidence)
        angles[index] = angle
        mean_curvature[index] = angle / window
        valid_mask[index] = True

    return LocalBendResult(
        coords=sampled,
        arc_lengths=arc_lengths,
        bend_scores=bend_scores,
        confidence=confidence,
        angles=angles,
        mean_curvature=mean_curvature,
        valid_mask=valid_mask,
    )


def resample_nodes_by_distance(nodes, distance=2, eps=1e-8):
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

    # Avoid duplicated endpoint caused by floating-point precision
    if len(new_nodes) == 0 or np.linalg.norm(new_nodes[-1] - nodes[-1]) > eps:
        new_nodes.append(nodes[-1])

    return np.array(new_nodes)


class SwcFiber:

    def __init__(self):
        self.nodes = []
        self._length = None

        self.resampled_coords = None
        self.last_sampled_dist = -1
        self.cktree = None
        self._resample_cache = None

    def _invalidate_cache(self):
        self._length = None
        self.resampled_coords = None
        self.last_sampled_dist = -1
        self.cktree = None
        self._resample_cache = None

    def __len__(self):
        return len(self.nodes)

    def __eq__(self, fiber):
        if self.nodes == fiber.nodes:
            return True
        elif self.nodes[::-1] == fiber.nodes:
            return True
        return False

    def __getitem__(self, idx):
        return self.nodes[idx]

    @property
    def coords(self):
        return np.array([node.coord for node in self.nodes])

    @property
    def length(self, update=True):
        if len(self.nodes) < 2:
            return 0.0
        if self._length is not None and not update:
            return self._length
        points = self.coords
        diffs = points[1:] - points[:-1]
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self._length = float(seg_lengths.sum())
        return self._length

    @property
    def center(self):
        return self.coords.mean(0)

    def append(self, node):
        self.nodes.append(node)
        self._invalidate_cache()

    def pop(self):
        node = self.nodes.pop()
        self._invalidate_cache()
        return node

    def reverse(self):
        self.nodes = self.nodes[::-1]
        self._invalidate_cache()
        return self

    def copy(self):
        new_fiber = SwcFiber()
        new_fiber.nodes = self.nodes[:]
        return new_fiber

    def local_bend_scores(
        self,
        window: float = 5.0,
        sample_step: float = 1.0,
        scale: Sequence[float] = (1.0, 1.0, 1.0),
        min_points: int = 4,
        robust: bool = True,
        robust_iterations: int = 3,
        huber_delta: float = 1.5,
    ) -> LocalBendResult:
        """Return robust local bend scores along this fiber.

        See :func:`local_bend_scores` for the score definition and parameter
        semantics. The returned arrays are aligned with uniformly resampled
        physical coordinates rather than the original, unevenly spaced nodes.
        """
        return local_bend_scores(
            self.coords,
            window=window,
            sample_step=sample_step,
            scale=scale,
            min_points=min_points,
            robust=robust,
            robust_iterations=robust_iterations,
            huber_delta=huber_delta,
        )

    def get_nearest_node(self, point, return_dist=False):
        coords = self.coords
        tree = cKDTree(coords)
        dist, idx = tree.query(point)
        if return_dist:
            return self.nodes[idx], dist
        else:
            return self.nodes[idx]

    def cache_resample_by_distance(self, dist_sample):
        if (
            self._resample_cache is not None
            and self._resample_cache["dist_sample"] == dist_sample
        ):
            return self._resample_cache["coords"], self._resample_cache["tree"]

        coords = resample_nodes_by_distance(self.coords, dist_sample)
        tree = cKDTree(coords)
        self._resample_cache = {
            "dist_sample": dist_sample,
            "coords": coords,
            "tree": tree,
        }

        self.resampled_coords = coords
        self.last_sampled_dist = dist_sample
        self.cktree = tree
        return coords, tree

    def cache_resample_coords_by_distance(self, dist_sample):
        return self.cache_resample_by_distance(dist_sample)[0]

    def cache_cKDTree(self, dist_sample):
        return self.cache_resample_by_distance(dist_sample)[1]

    def cahce_resample_coords_by_distance(self, dist_sample):
        return self.cache_resample_coords_by_distance(dist_sample)

    def cahce_cKDTree(self, coords):
        if self.resampled_coords is coords and self.cktree is not None:
            return self.cktree
        return cKDTree(coords)

    def get_overlap_length_with(
        self, fiber: "SwcFiber", dist_sample=1.0, dist_threshold=3.0
    ):
        coords1 = self.coords
        coords2 = fiber.coords
        if coords1.ndim == 1:
            coords1 = coords1[None, :]
        if coords2.ndim == 1:
            coords2 = coords2[None, :]
        if len(coords1) < 2 or len(coords2) < 2:
            return 0.0
        coords1, tree1 = self.cache_resample_by_distance(dist_sample)
        coords2, tree2 = fiber.cache_resample_by_distance(dist_sample)
        # fiber1->fiber2
        midpoints = (coords1[:-1] + coords1[1:]) * 0.5
        dists, _ = tree2.query(midpoints)
        mask = dists <= dist_threshold
        seglens1 = np.linalg.norm(coords1[1:] - coords1[:-1], axis=1)
        overlap1 = np.sum(seglens1[mask])
        # fiber2->fiber1
        midpoints = (coords2[:-1] + coords2[1:]) * 0.5
        dists, _ = tree1.query(midpoints)
        mask = dists <= dist_threshold
        seglens2 = np.linalg.norm(coords2[1:] - coords2[:-1], axis=1)
        overlap2 = np.sum(seglens2[mask])
        loverlap = (overlap1 + overlap2) / 2.0
        return loverlap

    def cal_iou(
        self,
        fiber: "SwcFiber",
        dist_sample=1.0,
        dist_threshold=3.0,
        min_iou_thres=0.5,
        eps=1e-7,
    ):
        if len(self.nodes) == 1:
            if (
                fiber.get_nearest_node(self.coords[0], return_dist=True)[1]
                <= dist_threshold
            ):
                return len(self.nodes) / len(fiber.nodes)
            else:
                return 0.0
        if len(fiber.nodes) == 1:
            if (
                self.get_nearest_node(fiber.coords[0], return_dist=True)[1]
                <= dist_threshold
            ):
                return len(fiber.nodes) / len(self.nodes)
            else:
                return 0.0
        l1 = self.length
        l2 = fiber.length
        if l1 / l2 < min_iou_thres or l2 / l1 < min_iou_thres:
            return 0.0
        overlap_length = self.get_overlap_length_with(
            fiber, dist_sample, dist_threshold
        )
        union_length = l1 + l2 - overlap_length
        return overlap_length / (union_length + eps)

    def is_sub_fiber_of(
        self,
        fiber: "SwcFiber",
        dist_sample=1.0,
        dist_threshold=3.0,
        same_fiber_iou_thres=0.9,
        return_prob=False,
    ):
        l1 = self.length
        l2 = fiber.length
        if l1 > l2 * 1.5:
            if return_prob:
                return 0.0
            else:
                return False
        overlap_length = self.get_overlap_length_with(
            fiber, dist_sample, dist_threshold
        )
        prob = overlap_length / (l1 + 1e-7)
        if return_prob:
            return prob
        else:
            return prob > same_fiber_iou_thres

    def to_str_list(self, scale=(1.0, 1.0, 1.0)):
        swc_str = []
        for node in self.nodes:
            if node.is_virtual():
                continue
            swc_str.append(node.to_swc_str(scale=scale))
        return "".join(swc_str)
