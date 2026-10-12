import numpy as np
import pytest

from swclib.data.swc_fiber import SwcFiber, local_bend_scores
from swclib.data.swc_node import SwcNode


def _straight_fiber(length=20):
    return np.column_stack(
        [np.arange(length + 1, dtype=float), np.zeros(length + 1), np.zeros(length + 1)]
    )


def _right_angle_fiber(arm_length=10):
    left = [[x, 0.0, 0.0] for x in range(-arm_length, 1)]
    right = [[0.0, y, 0.0] for y in range(1, arm_length + 1)]
    return np.asarray(left + right, dtype=float)


def test_local_bend_score_is_zero_for_straight_fiber():
    result = local_bend_scores(_straight_fiber())

    assert result.valid_mask.sum() == 11
    assert np.allclose(result.bend_scores[result.valid_mask], 0.0)
    assert np.allclose(result.angles[result.valid_mask], 0.0)
    assert np.allclose(result.mean_curvature[result.valid_mask], 0.0)
    assert np.allclose(result.confidence[result.valid_mask], 1.0)
    assert np.all(np.isnan(result.bend_scores[~result.valid_mask]))


def test_local_bend_score_is_half_for_right_angle():
    result = local_bend_scores(_right_angle_fiber())
    corner_index = np.argmin(np.linalg.norm(result.coords, axis=1))

    assert result.valid_mask[corner_index]
    assert result.bend_scores[corner_index] == pytest.approx(0.5)
    assert result.angles[corner_index] == pytest.approx(np.pi / 2)
    assert result.mean_curvature[corner_index] == pytest.approx(np.pi / 10)
    assert result.confidence[corner_index] == pytest.approx(1.0)


def test_local_bend_score_is_invariant_to_fiber_reversal():
    coords = _right_angle_fiber()
    forward = local_bend_scores(coords)
    backward = local_bend_scores(coords[::-1])

    assert np.array_equal(forward.valid_mask, backward.valid_mask[::-1])
    assert np.allclose(
        forward.bend_scores,
        backward.bend_scores[::-1],
        equal_nan=True,
    )
    assert np.allclose(
        forward.confidence,
        backward.confidence[::-1],
        equal_nan=True,
    )


def test_local_bend_score_suppresses_small_coordinate_noise():
    coords = _straight_fiber()
    coords[:, 1] = np.array(
        [
            0.00,
            0.08,
            -0.04,
            0.02,
            -0.06,
            0.05,
            -0.03,
            0.01,
            -0.08,
            0.04,
            0.00,
            -0.04,
            0.07,
            -0.02,
            0.03,
            -0.06,
            0.02,
            0.05,
            -0.03,
            0.01,
            0.00,
        ]
    )

    result = local_bend_scores(coords)

    assert np.nanmax(result.bend_scores) < 1e-3
    assert np.nanmin(result.confidence) > 0.99


def test_swc_fiber_method_uses_physical_scale():
    fiber = SwcFiber()
    for node_id, x in enumerate(range(0, 42, 2), start=1):
        fiber.append(SwcNode(nid=node_id, coord=[float(x), 0.0, 0.0]))

    result = fiber.local_bend_scores(scale=(0.5, 1.0, 1.0))

    assert result.arc_lengths[-1] == pytest.approx(20.0)
    assert result.valid_mask.sum() == 11
    assert np.allclose(result.bend_scores[result.valid_mask], 0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 0.0},
        {"sample_step": 0.0},
        {"window": 2.0, "sample_step": 1.0, "min_points": 4},
        {"scale": (1.0, 1.0)},
        {"scale": (1.0, -1.0, 1.0)},
    ],
)
def test_local_bend_score_validates_parameters(kwargs):
    with pytest.raises(ValueError):
        local_bend_scores(_straight_fiber(), **kwargs)
