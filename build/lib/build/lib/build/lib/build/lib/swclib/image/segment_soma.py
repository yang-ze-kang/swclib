import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gaussian
from skimage.morphology import (
    ball,
    binary_opening,
    binary_closing,
    binary_dilation,
    remove_small_objects,
)
from skimage.measure import label


def crop_roi_3d(image: np.ndarray, center, radius):
    z, y, x = [int(round(v)) for v in center]
    rz, ry, rx = [int(round(v)) for v in radius]

    z0 = max(0, z - rz)
    z1 = min(image.shape[0], z + rz + 1)
    y0 = max(0, y - ry)
    y1 = min(image.shape[1], y + ry + 1)
    x0 = max(0, x - rx)
    x1 = min(image.shape[2], x + rx + 1)

    slices = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    roi = image[slices]
    local_center = (z - z0, y - y0, x - x0)
    return roi, slices, local_center


def keep_component_with_seed(mask: np.ndarray, seed):
    labeled = label(mask)
    seed_label = labeled[seed]
    if seed_label == 0:
        return np.zeros_like(mask, dtype=bool)
    return labeled == seed_label


def ellipsoid_mask(shape, center, radii):
    zz, yy, xx = np.indices(shape)
    zc, yc, xc = center
    rz, ry, rx = radii
    rz = max(float(rz), 1.0)
    ry = max(float(ry), 1.0)
    rx = max(float(rx), 1.0)
    return (
        ((zz - zc) / rz) ** 2 +
        ((yy - yc) / ry) ** 2 +
        ((xx - xc) / rx) ** 2
    ) <= 1.0


def segment_soma_from_seed(
    image: np.ndarray,
    seed_point,
    roi_radius=(20, 40, 40),
    sigma=1.0,
    percentile=99.0,
    alpha=0.7,
    min_size=200,
    line_open_radius=1,
    core_min_dist=1.0,
    max_dist=80.0,
    ellipsoid_radii=None,
    recover_radius=3,
):
    """
    Segment an ellipsoid-like bright soma and suppress bright thin processes.

    Args:
        image: 3D image, shape (z, y, x), uint16 preferred
        seed_point: (z, y, x)
        roi_radius: crop size around seed
        sigma: Gaussian smoothing sigma
        percentile: high percentile threshold in ROI
        alpha: threshold relative to seed intensity
        min_size: minimum size after cleanup
        line_open_radius: opening radius used to remove thin lines
        core_min_dist: keep only voxels whose distance-to-background is large enough
        max_dist: max allowed distance from seed
        ellipsoid_radii: optional ellipsoid prior radii, e.g. (10, 18, 18)
        recover_radius: dilate soma core a bit to recover boundary

    Returns:
        full_mask: uint8 mask
        info: debug dictionary
    """
    assert image.ndim == 3, "image must be 3D"
    seed_point = tuple(int(round(v)) for v in seed_point)
    if image.max() > 255:
        image = np.sqrt(image.astype(np.float32))

    roi_raw, slices, local_seed = crop_roi_3d(image, seed_point, roi_radius)
    roi = roi_raw.astype(np.float32)
    roi_smooth = gaussian(roi, sigma=sigma, preserve_range=True)

    seed_value = float(roi_smooth[local_seed])
    thresh_p = float(np.percentile(roi_smooth, percentile))
    thresh_s = float(alpha * seed_value)

    # Step 1: high-intensity candidate
    mask = (roi_smooth >= thresh_p) & (roi_smooth >= thresh_s)

    # Step 2: keep only the component containing the seed
    mask = keep_component_with_seed(mask, local_seed)

    # Step 3: distance constraint around seed
    zz, yy, xx = np.indices(mask.shape)
    zc, yc, xc = local_seed
    dist_to_seed = np.sqrt((zz - zc) ** 2 + (yy - yc) ** 2 + (xx - xc) ** 2)
    mask &= (dist_to_seed <= float(max_dist))

    # Step 4: optional ellipsoid prior
    if ellipsoid_radii is not None:
        mask &= ellipsoid_mask(mask.shape, local_seed, ellipsoid_radii)

    # Step 5: remove thin lines by opening
    if line_open_radius > 0:
        opened = binary_opening(mask, footprint=ball(line_open_radius))
    else:
        opened = mask.copy()

    # If opening removes everything at seed, fall back to original mask
    if not opened[local_seed]:
        opened = mask.copy()

    # Step 6: keep only the seed component again
    opened = keep_component_with_seed(opened, local_seed)

    # Step 7: keep only "thick" part using distance transform
    dist_in_obj = ndi.distance_transform_edt(opened)
    core = opened & (dist_in_obj >= float(core_min_dist))

    # If core becomes empty, relax
    if not core[local_seed]:
        core = opened & (dist_in_obj >= max(1.0, float(core_min_dist) * 0.5))
    if not core[local_seed]:
        core = opened.copy()

    core = keep_component_with_seed(core, local_seed)

    # Step 8: recover soma boundary from thick core only
    if recover_radius > 0:
        soma = binary_dilation(core, footprint=ball(recover_radius)) & mask
    else:
        soma = core

    soma = keep_component_with_seed(soma, local_seed)

    # Step 9: cleanup
    soma = binary_closing(soma, footprint=ball(2))
    soma = ndi.binary_fill_holes(soma)
    soma = remove_small_objects(soma, min_size=min_size)
    soma = keep_component_with_seed(soma, local_seed)

    full_mask = np.zeros_like(image, dtype=np.uint8)
    full_mask[slices] = soma.astype(np.uint8)

    # info = {
    #     "seed_value": seed_value,
    #     "threshold_percentile": thresh_p,
    #     "threshold_seed": thresh_s,
    #     "percentile": percentile,
    #     "alpha": alpha,
    #     "line_open_radius": line_open_radius,
    #     "core_min_dist": core_min_dist,
    #     "max_dist": max_dist,
    #     "roi_min": float(roi_smooth.min()),
    #     "roi_max": float(roi_smooth.max()),
    # }
    return full_mask