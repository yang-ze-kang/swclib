import numpy as np
from scipy.ndimage import distance_transform_edt
import tifffile as tiff
import tqdm
import glob
import os


from swclib.data.swc import Swc


def swc_to_mask_line(swc: Swc, shape=(256, 256, 256), scale=(1.0, 1.0, 1.0), max_radius=5):
    mask = np.zeros(shape[::-1], dtype=np.uint8)
    def draw_line(p1, p2, r):
        p1 = np.array(p1)
        p2 = np.array(p2)
        diff = p2 - p1
        length = np.linalg.norm(diff)
        if length == 0:
            mask[int(p1[0]), int(p1[1]), int(p1[2])] = 1
            return
        steps = int(length) + 1
        for i in range(steps + 1):
            pos = p1 + diff * i / steps
            z, y, x = np.round(pos).astype(int)
            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                mask[z, y, x] = 1

    for nid, node in swc.nodes.items():
        x, y, z, r = node["x"], node["y"], node["z"], node["radius"]
        parent_id = node["parent"]
        if parent_id == -1:
            continue
        parent = swc.nodes[parent_id]
        draw_line((z, y, x), (parent["z"], parent["y"], parent["x"]), r)
    mask_dt = distance_transform_edt(1 - mask, sampling=scale[::-1])
    mask = (mask_dt <= max_radius).astype(np.uint8)
    return mask


class Swc2Mask:

    def __init__(
        self, shape=(300, 300, 300), scale=(1.0, 1.0, 1.0), radius=1, method="line"
    ):
        self.shape = shape
        self.scale = scale
        self.radius = radius
        self.method = method

    def run(self, swc, out_file=None):
        if isinstance(swc, str) or isinstance(swc, os.PathLike):
            swc = Swc(swc)
        if self.method == "line":
            mask = swc_to_mask_line(
                swc, shape=self.shape, scale=self.scale, max_radius=self.radius
            )
        else:
            raise NotImplementedError
        if out_file is not None:
            tiff.imwrite(out_file, mask.astype(np.uint8))
        return mask
