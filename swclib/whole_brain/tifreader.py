from tracemalloc import start

import rasterio
from rasterio.windows import Window
import tifffile
import numpy as np
import os
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


class WBTReader:
    """
    Reader class for 3D image slice data
    Supports reading arbitrary 3D regions from multiple 2D slice files
    """

    def __init__(self, slice_dir, slice_name_pattern):
        self.slice_dir = slice_dir
        files = os.listdir(slice_dir)
        pattern = re.compile(slice_name_pattern)
        files = sorted(files, key=lambda x: int(pattern.match(x).group(1)))
        start_z = int(pattern.match(files[0]).group(1))
        end_z  = int(pattern.match(files[-1]).group(1))
        if start_z > 0:
            self.slice_paths = [None] * start_z
        self.slice_paths += [os.path.join(slice_dir, file) for file in files]
        assert len(self.slice_paths) == end_z + 1, "Slice files are not continuous from 0 to max_z"
        self.start_z = start_z
        self.edn_z = end_z
        self._init_dimensions()

    def _init_dimensions(self):
        if not self.slice_paths:
            raise ValueError("Slice directory is empty")
        with rasterio.open(self.slice_paths[self.start_z], "r") as src:
            self.height = src.height
            self.width = src.width
            self.dtype = src.dtypes[0]

        self.depth = len(self.slice_paths)

    def read_region(
        self,
        start,
        end,
        mode="raster",
        num_workers=32,
        parallel_backend="thread",
        padding=None,
    ):
        """
        Read 3D region data

        Args:
            start: (Z1, Y1, X1)
            end:   (Z2, Y2, X2)
            mode: 'raster' or 'tiff'
            num_workers: 并行 worker 数
            parallel_backend: 'thread' or 'process'
        """
        z1, y1, x1 = start
        z2, y2, x2 = end
        self._validate_coords(start, end, padding)
        d, h, w = z2 - z1, y2 - y1, x2 - x1
        image = np.empty((d, h, w), dtype=self.dtype)

        subz2, suby2, subx2 = min(z2, self.depth), min(y2, self.height), min(x2, self.width)

        if mode == "raster":
            image_sub = self._read_with_rasterio_parallel(
                z1,
                subz2,
                y1,
                suby2,
                x1,
                subx2,
                num_workers=num_workers,
                parallel_backend=parallel_backend,
            )
        elif mode == "tiff":
            image_sub = self._read_with_tifffile_parallel(
                z1,
                subz2,
                y1,
                suby2,
                x1,
                subx2,
                num_workers=num_workers,
                parallel_backend=parallel_backend,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'raster' or 'tiff'")
        image[:subz2, :suby2, :subx2] = image_sub
        return image

    def _validate_coords(self, start, end, padding=None):
        z1, y1, x1 = start
        z2, y2, x2 = end
        if z1 < self.start_z or (z2 > self.depth and padding != "right") or z1 >= z2:
            raise ValueError(
                f"Invalid Z coordinates: ({z1}, {z2}), valid range: [0, {self.depth})"
            )
        if y1 < 0 or (y2 > self.height and padding != "right") or y1 >= y2:
            raise ValueError(
                f"Invalid Y coordinates: ({y1}, {y2}), valid range: [0, {self.height})"
            )
        if x1 < 0 or (x2 > self.width and padding != "right") or x1 >= x2:
            raise ValueError(
                f"Invalid X coordinates: ({x1}, {x2}), valid range: [0, {self.width})"
            )

    # ------------------- rasterio 并行版 -------------------
    @staticmethod
    def _rasterio_read_one(args):
        """worker: 读单张 slice 的 window；返回 (i, slice_2d)"""
        i, path, x1, y1, w, h = args
        window = Window(x1, y1, w, h)
        with rasterio.open(path) as src:
            arr = src.read(1, window=window)
        return i, arr

    def _read_with_rasterio_parallel(
        self, z1, z2, y1, y2, x1, x2, num_workers=8, parallel_backend="thread"
    ):
        d, h, w = z2 - z1, y2 - y1, x2 - x1
        image = np.empty((d, h, w), dtype=self.dtype)

        Executor = (
            ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
        )
        tasks = [
            (i, self.slice_paths[z], x1, y1, w, h) for i, z in enumerate(range(z1, z2))
        ]

        with Executor(max_workers=num_workers) as ex:
            for i, arr in ex.map(WBTReader._rasterio_read_one, tasks, chunksize=1):
                image[i] = arr

        return image

    # ------------------- tifffile 并行版 -------------------
    @staticmethod
    def _tifffile_read_one(args):
        """worker: 读单张 slice 并裁剪；返回 (i, slice_2d)"""
        i, path, y1, y2, x1, x2 = args
        with tifffile.TiffFile(path) as tif:
            full = tif.asarray()
        return i, full[y1:y2, x1:x2]

    def _read_with_tifffile_parallel(
        self, z1, z2, y1, y2, x1, x2, num_workers=8, parallel_backend="thread"
    ):
        d, h, w = z2 - z1, y2 - y1, x2 - x1
        image = np.empty((d, h, w), dtype=self.dtype)

        Executor = (
            ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
        )
        tasks = [
            (i, self.slice_paths[z], y1, y2, x1, x2)
            for i, z in enumerate(range(z1, z2))
        ]

        with Executor(max_workers=num_workers) as ex:
            for i, arr in ex.map(WBTReader._tifffile_read_one, tasks, chunksize=1):
                image[i] = arr

        return image

    def get_dimensions(self):
        return (self.depth, self.height, self.width)

    def __repr__(self):
        return f"WBTReader(slices={self.depth}, size=({self.depth}, {self.height}, {self.width}))"


# Usage example
if __name__ == "__main__":
    import time

    # Initialize reader
    reader = WBTReader("/data2/CH1/slices")
    # reader = WBTReader("/home/yangzekang/data1/neuron/whole_brain/test_tiled_tifs")

    # Print image information
    print(reader)
    print(f"Image dimensions: {reader.get_dimensions()}")

    # 1.0059558593750e+04 2.7955351562500e+03 7.9575000000000e+03
    x, y, z = 10059, 2795, 7957
    x = int(x / 0.35)
    y = int(y / 0.35)
    # z = 150
    t1 = time.time()
    region = reader.read_region(
        # start=(z-150, y-150, x-150),
        # end=(z+150, y+150, x+150),
        start=(8600, 11150, 24450),
        end=(8900, 11450, 24750),
        mode="raster",
    )
    t2 = time.time()
    print(t2 - t1)

    print(f"Region shape: {region.shape}")
    print(f"Data type: {region.dtype}")

    tifffile.imwrite("test.tif", region)
