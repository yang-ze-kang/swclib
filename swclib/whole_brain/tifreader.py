import rasterio
from rasterio.windows import Window
import tifffile
import numpy as np
import os
import re
from ._volume import VolumeReader, read_parallel


class WBTReader(VolumeReader):
    """
    Reader class for 3D image slice data
    Regions, dimensions and arrays follow axis_order (default ZYX).
    """

    def __init__(self, slice_dir, slice_name_pattern, slice_ext=".tif", axis_order="ZYX"):
        super().__init__(axis_order=axis_order)
        self.slice_dir = slice_dir
        slice_ext = slice_ext.lower()
        if not slice_ext.startswith("."):
            slice_ext = "." + slice_ext
        files = [
            os.path.join(root, file)
            for root, _, filenames in os.walk(slice_dir)
            for file in filenames
            if file.lower().endswith(slice_ext)
        ]
        pattern = re.compile(slice_name_pattern)
        if not files:
            raise ValueError("Slice directory contains no matching image files")
        if pattern.groups != 1:
            raise ValueError("slice_name_pattern must contain one Z-index capture group")
        if any(pattern.match(os.path.basename(path)) is None for path in files):
            raise ValueError("Some slice filenames do not match slice_name_pattern")
        files = sorted(files, key=lambda x: int(pattern.match(os.path.basename(x)).group(1)))
        start_z = int(pattern.match(os.path.basename(files[0])).group(1))
        end_z  = int(pattern.match(os.path.basename(files[-1])).group(1))
        indices = [int(pattern.match(os.path.basename(path)).group(1)) for path in files]
        if start_z < 0 or indices != list(range(start_z, end_z + 1)):
            raise ValueError("Slice Z indices must be nonnegative, unique and continuous")
        self.slice_paths = []
        if start_z > 0:
            self.slice_paths = [None] * start_z
        self.slice_paths += files
        self.start_z = start_z
        self.end_z = end_z
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
            start: inclusive coordinates in self.axis_order
            end: exclusive coordinates in self.axis_order
            mode: 'raster' or 'tiff'
            num_workers: 并行 worker 数
            parallel_backend: 'thread' or 'process'
            padding: None, or 'right' to zero-fill beyond positive boundaries
        """
        start, end = self._validate_coords(start, end, padding)
        z1, y1, x1 = start
        z2, y2, x2 = end
        d, h, w = z2 - z1, y2 - y1, x2 - x1
        image = np.zeros((d, h, w), dtype=self.dtype)

        subz2, suby2, subx2 = min(z2, self.depth), min(y2, self.height), min(x2, self.width)

        if mode not in ("raster", "tiff"):
            raise ValueError(f"Unsupported mode: {mode}. Use 'raster' or 'tiff'")
        if any(a >= b for a, b in zip(start, (subz2, suby2, subx2))):
            list(read_parallel(self._rasterio_read_one, (), num_workers, parallel_backend))
            return self._output_array(image)

        image_sub = image[:subz2 - z1, :suby2 - y1, :subx2 - x1]
        if mode == "raster":
            self._read_with_rasterio_parallel(
                z1,
                subz2,
                y1,
                suby2,
                x1,
                subx2,
                num_workers=num_workers,
                parallel_backend=parallel_backend,
                out=image_sub,
            )
        elif mode == "tiff":
            self._read_with_tifffile_parallel(
                z1,
                subz2,
                y1,
                suby2,
                x1,
                subx2,
                num_workers=num_workers,
                parallel_backend=parallel_backend,
                out=image_sub,
            )
        return self._output_array(image)

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
        self, z1, z2, y1, y2, x1, x2, num_workers=8, parallel_backend="thread", out=None
    ):
        d, h, w = z2 - z1, y2 - y1, x2 - x1
        image = np.empty((d, h, w), dtype=self.dtype) if out is None else out

        tasks = (
            (i, self.slice_paths[z], x1, y1, w, h) for i, z in enumerate(range(z1, z2))
        )

        for i, arr in read_parallel(WBTReader._rasterio_read_one, tasks, num_workers, parallel_backend):
            image[i] = arr

        return image

    # ------------------- tifffile 并行版 -------------------
    @staticmethod
    def _tifffile_read_one(args):
        """worker: 读单张 slice 并裁剪；返回 (i, slice_2d)"""
        i, path, y1, y2, x1, x2 = args
        with tifffile.TiffFile(path) as tif:
            full = tif.asarray()
        return i, full[y1:y2, x1:x2].copy()

    def _read_with_tifffile_parallel(
        self, z1, z2, y1, y2, x1, x2, num_workers=8, parallel_backend="thread", out=None
    ):
        d, h, w = z2 - z1, y2 - y1, x2 - x1
        image = np.empty((d, h, w), dtype=self.dtype) if out is None else out

        tasks = (
            (i, self.slice_paths[z], y1, y2, x1, x2)
            for i, z in enumerate(range(z1, z2))
        )

        for i, arr in read_parallel(WBTReader._tifffile_read_one, tasks, num_workers, parallel_backend):
            image[i] = arr

        return image

    def __repr__(self):
        return f"WBTReader(slices={self.depth}, axis_order={self.axis_order!r}, size={self.shape})"


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
