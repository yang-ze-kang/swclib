"""Region reader for Fast Neurite Tracer's catalog + NRRD cube exports."""

from configparser import ConfigParser
from itertools import product
from pathlib import Path
import re

import nrrd
import numpy as np

from ._volume import VolumeReader, read_parallel


def _header_dtype(header):
    # NRRD scalar type aliases (use a native dtype for the assembled volume).
    aliases = {
        "i1": "signed char,int8,int8_t",
        "u1": "uchar,unsigned char,uint8,uint8_t",
        "i2": "short,short int,signed short,signed short int,int16,int16_t",
        "u2": "ushort,unsigned short,unsigned short int,uint16,uint16_t",
        "i4": "int,signed int,int32,int32_t",
        "u4": "uint,unsigned int,uint32,uint32_t",
        "i8": "longlong,long long,long long int,signed long long,signed long long int,int64,int64_t",
        "u8": "ulonglong,unsigned long long,unsigned long long int,uint64,uint64_t",
        "f4": "float",
        "f8": "double",
    }
    for dtype, names in aliases.items():
        if header["type"] in names.split(","):
            return np.dtype(dtype)
    raise ValueError(f"Unsupported NRRD scalar type: {header['type']}")


class FNTReader(VolumeReader):
    """Read only intersecting cubes from an FNT export directory.

    ``fnt_dir`` contains ``catalog`` and channel subdirectories such as ``ch00``.
    Catalog size/cubesize, origin, spacing and direction use XYZ order.
    Regions, dimensions and returned NumPy arrays follow axis_order (default ZYX).
    Missing or malformed cubes raise errors instead of silently losing data.
    """

    _placeholder = re.compile(r"<(\d*)([XYZ])>")

    def __init__(self, fnt_dir, channel="ch00", axis_order="ZYX"):
        super().__init__(axis_order=axis_order)
        self.fnt_dir = Path(fnt_dir)
        catalog = ConfigParser(interpolation=None)
        with (self.fnt_dir / "catalog").open(encoding="utf-8-sig") as stream:
            catalog.read_file(stream)
        channels = {name.lower(): name for name in catalog.sections()
                    if "pattern" in catalog[name]}
        self.channels = tuple(channels)
        self.channel = str(channel).lower()
        if self.channel not in channels:
            raise ValueError(f"Unknown cube channel {channel!r}; available: {self.channels}")
        config = catalog[channels[self.channel]]
        self.pattern = config["pattern"]
        if {axis for _, axis in self._placeholder.findall(self.pattern)} != set("XYZ"):
            raise ValueError("Catalog pattern must contain <...X>, <...Y> and <...Z>")
        self.size = self._parse_vector(config["size"], 3, int, "size")
        self.cubesize = self._parse_vector(config["cubesize"], 3, int, "cubesize")
        if min(self.size + self.cubesize) <= 0:
            raise ValueError("Catalog size and cubesize must be positive")
        self.width, self.height, self.depth = self.size
        self.cube_shape = self.cubesize[::-1]
        self.start_z, self.end_z = 0, self.depth - 1
        self.origin = self._parse_vector(config.get("origin", "0 0 0"), 3, float, "origin")
        self.direction = np.asarray(self._parse_vector(
            config.get("direction", "1 0 0 0 1 0 0 0 1"), 9, float, "direction"
        )).reshape(3, 3)
        self.spacing = tuple(np.linalg.norm(self.direction, axis=1))
        # Header only: initialization neither scans the tree nor decodes a cube.
        header = nrrd.read_header(str(self._cube_path((0, 0, 0))))
        if header.get("dimension") != 3:
            raise ValueError("FNT cubes must be three-dimensional scalar NRRDs")
        self.dtype = _header_dtype(header)

    @staticmethod
    def _parse_vector(value, length, convert, name):
        result = tuple(convert(v) for v in value.split())
        if len(result) != length:
            raise ValueError(f"Catalog {name} must contain {length} values")
        return result

    def _cube_path(self, start):
        coords = dict(zip("ZYX", start))
        relative = self._placeholder.sub(
            lambda match: str(coords[match[2]]).zfill(int(match[1] or 0)), self.pattern
        )
        return self.fnt_dir / relative

    @staticmethod
    def _read_cube(task):
        path, source, target, expected_shape, cube_shape, dtype = task
        data, _ = nrrd.read(str(path), index_order="C")
        # Exporters may truncate edge cubes or retain their full padded size.
        if data.shape not in (expected_shape, cube_shape):
            raise ValueError(f"Unexpected cube shape in {path}: {data.shape}; expected {expected_shape}")
        if data.dtype.newbyteorder("=") != dtype:
            raise ValueError(f"Unexpected cube dtype in {path}: {data.dtype}; expected {dtype}")
        return target, data[source].copy()

    def read_region(self, start, end, mode="nrrd", num_workers=8,
                    parallel_backend="thread", padding=None):
        """Read ``[start, end)`` with coordinates and output in self.axis_order.

        ``padding='right'`` zero-fills beyond the positive volume boundaries.
        Gzip NRRDs are decoded per intersecting cube; pending reads are bounded
        by ``num_workers``. Use one worker for serial reading.
        """
        if mode != "nrrd":
            raise ValueError("FNTReader mode must be 'nrrd'")
        start, end = self._validate_coords(start, end, padding)
        image = np.zeros(tuple(b - a for a, b in zip(start, end)), dtype=self.dtype)
        clipped_end = tuple(min(b, n) for b, n in zip(end, self._shape_zyx))

        def tasks():
            if any(a >= b for a, b in zip(start, clipped_end)):
                return
            ranges = [range(a // c * c, b, c)
                      for a, b, c in zip(start, clipped_end, self.cube_shape)]
            for corner in product(*ranges):
                stop = tuple(min(a + c, n) for a, c, n in zip(corner, self.cube_shape, self._shape_zyx))
                lo = tuple(max(a, b) for a, b in zip(start, corner))
                hi = tuple(min(a, b) for a, b in zip(clipped_end, stop))
                source = tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, corner))
                target = tuple(slice(a - c, b - c) for a, b, c in zip(lo, hi, start))
                expected = tuple(b - a for a, b in zip(corner, stop))
                yield self._cube_path(corner), source, target, expected, self.cube_shape, self.dtype

        for target, block in read_parallel(self._read_cube, tasks(), num_workers, parallel_backend):
            image[target] = block
        return self._output_array(image)

    def __repr__(self):
        return (f"FNTReader(channel={self.channel!r}, axis_order={self.axis_order!r}, "
                f"size={self.shape}, cube_shape={self.cube_shape})")
