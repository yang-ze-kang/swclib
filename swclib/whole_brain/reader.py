"""Unified whole-brain image reader."""


class WholeBrainReader:
    """Select FNT cubes (mode='fnt') or 2D TIFF slices (mode='slice').

    Backend-specific constructor options are forwarded to the selected reader:
    ``channel`` for FNT, ``slice_name_pattern`` and ``slice_ext`` for slices.
    ``axis_order`` selects ZYX (default) or XYZ for region coordinates, returned
    arrays and dimensions. Bounding boxes and physical metadata always use XYZ.
    """

    def __init__(self, data_dir, mode="slice", axis_order="ZYX", **kwargs):
        self.mode = mode.lower()
        if self.mode == "fnt":
            from .fntreader import FNTReader
            self.reader = FNTReader(data_dir, axis_order=axis_order, **kwargs)
        elif self.mode == "slice":
            from .tifreader import WBTReader
            self.reader = WBTReader(data_dir, axis_order=axis_order, **kwargs)
        else:
            raise ValueError("mode must be 'fnt' or 'slice'")

    def get_dimensions(self):
        """Return the volume dimensions in the selected axis_order."""
        return self.reader.get_dimensions()

    def get_bbox(self):
        """Return XYZ voxel bounds with an exclusive upper corner."""
        return self.reader.get_bbox()

    def read_region(self, start, end, **kwargs):
        """Read coordinates and return an array in the selected axis_order."""
        return self.reader.read_region(start, end, **kwargs)

    def __getattr__(self, name):
        reader = self.__dict__.get("reader")
        if reader is None:
            raise AttributeError(name)
        return getattr(reader, name)

    def __repr__(self):
        return f"WholeBrainReader(mode={self.mode!r}, reader={self.reader!r})"
