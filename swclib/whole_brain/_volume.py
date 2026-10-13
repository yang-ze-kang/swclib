"""Shared coordinate conventions and bounded parallel I/O."""

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from operator import index


class VolumeReader:
    """Regions and arrays follow axis_order; bounding boxes always use XYZ."""

    def __init__(self, axis_order="ZYX"):
        if not isinstance(axis_order, str) or axis_order.upper() not in ("ZYX", "XYZ"):
            raise ValueError("axis_order must be 'ZYX' or 'XYZ'")
        self._axis_order = axis_order.upper()

    @property
    def axis_order(self):
        return self._axis_order

    @property
    def _shape_zyx(self):
        return (self.depth, self.height, self.width)

    @property
    def shape(self):
        return self.get_dimensions()

    def get_dimensions(self):
        """Return dimensions in the constructor's axis_order."""
        return self._shape_zyx if self.axis_order == "ZYX" else self._shape_zyx[::-1]

    def _output_array(self, image):
        """Expose the internal ZYX array in the requested order without copying."""
        return image if self.axis_order == "ZYX" else image.transpose(2, 1, 0)

    def get_bbox(self):
        """Return the XYZ voxel bounds, with an exclusive upper corner."""
        return ((0, 0, self.start_z), (self.width, self.height, self.depth))

    def _validate_coords(self, start, end, padding=None):
        """Validate public coordinates and return internal ZYX coordinates."""
        if padding not in (None, "right"):
            raise ValueError("padding must be None or 'right'")
        try:
            start = tuple(index(v) for v in start)
            end = tuple(index(v) for v in end)
        except TypeError as exc:
            raise ValueError("start and end must contain integer voxel coordinates") from exc
        if len(start) != 3 or len(end) != 3:
            raise ValueError(f"start and end must each contain three {self.axis_order} coordinates")
        if self.axis_order == "XYZ":
            start, end = start[::-1], end[::-1]
        for axis, lo, hi, minimum, maximum in zip(
            "ZYX", start, end, (self.start_z, 0, 0), self._shape_zyx
        ):
            if lo < minimum or lo >= hi or (hi > maximum and padding is None):
                raise ValueError(
                    f"Invalid {axis} coordinates: ({lo}, {hi}), "
                    f"valid range: [{minimum}, {maximum})"
                )
        return start, end


def read_parallel(worker, tasks, num_workers, parallel_backend):
    """Yield results with at most num_workers pending reads/results in memory."""
    if parallel_backend not in ("thread", "process"):
        raise ValueError("parallel_backend must be 'thread' or 'process'")
    if not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")
    if num_workers == 1:
        yield from map(worker, tasks)
        return
    executor = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
    tasks = iter(tasks)
    with executor(max_workers=num_workers) as pool:
        pending = deque()
        for _ in range(num_workers):
            task = next(tasks, None)
            if task is None:
                break
            pending.append(pool.submit(worker, task))
        while pending:
            yield pending.popleft().result()
            task = next(tasks, None)
            if task is not None:
                pending.append(pool.submit(worker, task))
