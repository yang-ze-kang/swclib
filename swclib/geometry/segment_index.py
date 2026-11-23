"""Mutable broad-phase segment index; exact geometric tests remain with callers."""

from collections import defaultdict
from itertools import product

import numpy as np


class SegmentIndex:
    """Spatial hash with stable integer IDs, incremental insertion and deletion.

    Long diagonal segments are tiled along their length, avoiding a cubic
    number of occupied cells. Queries return a conservative candidate set.
    """

    def __init__(self, cell_size=16.0):
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size must be positive and finite")
        self.cell_size = float(cell_size)
        self.cells = defaultdict(set)
        self.membership = {}

    def _cells(self, lower, upper):
        lo = np.floor(np.asarray(lower) / self.cell_size).astype(np.int64)
        hi = np.floor(np.asarray(upper) / self.cell_size).astype(np.int64)
        return product(*(range(int(a), int(b) + 1) for a, b in zip(lo, hi)))

    def add(self, index, start, end):
        self.remove(index)
        start, end = np.asarray(start), np.asarray(end)
        steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / self.cell_size)))
        cells = set()
        previous = start
        for step in range(1, steps + 1):
            point = start + (end - start) * (step / steps)
            cells.update(self._cells(np.minimum(previous, point), np.maximum(previous, point)))
            previous = point
        self.membership[index] = cells
        for cell in cells:
            self.cells[cell].add(index)

    def remove(self, index):
        for cell in self.membership.pop(index, ()):
            self.cells[cell].discard(index)
            if not self.cells[cell]:
                del self.cells[cell]

    def query(self, lower, upper):
        result = set()
        for cell in self._cells(lower, upper):
            result.update(self.cells.get(cell, ()))
        return result

    def query_segment(self, start, end, radius=0.0):
        """Conservative radius query without traversing a long diagonal's box."""
        start, end = np.asarray(start), np.asarray(end)
        steps = max(1, int(np.ceil(np.max(np.abs(end - start)) / self.cell_size)))
        result, previous = set(), start
        for step in range(1, steps + 1):
            point = start + (end - start) * (step / steps)
            result.update(self.query(np.minimum(previous, point) - radius,
                                     np.maximum(previous, point) + radius))
            previous = point
        return result
