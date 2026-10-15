from pathlib import Path

import numpy as np

from swclib.data.swc import Swc, merge_swcs


class SwcReader:
    """Read one SWC or a directory containing one SWC per neuron.

    Directory construction scans coordinates in bounded chunks, retaining only
    file paths, node counts and bounding boxes. Region queries load intersecting
    neurons one at a time. Source files must remain unchanged during use.

    Rows are ordered by sorted file path, then by row order within each file.
    Accessing a node-array attribute (e.g. ``xs``) explicitly materializes and
    caches the entire directory, with globally reindexed IDs and parents.
    """

    _COLUMNS = ("ids", "types", "xs", "ys", "zs", "rs", "parents")
    _DTYPES = (np.int64, np.int32, np.float32, np.float32,
               np.float32, np.float32, np.int64)
    _INDEX_CHUNK_SIZE = 65536

    def __init__(self, path, rescale=(1.0, 1.0, 1.0), *, recursive=False):
        self.path = Path(path)
        self.rescale = tuple(rescale)
        self._coordinate_dtypes = tuple(
            (np.empty(0, dtype=np.float32) * scale).dtype for scale in self.rescale
        )
        self._materialized = not self.path.is_dir()
        if self._materialized:
            self.ids, self.types, self.xs, self.ys, self.zs, self.rs, self.parents, self.id2idx = self.load_swc(path, rescale)
            self.swc_paths = (self.path,)
            return

        paths = self.path.rglob("*") if recursive else self.path.iterdir()
        self.swc_paths = tuple(sorted(
            p for p in paths if p.is_file() and p.suffix.lower() == ".swc"
        ))
        self._bounds = np.empty(
            (len(self.swc_paths), 6), dtype=np.result_type(*self._coordinate_dtypes)
        )
        self._node_offsets = np.zeros(len(self.swc_paths) + 1, dtype=np.int64)
        self._offsets = []
        for i, swc_path in enumerate(self.swc_paths):
            count, bounds = self._scan_swc(swc_path)
            self._bounds[i] = bounds
            self._node_offsets[i + 1] = self._node_offsets[i] + count

    def __getattr__(self, name):
        if (name in (*self._COLUMNS, "id2idx")
                and not self.__dict__.get("_materialized", True)):
            self._materialize()
            return self.__dict__[name]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    @staticmethod
    def _iter_swc_rows(path):
        # Share parsing between the index scan and full neuron reads so that
        # skipped rows do not change the global mask's row offsets.
        with open(path, "r", buffering=1024 * 1024) as stream:
            for line in stream:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                row = np.fromstring(line, sep=" ")
                if row.size >= 7:
                    yield row

    def _scan_swc(self, path):
        bounds = np.array([np.inf] * 3 + [-np.inf] * 3, dtype=self._bounds.dtype)
        points = np.empty((self._INDEX_CHUNK_SIZE, 3), dtype=np.float32)
        count = used = 0

        def update_bounds(size):
            for axis in range(3):
                values = points[:size, axis] * self.rescale[axis]
                # NaNs never match a cube and must not hide valid coordinates.
                values = values[~np.isnan(values)]
                if values.size:
                    bounds[axis] = min(bounds[axis], values.min())
                    bounds[axis + 3] = max(bounds[axis + 3], values.max())

        for row in self._iter_swc_rows(path):
            points[used] = row[2:5]
            used += 1
            count += 1
            if used == len(points):
                update_bounds(used)
                used = 0
        if used:
            update_bounds(used)
        return count, bounds

    def _candidate_indices(self, cube):
        xmin, ymin, zmin, xmax, ymax, zmax = cube
        # Match the point-mask scalar comparisons, including float32 rounding
        # when a query boundary is not exactly representable. NumPy-valued
        # scale factors can promote individual coordinate columns to float64.
        keep = np.ones(len(self.swc_paths), dtype=bool)
        for axis, lower, upper in zip(range(3), (xmin, ymin, zmin), (xmax, ymax, zmax)):
            dtype = self._coordinate_dtypes[axis]
            keep &= self._bounds[:, axis].astype(dtype, copy=False) < upper
            keep &= self._bounds[:, axis + 3].astype(dtype, copy=False) >= lower
        return np.flatnonzero(keep)

    def _load_neuron(self, index):
        reader = SwcReader(self.swc_paths[index], self.rescale)
        expected = self._node_offsets[index + 1] - self._node_offsets[index]
        if reader.ids.size != expected:
            raise ValueError(f"SWC changed after indexing; recreate SwcReader: {self.swc_paths[index]}")
        for offset in self._offsets:
            reader.add_offset(offset)
        return reader

    def _materialize(self):
        """Opt in to full arrays, allocating each output column only once."""
        size = int(self._node_offsets[-1])
        dtypes = list(self._DTYPES)
        dtypes[2:5] = self._coordinate_dtypes
        arrays = {name: np.empty(size, dtype=dtype)
                  for name, dtype in zip(self._COLUMNS, dtypes)}
        for i in range(len(self.swc_paths)):
            reader = self._load_neuron(i)
            lo, hi = map(int, self._node_offsets[i:i + 2])
            for name in ("types", "xs", "ys", "zs", "rs"):
                arrays[name][lo:hi] = getattr(reader, name)
            arrays["ids"][lo:hi] = np.arange(lo + 1, hi + 1, dtype=np.int64)
            for j, parent in enumerate(reader.parents, start=lo):
                parent_index = reader.id2idx.get(int(parent)) if parent != -1 else None
                arrays["parents"][j] = -1 if parent_index is None else lo + parent_index + 1
            del reader
        id2idx = {i + 1: i for i in range(size)}
        self.__dict__.update(arrays, id2idx=id2idx)
        self._materialized = True

    def load_swc(self, path, rescale):
        """
        Fast-ish SWC parser for large text files.
        Returns:
        ids(int64), types(int32), xs(float32), ys(float32), zs(float32), rs(float32), parents(int64)
        SWC columns: n T x y z R P
        """
        ids = []
        types = []
        xs = []
        ys = []
        zs = []
        rs = []
        parents = []

        for arr in self._iter_swc_rows(path):
            ids.append(int(arr[0]))
            types.append(int(arr[1]))
            xs.append(arr[2])
            ys.append(arr[3])
            zs.append(arr[4])
            rs.append(arr[5])
            parents.append(int(arr[6]))

        ids = np.asarray(ids, dtype=np.int64)
        types = np.asarray(types, dtype=np.int32)
        xs = np.asarray(xs, dtype=np.float32) * rescale[0]
        ys = np.asarray(ys, dtype=np.float32) * rescale[1]
        zs = np.asarray(zs, dtype=np.float32) * rescale[2]
        rs = np.asarray(rs, dtype=np.float32)
        parents = np.asarray(parents, dtype=np.int64)

        # Build id -> index map (SWC ids are not always 1..N contiguous)
        id2idx = {int(i): k for k, i in enumerate(ids)}
        return ids, types, xs, ys, zs, rs, parents, id2idx
    
    def add_offset(self, offset):
        """Translate XYZ coordinates, updating a directory's index lazily."""
        if not self._materialized:
            offset = tuple(offset)
            # Replay each offset on load to preserve float32 rounding, even
            # after multiple calls; summing offsets first can move boundaries.
            for axis in range(3):
                for column in (axis, axis + 3):
                    values = self._bounds[:, column].astype(self._coordinate_dtypes[axis], copy=False)
                    values += offset[axis]
                    self._bounds[:, column] = values
            self._offsets.append(offset)
            return
        self.xs += offset[0]
        self.ys += offset[1]
        self.zs += offset[2]
    
    def query_cube_mask(self, cube):
        """Return one boolean per node, in sorted-file / source-row order.

        A directory allocates the full boolean mask, but loads coordinates only
        for candidate neurons. Nonintersecting files contribute False entries.
        """
        if not self._materialized:
            indices = self._candidate_indices(cube)
            keep = np.zeros(int(self._node_offsets[-1]), dtype=bool)
            for i in indices:
                reader = self._load_neuron(i)
                lo, hi = self._node_offsets[i:i + 2]
                keep[lo:hi] = reader.query_cube_mask(cube)
                del reader
            return keep
        xmin, ymin, zmin, xmax, ymax, zmax = cube
        return (self.xs >= xmin) & (self.xs < xmax) & (self.ys >= ymin) & (self.ys < ymax) & (self.zs >= zmin) & (self.zs < zmax)
    
    def check_cube_nonempty(self, cube):
        """Check actual node hits, stopping after the first nonempty neuron."""
        if not self._materialized:
            for i in self._candidate_indices(cube):
                reader = self._load_neuron(i)
                if reader.check_cube_nonempty(cube):
                    return True
                del reader
            return False
        return bool(self.query_cube_mask(cube).any())
    
    def read_region(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        out_path: str = None,
        out_r = None,
    ) -> Swc:
        """
        Read nodes in a cube and return them as a local-coordinate Swc object.

        Nodes are reindexed to 1..M. If a kept node's parent is outside the
        region, that node becomes a root in the returned SWC.
        """
        if not self._materialized:
            regions = []
            for i in self._candidate_indices([*start, *end]):
                reader = self._load_neuron(i)
                region = reader.read_region(start, end, out_r=out_r)
                if region.nodes:
                    regions.append(region)
                del reader
            swc = merge_swcs(regions)
            # Keep the reader's edge convention, including (root, -1).
            swc.edges = [(nid, node["parent"]) for nid, node in swc.nodes.items()]
            if out_path is not None:
                swc.save_to_swc(out_path, write_header=False)
            return swc
        swc = Swc()
        keep = self.query_cube_mask([*start, *end])
        keep_idx = np.flatnonzero(keep)
        if keep_idx.size == 0:
            if out_path is not None:
                swc.save_to_swc(out_path, write_header=False)
            return swc

        # Reindex kept nodes to 1..M
        # Map old node id -> new node id
        old_ids_kept = self.ids[keep_idx]
        new_ids = np.arange(1, keep_idx.size + 1, dtype=np.int64)
        oldid2newid = {int(oid): int(nid) for oid, nid in zip(old_ids_kept, new_ids)}

        # Prepare output arrays
        out_types = self.types[keep_idx]
        out_xs = self.xs[keep_idx] - start[0]
        out_ys = self.ys[keep_idx] - start[1]
        out_zs = self.zs[keep_idx] - start[2]
        out_rs = self.rs[keep_idx]
        if out_r is not None:
            out_rs = np.full_like(out_rs, out_r, dtype=np.float32)
        out_parents_old = self.parents[keep_idx]

        out_parents_new = np.empty_like(out_parents_old)
        for i in range(keep_idx.size):
            pid = int(out_parents_old[i])
            if pid == -1:
                out_parents_new[i] = -1
            else:
                # clip: if parent not kept => -1
                out_parents_new[i] = oldid2newid.get(pid, -1)

        for nid, t, x, y, z, r, pid_new in zip(
            new_ids,
            out_types,
            out_xs,
            out_ys,
            out_zs,
            out_rs,
            out_parents_new,
        ):
            nid = int(nid)
            pid_new = int(pid_new)
            swc.nodes[nid] = {
                "id": nid,
                "type": int(t),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "radius": float(r),
                "parent": pid_new,
            }
            swc.edges.append((nid, pid_new))

        swc._refresh_bound_box()

        if out_path is not None:
            swc.save_to_swc(out_path, write_header=False)
        return swc
