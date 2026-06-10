import numpy as np


class SwcReader:

    def __init__(self, path, rescale=(1.0, 1.0, 1.0)):
        self.ids, self.types, self.xs, self.ys, self.zs, self.rs, self.parents, self.id2idx = self.load_swc(path, rescale)

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

        # big buffer helps
        with open(path, "r", buffering=1024 * 1024) as f:
            for line in f:
                if not line or line[0] == "#":
                    continue
                s = line.strip()
                if not s:
                    continue
                # Using numpy.fromstring on each line is often faster than split+float for big files.
                arr = np.fromstring(s, sep=" ")
                if arr.size < 7:
                    # tolerate odd formatting
                    continue
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
        self.xs += offset[0]
        self.ys += offset[1]
        self.zs += offset[2]
    
    def query_cube_mask(self, cube):
        xmin, ymin, zmin, xmax, ymax, zmax = cube
        return (self.xs >= xmin) & (self.xs < xmax) & (self.ys >= ymin) & (self.ys < ymax) & (self.zs >= zmin) & (self.zs < zmax)
    
    def check_cube_nonempty(self, cube):
        keep = self.query_cube_mask(cube)
        keep_idx = np.flatnonzero(keep)
        return keep_idx.size > 0
    
    def read_region(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        out_path: str,
        out_r = None,
    ):
        keep = self.query_cube_mask([*start, *end])
        keep_idx = np.flatnonzero(keep)
        if keep_idx.size == 0:
            return False

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
        out_parents_old = self.parents[keep_idx]

        out_parents_new = np.empty_like(out_parents_old)
        for i in range(keep_idx.size):
            pid = int(out_parents_old[i])
            if pid == -1:
                out_parents_new[i] = -1
            else:
                # clip: if parent not kept => -1
                out_parents_new[i] = oldid2newid.get(pid, -1)

        # Write SWC
        # Note: keep formatting simple and fast
        if out_path is not None:
            with open(out_path, "w", buffering=1024 * 1024) as f:
                f.write(f"# cube: {start}->{end}\n")
                for nid, t, x, y, z, r, pid_new in zip(new_ids, out_types, out_xs, out_ys, out_zs, out_rs, out_parents_new):
                    if out_r is not None:
                        r = out_r
                    f.write(f"{nid} {int(t)} {float(x):.6f} {float(y):.6f} {float(z):.6f} {float(r):.6f} {int(pid_new)}\n")
        return True