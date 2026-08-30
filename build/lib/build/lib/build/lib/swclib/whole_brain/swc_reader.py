import numpy as np

from swclib.data.swc import Swc


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
        out_path: str = None,
        out_r = None,
    ) -> Swc:
        """
        Read nodes in a cube and return them as a local-coordinate Swc object.

        Nodes are reindexed to 1..M. If a kept node's parent is outside the
        region, that node becomes a root in the returned SWC.
        """
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
