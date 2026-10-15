"""Read-only adapters around swclib; browser payloads use little-endian arrays."""

import json
import math
import os
from pathlib import Path
import struct
import threading
import uuid

import numpy as np

MIB = 1024 ** 2
MAX_VOLUME_EDGE = 300
MAX_VOLUME_VOXELS = MAX_VOLUME_EDGE ** 3
DEFAULT_VOLUME_MIB = 128
IMAGE_EXTENSIONS = {".tif", ".tiff", ".nrrd", ".npy"}


def vector(value, name, integer=False):
    try:
        result = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须包含三个有限数值（XYZ）") from exc
    if len(result) != 3 or not all(math.isfinite(v) for v in result):
        raise ValueError(f"{name} 必须包含三个有限数值（XYZ）")
    if integer:
        if any(v != int(v) for v in result):
            raise ValueError(f"{name} 必须为整数体素坐标")
        return tuple(int(v) for v in result)
    return result


def region(start, end, integer=False):
    start, end = vector(start, "start", integer), vector(end, "end", integer)
    if any(a >= b for a, b in zip(start, end)):
        raise ValueError("每个轴均要求 start < end；end 不包含在区域内")
    return start, end


def packet(metadata, *arrays):
    header = json.dumps(metadata, ensure_ascii=False, allow_nan=False).encode("utf-8")
    header += b" " * (-len(header) % 4)
    return [struct.pack("<I", len(header)), header] + [memoryview(a).cast("B") if a.size else b"" for a in arrays]


class Catalog:
    """Only paths/options live here. Registration never decodes image pixels."""

    def __init__(self):
        self.entries = {}
        self.sources = {}
        self.lock = threading.Lock()

    def list(self):
        with self.lock:
            return list(self.entries.values())

    def get(self, key, kind):
        with self.lock:
            entry = self.entries.get(key)
        if entry is None or entry["kind"] != kind:
            raise ValueError("未找到对应的数据条目")
        return entry

    def remove(self, ids):
        """Forget catalog records only: never unlink or otherwise alter sources."""
        if not isinstance(ids, list) or not ids or any(not isinstance(key, str) for key in ids):
            raise ValueError("ids 必须是非空的条目 ID 列表")
        removed = []
        with self.lock:
            for key in dict.fromkeys(ids):
                entry = self.entries.pop(key, None)
                if entry is not None:
                    source_key = (entry["kind"], entry["path"], json.dumps(entry["options"], sort_keys=True))
                    self.sources.pop(source_key, None)
                    removed.append(key)
        return removed

    def add(self, path, kind, recursive=False, options=None):
        if kind not in ("swc", "image", "brain"):
            raise ValueError("不支持的数据类型")
        path = Path(path).expanduser().resolve(strict=True)
        extensions = {".swc"} if kind == "swc" else IMAGE_EXTENSIONS
        if kind == "brain":
            if not path.is_dir():
                raise ValueError("全脑路径必须是目录")
            options = dict(options or {})
            if options.get("mode") not in ("slice", "fnt"):
                raise ValueError("全脑模式应为 slice 或 fnt")
            paths = [path]
        elif path.is_dir():
            paths = sorted(p for p in (path.rglob("*") if recursive else path.iterdir())
                           if p.is_file() and p.suffix.lower() in extensions)
        else:
            paths = [path] if path.is_file() and path.suffix.lower() in extensions else []
        if not paths:
            raise ValueError("路径下没有支持的文件")
        added = []
        with self.lock:
            for candidate in paths:
                entry = dict(kind=kind, path=str(candidate), name=candidate.name, options=options or {})
                source_key = (kind, str(candidate), json.dumps(entry["options"], sort_keys=True))
                existing = self.sources.get(source_key)
                if existing:
                    added.append(existing)
                    continue
                entry["id"] = uuid.uuid4().hex
                self.entries[entry["id"]] = entry
                self.sources[source_key] = entry
                added.append(entry)
        return added

    def add_region(self, source_id, start, end, limits, client=None):
        """Register an immutable ROI recipe; never decode or save image pixels."""
        source = self.get(source_id, "brain")
        _, start, end, _ = checked_brain_region(source, start, end, limits, client)
        options = dict(source["options"], region=dict(start=start, end=end))
        name = source["name"] + "_[" + ",".join(map(str, start)) + "]-[" + ",".join(map(str, end)) + "]"
        entry = dict(kind="brain-region", path=source["path"], name=name, options=options)
        key = (entry["kind"], entry["path"], json.dumps(options, sort_keys=True))
        with self.lock:
            if self.entries.get(source_id) is not source:
                raise ValueError("全脑数据源已移除，请重新添加")
            if key in self.sources:
                return self.sources[key]
            entry["id"] = uuid.uuid4().hex
            self.entries[entry["id"]] = self.sources[key] = entry
        return entry


def swc_packet(path, crop=None, *, swc=None):
    from swclib.data.swc import Swc

    if swc is None:
        swc = Swc(str(path))
    if crop:
        start, end = region(crop.get("start"), crop.get("end"))
        swc = swc.read_region(start, end)
    nodes = list(swc.nodes.values())
    count = len(nodes)
    positions = np.asarray([[n["x"], n["y"], n["z"]] for n in nodes], dtype="<f4").reshape(-1, 3)
    if not np.isfinite(positions).all():
        raise ValueError("SWC 包含非有限或超出 float32 范围的坐标")
    indices = {n["id"]: i for i, n in enumerate(nodes)}
    edges = np.asarray([(i, indices[n["parent"]]) for i, n in enumerate(nodes)
                        if n["parent"] in indices and n["parent"] != n["id"]], dtype="<u4").reshape(-1, 2)
    degree = np.bincount(edges.ravel(), minlength=count)
    roots = np.asarray([n["parent"] not in indices for n in nodes], dtype=bool)
    colors = np.tile(np.array([255, 205, 71, 255], dtype="u1"), (count, 1))
    colors[degree == 2] = (249, 87, 103, 255)
    colors[degree > 2] = (63, 216, 147, 255)
    colors[roots] = (79, 151, 255, 255)
    bounds = [positions.min(0).tolist(), positions.max(0).tolist()] if count else None
    return packet(dict(kind="swc", count=count, edge_count=len(edges), bounds=bounds,
                       roots=int(roots.sum()), coordinate_order="XYZ"), positions, colors, edges)


def available_memory():
    """Linux available memory/cgroup headroom, portable conservative fallback."""
    candidates = []
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    candidates.append(int(line.split()[1]) * 1024)
                    break
    except OSError:
        try:
            candidates.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError, AttributeError):
            pass
    for maximum, current in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                             ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
        try:
            candidates.append(max(0, int(Path(maximum).read_text()) - int(Path(current).read_text())))
        except (OSError, ValueError):
            pass
    return min(candidates) if candidates else 512 * MIB


class Limits:
    def __init__(self, volume_mib=DEFAULT_VOLUME_MIB):
        self.volume_bytes = int(volume_mib) * MIB

    def describe(self, dtype="uint16", scratch=0, client=None):
        client = client or {}
        itemsize = np.dtype(dtype).itemsize
        available = available_memory()
        budget = min(int(available * 0.20), 2 * 1024 ** 3)
        # Raw volume + decoder output, transfer and GPU copies, normalization workspace.
        server_voxels = max(0, budget - int(scratch) - 2 * MIB) // (2 * itemsize + 24)
        client_mib = float(client.get("volume_mib", DEFAULT_VOLUME_MIB))
        edge = int(client.get("max_texture_size", 2048))
        if not math.isfinite(client_mib) or client_mib <= 0 or edge < 1:
            raise ValueError("无效的浏览器显存限制")
        edge = min(edge, MAX_VOLUME_EDGE)
        maximum = max(0, min(MAX_VOLUME_VOXELS, edge ** 3, self.volume_bytes // 4,
                             int(client_mib * MIB) // 4, server_voxels))
        cube = min(edge, int(maximum ** (1 / 3)))
        while (cube + 1) ** 3 <= maximum and cube < edge:
            cube += 1
        return dict(max_voxels=maximum, max_edge=edge, suggested_cube=cube,
                    available_mib=round(available / MIB), budget_mib=round(budget / MIB),
                    scratch_mib=round(scratch / MIB, 2), display_mib=round(maximum * 4 / MIB, 2))

    def check(self, shape, dtype, scratch=0, client=None):
        limit = self.describe(dtype, scratch, client)
        count = math.prod(int(v) for v in shape)
        if min(shape) < 1 or max(shape) > limit["max_edge"] or count > limit["max_voxels"]:
            raise ValueError(f"区域 {tuple(shape[::-1])} (XYZ) 超过当前显示上限："
                             f"{limit['max_voxels']:,} 体素，单轴 ≤ {limit['max_edge']}；"
                             f"建议立方体边长 ≤ {limit['suggested_cube']}。请缩小 crop。")
        return limit


def image_info(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            shape, dtype = series.shape, series.dtype
            if "S" in series.axes or len(shape) not in (2, 3) or not series.axes.endswith("YX"):
                raise ValueError("仅支持二维/三维灰度 TIFF（ZYX），请先选择通道/时间点")
    elif suffix == ".npy":
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        shape, dtype = data.shape, data.dtype
        del data
    elif suffix == ".nrrd":
        import nrrd
        from swclib.whole_brain.fntreader import _header_dtype
        header = nrrd.read_header(str(path))
        shape, dtype = tuple(int(v) for v in header["sizes"][::-1]), _header_dtype(header)
    else:
        raise ValueError("支持的图像格式：TIFF、NRRD、NPY")
    dtype = np.dtype(dtype)
    if len(shape) not in (2, 3) or dtype.kind not in "uifb":
        raise ValueError("仅支持二维/三维标量灰度图像；数组轴顺序为 ZYX")
    shape = (1,) + tuple(shape) if len(shape) == 2 else tuple(shape)
    return shape, dtype


def brain_reader(entry):
    from swclib.whole_brain import WholeBrainReader
    opts = entry["options"]
    mode = opts["mode"]
    kwargs = dict(channel=opts.get("channel", "ch00")) if mode == "fnt" else dict(
        slice_name_pattern=opts.get("pattern", r".*?(\d+)\.tif"),
        slice_ext=opts.get("extension", ".tif"))
    return WholeBrainReader(entry["path"], mode=mode, axis_order="ZYX", **kwargs)


def brain_scratch(reader):
    # One worker: include the decoded cube / full slice even for a tiny ROI.
    shape = reader.cube_shape if reader.mode == "fnt" else (reader.height, reader.width)
    cache = 16 * MIB if reader.mode == "slice" else 0
    return math.prod(shape) * np.dtype(reader.dtype).itemsize * 3 + cache


def checked_brain_region(entry, start, end, limits, client=None):
    start, end = region(start, end, integer=True)
    reader = brain_reader(entry)
    shape = tuple(b - a for a, b in zip(start, end))[::-1]
    limit = limits.check(shape, reader.dtype, brain_scratch(reader), client)
    lower, upper = reader.get_bbox()
    if any(a < lo or b > hi for a, b, lo, hi in zip(start, end, lower, upper)):
        raise ValueError("裁剪区域超出全脑数据边界")
    return reader, start, end, limit


def describe_image(entry, limits, client=None):
    if entry["kind"] == "brain-region":
        roi = entry["options"]["region"]
        reader, start, end, limit = checked_brain_region(entry, roi["start"], roi["end"], limits, client)
        return dict(shape=[b - a for a, b in zip(start, end)], dtype=str(np.dtype(reader.dtype)),
                    bounds=[start, end], origin=start, limits=limit)
    if entry["kind"] == "brain":
        reader = brain_reader(entry)
        shape, dtype = reader.shape, np.dtype(reader.dtype)
        bounds, scratch = reader.get_bbox(), brain_scratch(reader)
    else:
        shape, dtype = image_info(entry["path"])
        bounds, scratch = ((0, 0, 0), shape[::-1]), 0
    return dict(shape=list(shape[::-1]), dtype=str(dtype), bounds=bounds,
                limits=limits.describe(dtype, scratch, client))


def volume_packet(entry, request, limits):
    client = request.get("client")
    if entry["kind"] in ("brain", "brain-region"):
        if entry["kind"] == "brain-region":
            if any(key in request for key in ("start", "end", "origin")):
                raise ValueError("小块区域已固定，请新增条目来读取其他区域")
            roi = entry["options"]["region"]
        else:
            roi = request
        reader, start, end, limit = checked_brain_region(entry, roi.get("start"), roi.get("end"), limits, client)
        # Bound output before invoking a reader, including its backend validation.
        options = dict(num_workers=1)
        if reader.mode == "slice":
            options["mode"] = "raster"
            import rasterio
            with rasterio.Env(GDAL_CACHEMAX=16 * MIB):
                data = reader.read_region(start[::-1], end[::-1], **options)
        else:
            data = reader.read_region(start[::-1], end[::-1], **options)
        origin = start
    else:
        shape, dtype = image_info(entry["path"])
        limit = limits.check(shape, dtype, client=client)
        suffix = Path(entry["path"]).suffix.lower()
        if suffix in (".tif", ".tiff"):
            import tifffile
            data = tifffile.imread(entry["path"])
        elif suffix == ".nrrd":
            import nrrd
            data, _ = nrrd.read(entry["path"], index_order="C")
        else:
            data = np.load(entry["path"], mmap_mode="r", allow_pickle=False)
        data = data.reshape(shape)
        origin = vector(request.get("origin", [0, 0, 0]), "origin")
    if data.dtype.kind == "f" and not all(np.isfinite(plane).all() for plane in data):
        raise ValueError("图像含 NaN 或无穷值，请先清理数据")
    low, high = float(data.min()), float(data.max())
    display = np.empty(data.shape, dtype="<f4")
    scale = 1.0 / (high - low) if high > low else 0.0
    # Bound normalization scratch even for wide, shallow image volumes.
    for source, target in zip(data, display):
        for row in range(source.shape[0]):
            target[row] = (source[row].astype(np.float64) - low) * scale if high > low else float(high > 0)
    del data
    sample = display.ravel()[::max(1, display.size // 250000)]
    lo, hi = np.percentile(sample, [1, 99.5]).tolist()
    if hi <= lo:
        lo, hi = 0, 1
    return packet(dict(kind="volume", shape=list(display.shape[::-1]), origin=origin,
                       intensity_range=[low, high], window=[lo, hi],
                       limits=limit, coordinate_order="XYZ", storage_order="ZYX",
                       display_dtype="float32"), display)
