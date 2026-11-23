"""Indexed prediction-only replay for hierarchical v1 and incremental direct v2 traces."""

import json
from pathlib import Path
import threading
import uuid

import numpy as np

from swclib.data.swc import Swc

MAX_LINE = 16 * 1024 * 1024


def _line(stream):
    line = stream.readline(MAX_LINE + 1)
    if len(line) > MAX_LINE:
        raise ValueError("追踪记录单行超过 16 MiB")
    if not line or not line.endswith(b"\n"):
        return None  # A still-running/crashed writer may leave a partial last line.
    return json.loads(line)


def _edges(nodes):
    return [[nodes[parent][1:], data[1:]] for data in nodes.values()
            for parent in [data[0]] if parent in nodes]


def _snapshot(path):
    if not path.is_file():
        return []
    with np.load(path, allow_pickle=False) as data:
        points, parents = data["points_xyz"], data["parents"]
    if (points.shape != (len(parents), 3) or not np.isfinite(points).all()
            or np.any(parents < -1) or np.any(parents >= len(points))):
        raise ValueError("全脑树记录无效")
    children = np.flatnonzero(parents >= 0)
    return np.stack([points[parents[children]], points[children]], axis=1).tolist()


def _swc_edges(path, scale, offset):
    swc = Swc(str(path))
    nodes = {nid: [node["parent"], *(
        np.array([node[axis] for axis in ("x", "y", "z")]) * scale + offset).tolist()]
        for nid, node in swc.nodes.items()}
    return _edges(nodes)


class Replay:
    def __init__(self, path, gt_path=None):
        path = Path(path).expanduser().resolve(strict=True)
        self.root = path if path.is_dir() else path.parent
        manifest = self.root / "trace_replay.json"
        self.metadata = json.loads(manifest.read_text()) if manifest.is_file() else {}
        sources = ([path] if path.is_file() else [path / "trace_steps.jsonl"]
                   if (path / "trace_steps.jsonl").is_file()
                   else sorted(path.glob("block_results/iter*/trace_steps.jsonl")))
        self.blocks, self.frames, self.warnings = [], [], []
        for source in sources:
            offsets, end = [], None
            with source.open("rb") as stream:
                header = _line(stream)
                if not header or header.get("version") not in (1, 2) or header.get("event") != "header":
                    raise ValueError(f"不支持的追踪记录：{source}")
                if header.get("coordinate_frame") != "global_xyz_voxels":
                    raise ValueError("回放记录必须使用 global_xyz_voxels")
                while True:
                    position = stream.tell()
                    record = _line(stream)
                    if record is None:
                        break
                    if record.get("event") == "prediction":
                        offsets.append(position)
                    elif record.get("event") == "end":
                        end = record
            block = dict(path=source, header=header, offsets=offsets, end=end)
            index = len(self.blocks)
            self.blocks.append(block)
            for step in range(len(offsets)):
                self.frames.append((index, step, "prediction"))
            if header.get("version") == 2 and not (source.parent / "trace_initial.npz").is_file():
                raise ValueError("直接追踪记录缺少 trace_initial.npz")
            if header.get("version") == 1 and not (source.parent / "trace_tree_before.npz").is_file():
                self.warnings.append(f"{source.parent.name} 没有全脑树快照，仅显示局部历史与追踪树")
            if end is None or end.get("status") != "complete":
                self.warnings.append(f"{source.parent.name} 记录未完成：{(end or {}).get('error') or '仍在写入或已中断'}")
        if not self.frames:
            raise ValueError("没有可回放的预测；请开启 infer.save_steps 后重新推理")
        gt = gt_path or self.metadata.get("ground_truth")
        scale = np.asarray(self.metadata.get("swc_to_voxel_scale_xyz", [1, 1, 1]))
        offset = np.asarray(self.metadata.get("swc_to_voxel_offset_xyz", [0, 0, 0]))
        self.gt = _swc_edges(gt, scale, offset) if gt else []
        self.gt_path = str(gt) if gt else None
        self.cached_block, self.applied, self.nodes, self.global_edges = None, -1, {}, []

    def describe(self):
        return dict(frame_count=len(self.frames), prediction_count=sum(len(b["offsets"]) for b in self.blocks),
                    block_count=len(self.blocks), gt_path=self.gt_path, gt_edges_xyz=self.gt,
                    warnings=self.warnings, metadata=self.metadata)

    def frame(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.frames):
            raise ValueError("回放位置超出范围")
        block_index, step, phase = self.frames[index]
        block = self.blocks[block_index]
        folder = block["path"].parent
        wanted = step
        if not block["offsets"]:
            wanted = -1
        if self.cached_block != block_index or self.applied > wanted:
            self.cached_block, self.applied = block_index, -1
            if block["header"].get("version") == 2:
                with np.load(folder / "trace_initial.npz", allow_pickle=False) as initial:
                    points, parents = initial["points_xyz"], initial["parents"]
                if (points.shape != (len(parents), 3) or not np.isfinite(points).all()
                        or np.any(parents < -1) or np.any(parents >= len(points))):
                    raise ValueError("全脑树初始记录无效")
                self.nodes = {node: [int(parent), *point.tolist()]
                              for node, (point, parent) in enumerate(zip(points, parents))}
                self.global_edges = []
            else:
                self.nodes = {0: [-1, *block["header"]["seed_xyz"]]}
                self.global_edges = _snapshot(folder / "trace_tree_before.npz")
                if not (folder / "trace_tree_before.npz").exists():
                    history = block["header"]["history_xyz"]
                    self.global_edges = [[a, b] for a, b in zip(history[:-1], history[1:])]
        with block["path"].open("rb") as stream:
            for local_index in range(self.applied + 1, wanted + 1):
                stream.seek(block["offsets"][local_index])
                record = _line(stream)
                for node in record["removed_nodes"]:
                    self.nodes.pop(node, None)
                for node, *data in record["added_nodes"]:
                    self.nodes[node] = data
                self.applied = local_index
            record = None
            if block["offsets"]:
                stream.seek(block["offsets"][step])
                record = _line(stream)
        last_in_block = step == len(block["offsets"]) - 1
        # The last prediction also carries the outer transaction outcome in old
        # recordings. No separate advance/stitch frames are needed.
        committed = last_in_block and (folder / "trace_tree_after.npz").is_file()
        edges = (_snapshot(folder / "trace_tree_after.npz") if committed
                 else self.global_edges + _edges(self.nodes))
        state = folder / "trace_state.json"
        outcome = json.loads(state.read_text()) if committed and state.is_file() else None
        return dict(index=index, phase=phase, block_index=block_index,
                    iteration=step+1 if block["header"].get("version") == 2 else block["header"].get("iteration"), step=step+1,
                    mode=block["header"].get("mode", "hierarchical"),
                    seed_xyz=record["seed_xyz"] if record else block["header"]["seed_xyz"],
                    traced_edges_xyz=edges, prediction=record, outcome=outcome,
                    end=block["end"])


class ReplayService:
    def __init__(self):
        self.lock = threading.Lock()
        self.replay = None
        self.key = None

    def open(self, path, gt_path=None):
        with self.lock:
            replay = Replay(path, gt_path)
            self.replay, self.key = replay, uuid.uuid4().hex
            return dict(id=self.key, **replay.describe())

    def frame(self, key, index):
        with self.lock:
            if key != self.key or self.replay is None:
                raise ValueError("回放已被替换，请重新加载")
            return self.replay.frame(index)
