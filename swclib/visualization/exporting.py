"""Atomic JSON export with explicit overwrite confirmation from the client.

The single writer thread is started before request threads install seccomp.
Source parsing and statistics still run exclusively in read-only workers.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import uuid


def _ready():
    return True


class ExportExistsError(ValueError):
    def __init__(self, path):
        self.path = str(path)
        super().__init__(f"目标文件已存在：{path}，请确认是否覆盖")


def export_path(path):
    if not isinstance(path, str) or not path.strip():
        raise ValueError("请输入导出目录或 JSON 文件路径")
    target = Path(path).expanduser()
    if target.is_dir():
        target = target / "brain_hemisphere.json"
    elif target.suffix.lower() != ".json":
        if not target.exists():
            raise FileNotFoundError("导出目录不存在；指定文件时请使用 .json 后缀")
        raise ValueError("导出文件必须使用 .json 后缀")
    # Resolve only the parent: replacing a symlink must not overwrite its target.
    return target.parent.resolve(strict=True) / target.name


def write_report(path, report, *, overwrite=False):
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite 必须为布尔值")
    target = export_path(path)
    destination, filename = target.parent, target.name
    payload = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=4).encode("utf-8") + b"\n"
    descriptor = os.open(destination, os.O_RDONLY | os.O_DIRECTORY)
    temporary = ".brain_hemisphere-" + uuid.uuid4().hex + ".tmp"
    created = False
    try:
        # A first attempt never replaces an existing file or link.
        try:
            os.stat(filename, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if not overwrite:
                raise ExportExistsError(target)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=descriptor)
        created = True
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush();os.fsync(stream.fileno())
        try:
            if overwrite:
                os.replace(temporary, filename, src_dir_fd=descriptor, dst_dir_fd=descriptor)
                created = False
            else:
                # Fail rather than overwrite if another export won the race.
                os.link(temporary, filename, src_dir_fd=descriptor, dst_dir_fd=descriptor, follow_symlinks=False)
        except FileExistsError as exc:
            raise ExportExistsError(target) from exc
        return str(target)
    finally:
        if created:
            os.unlink(temporary, dir_fd=descriptor)
        os.close(descriptor)


class ExportWriter:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="viewer-json-export")
        self.pool.submit(_ready).result()

    def write(self, path, report, *, overwrite=False):
        return self.pool.submit(write_report, path, report, overwrite=overwrite).result()

    def close(self):
        self.pool.shutdown(wait=True)
