"""Read-only source service with explicit JSON exports."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import secrets
import threading
from urllib.parse import urlsplit

from . import __version__
from .data import Catalog, DEFAULT_VOLUME_MIB, Limits, describe_image, volume_packet
from .loading import SWCReaders
from .readonly import ReadOnlyGuard
from .exporting import ExportExistsError, ExportWriter
from .hemisphere import DESCRIPTION, hemisphere_counts, plane_parameters
from .plane_optimization import optimize_plane

STATIC = Path(__file__).with_name("static")


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, volume_mib, swc_workers=None):
        self.catalog = Catalog()
        self.limits = Limits(volume_mib)
        self.token = secrets.token_urlsafe(32)
        self.read_lock = threading.Lock()
        self.export_lock = threading.Lock()
        self.readonly = ReadOnlyGuard()
        super().__init__(address, ViewerHandler)
        try:
            self.swc_readers = SWCReaders(swc_workers)
            self.export_writer = ExportWriter()
        except BaseException:
            if hasattr(self, "swc_readers"):
                self.swc_readers.close()
            super().server_close()
            raise

    def server_close(self):
        super().server_close()
        self.swc_readers.close()
        self.export_writer.close()

    def process_request_thread(self, request, client_address):
        try:
            self.readonly.install()
        except Exception:
            logging.getLogger(__name__).exception("Read-only protection failed; request refused")
            try:
                request.sendall(b"HTTP/1.0 503 Read-only protection unavailable\r\nContent-Length: 0\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        super().process_request_thread(request, client_address)


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "SWCLibViewer/" + __version__

    def log_message(self, fmt, *args):
        logging.getLogger(__name__).debug(fmt, *args)

    def respond(self, status, parts, content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(sum(len(p) for p in parts)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        for part in parts:
            self.wfile.write(part)

    def json(self, value, status=200):
        try:
            self.respond(status, [json.dumps(value, ensure_ascii=False, allow_nan=False).encode()])
        except (BrokenPipeError, ConnectionResetError):
            pass  # A closed tab may also disconnect while an error is reported.

    def valid_host(self):
        host = urlsplit("http://" + self.headers.get("Host", "")).hostname
        if self.server.server_address[0] == "127.0.0.1" and host not in ("localhost", "127.0.0.1"):
            self.json({"error": "Invalid local Host"}, 403)
            return False
        return True

    def do_GET(self):
        if not self.valid_host():
            return
        path = urlsplit(self.path).path
        if path == "/api/config":
            self.json(dict(version=__version__, token=self.server.token,
                           read_only=True, filesystem_protection="seccomp",
                           hemisphere_export=True,
                           swc_workers=self.server.swc_readers.workers,
                           entries=self.server.catalog.list(), limits=self.server.limits.describe()))
            return
        resources = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/renderer.js": ("renderer.js", "text/javascript; charset=utf-8"),
                     "/geometry-worker.js": ("geometry-worker.js", "text/javascript; charset=utf-8"),
                     "/plugins.js": ("plugins.js", "text/javascript; charset=utf-8"),
                     "/hemisphere.js": ("hemisphere.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path not in resources:
            self.json({"error": "Not found"}, 404)
            return
        filename, mime = resources[path]
        self.respond(200, [(STATIC / filename).read_bytes()], mime)

    def do_POST(self):
        if not self.valid_host():
            return
        if not secrets.compare_digest(self.headers.get("X-Viewer-Token", ""), self.server.token):
            self.json({"error": "刷新页面后重试（会话令牌无效）"}, 403)
            return
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.json({"error": "Cross-origin request denied"}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("请求大小无效")
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise ValueError("请求必须为 JSON 对象")
            path = urlsplit(self.path).path
            fields = {
                "/api/add": {"path", "kind", "recursive", "options"},
                "/api/remove": {"ids"},
                "/api/swc": {"id", "crop"},
                "/api/info": {"id", "kind", "client"},
                "/api/volume": {"id", "kind", "client", "start", "end", "origin"},
                "/api/region": {"id", "start", "end", "client"},
                "/api/somas": {"id", "rule"},
                "/api/hemisphere/export": {"ids", "plane", "directory", "path", "overwrite"},
                "/api/hemisphere/count": {"ids", "plane"},
                "/api/hemisphere/optimize": {"ids", "plane", "center", "max_angle_deg"},
            }
            if path not in fields:
                self.json({"error": "Not found"}, 404)
                return
            if set(request) - fields[path]:
                raise ValueError("不支持的请求参数；源数据只读，导出仅支持结果 JSON")
            if path in ("/api/hemisphere/export", "/api/hemisphere/count", "/api/hemisphere/optimize"):
                exporting = path.endswith("/export")
                if "path" in request and "directory" in request:
                    raise ValueError("path 和 directory 只能指定一个")
                overwrite = request.get("overwrite", False)
                if not isinstance(overwrite, bool):
                    raise ValueError("overwrite 必须为布尔值")
                ids = request.get("ids")
                if not isinstance(ids, list) or not ids or any(not isinstance(key, str) for key in ids):
                    raise ValueError("请选择至少一个已加载显示的 SWC")
                entries = [self.server.catalog.get(key, "swc") for key in dict.fromkeys(ids)]
                names = [entry["name"] for entry in entries]
                if exporting and len(set(names)) != len(names):
                    raise ValueError("存在同名 SWC，示例格式的 swc_name 无法区分，请分别导出")
                plane = plane_parameters(request.get("plane"))
                if not self.server.export_lock.acquire(blocking=False):
                    self.json({"error": "另一个脑区统计、微调或导出正在进行，请完成后重试"}, 409)
                    return
                try:
                    if path.endswith("/optimize"):
                        summaries = None
                        def evaluate(planes):
                            nonlocal summaries
                            if summaries is None:
                                summaries = self.server.swc_readers.plane_summaries(entries)
                            return self.server.swc_readers.plane_counts(entries, planes, summaries)
                        result = optimize_plane(
                            request["plane"], request.get("center"), request.get("max_angle_deg", 10),
                            evaluate,
                        )
                        self.json(result)
                        return
                    neurons = self.server.swc_readers.hemispheres(entries, request["plane"], include_soma=exporting)
                    if not exporting:
                        self.json(dict(plane=plane, counts=hemisphere_counts(neurons)))
                        return
                    report = dict(description=DESCRIPTION, plane=plane, neurons=neurons)
                    destination = self.server.export_writer.write(
                        request.get("path", request.get("directory")), report, overwrite=overwrite
                    )
                    self.json(dict(path=destination, count=len(neurons)))
                finally:
                    self.server.export_lock.release()
                return
            if path == "/api/remove":
                self.json({"removed": self.server.catalog.remove(request.get("ids"))})
                return
            if path == "/api/add":
                added = self.server.catalog.add(request["path"], request["kind"],
                                                request.get("recursive", False), request.get("options"))
                self.json({"entries": added})
                return
            kind = "swc" if path in ("/api/swc", "/api/somas") else "brain" if path == "/api/region" else request.get("kind", "image")
            entry = self.server.catalog.get(request["id"], kind)
            if path in ("/api/swc", "/api/somas"):
                readers = self.server.swc_readers
                reservation = readers.acquire(entry["path"])
                if reservation is None:
                    self.json({"error": "SWC 读取队列忙，请稍后重试"}, 409)
                    return
                try:
                    if path == "/api/somas":
                        self.json(readers.somas(entry["path"], request.get("rule")))
                    else:
                        self.respond(200, readers.read(entry["path"], request.get("crop")), "application/octet-stream")
                finally:
                    readers.release(reservation)
                return
            # Image decoding stays serial across tabs; SWC has a separate bound.
            if not self.server.read_lock.acquire(blocking=False):
                self.json({"error": "另一个数据请求正在读取，请稍后重试"}, 409)
                return
            try:
                if path == "/api/region":
                    block = self.server.catalog.add_region(entry["id"], request.get("start"), request.get("end"),
                                                           self.server.limits, request.get("client"))
                    self.json({"entry": block})
                elif path == "/api/info":
                    self.json(describe_image(entry, self.server.limits, request.get("client")))
                elif path == "/api/volume":
                    self.respond(200, volume_packet(entry, request, self.server.limits), "application/octet-stream")
                else:
                    self.json({"error": "Not found"}, 404)
            finally:
                self.server.read_lock.release()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ExportExistsError as exc:
            self.json({"error": str(exc), "code": "export_exists", "path": exc.path}, 409)
        except (ValueError, TypeError, KeyError, OSError, IndexError) as exc:
            self.json({"error": str(exc)}, 400)
        except Exception:
            logging.getLogger(__name__).exception("Viewer request failed")
            self.json({"error": "读取失败，详细信息见服务终端"}, 500)


def create_server(host="127.0.0.1", port=8000, volume_mib=DEFAULT_VOLUME_MIB, swc_workers=None):
    return ViewerServer((host, port), volume_mib, swc_workers)
