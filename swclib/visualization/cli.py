"""Installed console entry point."""

import argparse
import signal
import webbrowser

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(description="SWC / image viewer (local WebGL2 website)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--open", action="store_true", help="Open the website in your browser")
    parser.add_argument("--swc", action="append", default=[], metavar="PATH", help="SWC file/directory; repeatable")
    parser.add_argument("--image", action="append", default=[], metavar="PATH", help="Image file/directory; repeatable")
    parser.add_argument("--max-volume-mib", type=int, default=128,
                        help="Float32 volume budget in MiB (default: 128); each axis is always <= 300")
    parser.add_argument("--swc-workers", type=int, choices=range(1, 9), metavar="1..8",
                        help="Parallel SWC reader processes (default: auto, at most 4); 1 for serial reads")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535 or args.max_volume_mib < 1:
        parser.error("port must be in 0..65535 and max-volume-mib must be positive")
    from .server import create_server

    try:
        server = create_server(args.host, args.port, args.max_volume_mib, args.swc_workers)
    except (RuntimeError, OSError) as exc:
        parser.error(str(exc))
    def stop(signum, frame):
        raise KeyboardInterrupt
    previous_terminate = signal.signal(signal.SIGTERM, stop)
    try:
        for kind, paths in (("swc", args.swc), ("image", args.image)):
            for path in paths:
                server.catalog.add(path, kind)
        host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
        url = f"http://{host}:{server.server_port}"
        print(f"SWC Studio {__version__}  {url}", flush=True)
        print("Press Ctrl+C to stop. Paths refer to files on this server.", flush=True)
        if args.open:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        signal.signal(signal.SIGTERM, previous_terminate)


if __name__ == "__main__":
    main()
