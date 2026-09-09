"""Zero-dependency web server for the three.js spinor demos.

    python demos/server.py            # http://localhost:8765
    python demos/server.py --port 9000

Static files are served from ``demos/web``; ``POST /api/<name>`` with a JSON
body runs the matching handler in ``demos/api.py`` (which calls spinor_lib)
and returns JSON. Only the Python standard library, torch and numpy are needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from demos.api import HANDLERS  # noqa: E402

WEB_DIR = os.path.join(HERE, "web")


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if not self.path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        name = self.path[len("/api/"):].split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            params = json.loads(self.rfile.read(length) or b"{}")
            fn = HANDLERS.get(name)
            if fn is None:
                raise KeyError(f"no handler named {name!r}")
            body = json.dumps(fn(params), allow_nan=False).encode()  # NaN is not JSON
            status = HTTPStatus.OK
        except Exception as exc:  # report to the browser instead of dying
            traceback.print_exc()
            body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
            status = HTTPStatus.BAD_REQUEST
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # API calls fire many times per second while sliders move; keep quiet.
        if not (args and "/api/" in str(args[0])):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"spinor demos: http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
