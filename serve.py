"""Static server for the demo pages with caching disabled.

Chrome's memory cache otherwise serves stale assets while we iterate.
Usage: python3 serve.py [port]
"""

import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8321
    handler = partial(NoCacheHandler, directory="web")
    print(f"serving web/ on http://127.0.0.1:{port} (no-store)")
    HTTPServer(("127.0.0.1", port), handler).serve_forever()
