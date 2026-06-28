import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "api"))

from optimize import handler

PORT = 8765

# One slow/stalled connection on a single-threaded HTTPServer blocks every
# other request indefinitely; ThreadingHTTPServer + a socket timeout avoids that.
handler.timeout = 10

if __name__ == "__main__":
    server = ThreadingHTTPServer(("localhost", PORT), handler)
    print(f"Serving POST /api/optimize at http://localhost:{PORT}/api/optimize")
    server.serve_forever()
