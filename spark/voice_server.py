"""Minimal HTTPS server for voice.html — serves static file, proxies to v4 API."""
import http.server, ssl, os

PORT = 8422
CERT = os.path.expanduser("~/Vybn/spark/ts.crt")
KEY = os.path.expanduser("~/Vybn/spark/ts.key")
ROOT = os.path.expanduser("~/Vybn/spark")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)
    def log_message(self, fmt, *args):
        pass  # quiet

ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(CERT, KEY)

srv = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
print(f"voice server on https://0.0.0.0:{PORT}")
srv.serve_forever()
