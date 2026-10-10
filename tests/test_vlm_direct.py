import http.server
import os
import threading
import unittest
import urllib.request

import smart_prompts


class _Ok(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"data": []}')

    def log_message(self, *args):
        pass


class DirectRetryTest(unittest.TestCase):
    def test_a_dead_proxy_falls_back_to_a_direct_request(self):
        server = http.server.HTTPServer(("127.0.0.1", 0), _Ok)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        saved = {k: os.environ.get(k) for k in ("http_proxy", "HTTP_PROXY", "no_proxy", "NO_PROXY")}
        # a proxy on a port nothing listens on, and nothing exempted from it
        os.environ["http_proxy"] = os.environ["HTTP_PROXY"] = "http://127.0.0.1:9"
        os.environ["no_proxy"] = os.environ["NO_PROXY"] = ""
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/models"
            with self.assertRaises(OSError):
                urllib.request.urlopen(url, timeout=5)           # what failed every job
            with smart_prompts.open_url(urllib.request.Request(url), 5) as response:
                self.assertEqual(response.read(), b'{"data": []}')
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
