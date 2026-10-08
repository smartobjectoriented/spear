"""The context window is the server's, unless the operator says otherwise.

Started without SPEAR_CTX, the client budgeted 32768 tokens against a server
holding 524288, and compacted a turn's history twice with a twentieth of the
window in use. Only one launcher mode asked the server; every other way of
starting the client trusted the default. The client now asks itself, and
says where its number came from.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli import model_io


def serve(test, routes):
    """A throwaway HTTP server answering GET on the given paths."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in routes:
                self.send_response(404)
                self.end_headers()
                return

            body = json.dumps(routes[self.path]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    test.addCleanup(server.server_close)
    test.addCleanup(server.shutdown)

    return f"http://127.0.0.1:{server.server_port}/v1"


LLAMA = {"/props": {"default_generation_settings": {"n_ctx": 131072}},
         "/v1/models": {"data": [{"id": "m", "meta": {"n_ctx_train": 262144}}]}}
VLLM = {"/v1/models": {"data": [{"id": "m", "max_model_len": 65536}]}}


class Discovery(unittest.TestCase):
    def test_llama_cpp_reports_its_slot_window(self):
        self.assertEqual(model_io.served_context_window(serve(self, LLAMA)),
                         (131072, "/props"))

    def test_vllm_reports_max_model_len(self):
        self.assertEqual(model_io.served_context_window(serve(self, VLLM)),
                         (65536, "/v1/models"))

    def test_a_training_length_is_not_a_served_window(self):
        routes = {"/v1/models": LLAMA["/v1/models"]}
        value, _ = model_io.served_context_window(serve(self, routes))

        self.assertIsNone(value)

    def test_a_base_without_v1_is_accepted(self):
        base = serve(self, LLAMA).removesuffix("/v1")

        self.assertEqual(model_io.served_context_window(base)[0], 131072)


class Resolution(unittest.TestCase):
    def test_an_explicit_value_wins_and_the_server_is_not_asked(self):
        def discover(_):
            raise AssertionError("the server must not be asked")

        self.assertEqual(model_io.resolve_context_window(
            {"SPEAR_CTX": "40000"}, "http://unused/v1", discover=discover),
            (40000, "SPEAR_CTX"))

    def test_the_server_value_is_used_and_attributed(self):
        self.assertEqual(model_io.resolve_context_window(
            {}, serve(self, LLAMA)), (131072, "server /props"))
        self.assertEqual(model_io.resolve_context_window(
            {}, serve(self, VLLM)), (65536, "server /v1/models"))

    def test_silence_falls_back_and_says_so(self):
        value, source = model_io.resolve_context_window({}, serve(self, {}))

        self.assertEqual(value, model_io.DEFAULT_CTX)
        self.assertTrue(source.startswith("default:"), source)
        self.assertNotIn("server /", source)

    def test_an_unreachable_server_falls_back_and_says_so(self):
        value, source = model_io.resolve_context_window(
            {}, "http://127.0.0.1:9/v1",
            discover=lambda base: model_io.served_context_window(base, timeout=1))

        self.assertEqual(value, model_io.DEFAULT_CTX)
        self.assertIn("did not report", source)

    def test_a_provider_with_no_such_endpoint_is_not_asked(self):
        def discover(_):
            raise AssertionError("the server must not be asked")

        value, source = model_io.resolve_context_window(
            {}, "http://unused/v1", provider="anthropic", discover=discover)

        self.assertEqual(value, model_io.DEFAULT_CTX)
        self.assertTrue(source.startswith("default:"))


if __name__ == "__main__":
    unittest.main()
