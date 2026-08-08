#!/usr/bin/env python3
"""Tests for the picklOS#444 persistent-engine adapter in server-wrapper.py.

Run:  /usr/bin/python3 test-server-wrapper.py -v
(interpreter matches the com.vi.whisper-server plist: /usr/bin/python3)

Design:
- The whisper-cli SPAWN IS STUBBED (module-level function patch) — no test
  ever executes the real whisper-cli binary or needs the model file.
- ffmpeg runs for REAL on a tiny generated WAV fixture, so the normalize leg
  of the pipeline is exercised (FFMPEG path must exist — it does on M1 + M4).
- The engine is a local stub HTTP server that records every request it
  receives and returns a configurable response — so tests can assert both
  that it WAS hit (engine mode) and that it was NEVER hit (negative control).
"""

import base64
import contextlib
import importlib.util
import io
import json
import os
import re
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- fixtures


def load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "server_wrapper", os.path.join(HERE, "server-wrapper.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # __name__ != "__main__" -> no server starts
    return mod


def tiny_wav_bytes():
    """A real (tiny) 16kHz mono 16-bit WAV: 0.1s of silence."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


WAV = tiny_wav_bytes()

CLI_TEXT = "Hello world."          # what the stubbed spawn path returns
ENGINE_RAW = " Hello world.\n"     # same text as the engine emits it:
                                   # leading space + trailing \n (delta 3)


class StubEngineHandler(BaseHTTPRequestHandler):
    """Records requests; replies per the owning server's `behavior`."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        self.server.requests.append(
            {
                "path": self.path,
                "content_type": self.headers.get("Content-Type", ""),
                "body": body,
            }
        )
        behavior = self.server.behavior
        if behavior.get("sleep"):
            time.sleep(behavior["sleep"])
        status = behavior.get("status", 200)
        payload = behavior.get("body", json.dumps({"text": ENGINE_RAW}).encode())
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client gave up (timeout arm) — fine

    def log_message(self, fmt, *args):
        pass


def start_stub_engine():
    server = ThreadingHTTPServer(("127.0.0.1", 0), StubEngineHandler)
    server.requests = []
    server.behavior = {}
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def closed_port_url():
    """A URL nothing listens on (bind, read port, close) — connection refused."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}/inference"


FALLBACK_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\] ENGINE FALLBACK -> whisper-cli spawn path: "
)

# ------------------------------------------------------------------- tests


class WrapperTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_wrapper()
        cls.wrapper = cls.mod.HTTPServer(
            ("127.0.0.1", 0), cls.mod.TranscribeHandler
        )
        cls.wrapper_port = cls.wrapper.server_address[1]
        threading.Thread(target=cls.wrapper.serve_forever, daemon=True).start()
        cls.engine = start_stub_engine()
        cls.engine_url = (
            f"http://127.0.0.1:{cls.engine.server_address[1]}/inference"
        )

    @classmethod
    def tearDownClass(cls):
        cls.wrapper.shutdown()
        cls.engine.shutdown()

    def setUp(self):
        mod = self.mod
        self.cli_calls = []
        self.engine_fn_calls = []
        self._orig_cli = mod._transcribe_via_cli
        self._orig_engine = mod._transcribe_via_engine
        self._orig_timeout = mod.ENGINE_TIMEOUT

        def stub_cli(wav_path):
            self.cli_calls.append(wav_path)
            return CLI_TEXT

        real_engine = mod._transcribe_via_engine

        def counting_engine(wav_path, engine_url):
            self.engine_fn_calls.append(engine_url)
            return real_engine(wav_path, engine_url)

        mod._transcribe_via_cli = stub_cli
        mod._transcribe_via_engine = counting_engine
        self.engine.requests.clear()
        self.engine.behavior = {}
        os.environ.pop("WHISPER_ENGINE_URL", None)

    def tearDown(self):
        self.mod._transcribe_via_cli = self._orig_cli
        self.mod._transcribe_via_engine = self._orig_engine
        self.mod.ENGINE_TIMEOUT = self._orig_timeout
        os.environ.pop("WHISPER_ENGINE_URL", None)

    # -- HTTP helpers against the wrapper under test

    def post(self, path, body, content_type):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.wrapper_port}{path}",
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read()

    def get(self, path):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.wrapper_port}{path}", timeout=10
        ) as resp:
            return resp.status, resp.read()

    def post_multipart(self, path):
        b = "testboundary123"
        body = (
            (
                f"--{b}\r\n"
                'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                "Content-Type: audio/wav\r\n\r\n"
            ).encode()
            + WAV
            + f"\r\n--{b}--\r\n".encode()
        )
        return self.post(path, body, f"multipart/form-data; boundary={b}")

    def post_json(self, path):
        body = json.dumps(
            {"audioBase64": base64.b64encode(WAV).decode()}
        ).encode()
        return self.post(path, body, "application/json")

    def post_raw(self, path):
        return self.post(path, WAV, "application/octet-stream")


class TestSpawnPathAndNegativeControl(WrapperTestBase):
    def test_env_unset_uses_spawn_path_engine_never_contacted(self):
        """Negative control: env unset -> spawn path; engine function never
        invoked AND zero requests hit the stub."""
        status, body = self.post_multipart("/transcribe")
        self.assertEqual(status, 200)
        self.assertEqual(body, json.dumps({"text": CLI_TEXT}).encode())
        self.assertEqual(len(self.cli_calls), 1)
        self.assertEqual(self.engine_fn_calls, [])
        self.assertEqual(self.engine.requests, [])

    def test_all_request_shapes_both_paths_spawn_mode(self):
        """/transcribe + /inference x JSON/multipart/raw — contract intact."""
        for path in ("/transcribe", "/inference"):
            for shape, fn in (
                ("json", self.post_json),
                ("multipart", self.post_multipart),
                ("raw", self.post_raw),
            ):
                with self.subTest(path=path, shape=shape):
                    status, body = fn(path)
                    self.assertEqual(status, 200)
                    self.assertEqual(
                        body, json.dumps({"text": CLI_TEXT}).encode()
                    )
        self.assertEqual(self.engine.requests, [])


class TestEngineMode(WrapperTestBase):
    def test_engine_text_stripped_byte_compatible(self):
        """Engine mode strips ' Hello world.\\n' to the exact bytes the spawn
        path returns for the same transcription."""
        _, spawn_body = self.post_multipart("/transcribe")

        os.environ["WHISPER_ENGINE_URL"] = self.engine_url
        status, engine_body = self.post_multipart("/transcribe")
        self.assertEqual(status, 200)
        self.assertEqual(engine_body, spawn_body)  # byte-compatible
        self.assertEqual(engine_body, json.dumps({"text": CLI_TEXT}).encode())

        # exactly one engine request, none via the cli after the first call
        self.assertEqual(len(self.engine.requests), 1)
        self.assertEqual(len(self.cli_calls), 1)  # only the spawn-mode call

        req = self.engine.requests[0]
        self.assertEqual(req["path"], "/inference")
        self.assertTrue(req["content_type"].startswith("multipart/form-data"))
        self.assertIn(b'name="file"', req["body"])
        self.assertIn(b"RIFF", req["body"])  # ffmpeg-normalized WAV forwarded

    def test_all_request_shapes_both_paths_engine_mode(self):
        os.environ["WHISPER_ENGINE_URL"] = self.engine_url
        for path in ("/transcribe", "/inference"):
            for shape, fn in (
                ("json", self.post_json),
                ("multipart", self.post_multipart),
                ("raw", self.post_raw),
            ):
                with self.subTest(path=path, shape=shape):
                    status, body = fn(path)
                    self.assertEqual(status, 200)
                    self.assertEqual(
                        body, json.dumps({"text": CLI_TEXT}).encode()
                    )
        self.assertEqual(len(self.engine.requests), 6)
        self.assertEqual(self.cli_calls, [])


class TestEngineFallback(WrapperTestBase):
    def _assert_fell_back(self, stderr_text):
        lines = [l for l in stderr_text.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1, f"expected 1 loud line, got: {lines}")
        self.assertRegex(lines[0], FALLBACK_RE)

    def test_engine_down_falls_back_loudly(self):
        """The load-bearing arm: connection refused -> spawn path + loud line."""
        os.environ["WHISPER_ENGINE_URL"] = closed_port_url()
        cap = io.StringIO()
        with contextlib.redirect_stderr(cap):
            status, body = self.post_multipart("/transcribe")
        self.assertEqual(status, 200)
        self.assertEqual(body, json.dumps({"text": CLI_TEXT}).encode())
        self.assertEqual(len(self.cli_calls), 1)
        self._assert_fell_back(cap.getvalue())

    def test_engine_non_200_falls_back(self):
        os.environ["WHISPER_ENGINE_URL"] = self.engine_url
        self.engine.behavior = {"status": 500, "body": b"boom"}
        cap = io.StringIO()
        with contextlib.redirect_stderr(cap):
            status, body = self.post_multipart("/transcribe")
        self.assertEqual(status, 200)
        self.assertEqual(body, json.dumps({"text": CLI_TEXT}).encode())
        self.assertEqual(len(self.cli_calls), 1)
        self.assertEqual(len(self.engine.requests), 1)  # it WAS tried
        self._assert_fell_back(cap.getvalue())

    def test_malformed_engine_json_falls_back(self):
        os.environ["WHISPER_ENGINE_URL"] = self.engine_url
        for label, payload in (
            ("not-json", b"this is not json"),
            ("missing-text-key", json.dumps({"result": "x"}).encode()),
            ("text-not-string", json.dumps({"text": 42}).encode()),
        ):
            with self.subTest(payload=label):
                self.cli_calls.clear()
                self.engine.behavior = {"status": 200, "body": payload}
                cap = io.StringIO()
                with contextlib.redirect_stderr(cap):
                    status, body = self.post_multipart("/transcribe")
                self.assertEqual(status, 200)
                self.assertEqual(
                    body, json.dumps({"text": CLI_TEXT}).encode()
                )
                self.assertEqual(len(self.cli_calls), 1)
                self._assert_fell_back(cap.getvalue())

    def test_engine_timeout_falls_back(self):
        os.environ["WHISPER_ENGINE_URL"] = self.engine_url
        self.mod.ENGINE_TIMEOUT = 0.5
        self.engine.behavior = {"sleep": 1.5}
        cap = io.StringIO()
        with contextlib.redirect_stderr(cap):
            status, body = self.post_multipart("/transcribe")
        self.assertEqual(status, 200)
        self.assertEqual(body, json.dumps({"text": CLI_TEXT}).encode())
        self.assertEqual(len(self.cli_calls), 1)
        self._assert_fell_back(cap.getvalue())


class TestHealth(WrapperTestBase):
    EXPECTED = json.dumps({"status": "ok", "model": "base.en"}).encode()

    def test_health_unchanged_spawn_mode(self):
        status, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, self.EXPECTED)  # includes the `model` field

    def test_health_unchanged_engine_mode(self):
        os.environ["WHISPER_ENGINE_URL"] = self.engine_url
        status, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(body, self.EXPECTED)
        self.assertEqual(self.engine.requests, [])  # /health never hits engine


if __name__ == "__main__":
    unittest.main(verbosity=2)
