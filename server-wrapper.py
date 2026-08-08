#!/usr/bin/env python3
"""Whisper HTTP server wrapper — accepts any audio format, converts to WAV, transcribes."""

import base64
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

# Paths derive from the running user's home so the same file is correct on
# both hosts (M1 dev: /Users/tomvan, M4 prod: /Users/tomtomxyz). Before
# picklOS#444 the two copies diverged only in these hardcoded prefixes.
HOME = os.path.expanduser("~")
WHISPER_CLI = os.path.join(HOME, "projects/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.path.join(HOME, "projects/whisper.cpp/models/ggml-base.en.bin")
FFMPEG = "/opt/homebrew/bin/ffmpeg"
PORT = 8178

# picklOS#444 persistent-engine adapter. When WHISPER_ENGINE_URL is set
# (e.g. http://127.0.0.1:8380/inference — the com.vi.whisper-engine
# whisper-server), the ffmpeg-normalized WAV is POSTed there instead of
# spawning whisper-cli per request. Unset -> behavior identical to before.
ENGINE_TIMEOUT = 20  # seconds; on expiry we fall back to the spawn path


def _log_fallback(reason):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(
        f"[{ts}] ENGINE FALLBACK -> whisper-cli spawn path: {reason}",
        file=sys.stderr,
        flush=True,
    )


def _transcribe_via_engine(wav_path, engine_url):
    """POST the normalized WAV to the persistent whisper-server engine.

    S419 spike nondeterminism note: a persistent whisper-server reusing one
    whisper_state across whisper_full calls can occasionally flip a marginal
    word on identical input bytes (2-3/10 on near-tie clips); a fresh-boot
    server is deterministic (4/4). Falsified as ffmpeg / Metal / temp-fallback
    RNG / params — hypothesis points at reused whisper_state upstream. Not a
    blocker for voice chat (real-mic floor is higher), but an upstream-issue
    search is still owed before/at cutover (picklOS#444).

    The engine's boot flags (-bs 5 -bo 5 -nt) match prod whisper-cli; we send
    ONLY the `file` field so no per-request form field overrides them.
    """
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()
    boundary = uuid.uuid4().hex
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + wav_bytes
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(
        engine_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=ENGINE_TIMEOUT) as resp:
        if resp.status != 200:
            raise RuntimeError(f"engine HTTP {resp.status}")
        payload = json.loads(resp.read().decode("utf-8"))
    text = payload["text"]  # missing key -> KeyError -> fallback
    if not isinstance(text, str):
        raise RuntimeError("engine JSON 'text' is not a string")
    # Contract delta 3 (S419 spike): engine text carries a leading space +
    # trailing newline(s) that stripped whisper-cli stdout does not. .strip()
    # is the exact treatment the spawn path applies to cli stdout, so response
    # bytes stay byte-compatible with today whenever the text matches.
    return text.strip()


def _transcribe_via_cli(wav_path):
    """The pre-#444 per-request spawn path — byte-identical known-good arm."""
    result = subprocess.run(
        [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path, "--no-timestamps", "-nt"],
        capture_output=True, timeout=30,
    )
    return result.stdout.decode().strip()


def transcribe(wav_path):
    """Engine if configured; ANY engine failure (connection refused, timeout,
    non-200, unparseable JSON) falls back to the spawn path with one loud
    ISO-stamped log line. The fallback IS the rollback: production keeps
    working per-request if the engine dies."""
    engine_url = os.environ.get("WHISPER_ENGINE_URL")
    if engine_url:
        try:
            return _transcribe_via_engine(wav_path, engine_url)
        except Exception as e:
            _log_fallback(f"{type(e).__name__}: {e}")
    return _transcribe_via_cli(wav_path)


class TranscribeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path not in ("/transcribe", "/inference"):
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        content_type = self.headers.get("Content-Type", "")

        audio_bytes = None

        if "application/json" in content_type:
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                b64 = data.get("audioBase64", "")
                if b64:
                    audio_bytes = base64.b64decode(b64)
            except Exception as e:
                self.send_error(400, f"Invalid JSON: {e}")
                return
        elif "multipart/form-data" in content_type:
            # Simple multipart parsing — extract first file
            body = self.rfile.read(length)
            boundary = content_type.split("boundary=")[-1].encode()
            parts = body.split(b"--" + boundary)
            for part in parts:
                if b"filename=" in part:
                    # Extract file content after double CRLF
                    idx = part.find(b"\r\n\r\n")
                    if idx >= 0:
                        audio_bytes = part[idx + 4:].rstrip(b"\r\n--")
                    break
        else:
            # Treat as raw audio
            audio_bytes = self.rfile.read(length)

        if not audio_bytes:
            self.send_error(400, "No audio data")
            return

        try:
            with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as inp:
                inp.write(audio_bytes)
                input_path = inp.name

            wav_path = input_path + ".wav"

            # Convert to 16kHz mono WAV using ffmpeg
            conv = subprocess.run(
                [FFMPEG, "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", wav_path, "-y"],
                capture_output=True, timeout=15,
            )
            if conv.returncode != 0:
                self.send_error(500, f"ffmpeg error: {conv.stderr.decode()[:200]}")
                return

            # Transcribe: persistent engine when WHISPER_ENGINE_URL is set,
            # else the original whisper-cli spawn (picklOS#444)
            text = transcribe(wav_path)

            # Clean up
            os.unlink(input_path)
            os.unlink(wav_path)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"text": text}).encode())

        except subprocess.TimeoutExpired:
            self.send_error(504, "Transcription timeout")
        except Exception as e:
            self.send_error(500, str(e)[:200])

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "model": "base.en"}).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), TranscribeHandler)  # 0.0.0.0 needed for Docker host.docker.internal; macOS firewall blocks LAN
    print(f"Whisper transcription server running on port {PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()
