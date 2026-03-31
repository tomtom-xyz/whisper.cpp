#!/usr/bin/env python3
"""Whisper HTTP server wrapper — accepts any audio format, converts to WAV, transcribes."""

import base64
import json
import os
import subprocess
import tempfile
from http.server import HTTPServer, BaseHTTPRequestHandler

WHISPER_CLI = "/Users/tomvan/projects/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/Users/tomvan/projects/whisper.cpp/models/ggml-base.en.bin"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
PORT = 8178


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

            # Transcribe with whisper-cli
            result = subprocess.run(
                [WHISPER_CLI, "-m", WHISPER_MODEL, "-f", wav_path, "--no-timestamps", "-nt"],
                capture_output=True, timeout=30,
            )

            text = result.stdout.decode().strip()

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
