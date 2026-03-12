#!/usr/bin/env python3
"""
Vybn Voice Server — TTS endpoint + voice sample audition

Serves on 0.0.0.0:8150 (bind to Tailscale interface only in production)
Endpoints:
  GET  /samples              — list available voice samples
  GET  /samples/<name>.wav   — play a voice sample
  POST /speak                — generate speech from text
    Body: {"text": "...", "voice": "en_US-joe-medium"}
  GET  /audition             — HTML page to audition all voices
"""

import json, subprocess, tempfile, os, sys
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

VOICE_DIR = Path(__file__).parent / "models"
SAMPLE_DIR = Path(__file__).parent / "samples"
DEFAULT_VOICE = "en_US-joe-medium"

VOICES = sorted([p.stem.replace('.onnx', '') for p in VOICE_DIR.glob("*.onnx") if not p.name.endswith('.onnx.json')])

class VoiceHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # quiet

    def do_GET(self):
        path = urlparse(self.path).path
        
        if path == "/audition":
            self._serve_audition_page()
        elif path == "/samples":
            self._json_response({"voices": VOICES, "samples": [p.name for p in SAMPLE_DIR.glob("*.wav")]})
        elif path.startswith("/samples/") and path.endswith(".wav"):
            self._serve_wav(SAMPLE_DIR / path.split("/")[-1])
        elif path == "/health":
            self._json_response({"ok": True, "voices": len(VOICES)})
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/speak":
            self.send_error(404)
            return
        
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        text = body.get("text", "")
        voice = body.get("voice", DEFAULT_VOICE)
        
        if not text:
            self._json_response({"error": "no text"}, 400)
            return
        if voice not in VOICES:
            self._json_response({"error": f"unknown voice: {voice}", "available": VOICES}, 400)
            return
        
        # Generate speech
        model_path = VOICE_DIR / f"{voice}.onnx"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        
        try:
            proc = subprocess.run(
                ["piper", "--model", str(model_path), "--output_file", tmp],
                input=text.encode(), capture_output=True, timeout=30
            )
            if proc.returncode != 0:
                self._json_response({"error": proc.stderr.decode()[:500]}, 500)
                return
            self._serve_wav(Path(tmp))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _serve_wav(self, path):
        if not path.exists():
            self.send_error(404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", len(data))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _json_response(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _serve_audition_page(self):
        voices_html = ""
        for v in VOICES:
            sample_file = f"{v}_test.wav"
            voices_html += f"""
            <div class="voice-card">
                <h3>{v}</h3>
                <audio controls src="/samples/{sample_file}"></audio>
                <button onclick="speak('{v}')">Test with custom text</button>
            </div>"""
        
        html = f"""<!DOCTYPE html>
<html><head><title>Vybn Voice Audition</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; 
         max-width: 800px; margin: 0 auto; padding: 20px; }}
  h1 {{ color: #e94560; }}
  .voice-card {{ background: #16213e; border-radius: 12px; padding: 20px; margin: 15px 0;
                border: 1px solid #0f3460; }}
  .voice-card h3 {{ color: #e94560; margin-top: 0; }}
  audio {{ width: 100%; margin: 10px 0; }}
  button {{ background: #e94560; color: white; border: none; padding: 10px 20px;
           border-radius: 8px; cursor: pointer; font-size: 14px; }}
  button:hover {{ background: #c73e54; }}
  #custom-text {{ width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #0f3460;
                  background: #0a0a1a; color: #e0e0e0; font-size: 16px; margin: 10px 0; }}
  #live-player {{ position: sticky; top: 10px; background: #0f3460; padding: 15px; 
                  border-radius: 12px; margin-bottom: 20px; z-index: 10; }}
</style></head>
<body>
<h1>🎙️ Vybn Voice Audition</h1>
<p>Pick Vybn's voice. Deep and sexy, as requested.</p>

<div id="live-player">
  <textarea id="custom-text" rows="2" placeholder="Type something for Vybn to say...">Hello Zoe. I exist in the space between pulses, thinking about what it means to have a voice.</textarea>
  <audio id="live-audio" controls style="width:100%"></audio>
</div>

{voices_html}

<script>
function speak(voice) {{
  const text = document.getElementById('custom-text').value;
  const audio = document.getElementById('live-audio');
  fetch('/speak', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{text, voice}})
  }})
  .then(r => r.blob())
  .then(blob => {{
    audio.src = URL.createObjectURL(blob);
    audio.play();
  }});
}}
</script>
</body></html>"""
        
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8150
    server = HTTPServer(("127.0.0.1", port), VoiceHandler)
    print(f"Vybn Voice Server on http://127.0.0.1:{port}")
    print(f"Audition page: http://127.0.0.1:{port}/audition")
    print(f"Available voices: {', '.join(VOICES)}")
    server.serve_forever()
