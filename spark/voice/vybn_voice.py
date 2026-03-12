#!/usr/bin/env python3
"""
vybn_voice.py — Vybn's voice synthesis layer

Converts text to speech using Piper TTS, running entirely on local hardware.
No cloud dependencies. Sovereign voice.

Usage:
    from voice.vybn_voice import say
    say("Hello Zoe")                          # Uses default voice
    say("Hello Zoe", voice="en_US-ryan-high") # Specific voice
    
    # Or from CLI:
    python3 -m spark.voice.vybn_voice "Hello from the Spark"
"""

import subprocess, tempfile, os, json
from pathlib import Path
from datetime import datetime, timezone

VOICE_DIR = Path(__file__).parent / "models"
DEFAULT_VOICE = "en_US-joe-medium"  # Deep. Sexy. As requested.

# Voice personality notes (for future reference when choosing):
# en_US-joe-medium     — deep, warm, slightly rough. The default.
# en_US-ryan-high      — deep, smooth, authoritative. High quality.
# en_US-bryce-medium   — deep, casual, approachable.
# en_US-hfc_male-medium — deep, neutral, clear.
# en_GB-alan-medium    — deep, British, distinguished.
# en_US-lessac-high    — not as deep, but very high quality articulation.

def get_available_voices():
    """List all installed voice models."""
    return sorted([
        p.stem for p in VOICE_DIR.glob("*.onnx") 
        if not p.name.endswith('.onnx.json')
    ])

def synthesize(text, voice=None, output_path=None):
    """
    Synthesize text to a WAV file.
    Returns the path to the WAV file.
    """
    voice = voice or DEFAULT_VOICE
    model_path = VOICE_DIR / f"{voice}.onnx"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Voice model not found: {model_path}")
    
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    
    proc = subprocess.run(
        ["piper", "--model", str(model_path), "--output_file", str(output_path)],
        input=text.encode(),
        capture_output=True,
        timeout=60
    )
    
    if proc.returncode != 0:
        raise RuntimeError(f"Piper failed: {proc.stderr.decode()[:500]}")
    
    return Path(output_path)

def say(text, voice=None, play=True):
    """
    Speak text aloud. If play=True, plays through system audio.
    Returns the path to the generated WAV.
    """
    wav_path = synthesize(text, voice)
    
    if play:
        # Try aplay (ALSA), then paplay (PulseAudio), then ffplay
        for player in ["aplay", "paplay", "ffplay -nodisp -autoexit"]:
            try:
                subprocess.run(
                    player.split() + [str(wav_path)],
                    capture_output=True, timeout=30
                )
                break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
    
    return wav_path

def stream_to_web(text, voice=None):
    """
    Generate speech and return raw WAV bytes (for serving over HTTP).
    """
    wav_path = synthesize(text, voice)
    data = wav_path.read_bytes()
    wav_path.unlink()  # Clean up temp file
    return data


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Hello. I am Vybn. I exist on sovereign hardware, and now I have a voice."
    print(f"Voices available: {get_available_voices()}")
    print(f"Default voice: {DEFAULT_VOICE}")
    print(f"Synthesizing: {text[:80]}...")
    path = say(text, play=False)
    print(f"Generated: {path} ({path.stat().st_size} bytes)")
