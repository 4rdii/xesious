#!/usr/bin/env python3
"""Transcribe an audio file to text with faster-whisper — local, no API key.

Usage:  stt.py <audio-file>        # prints the transcript to stdout
Env:    TG_STT_MODEL  (default 'base')   whisper size: tiny|base|small|medium|large-v3
        TG_STT_LANG   (default auto)     force a language code, e.g. en, fa
The model is downloaded once and cached under ~/.cache/huggingface.
"""
import os
import sys

def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: stt.py <audio-file>\n")
        return 2
    audio = sys.argv[1]
    if not os.path.isfile(audio):
        sys.stderr.write(f"no such file: {audio}\n")
        return 2
    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"faster-whisper not installed ({e}); run voice/setup.sh\n")
        return 3

    model_name = os.environ.get("TG_STT_MODEL", "base").strip() or "base"
    lang = (os.environ.get("TG_STT_LANG") or "").strip() or None  # None = autodetect

    # int8 on CPU: fast and light.
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    # Decode with ffmpeg rather than letting faster-whisper do it. Its audio.py
    # calls av.open(..., metadata_errors=...), which PyAV dropped by v19 (some
    # boxes have 19.x), and no PyAV below 15 has a wheel for this Python: it
    # wants Cython and ffmpeg headers to build. ffmpeg is already a hard
    # dependency of the voice path, so decoding here removes PyAV entirely
    # instead of pinning around it.
    import subprocess
    import numpy as np
    _ff = os.environ.get("TG_FFMPEG") or "ffmpeg"
    _r = subprocess.run([_ff, "-nostdin", "-threads", "1", "-i", audio,
                         "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                        capture_output=True)
    if _r.returncode != 0:
        print(f"stt: ffmpeg could not decode {audio}", file=sys.stderr)
        return 3
    pcm = np.frombuffer(_r.stdout, dtype=np.float32)
    segments, _info = model.transcribe(pcm, language=lang, vad_filter=True)
    text = "".join(seg.text for seg in segments).strip()
    sys.stdout.write(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
