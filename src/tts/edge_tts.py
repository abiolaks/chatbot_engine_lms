# src/tts/edge_tts.py
#
# Primary TTS: Microsoft Edge TTS (requires network).
# Fallback TTS: Piper (fully local, no network required).
#
# synthesize() tries Edge TTS first. On any network / connectivity failure
# it transparently falls back to PiperTTS and returns WAV bytes instead of
# MP3.  All callers (routes.py, MuseTalk worker) handle both formats
# because ffmpeg and browsers both auto-detect WAV vs MP3 from magic bytes.

import asyncio
import io
import logging
from pathlib import Path
from typing import Optional

import edge_tts

from config import Config

logger = logging.getLogger(__name__)

# ── Lazy Piper singleton ──────────────────────────────────────────────────────
_piper = None


def _get_piper():
    global _piper
    if _piper is None:
        from src.tts.piper_tts import PiperTTS
        model_path = Path(Config.PIPER_MODEL_PATH)
        _piper = PiperTTS(model_path=model_path)
    return _piper


# ── Main class ────────────────────────────────────────────────────────────────

class EdgeTTS:
    def __init__(self, voice: str = "en-US-JennyNeural"):
        self.voice = voice

    async def synthesize(self, text: str, output_path: Optional[Path] = None) -> bytes:
        """
        Synthesize speech. Tries Edge TTS (online) first; falls back to
        Piper (offline) on any network error.

        Returns:
          - MP3 bytes  if Edge TTS succeeds
          - WAV bytes  if Piper fallback is used
          - b''        if both engines fail
        """
        # ── Attempt Edge TTS ──────────────────────────────────────────────
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            if output_path:
                await communicate.save(str(output_path))
                return b""
            else:
                audio_bytes = io.BytesIO()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes.write(chunk["data"])
                result = audio_bytes.getvalue()
                if result:
                    return result
                # Empty result → treat as failure
                raise RuntimeError("Edge TTS returned empty audio")

        except Exception as exc:
            logger.warning(f"Edge TTS failed ({exc!r}) — falling back to Piper.")

        # ── Piper fallback ────────────────────────────────────────────────
        try:
            piper = _get_piper()
            if piper.available:
                wav = await piper.synthesize(text)
                if wav:
                    logger.info("Piper fallback: audio synthesized successfully.")
                    return wav
        except Exception as exc:
            logger.error(f"Piper fallback also failed: {exc!r}")

        logger.error("Both Edge TTS and Piper failed — returning silent audio.")
        return b""

    # Synchronous convenience wrapper (not used in the async pipeline)
    def speak(self, text: str, output_path: Optional[Path] = None) -> bytes:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.synthesize(text, output_path))
        finally:
            loop.close()
