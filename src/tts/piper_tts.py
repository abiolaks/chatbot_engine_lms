# src/tts/piper_tts.py
#
# Offline TTS fallback using Piper (https://github.com/rhasspy/piper).
# Returns WAV bytes — browsers and MuseTalk both handle WAV natively.
#
# Setup (one-time, on the server):
#   pip install piper-tts
#   mkdir -p models/piper
#   # Download a voice from https://huggingface.co/rhasspy/piper-voices
#   # e.g. en_US-amy-medium — need both .onnx and .onnx.json
#   wget -O models/piper/en_US-amy-medium.onnx \
#     https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
#   wget -O models/piper/en_US-amy-medium.onnx.json \
#     https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

import asyncio
import io
import logging
import wave
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_voice = None          # singleton PiperVoice — loaded once
_model_path: Optional[Path] = None


def _load_voice(model_path: Path) -> bool:
    """Load Piper voice model. Returns True on success, False if unavailable."""
    global _voice, _model_path
    if _voice is not None and _model_path == model_path:
        return True
    try:
        from piper.voice import PiperVoice   # type: ignore
        if not model_path.exists():
            logger.warning(
                f"Piper model not found at {model_path}. "
                "Download a voice from https://huggingface.co/rhasspy/piper-voices"
            )
            return False
        logger.info(f"Loading Piper voice from {model_path} …")
        _voice = PiperVoice.load(str(model_path))
        _model_path = model_path
        logger.info("Piper voice loaded.")
        return True
    except ImportError:
        logger.warning("piper-tts not installed — run: pip install piper-tts")
        return False
    except Exception as exc:
        logger.error(f"Failed to load Piper voice: {exc}")
        return False


def _synthesize_sync(text: str) -> bytes:
    """Synchronous Piper synthesis — returns WAV bytes."""
    if _voice is None:
        return b""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        _voice.synthesize(text, wav_file)
    return buf.getvalue()


class PiperTTS:
    """Local, offline TTS using Piper. Returns WAV bytes."""

    def __init__(self, model_path: Path):
        self._model_path = model_path
        self._available = _load_voice(model_path)

    @property
    def available(self) -> bool:
        return self._available

    async def synthesize(self, text: str) -> bytes:
        """Synthesize speech offline. Returns WAV bytes, or b'' if unavailable."""
        if not self._available:
            return b""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _synthesize_sync, text)
