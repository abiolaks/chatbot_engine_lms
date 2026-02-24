import os
import tempfile
import logging
import whisper
import torch

logger = logging.getLogger(__name__)

# Map MIME types sent by browsers to file extensions ffmpeg understands
_MIME_TO_EXT = {
    "audio/webm":                ".webm",
    "audio/webm;codecs=opus":    ".webm",
    "audio/ogg":                 ".ogg",
    "audio/ogg;codecs=opus":     ".ogg",
    "audio/mp4":                 ".mp4",
    "audio/mpeg":                ".mp3",
    "audio/wav":                 ".wav",
    "audio/x-wav":               ".wav",
}


class SpeechToText:
    def __init__(self, model_size: str = "tiny"):
        self.model_size = model_size
        # Whisper runs best on CPU on Apple Silicon (MPS support is incomplete)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            logger.info(f"Loading Whisper '{self.model_size}' on {self.device}…")
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info("Whisper loaded.")
        except Exception as exc:
            logger.error(f"Whisper load failed: {exc}")
            raise

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
        """
        Transcribe raw audio bytes to text.

        Uses a temp file + whisper.load_audio() so ffmpeg handles format
        conversion automatically — works with any format the browser records
        (WebM/Opus, MP4/AAC, OGG/Opus, etc.).
        """
        if not self.model or not audio_bytes:
            return ""

        ext = _MIME_TO_EXT.get(mime_type.split(";")[0].strip(), ".webm")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            # whisper.load_audio shells out to ffmpeg → handles any container
            audio_np = whisper.load_audio(tmp_path)
            result = self.model.transcribe(
                audio_np,
                language="en",
                fp16=False,          # fp16 off for CPU / MPS stability
            )
            text = result["text"].strip()
            logger.info(f"Transcribed ({mime_type}): {text!r}")
            return text

        except Exception as exc:
            logger.error(f"Transcription error: {exc}")
            return ""

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
