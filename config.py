import os
from pathlib import Path

# folder structure and configuration settings for the chatbot engine
class Config:
    BASE_DIR = Path(__file__).parent
    MODELS_DIR = BASE_DIR / "models"
    STATIC_DIR = BASE_DIR / "static"

    MODELS_DIR.mkdir(exist_ok=True)
    (STATIC_DIR / "images").mkdir(parents=True, exist_ok=True)
    (STATIC_DIR / "videos").mkdir(exist_ok=True)

    AVATAR_IMAGE_PATH = STATIC_DIR / "images" / "crop_portrait_gen.png"
    OUTPUT_VIDEO_PATH = STATIC_DIR / "videos" / "response_"

    WHISPER_MODEL = "tiny"

    # Edge TTS models
    EDGE_TTS_VOICE = "en-US-JennyNeural"
    EDGE_TTS_RATE = "+0%"
    EDGE_TTS_PITCH = "+0%"

    # conversation settings
    MAX_HISTORY = 20

    # Recommendation engine
    RECOMMENDATION_API_URL = os.getenv(
        "RECOMMENDATION_API_URL", "http://localhost:8001/recommend"
    )

    # Ollama (local LLM)
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "30"))

    # Avatar persona
    AVATAR_NAME = "Genevieve"

    # Lip-sync mode: "viseme" (CPU, default) | "musetalk" (GPU)
    LIPSYNC_MODE = os.getenv("LIPSYNC_MODE", "viseme")

    # Piper offline TTS fallback — set path to your downloaded .onnx voice model
    # Download from: https://huggingface.co/rhasspy/piper-voices
    # e.g. models/piper/en_US-amy-medium.onnx (needs .onnx + .onnx.json alongside)
    PIPER_MODEL_PATH = os.getenv(
        "PIPER_MODEL_PATH", "models/piper/en_US-amy-medium.onnx"
    )
