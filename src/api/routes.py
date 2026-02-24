# src/api/routes.py
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    HTTPException,
)
import base64
import json
import uuid
import logging

from ..stt.speech_to_text import SpeechToText
from ..nlp.ollama_conversation import OllamaConversationManager
from ..tts.edge_tts import EdgeTTS
from config import Config

logger = logging.getLogger(__name__)

router = APIRouter()
config = Config()

# ── Component initialisation ─────────────────────────────────────────────────
stt = SpeechToText(model_size=config.WHISPER_MODEL)
conversation = OllamaConversationManager()
tts = EdgeTTS(voice=config.EDGE_TTS_VOICE)

# session_id → WebSocket
active_connections: dict = {}


# ── WebSocket endpoint ───────────────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    active_connections[session_id] = websocket
    conversation.new_session(session_id)
    logger.info(f"WebSocket connected: {session_id}")

    # Send the greeting immediately on connect
    greeting = await conversation.process_message(
        session_id, "__init__"
    )
    audio_bytes = await tts.synthesize(greeting["text"])
    if Config.LIPSYNC_MODE == "musetalk" and audio_bytes:
        from ..lipsync.musetalk_worker import generate_video
        video_out = await generate_video(audio_bytes)
        await websocket.send_json({
            "type":           "response",
            "text":           greeting["text"],
            "action":         greeting["action"],
            "video":          base64.b64encode(video_out).decode(),
            "collected_info": greeting["collected_info"],
        })
    else:
        await websocket.send_json({
            "type":           "response",
            "text":           greeting["text"],
            "action":         greeting["action"],
            "audio":          base64.b64encode(audio_bytes).decode() if audio_bytes else None,
            "collected_info": greeting["collected_info"],
        })

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type", "text")

            # ── Decode input ──────────────────────────────────────────────
            if msg_type == "audio":
                audio_b64 = message.get("audio", "")
                mime = message.get("mime", "audio/webm")
                if audio_b64:
                    audio_bytes_in = base64.b64decode(audio_b64)
                    user_text = stt.transcribe(audio_bytes_in, mime_type=mime)
                    if not user_text:
                        user_text = "[Could not understand audio]"
                else:
                    user_text = ""
            else:
                user_text = message.get("text", "")

            if not user_text.strip():
                continue

            # ── Process through Ollama ────────────────────────────────────
            response = await conversation.process_message(session_id, user_text)

            # ── Synthesise speech ─────────────────────────────────────────
            audio_out = await tts.synthesize(response["text"])

            # ── Send main response ────────────────────────────────────────
            if Config.LIPSYNC_MODE == "musetalk" and audio_out:
                from ..lipsync.musetalk_worker import generate_video
                video_out = await generate_video(audio_out)
                await websocket.send_json({
                    "type":           "response",
                    "text":           response["text"],
                    "action":         response["action"],
                    "video":          base64.b64encode(video_out).decode(),
                    "collected_info": response["collected_info"],
                })
            else:
                await websocket.send_json({
                    "type":           "response",
                    "text":           response["text"],
                    "action":         response["action"],
                    "audio":          base64.b64encode(audio_out).decode() if audio_out else None,
                    "collected_info": response["collected_info"],
                })

            # ── Send recommendations if ready ─────────────────────────────
            if response["action"] == "recommend" and response["recommendations"]:
                await websocket.send_json({
                    "type": "recommendations",
                    "courses": response["recommendations"],
                })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    finally:
        active_connections.pop(session_id, None)


# ── REST endpoints ────────────────────────────────────────────────────────────

@router.post("/chat")
async def chat_text(session_id: str = None, message: str = ""):
    """Text-based chat — returns JSON response."""
    if not session_id:
        session_id = str(uuid.uuid4())
        conversation.new_session(session_id)

    response = await conversation.process_message(session_id, message)
    audio_bytes = await tts.synthesize(response["text"])

    payload = {
        "session_id": session_id,
        "response": response["text"],
        "action": response["action"],
        "audio": base64.b64encode(audio_bytes).decode() if audio_bytes else None,
        "collected_info": response["collected_info"],
    }
    if response["action"] == "recommend":
        payload["recommendations"] = response["recommendations"]
    return payload


@router.post("/chat/audio")
async def chat_audio(session_id: str = None, audio: UploadFile = File(...)):
    """Audio file upload → transcribe → chat."""
    if not session_id:
        session_id = str(uuid.uuid4())
        conversation.new_session(session_id)

    audio_bytes = await audio.read()
    user_text = stt.transcribe(audio_bytes)
    response = await conversation.process_message(session_id, user_text)
    audio_out = await tts.synthesize(response["text"])

    payload = {
        "session_id": session_id,
        "transcription": user_text,
        "response": response["text"],
        "action": response["action"],
        "audio": base64.b64encode(audio_out).decode() if audio_out else None,
        "collected_info": response["collected_info"],
    }
    if response["action"] == "recommend":
        payload["recommendations"] = response["recommendations"]
    return payload


@router.get("/session/{session_id}")
async def get_session_info(session_id: str):
    session = conversation.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "collected_info": session["collected"],
        "history": session["messages"][-5:],
    }


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if not conversation.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "cleared", "session_id": session_id}


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "components": {
            "stt": "loaded",
            "tts": "ready",
            "llm": f"ollama/{config.OLLAMA_MODEL}",
        },
    }
