# Main application file for the chatbot engine LMS
# app.py
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import Config
from src.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Lifespan (startup + shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Generate static assets on startup; clean up on shutdown."""

    # ── STARTUP ───────────────────────────────────────────────────────────
    logger.info("Starting AI Learning Advisor chatbot…")

    # Generate viseme sprites so the browser can load them immediately
    avatar = Path("static/images/portrait-business-woman-office.jpg")
    if avatar.exists():
        try:
            from src.lipsync.viseme_generator import ensure_visemes
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, ensure_visemes)
            logger.info("Viseme sprites ready.")
        except Exception as exc:
            logger.error(
                f"Viseme generation failed (lip-sync unavailable): {exc}"
            )
    else:
        logger.warning(
            f"Avatar image not found at {avatar} — viseme generation skipped."
        )

    # Load MuseTalk avatar on GPU servers (no-op in viseme mode)
    if Config.LIPSYNC_MODE == "musetalk":
        try:
            from src.lipsync.musetalk_worker import load_avatar
            await asyncio.get_event_loop().run_in_executor(None, load_avatar)
            logger.info("MuseTalk Avatar loaded.")
        except Exception as exc:
            logger.error(f"MuseTalk Avatar load failed: {exc}")

    logger.info("Server ready →  http://localhost:8000/static/index.html")
    logger.info("API docs    →  http://localhost:8000/docs")

    yield   # ── APPLICATION RUNS ──────────────────────────────────────────

    # ── SHUTDOWN ──────────────────────────────────────────────────────────
    logger.info("Shutting down chatbot server.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Chatbot with Face & Voice",
    description="Conversational AI avatar with STT, TTS, and real-time lip-sync",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router, prefix="/api/v1", tags=["api"])

# Static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
(static_dir / "images").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Root redirect ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect browser to the chat frontend."""
    return """
    <html>
      <head>
        <title>AI Chatbot with Face &amp; Voice</title>
        <meta http-equiv="refresh" content="0;url=/static/index.html">
        <style>body{font-family:Arial;margin:40px;} a{color:#0066cc;}</style>
      </head>
      <body>
        <p>Redirecting to <a href="/static/index.html">chat interface</a>…</p>
      </body>
    </html>
    """


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "chatbot-engine", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
