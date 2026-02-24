# Genevieve — AI Avatar Learning Advisor

Genevieve is a fully local, voice-driven AI learning advisor. She listens to you, collects your learning goal, experience level, and career aspiration through natural conversation, then recommends personalised courses — all spoken aloud with real-time lip-sync animation running entirely in the browser.

---

## Table of Contents

1. [Changelog — pipeline hardening & lip-sync fixes](#changelog)
2. [How the lip-sync works without MuseTalk](#how-the-lip-sync-works-without-musetalk)
3. [The autoplay fix — what was broken and how it was solved](#the-autoplay-fix)
3. [Architecture overview](#architecture-overview)
4. [Prerequisites](#prerequisites)
5. [Step-by-step local setup](#step-by-step-local-setup)
6. [Running and using the app](#running-and-using-the-app)
7. [Configuration reference](#configuration-reference)
8. [Project structure](#project-structure)
9. [Deployment guide](#deployment-guide)
10. [MuseTalk GPU upgrade (optional)](#musetalk-gpu-upgrade-optional)

---

---

## Changelog

### Pipeline hardening & lip-sync improvements (Feb 2026)

#### Problem 1 — LLM hallucination and off-topic responses

**Root causes found:**
- `_BASE_PROMPT` allowed too much latitude — the model could give career advice, mention tools, or answer unrelated questions
- A "chatting" phase after recommendations gave the LLM free-form conversation rights with almost no constraints
- The `intro` spoken when recommendations fired was LLM-generated, opening a hallucination window for course names or false claims
- A "sentinel" JSON extraction path could be manipulated by a crafted user message to inject goal/level/career values

**What was changed in `src/nlp/ollama_conversation.py`:**

1. **Replaced `_BASE_PROMPT` with `_SYSTEM_PROMPT`** — new prompt has 9 numbered absolute rules, including hardcoded fallback sentences for off-topic and help requests, zero allowed diversions
2. **Removed the "chatting" phase entirely** — after recommendations the session resets immediately to collecting, no free-form conversation
3. **Hardcoded the recommendation intro** (`_REC_INTRO`) — no LLM call at the moment of recommendation; a fixed string is spoken and added to history, eliminating any chance of a hallucinated course name
4. **Added `_POST_REC_BRIDGE`** — a hardcoded sentence inviting the next search, sent after the recommendation cards, also not LLM-generated
5. **Removed sentinel extraction completely** — `_extract_sentinel`, `_SENTINEL_RE`, and all related code paths deleted; server-side regex is the only source of truth
6. **Lowered temperature from 0.3 → 0.2** and **reduced `num_predict` from 120 → 80** — tighter constraint on output length and creativity
7. **Added `stop` tokens** (`["\n", ".", "?", "!"]`) — Ollama stops generating after the first sentence, enforcing the one-sentence rule at the API level
8. **Removed `_is_farewell` / `_FAREWELL_RE`** — the "end" action is no longer used; the session model is reset by recommendation, not by farewell detection

**What was changed in `src/api/routes.py`:**

9. **Bridge message is synthesised and sent over WebSocket** after the recommendations payload — Genevieve speaks `_POST_REC_BRIDGE` so the user hears that they can search again

#### Problem 2 — Lip-sync lagging behind the audio

**Root causes found:**
- The smoothing factor `0.18` applied symmetrically meant the mouth opened at the same (slow) rate it closed — causing noticeable lag between the start of each spoken syllable and visible mouth movement
- The amplitude scaling `avg * 3.5` was linear — quiet Edge TTS output (typical avg energy ~0.05–0.10) produced weak mouth movement even during clear speech
- The frequency band `300–3000 Hz` excluded the 100–300 Hz fundamental frequency range where low-frequency voiced speech energy lives

**What was changed in `static/index.html`:**

10. **Asymmetric smoothing** — opening factor `0.40`, closing factor `0.12`. The mouth now snaps open within ~2 frames of a new syllable and fades out slowly, matching how human lips actually move
11. **Wider frequency band** — `100–3500 Hz` (was `300–3000 Hz`). Captures the fundamental frequency and first formant which carry most of the voiced energy in Edge TTS output
12. **Non-linear amplitude curve** — `Math.pow(avg, 0.6) * 2.8` (was `avg * 3.5`). The power of 0.6 boosts small input values so quiet speech produces visible mouth movement, while loud speech still clamps to 1.0
13. **Talking-loop video rate range widened** — `0.5–2.0×` (was `0.6–1.5×`). Gives more expressive range at both ends of amplitude
14. **Frequency bin indices moved outside `lipLoop`** — they never change per audio clip; computing them once avoids a division and two `Math.floor` calls every animation frame (~60/s)

---

## How the lip-sync works without MuseTalk

The project ships with two lip-sync modes. **Viseme mode** (the default, no GPU required) is what makes Genevieve's mouth move today. **MuseTalk mode** (optional, GPU required) replaces viseme mode with photorealistic video.

### Viseme mode — how it works end to end

**Step 1 — Six mouth sprites are baked at server startup**

`src/lipsync/viseme_generator.py` runs once when the server starts. It:

1. Loads the portrait photo (`static/images/portrait-business-woman-office.jpg`).
2. Crops and resizes the face to the 220×220 px display size.
3. Generates 6 JPEG images (`static/images/visemes/v0.jpg` … `v5.jpg`), each showing the mouth at a different degree of openness — from fully closed (v0) to wide open (v5).
4. Each sprite is created by compositing a Gaussian-feathered dark ellipse (the mouth opening) and a smaller ivory ellipse (teeth) over the original face using OpenCV. The soft Gaussian blur on the mask means the edges blend into the real skin instead of looking painted on.

These files are served as static assets so the browser can preload all six before the first audio plays.

**Step 2 — Audio is synthesised and streamed to the browser**

When Genevieve responds, `src/tts/edge_tts.py` converts her text to MP3 using Microsoft Edge TTS (online). The MP3 bytes are base64-encoded and sent inside the WebSocket JSON message as the `audio` field.

**Step 3 — The browser decodes the audio and drives the mouth in real time**

`static/index.html` contains a Web Audio API pipeline:

1. The base64 string is decoded to a `Uint8Array`.
2. `audioCtx.decodeAudioData()` decodes the MP3 into a PCM `AudioBuffer`.
3. An `AnalyserNode` sits between the `BufferSource` and the speakers. Each animation frame it reads the frequency-domain data (`getByteFrequencyData`).
4. The speech band (300–3000 Hz) is summed and averaged, giving a number between 0 and 1 that represents how loud Genevieve is speaking at that exact moment.
5. That amplitude number directly controls `mouthTarget`. A smoothing loop (`mouthCurrent += (mouthTarget - mouthCurrent) * 0.18`) eases toward the target so the mouth opens and closes fluidly rather than jumping.
6. `mouthCurrent` is mapped to a fractional sprite index (0–5). The `mouthLoop` renders two adjacent sprites on a `<canvas>` with alpha cross-fading: if the index is 2.7 it draws sprite 2 at full opacity then sprite 3 at 70% opacity on top, giving a smooth intermediate position that the 6 static images alone could not achieve.

The result is a face that opens its mouth exactly in time with the audio, with no ML or GPU involved at all.

---

## The autoplay fix

### What was broken

The server sends Genevieve's greeting the instant the WebSocket connects — before the user has clicked or typed anything. Modern browsers (Chrome, Firefox, Safari) enforce an **autoplay policy**: audio can only start playing after a direct user interaction (a click, keypress, or tap). Any attempt to start audio without a prior gesture is silently ignored.

The original code called `audioCtx.resume()` inside the WebSocket `onmessage` handler. Because `onmessage` fires from a network event — not a user gesture — the browser blocked it every time. The greeting audio was decoded successfully but never played. The same thing happened for all subsequent responses because `resume()` was still being called from the wrong place.

### How it was fixed

Three changes were made to `static/index.html`:

**1. A gesture gate was added**

```javascript
let _audioUnlocked = false;
let _unlockResolvers = [];

function _audioUnlockedPromise() {
  if (_audioUnlocked) return Promise.resolve();
  return new Promise(res => _unlockResolvers.push(res));
}
```

Audio playback now waits on this promise before doing anything.

**2. `_unlockAudio()` is called synchronously inside every user-gesture handler**

```javascript
function _unlockAudio() {
  if (_audioUnlocked) return;
  if (!audioCtx) audioCtx = new AudioContext();
  // resume() is called HERE — synchronously inside the click handler stack
  const p = audioCtx.state !== 'running' ? audioCtx.resume() : Promise.resolve();
  p.then(() => {
    _audioUnlocked = true;
    _unlockResolvers.forEach(r => r());
    _unlockResolvers = [];
  });
}
```

This is added to the send button click, the Enter key handler, and the mic button click — the three places where the user is definitely interacting with the page. The browser sees `resume()` called synchronously in the gesture stack and allows it.

**3. `playAudioWithLipSync()` queues audio until the gate opens**

```javascript
function playAudioWithLipSync(base64Audio) {
  const bytes = Uint8Array.from(atob(base64Audio), c => c.charCodeAt(0));

  _audioUnlockedPromise().then(() => {
    audioCtx.decodeAudioData(bytes.buffer.slice(0), buffer => {
      // ... play the audio ...
    });
  });
}
```

The greeting audio now arrives over the WebSocket, is decoded, and sits waiting behind the promise. The moment the user clicks "Send" or presses Enter for the first time, `_unlockAudio()` fires, the context resumes, the promise resolves, and the greeting plays immediately.

---

## Architecture overview

```
Browser  ──── WebSocket ws://host/api/v1/ws ────  FastAPI (app.py)
  │                                                     │
  │  sends: { type:"audio"|"text", audio/text }         │
  │  receives: { type:"response", text, audio,          │
  │              collected_info }                        │
  │  receives: { type:"recommendations", courses }       │
  │                                                     │
  ▼                                                     ▼
index.html                                    src/api/routes.py
  ├── Web Audio API (decode + play MP3)           │
  ├── AnalyserNode → amplitude → mouthTarget      ├── STT: SpeechToText (Whisper tiny)
  ├── Canvas viseme cross-fade                    │       audio bytes → English text
  └── Course cards rendered from JSON            ├── NLP: OllamaConversationManager
                                                  │       text → intent extraction
                                                  │       → Ollama/gemma3:4b API call
                                                  │       → structured response
                                                  ├── Recommendations: recommend_courses()
                                                  │       goal + level + career → top 3 courses
                                                  └── TTS: EdgeTTS
                                                          text → MP3 bytes
```

### Component responsibilities

| File | What it does |
|---|---|
| `app.py` | FastAPI entry point, lifespan startup (generates visemes, optionally loads MuseTalk), mounts static files |
| `config.py` | All configuration via environment variables with sensible defaults |
| `src/api/routes.py` | WebSocket endpoint, REST endpoints (`/chat`, `/chat/audio`, `/session/{id}`) |
| `src/stt/speech_to_text.py` | Wraps OpenAI Whisper (tiny). Accepts any browser audio format, converts via ffmpeg |
| `src/nlp/ollama_conversation.py` | Conversation brain: per-session state, server-side intent extraction (regex), dynamic system prompt, Ollama API calls |
| `src/tts/edge_tts.py` | Primary TTS via Microsoft Edge TTS (online). Falls back to Piper (offline) if network fails |
| `src/tts/piper_tts.py` | Offline TTS fallback using Piper — returns WAV bytes |
| `src/recommendations/engine.py` | Scores all 24 courses against goal/level/career using keyword overlap + level distance |
| `src/recommendations/courses.py` | Hard-coded 24-course catalogue spanning Python, JS, ML, data science, DevOps, SQL, cloud, design |
| `src/lipsync/viseme_generator.py` | Generates 6 mouth-state sprites from the avatar photo at startup (OpenCV only) |
| `src/lipsync/musetalk_worker.py` | Optional GPU lip-sync: loads MuseTalk models, runs per-response inference |
| `static/index.html` | Single-page app: WebSocket client, Web Audio API pipeline, canvas viseme renderer, chat UI |

### Conversation state machine

```
Session created
      │
      ▼
phase = "collecting"
  collected = { goal: None, level: None, career: None }
      │
      │  Each user message:
      │  1. Regex extracts any of goal / level / career
      │  2. Ollama generates a spoken reply asking for missing info
      │
      ▼  (all three collected)
  recommend_courses() fires immediately
  phase → "chatting"
  collected reset to all-None
      │
      ▼
phase = "chatting"
  Follow-up questions answered directly.
  If user mentions a new topic → phase resets to "collecting"
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or 3.11 | 3.12+ untested with some ML deps |
| ffmpeg | any modern | Must be on `$PATH` — used by Whisper and TTS fallback |
| Ollama | latest | Must be running before starting the server |
| gemma3:4b model | — | Pulled via `ollama pull gemma3:4b` |
| Internet connection | — | Required for Edge TTS (or set up Piper fallback for offline) |

---

## Step-by-step local setup

### 1. Clone the repository

```bash
git clone <your-repo-url> chatbot_engine_lms
cd chatbot_engine_lms
```

### 2. Create and activate a Python virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs:
- `fastapi` + `uvicorn[standard]` — web server
- `openai-whisper` + `torch` + `torchaudio` — speech-to-text
- `edge-tts` — text-to-speech (online)
- `piper-tts` — text-to-speech (offline fallback)
- `opencv-python` + `numpy` — viseme sprite generation
- `httpx` — async HTTP client for Ollama
- `python-multipart` + `websockets` — FastAPI extras

### 4. Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install -y ffmpeg

# Windows — download from https://ffmpeg.org/download.html
# and add the bin/ folder to your PATH
```

Verify: `ffmpeg -version`

### 5. Install and start Ollama

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Then pull the model Genevieve uses
ollama pull gemma3:4b

# Start the server (runs in the background)
ollama serve &
```

Verify: `curl http://localhost:11434/api/tags`

### 6. Verify the avatar image is present

The file `static/images/portrait-business-woman-office.jpg` must exist. It is the source image for all viseme sprites. If you want to use a different avatar, replace this file — the viseme generator will automatically detect new coordinates on the next startup if you adjust the `CROP` / `MOUTH_*` constants in `src/lipsync/viseme_generator.py`.

### 7. (Optional) Set up Piper offline TTS fallback

If you want the app to speak even without internet access:

```bash
mkdir -p models/piper

# Download voice model (two files needed)
wget -O models/piper/en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx

wget -O models/piper/en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
```

Then set the env var:

```bash
export PIPER_MODEL_PATH=models/piper/en_US-amy-medium.onnx
```

Without this, Edge TTS works fine as long as you have internet. If Edge TTS fails and Piper is not set up, audio returns empty bytes and the text response is still shown in the chat.

### 8. (Optional) Create a .env file

```bash
cat > .env << 'EOF'
LIPSYNC_MODE=viseme
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_TIMEOUT=30
WHISPER_MODEL=tiny
EDGE_TTS_VOICE=en-US-JennyNeural
EDGE_TTS_RATE=+0%
EDGE_TTS_PITCH=+0%
# PIPER_MODEL_PATH=models/piper/en_US-amy-medium.onnx
EOF
```

---

## Running and using the app

### Start the server

```bash
python app.py
```

You will see startup logs like:

```
INFO  Generating 6 viseme sprites from static/images/portrait-business-woman-office.jpg …
INFO  Viseme sprites ready.
INFO  Server ready → http://localhost:8000/static/index.html
INFO  API docs    → http://localhost:8000/docs
```

If the viseme sprites already exist from a previous run, the generator skips them instantly.

### Open the chat interface

Navigate to: `http://localhost:8000/static/index.html`

### How to have a conversation

Genevieve needs three pieces of information before she can recommend courses:

| What she needs | Examples |
|---|---|
| **Learning goal** | "Python", "machine learning", "web development", "SQL", "DevOps" |
| **Experience level** | "beginner", "intermediate", "advanced" |
| **Career goal** | "data scientist", "web developer", "software engineer", "ML engineer" |

You can give all three in one message:

> "I'm a complete beginner and I want to learn Python to become a data scientist."

Or answer her questions one at a time. Once she has all three, she immediately shows and speaks the top 3 recommended courses.

After recommendations, you can ask follow-up questions. If you mention a new topic, she resets and collects fresh information for new recommendations.

### Voice input

Click the microphone button (🎤), speak, then click Stop (■). Your voice is:

1. Recorded by the browser as WebM/Opus
2. Base64-encoded and sent over the WebSocket
3. Decoded and transcribed server-side by Whisper
4. Processed as text through the normal pipeline

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/ws` | WebSocket | Real-time chat (audio + text) |
| `/api/v1/chat` | POST | Text-only REST endpoint |
| `/api/v1/chat/audio` | POST | Audio file upload → transcribe → chat |
| `/api/v1/session/{id}` | GET | Get session state and recent history |
| `/api/v1/session/{id}` | DELETE | Clear a session |
| `/api/v1/health` | GET | Component health check |
| `/docs` | GET | Interactive Swagger UI |

---

## Configuration reference

All configuration is in `config.py` and read from environment variables.

| Variable | Default | Description |
|---|---|---|
| `LIPSYNC_MODE` | `viseme` | `viseme` = CPU sprites (default). `musetalk` = GPU video. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3:4b` | Model name to use (must be pulled) |
| `OLLAMA_TIMEOUT` | `30` | Seconds before Ollama call times out |
| `WHISPER_MODEL` | `tiny` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` |
| `EDGE_TTS_VOICE` | `en-US-JennyNeural` | Edge TTS voice name |
| `EDGE_TTS_RATE` | `+0%` | Speech rate adjustment |
| `EDGE_TTS_PITCH` | `+0%` | Pitch adjustment |
| `PIPER_MODEL_PATH` | `models/piper/en_US-amy-medium.onnx` | Path to Piper .onnx voice file |

**Whisper model tradeoff:**

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `tiny` | 75 MB | fastest | good for clear English speech |
| `base` | 145 MB | fast | better accuracy |
| `small` | 465 MB | moderate | noticeably better |
| `medium` | 1.5 GB | slow on CPU | near-human accuracy |

---

## Project structure

```
chatbot_engine_lms/
├── app.py                              # FastAPI entry point + lifespan
├── config.py                           # All env-var config
├── requirements.txt
│
├── static/
│   ├── index.html                      # Single-page chat UI
│   ├── images/
│   │   ├── portrait-business-woman-office.jpg  # Avatar source photo
│   │   └── visemes/
│   │       ├── v0.jpg                  # Mouth closed (generated at startup)
│   │       ├── v1.jpg
│   │       ├── v2.jpg
│   │       ├── v3.jpg
│   │       ├── v4.jpg
│   │       └── v5.jpg                 # Mouth wide open
│   └── videos/                         # (optional) talking_loop.mp4
│
├── src/
│   ├── api/
│   │   └── routes.py                  # WebSocket + REST handlers
│   ├── stt/
│   │   └── speech_to_text.py          # Whisper wrapper
│   ├── tts/
│   │   ├── edge_tts.py                # Edge TTS (primary) + Piper fallback
│   │   └── piper_tts.py               # Offline Piper TTS
│   ├── nlp/
│   │   └── ollama_conversation.py     # Conversation manager + intent extraction
│   ├── recommendations/
│   │   ├── courses.py                 # 24-course catalogue
│   │   └── engine.py                  # Keyword + level scoring
│   └── lipsync/
│       ├── viseme_generator.py        # CPU sprite generator (active)
│       └── musetalk_worker.py         # GPU lip-sync wrapper (optional)
│
├── MuseTalk/                          # MuseTalk repo (for GPU mode only)
│   ├── models/
│   │   ├── musetalkV15/unet.pth       # 3.2 GB UNet weights
│   │   ├── sd-vae/                    # 319 MB VAE
│   │   └── whisper/                   # 144 MB audio encoder
│   └── results/v15/avatars/genevieve/ # Pre-computed face latents (cached)
│
└── models/
    └── piper/                         # (optional) Piper offline voice models
```

---

## Deployment guide

### Option A — Single server (no Docker)

This is the simplest production setup: one Linux VPS running everything.

**Tested on:** Ubuntu 22.04, 2 vCPU, 4 GB RAM (adequate for `tiny` Whisper + `gemma3:4b` via Ollama)

#### 1. Provision the server and install system packages

```bash
sudo apt-get update && sudo apt-get install -y \
  python3.10 python3.10-venv python3-pip \
  ffmpeg git curl nginx
```

#### 2. Install Ollama and pull the model

```bash
curl -fsSL https://ollama.ai/install.sh | sh
sudo systemctl enable --now ollama
ollama pull gemma3:4b
```

#### 3. Clone the repo and install Python deps

```bash
git clone <your-repo-url> /opt/chatbot
cd /opt/chatbot
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 4. Create the environment file

```bash
cat > /opt/chatbot/.env << 'EOF'
LIPSYNC_MODE=viseme
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
WHISPER_MODEL=tiny
EDGE_TTS_VOICE=en-US-JennyNeural
EOF
```

#### 5. Create a systemd service

```bash
sudo tee /etc/systemd/system/chatbot.service << 'EOF'
[Unit]
Description=Genevieve AI Learning Advisor
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/chatbot
EnvironmentFile=/opt/chatbot/.env
ExecStart=/opt/chatbot/.venv/bin/python app.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now chatbot
```

Check logs: `sudo journalctl -fu chatbot`

#### 6. Set up Nginx as a reverse proxy (adds HTTPS support)

WebSockets require special proxy configuration:

```nginx
# /etc/nginx/sites-available/chatbot
server {
    listen 80;
    server_name your-domain.com;

    # Redirect HTTP → HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # WebSocket endpoint — must use wss:// in production
    location /api/v1/ws {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 86400;   # keep WebSocket alive
    }

    # All other requests
    location / {
        proxy_pass       http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

# Free TLS cert
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

### Option B — Docker Compose

The easiest way to run everything reproducibly, including Ollama.

#### `docker-compose.yml`

```yaml
version: "3.9"
services:

  chatbot:
    build: .
    ports:
      - "8000:8000"
    environment:
      LIPSYNC_MODE:    ${LIPSYNC_MODE:-viseme}
      OLLAMA_BASE_URL: http://ollama:11434
      OLLAMA_MODEL:    gemma3:4b
      WHISPER_MODEL:   tiny
      EDGE_TTS_VOICE:  en-US-JennyNeural
    depends_on:
      ollama:
        condition: service_healthy
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    environment:
      OLLAMA_KEEP_ALIVE: "24h"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  ollama_data:
```

#### `Dockerfile`

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "app.py"]
```

#### Run it

```bash
# Build and start
docker-compose up --build -d

# Pull the Ollama model (first time only)
docker-compose exec ollama ollama pull gemma3:4b

# Check logs
docker-compose logs -f chatbot

# Open
open http://localhost:8000/static/index.html
```

---

### Option C — GPU server with MuseTalk

Use this only if you have a machine with an NVIDIA GPU (8 GB VRAM minimum) and want photorealistic lip-sync video instead of viseme sprites. **The entire rest of the pipeline is unchanged** — only the lip-sync step swaps.

#### Additional requirements

- CUDA 12 + cuDNN
- PyTorch 2.0.1 with CUDA support
- MuseTalk model weights (already in `MuseTalk/models/` if you cloned with LFS)

#### Install MuseTalk dependencies

```bash
cd MuseTalk
pip install -r requirements.txt
pip install -U openmim
mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"
cd ..
```

#### Verify model weights exist

```bash
ls MuseTalk/models/musetalkV15/unet.pth      # 3.2 GB
ls MuseTalk/models/sd-vae/                   # 319 MB
ls MuseTalk/models/whisper/                  # 144 MB
ls MuseTalk/models/dwpose/
ls MuseTalk/models/face-parse-bisent/
```

#### Start in MuseTalk mode

```bash
LIPSYNC_MODE=musetalk python app.py
```

On first startup the server prepares the avatar (extracts face landmarks and VAE latents, ~60 seconds). The latents are cached to `MuseTalk/results/v15/avatars/genevieve/` — subsequent starts skip preparation and load in seconds.

#### Docker Compose for GPU

```yaml
# docker-compose.gpu.yml
services:
  chatbot-gpu:
    build: .
    runtime: nvidia
    environment:
      LIPSYNC_MODE:    musetalk
      OLLAMA_BASE_URL: http://ollama:11434
      OLLAMA_MODEL:    gemma3:4b
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "8000:8000"
    depends_on: [ollama]

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama

volumes:
  ollama_data:
```

```bash
docker-compose -f docker-compose.gpu.yml up --build -d
```

---

### Common deployment issues

| Symptom | Cause | Fix |
|---|---|---|
| "Reconnecting…" in browser | WebSocket blocked by proxy | Add `Upgrade` + `Connection` headers in Nginx (see above) |
| No audio in browser | HTTPS required for microphone | Add TLS with certbot; `wss://` is used automatically when the page is HTTPS |
| Ollama timeout errors | Model not loaded or too slow | Increase `OLLAMA_TIMEOUT`; use a faster model like `gemma3:2b` |
| Whisper crashes on startup | PyTorch / CUDA mismatch | Install CPU-only torch: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| Viseme sprites missing | Avatar image not found | Confirm `static/images/portrait-business-woman-office.jpg` exists before starting |
| Edge TTS silent | No internet / Microsoft API down | Set up Piper fallback (see step 7 in local setup) |
| MuseTalk OOM | Not enough VRAM | Reduce `batch_size` in `musetalk_worker.py` `_make_args()` from 8 to 4 |
