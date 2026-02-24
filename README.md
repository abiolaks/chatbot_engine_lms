# Genevieve — AI Avatar Learning Advisor

Genevieve is a fully local, voice-driven AI learning advisor. She listens to you, collects your learning goal, experience level, and career aspiration through natural conversation, then recommends personalised courses — all spoken aloud with real-time lip-sync animation and word-by-word text streaming running entirely in the browser.

---

## Table of Contents

1. [Changelog](#changelog)
2. [How the lip-sync works without MuseTalk](#how-the-lip-sync-works-without-musetalk)
3. [The autoplay fix](#the-autoplay-fix)
4. [Architecture overview](#architecture-overview)
5. [Prerequisites](#prerequisites)
6. [Step-by-step local setup](#step-by-step-local-setup)
7. [Running and using the app](#running-and-using-the-app)
8. [Configuration reference](#configuration-reference)
9. [Project structure](#project-structure)
10. [Deployment guide](#deployment-guide)
11. [MuseTalk GPU upgrade (optional)](#musetalk-gpu-upgrade-optional)

---

## Changelog

### Audio queue + word streaming (Feb 2026)

**Problems fixed:**

- Audio was cut off mid-sentence when a new response arrived because each call to `playAudioWithLipSync` unconditionally stopped whatever was playing
- The full response text was dumped into the chat bubble instantly before Genevieve had spoken a single word

**What was changed in `static/index.html`:**

1. **Sequential audio queue** — `playAudioWithLipSync` replaced with `enqueueAudio(base64, text, bubble)` + `_drainQueue()`. Each clip plays to 100% completion via `activeSource.onended` before the next starts. Nothing ever calls `.stop()` on a playing source.
2. **Word-by-word text streaming** — `appendBotStreaming()` creates an empty chat bubble and returns the DOM element. During playback, `_playItem` calculates `msPerWord = buffer.duration × 1000 / wordCount` and uses `setInterval` to reveal one word at a time, keeping text appearance in step with speech. When the clip ends, any remaining words are flushed immediately as a safety net for rounding drift.

---

### Pipeline hardening & lip-sync improvements (Feb 2026)

#### LLM hallucination and off-topic responses

**Root causes:**
- The system prompt was a set of soft guidelines the model could drift away from
- A post-recommendation "chatting" phase gave the LLM near-free-form conversation rights
- The recommendation intro was LLM-generated — opening a window for hallucinated course names
- A "sentinel" JSON extraction path could be injected by a crafted user message

**What was changed in `src/nlp/ollama_conversation.py`:**

1. **New `_SYSTEM_PROMPT`** — 9 numbered absolute rules including hardcoded fallback sentences for off-topic prompts and help requests; zero allowed diversions
2. **Chatting phase removed** — after recommendations the session resets immediately to collecting; there is no free-form conversation mode
3. **Hardcoded `_REC_INTRO`** — spoken the moment recommendations fire; no LLM call, eliminating any chance of a hallucinated course name
4. **Hardcoded `_POST_REC_BRIDGE`** — invites the next search after recommendation cards are shown; also not LLM-generated
5. **Sentinel extraction deleted** — `_extract_sentinel`, `_SENTINEL_RE`, and the `json` import removed; server-side regex is the only source of truth
6. **Temperature `0.3 → 0.2`**, **`num_predict` `120 → 80`** — tighter constraint on creativity and length
7. **`stop` tokens added** (`[".", "?", "!"]`) — Ollama stops generating after the first sentence at the API level
8. **`_is_farewell` / `_FAREWELL_RE` removed** — the "end" action is gone; state resets via recommendation, not farewells

**What was changed in `src/api/routes.py`:**

9. **Bridge message synthesised and sent** after the recommendations payload so Genevieve speaks `_POST_REC_BRIDGE` aloud

#### Lip-sync timing

**Root causes:**
- Symmetric smoothing (`0.18` both ways) meant the mouth opened as slowly as it closed — ~130 ms lag before visible movement on each syllable
- Linear amplitude scaling (`avg × 3.5`) produced weak mouth movement for Edge TTS's typically quiet output
- Frequency band `300–3000 Hz` missed the `100–300 Hz` fundamental frequency range of the Jenny Neural voice

**What was changed in `static/index.html`:**

10. **Asymmetric smoothing** — open `0.40`, close `0.12`; mouth snaps open within ~2 frames and fades out naturally
11. **Wider frequency band** — `100–3500 Hz` (was `300–3000 Hz`)
12. **Non-linear amplitude curve** — `Math.pow(avg, 0.6) × 2.8` (was `avg × 3.5`); quiet speech now drives visible movement
13. **Talking-loop video rate widened** — `0.5–2.0×` (was `0.6–1.5×`)
14. **Bin-index computation moved outside `lipLoop`** — computed once per clip, not every animation frame

---

### Browser autoplay fix (Feb 2026)

**Problem:** Genevieve's greeting audio (and all subsequent audio) was silently dropped. `audioCtx.resume()` was called inside `ws.onmessage` — a network event, not a user gesture — so the browser refused it every time.

**Fix:** `_unlockAudio()` creates the `AudioContext` and calls `resume()` **synchronously inside** the click/keydown/mic handlers where the browser trusts the gesture. All audio is then queued behind `_audioUnlockedPromise()` and plays the instant the context is running.

---

## How the lip-sync works without MuseTalk

The project ships with two lip-sync modes. **Viseme mode** (default, no GPU required) makes Genevieve's mouth move in the browser. **MuseTalk mode** (optional, GPU required) replaces viseme sprites with photorealistic video.

### Viseme mode — end to end

**Step 1 — Six mouth sprites baked at startup**

`src/lipsync/viseme_generator.py` runs once when the server starts:

1. Loads the portrait photo (`static/images/portrait-business-woman-office.jpg`)
2. Crops and resizes the face to the 220×220 px canvas
3. Generates 6 JPEGs (`v0.jpg` … `v5.jpg`) — fully closed to wide open — by compositing a Gaussian-feathered dark ellipse (mouth cavity) and an ivory ellipse (teeth) over the original face with OpenCV. The soft blur blends edges into real skin.

**Step 2 — Audio synthesised server-side**

`src/tts/edge_tts.py` converts Genevieve's reply to MP3 via Microsoft Edge TTS. The bytes are base64-encoded and sent in the WebSocket `audio` field alongside the full text.

**Step 3 — Browser queues the clip and streams words**

On arrival the message is pushed onto `_audioQueue`. `_drainQueue` picks it up and calls `_playItem`:

1. `decodeAudioData` converts MP3 → PCM `AudioBuffer`
2. A `BufferSource → AnalyserNode → destination` graph starts playback
3. `msPerWord = buffer.duration × 1000 / wordCount` is calculated; `setInterval` appends one word at a time to the chat bubble
4. Each animation frame, `getByteFrequencyData` reads the `100–3500 Hz` band; energy is averaged, passed through `Math.pow(avg, 0.6) × 2.8`, and stored as `mouthTarget`
5. `mouthLoop` eases `mouthCurrent` toward `mouthTarget` asymmetrically (open `0.40`, close `0.12`) and maps it to a fractional sprite index. Two adjacent sprites are cross-faded on a `<canvas>` for smooth intermediate positions.
6. `onended` flushes remaining words, then calls `_drainQueue` to start the next clip.

---

## The autoplay fix

Modern browsers block audio until the user has physically interacted with the page. Genevieve's greeting arrives before any click, so `AudioContext.resume()` must be called from inside a gesture handler — not from `ws.onmessage`.

**The fix has three parts:**

**1. A gesture gate** resolves when the context is first unlocked:

```javascript
let _audioUnlocked = false;
let _unlockResolvers = [];

function _audioUnlockedPromise() {
  if (_audioUnlocked) return Promise.resolve();
  return new Promise(res => _unlockResolvers.push(res));
}
```

**2. `_unlockAudio()` is called synchronously inside every gesture handler** (send button, Enter key, mic button):

```javascript
function _unlockAudio() {
  if (_audioUnlocked) return;
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const p = audioCtx.state !== 'running' ? audioCtx.resume() : Promise.resolve();
  p.then(() => {
    _audioUnlocked = true;
    _unlockResolvers.forEach(r => r());
    _unlockResolvers = [];
  });
}
```

**3. `enqueueAudio()` waits behind the gate:**

```javascript
function _playItem({ bytes, text, bubble }) {
  _audioUnlockedPromise().then(() => {
    audioCtx.decodeAudioData(bytes.buffer.slice(0), buffer => { /* play */ });
  });
}
```

Greeting audio queues up immediately on connection and plays the instant the user first clicks Send or presses Enter.

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
  ├── Audio queue (_audioQueue)                   │
  │   └── clips play sequentially to completion   ├── STT: SpeechToText (Whisper tiny)
  ├── Word streaming (setInterval / msPerWord)    │       audio bytes → English text
  ├── Web Audio AnalyserNode → mouthTarget        ├── NLP: OllamaConversationManager
  ├── Asymmetric smoothing → mouthCurrent         │       regex extraction (goal/level/career)
  ├── Canvas viseme cross-fade (v0–v5)            │       Ollama/gemma3:4b — ask for missing info
  └── Course cards rendered from JSON            ├── Recommendations: recommend_courses()
                                                  │       all three → top 3 courses scored
                                                  └── TTS: EdgeTTS (MP3) → Piper fallback (WAV)
```

### Component responsibilities

| File | What it does |
|---|---|
| `app.py` | FastAPI entry point, lifespan (generates visemes, optionally loads MuseTalk) |
| `config.py` | All configuration via environment variables |
| `src/api/routes.py` | WebSocket handler, REST endpoints, post-rec bridge message |
| `src/stt/speech_to_text.py` | Whisper wrapper — accepts any browser audio format via ffmpeg |
| `src/nlp/ollama_conversation.py` | Strict conversation manager: server-side regex extraction, dynamic system prompt, hardcoded intros |
| `src/tts/edge_tts.py` | Edge TTS (online, MP3) with Piper offline fallback |
| `src/tts/piper_tts.py` | Offline Piper TTS — WAV bytes, no network required |
| `src/recommendations/engine.py` | Scores 24 courses by keyword overlap + level distance |
| `src/recommendations/courses.py` | 24-course catalogue: Python, JS, ML, data science, DevOps, SQL, cloud, design |
| `src/lipsync/viseme_generator.py` | Generates 6 mouth-state sprites from avatar photo at startup (OpenCV only) |
| `src/lipsync/musetalk_worker.py` | Optional GPU lip-sync: loads MuseTalk, runs per-response inference |
| `static/index.html` | Single-page app: audio queue, word streaming, Web Audio pipeline, canvas viseme renderer |

### Conversation flow

```
WebSocket connects
       │
       ▼
Hardcoded greeting spoken (no LLM call)
       │
       ▼
┌─────────────────────────────────────────────┐
│  COLLECTING                                 │
│  collected = { goal: ?, level: ?, career: ? }│
│                                             │
│  Each user turn:                            │
│  1. Regex extracts goal / level / career    │
│  2. LLM asks for still-missing items only   │
│     (one sentence, temperature 0.2)         │
└──────────────────┬──────────────────────────┘
                   │ all three filled
                   ▼
       recommend_courses() fires
       Hardcoded _REC_INTRO spoken
       Recommendation cards sent to browser
       Hardcoded _POST_REC_BRIDGE spoken
       collected reset to all-None
                   │
                   └──────► back to COLLECTING
                             (ready for next search)
```

The LLM is never called at the moment of recommendation. It is only called when asking for missing info. All other spoken strings are hardcoded.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10 or 3.11 | 3.12+ untested with some ML deps |
| ffmpeg (any modern) | Must be on `$PATH` — used by Whisper and TTS fallback |
| Ollama (latest) | Must be running before starting the server |
| `gemma3:4b` model | Pull with `ollama pull gemma3:4b` |
| Internet connection | Required for Edge TTS (or set up Piper for offline) |

---

## Step-by-step local setup

### 1. Clone the repository

```bash
git clone https://github.com/abiolaks/chatbot_engine_lms.git
cd chatbot_engine_lms
```

### 2. Create and activate a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Installs: `fastapi`, `uvicorn`, `openai-whisper`, `torch`, `edge-tts`, `piper-tts`, `opencv-python`, `httpx`, `python-multipart`, `websockets`, and supporting libraries.

### 4. Install ffmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install -y ffmpeg

# Windows — download from https://ffmpeg.org/download.html, add bin/ to PATH
```

Verify: `ffmpeg -version`

### 5. Install and start Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull gemma3:4b
ollama serve &
```

Verify: `curl http://localhost:11434/api/tags`

### 6. Verify the avatar image

`static/images/portrait-business-woman-office.jpg` must exist — it is the source for all six viseme sprites. To use a different avatar, replace this file and adjust the `CROP` / `MOUTH_*` constants in `src/lipsync/viseme_generator.py`.

### 7. (Optional) Piper offline TTS fallback

If you want audio even without internet access:

```bash
mkdir -p models/piper
wget -O models/piper/en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget -O models/piper/en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
export PIPER_MODEL_PATH=models/piper/en_US-amy-medium.onnx
```

Without Piper, Edge TTS works when online. If both fail, audio is silent but text still appears in the chat.

### 8. (Optional) Create a `.env` file

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

Startup output:

```
INFO  Generating 6 viseme sprites from static/images/portrait-business-woman-office.jpg …
INFO  Viseme sprites ready.
INFO  Server ready → http://localhost:8000/static/index.html
INFO  API docs    → http://localhost:8000/docs
```

Viseme sprites are cached — subsequent starts skip generation instantly.

### Open the chat interface

`http://localhost:8000/static/index.html`

### How to have a conversation

Genevieve collects three things before recommending courses:

| What she needs | Examples |
|---|---|
| **Learning goal** | Python, machine learning, web development, SQL, DevOps, cybersecurity |
| **Experience level** | beginner, intermediate, advanced |
| **Career goal** | data scientist, web developer, software engineer, ML engineer |

You can give all three at once:

> "I'm a complete beginner and I want to learn Python to become a data scientist."

Or answer her follow-up questions. The moment she has all three, she:
1. Calls the recommendation engine immediately (server-side, no LLM)
2. Speaks a hardcoded confirmation: *"Your personalised course recommendations are ready — take a look below."*
3. Shows the top 3 course cards in the chat
4. Speaks a hardcoded bridge: *"If you'd like to explore a different topic, just tell me what you want to learn next."*
5. Resets — ready for a new search straight away

Text streams word-by-word into the chat bubble in time with speech for every response.

### Voice input

Click the microphone button (🎤), speak, then click Stop (■). Your audio is recorded as WebM/Opus, sent over WebSocket, transcribed server-side by Whisper, and processed normally.

### API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/ws` | WebSocket | Real-time chat (audio + text) |
| `/api/v1/chat` | POST | Text-only REST endpoint |
| `/api/v1/chat/audio` | POST | Audio file upload → transcribe → chat |
| `/api/v1/session/{id}` | GET | Session state and recent history |
| `/api/v1/session/{id}` | DELETE | Clear a session |
| `/api/v1/health` | GET | Component health check |
| `/docs` | GET | Swagger UI |

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `LIPSYNC_MODE` | `viseme` | `viseme` = CPU sprites. `musetalk` = GPU video. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `gemma3:4b` | Model name (must be pulled) |
| `OLLAMA_TIMEOUT` | `30` | Seconds before Ollama times out |
| `WHISPER_MODEL` | `tiny` | `tiny` / `base` / `small` / `medium` / `large` |
| `EDGE_TTS_VOICE` | `en-US-JennyNeural` | Edge TTS voice |
| `EDGE_TTS_RATE` | `+0%` | Speech rate |
| `EDGE_TTS_PITCH` | `+0%` | Pitch |
| `PIPER_MODEL_PATH` | `models/piper/en_US-amy-medium.onnx` | Piper offline voice |

**Whisper model tradeoff:**

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `tiny` | 75 MB | fastest | good for clear English |
| `base` | 145 MB | fast | better |
| `small` | 465 MB | moderate | noticeably better |
| `medium` | 1.5 GB | slow on CPU | near-human |

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
│   │                                   #   audio queue, word streaming,
│   │                                   #   Web Audio lip-sync, viseme canvas
│   ├── images/
│   │   ├── portrait-business-woman-office.jpg  # Avatar source photo
│   │   └── visemes/
│   │       ├── v0.jpg                  # Mouth closed  (generated at startup)
│   │       ├── v1.jpg … v4.jpg
│   │       └── v5.jpg                 # Mouth wide open
│   └── videos/
│       └── talking_loop.mp4           # Optional ambient motion overlay
│
├── src/
│   ├── api/
│   │   └── routes.py                  # WebSocket + REST handlers
│   ├── stt/
│   │   └── speech_to_text.py          # Whisper wrapper
│   ├── tts/
│   │   ├── edge_tts.py                # Edge TTS primary + Piper fallback
│   │   └── piper_tts.py               # Offline Piper TTS
│   ├── nlp/
│   │   └── ollama_conversation.py     # Strict conversation manager
│   │                                  #   server-side regex extraction
│   │                                  #   hardcoded intros, no chatting phase
│   ├── recommendations/
│   │   ├── courses.py                 # 24-course catalogue
│   │   └── engine.py                  # Keyword + level scoring
│   └── lipsync/
│       ├── viseme_generator.py        # CPU sprite generator (active)
│       └── musetalk_worker.py         # GPU lip-sync wrapper (optional)
│
├── MuseTalk/                          # MuseTalk repo — excluded from git
│   ├── models/                        # 8.6 GB weights (download separately)
│   └── results/v15/avatars/genevieve/ # Cached face latents
│
└── models/
    └── piper/                         # Optional offline TTS voice models
```

---

## Deployment guide

### Option A — Single server (systemd + Nginx)

**Minimum spec:** Ubuntu 22.04, 2 vCPU, 4 GB RAM

#### 1. System packages

```bash
sudo apt-get update && sudo apt-get install -y \
  python3.10 python3.10-venv python3-pip ffmpeg git curl nginx
```

#### 2. Ollama

```bash
curl -fsSL https://ollama.ai/install.sh | sh
sudo systemctl enable --now ollama
ollama pull gemma3:4b
```

#### 3. App

```bash
git clone https://github.com/abiolaks/chatbot_engine_lms.git /opt/chatbot
cd /opt/chatbot
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

#### 4. Environment file

```bash
cat > /opt/chatbot/.env << 'EOF'
LIPSYNC_MODE=viseme
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
WHISPER_MODEL=tiny
EDGE_TTS_VOICE=en-US-JennyNeural
EOF
```

#### 5. systemd service

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

sudo systemctl daemon-reload && sudo systemctl enable --now chatbot
```

Logs: `sudo journalctl -fu chatbot`

#### 6. Nginx reverse proxy

WebSockets require explicit `Upgrade` headers — without them the browser connection silently fails:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;
    ssl_certificate     /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    location /api/v1/ws {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 86400;
    }

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
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

### Option B — Docker Compose

#### `docker-compose.yml`

```yaml
version: "3.9"
services:
  chatbot:
    build: .
    ports: ["8000:8000"]
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
    volumes: [ollama_data:/root/.ollama]
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

#### Run

```bash
docker-compose up --build -d
docker-compose exec ollama ollama pull gemma3:4b
# open http://localhost:8000/static/index.html
```

---

### Option C — GPU server (MuseTalk photorealistic lip-sync)

Replaces the viseme sprite step with a real talking-head video. Everything else in the pipeline is unchanged.

**Requirements:** NVIDIA GPU 8 GB VRAM+, CUDA 12, PyTorch 2.0.1

#### Clone MuseTalk and download weights

```bash
git clone https://github.com/TMElyralab/MuseTalk MuseTalk
cd MuseTalk && bash download_weights.sh && cd ..
```

#### Install MuseTalk Python dependencies

```bash
cd MuseTalk
pip install -r requirements.txt
pip install -U openmim
mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"
cd ..
```

#### Start in MuseTalk mode

```bash
LIPSYNC_MODE=musetalk python app.py
```

On first run the avatar face latents are prepared (~60 s) and cached to `MuseTalk/results/v15/avatars/genevieve/`. Subsequent starts skip preparation.

---

### Common deployment issues

| Symptom | Cause | Fix |
|---|---|---|
| "Reconnecting…" in browser | WebSocket blocked by proxy | Add `Upgrade` + `Connection` headers in Nginx config |
| Microphone button does nothing | HTTPS required for `getUserMedia` | Add TLS via certbot; browser auto-uses `wss://` on HTTPS pages |
| No audio at all | AudioContext not unlocked | User must click Send or mic before audio plays (autoplay policy) |
| Ollama timeout errors | Model cold or overloaded | Increase `OLLAMA_TIMEOUT`; try `gemma3:2b` for faster responses |
| Whisper crashes on startup | PyTorch / CUDA mismatch | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| Viseme sprites missing | Avatar image not found | Confirm `static/images/portrait-business-woman-office.jpg` exists |
| Edge TTS silent | No internet / API down | Set up Piper fallback (setup step 7) |
| MuseTalk OOM | Insufficient VRAM | Reduce `batch_size` in `musetalk_worker.py` `_make_args()` from 8 to 4 |

---

## MuseTalk GPU upgrade (optional)

See [Option C](#option-c--gpu-server-musetalk-photorealistic-lip-sync) above. The `src/lipsync/musetalk_worker.py` integration is already written and the avatar latents are pre-computed — enabling MuseTalk requires only cloning the MuseTalk repo, downloading weights, and setting `LIPSYNC_MODE=musetalk`.
