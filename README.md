# Genevieve — AI Avatar Learning Advisor

Genevieve is a fully local, voice-driven AI learning advisor. She listens to you, collects your learning goal, experience level, and career aspiration through natural conversation, then recommends personalised courses — all spoken aloud with real-time lip-sync animation and word-by-word text streaming running entirely in the browser.

---

## Table of Contents

1. [Changelog](#changelog)
2. [How the lip-sync works](#how-the-lip-sync-works)
3. [The autoplay fix](#the-autoplay-fix)
4. [Architecture overview](#architecture-overview)
5. [Prerequisites](#prerequisites)
6. [Step-by-step local setup](#step-by-step-local-setup)
7. [Running and using the app](#running-and-using-the-app)
8. [Configuration reference](#configuration-reference)
9. [Project structure](#project-structure)
10. [Deployment guide](#deployment-guide)
11. [MuseTalk GPU upgrade (optional)](#musetalk-gpu-upgrade-optional)
12. [API contract](#api-contract)
13. [Backend engineer handoff](#backend-engineer-handoff)
14. [Frontend developer handoff — LMS integration](#frontend-developer-handoff--lms-integration)

---

## Changelog

### Full pipeline reliability hardening (Feb 2026)

**Problems fixed:**

1. **Slow-motion mouth at idle** — the talking-loop video was playing at 0.2× speed when Genevieve was not speaking, causing the mouth to animate slowly with no audio. Fixed by keeping the video hidden during idle; the canvas shows the neutral v0 (closed-mouth) sprite instead.
2. **Lip sync not following TTS** — the video was being shown during speech but playback rate was not correctly driving it from audio amplitude. Fixed: video shown during speech, rate 1.0–1.8× driven by TTS analyser amplitude, paused between words (mouthTarget < 0.10) so the mouth never drifts to a wide-open frame on silence.
3. **"I'm here to match you with courses" response loop** — Ollama `stop` tokens `["\n", ".", "?", "!"]` were stripping terminal punctuation from responses. "What do you want to learn?" stopped at "?" without including it; code appended "." making it a statement, which the LLM then treated as off-topic. Fixed by removing stop tokens and using a first-sentence regex extractor to preserve original punctuation.
4. **LLM ignores instructions** — system prompt replaced with a shorter, simpler version that includes concrete few-shot examples, making `gemma3:4b` reliably ask for missing info instead of triggering the off-topic rule.
5. **Whisper `tiny` mis-transcribes conversational speech** — upgraded to `base` model (3× more accurate for everyday speech). Set `WHISPER_MODEL=base` in config.
6. **Empty recordings sent to LLM** — when Whisper returns nothing (silence, background noise, sub-second clip), the string `"[Could not understand audio]"` was passed to the LLM, which responded with the off-topic fallback. Now returns a TTS "I didn't catch that — please try again." without calling the LLM at all.
7. **Greeting plays simultaneously with microphone recording** — clicking mic unlocked the AudioContext, releasing the queued greeting audio at the same moment the mic opened. The mic captured Genevieve's voice. Fixed: mic click waits for the audio queue to drain (`_onQueueDrained` callback) before opening the microphone.
8. **VAD always returned zero energy** — `_vadAnalyser` was connected as a side branch (`micSrc → _vadAnalyser`) but not in the path to `destination`. Chrome does not process nodes outside the active graph. Fixed: `micSrc → _vadAnalyser → silentGain(0) → destination`.
9. **User corrections ignored** — once a value was extracted (e.g. level = beginner), saying "actually I'm advanced" was silently ignored. Fixed: correction trigger words ("actually", "no", "I meant") allow the extracted value to overwrite the stored one.
10. **Intent extraction too narrow** — added patterns for "just getting started", "comfortable with", "a few years", TypeScript, Next.js, AWS, Azure, PyTorch, TensorFlow, Kotlin, Flutter, "data analytics", "data pipeline", "build websites", "programming", "coding", etc.
11. **Old context after recommendations** — after firing recommendations the message history was not fully cleared, causing the LLM to reference the previous search topic in the next conversation. Fixed: `session["messages"]` reset to `[_REC_INTRO]` on recommendation.

---

### SadTalker talking-head video (Feb 2026)

Generated `static/videos/talking_loop.mp4` (438 KB, 440×440, H.264, 15.56 s) using SadTalker on CPU with `~/SadTalker/checkpoints/` individual `.pth` files. Shown as the face layer during speech; hidden during idle.

**Key notes for future regeneration:**
- SadTalker must be run with `--cpu` flag (not `--device cpu`)
- Use individual `.pth` files — do NOT use `curl -C -` to resume a safetensors download (byte-offset resume from HuggingFace causes data corruption with all tensor weights reading as 0.0 → black output frames)
- After render: `bash scripts/install_talking_loop.sh` resizes to 440×440 and installs to `static/videos/`

---

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

## How the lip-sync works

The project ships with three lip-sync layers. **Talking-head video** (SadTalker render, shown during speech) gives a natural animated face. **Viseme sprites** (canvas, always running) provide the mouth animation driven by live TTS audio amplitude. **MuseTalk mode** (optional, GPU) replaces both with photorealistic per-response video.

### During speech

The SadTalker video (`talking_loop.mp4`) is shown over the canvas. Its playback rate is driven by the TTS audio analyser: 1.0–1.8× proportional to amplitude, paused between words when `mouthTarget < 0.10`. When the video is visible the canvas sprite animation still runs underneath it and controls the ring glow.

### During idle

The video is hidden (`display: none`). The canvas shows sprite v0 — neutral closed mouth. No animation plays.

### MuseTalk GPU mode

See the [MuseTalk GPU upgrade](#musetalk-gpu-upgrade-optional) section.

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
  │              action, collected_info }                │
  │  receives: { type:"recommendations", courses }       │
  │                                                     │
  ▼                                                     ▼
index.html                                    src/api/routes.py
  ├── Audio queue (_audioQueue)                   │
  │   ├── clips play sequentially to completion   ├── STT: SpeechToText (Whisper base)
  │   └── _onQueueDrained — mic waits here        │       audio bytes → English text
  ├── Word streaming (setInterval / msPerWord)    ├── NLP: OllamaConversationManager
  ├── Web Audio AnalyserNode → mouthTarget        │       server-side regex extraction
  ├── Asymmetric smoothing → mouthCurrent         │       (goal / level / career)
  ├── Canvas viseme cross-fade (v0–v5)            │       Ollama/gemma3:4b — ask for missing
  ├── SadTalker video (rate 1.0–1.8×)            │       first-sentence extraction (no stop tokens)
  │   └── hidden at idle, shown during speech     ├── Recommendations: recommend_courses()
  ├── VAD: mic energy → auto-stop after 1.5 s     │       all three → top scored courses
  └── Course cards rendered from JSON            └── TTS: EdgeTTS (MP3) → Piper fallback (WAV)
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
WHISPER_MODEL=base
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
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large` |
| `EDGE_TTS_VOICE` | `en-US-JennyNeural` | Edge TTS voice |
| `EDGE_TTS_RATE` | `+0%` | Speech rate |
| `EDGE_TTS_PITCH` | `+0%` | Pitch |
| `PIPER_MODEL_PATH` | `models/piper/en_US-amy-medium.onnx` | Piper offline voice |

**Whisper model tradeoff:**

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `tiny` | 75 MB | fastest | poor for conversational speech |
| `base` | 145 MB | fast | **recommended — reliable for everyday English** |
| `small` | 465 MB | moderate | noticeably better, good for accented speech |
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
WHISPER_MODEL=base
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

---

## API contract

This section is the source of truth for both the backend engineer and the frontend/LMS developer. All integrations must conform to this contract.

### WebSocket — `/api/v1/ws`

#### Client → Server

```json
// Text input
{ "type": "text", "text": "I want to learn Python as a beginner" }

// Audio input (browser MediaRecorder output, base64-encoded)
{ "type": "audio", "audio": "<base64>", "mime": "audio/webm;codecs=opus" }
```

#### Server → Client (messages arrive in this order)

**1. Response message** — always sent after each user turn.

```json
{
  "type":           "response",
  "text":           "What's your experience level and your target career?",
  "audio":          "<base64 MP3>",
  "action":         "continue",
  "collected_info": {
    "goal":   "Python",
    "level":  null,
    "career": null
  }
}
```

| Field | Values | Notes |
|-------|--------|-------|
| `type` | `"response"` | Always this string |
| `text` | string | Spoken text — render in chat bubble |
| `audio` | base64 MP3 or `null` | Decode with `audioCtx.decodeAudioData()`. Null if TTS failed — show text only. |
| `action` | `"continue"` \| `"recommend"` | `"recommend"` means the next two messages follow immediately |
| `collected_info` | object | Current extraction state — use to show a progress indicator if desired |

**2. Recommendations message** — sent immediately after a `"recommend"` response.

```json
{
  "type": "recommendations",
  "courses": [
    {
      "title":    "Python for Everybody",
      "provider": "Coursera",
      "level":    "beginner",
      "duration": "4 weeks",
      "rating":   4.8,
      "reason":   "Highly rated beginner Python course aligned with data science goal"
    }
  ]
}
```

Up to 5 courses, sorted by relevance score. The `level` field is always `"beginner"`, `"intermediate"`, or `"advanced"`.

**3. Bridge message** — sent as a second `"response"` after recommendations, inviting the next search.

```json
{
  "type":   "response",
  "text":   "If you'd like to explore a different topic, just tell me what you want to learn next.",
  "audio":  "<base64 MP3>",
  "action": "continue",
  "collected_info": { "goal": null, "level": null, "career": null }
}
```

#### On connect

Immediately after the WebSocket handshake the server sends a greeting `response` message (with `audio`) and no user turn is needed. Clients must queue this audio and play it after the first user gesture (browser autoplay policy).

---

### REST endpoints

| Endpoint | Method | Body / Params | Returns |
|----------|--------|---------------|---------|
| `/api/v1/health` | GET | — | `{ "status": "ok", "components": {...} }` |
| `/api/v1/chat` | POST | `?session_id=&message=` | `{ session_id, response, action, audio, collected_info }` |
| `/api/v1/chat/audio` | POST | multipart: `session_id?`, `audio` file | `{ session_id, transcription, response, action, audio, collected_info }` |
| `/api/v1/session/{id}` | GET | — | `{ session_id, collected_info, history[-5:] }` |
| `/api/v1/session/{id}` | DELETE | — | `{ status: "cleared" }` |

Full Swagger UI: `/docs`

---

## Backend engineer handoff

### What is already built

The FastAPI application is functionally complete. The backend engineer's job is to harden it for production scale, not to rebuild it.

| Component | Status | File |
|-----------|--------|------|
| WebSocket handler | Done | `src/api/routes.py` |
| STT (Whisper base) | Done | `src/stt/speech_to_text.py` |
| LLM conversation manager | Done | `src/nlp/ollama_conversation.py` |
| TTS (Edge + Piper fallback) | Done | `src/tts/edge_tts.py` |
| Recommendation engine | Done | `src/recommendations/engine.py` |
| Course catalogue (24 courses) | Done | `src/recommendations/courses.py` |
| Viseme lip-sync sprites | Done | `src/lipsync/viseme_generator.py` |
| Deployment (systemd + Nginx + Docker) | Done | README |

### Tasks required for production

#### 1. Authentication

The WebSocket currently accepts any connection with no auth. Add token-based auth:

```python
# In websocket_endpoint, read a token from the query string:
# ws://host/api/v1/ws?token=<jwt>
# Validate against your LMS user system before calling websocket.accept()
```

#### 2. Replace in-memory session store with Redis

`OllamaConversationManager.sessions` is a plain Python dict. Server restarts or horizontal scaling silently loses all sessions. Replace with Redis:

```python
# pip install redis
import redis.asyncio as redis

class OllamaConversationManager:
    def __init__(self):
        self.redis = redis.from_url("redis://localhost:6379")

    async def get_session(self, session_id):
        raw = await self.redis.get(f"session:{session_id}")
        return json.loads(raw) if raw else None

    async def save_session(self, session_id, session):
        await self.redis.set(
            f"session:{session_id}", json.dumps(session), ex=3600
        )
```

#### 3. Move Whisper to async worker

`stt.transcribe()` is a synchronous blocking call that runs on the FastAPI event loop thread. For concurrent users this stalls all other requests. Move it to a thread pool:

```python
import asyncio, concurrent.futures
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

async def transcribe_async(audio_bytes, mime_type):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, stt.transcribe, audio_bytes, mime_type
    )
```

#### 4. CORS middleware

The LMS frontend will be on a different origin. Add CORS before shipping:

```python
# In app.py
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-lms-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 5. Piper offline TTS

Edge TTS requires internet. Download a Piper voice model for offline fallback (see setup step 7 above). Set `PIPER_MODEL_PATH` in the deployment `.env`.

#### 6. Environment variables for production

```bash
LIPSYNC_MODE=viseme
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_TIMEOUT=30
WHISPER_MODEL=base
EDGE_TTS_VOICE=en-US-JennyNeural
PIPER_MODEL_PATH=models/piper/en_US-amy-medium.onnx
REDIS_URL=redis://localhost:6379      # add when Redis is integrated
```

#### 7. Extend the course catalogue

`src/recommendations/courses.py` contains 24 courses. Add more by appending to the `COURSES` list using the same dict schema:

```python
{
    "id":       "py-adv-01",
    "title":    "Advanced Python — Decorators and Metaclasses",
    "provider": "Pluralsight",
    "level":    "advanced",         # "beginner" | "intermediate" | "advanced"
    "duration": "6 hours",
    "rating":   4.7,
    "tags":     ["python", "advanced", "software engineer", "backend developer"],
}
```

Tags drive the keyword scoring in `src/recommendations/engine.py`. Match them to the values extracted by the regex patterns in `ollama_conversation.py`.

---

## Frontend developer handoff — LMS integration

### What you are integrating

A voice-driven AI advisor widget that:
1. Opens a WebSocket connection to the backend
2. Speaks to the user via TTS audio + animated avatar
3. Listens to the user via microphone (VAD auto-stop)
4. Returns up to 5 structured course recommendations as JSON

The standalone prototype is `static/index.html` — use it as the definitive reference implementation. All production logic is there.

### Minimum integration requirements

| Requirement | Why |
|-------------|-----|
| **HTTPS only** | `getUserMedia()` (microphone) is blocked on plain HTTP in all modern browsers |
| **WebSocket upgrade headers in proxy** | Nginx must pass `Upgrade: websocket` and `Connection: upgrade` — see deployment guide |
| **CORS** | Backend must whitelist your LMS origin — ask the backend engineer to set this up |
| **No iframe for audio** | `AudioContext` and `getUserMedia` do not work in cross-origin iframes without `allow="microphone; autoplay"` |

### Integration approach

Extract three pieces from `static/index.html` into your LMS:

#### A. WebSocket + audio engine (`genevieve-sdk.js`)

Copy the JavaScript from `static/index.html` into a module. The public API surface you need:

```javascript
// Connect and receive events
connect();                          // opens ws, sends greeting automatically

// Send a text message
sendText("I want to learn Python"); // calls ws.send(), shows bubble, triggers thinking

// Send audio (base64 blob from MediaRecorder)
ws.send(JSON.stringify({ type: "audio", audio: b64, mime: "audio/webm;codecs=opus" }));

// Receive events via ws.onmessage — two message types to handle:
//   msg.type === "response"        → play audio, show text bubble
//   msg.type === "recommendations" → render course cards
```

The full audio queue, VAD, `_unlockAudio()`, and word streaming logic must be included unchanged. Do not replace these with simpler audio playback — the autoplay unlock and queue sequencing are critical for correct behaviour.

#### B. Avatar panel

The avatar requires:

- A `220×220` `<canvas id="avatar-canvas">` for the viseme sprites
- A `<video id="talking-vid" loop muted playsinline preload="auto">` positioned absolutely over the canvas, hidden by default
- Six sprite images served from `/static/images/visemes/v0.jpg` … `v5.jpg`
- The `talking_loop.mp4` video served from `/static/videos/talking_loop.mp4`

The canvas renders at 60 fps via `requestAnimationFrame`. No GPU is required.

#### C. Course card rendering

On `msg.type === "recommendations"`, render `msg.courses` using your LMS card style. Each course object:

```typescript
interface Course {
  title:    string;   // "Python for Everybody"
  provider: string;   // "Coursera"
  level:    "beginner" | "intermediate" | "advanced";
  duration: string;   // "4 weeks"
  rating:   number;   // 4.8
  reason:   string;   // one-sentence relevance explanation
}
```

### Audio format

The server sends MP3 audio base64-encoded in the `audio` field. Decode with the Web Audio API:

```javascript
const bytes = Uint8Array.from(atob(msg.audio), c => c.charCodeAt(0));
audioCtx.decodeAudioData(bytes.buffer.slice(0), buffer => {
  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);
  source.start();
});
```

The analyser node for lip sync must sit between the source and destination — see `_playItem()` in `index.html` for the complete graph.

### Microphone and VAD

`getUserMedia` requires a user gesture and HTTPS. The mic button flow:

1. User clicks → `_unlockAudio()` → `getUserMedia()` → `MediaRecorder.start(250)`
2. VAD monitors 100–3500 Hz speech-band energy every 50 ms
3. After 1.5 s of silence following speech → `mediaRecorder.stop()` → audio blob sent
4. User can also click Stop manually at any time

Set `VAD_SPEECH_THRESHOLD` (default `30`, range 0–255) to calibrate for your deployment environment. Increase if false-positives on background noise; decrease if speech is not detected.

### Browser compatibility

| Browser | Status |
|---------|--------|
| Chrome 90+ | Full support |
| Edge 90+ | Full support |
| Firefox 90+ | Full support |
| Safari 15.4+ | Full support (uses `webkitAudioContext` — already handled) |
| Mobile Chrome | Full support over HTTPS |
| Mobile Safari | Requires HTTPS; test microphone permissions carefully |
