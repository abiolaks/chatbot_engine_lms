"""
src/lipsync/musetalk_worker.py

Self-contained MuseTalk v1.5 worker for the FastAPI pipeline.
Active only when LIPSYNC_MODE=musetalk.

Architecture
────────────
  load_avatar()   — called ONCE at startup from app.py lifespan.
                    Loads all models (vae, unet, pe, whisper, fp, audio_processor)
                    and prepares avatar face latents (one-time ~60 s on GPU).

  generate_video(audio_bytes) → bytes (mp4)
                    Called per response. Converts TTS audio → lip-synced MP4.
                    Non-blocking: runs in a thread-pool executor.
                    Serialised via a threading.Lock (one GPU inference at a time).

Why we load models here instead of importing Avatar directly
────────────────────────────────────────────────────────────
  MuseTalk's realtime_inference.py defines Avatar as a class that references
  module-level globals (args, vae, unet, pe, whisper, …) which are only
  populated inside its  if __name__ == "__main__":  block.  We therefore:
    1. Load the models ourselves.
    2. Use importlib to load realtime_inference.py as a module WITHOUT
       running its __main__ block.
    3. Inject our loaded objects into that module's namespace so Avatar
       methods resolve them correctly at runtime.

Working-directory contract
──────────────────────────
  MuseTalk uses relative paths everywhere (./models/…, ./results/…).
  We chdir to MUSETALK_ROOT around every operation that needs it, and
  restore the original cwd afterwards.  A threading.Lock serialises
  inference so the chdir is safe even in a multi-threaded server.
"""

import asyncio
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path

# ── mmpose / face_detection stubs ─────────────────────────────────────────────
# preprocessing.py imports mmpose and face_detection at module level.
# Those packages are only used during avatar *preparation* (one-time).
# For inference (using cached latents) they are never called.
# We inject stub modules so realtime_inference.py can be loaded in any env.
def _inject_stubs() -> None:
    import cv2 as _cv2

    def _read_imgs(img_list):
        """Real read_imgs used by Avatar.init() to reload cached frames."""
        frames = []
        for img_path in img_list:
            frame = _cv2.imread(img_path)
            frames.append(frame)
        return frames

    for name in [
        "mmpose", "mmpose.apis", "mmpose.structures",
        "face_detection",
        "musetalk.utils.preprocessing",
    ]:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            # get_landmark_and_bbox is only called during preparation (never at inference)
            stub.get_landmark_and_bbox = None
            # read_imgs IS called in Avatar.init() to reload cached frames from disk
            stub.read_imgs = _read_imgs
            sys.modules[name] = stub

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
MUSETALK_ROOT = Path(__file__).parents[2] / "MuseTalk"
AVATAR_IMG    = str(Path(__file__).parents[2] / "static" / "images" / "portrait-business-woman-office.jpg")
AVATAR_ID     = "genevieve"
VERSION       = "v15"

# Output path that MuseTalk writes to (relative to MUSETALK_ROOT):
#   results/v15/avatars/genevieve/vid_output/response.mp4
_OUTPUT_MP4 = (
    MUSETALK_ROOT / "results" / VERSION / "avatars" / AVATAR_ID / "vid_output" / "response.mp4"
)

# ── Module-level singletons ────────────────────────────────────────────────────
_avatar  = None          # Avatar instance
_ri_mod  = None          # realtime_inference module with injected globals
_lock    = threading.Lock()   # serialise inference (GPU handles one at a time)


# ── args namespace ─────────────────────────────────────────────────────────────

def _make_args() -> types.SimpleNamespace:
    """
    Simulate the argparse.Namespace that Avatar class methods reference
    as module-level globals in realtime_inference.py.
    """
    return types.SimpleNamespace(
        version                    = VERSION,
        gpu_id                     = 0,
        vae_type                   = "sd-vae",
        # Relative paths — valid when cwd == MUSETALK_ROOT
        unet_config                = os.path.join("models", "musetalkV15", "musetalk.json"),
        unet_model_path            = os.path.join("models", "musetalkV15", "unet.pth"),
        whisper_dir                = os.path.join("models", "whisper"),
        bbox_shift                 = 0,
        result_dir                 = "results",
        extra_margin               = 10,
        fps                        = 25,
        audio_padding_length_left  = 2,
        audio_padding_length_right = 2,
        batch_size                 = 8,
        output_vid_name            = None,
        use_saved_coord            = False,
        saved_coord                = False,
        parsing_mode               = "jaw",
        left_cheek_width           = 90,
        right_cheek_width          = 90,
        # False → Avatar.inference() saves frames and assembles the MP4
        skip_save_images           = False,
    )


# ── Public: load once at startup ──────────────────────────────────────────────

def load_avatar(force_preparation: bool = False) -> None:
    """
    Load MuseTalk models and prepare the avatar (face latents).
    Call this once from app.py lifespan when LIPSYNC_MODE=musetalk.
    Requires a CUDA GPU and MuseTalk dependencies.
    """
    global _avatar, _ri_mod

    # Add MuseTalk root to sys.path so its packages are importable
    if str(MUSETALK_ROOT) not in sys.path:
        sys.path.insert(0, str(MUSETALK_ROOT))

    # Stub mmpose/face_detection before any MuseTalk import
    _inject_stubs()

    # MuseTalk uses relative paths — must run from its root
    orig_cwd = os.getcwd()
    os.chdir(MUSETALK_ROOT)
    try:
        import torch
        from transformers import WhisperModel
        from musetalk.utils.utils          import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing   import FaceParsing

        args = _make_args()

        # ── Device selection: CUDA → MPS (M1/M2) → CPU ────────────────────
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu_id}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info(f"MuseTalk: using device {device}")

        # float16 is supported on CUDA and on MPS (PyTorch ≥ 2.0)
        use_half = (device.type in ("cuda", "mps"))

        # ── Load neural network weights ────────────────────────────────────
        logger.info("Loading MuseTalk models (vae / unet / pe) …")
        vae, unet, pe = load_all_model(
            unet_model_path = args.unet_model_path,
            vae_type        = args.vae_type,
            unet_config     = args.unet_config,
            device          = device,
        )
        timesteps = torch.tensor([0], device=device)
        if use_half:
            pe         = pe.half().to(device)
            vae.vae    = vae.vae.half().to(device)
            unet.model = unet.model.half().to(device)
        else:
            pe         = pe.to(device)
            vae.vae    = vae.vae.to(device)
            unet.model = unet.model.to(device)
        weight_dtype = unet.model.dtype

        logger.info("Loading Whisper audio encoder …")
        audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
        whisper = WhisperModel.from_pretrained(args.whisper_dir)
        whisper = whisper.to(device=device, dtype=weight_dtype).eval()
        whisper.requires_grad_(False)

        logger.info("Initialising face parser …")
        fp = FaceParsing(
            left_cheek_width  = args.left_cheek_width,
            right_cheek_width = args.right_cheek_width,
        )

        # ── Load Avatar class via importlib (skips __main__ block) ─────────
        script_path = MUSETALK_ROOT / "scripts" / "realtime_inference.py"
        spec    = importlib.util.spec_from_file_location("musetalk_realtime", script_path)
        _ri_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ri_mod)

        # Inject all globals that Avatar methods reference at runtime
        _ri_mod.args            = args
        _ri_mod.vae             = vae
        _ri_mod.unet            = unet
        _ri_mod.pe              = pe
        _ri_mod.audio_processor = audio_processor
        _ri_mod.whisper         = whisper
        _ri_mod.device          = device
        _ri_mod.weight_dtype    = weight_dtype
        _ri_mod.timesteps       = timesteps
        _ri_mod.fp              = fp

        # datagen() defaults device="cuda:0" — patch it to use the real device
        import torch as _torch
        _real_device = device
        def _datagen_patched(whisper_chunks, vae_encode_latents,
                             batch_size=8, delay_frame=0, device=None):
            _dev = _real_device if device is None else device
            whisper_batch, latent_batch = [], []
            for i, w in enumerate(whisper_chunks):
                idx = (i + delay_frame) % len(vae_encode_latents)
                latent = vae_encode_latents[idx]
                whisper_batch.append(w)
                latent_batch.append(latent)
                if len(latent_batch) >= batch_size:
                    yield _torch.stack(whisper_batch), _torch.cat(latent_batch, dim=0)
                    whisper_batch, latent_batch = [], []
            if latent_batch:
                yield (_torch.stack(whisper_batch).to(_dev),
                       _torch.cat(latent_batch, dim=0).to(_dev))
        _ri_mod.datagen = _datagen_patched

        # ── Prepare avatar latents (one-time, ~60 s) ───────────────────────
        avatar_cache = MUSETALK_ROOT / "results" / VERSION / "avatars" / AVATAR_ID
        needs_prep   = force_preparation or not avatar_cache.exists()

        if needs_prep and avatar_cache.exists():
            # Remove old cache so Avatar() won't prompt for user input
            shutil.rmtree(avatar_cache)
            logger.info(f"Removed stale avatar cache at {avatar_cache}")

        logger.info(f"Creating Avatar (preparation={needs_prep}) …")
        _avatar = _ri_mod.Avatar(
            avatar_id   = AVATAR_ID,
            video_path  = AVATAR_IMG,
            bbox_shift  = 0,
            batch_size  = args.batch_size,
            preparation = needs_prep,
        )
        logger.info("MuseTalk Avatar ready — inference is available.")

        # ── MPS warm-up: compile kernels now so the first request is fast ──
        if device.type in ("mps", "cuda"):
            logger.info("Warming up MPS/CUDA kernels (first-request latency fix) …")
            try:
                _dummy_latent = torch.randn(
                    args.batch_size, 8, 32, 32,
                    dtype=weight_dtype, device=device
                )
                _dummy_audio = torch.randn(
                    args.batch_size, 50, 384,
                    dtype=weight_dtype, device=device
                )
                with torch.no_grad():
                    _ = unet.model(
                        _dummy_latent, timesteps,
                        encoder_hidden_states=_dummy_audio
                    ).sample
                    _ = vae.decode_latents(_)
                if device.type == "mps":
                    torch.mps.synchronize()
                del _dummy_latent, _dummy_audio, _
                logger.info("Kernel warm-up complete — first request will be fast.")
            except Exception as exc:
                logger.warning(f"Warm-up failed (non-fatal): {exc}")

    finally:
        os.chdir(orig_cwd)


# ── Public: per-response inference ────────────────────────────────────────────

async def generate_video(audio_bytes: bytes) -> bytes:
    """
    Convert TTS audio bytes → lip-synced MP4 bytes.
    Non-blocking: delegates to a thread-pool executor.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _generate_sync, audio_bytes)


def _generate_sync(audio_bytes: bytes) -> bytes:
    """Synchronous inference — called from thread-pool executor."""
    if _avatar is None:
        raise RuntimeError(
            "MuseTalk Avatar not loaded — call load_avatar() at startup."
        )

    with _lock:
        orig_cwd = os.getcwd()
        os.chdir(MUSETALK_ROOT)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp = Path(tmp_dir)

                # Detect audio format from magic bytes (WAV=RIFF, else assume MP3)
                ext       = "wav" if audio_bytes[:4] == b"RIFF" else "mp3"
                audio_in  = tmp / f"tts_input.{ext}"
                wav_16k   = tmp / "tts_16k.wav"
                audio_in.write_bytes(audio_bytes)

                # Convert to 16 kHz mono WAV (MuseTalk's audio encoder requirement)
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(audio_in),
                     "-ar", "16000", "-ac", "1", str(wav_16k)],
                    check=True, capture_output=True,
                )

                # Run lip-sync inference
                # Avatar.inference() writes the MP4 to:
                #   ./results/v15/avatars/genevieve/vid_output/response.mp4
                _avatar.inference(
                    audio_path       = str(wav_16k),
                    out_vid_name     = "response",
                    fps              = 25,
                    skip_save_images = False,
                )

                # Read the output and return as bytes
                if not _OUTPUT_MP4.exists():
                    raise FileNotFoundError(
                        f"MuseTalk did not produce output at {_OUTPUT_MP4}"
                    )
                return _OUTPUT_MP4.read_bytes()

        finally:
            os.chdir(orig_cwd)
