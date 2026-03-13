"""
src/lipsync/viseme_generator.py

Generates 6 viseme sprites (v0 … v5) from the portrait photo at server startup.
Uses only OpenCV + NumPy — no ML, no GPU.

Technique
─────────
For each mouth-open state (open_h = 0 … MAX_OPEN_H pixels):

  v0  open_h = 0   →  natural photo, unmodified (lips closed)
  v1…v5             →  jaw-warp: the lower face is physically shifted DOWN
                        by `open_h` pixels so the jaw actually drops and the
                        mouth opens using real face pixels — nothing is painted.

                     The gap that forms between the fixed upper lip and the
                     dropped jaw is filled with a Gaussian-feathered ellipse
                     whose colour is sampled from the actual lip pixels in the
                     portrait and darkened to ~20 % brightness.  This keeps
                     the mouth interior in exactly the right hue for this face.

The browser then cross-fades between adjacent sprites based on real-time audio
amplitude (analyser FFT), so the mouth appears to open/close smoothly in sync
with the TTS voice.

Auto-regeneration
─────────────────
ensure_visemes() writes a .source marker beside the sprites recording which
avatar image they came from.  On the next startup the marker is compared to
AVATAR_IMG.name; a mismatch (or missing marker) triggers regeneration.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Crop parameters computed by OpenCV face detection on Genevieve.png (864×1184).
CROP   = {"sx": 263, "sy": 130, "sw": 310, "sh": 310}
CANVAS = 220          # output canvas size in pixels

# Mouth geometry on the 220 × 220 canvas (OpenCV face + mouth detection)
MOUTH_CX  = 111       # lip-centre x
MOUTH_CY  = 161       # lip-parting line y  (upper–lower boundary)
MOUTH_HW  = 36        # half-width of the mouth region
NUM_VISEMES = 6       # v0 (closed) … v5 (wide open)
MAX_OPEN_H  = 12      # max jaw-drop in pixels at v5

VISEME_DIR = Path("static/images/visemes")
AVATAR_IMG = Path("static/images/Genevieve.png")


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_visemes(
    image_path: Path = AVATAR_IMG,
    output_dir: Path = VISEME_DIR,
    force: bool = False,
) -> list[str]:
    """
    Generate viseme sprites, skipping only when they were previously built
    from the same source image (tracked via a .source marker file).

    Returns a list of absolute file paths [v0.jpg … v5.jpg].
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths  = [str(output_dir / f"v{i}.jpg") for i in range(NUM_VISEMES)]
    marker = output_dir / ".source"

    if not force and all(Path(p).exists() for p in paths):
        if marker.exists() and marker.read_text().strip() == image_path.name:
            logger.info("Viseme sprites up-to-date — skipping generation.")
            return paths
        logger.info(f"Avatar changed to {image_path.name} — regenerating sprites.")

    logger.info(f"Generating {NUM_VISEMES} viseme sprites from {image_path} …")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Avatar image not found: {image_path}")

    c    = CROP
    crop = img[c["sy"]: c["sy"] + c["sh"], c["sx"]: c["sx"] + c["sw"]]
    face = cv2.resize(crop, (CANVAS, CANVAS), interpolation=cv2.INTER_LANCZOS4)

    for i in range(NUM_VISEMES):
        t      = i / (NUM_VISEMES - 1)
        open_h = int(round(t * MAX_OPEN_H))
        frame  = _make_viseme(face, MOUTH_CX, MOUTH_CY, MOUTH_HW, open_h)
        cv2.imwrite(paths[i], frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"  v{i}.jpg  open_h={open_h} px")

    marker.write_text(image_path.name)
    logger.info("Viseme sprites generated.")
    return paths


# ── Core: jaw-warp viseme ─────────────────────────────────────────────────────

def _make_viseme(
    face: np.ndarray,
    mx: int, my: int,
    hw: int,
    open_h: int,
) -> np.ndarray:
    """
    Produce one viseme frame via jaw-warp.

    Parameters
    ----------
    face   : 220×220 BGR portrait (the cropped + resized original photo)
    mx, my : mouth centre x and lip-parting y on the canvas
    hw     : mouth half-width
    open_h : how many pixels to drop the jaw (0 = closed)

    How it works
    ------------
    1. Keep the upper face (y < my) pixel-perfect from the original.
    2. Copy the lower face (y ≥ my) downward by open_h rows — the jaw
       physically drops using the real portrait pixels.
    3. Sample the lip-line pixel colour from the original photo and darken it
       to ~20 % to get a realistic mouth-interior colour for this face.
    4. Fill the gap (y = my … my+open_h) with that colour using a
       Gaussian-feathered ellipse mask so the edges blend naturally into both
       the upper lip above and the dropped jaw below.
    """
    if open_h <= 0:
        return face.copy()

    H, W   = face.shape[:2]
    canvas = face.astype(np.float32).copy()

    # ── 1. Jaw drop — shift entire lower face down ────────────────────────
    jaw_rows = H - my - open_h          # how many rows survive after the shift
    if jaw_rows > 0:
        canvas[my + open_h : H, :] = face[my : H - open_h, :].astype(np.float32)

    # ── 2. Sample natural lip colour → dark mouth interior ────────────────
    # Read from the original photo (before warp) at the parting line.
    sy1 = max(0,  my - 5);     sy2 = min(H, my + 5)
    sx1 = max(0,  mx - hw + 14); sx2 = min(W, mx + hw - 14)
    samp = face[sy1:sy2, sx1:sx2].astype(np.float32)
    lip_color = (
        np.median(samp.reshape(-1, 3), axis=0)
        if samp.size > 0
        else np.array([65., 42., 75.])
    )
    interior = lip_color * 0.22          # 22 % brightness → dark but hue-matched

    # ── 3. Feathered gap fill ─────────────────────────────────────────────
    # Ellipse centred in the gap, width = mouth width, height = gap height.
    # Gaussian blur on the mask feathers the boundary into both lip rows.
    gap_cy = my + open_h // 2
    gap_rx = max(1, hw - 10)             # slightly narrower than full lip width
    gap_ry = max(1, open_h // 2 + 1)

    mask   = np.zeros((H, W), dtype=np.float32)
    cv2.ellipse(mask, (mx, gap_cy), (gap_rx, gap_ry), 0, 0, 360, 1.0, -1)

    sigma  = max(1.5, open_h * 0.30)
    ks     = max(3, int(sigma * 3) | 1)
    mask   = cv2.GaussianBlur(mask, (ks, ks), sigmaX=sigma)

    layer  = np.full((H, W, 3), interior, dtype=np.float32)
    m3     = mask[:, :, np.newaxis]
    canvas = layer * m3 + canvas * (1.0 - m3)

    return np.clip(canvas, 0, 255).astype(np.uint8)
