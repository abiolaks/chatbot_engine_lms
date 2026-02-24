"""
src/lipsync/viseme_generator.py

Generates N mouth-state (viseme) images from the real face photo at server
startup.  Uses only OpenCV — no ML, no GPU.

Technique per frame
───────────────────
1. Crop + resize the portrait to the 220 × 220 display canvas.
2. Erase the original closed-mouth detail with a soft skin-coloured patch
   (colour sampled from the cheek directly above the mouth).
3. Draw layered mouth shapes using Gaussian-feathered alpha compositing:
     a. Dark interior ellipse (the opening)
     b. Teeth strip (upper part of opening, when open ≥ 4 px)
     c. Upper lip band  (at the top edge of the opening)
     d. Lower lip band  (at the bottom edge of the opening)
   Each layer is a soft ellipse blended with a Gaussian-blurred mask,
   so edges feather naturally into the surrounding skin.
4. Save as JPEG to static/images/visemes/v{N}.jpg

All mouth coordinates were computed by OpenCV face detection against
portrait-business-woman-office.jpg (3840 × 5760 px).
"""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CROP = {"sx": 242, "sy": 793, "sw": 3269, "sh": 3269}
CANVAS = 220          # output size (px)

# Mouth on the 220 × 220 canvas (from OpenCV face detection)
MOUTH_CX = 110
MOUTH_CY = 149
MOUTH_HW = 37         # half-width  — matches JS constant M.hw
NUM_VISEMES = 6        # v0 (closed) … v5 (wide open)
MAX_OPEN_H  = 22       # max vertical half-height in px — matches JS M.maxH

VISEME_DIR = Path("static/images/visemes")
AVATAR_IMG = Path("static/images/portrait-business-woman-office.jpg")


# ── Public API ────────────────────────────────────────────────────────────────

def ensure_visemes(
    image_path: Path = AVATAR_IMG,
    output_dir: Path = VISEME_DIR,
    force: bool = False,
) -> list[str]:
    """
    Generate viseme sprites if they don't already exist.
    Returns a list of absolute file paths [v0.jpg … v5.jpg].
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(output_dir / f"v{i}.jpg") for i in range(NUM_VISEMES)]

    if not force and all(Path(p).exists() for p in paths):
        logger.info("Viseme sprites already up-to-date — skipping generation.")
        return paths

    logger.info(f"Generating {NUM_VISEMES} viseme sprites from {image_path} …")

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Avatar image not found: {image_path}")

    # Crop and resize to display canvas
    c = CROP
    crop = img[c["sy"]: c["sy"] + c["sh"], c["sx"]: c["sx"] + c["sw"]]
    face = cv2.resize(crop, (CANVAS, CANVAS), interpolation=cv2.INTER_LANCZOS4)

    for i in range(NUM_VISEMES):
        t      = i / (NUM_VISEMES - 1)       # 0.0 … 1.0
        open_h = int(round(t * MAX_OPEN_H))
        frame  = _make_viseme(face, MOUTH_CX, MOUTH_CY, MOUTH_HW, open_h)
        cv2.imwrite(paths[i], frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
        logger.info(f"  v{i}.jpg  (open_h={open_h}px)")

    logger.info("Viseme generation complete.")
    return paths


# ── Core: layered alpha-composite mouth blend ─────────────────────────────────

def _make_viseme(
    face: np.ndarray,
    mx: int, my: int,
    hw: int,
    open_h: int,
) -> np.ndarray:
    """
    Composite an open-mouth shape into `face` using layered soft fills.

    Each layer is a Gaussian-feathered ellipse so edges blend naturally
    into the surrounding skin without hard boundaries or colour mismatch.

    Parameters
    ----------
    face   : 220 × 220 BGR image
    mx, my : mouth centre on the canvas
    hw     : half-width of the mouth opening ellipse
    open_h : vertical half-height  (0 = closed mouth)
    """
    if open_h <= 0:
        return face.copy()

    H, W = face.shape[:2]
    canvas = face.astype(np.float32).copy()

    # ── Colour constants ───────────────────────────────────────────────────
    dark  = np.array([20., 12., 15.])       # dark mouth interior (BGR)
    ivory = np.array([218., 213., 208.])    # teeth (off-white, BGR)

    # ── 1. Dark mouth interior ─────────────────────────────────────────────
    # Draw directly onto the original face — the Gaussian feathering lets
    # the real lip texture show through at the edges, so no artificial
    # lip-colour bands are needed.
    _soft_fill(canvas, mx, my, max(1, hw - 4), max(1, open_h), dark, sigma=3)

    # ── 2. Teeth strip (upper portion of the opening) ─────────────────────
    if open_h >= 5:
        teeth_h = max(1, open_h // 3)
        teeth_w = max(1, hw - 10)
        teeth_y = my - open_h + teeth_h + 1
        _soft_fill(canvas, mx, teeth_y, teeth_w, teeth_h, ivory, sigma=1)

    return np.clip(canvas, 0, 255).astype(np.uint8)


# ── Helper ────────────────────────────────────────────────────────────────────

def _soft_fill(
    canvas: np.ndarray,
    cx: int, cy: int,
    rx: int, ry: int,
    color,
    sigma: float = 3.0,
) -> None:
    """
    Paint a Gaussian-feathered ellipse of `color` into float32 `canvas` in-place.

    Parameters
    ----------
    canvas  : float32 H×W×3 image (modified in-place)
    cx, cy  : ellipse centre
    rx, ry  : semi-axes (half-width, half-height)
    color   : BGR colour (array-like, float values)
    sigma   : Gaussian blur sigma for edge feathering
    """
    H, W = canvas.shape[:2]
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.ellipse(
        mask,
        (int(cx), int(cy)),
        (max(1, int(rx)), max(1, int(ry))),
        0, 0, 360, 1.0, -1,
    )
    if sigma > 0:
        ks = max(3, int(sigma * 3) | 1)   # odd kernel ≥ 3
        mask = cv2.GaussianBlur(mask, (ks, ks), sigmaX=sigma)

    layer = np.full((H, W, 3), np.array(color, dtype=np.float32), dtype=np.float32)
    m3 = mask[:, :, np.newaxis]
    canvas[:] = layer * m3 + canvas * (1.0 - m3)
