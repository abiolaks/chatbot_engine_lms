"""
prepare_avatar.py — One-time avatar preparation for MuseTalk (no mmpose needed).

Run from the project root:
    python prepare_avatar.py

What it does:
  Replicates MuseTalk's Avatar.prepare_material() using `face_alignment`
  (pip-installed, 68-point model) instead of mmpose/dwpose.
  Saves latents.pt, coords.pkl, mask*.pkl and PNG frames so that
  musetalk_worker.py can load the avatar with preparation=False.
"""

import os
import sys
import glob
import json
import pickle
import shutil
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths (mirrored from musetalk_worker.py) ─────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent
MUSETALK_ROOT = PROJECT_ROOT / "MuseTalk"
AVATAR_IMG    = PROJECT_ROOT / "static" / "images" / "portrait-business-woman-office.jpg"
AVATAR_ID     = "genevieve"
VERSION       = "v15"

AVATAR_DIR    = MUSETALK_ROOT / "results" / VERSION / "avatars" / AVATAR_ID
FULL_IMGS_DIR = AVATAR_DIR / "full_imgs"
MASK_DIR      = AVATAR_DIR / "mask"
VID_OUT_DIR   = AVATAR_DIR / "vid_output"


# ── Bootstrap MuseTalk imports ────────────────────────────────────────────────
if str(MUSETALK_ROOT) not in sys.path:
    sys.path.insert(0, str(MUSETALK_ROOT))
os.chdir(MUSETALK_ROOT)   # MuseTalk uses relative paths


def _load_models():
    """Load VAE, UNet, pe, FaceParsing — same as musetalk_worker.load_avatar."""
    import types
    # Stub mmpose so realtime_inference.py can be imported later
    for name in ["mmpose", "mmpose.apis", "mmpose.structures",
                 "face_detection", "musetalk.utils.preprocessing"]:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.get_landmark_and_bbox = None
            stub.read_imgs = None
            sys.modules[name] = stub

    from musetalk.utils.utils import load_all_model
    from musetalk.utils.face_parsing import FaceParsing

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    log.info(f"Using device: {device}")

    vae, unet, pe = load_all_model(
        unet_model_path="models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        unet_config="models/musetalkV15/musetalk.json",
        device=device,
    )
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
    return vae, fp, device


def _get_landmark_and_bbox_fa(img_list: list, bbox_shift: int = 0):
    """
    face_alignment-based replacement for musetalk.utils.preprocessing.get_landmark_and_bbox.

    Returns (coord_list, frame_list) matching MuseTalk's expected format.
    coord_list: list of (x1, y1, x2, y2) tuples — crop box per frame
    frame_list: list of BGR numpy arrays
    """
    import face_alignment as fa_lib

    device_str = "mps" if torch.backends.mps.is_available() else (
                 "cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"face_alignment device: {device_str}")
    fa = fa_lib.FaceAlignment(
        fa_lib.LandmarksType.TWO_D,
        flip_input=False,
        device=device_str,
    )

    coord_placeholder = (0.0, 0.0, 0.0, 0.0)
    coord_list = []
    frame_list = []

    for img_path in tqdm(img_list, desc="Landmark detection"):
        frame = cv2.imread(str(img_path))
        frame_list.append(frame)

        # face_alignment expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(rgb)

        if preds is None or len(preds) == 0:
            log.warning(f"No face detected in {img_path}")
            coord_list.append(coord_placeholder)
            continue

        # preds[0] shape: (68, 2) — 68 face landmarks
        lm = preds[0].astype(np.int32)

        # Replicate MuseTalk's half-face boundary logic:
        #   half_face_coord = lm[29]  (nose bridge / mid-nose)
        #   upper_bond = half_face_coord[1] - (max_y - half_face_coord[1])
        half_face_coord = lm[29].copy()
        if bbox_shift != 0:
            half_face_coord[1] += bbox_shift

        half_face_dist = np.max(lm[:, 1]) - half_face_coord[1]
        upper_bond = max(0, half_face_coord[1] - half_face_dist)

        x1 = int(np.min(lm[:, 0]))
        y1 = int(upper_bond)
        x2 = int(np.max(lm[:, 0]))
        y2 = int(np.max(lm[:, 1]))

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            log.warning(f"Bad landmark bbox {(x1,y1,x2,y2)} for {img_path}; falling back to face bbox")
            # fallback: use face bounding box from landmarks
            margin = 20
            x1 = max(0, int(np.min(lm[:, 0])) - margin)
            y1 = max(0, int(np.min(lm[:, 1])) - margin)
            x2 = int(np.max(lm[:, 0])) + margin
            y2 = int(np.max(lm[:, 1])) + margin

        coord_list.append((x1, y1, x2, y2))

    return coord_list, frame_list


def prepare_avatar(force: bool = False):
    """Run the full avatar preparation pipeline."""

    if AVATAR_DIR.exists() and not force:
        log.info(f"Avatar cache already exists at {AVATAR_DIR}. Use force=True to re-run.")
        return

    if AVATAR_DIR.exists():
        shutil.rmtree(AVATAR_DIR)

    for d in [AVATAR_DIR, FULL_IMGS_DIR, MASK_DIR, VID_OUT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. Save avatar info ───────────────────────────────────────────────────
    avatar_info = {
        "avatar_id": AVATAR_ID,
        "video_path": str(AVATAR_IMG),
        "bbox_shift": 0,
        "version": VERSION,
    }
    with open(AVATAR_DIR / "avator_info.json", "w") as f:
        json.dump(avatar_info, f)

    # ── 2. Read + resize avatar image, save as frame 00000000.png ────────────
    log.info("Reading and resizing avatar image …")
    img = cv2.imread(str(AVATAR_IMG))
    if img is None:
        raise FileNotFoundError(f"Avatar image not found: {AVATAR_IMG}")

    # MuseTalk is designed for ~512-1080p input; very large images break face
    # detectors and consume excessive memory.  Resize so the longest side ≤ 1024.
    MAX_SIDE = 1024
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        log.info(f"Resized from ({w},{h}) → ({img.shape[1]},{img.shape[0]})")

    cv2.imwrite(str(FULL_IMGS_DIR / "00000000.png"), img)

    # ── 3. Load MuseTalk models ────────────────────────────────────────────────
    log.info("Loading MuseTalk models …")
    vae, fp, device = _load_models()

    # ── 4. Get face crop coordinates via face_alignment ───────────────────────
    img_list = sorted(glob.glob(str(FULL_IMGS_DIR / "*.png")))
    log.info(f"Running landmark detection on {len(img_list)} frame(s) …")
    coord_list, frame_list = _get_landmark_and_bbox_fa(img_list, bbox_shift=0)

    # ── 5. Build latents for each crop ───────────────────────────────────────
    log.info("Encoding face crops into VAE latents …")
    input_latent_list = []
    coord_placeholder = (0.0, 0.0, 0.0, 0.0)

    EXTRA_MARGIN = 10   # matches args.extra_margin in musetalk_worker.py

    for i, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
        if bbox == coord_placeholder:
            log.warning(f"Skipping frame {i} (no face).")
            continue
        x1, y1, x2, y2 = bbox
        # v15 extra margin
        y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])
        coord_list[i] = (x1, y1, x2, y2)

        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latent = vae.get_latents_for_unet(resized)
        input_latent_list.append(latent)

    if not input_latent_list:
        raise RuntimeError("No faces detected in the avatar image — preparation failed.")

    # ── 6. Cycle (ping-pong) the lists ────────────────────────────────────────
    frame_list_cycle  = frame_list  + frame_list[::-1]
    coord_list_cycle  = coord_list  + coord_list[::-1]
    latent_list_cycle = input_latent_list + input_latent_list[::-1]

    # ── 7. Save full-res frames + mask data ───────────────────────────────────
    log.info(f"Saving {len(frame_list_cycle)} cycle frames and masks …")
    from musetalk.utils.blending import get_image_prepare_material

    mask_coords_list = []
    mask_list = []

    for i, frame in enumerate(tqdm(frame_list_cycle, desc="Saving frames + masks")):
        out_path = str(FULL_IMGS_DIR / f"{str(i).zfill(8)}.png")
        cv2.imwrite(out_path, frame)

        x1, y1, x2, y2 = coord_list_cycle[i]
        mask, crop_box = get_image_prepare_material(
            frame, [x1, y1, x2, y2], fp=fp, mode="jaw"
        )
        mask_coords_list.append(crop_box)
        mask_list.append(mask)

        cv2.imwrite(str(MASK_DIR / f"{str(i).zfill(8)}.png"), mask)

    # ── 8. Persist to disk ───────────────────────────────────────────────────
    log.info("Saving latents.pt …")
    torch.save(latent_list_cycle, str(AVATAR_DIR / "latents.pt"))

    log.info("Saving coords.pkl …")
    with open(AVATAR_DIR / "coords.pkl", "wb") as f:
        pickle.dump(coord_list_cycle, f)

    log.info("Saving mask_coords.pkl …")
    with open(AVATAR_DIR / "mask_coords.pkl", "wb") as f:
        pickle.dump(mask_coords_list, f)

    log.info(f"Avatar preparation complete → {AVATAR_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Re-prepare even if avatar cache already exists")
    args = parser.parse_args()
    prepare_avatar(force=args.force)
