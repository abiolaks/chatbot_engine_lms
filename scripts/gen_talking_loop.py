#!/usr/bin/env python3
"""
Generate a talking_loop.mp4 where Genevieve's face makes the same natural
mouth movements as the business-woman driving video.

Algorithm
─────────
1. Detect 68-point face landmarks on Genevieve (once).
2. For every frame of the driver video, detect 68 landmarks on the
   business woman's face.
3. Compute the similarity transform (rotation + scale + translation) that
   maps the *driver's* eye/nose region onto *Genevieve's* — this normalises
   head pose and scale so only the mouth movement is transferred.
4. Replace Genevieve's lower-face (jaw + lips) landmarks with the
   rigidly-aligned driver's lower-face landmarks.
5. Use Delaunay-triangulated piecewise-affine warping to deform Genevieve's
   face from her original landmark positions to the new target positions.
6. Apply temporal smoothing (Gaussian) across all frames to remove jitter.
7. Encode the output frames as static/videos/talking_loop.mp4 via ffmpeg.

Usage:  python scripts/gen_talking_loop.py
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import face_alignment
import imageio
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import Delaunay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_IMG = Path("static/images/Genevieve.png")
DRV_VID = Path("static/videos/talking_loop.mp4")   # business-woman driver
OUT_VID = Path("static/videos/talking_loop.mp4")   # overwritten in-place
TMP_VID = Path("static/videos/_gen_tmp.mp4")

# Genevieve crop (matches viseme_generator.py)
CROP    = dict(sx=263, sy=130, sw=310, sh=310)
SIZE    = 440   # match driver-video dimensions

# ── Landmark index groups ─────────────────────────────────────────────────────
#   0-16  jaw outline       17-26 eyebrows       27-35 nose
#  36-47  eyes             48-59 outer lips      60-67 inner lips
_LOWER = list(range(0, 17)) + list(range(48, 68))   # jaw + lips  → animated
_REF   = list(range(36, 48)) + list(range(27, 31))  # eyes + nose  → rigid ref


# ── Boundary points to prevent extrapolation at image edges ──────────────────
def _border(W, H):
    return np.float32([
        [0, 0],       [W//4, 0],   [W//2, 0],   [3*W//4, 0],   [W-1, 0],
        [0, H//4],    [W-1, H//4],
        [0, H//2],    [W-1, H//2],
        [0, 3*H//4],  [W-1, 3*H//4],
        [0, H-1],     [W//4, H-1], [W//2, H-1], [3*W//4, H-1], [W-1, H-1],
    ])


# ── Rigid alignment ───────────────────────────────────────────────────────────
def _rigid_align(src_lm: np.ndarray, drv_lm: np.ndarray) -> np.ndarray:
    """
    Find the similarity transform (rot + scale + translate) that maps drv's
    reference landmarks onto src's.  Apply it to ALL drv landmarks so they
    live in src's image space.
    """
    s = src_lm[_REF].astype(np.float64)
    d = drv_lm[_REF].astype(np.float64)

    sc, dc   = s.mean(0), d.mean(0)
    s_std    = np.sqrt(((s - sc) ** 2).sum() / len(s))
    d_std    = np.sqrt(((d - dc) ** 2).sum() / len(d))
    scale    = s_std / d_std if d_std > 1e-8 else 1.0

    sn = (s - sc) / s_std
    dn = (d - dc) / d_std
    U, _, Vt = np.linalg.svd(dn.T @ sn)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    aligned = (drv_lm.astype(np.float64) - dc) / d_std @ R.T * s_std + sc
    return aligned.astype(np.float32)


def _target_lm(src_lm: np.ndarray, drv_lm: np.ndarray) -> np.ndarray:
    """
    Build target landmarks: Genevieve's upper face + driver's aligned
    lower face (jaw + lips).  This preserves her eyes/nose exactly while
    animating her mouth with the driver's expression.
    """
    aligned        = _rigid_align(src_lm, drv_lm)
    target         = src_lm.copy()
    target[_LOWER] = aligned[_LOWER]
    return target


# ── Piecewise affine warp ─────────────────────────────────────────────────────
def _warp_tri(src: np.ndarray, dst: np.ndarray,
              st: np.ndarray, dt: np.ndarray) -> None:
    """In-place: copy the src triangle st into the dst triangle dt."""
    Hs, Ws = src.shape[:2]
    Hd, Wd = dst.shape[:2]

    rs = cv2.boundingRect(st.reshape(1, -1, 2).astype(np.float32))
    rd = cv2.boundingRect(dt.reshape(1, -1, 2).astype(np.float32))
    if rs[2] <= 0 or rs[3] <= 0 or rd[2] <= 0 or rd[3] <= 0:
        return

    stl = st - [rs[0], rs[1]]
    dtl = dt - [rd[0], rd[1]]

    x1s, y1s = max(0, rs[0]), max(0, rs[1])
    x2s, y2s = min(Ws, rs[0] + rs[2]), min(Hs, rs[1] + rs[3])
    if x2s <= x1s or y2s <= y1s:
        return

    patch = src[y1s:y2s, x1s:x2s]
    stl_adj = (stl - np.array([x1s - rs[0], y1s - rs[1]])).astype(np.float32)

    M   = cv2.getAffineTransform(stl_adj, dtl.astype(np.float32))
    wp  = cv2.warpAffine(patch, M, (rd[2], rd[3]),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    mask = np.zeros((rd[3], rd[2]), np.uint8)
    cv2.fillConvexPoly(mask, dtl.round().astype(int), 255)

    x1d, y1d = max(0, rd[0]), max(0, rd[1])
    x2d, y2d = min(Wd, rd[0] + rd[2]), min(Hd, rd[1] + rd[3])
    if x2d <= x1d or y2d <= y1d:
        return

    ox1, ox2 = x1d - rd[0], x2d - rd[0]
    oy1, oy2 = y1d - rd[1], y2d - rd[1]

    roi  = dst[y1d:y2d, x1d:x2d]
    wp_r = wp[oy1:oy2, ox1:ox2]
    mk_r = mask[oy1:oy2, ox1:ox2, None].astype(np.float32) / 255.0
    roi[:] = np.clip(wp_r * mk_r + roi * (1 - mk_r), 0, 255).astype(np.uint8)


def _pa_warp(src: np.ndarray, src_pts: np.ndarray,
             dst_pts: np.ndarray) -> np.ndarray:
    """Piecewise affine warp: move src's pixels from src_pts → dst_pts."""
    out = np.zeros_like(src)
    tri = Delaunay(dst_pts)
    for s in tri.simplices:
        _warp_tri(src, out, src_pts[s], dst_pts[s])
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("Loading Genevieve.png …")
    raw = cv2.imread(str(SRC_IMG))
    if raw is None:
        raise FileNotFoundError(SRC_IMG)

    c = CROP
    crop     = raw[c["sy"]:c["sy"]+c["sh"], c["sx"]:c["sx"]+c["sw"]]
    src_face = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_LANCZOS4)
    src_rgb  = cv2.cvtColor(src_face, cv2.COLOR_BGR2RGB)

    log.info("Initialising face-alignment model (CPU) …")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False
    )

    log.info("Detecting landmarks on Genevieve …")
    src_lm = fa.get_landmarks(src_rgb)
    if src_lm is None:
        raise RuntimeError("No face detected in Genevieve.png — check crop.")
    src_lm = src_lm[0]   # (68, 2)
    log.info(f"  OK: {src_lm.shape}")

    bnd         = _border(SIZE, SIZE)
    src_pts_all = np.vstack([src_lm, bnd])

    # ── Read ALL driver frames ────────────────────────────────────────────────
    log.info("Reading driver video …")
    reader = imageio.get_reader(str(DRV_VID))
    fps    = reader.get_meta_data().get("fps", 25)
    drv_frames = []
    for frame in reader:
        f = frame[:, :, :3].astype(np.uint8)
        drv_frames.append(cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
    reader.close()
    N = len(drv_frames)
    log.info(f"  {N} frames @ {fps} fps")

    # ── Detect landmarks on every driver frame (cached to disk) ──────────────
    LM_CACHE = Path("static/videos/_drv_landmarks.npy")
    if LM_CACHE.exists():
        log.info("Loading cached driver landmarks …")
        drv_lms = list(np.load(str(LM_CACHE)))
    else:
        log.info("Detecting landmarks on driver frames (this takes a few minutes) …")
        drv_lms = []
        last_ok = None
        for i, frame in enumerate(drv_frames):
            lm = fa.get_landmarks(frame)
            if lm is None:
                lm = last_ok if last_ok is not None else src_lm
                log.warning(f"  frame {i}: no face — using previous")
            else:
                lm      = lm[0]
                last_ok = lm
            drv_lms.append(lm)
            if (i + 1) % 50 == 0:
                log.info(f"  {i+1}/{N} …")
        np.save(str(LM_CACHE), np.stack(drv_lms))

    # ── Temporal smoothing (Gaussian, sigma ≈ 1 frame) ───────────────────────
    log.info("Smoothing landmarks …")
    lm_arr = np.stack(drv_lms, axis=0)           # (N, 68, 2)
    lm_arr = gaussian_filter1d(lm_arr, sigma=1.2, axis=0)

    # ── Warp Genevieve for every frame ────────────────────────────────────────
    log.info("Warping Genevieve …")
    frames_out = []
    for i in range(N):
        tgt_lm  = _target_lm(src_lm, lm_arr[i])
        tgt_pts = np.vstack([tgt_lm, bnd])
        warped  = _pa_warp(src_rgb, src_pts_all, tgt_pts)
        frames_out.append(warped)
        if (i + 1) % 50 == 0:
            log.info(f"  {i+1}/{N} …")

    # ── Encode via ffmpeg for maximum browser compatibility ───────────────────
    log.info("Encoding …")
    with tempfile.TemporaryDirectory() as tmp:
        for i, frame in enumerate(frames_out):
            cv2.imwrite(
                f"{tmp}/f{i:05d}.png",
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(int(fps)),
                "-i", f"{tmp}/f%05d.png",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "20",
                "-preset", "fast",
                str(TMP_VID),
            ],
            check=True,
            capture_output=True,
        )

    shutil.move(str(TMP_VID), str(OUT_VID))
    log.info(f"Done → {OUT_VID}")
    log.info("Restart the server and reload the browser to use the new video.")


if __name__ == "__main__":
    main()
