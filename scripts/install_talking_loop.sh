#!/bin/bash
# After SadTalker finishes, run this to install the output as talking_loop.mp4
# Usage: bash scripts/install_talking_loop.sh

SADTALKER_OUT="/tmp/sadtalker_out"
DEST="static/videos/talking_loop.mp4"
CANVAS_SIZE=440

# Find the latest .mp4 in sadtalker output
RESULT=$(find "$SADTALKER_OUT" -name "*.mp4" | sort | tail -1)
if [ -z "$RESULT" ]; then
  echo "ERROR: No .mp4 found in $SADTALKER_OUT"
  exit 1
fi
echo "Found: $RESULT"

# Resize to 440x440 (match the canvas), encode browser-compatible H.264
ffmpeg -y -i "$RESULT" \
  -vf "scale=${CANVAS_SIZE}:${CANVAS_SIZE}:force_original_aspect_ratio=decrease,pad=${CANVAS_SIZE}:${CANVAS_SIZE}:(ow-iw)/2:(oh-ih)/2" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 -preset fast \
  -an \
  "$DEST"

echo "Installed → $DEST"
echo "Restart the server and reload the browser."
