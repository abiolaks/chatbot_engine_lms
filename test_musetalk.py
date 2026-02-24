"""
Quick smoke-test for the MuseTalk integration.
Run from the project root:
    LIPSYNC_MODE=musetalk python test_musetalk.py
"""
import asyncio
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from src.lipsync.musetalk_worker import load_avatar, generate_video

async def main():
    print("\n=== Step 1: load_avatar() ===")
    load_avatar()
    print("load_avatar() completed successfully.\n")

    print("=== Step 2: generate_video() with a silent WAV ===")
    # Minimal valid 16-kHz mono WAV (1 second of silence)
    import struct, wave, io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b'\x00\x00' * 16000)   # 1 s silence
    audio_bytes = buf.getvalue()

    video_bytes = await generate_video(audio_bytes)
    print(f"generate_video() returned {len(video_bytes):,} bytes of MP4.")

    out = "test_musetalk_output.mp4"
    with open(out, "wb") as f:
        f.write(video_bytes)
    print(f"Saved to {out}")
    print("\n=== All tests passed ===")

if __name__ == "__main__":
    asyncio.run(main())
