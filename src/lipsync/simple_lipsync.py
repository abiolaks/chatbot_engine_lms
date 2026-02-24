# src/lipsync/simple_lipsync.py
import cv2
import numpy as np
from pathlib import Path
import logging
from moviepy.editor import VideoFileClip, AudioFileClip, ImageClip, CompositeVideoClip
import subprocess
import math
import wave
import struct

logger = logging.getLogger(__name__)

class SimpleLipSync:
    """
    Creates a video with a still image and subtle mouth movement based on audio amplitude.
    """
    def __init__(self, avatar_path: Path):
        self.avatar_path = avatar_path
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
    
    def create_video(self, audio_path: Path, output_path: Path, fps=30) -> Path:
        """
        Generate video with lip movement synchronized to audio amplitude.
        """
        # Load avatar image
        img = cv2.imread(str(self.avatar_path))
        if img is None:
            raise ValueError(f"Cannot load avatar from {self.avatar_path}")
        height, width = img.shape[:2]
        
        # Get audio duration and amplitude envelope
        duration, amplitudes = self._get_audio_amplitude(audio_path, fps)
        if duration == 0:
            # fallback: just still image
            return self._create_still_video(audio_path, output_path)
        
        total_frames = int(duration * fps)
        # Ensure amplitudes list length matches total_frames
        if len(amplitudes) < total_frames:
            # repeat last amplitude
            amplitudes.extend([amplitudes[-1]] * (total_frames - len(amplitudes)))
        elif len(amplitudes) > total_frames:
            amplitudes = amplitudes[:total_frames]
        
        # Video writer
        temp_video = self.temp_dir / f"temp_vid_{output_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
        
        mouth_center = (width // 2, int(height * 0.8))
        mouth_width = int(width * 0.15)
        mouth_height = int(height * 0.03)
        
        for i in range(total_frames):
            frame = img.copy()
            # Amplitude between 0 and 1, scale mouth height
            amp = amplitudes[i] if i < len(amplitudes) else 0
            # Map amp to mouth open factor (0 to 1)
            open_factor = min(1.0, amp * 3)  # amplify effect
            current_mouth_height = int(mouth_height * (0.5 + open_factor * 0.5))
            
            # Draw a simple mouth (ellipse or rectangle)
            cv2.ellipse(frame,
                        mouth_center,
                        (mouth_width, current_mouth_height),
                        0, 0, 360,
                        (0, 0, 255), -1)  # red mouth
            out.write(frame)
        
        out.release()
        
        # Combine with audio
        final_path = self._combine_audio_video(temp_video, audio_path, output_path)
        temp_video.unlink(missing_ok=True)
        return final_path
    
    def _get_audio_amplitude(self, audio_path: Path, fps: int):
        """
        Extract amplitude envelope from audio file.
        Returns (duration_in_seconds, list of amplitudes per frame)
        """
        try:
            import wave
            import numpy as np
            wf = wave.open(str(audio_path), 'rb')
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / framerate
            
            # Read all frames
            frames = wf.readframes(n_frames)
            if sampwidth == 2:
                dtype = np.int16
            elif sampwidth == 1:
                dtype = np.int8
            else:
                raise ValueError("Unsupported sample width")
            
            samples = np.frombuffer(frames, dtype=dtype)
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)  # mono
            
            # Normalize to 0-1
            samples = samples.astype(np.float32) / 32768.0
            
            # Compute amplitude per frame (RMS)
            frame_length = int(framerate / fps)
            amplitudes = []
            for i in range(0, len(samples), frame_length):
                segment = samples[i:i+frame_length]
                if len(segment) == 0:
                    break
                rms = np.sqrt(np.mean(segment**2))
                amplitudes.append(rms)
            
            wf.close()
            return duration, amplitudes
        except Exception as e:
            logger.error(f"Failed to extract amplitude: {e}")
            # fallback: assume 0 amplitude
            return 5.0, [0.0]
    
    def _combine_audio_video(self, video_path: Path, audio_path: Path, output_path: Path) -> Path:
        """Use ffmpeg to combine audio and video"""
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-shortest',
                str(output_path)
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            return output_path
        except Exception as e:
            logger.error(f"FFmpeg combine failed: {e}")
            return video_path
    
    def _create_still_video(self, audio_path: Path, output_path: Path) -> Path:
        """Fallback: create video with still image and audio"""
        img = cv2.imread(str(self.avatar_path))
        height, width = img.shape[:2]
        fps = 30
        duration = self._get_duration_ffprobe(audio_path)
        if duration <= 0:
            duration = 5.0
        
        temp_video = self.temp_dir / f"still_{output_path.stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
        total_frames = int(duration * fps)
        for _ in range(total_frames):
            out.write(img)
        out.release()
        
        return self._combine_audio_video(temp_video, audio_path, output_path)
    
    def _get_duration_ffprobe(self, audio_path: Path) -> float:
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', str(audio_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip())
        except:
            return 5.0