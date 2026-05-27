"""
Video Processor for RAG Evaluation

Procesa videos para extracción de:
1. Transcripción de audio (Whisper)
2. Frames representativos (OpenCV)
3. Detección de escenas (cambio de escenas)
4. OCR de frames (texto en pantalla)

Este módulo convierte videos en "documentos" procesables para evaluación RAG.
"""

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Scene:
    """
    Represents a detected scene in a video.

    Attributes:
        start_frame: First frame of the scene
        end_frame: Last frame of the scene
        start_time_sec: Start time in seconds
        end_time_sec: End time in seconds
        representative_frame: Path to representative frame image
        frame_index: Index of representative frame
    """
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    representative_frame: str = ""
    frame_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
            "representative_frame": self.representative_frame,
            "frame_index": self.frame_index,
        }


@dataclass
class VideoInfo:
    """
    Complete information about a processed video.

    Attributes:
        video_path: Original video path
        video_hash: SHA-256 hash for cache validation
        video_duration_sec: Duration in seconds
        total_frames: Total number of frames
        fps: Frames per second
        scenes_detected: Number of scenes detected
        frames_extracted: Number of frames extracted
        scenes: List of detected scenes
        frame_paths: Paths to extracted frames
        transcript: Full video transcript
        transcript_path: Path to transcript file
    """
    video_path: str
    video_hash: str
    video_duration_sec: float = 0.0
    total_frames: int = 0
    fps: float = 0.0
    scenes_detected: int = 0
    frames_extracted: int = 0
    scenes: list[Scene] = field(default_factory=list)
    frame_paths: list[str] = field(default_factory=list)
    transcript: str = ""
    transcript_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_path": self.video_path,
            "video_hash": self.video_hash,
            "video_duration_sec": self.video_duration_sec,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "scenes_detected": self.scenes_detected,
            "frames_extracted": self.frames_extracted,
            "scenes": [s.to_dict() for s in self.scenes],
            "frame_paths": self.frame_paths,
            "transcript": self.transcript,
            "transcript_path": self.transcript_path,
        }

    def to_markdown(self) -> str:
        """Convert to Markdown for human-readable report."""
        lines = [
            f"# Video Information",
            f"",
            f"**Path:** {self.video_path}",
            f"**Hash:** {self.video_hash}",
            f"**Duration:** {self.video_duration_sec:.2f} seconds ({self.video_duration_sec/60:.1f} min)",
            f"**FPS:** {self.fps}",
            f"**Total Frames:** {self.total_frames}",
            f"**Scenes Detected:** {self.scenes_detected}",
            f"**Frames Extracted:** {self.frames_extracted}",
            f"",
            f"## Scenes",
            f"",
        ]

        for i, scene in enumerate(self.scenes):
            lines.append(f"### Scene {i+1}")
            lines.append(f"- **Time:** {scene.start_time_sec:.1f}s - {scene.end_time_sec:.1f}s")
            lines.append(f"- **Frames:** {scene.start_frame} - {scene.end_frame}")
            lines.append(f"- **Representative:** `{scene.representative_frame}`")
            lines.append("")

        if self.transcript:
            lines.append("## Transcript")
            lines.append("")
            lines.append(self.transcript[:2000] + "..." if len(self.transcript) > 2000 else self.transcript)

        return "\n".join(lines)


# =============================================================================
# VIDEO PROCESSOR
# =============================================================================

class VideoProcessor:
    """
    Processes videos for RAG evaluation.

    Extracts:
    1. Audio transcript (using Whisper or ffsubsync)
    2. Representative frames (using OpenCV scene detection)
    3. Scene boundaries (using shot detection)

    Attributes:
        frames_dir: Directory to save extracted frames
        processed_dir: Directory to save processed info
        use_whisper: Whether to use Whisper for transcription
    """

    def __init__(
        self,
        frames_dir: str = "back/videos/frames",
        processed_dir: str = "back/videos/processed",
        use_whisper: bool = True,
    ):
        """
        Initialize the video processor.

        Args:
            frames_dir: Directory to save extracted frames
            processed_dir: Directory to save processed metadata
            use_whisper: Use Whisper for transcription (requires whisper package)
        """
        self.frames_dir = Path(frames_dir)
        self.processed_dir = Path(processed_dir)
        self.use_whisper = use_whisper

        # Create directories
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, video_path: Path) -> str:
        """Compute SHA-256 hash of video file."""
        sha256 = hashlib.sha256()
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_video_info(self, video_path: Path) -> tuple[float, int, float]:
        """
        Get video metadata using OpenCV.

        Returns:
            Tuple of (duration_sec, total_frames, fps)
        """
        cap = cv2.VideoCapture(str(video_path))

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        cap.release()

        return duration_sec, total_frames, fps

    def _detect_scenes(
        self,
        video_path: Path,
        threshold: float = 30.0,
        min_scene_length_sec: float = 10.0,
        sample_rate: int = 30,  # Process every Nth frame for speed
    ) -> list[Scene]:
        """
        Detect scene changes in video.

        Uses histogram comparison to detect shot boundaries.
        Samples frames for faster processing of long videos.

        Args:
            video_path: Path to video file
            threshold: Sensitivity threshold (higher = fewer scenes)
            min_scene_length_sec: Minimum scene length in seconds
            sample_rate: Process every Nth frame (default: 30 for ~1fps)

        Returns:
            List of detected Scene objects
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        min_scene_frames = int(min_scene_length_sec * fps)

        scenes = []
        current_scene_start = 0
        prev_frame = None

        frame_idx = 0
        processed_frames = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"      Detecting scenes (sampling 1/{sample_rate} frames, ~{total_frames//sample_rate//60} min)...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames based on sample rate
            if frame_idx % sample_rate != 0:
                frame_idx += 1
                continue

            # Progress indicator
            processed_frames += 1
            if processed_frames % 100 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"        Progress: {progress:.1f}% (frame {frame_idx}/{total_frames})")

            # Convert to histogram
            hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = hist.flatten() / hist.sum()  # Normalize

            if prev_frame is not None:
                # Compare histograms
                score = cv2.compareHist(prev_frame, hist, cv2.HISTCMP_CORREL)

                # Scene change detected
                if score < (1.0 - threshold / 100.0):
                    # Adjust for sample rate
                    actual_frame_idx = frame_idx
                    actual_scene_start = current_scene_start * sample_rate
                    scene_length = (actual_frame_idx - actual_scene_start) // sample_rate
                    
                    if scene_length >= min_scene_frames:
                        scenes.append(Scene(
                            start_frame=actual_scene_start,
                            end_frame=actual_frame_idx - 1,
                            start_time_sec=actual_scene_start / fps,
                            end_time_sec=(actual_frame_idx - 1) / fps,
                        ))
                    current_scene_start = frame_idx

            prev_frame = hist
            frame_idx += 1

        # Add last scene
        actual_total = frame_idx
        actual_scene_start = current_scene_start * sample_rate
        scene_length = (actual_total - actual_scene_start) // sample_rate
        if scene_length >= min_scene_frames:
            scenes.append(Scene(
                start_frame=actual_scene_start,
                end_frame=actual_total - 1,
                start_time_sec=actual_scene_start / fps,
                end_time_sec=(actual_total - 1) / fps,
            ))

        cap.release()
        print(f"      Detected {len(scenes)} scenes")
        return scenes

    def _extract_representative_frames(
        self,
        video_path: Path,
        scenes: list[Scene],
        video_prefix: str,
    ) -> list[str]:
        """
        Extract representative frames for each scene.

        Args:
            video_path: Path to video file
            scenes: List of detected scenes
            video_prefix: Prefix for output filenames

        Returns:
            List of paths to extracted frames
        """
        cap = cv2.VideoCapture(str(video_path))
        frame_paths = []

        for i, scene in enumerate(scenes):
            # Get middle frame of scene
            mid_frame = (scene.start_frame + scene.end_frame) // 2
            scene.frame_index = mid_frame

            # Extract frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
            ret, frame = cap.read()

            if ret:
                # Save frame
                frame_path = self.frames_dir / f"{video_prefix}_scene_{i:03d}_frame_{mid_frame}.jpg"
                cv2.imwrite(str(frame_path), frame)
                scene.representative_frame = str(frame_path)
                frame_paths.append(str(frame_path))

        cap.release()
        return frame_paths

    def _transcribe_audio(
        self,
        video_path: Path,
        video_prefix: str,
    ) -> str:
        """
        Transcribe audio from video.

        Tries multiple methods:
        1. Whisper (if installed)
        2. ffsubsync (fallback)
        3. Skip if neither available

        Args:
            video_path: Path to video file
            video_prefix: Prefix for output files

        Returns:
            Transcribed text
        """
        transcript_path = self.processed_dir / f"{video_prefix}_transcript.txt"

        # Check if already transcribed
        if transcript_path.exists():
            return transcript_path.read_text(encoding="utf-8")

        # Check if audio file exists
        audio_path = video_path.with_suffix(".mp3")
        if not audio_path.exists():
            # Extract audio from video
            print(f"    Extracting audio from video...")
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(video_path),
                    "-vn", "-acodec", "libmp3lame", "-ab", "128k",
                    str(audio_path),
                ],
                capture_output=True,
                check=False,
            )

        if not audio_path.exists():
            print(f"    Warning: Could not extract audio")
            return ""

        # Try Whisper
        if self.use_whisper:
            try:
                import whisper
                print(f"    Transcribing with Whisper...")
                model = whisper.load_model("base")
                result = model.transcribe(str(audio_path), language="es")
                transcript = result["text"]

                # Save transcript
                transcript_path.write_text(transcript, encoding="utf-8")
                return transcript

            except ImportError:
                print(f"    Whisper not installed, skipping transcription")
            except Exception as e:
                print(f"    Whisper error: {e}")

        # Fallback: return empty
        print(f"    Skipping transcription (no Whisper)")
        return ""

    def process_video(
        self,
        video_path: Path,
        force_reprocess: bool = False,
    ) -> VideoInfo:
        """
        Process a single video file.

        Args:
            video_path: Path to video file
            force_reprocess: Force reprocessing even if cached

        Returns:
            VideoInfo with complete processing results
        """
        import re
        video_prefix = re.sub(r'[^\w-]', '_', video_path.stem)
        info_path = self.processed_dir / f"{video_prefix}_info.json"

        # Check cache
        if info_path.exists() and not force_reprocess:
            print(f"    Loading cached video info...")
            data = json.loads(info_path.read_text(encoding="utf-8"))
            return VideoInfo(**data)

        print(f"  Processing video: {video_path.name}")

        # Compute hash
        video_hash = self._compute_hash(video_path)

        # Get video metadata
        print(f"    Getting video metadata...")
        duration_sec, total_frames, fps = self._get_video_info(video_path)

        # Detect scenes
        print(f"    Detecting scenes...")
        scenes = self._detect_scenes(video_path)

        # Extract representative frames
        print(f"    Extracting representative frames...")
        frame_paths = self._extract_representative_frames(video_path, scenes, video_prefix)

        # Transcribe audio
        print(f"    Transcribing audio...")
        transcript = self._transcribe_audio(video_path, video_prefix)

        # Create VideoInfo
        video_info = VideoInfo(
            video_path=str(video_path),
            video_hash=video_hash,
            video_duration_sec=duration_sec,
            total_frames=total_frames,
            fps=fps,
            scenes_detected=len(scenes),
            frames_extracted=len(frame_paths),
            scenes=scenes,
            frame_paths=frame_paths,
            transcript=transcript,
            transcript_path=str(self.processed_dir / f"{video_prefix}_transcript.txt") if transcript else "",
        )

        # Save info
        info_path.write_text(json.dumps(video_info.to_dict(), indent=2), encoding="utf-8")
        print(f"    Saved info to {info_path}")

        return video_info

    def process_all_videos(
        self,
        videos_dir: Path,
        force_reprocess: bool = False,
    ) -> list[VideoInfo]:
        """
        Process all videos in a directory.

        Args:
            videos_dir: Directory containing video files
            force_reprocess: Force reprocessing

        Returns:
            List of VideoInfo objects
        """
        video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".webm"]
        video_files = [
            f for f in videos_dir.iterdir()
            if f.suffix.lower() in video_extensions
        ]

        if not video_files:
            print(f"  No video files found in {videos_dir}")
            return []

        print(f"  Found {len(video_files)} video(s)")

        video_infos = []
        for video_path in video_files:
            video_info = self.process_video(video_path, force_reprocess)
            video_infos.append(video_info)

        return video_infos
