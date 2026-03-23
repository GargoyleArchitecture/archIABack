"""
Video Processor for EVRAG

Procesamiento de video con scene-change detection para extraer frames clave.
Basado en el paper EVRAG: usa detección de cambios de escena en lugar de muestreo fijo.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Scene:
    """
    Represents a detected scene in a video.
    
    Attributes:
        start_frame: Frame number where scene starts
        end_frame: Frame number where scene ends
        start_time_sec: Start time in seconds
        end_time_sec: End time in seconds
        representative_frame: Path to saved representative frame
        frame_index: Index of representative frame within scene
    """
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    representative_frame: Path | None = None
    frame_index: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
            "representative_frame": str(self.representative_frame) if self.representative_frame else None,
            "frame_index": self.frame_index,
        }
    
    @property
    def duration_sec(self) -> float:
        """Returns scene duration in seconds."""
        return self.end_time_sec - self.start_time_sec
    
    @property
    def frame_count(self) -> int:
        """Returns number of frames in scene."""
        return self.end_frame - self.start_frame + 1


class VideoProcessor:
    """
    Video processor with scene-change detection.
    
    Uses OpenCV to detect scene changes and extract representative frames.
    This is more efficient than fixed-rate sampling (1-2 fps) while preserving
    key visual content.
    
    Attributes:
        config: Configuration dictionary
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize video processor.
        
        Args:
            config: Configuration dictionary (default: EVRAG_CONFIG)
        """
        from .config import EVRAG_CONFIG
        self.config = config or EVRAG_CONFIG
        
        self.scenes: list[Scene] = []
        self.video_path: Path | None = None
        self.video_duration_sec: float = 0
        self.total_frames: int = 0
        self.fps: float = 0
    
    def compute_video_hash(self, video_path: Path) -> str:
        """
        Compute SHA256 hash of video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        
        with open(video_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def detect_scenes(self, video_path: Path | str) -> list[Scene]:
        """
        Detect scene changes in video using OpenCV.
        
        Uses HSV color space differences between consecutive frames.
        A scene cut is detected when the average pixel change exceeds threshold.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of detected Scene objects
        """
        import cv2
        
        video_path = Path(video_path)
        self.video_path = video_path
        
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration_sec = self.total_frames / self.fps
        
        print(f"Video: {video_path.name}")
        print(f"  Duration: {self.video_duration_sec:.1f}s ({self.video_duration_sec/60:.1f} min)")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
        
        # Scene detection parameters
        threshold = self.config["scene_detection_threshold"] / 100.0
        min_scene_frames = int(self.config["min_scene_length_sec"] * self.fps)
        
        scenes: list[Scene] = []
        current_scene_start = 0
        prev_frame = None
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to HSV for better color difference detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, hsv)
                diff_mean = np.mean(diff) / 255.0  # Normalize to 0-1
                
                # Detect scene change
                if diff_mean > threshold and (frame_idx - current_scene_start) >= min_scene_frames:
                    # End current scene, start new one
                    scenes.append(Scene(
                        start_frame=current_scene_start,
                        end_frame=frame_idx - 1,
                        start_time_sec=current_scene_start / self.fps,
                        end_time_sec=(frame_idx - 1) / self.fps,
                    ))
                    current_scene_start = frame_idx
            
            prev_frame = hsv
            frame_idx += 1
        
        # Don't forget the last scene
        scenes.append(Scene(
            start_frame=current_scene_start,
            end_frame=frame_idx - 1,
            start_time_sec=current_scene_start / self.fps,
            end_time_sec=(frame_idx - 1) / self.fps,
        ))
        
        cap.release()
        
        self.scenes = scenes
        
        print(f"  Detected scenes: {len(scenes)}")
        
        return scenes
    
    def extract_representative_frames(
        self,
        output_dir: Path | None = None,
        max_frames: int | None = None,
    ) -> list[Path]:
        """
        Extract representative frame from each scene.
        
        For each detected scene, extracts the middle frame as representative.
        
        Args:
            output_dir: Directory to save frames (default: config.frames_dir)
            max_frames: Maximum frames to extract (for long videos)
            
        Returns:
            List of paths to extracted frames
        """
        import cv2
        
        if not self.scenes:
            raise ValueError("No scenes detected. Call detect_scenes() first.")
        
        output_dir = output_dir or Path(self.config["frames_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Limit frames if needed
        scenes_to_process = self.scenes
        if max_frames and len(scenes_to_process) > max_frames:
            # Select evenly distributed scenes
            indices = np.linspace(0, len(scenes_to_process) - 1, max_frames, dtype=int)
            scenes_to_process = [self.scenes[i] for i in indices]
        
        cap = cv2.VideoCapture(str(self.video_path))
        extracted_frames: list[Path] = []
        
        for scene_idx, scene in enumerate(scenes_to_process):
            # Extract middle frame of scene
            middle_frame_idx = (scene.start_frame + scene.end_frame) // 2
            scene.frame_index = middle_frame_idx
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Save frame
                frame_path = output_dir / f"{self.video_path.stem}_scene_{scene_idx:03d}_frame_{middle_frame_idx:05d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                
                scene.representative_frame = frame_path
                extracted_frames.append(frame_path)
        
        cap.release()
        
        print(f"  Extracted frames: {len(extracted_frames)}")
        
        return extracted_frames
    
    def process_video(
        self,
        video_path: Path | str,
        force_reprocess: bool = False,
    ) -> dict[str, Any]:
        """
        Complete video processing pipeline.
        
        Args:
            video_path: Path to video file
            force_reprocess: If True, reprocess even if already processed
            
        Returns:
            Dictionary with processing results
        """
        video_path = Path(video_path)
        
        # Check if already processed
        processed_info_path = Path(self.config["processed_dir"]) / f"{video_path.stem}_info.json"
        
        if processed_info_path.exists() and not force_reprocess:
            import json
            print(f"Using cached processing info for {video_path.name}")
            return json.loads(processed_info_path.read_text())
        
        # Process video
        print(f"\nProcessing video: {video_path.name}")
        
        # Detect scenes
        scenes = self.detect_scenes(video_path)
        
        # Extract frames
        frames = self.extract_representative_frames()
        
        # Build result
        result = {
            "video_path": str(video_path),
            "video_hash": self.compute_video_hash(video_path),
            "video_duration_sec": self.video_duration_sec,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "scenes_detected": len(scenes),
            "frames_extracted": len(frames),
            "scenes": [s.to_dict() for s in scenes],
            "frame_paths": [str(f) for f in frames],
        }
        
        # Save processing info
        processed_info_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        processed_info_path.write_text(json.dumps(result, indent=2))
        
        return result
    
    def get_scenes_summary(self) -> str:
        """
        Returns human-readable summary of detected scenes.
        
        Returns:
            Formatted string with scene information
        """
        if not self.scenes:
            return "No scenes detected"
        
        lines = [
            f"Video: {self.video_path.name if self.video_path else 'Unknown'}",
            f"Duration: {self.video_duration_sec:.1f}s",
            f"Scenes detected: {len(self.scenes)}",
            "",
            "Scenes:",
        ]
        
        for i, scene in enumerate(self.scenes[:10], 1):  # Show first 10
            lines.append(
                f"  {i:2d}. [{scene.start_time_sec:5.1f}s - {scene.end_time_sec:5.1f}s] "
                f"({scene.duration_sec:5.1f}s, {scene.frame_count:3d} frames)"
            )
        
        if len(self.scenes) > 10:
            lines.append(f"  ... and {len(self.scenes) - 10} more scenes")
        
        return "\n".join(lines)
