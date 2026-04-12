"""
Audio Transcriber for EVRAG

Transcripción de audio usando Whisper (open-source, local).
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionResult:
    """
    Result of audio transcription.
    
    Attributes:
        text: Full transcribed text
        segments: List of segments with timestamps
        language: Detected language
        duration_sec: Audio duration in seconds
    """
    text: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    language: str = ""
    duration_sec: float = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "segments": self.segments,
            "language": self.language,
            "duration_sec": self.duration_sec,
        }
    
    def save(self, output_path: Path) -> None:
        """Save transcription to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
    
    @classmethod
    def load(cls, json_path: Path) -> "TranscriptionResult":
        """Load transcription from JSON file."""
        data = json.loads(json_path.read_text())
        return cls(
            text=data["text"],
            segments=data["segments"],
            language=data.get("language", ""),
            duration_sec=data.get("duration_sec", 0),
        )


class AudioTranscriber:
    """
    Audio transcriber using Whisper.
    
    Uses OpenAI's Whisper model for speech-to-text transcription.
    Supports both local (open-source) and API versions.
    
    Attributes:
        config: Configuration dictionary
        model: Loaded Whisper model (for local mode)
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize audio transcriber.
        
        Args:
            config: Configuration dictionary (default: EVRAG_CONFIG)
        """
        from .config import EVRAG_CONFIG
        self.config = config or EVRAG_CONFIG
        
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load Whisper model (local)."""
        if self._model_loaded:
            return
        
        try:
            import whisper
            
            model_name = self.config["whisper_model"]
            print(f"Loading Whisper model: {model_name}")
            
            self.model = whisper.load_model(model_name)
            self._model_loaded = True
            
        except ImportError:
            print("Warning: Whisper not installed. Falling back to Whisper API.")
            self.config["transcriber"] = "whisper_api"
            self._model_loaded = True
    
    def transcribe_audio(
        self,
        audio_path: Path | str,
        use_cache: bool = True,
    ) -> TranscriptionResult:
        """
        Transcribe audio from video file.
        
        Args:
            audio_path: Path to audio/video file
            use_cache: If True, use cached transcription if available
            
        Returns:
            TranscriptionResult with transcribed text and metadata
        """
        audio_path = Path(audio_path)
        
        # Check cache
        cache_dir = Path(self.config["transcripts_dir"])
        cache_file = cache_dir / f"{audio_path.stem}_transcript.json"
        
        if use_cache and cache_file.exists():
            print(f"Using cached transcription for {audio_path.name}")
            return TranscriptionResult.load(cache_file)
        
        # Transcribe
        print(f"Transcribing audio from {audio_path.name}...")
        
        if self.config["transcriber"] == "whisper_local":
            result = self._transcribe_local(audio_path)
        else:
            result = self._transcribe_api(audio_path)
        
        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        result.save(cache_file)
        
        print(f"  Transcribed {len(result.text)} characters")
        print(f"  Segments: {len(result.segments)}")
        
        return result
    
    def _transcribe_local(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe using local Whisper model."""
        self._load_model()
        
        # Whisper options
        options = {
            "task": "transcribe",
            "language": self.config["language"],
        }
        
        # Transcribe
        result = self.model.transcribe(str(audio_path), **options)
        
        return TranscriptionResult(
            text=result["text"],
            segments=result.get("segments", []),
            language=result.get("language", ""),
            duration_sec=result.get("duration_sec", 0),
        )
    
    def _transcribe_api(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe using Whisper API (fallback)."""
        from openai import OpenAI
        
        client = OpenAI()
        
        print("Using Whisper API (slower, costs money)...")
        
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language=self.config["language"],
                response_format="verbose_json",
            )
        
        return TranscriptionResult(
            text=transcript.text,
            segments=[
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in transcript.segments
            ],
            language=transcript.language,
            duration_sec=transcript.duration,
        )
    
    def extract_audio_from_video(
        self,
        video_path: Path | str,
        output_dir: Path | None = None,
    ) -> Path:
        """
        Extract audio track from video file.
        
        Uses ffmpeg to extract audio as MP3.
        
        Args:
            video_path: Path to video file
            output_dir: Directory for audio output (default: same as video)
            
        Returns:
            Path to extracted audio file
        """
        import subprocess
        
        video_path = Path(video_path)
        output_dir = output_dir or video_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_path = output_dir / f"{video_path.stem}.mp3"
        
        # Check if already extracted
        if audio_path.exists():
            print(f"Audio already extracted: {audio_path.name}")
            return audio_path
        
        print(f"Extracting audio from {video_path.name}...")
        
        # ffmpeg command
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vn",  # No video
            "-acodec", "libmp3lame",
            "-ab", "128k",
            "-y",  # Overwrite
            str(audio_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr}")
        
        print(f"  Audio extracted: {audio_path.name}")
        
        return audio_path
    
    def process_video_audio(
        self,
        video_path: Path | str,
        use_cache: bool = True,
    ) -> TranscriptionResult:
        """
        Complete audio processing pipeline for video.
        
        1. Extract audio from video
        2. Transcribe audio
        
        Args:
            video_path: Path to video file
            use_cache: If True, use cached results
            
        Returns:
            TranscriptionResult
        """
        video_path = Path(video_path)
        
        # Extract audio
        audio_path = self.extract_audio_from_video(video_path)
        
        # Transcribe
        return self.transcribe_audio(audio_path, use_cache=use_cache)
