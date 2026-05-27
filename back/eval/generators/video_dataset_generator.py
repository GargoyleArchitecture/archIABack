"""
Video Dataset Generator for RAG Evaluation

Genera datasets de evaluación (QA pairs) a partir de videos procesados.
Utiliza la transcripción y frames del video para crear preguntas y respuestas.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .dataset_generator import DocumentDataset, QAPair
from .video_processor import VideoInfo, VideoProcessor


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VideoDataset:
    """
    Dataset de evaluación generado desde un video.

    Attributes:
        video_path: Path to original video
        video_hash: SHA-256 hash for cache validation
        generated_at: Timestamp of generation
        qa_pairs: List of QA pairs
        transcript: Video transcript
        scenes_info: Information about detected scenes
    """
    video_path: str
    video_hash: str
    generated_at: str
    qa_pairs: list[QAPair] = field(default_factory=list)
    transcript: str = ""
    scenes_info: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_path": self.video_path,
            "video_hash": self.video_hash,
            "generated_at": self.generated_at,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
            "transcript": self.transcript,
            "scenes_info": self.scenes_info,
        }

    def to_document_dataset(self) -> DocumentDataset:
        """Convert to DocumentDataset format for compatibility."""
        return DocumentDataset(
            document_path=self.video_path,
            document_hash=self.video_hash,
            qa_pairs=self.qa_pairs,
            generated_at=self.generated_at,
            generation_model="video_generator",
        )


# =============================================================================
# VIDEO DATASET GENERATOR
# =============================================================================

class VideoDatasetGenerator:
    """
    Generates evaluation datasets from processed videos.

    Creates QA pairs based on:
    1. Video transcript (audio content)
    2. Scene information (visual content)
    3. Combined audio-visual questions

    Attributes:
        qa_pairs_per_video: Number of QA pairs to generate per video
        question_types: Distribution of question types
        generation_model: LLM model for generation
    """

    def __init__(
        self,
        qa_pairs_per_video: int = 10,
        question_types: dict[str, int] | None = None,
        generation_model: str = "gpt-4o-mini",
        use_mock: bool = False,
    ):
        """
        Initialize the video dataset generator.

        Args:
            qa_pairs_per_video: Number of QA pairs to generate
            question_types: Distribution of question types
            generation_model: LLM model for generation
            use_mock: Use mock QA generation (no LLM required)
        """
        self.qa_pairs_per_video = qa_pairs_per_video
        self.question_types = question_types or {
            "factual": 4,       # 40% - Basic retrieval
            "multi_hop": 4,     # 40% - Multi-hop reasoning
            "synthesis": 2,     # 20% - Global understanding
        }
        self.generation_model = generation_model
        self.use_mock = use_mock

        # Initialize LLM (lazy loading)
        self._llm = None

    def _get_llm(self):
        """Lazy load LLM."""
        if self._llm is None:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self.generation_model,
                temperature=0.7,
            )
        return self._llm

    def _generate_mock_qa(
        self,
        transcript: str,
        video_path: str,
        num_questions: int,
        question_type: str,
    ) -> list[QAPair]:
        """
        Generate mock QA pairs for testing without LLM.

        Args:
            transcript: Video transcript text
            video_path: Path to video file
            num_questions: Number of questions to generate
            question_type: Type of questions

        Returns:
            List of QAPair objects
        """
        if not transcript or len(transcript.strip()) < 100:
            return []

        qa_pairs = []
        for i in range(num_questions):
            qa_pairs.append(QAPair(
                question=f"Pregunta {question_type} #{i+1} sobre el video",
                answer=f"Respuesta mock para la pregunta #{i+1}. El video trata sobre arquitectura de software.",
                context=transcript[:500],
                metadata={
                    "type": question_type,
                    "source": "transcript_mock",
                    "timestamp_sec": i * 60,
                },
            ))

        return qa_pairs

    def _generate_transcript_qa(
        self,
        transcript: str,
        video_path: str,
        num_questions: int,
        question_type: str,
    ) -> list[QAPair]:
        """
        Generate QA pairs from video transcript.

        Args:
            transcript: Video transcript text
            video_path: Path to video file
            num_questions: Number of questions to generate
            question_type: Type of questions (factual, multi_hop, synthesis)

        Returns:
            List of QAPair objects
        """
        if not transcript or len(transcript.strip()) < 100:
            return []

        llm = self._get_llm()

        # Create prompt for QA generation
        prompt = f"""You are an expert at creating evaluation questions for educational videos.

Based on the following video transcript, generate {num_questions} {question_type} questions.

**Transcript:**
{transcript[:8000]}  # Limit to avoid token limits

**Instructions:**
- Questions should be in Spanish (the transcript is in Spanish)
- Each question should be answerable using ONLY the transcript
- Provide clear, concise answers
- Include timestamps when relevant

**Question Types:**
- factual: Direct fact retrieval (e.g., "¿Qué es...?", "¿Cuándo...?")
- multi_hop: Requires combining multiple pieces of information
- synthesis: Requires understanding the overall message

**Output Format (JSON):**
```json
[
    {{
        "question": "¿Pregunta en español?",
        "answer": "Respuesta clara y concisa.",
        "context": "Fragmento relevante del transcript...",
        "timestamp_sec": 123.45
    }}
]
```

Generate ONLY the JSON array, no additional text."""

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Extract JSON from response
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            qa_data = json.loads(content)

            qa_pairs = []
            for item in qa_data:
                qa_pairs.append(QAPair(
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    context=item.get("context", transcript[:500]),
                    metadata={
                        "type": question_type,
                        "source": "transcript",
                        "timestamp_sec": item.get("timestamp_sec", 0),
                    },
                ))

            return qa_pairs

        except Exception as e:
            print(f"      Error generating transcript QA: {e}")
            return []

    def _generate_visual_qa(
        self,
        scenes_info: list[dict],
        video_path: str,
        num_questions: int,
    ) -> list[QAPair]:
        """
        Generate QA pairs from visual scene information.

        Args:
            scenes_info: Information about detected scenes
            video_path: Path to video file
            num_questions: Number of questions to generate

        Returns:
            List of QAPair objects
        """
        if not scenes_info or len(scenes_info) < 2:
            return []

        llm = self._get_llm()

        # Create prompt for visual QA generation
        prompt = f"""You are an expert at creating evaluation questions for educational videos.

Based on the following scene information from a video, generate {num_questions} questions about the visual structure.

**Scene Information:**
{json.dumps(scenes_info, indent=2)}

**Instructions:**
- Questions should be in Spanish
- Focus on visual structure, scene changes, and video organization
- Provide clear answers based on the scene data

**Output Format (JSON):**
```json
[
    {{
        "question": "¿Pregunta sobre estructura visual?",
        "answer": "Respuesta basada en información de escenas.",
        "context": "Información de escenas relevante...",
        "scene_index": 1
    }}
]
```

Generate ONLY the JSON array, no additional text."""

        try:
            response = llm.invoke(prompt)
            content = response.content.strip()

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            qa_data = json.loads(content)

            qa_pairs = []
            for item in qa_data:
                qa_pairs.append(QAPair(
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    context=item.get("context", str(scenes_info)),
                    metadata={
                        "type": "visual",
                        "source": "scenes",
                        "scene_index": item.get("scene_index", 0),
                    },
                ))

            return qa_pairs

        except Exception as e:
            print(f"      Error generating visual QA: {e}")
            return []

    def generate_dataset(
        self,
        video_info: VideoInfo,
        force_regenerate: bool = False,
    ) -> VideoDataset:
        """
        Generate evaluation dataset for a video.

        Args:
            video_info: VideoInfo from VideoProcessor
            force_regenerate: Force regeneration even if cached

        Returns:
            VideoDataset with QA pairs
        """
        video_path = Path(video_info.video_path)
        video_prefix = video_path.stem
        dataset_path = Path(f"back/eval/datasets/{video_prefix}_video_dataset.json")

        # Check cache
        if dataset_path.exists() and not force_regenerate:
            print(f"    Loading cached video dataset...")
            data = json.loads(dataset_path.read_text(encoding="utf-8"))
            return VideoDataset(**data)

        print(f"    Generating QA pairs for video...")

        qa_pairs = []

        if self.use_mock:
            # Generate mock QA pairs for testing
            print(f"      Generating mock QA pairs...")
            
            # Generate questions from transcript if available
            transcript = video_info.transcript
            if transcript and len(transcript.strip()) > 100:
                for q_type, count in self.question_types.items():
                    mock_qas = self._generate_mock_qa(
                        transcript=transcript,
                        video_path=video_info.video_path,
                        num_questions=count,
                        question_type=q_type,
                    )
                    qa_pairs.extend(mock_qas)
            else:
                # No transcript, generate questions from video metadata
                print(f"      No transcript available, generating from video info...")
                duration_min = video_info.video_duration_sec / 60
                for q_type, count in self.question_types.items():
                    for i in range(count):
                        qa_pairs.append(QAPair(
                            question=f"Pregunta {q_type} #{i+1} sobre el video de {duration_min:.0f} minutos",
                            answer=f"Respuesta mock #{i+1}. El video trata sobre arquitectura de software y dura {duration_min:.0f} minutos.",
                            question_type=q_type,
                            context=f"Video: {video_info.video_path}, Duración: {duration_min:.0f} minutos",
                            requires_multimodal=True,  # Video is multimodal
                        ))
            
            # Add visual questions from scenes
            if video_info.scenes:
                print(f"      Generating {len(video_info.scenes)} visual questions from scenes...")
                for i, scene in enumerate(video_info.scenes):
                    scene_data = scene if isinstance(scene, dict) else scene.to_dict()
                    qa_pairs.append(QAPair(
                        question=f"¿Qué ocurre en la escena {i+1} del video (min {scene_data['start_time_sec']/60:.1f} - {scene_data['end_time_sec']/60:.1f})?",
                        answer=f"La escena {i+1} ocurre entre los frames {scene_data['start_frame']} y {scene_data['end_frame']}, durando {scene_data['end_time_sec']-scene_data['start_time_sec']:.1f} segundos.",
                        question_type="factual",
                        context=f"Scene {i+1}: {json.dumps(scene_data)}",
                        requires_multimodal=True,
                    ))
        else:
            # Generate questions from transcript using LLM
            transcript = video_info.transcript
            if transcript:
                for q_type, count in self.question_types.items():
                    print(f"      Generating {count} {q_type} questions from transcript...")
                    transcript_qas = self._generate_transcript_qa(
                        transcript=transcript,
                        video_path=video_info.video_path,
                        num_questions=count,
                        question_type=q_type,
                    )
                    qa_pairs.extend(transcript_qas)

            # Generate questions from visual structure
            if video_info.scenes:
                print(f"      Generating visual structure questions...")
                # Handle both Scene objects and dicts (from cache)
                scenes_data = []
                for s in video_info.scenes:
                    if hasattr(s, 'to_dict'):
                        scenes_data.append(s.to_dict())
                    else:
                        scenes_data.append(s)

                visual_qas = self._generate_visual_qa(
                    scenes_info=scenes_data,
                    video_path=video_info.video_path,
                    num_questions=2,  # Fixed number for visual questions
                )
                qa_pairs.extend(visual_qas)

        # Limit to target number
        if len(qa_pairs) > self.qa_pairs_per_video:
            qa_pairs = qa_pairs[:self.qa_pairs_per_video]

        # Create dataset
        # Handle both Scene objects and dicts (from cache)
        scenes_data = []
        for s in video_info.scenes:
            if hasattr(s, 'to_dict'):
                scenes_data.append(s.to_dict())
            else:
                scenes_data.append(s)

        dataset = VideoDataset(
            video_path=video_info.video_path,
            video_hash=video_info.video_hash,
            generated_at=datetime.now().isoformat(),
            qa_pairs=qa_pairs,
            transcript=video_info.transcript,
            scenes_info=scenes_data,
        )

        # Save dataset
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset_path.write_text(json.dumps(dataset.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"    Saved dataset to {dataset_path}")

        return dataset

    def generate_datasets_for_all_videos(
        self,
        videos_dir: Path,
        force_regenerate: bool = False,
    ) -> list[VideoDataset]:
        """
        Generate datasets for all videos in a directory.

        Args:
            videos_dir: Directory containing video files (or processed info)
            force_regenerate: Force regeneration

        Returns:
            List of VideoDataset objects
        """
        # First, process videos to get VideoInfo
        processor = VideoProcessor()
        raw_videos_dir = videos_dir / "raw"

        if not raw_videos_dir.exists():
            raw_videos_dir = videos_dir

        video_infos = processor.process_all_videos(raw_videos_dir, force_reprocess=force_regenerate)

        if not video_infos:
            return []

        # Generate datasets for each video
        datasets = []
        for video_info in video_infos:
            print(f"\n  Generating dataset for {Path(video_info.video_path).name}...")
            dataset = self.generate_dataset(video_info, force_regenerate)
            datasets.append(dataset)

        return datasets


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_video_datasets(
    videos_dir: str = "back/videos",
    qa_pairs_per_video: int = 10,
    force_regenerate: bool = False,
) -> list[VideoDataset]:
    """
    Generate evaluation datasets for all videos.

    Args:
        videos_dir: Directory containing videos
        qa_pairs_per_video: Number of QA pairs per video
        force_regenerate: Force regeneration

    Returns:
        List of VideoDataset objects
    """
    generator = VideoDatasetGenerator(qa_pairs_per_video=qa_pairs_per_video)
    return generator.generate_datasets_for_all_videos(Path(videos_dir), force_regenerate)
