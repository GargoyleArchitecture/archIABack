#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Verification Script for EVRAG

Verifica el video procesado generando transcripción real y QA pairs reales.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / "back" / "src" / ".env")
load_dotenv(project_root / "back" / ".env")

from back.evrag.transcriber import AudioTranscriber
from back.evrag.config import EVRAG_CONFIG


VIDEO_PATH = project_root / "back" / "videos" / "raw" / "YTDown.com_YouTube_Clase-abierta-Maestria-en-Ingenieria-de-_Media_M8XT7F8DZdg_001_1080p.mp4"
OUTPUT_DIR = project_root / "back" / "videos" / "processed"


def transcribe_video():
    """Transcribe video audio using Whisper."""
    print("=" * 70)
    print("VIDEO TRANSCRIPTION - EVRAG")
    print("=" * 70)
    print(f"\nVideo: {VIDEO_PATH.name}")
    print(f"Duration: ~70 minutes")
    print()
    
    transcriber = AudioTranscriber(config=EVRAG_CONFIG)
    
    print("Step 1: Transcribing audio with Whisper...")
    print("(This may take 5-10 minutes for a 70-minute video)\n")
    
    try:
        result = transcriber.transcribe_audio(str(VIDEO_PATH))
        
        print(f"\n✓ Transcription complete!")
        print(f"  - Language: {result.language}")
        print(f"  - Duration: {result.duration_sec:.2f}s")
        print(f"  - Text length: {len(result.text)} characters")
        print(f"  - Segments: {len(result.segments)}")
        
        # Save transcription
        transcript_path = OUTPUT_DIR / f"{VIDEO_PATH.stem}_transcript.json"
        result.save(transcript_path)
        print(f"\n✓ Saved to: {transcript_path}")
        
        # Also save plain text version
        txt_path = OUTPUT_DIR / f"{VIDEO_PATH.stem}_transcript.txt"
        txt_path.write_text(result.text, encoding="utf-8")
        print(f"✓ Plain text: {txt_path}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        return None


def generate_qa_pairs(transcript_text: str, num_pairs: int = 10):
    """Generate real QA pairs from transcript using LLM."""
    from openai import OpenAI
    
    print("\n" + "=" * 70)
    print("GENERATING QA PAIRS FROM TRANSCRIPT")
    print("=" * 70)
    
    client = OpenAI()
    
    # Truncate transcript if too long (keep first 15k chars for context)
    max_context = 15000
    context = transcript_text[:max_context] if len(transcript_text) > max_context else transcript_text
    
    print(f"\nUsing {len(context)} characters of transcript as context...")
    
    prompt = f"""Based on the following transcript from a 70-minute educational video about software architecture,
generate 10 high-quality question-answer pairs for evaluation purposes.

The questions should be:
- 4 factual questions (direct information retrieval)
- 4 multi-hop questions (require connecting multiple pieces of information)
- 2 synthesis questions (require understanding the overall content)

Format your response as a JSON array with this exact structure:
[
  {{
    "question": "Your question here",
    "answer": "Detailed ground truth answer",
    "type": "factual|multi_hop|synthesis"
  }}
]

TRANSCRIPT:
{context}

Generate the 10 QA pairs now:"""

    print("\nCalling GPT-4o-mini to generate QA pairs...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at generating evaluation questions from educational content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        
        qa_text = response.choices[0].message.content
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', qa_text, re.DOTALL)
        if json_match:
            qa_pairs = json.loads(json_match.group())
            print(f"\n✓ Generated {len(qa_pairs)} QA pairs")
            return qa_pairs
        else:
            print("❌ Could not parse JSON from response")
            print(qa_text)
            return []
            
    except Exception as e:
        print(f"\n❌ Error generating QA pairs: {e}")
        return []


def create_verification_dataset(qa_pairs: list, transcript_text: str):
    """Create verification dataset with real content."""
    print("\n" + "=" * 70)
    print("CREATING VERIFICATION DATASET")
    print("=" * 70)
    
    dataset = {
        "video_path": str(VIDEO_PATH),
        "video_hash": "6bb135f0b47e88c01b245470ebaeadc166be797b05d79c02a4bb8492e640ad02",
        "generated_at": datetime.now().isoformat(),
        "transcript_length": len(transcript_text),
        "transcript_sample": transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text,
        "qa_pairs": [
            {
                "question": qa["question"],
                "answer": qa["answer"],
                "question_type": qa["type"],
                "context": "Video transcript: " + VIDEO_PATH.name,
                "page_numbers": [],
                "requires_multimodal": False,  # Text-only from transcript
                "verified": True,
                "verification_notes": f"Generated from real transcript at {datetime.now().isoformat()}"
            }
            for qa in qa_pairs
        ],
        "scenes_info": json.loads(
            (OUTPUT_DIR / f"{VIDEO_PATH.stem}_info.json").read_text()
        ).get("scenes", [])
    }
    
    # Save dataset
    dataset_path = OUTPUT_DIR / f"{VIDEO_PATH.stem}_verified_dataset.json"
    dataset_path.write_text(json.dumps(dataset, indent=2, ensure_ascii=False))
    print(f"\n✓ Dataset saved to: {dataset_path}")
    
    return dataset


def main():
    """Main verification workflow."""
    print("\n" + "=" * 70)
    print("EVRAG VIDEO VERIFICATION")
    print("=" * 70)
    print(f"\nTarget: {VIDEO_PATH.name}")
    print(f"Previous status: Mock evaluation (no real transcription)")
    print(f"Goal: Generate real transcription and QA pairs\n")
    
    # Step 1: Transcribe
    transcript_result = transcribe_video()
    
    if not transcript_result:
        print("\n❌ Transcription failed. Cannot proceed.")
        return
    
    # Step 2: Generate QA pairs
    qa_pairs = generate_qa_pairs(transcript_result.text)
    
    if not qa_pairs:
        print("\n❌ QA generation failed.")
        return
    
    # Step 3: Create verification dataset
    dataset = create_verification_dataset(qa_pairs, transcript_result.text)
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print(f"✓ Video transcribed: {len(transcript_result.text)} characters")
    print(f"✓ QA pairs generated: {len(qa_pairs)}")
    print(f"✓ Dataset created with real content")
    print(f"\nNext step: Run evaluation with real RAG system")
    print("=" * 70)


if __name__ == "__main__":
    main()
