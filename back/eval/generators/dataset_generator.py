"""
Dataset Generator - MiRAGE-style multi-agent QA pair generation

Genera datasets de evaluación para RAG usando un enfoque multi-agente:
1. Generator Agent: Crea QA pairs basados en el contenido del documento
2. Verifier Agent: Verificación adversarial para detectar alucinaciones

Soporte para 3 tipos de preguntas:
- Factual: Información explícita en el documento
- Multi-hop: Requiere sintetizar información de múltiples secciones
- Synthesis: Comprensión global y conclusiones
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import (
    EVAL_CONFIG,
    QuestionType,
    get_question_type_distribution,
)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QAPair:
    """
    Represents a single Question-Answer pair for evaluation.
    
    Attributes:
        question: The evaluation question
        answer: Ground truth answer
        question_type: Type of question (factual, multi_hop, synthesis)
        context: Source context used to generate the QA pair
        page_numbers: Page numbers where the answer can be found
        requires_multimodal: Whether this QA requires images/diagrams
    """
    question: str
    answer: str
    question_type: QuestionType
    context: str = ""
    page_numbers: list[int] = field(default_factory=list)
    requires_multimodal: bool = False
    verified: bool = False
    verification_notes: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "context": self.context,
            "page_numbers": self.page_numbers,
            "requires_multimodal": self.requires_multimodal,
            "verified": self.verified,
            "verification_notes": self.verification_notes,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAPair":
        """Create QAPair from dictionary."""
        return cls(
            question=data["question"],
            answer=data["answer"],
            question_type=data["question_type"],
            context=data.get("context", ""),
            page_numbers=data.get("page_numbers", []),
            requires_multimodal=data.get("requires_multimodal", False),
            verified=data.get("verified", False),
            verification_notes=data.get("verification_notes", ""),
        )


@dataclass
class DocumentDataset:
    """
    Represents a complete evaluation dataset for a single document.
    
    Attributes:
        document_path: Path to the source document
        document_hash: SHA256 hash of the document (for change detection)
        qa_pairs: List of QA pairs for this document
        generated_at: Timestamp when dataset was generated
        generation_model: Model used for generation
    """
    document_path: str
    document_hash: str
    qa_pairs: list[QAPair] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_model: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "document_path": self.document_path,
            "document_hash": self.document_hash,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
            "generated_at": self.generated_at,
            "generation_model": self.generation_model,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentDataset":
        """Create DocumentDataset from dictionary."""
        return cls(
            document_path=data["document_path"],
            document_hash=data["document_hash"],
            qa_pairs=[QAPair.from_dict(qa) for qa in data["qa_pairs"]],
            generated_at=data.get("generated_at", ""),
            generation_model=data.get("generation_model", ""),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "DocumentDataset":
        """Create DocumentDataset from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @property
    def total_qa_pairs(self) -> int:
        """Returns total number of QA pairs."""
        return len(self.qa_pairs)
    
    @property
    def verified_count(self) -> int:
        """Returns number of verified QA pairs."""
        return sum(1 for qa in self.qa_pairs if qa.verified)


# =============================================================================
# PROMPTS
# =============================================================================

GENERATOR_PROMPT_TEMPLATE = """You are an expert at creating evaluation questions for RAG systems.

Your task is to generate {num_questions} questions of type "{question_type}" based on the provided document context.

**Question Type Guidelines:**

- **factual**: Questions about explicit facts, definitions, or specific information stated in the text.
  Example: "What are the three components of the ASR format?"

- **multi_hop**: Questions that require synthesizing information from multiple sections or pages.
  Example: "How does the choice of architectural style affect the tactics recommended for latency optimization?"

- **synthesis**: Questions that test overall understanding and ability to draw conclusions.
  Example: "Based on the document, what is the most appropriate architectural approach for a high-availability system?"

**Requirements:**
1. Each question must be answerable ONLY using the provided context
2. Include the exact page numbers where the answer can be found
3. Provide a complete ground truth answer with citations
4. Questions should be in Spanish (for Spanish documents) or English (for English documents)
5. Avoid questions that could be answered with general knowledge

**Document Context:**
{context}

**Output Format:**
Generate a JSON array with this exact structure:
[
  {{
    "question": "The question text",
    "answer": "Complete ground truth answer",
    "page_numbers": [1, 2],
    "confidence": 0.95
  }}
]

Generate exactly {num_questions} questions of type "{question_type}".
"""

VERIFIER_PROMPT_TEMPLATE = """You are an adversarial verifier for RAG evaluation datasets.

Your task is to critically evaluate whether a QA pair is valid and properly grounded in the source document.

**Verification Criteria:**

1. **Answerability**: Can the question be answered using ONLY the provided context?
2. **Correctness**: Is the ground truth answer accurate according to the context?
3. **Specificity**: Does the answer include specific details from the context (not generic)?
4. **Page Citations**: Are the page numbers correct and relevant?
5. **No Hallucination**: Does the answer avoid adding information not in the context?

**Document Context:**
{context}

**QA Pair to Verify:**

Question: {question}

Ground Truth Answer: {answer}

Question Type: {question_type}

Claimed Page Numbers: {page_numbers}

**Your Task:**
1. Identify any issues with this QA pair
2. Determine if it should be ACCEPTED or REJECTED
3. Provide specific feedback

**Output Format:**
Respond with a JSON object:
{{
  "status": "ACCEPT" or "REJECT",
  "issues": ["list of specific issues found, or empty if none"],
  "confidence": 0.0-1.0,
  "feedback": "Detailed explanation of your decision"
}}

Be strict - if there's any doubt about the QA pair's validity, REJECT it.
"""


# =============================================================================
# DATASET GENERATOR
# =============================================================================

class DatasetGenerator:
    """
    MiRAGE-style dataset generator using multi-agent approach.
    
    This generator creates evaluation datasets for RAG systems by:
    1. Extracting text from documents
    2. Generating QA pairs using a Generator LLM
    3. Verifying QA pairs using an Adversarial Verifier LLM
    
    Attributes:
        generation_model: LLM model for generating QA pairs
        verification_model: LLM model for verification
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        generation_model: str | None = None,
        verification_model: str | None = None,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the Dataset Generator.

        Args:
            generation_model: Model name for generation (default: from config)
            verification_model: Model name for verification (default: from config)
            config: Configuration dictionary (default: EVAL_CONFIG)
        """
        from back.services.llm_factory import get_llm
        
        self.config = config or EVAL_CONFIG
        
        # Usar Ollama (local) o OpenAI según config
        provider = self.config.get("llm_provider", "ollama")
        model = self.config.get("llm_model", "llama3.1")
        
        # Initialize LLMs
        self.generator_llm = get_llm(
            provider=provider,
            model=model,
            temperature=self.config.get("generation_temperature", 0.7),
        )

        self.verifier_llm = get_llm(
            provider=provider,
            model=model,
            temperature=self.config.get("evaluation_temperature", 0.0),
        )

        # Statistics
        self.stats = {
            "documents_processed": 0,
            "qa_pairs_generated": 0,
            "qa_pairs_verified": 0,
            "qa_pairs_rejected": 0,
        }
    
    def compute_document_hash(self, document_path: Path) -> str:
        """
        Compute SHA256 hash of a document for change detection.
        
        Args:
            document_path: Path to the document
            
        Returns:
            SHA256 hash as hex string
        """
        sha256_hash = hashlib.sha256()
        
        with open(document_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def extract_text_from_pdf(
        self, 
        document_path: Path,
        max_chars: int = 50000,
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Extract text from PDF document with page metadata.
        
        Args:
            document_path: Path to the PDF file
            max_chars: Maximum characters to extract
            
        Returns:
            Tuple of (full_text, page_info_list)
            where page_info_list contains {"page": int, "text": str, "start_char": int}
        """
        import fitz  # PyMuPDF
        
        doc = fitz.open(document_path)
        full_text = ""
        page_info = []
        
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text()
            start_char = len(full_text)
            
            if len(full_text) + len(page_text) > max_chars:
                # Truncate if exceeds max
                remaining = max_chars - len(full_text)
                page_text = page_text[:remaining]
                full_text += page_text
                page_info.append({
                    "page": page_num,
                    "text": page_text,
                    "start_char": start_char,
                })
                break
            
            full_text += page_text
            page_info.append({
                "page": page_num,
                "text": page_text,
                "start_char": start_char,
            })
        
        doc.close()
        
        return full_text, page_info
    
    def _chunk_context(
        self, 
        page_info: list[dict[str, Any]],
        max_chunk_chars: int = 8000,
    ) -> list[dict[str, Any]]:
        """
        Split document into chunks for QA generation.
        
       Chunks are created to fit within LLM context limits while
        maintaining page boundaries for accurate citations.
        
        Args:
            page_info: List of page information from PDF extraction
            max_chunk_chars: Maximum characters per chunk
            
        Returns:
            List of chunks with page metadata
        """
        chunks = []
        current_chunk = ""
        current_pages = []
        current_start = 0
        
        for page_data in page_info:
            page_text = page_data["text"]
            page_num = page_data["page"]
            
            if len(current_chunk) + len(page_text) <= max_chunk_chars:
                # Add to current chunk
                current_chunk += page_text
                if page_num not in current_pages:
                    current_pages.append(page_num)
                if not current_start:
                    current_start = page_data["start_char"]
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "pages": current_pages.copy(),
                        "start_char": current_start,
                    })
                
                # Start new chunk with current page
                current_chunk = page_text
                current_pages = [page_num]
                current_start = page_data["start_char"]
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "pages": current_pages,
                "start_char": current_start,
            })
        
        return chunks
    
    def _generate_qa_for_type(
        self,
        chunk: dict[str, Any],
        question_type: QuestionType,
        num_questions: int,
    ) -> list[QAPair]:
        """
        Generate QA pairs for a specific question type.
        
        Args:
            chunk: Document chunk with text and metadata
            question_type: Type of questions to generate
            num_questions: Number of questions to generate
            
        Returns:
            List of generated QAPair objects
        """
        prompt = GENERATOR_PROMPT_TEMPLATE.format(
            num_questions=num_questions,
            question_type=question_type,
            context=chunk["text"][:15000],  # Limit context for LLM
        )
        
        messages = [
            SystemMessage(content="You are an expert at creating evaluation questions for RAG systems."),
            HumanMessage(content=prompt),
        ]
        
        response = self.generator_llm.invoke(messages)
        response_text = response.content
        
        # Parse JSON from response
        try:
            # Try to extract JSON array from response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if json_match:
                qa_data = json.loads(json_match.group())
            else:
                qa_data = json.loads(response_text)
            
            qa_pairs = []
            for qa in qa_data:
                qa_pair = QAPair(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    question_type=question_type,
                    context=chunk["text"][:2000],  # Store abbreviated context
                    page_numbers=qa.get("page_numbers", chunk["pages"][:3]),
                    requires_multimodal=False,  # TODO: Detect diagrams/images
                )
                qa_pairs.append(qa_pair)
            
            return qa_pairs
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing QA generation response: {e}")
            return []
    
    def _verify_qa_pair(
        self,
        qa_pair: QAPair,
        context: str,
    ) -> tuple[bool, str]:
        """
        Verify a QA pair using adversarial verification.
        
        Args:
            qa_pair: QA pair to verify
            context: Source document context
            
        Returns:
            Tuple of (is_valid, verification_notes)
        """
        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            context=context[:15000],
            question=qa_pair.question,
            answer=qa_pair.answer,
            question_type=qa_pair.question_type,
            page_numbers=qa_pair.page_numbers,
        )
        
        messages = [
            SystemMessage(content="You are an adversarial verifier for RAG evaluation datasets."),
            HumanMessage(content=prompt),
        ]
        
        response = self.verifier_llm.invoke(messages)
        response_text = response.content
        
        # Parse JSON response
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                verification = json.loads(json_match.group())
            else:
                verification = json.loads(response_text)
            
            is_accepted = verification.get("status", "REJECT") == "ACCEPT"
            issues = verification.get("issues", [])
            feedback = verification.get("feedback", "")
            
            notes = f"Issues: {', '.join(issues) if issues else 'None'}. Feedback: {feedback}"
            
            return is_accepted, notes
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing verification response: {e}")
            return False, f"Parse error: {e}"
    
    def generate_dataset(
        self,
        document_path: Path | str,
        force_regenerate: bool = False,
        cache_dir: Path | None = None,
    ) -> DocumentDataset:
        """
        Generate evaluation dataset for a document.
        
        Args:
            document_path: Path to the PDF document
            force_regenerate: If True, regenerate even if cached
            cache_dir: Directory for caching datasets
            
        Returns:
            DocumentDataset with generated QA pairs
        """
        document_path = Path(document_path)
        cache_dir = cache_dir or Path(self.config["datasets_dir"])
        
        # Check cache
        cache_file = cache_dir / f"{document_path.stem}_eval.json"
        if cache_file.exists() and not force_regenerate:
            cached_data = DocumentDataset.from_json(cache_file.read_text())
            
            # Check if document has changed
            current_hash = self.compute_document_hash(document_path)
            if cached_data.document_hash == current_hash:
                print(f"Using cached dataset for {document_path.name}")
                return cached_data
        
        print(f"Generating dataset for {document_path.name}...")
        
        # Compute document hash
        doc_hash = self.compute_document_hash(document_path)
        
        # Extract text
        full_text, page_info = self.extract_text_from_pdf(document_path)
        
        # Create chunks
        chunks = self._chunk_context(page_info)
        
        # Get question type distribution
        qa_distribution = get_question_type_distribution()
        
        # Generate QA pairs
        dataset = DocumentDataset(
            document_path=str(document_path),
            document_hash=doc_hash,
            generation_model=self.generation_model_name,
        )
        
        for chunk_idx, chunk in enumerate(chunks):
            print(f"  Processing chunk {chunk_idx + 1}/{len(chunks)}...")
            
            for q_type, num_qa in qa_distribution.items():
                # Distribute questions across chunks
                qa_per_chunk = max(1, num_qa // len(chunks))
                
                qa_pairs = self._generate_qa_for_type(
                    chunk=chunk,
                    question_type=q_type,
                    num_questions=qa_per_chunk,
                )
                
                # Verify each QA pair
                for qa_pair in qa_pairs:
                    is_valid, notes = self._verify_qa_pair(
                        qa_pair=qa_pair,
                        context=chunk["text"],
                    )
                    
                    if is_valid:
                        qa_pair.verified = True
                        qa_pair.verification_notes = notes
                        dataset.qa_pairs.append(qa_pair)
                        self.stats["qa_pairs_verified"] += 1
                    else:
                        self.stats["qa_pairs_rejected"] += 1
                
                self.stats["qa_pairs_generated"] += len(qa_pairs)
        
        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(dataset.to_json(), encoding="utf-8")
        
        self.stats["documents_processed"] += 1
        
        print(f"  Generated {dataset.total_qa_pairs} QA pairs ({dataset.verified_count} verified)")
        
        return dataset
    
    def generate_datasets_for_all_docs(
        self,
        docs_dir: Path | None = None,
        force_regenerate: bool = False,
    ) -> list[DocumentDataset]:
        """
        Generate evaluation datasets for all PDF documents in a directory.
        
        Args:
            docs_dir: Directory containing PDF documents
            force_regenerate: If True, regenerate all datasets
            
        Returns:
            List of DocumentDataset objects
        """
        docs_dir = docs_dir or Path(self.config["docs_dir"])
        
        pdf_files = list(docs_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {docs_dir}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        datasets = []
        for pdf_file in pdf_files:
            dataset = self.generate_dataset(
                document_path=pdf_file,
                force_regenerate=force_regenerate,
            )
            datasets.append(dataset)
        
        return datasets
    
    def get_stats(self) -> dict[str, Any]:
        """
        Returns generation statistics.
        
        Returns:
            Dictionary with generation statistics
        """
        return self.stats.copy()
