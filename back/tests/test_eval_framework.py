"""
Tests for RAG Evaluation Framework

Pruebas para el framework de evaluación RAG.
Nota: Los imports son desde 'eval' no 'back.eval' porque pytest usa pythonpath = ["back"]
"""

import json
import pytest
from pathlib import Path

# Imports desde eval (pythonpath apunta a back/)
from eval.config import EVAL_CONFIG, get_enabled_metrics, get_total_qa_pairs
from eval.generators.dataset_generator import DatasetGenerator, DocumentDataset, QAPair
from eval.metrics.hybrid_evaluator import HybridEvaluator, DocumentEvaluationResult, MetricResult, QAEvaluationResult
from eval.pipeline import RAGEvaluationPipeline, EvaluationReport


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_qa_pair():
    """Create a sample QA pair for testing."""
    return QAPair(
        question="What is the purpose of ASR in software architecture?",
        answer="ASR (Architecture Significant Requirement) is used to capture architecturally significant requirements in a structured format.",
        question_type="factual",
        context="This is sample context about ASR...",
        page_numbers=[1, 2],
    )


@pytest.fixture
def sample_dataset():
    """Create a sample document dataset."""
    return DocumentDataset(
        document_path="test_document.pdf",
        document_hash="abc123",
        qa_pairs=[
            QAPair(
                question="What is ASR?",
                answer="Architecture Significant Requirement",
                question_type="factual",
            ),
            QAPair(
                question="How does ASR relate to quality attributes?",
                answer="ASR captures quality attributes in a structured format",
                question_type="multi_hop",
            ),
        ],
    )


@pytest.fixture
def mock_rag_func():
    """Mock RAG invoke function."""
    def _mock(question: str, session_id: str) -> dict:
        return {
            "retrieved_context": "Mock context for testing",
            "generated_answer": f"Mock answer to: {question}",
        }
    return _mock


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Tests for configuration."""
    
    def test_qa_pairs_per_doc(self):
        """Test QA pairs per document configuration."""
        assert EVAL_CONFIG["qa_pairs_per_doc"] == 10
    
    def test_question_types_sum(self):
        """Test that question types sum to qa_pairs_per_doc."""
        total = sum(EVAL_CONFIG["question_types"].values())
        assert total == EVAL_CONFIG["qa_pairs_per_doc"]
    
    def test_get_total_qa_pairs(self):
        """Test get_total_qa_pairs helper function."""
        assert get_total_qa_pairs() == EVAL_CONFIG["qa_pairs_per_doc"]
    
    def test_get_enabled_metrics(self):
        """Test get_enabled_metrics returns non-empty list."""
        metrics = get_enabled_metrics()
        assert len(metrics) > 0
        assert isinstance(metrics, list)
    
    def test_is_metric_enabled(self):
        """Test is_metric_enabled for known metrics."""
        from eval.config import is_metric_enabled
        
        # At least some metrics should be enabled
        enabled_count = sum(
            1 for metric in EVAL_CONFIG["metrics"]
            if is_metric_enabled(metric)
        )
        assert enabled_count > 0


# =============================================================================
# DATA CLASS TESTS
# =============================================================================

class TestQAPair:
    """Tests for QAPair dataclass."""
    
    def test_create_qa_pair(self, sample_qa_pair):
        """Test creating a QA pair."""
        assert sample_qa_pair.question_type == "factual"
        assert len(sample_qa_pair.page_numbers) == 2
    
    def test_qa_pair_to_dict(self, sample_qa_pair):
        """Test converting QA pair to dictionary."""
        data = sample_qa_pair.to_dict()
        assert data["question"] == sample_qa_pair.question
        assert data["answer"] == sample_qa_pair.answer
        assert data["question_type"] == sample_qa_pair.question_type
    
    def test_qa_pair_from_dict(self):
        """Test creating QA pair from dictionary."""
        data = {
            "question": "Test question?",
            "answer": "Test answer",
            "question_type": "factual",
            "context": "Test context",
            "page_numbers": [1],
            "verified": True,
        }
        qa_pair = QAPair.from_dict(data)
        assert qa_pair.question == "Test question?"
        assert qa_pair.verified is True


class TestDocumentDataset:
    """Tests for DocumentDataset dataclass."""
    
    def test_create_dataset(self, sample_dataset):
        """Test creating a document dataset."""
        assert sample_dataset.total_qa_pairs == 2
        assert sample_dataset.document_path == "test_document.pdf"
    
    def test_dataset_to_json(self, sample_dataset):
        """Test converting dataset to JSON."""
        json_str = sample_dataset.to_json()
        data = json.loads(json_str)
        assert data["document_path"] == sample_dataset.document_path
        assert len(data["qa_pairs"]) == 2
    
    def test_dataset_from_json(self, sample_dataset):
        """Test loading dataset from JSON."""
        json_str = sample_dataset.to_json()
        loaded = DocumentDataset.from_json(json_str)
        assert loaded.document_path == sample_dataset.document_path
        assert loaded.total_qa_pairs == sample_dataset.total_qa_pairs
    
    def test_verified_count(self):
        """Test verified count property."""
        dataset = DocumentDataset(
            document_path="test.pdf",
            document_hash="xyz",
            qa_pairs=[
                QAPair(question="Q1", answer="A1", question_type="factual", verified=True),
                QAPair(question="Q2", answer="A2", question_type="factual", verified=False),
                QAPair(question="Q3", answer="A3", question_type="factual", verified=True),
            ],
        )
        assert dataset.verified_count == 2


# =============================================================================
# EVALUATOR TESTS
# =============================================================================

class TestMetricResult:
    """Tests for MetricResult dataclass."""
    
    def test_create_metric_result(self):
        """Test creating a metric result."""
        result = MetricResult(
            metric_name="faithfulness",
            score=0.85,
            explanation="Good faithfulness",
            framework="ragas",
        )
        assert result.score == 0.85
        assert result.framework == "ragas"
    
    def test_metric_result_to_dict(self):
        """Test converting metric result to dictionary."""
        result = MetricResult(
            metric_name="test_metric",
            score=0.75,
            explanation="Test explanation",
        )
        data = result.to_dict()
        assert data["metric_name"] == "test_metric"
        assert data["score"] == 0.75


class TestQAEvaluationResult:
    """Tests for QAEvaluationResult dataclass."""
    
    def test_create_qa_evaluation_result(self, sample_qa_pair):
        """Test creating QA evaluation result."""
        result = QAEvaluationResult(
            qa_pair=sample_qa_pair,
            retrieved_context="Test context",
            generated_answer="Test answer",
        )
        assert result.qa_pair == sample_qa_pair
        assert result.overall_score == 0.0  # No metrics yet
    
    def test_add_metric(self, sample_qa_pair):
        """Test adding metrics to evaluation result."""
        result = QAEvaluationResult(qa_pair=sample_qa_pair)
        
        result.add_metric(MetricResult(
            metric_name="faithfulness",
            score=0.8,
            framework="ragas",
        ))
        
        assert "faithfulness" in result.metrics
        assert result.overall_score == 0.8
        
        # Add another metric
        result.add_metric(MetricResult(
            metric_name="answer_relevance",
            score=0.9,
            framework="ragas",
        ))
        
        # Average should be (0.8 + 0.9) / 2 = 0.85
        assert abs(result.overall_score - 0.85) < 0.001


class TestDocumentEvaluationResult:
    """Tests for DocumentEvaluationResult dataclass."""
    
    def test_create_document_evaluation_result(self, sample_dataset):
        """Test creating document evaluation result."""
        result = DocumentEvaluationResult(
            document_path="test.pdf",
            dataset=sample_dataset,
        )
        assert result.document_path == "test.pdf"
        assert len(result.qa_results) == 0
    
    def test_add_qa_result(self, sample_dataset):
        """Test adding QA results."""
        doc_result = DocumentEvaluationResult(
            document_path="test.pdf",
            dataset=sample_dataset,
        )
        
        qa_result = QAEvaluationResult(
            qa_pair=sample_dataset.qa_pairs[0],
            overall_score=0.8,
        )
        
        doc_result.add_qa_result(qa_result)
        
        assert len(doc_result.qa_results) == 1
        assert doc_result.average_overall_score == 0.8


# =============================================================================
# PIPELINE TESTS
# =============================================================================

class TestRAGEvaluationPipeline:
    """Tests for RAG evaluation pipeline."""
    
    def test_create_pipeline(self):
        """Test creating a pipeline."""
        pipeline = RAGEvaluationPipeline()
        assert pipeline.dataset_generator is not None
        assert pipeline.evaluator is not None
    
    def test_create_pipeline_with_rag_func(self, mock_rag_func):
        """Test creating pipeline with RAG invoke function."""
        pipeline = RAGEvaluationPipeline(rag_invoke_func=mock_rag_func)
        assert pipeline.rag_invoke_func is not None
    
    def test_evaluate_layer_mock(self, mock_rag_func):
        """Test evaluating a layer with mock RAG."""
        pipeline = RAGEvaluationPipeline(rag_invoke_func=mock_rag_func)
        
        # Use mock mode for testing
        report = pipeline.evaluate_layer(
            layer_name="test_layer",
            docs_dir=Path("back/docs"),
            use_mock_rag=True,
            mock_accuracy=0.8,
        )
        
        assert report.layer_name == "test_layer"
        assert report.report_id.startswith("test_layer_")
    
    def test_get_stats(self):
        """Test getting pipeline statistics."""
        pipeline = RAGEvaluationPipeline()
        stats = pipeline.get_stats()
        
        assert "datasets_generated" in stats
        assert "documents_evaluated" in stats
        assert "qa_pairs_evaluated" in stats


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""
    
    def test_create_report(self):
        """Test creating an evaluation report."""
        report = EvaluationReport(
            report_id="test_001",
            evaluated_at="2026-03-22T00:00:00",
            layer_name="layer1_books",
        )
        
        assert report.report_id == "test_001"
        assert report.layer_name == "layer1_books"
    
    def test_report_to_json(self):
        """Test converting report to JSON."""
        report = EvaluationReport(
            report_id="test_001",
            evaluated_at="2026-03-22T00:00:00",
            layer_name="layer1_books",
            aggregate_metrics={"overall": 0.85, "faithfulness": 0.9},
        )
        
        json_str = report.to_json()
        data = json.loads(json_str)
        
        assert data["report_id"] == "test_001"
        assert data["aggregate_metrics"]["overall"] == 0.85
    
    def test_report_to_markdown(self):
        """Test converting report to Markdown."""
        report = EvaluationReport(
            report_id="test_001",
            evaluated_at="2026-03-22T00:00:00",
            layer_name="layer1_books",
            aggregate_metrics={"overall": 0.85},
        )
        
        md = report.to_markdown()
        
        assert "# RAG Evaluation Report" in md
        assert "test_001" in md
        assert "layer1_books" in md
        assert "0.8500" in md


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the evaluation framework."""
    
    def test_full_evaluation_flow(self, sample_dataset, mock_rag_func):
        """Test complete evaluation flow."""
        # Create evaluator
        evaluator = HybridEvaluator()
        
        # Create mock RAG results
        rag_results = [
            {
                "question": qa.question,
                "retrieved_context": "Mock context",
                "generated_answer": f"Answer to: {qa.question}",
            }
            for qa in sample_dataset.qa_pairs
        ]
        
        # Evaluate
        doc_result = evaluator.evaluate_dataset(
            dataset=sample_dataset,
            rag_results=rag_results,
            metrics=["faithfulness", "answer_relevance"],
        )
        
        # Verify
        assert len(doc_result.qa_results) == len(sample_dataset.qa_pairs)
        assert "faithfulness" in doc_result.aggregate_metrics
        assert "answer_relevance" in doc_result.aggregate_metrics
