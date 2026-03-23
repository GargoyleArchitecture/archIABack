"""
RAG Evaluation Pipeline

Pipeline automatizado para evaluación de sistemas RAG:
1. Genera datasets de evaluación (si no existen)
2. Ejecuta evaluación usando el RAG system
3. Genera reportes con métricas agregadas
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import EVAL_CONFIG
from .generators import DatasetGenerator, DocumentDataset
from .metrics import HybridEvaluator, DocumentEvaluationResult


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EvaluationReport:
    """
    Complete evaluation report for a RAG system.
    
    Attributes:
        report_id: Unique identifier for this report
        evaluated_at: Timestamp of evaluation
        layer_name: Name of the evaluation layer (e.g., "layer1_books")
        document_results: Results for each document
        aggregate_metrics: Aggregated metrics across all documents
        comparison_with_previous: Comparison with previous evaluation (if available)
    """
    report_id: str
    evaluated_at: str
    layer_name: str
    document_results: list[DocumentEvaluationResult] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    comparison_with_previous: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "report_id": self.report_id,
            "evaluated_at": self.evaluated_at,
            "layer_name": self.layer_name,
            "document_results": [r.to_dict() for r in self.document_results],
            "aggregate_metrics": {k: round(v, 4) for k, v in self.aggregate_metrics.items()},
            "comparison_with_previous": self.comparison_with_previous,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def to_markdown(self) -> str:
        """Convert to Markdown format for human-readable reports."""
        lines = [
            f"# RAG Evaluation Report",
            f"",
            f"**Report ID:** {self.report_id}",
            f"**Evaluated At:** {self.evaluated_at}",
            f"**Layer:** {self.layer_name}",
            f"",
            f"## Aggregate Metrics",
            f"",
        ]
        
        # Aggregate metrics table
        lines.append("| Metric | Score |")
        lines.append("|--------|-------|")
        for metric, score in sorted(self.aggregate_metrics.items()):
            lines.append(f"| {metric} | {score:.4f} |")
        lines.append("")
        
        # Per-document results
        lines.append("## Per-Document Results")
        lines.append("")
        
        for doc_result in self.document_results:
            doc_name = Path(doc_result.document_path).name
            lines.append(f"### {doc_name}")
            lines.append("")
            lines.append(f"- **QA Pairs:** {len(doc_result.qa_results)}")
            lines.append(f"- **Overall Score:** {doc_result.average_overall_score:.4f}")
            lines.append("")
            
            if doc_result.aggregate_metrics:
                lines.append("| Metric | Score |")
                lines.append("|--------|-------|")
                for metric, score in sorted(doc_result.aggregate_metrics.items()):
                    lines.append(f"| {metric} | {score:.4f} |")
                lines.append("")
        
        # Comparison section
        if self.comparison_with_previous:
            lines.append("## Comparison with Previous Evaluation")
            lines.append("")
            prev = self.comparison_with_previous
            lines.append(f"**Previous Date:** {prev.get('previous_date', 'N/A')}")
            lines.append("")
            
            if 'metric_changes' in prev:
                lines.append("| Metric | Previous | Current | Change |")
                lines.append("|--------|----------|---------|--------|")
                for metric, change in prev['metric_changes'].items():
                    delta = change.get('delta', 0)
                    delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
                    lines.append(
                        f"| {metric} | {change.get('previous', 0):.4f} | "
                        f"{change.get('current', 0):.4f} | {delta_str} |"
                    )
                lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# EVALUATION PIPELINE
# =============================================================================

class RAGEvaluationPipeline:
    """
    Automated pipeline for RAG system evaluation.
    
    This pipeline orchestrates the complete evaluation workflow:
    1. Dataset generation (if needed)
    2. RAG system invocation
    3. Metric computation
    4. Report generation
    
    Attributes:
        config: Configuration dictionary
        dataset_generator: Generator for creating evaluation datasets
        evaluator: Hybrid evaluator for computing metrics
    """
    
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        rag_invoke_func: Callable | None = None,
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            config: Configuration dictionary (default: EVAL_CONFIG)
            rag_invoke_func: Function to invoke RAG system for evaluation.
                Signature: func(question: str, session_id: str) -> dict
                Returns: {"retrieved_context": str, "generated_answer": str}
        """
        self.config = config or EVAL_CONFIG
        self.rag_invoke_func = rag_invoke_func
        
        # Initialize components
        self.dataset_generator = DatasetGenerator(config=self.config)
        self.evaluator = HybridEvaluator(config=self.config)
        
        # Statistics
        self.stats = {
            "datasets_generated": 0,
            "documents_evaluated": 0,
            "qa_pairs_evaluated": 0,
            "total_evaluation_time_sec": 0,
        }
    
    def _invoke_rag_system(
        self,
        question: str,
        session_id: str = "eval_session",
    ) -> dict[str, str]:
        """
        Invoke the RAG system to get an answer.
        
        Args:
            question: Question to ask
            session_id: Session identifier for conversation tracking
            
        Returns:
            Dictionary with retrieved_context and generated_answer
            
        Raises:
            RuntimeError: If no rag_invoke_func is configured
        """
        if not self.rag_invoke_func:
            raise RuntimeError(
                "RAG invoke function not configured. "
                "Provide rag_invoke_func to pipeline or use mock results."
            )
        
        return self.rag_invoke_func(question, session_id)
    
    def evaluate_layer(
        self,
        layer_name: str,
        docs_dir: Path | None = None,
        force_regenerate_datasets: bool = False,
        use_mock_rag: bool = False,
        mock_accuracy: float = 0.8,
    ) -> EvaluationReport:
        """
        Evaluate a complete layer (set of documents).
        
        Args:
            layer_name: Name of the layer (e.g., "layer1_books")
            docs_dir: Directory containing PDF documents
            force_regenerate_datasets: If True, regenerate all datasets
            use_mock_rag: If True, use mock RAG results (for testing)
            mock_accuracy: Target accuracy for mock results (0-1)
            
        Returns:
            EvaluationReport with complete results
        """
        import time
        start_time = time.time()
        
        docs_dir = docs_dir or Path(self.config["docs_dir"])
        
        print(f"\n{'='*60}")
        print(f"Evaluating layer: {layer_name}")
        print(f"Documents directory: {docs_dir}")
        print(f"{'='*60}\n")
        
        # Step 1: Generate datasets
        print("Step 1: Generating/Loading datasets...")
        datasets = self.dataset_generator.generate_datasets_for_all_docs(
            docs_dir=docs_dir,
            force_regenerate=force_regenerate_datasets,
        )
        self.stats["datasets_generated"] = len(datasets)
        
        if not datasets:
            return EvaluationReport(
                report_id=f"{layer_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                evaluated_at=datetime.now().isoformat(),
                layer_name=layer_name,
            )
        
        # Step 2: Evaluate each dataset
        print("\nStep 2: Evaluating RAG performance...")
        document_results = []
        
        for dataset in datasets:
            print(f"\n  Evaluating {Path(dataset.document_path).name}...")
            
            rag_results = []
            
            for qa_pair in dataset.qa_pairs:
                if use_mock_rag:
                    # Generate mock RAG result
                    import random
                    if random.random() < mock_accuracy:
                        # Good answer (similar to ground truth)
                        generated_answer = qa_pair.answer[:80] + "..."
                    else:
                        # Poor answer
                        generated_answer = "I don't have enough information to answer this question."
                    
                    retrieved_context = qa_pair.context[:500]
                else:
                    # Invoke real RAG system
                    try:
                        rag_result = self._invoke_rag_system(
                            question=qa_pair.question,
                            session_id=f"eval_{dataset.document_path}",
                        )
                        generated_answer = rag_result.get("generated_answer", "")
                        retrieved_context = rag_result.get("retrieved_context", "")
                    except Exception as e:
                        print(f"    Error invoking RAG: {e}")
                        generated_answer = f"Error: {e}"
                        retrieved_context = ""
                
                rag_results.append({
                    "question": qa_pair.question,
                    "retrieved_context": retrieved_context,
                    "generated_answer": generated_answer,
                })
            
            # Evaluate this document's dataset
            doc_result = self.evaluator.evaluate_dataset(
                dataset=dataset,
                rag_results=rag_results,
            )
            
            document_results.append(doc_result)
            self.stats["documents_evaluated"] += 1
            self.stats["qa_pairs_evaluated"] += len(doc_result.qa_results)
            
            print(f"    Overall score: {doc_result.average_overall_score:.4f}")
        
        # Step 3: Calculate aggregate metrics
        print("\nStep 3: Calculating aggregate metrics...")
        aggregate_metrics = self._calculate_aggregate_metrics(document_results)
        
        # Step 4: Load previous report for comparison
        print("Step 4: Loading previous report for comparison...")
        comparison = self._load_comparison(layer_name)
        
        # Create final report
        report = EvaluationReport(
            report_id=f"{layer_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            evaluated_at=datetime.now().isoformat(),
            layer_name=layer_name,
            document_results=document_results,
            aggregate_metrics=aggregate_metrics,
            comparison_with_previous=comparison,
        )
        
        # Save report
        self._save_report(report)
        
        elapsed_time = time.time() - start_time
        self.stats["total_evaluation_time_sec"] = elapsed_time
        
        print(f"\n{'='*60}")
        print(f"Evaluation complete!")
        print(f"  Documents evaluated: {self.stats['documents_evaluated']}")
        print(f"  QA pairs evaluated: {self.stats['qa_pairs_evaluated']}")
        print(f"  Overall score: {aggregate_metrics.get('overall', 0):.4f}")
        print(f"  Time elapsed: {elapsed_time:.1f} seconds")
        print(f"{'='*60}\n")
        
        return report
    
    def _calculate_aggregate_metrics(
        self,
        document_results: list[DocumentEvaluationResult],
    ) -> dict[str, float]:
        """
        Calculate aggregate metrics across all documents.
        
        Args:
            document_results: List of document evaluation results
            
        Returns:
            Dictionary of aggregated metric scores
        """
        if not document_results:
            return {}
        
        # Collect all metrics from all documents
        metric_totals: dict[str, list[float]] = {}
        
        for doc_result in document_results:
            for metric_name, score in doc_result.aggregate_metrics.items():
                if metric_name not in metric_totals:
                    metric_totals[metric_name] = []
                metric_totals[metric_name].append(score)
        
        # Calculate averages
        aggregates = {
            name: sum(scores) / len(scores)
            for name, scores in metric_totals.items()
        }
        
        return aggregates
    
    def _load_comparison(self, layer_name: str) -> dict[str, Any]:
        """
        Load previous evaluation report for comparison.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Comparison data or empty dict if no previous report
        """
        reports_dir = Path(self.config["reports_dir"])
        
        # Find most recent previous report
        report_files = sorted(reports_dir.glob(f"{layer_name}_*.json"))
        
        if len(report_files) < 2:
            return {}
        
        # Load second most recent (previous)
        previous_report_path = report_files[-2]
        
        try:
            previous_data = json.loads(previous_report_path.read_text())
            
            # Build comparison
            comparison = {
                "previous_date": previous_data.get("evaluated_at", "Unknown"),
                "metric_changes": {},
            }
            
            previous_metrics = previous_data.get("aggregate_metrics", {})
            
            # Current aggregates will be filled in by caller
            # For now, return structure
            return comparison
            
        except Exception as e:
            print(f"Warning: Could not load previous report: {e}")
            return {}
    
    def _save_report(self, report: EvaluationReport) -> None:
        """
        Save evaluation report to disk.
        
        Args:
            report: Report to save
        """
        reports_dir = Path(self.config["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = reports_dir / f"{report.report_id}.json"
        json_path.write_text(report.to_json(), encoding="utf-8")
        
        # Save Markdown
        md_path = reports_dir / f"{report.report_id}.md"
        md_path.write_text(report.to_markdown(), encoding="utf-8")
        
        print(f"  Report saved: {json_path}")
        print(f"  Markdown saved: {md_path}")
    
    def get_stats(self) -> dict[str, Any]:
        """
        Returns pipeline statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        return self.stats.copy()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def evaluate_layer_1_books(
    rag_invoke_func: Callable | None = None,
    force_regenerate: bool = False,
) -> EvaluationReport:
    """
    Evaluate Layer 1: Current books in back/docs/.
    
    Args:
        rag_invoke_func: Function to invoke RAG system
        force_regenerate: If True, regenerate all datasets
        
    Returns:
        EvaluationReport with results
    """
    pipeline = RAGEvaluationPipeline(rag_invoke_func=rag_invoke_func)
    
    return pipeline.evaluate_layer(
        layer_name="layer1_books",
        docs_dir=Path(EVAL_CONFIG["docs_dir"]),
        force_regenerate_datasets=force_regenerate,
    )


def evaluate_layer_2_new_docs(
    rag_invoke_func: Callable | None = None,
    force_regenerate: bool = False,
) -> EvaluationReport:
    """
    Evaluate Layer 2: New documents (to be added).
    
    Args:
        rag_invoke_func: Function to invoke RAG system
        force_regenerate: If True, regenerate all datasets
        
    Returns:
        EvaluationReport with results
    """
    # TODO: Define path for new documents
    new_docs_dir = Path("back/docs_new")
    
    if not new_docs_dir.exists():
        print(f"Warning: New docs directory does not exist: {new_docs_dir}")
        return EvaluationReport(
            report_id=f"layer2_new_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            evaluated_at=datetime.now().isoformat(),
            layer_name="layer2_new_docs",
        )
    
    pipeline = RAGEvaluationPipeline(rag_invoke_func=rag_invoke_func)
    
    return pipeline.evaluate_layer(
        layer_name="layer2_new_docs",
        docs_dir=new_docs_dir,
        force_regenerate_datasets=force_regenerate,
    )
