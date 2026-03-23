"""
Command-line interface for RAG Evaluation

Ejecutar evaluación desde la línea de comandos:

    # Evaluar capa 1 (libros actuales) con RAG real
    poetry run python -m back.eval --layer layer1_books
    
    # Evaluar con mock (para testing sin RAG)
    poetry run python -m back.eval --layer layer1_books --mock
    
    # Forzar regeneración de datasets
    poetry run python -m back.eval --layer layer1_books --regenerate
    
    # Ver estadísticas
    poetry run python -m back.eval --stats
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework for ArchIA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --layer layer1_books
  %(prog)s --layer layer1_books --mock --accuracy 0.75
  %(prog)s --layer layer1_books --regenerate
  %(prog)s --stats
        """,
    )
    
    parser.add_argument(
        "--layer",
        type=str,
        choices=["layer1_books", "layer2_new_docs"],
        help="Evaluation layer to run",
    )
    
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock RAG results (for testing without RAG system)",
    )
    
    parser.add_argument(
        "--accuracy",
        type=float,
        default=0.8,
        help="Target accuracy for mock results (0-1). Default: 0.8",
    )
    
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force regeneration of evaluation datasets",
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show evaluation statistics",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format for reports. Default: both",
    )
    
    args = parser.parse_args()
    
    # Show stats only
    if args.stats:
        from .pipeline import RAGEvaluationPipeline
        pipeline = RAGEvaluationPipeline()
        print("\nEvaluation Statistics:")
        print("-" * 40)
        for key, value in pipeline.get_stats().items():
            print(f"  {key}: {value}")
        return
    
    # Require layer for evaluation
    if not args.layer:
        parser.print_help()
        print("\nError: --layer is required for evaluation")
        sys.exit(1)
    
    # Import here to avoid loading dependencies unnecessarily
    from .pipeline import evaluate_layer_1_books, evaluate_layer_2_new_docs
    
    # Define mock RAG function if needed
    def mock_rag_func(question: str, session_id: str) -> dict:
        import random
        if random.random() < args.accuracy:
            return {
                "retrieved_context": "Mock context for testing...",
                "generated_answer": f"Mock answer to: {question[:50]}...",
            }
        else:
            return {
                "retrieved_context": "",
                "generated_answer": "I don't have enough information to answer this question.",
            }
    
    # Run evaluation
    if args.layer == "layer1_books":
        print(f"\n{'='*60}")
        print("EVALUACIÓN CAPA 1: Libros Actuales")
        print(f"{'='*60}\n")
        
        report = evaluate_layer_1_books(
            rag_invoke_func=None if args.mock else mock_rag_func,
            force_regenerate=args.regenerate,
        )
        
    elif args.layer == "layer2_new_docs":
        print(f"\n{'='*60}")
        print("EVALUACIÓN CAPA 2: Nuevos Documentos")
        print(f"{'='*60}\n")
        
        report = evaluate_layer_2_new_docs(
            rag_invoke_func=None if args.mock else mock_rag_func,
            force_regenerate=args.regenerate,
        )
    
    else:
        print(f"Error: Unknown layer: {args.layer}")
        sys.exit(1)
    
    # Print summary
    if report.document_results:
        print(f"\n{'='*60}")
        print("RESUMEN DE EVALUACIÓN")
        print(f"{'='*60}")
        print(f"  Report ID: {report.report_id}")
        print(f"  Documentos evaluados: {len(report.document_results)}")
        print(f"  QA pairs evaluados: {sum(len(r.qa_results) for r in report.document_results)}")
        print(f"  Overall score: {report.aggregate_metrics.get('overall', 0):.4f}")
        print(f"{'='*60}\n")
        
        # Print per-document scores
        print("Scores por documento:")
        for doc_result in report.document_results:
            from pathlib import Path
            doc_name = Path(doc_result.document_path).name
            score = doc_result.average_overall_score
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {doc_name:<40} [{bar}] {score:.4f}")
        print()


if __name__ == "__main__":
    main()
