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
    from dotenv import load_dotenv
    import os
    from pathlib import Path
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        
    parser = argparse.ArgumentParser(
        description="RAG Evaluation Framework for ArchIA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --layer layer1_books --mode standard   # 10 QA/doc (rápido)
  %(prog)s --layer layer1_books --mode comprehensive  # 30 QA/doc (completo)
  %(prog)s --layer layer1_books --mock            # Con mock (testing)
  %(prog)s --layer layer1_books --regenerate      # Forzar regeneración
  %(prog)s --layer layer3_videos                  # Evaluar videos
  %(prog)s --stats                                # Ver estadísticas
        """,
    )

    parser.add_argument(
        "--layer",
        type=str,
        choices=["layer1_books", "layer2_new_docs", "layer3_videos"],
        help="Evaluation layer to run",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "comprehensive"],
        default="standard",
        help="Evaluation mode: standard (10 QA/doc) or comprehensive (30 QA/doc). Default: standard",
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

    # Set evaluation mode
    from .config import set_eval_mode
    set_eval_mode(args.mode)

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
    from .pipeline import evaluate_layer_1_books, evaluate_layer_2_new_docs, evaluate_layer_3_videos

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

    # Define real RAG function using LangGraph
    def real_rag_func(question: str, session_id: str) -> dict:
        import sys
        import os
        
        # Prevenir crashes silenciosos de PyTorch en Windows
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        from pathlib import Path
        backend_dir = Path(__file__).resolve().parent.parent
        if str(backend_dir) not in sys.path:
            sys.path.append(str(backend_dir))
            
        from src.graph import graph
        from langchain_core.messages import HumanMessage
        
        turn_messages = [HumanMessage(content=question)]
        config = {"configurable": {"thread_id": session_id}, "recursion_limit": 20}
        
        # Desactivar EVRAG si estamos evaluando libros para ahorrar RAM/VRAM de CLIP
        current_evrag_mode = "hybrid" if args.layer == "layer3_videos" else "none"
        current_rag_mode = "both" if args.layer == "layer3_videos" else "text"
        
        try:
            result = graph.invoke(
                {
                    "messages": turn_messages,
                    "userQuestion": question,
                    "imagePath1": "",
                    "imagePath2": "",
                    "doc_only": False,
                    "doc_context": "",
                    "rag_mode": current_rag_mode,
                    "evrag_mode": current_evrag_mode,
                    "force_rag": True,
                    "language": "es",
                    "intent": "general",
                    "nextNode": "supervisor",
                },
                config,
            )
            # Extraemos el contexto si el grafo lo provee, o un string dummy
            docs = result.get("retrieved_docs", [])
            context = "\n".join([str(d) for d in docs]) if docs else "Context retrieved from vector DB."
            
            return {
                "retrieved_context": context,
                "generated_answer": result.get("endMessage", "")
            }
        except Exception as e:
            return {
                "retrieved_context": "",
                "generated_answer": f"Error invoking LangGraph: {str(e)}"
            }

    rag_func = mock_rag_func if args.mock else real_rag_func

    # Run evaluation
    if args.layer == "layer1_books":
        print(f"\n{'='*60}")
        print("EVALUACIÓN CAPA 1: Libros Actuales")
        print(f"{'='*60}\n")

        report = evaluate_layer_1_books(
            rag_invoke_func=rag_func,
            force_regenerate=args.regenerate,
        )

    elif args.layer == "layer2_new_docs":
        print(f"\n{'='*60}")
        print("EVALUACIÓN CAPA 2: Nuevos Documentos")
        print(f"{'='*60}\n")

        report = evaluate_layer_2_new_docs(
            rag_invoke_func=rag_func,
            force_regenerate=args.regenerate,
        )

    elif args.layer == "layer3_videos":
        print(f"\n{'='*60}")
        print("EVALUACIÓN CAPA 3: Videos (EVRAG)")
        print(f"{'='*60}\n")

        # For mock mode, we use mock QA generation
        report = evaluate_layer_3_videos(
            rag_invoke_func=rag_func,
            force_regenerate=args.regenerate,
            use_mock=args.mock,  # Use mock QA generation when --mock flag is set
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
