import os
import sys
import asyncio
import json
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any

# Configurar rutas para importar desde el backend
BACK_DIR = Path(__file__).resolve().parent / "back"
sys.path.append(str(BACK_DIR))

from src.graph import graph as graph_app
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Cargar variables de entorno
load_dotenv(BACK_DIR / "src" / ".env")

# --- DATASET DE EVALUACIÓN ---
EVAL_DATASET = [
    {
        "id": "Q1",
        "question": "¿Cuáles son las tácticas fundamentales para mejorar la disponibilidad (Availability)?",
        "expected_topics": ["redundancia", "detección de fallos", "recuperación", "retry", "circuit breaker"]
    },
    {
        "id": "Q2",
        "question": "¿Qué dice el contenido sobre el particionamiento de bases de datos y escalabilidad?",
        "expected_topics": ["sharding", "escalabilidad horizontal", "particionamiento", "throughput"]
    },
    {
        "id": "Q3",
        "question": "Genera un requisito significativo de arquitectura (ASR) para un sistema bancario que requiere alta seguridad y latencia menor a 200ms.",
        "expected_topics": ["ASR", "latencia", "seguridad", "escenario de calidad"]
    }
]

# --- MODOS DE EVALUACIÓN ---
MODES = [
    {"name": "Text-Only (PDF)", "rag_mode": "text", "evrag_mode": "visual"},
    {"name": "EVRAG Visual (CLIP)", "rag_mode": "video", "evrag_mode": "visual"},
    {"name": "EVRAG Descriptive", "rag_mode": "video", "evrag_mode": "descriptive"},
    {"name": "EVRAG Hybrid", "rag_mode": "video", "evrag_mode": "hybrid"},
    {"name": "Hybrid Visual (PDF+CLIP)", "rag_mode": "both", "evrag_mode": "visual"},
    {"name": "Hybrid Descriptive (PDF+Desc)", "rag_mode": "both", "evrag_mode": "descriptive"},
    {"name": "Hybrid Total (Full)", "rag_mode": "both", "evrag_mode": "hybrid"}
]

class MiRAGEJudge:
    """LLM-as-a-Judge implementando métricas MiRAGE (InfoF1, CiteF1)."""
    
    def __init__(self, model="gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model, temperature=0)

    async def _call_llm(self, prompt: str, json_mode=True) -> Any:
        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            content = response.content
            if json_mode:
                # Limpiar markdown si existe
                content = re.sub(r'```json\s*|\s*```', '', content).strip()
                return json.loads(content)
            return content
        except Exception as e:
            print(f"Error calling LLM Judge: {e}")
            return {"error": str(e)}

    async def evaluate_comprehensive(self, question: str, answer: str, context: List[str]) -> Dict[str, Any]:
        """Calcula todas las métricas (MiRAGE + RAGAS/CCRS) en un set optimizado de llamadas."""
        context_text = "\n---\n".join(context)[:15000]
        
        # 1. MiRAGE: Decomposición en Subclaims
        decomposition_prompt = f"""
        Descompone la siguiente respuesta en una lista de 'subclaims' (afirmaciones atómicas).
        RESPUESTA: {answer}
        Responde solo en JSON: {{"subclaims": ["afirmacion 1", "afirmacion 2"]}}
        """
        subclaims_res = await self._call_llm(decomposition_prompt)
        subclaims = subclaims_res.get("subclaims", [])
        
        # 2. MiRAGE Precision + RAGAS Faithfulness + CCRS Correctness
        # Combinamos estas evaluaciones para ahorrar tokens y tiempo
        multi_eval_prompt = f"""
        Eres un evaluador experto. Analiza la RESPUESTA y el CONTEXTO.
        
        CONTEXTO: {context_text}
        RESPUESTA: {answer}
        AFIRMACIONES ESPECÍFICAS: {json.dumps(subclaims)}
        
        Evalúa y responde en JSON con:
        1. "verifications": lista de true/false para cada AFIRMACIÓN ESPECÍFICA basada en el contexto.
        2. "faithfulness": score 0-10 (¿La respuesta se deriva solo del contexto?).
        3. "correctness": score 0-10 (¿Es técnicamente correcta?).
        4. "relevancy": score 0-10 (¿El contexto fue útil para la pregunta?).
        
        JSON:
        {{
            "verifications": [true, false, ...],
            "faithfulness": 0-10,
            "correctness": 0-10,
            "relevancy": 0-10
        }}
        """
        eval_res = await self._call_llm(multi_eval_prompt)
        
        # Calcular InfoF1 Precision
        verif_results = eval_res.get("verifications", [])
        precision = sum(1 for r in verif_results if r) / len(subclaims) if subclaims else 0
        
        # 3. MiRAGE Recall (Cobertura)
        recall_prompt = f"""
        Identifica los 5 puntos clave de información en el CONTEXTO para esta PREGUNTA: {question}
        ¿La RESPUESTA cubre cada uno?
        Responde solo en JSON: {{"recall_score": 0-10}}
        """
        recall_res = await self._call_llm(recall_prompt)
        recall = recall_res.get("recall_score", 0) / 10

        # InfoF1 calculation
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 4. CiteF1 (Calidad de Citas)
        cite_prompt = f"""
        Analiza las citas en la respuesta (ej. [1], VIDEO_SOURCES).
        ¿Son precisas y respaldadas por el contexto?
        RESPUESTA: {answer}
        Responde solo en JSON: {{"cite_score": 0-10}}
        """
        cite_res = await self._call_llm(cite_prompt)

        return {
            "info_f1": round(f1 * 10, 2),
            "cite_f1": cite_res.get("cite_score", 0),
            "faithfulness": eval_res.get("faithfulness", 0),
            "correctness": eval_res.get("correctness", 0),
            "relevancy": eval_res.get("relevancy", 0)
        }

async def get_system_response(question: str, rag_mode: str, evrag_mode: str) -> Dict[str, Any]:
    """Invoca el grafo con una configuración específica."""
    config = {"configurable": {"thread_id": f"eval_{rag_mode}_{evrag_mode}_{os.urandom(2).hex()}"}}
    
    state = {
        "messages": [HumanMessage(content=question)],
        "userQuestion": question,
        "rag_mode": rag_mode,
        "evrag_mode": evrag_mode,
        "force_rag": True,
        "language": "es",
        "retrieved_docs": [],
        "nextNode": "supervisor"
    }
    
    # Valores por defecto para evitar errores de TypedDict si faltan
    default_state = {
        "localQuestion": "", "turn_messages": [], "requested_nodes": [], 
        "pending_nodes": [], "completed_nodes": [], "endMessage": "",
        "imagePath1": "", "imagePath2": "", "doc_only": False, "doc_context": "",
        "memory_text": "", "suggestions": [], "hasVisitedInvestigator": False,
        "hasVisitedEvaluator": False, "hasVisitedASR": False, "hasVisitedDiagram": False,
        "intent": "architecture", "current_asr": "", "arch_stage": "",
        "quality_attribute": "", "add_context": "", "tactics_list": [],
        "diagram": {}, "diagram_history": {}
    }
    
    full_state = {**default_state, **state}
    
    try:
        result = await graph_app.ainvoke(full_state, config)
        return {
            "answer": result.get("endMessage", "No response"),
            "context": result.get("retrieved_docs", [])
        }
    except Exception as e:
        print(f"Error in graph execution: {e}")
        return {"answer": "Error", "context": []}

async def run_advanced_evaluation():
    print(f"\n{'='*70}")
    print("🚀 INICIANDO EVALUACIÓN COMPARATIVA AVANZADA (MiRAGE FRAMEWORK)")
    print(f"{'='*70}\n")
    
    judge = MiRAGEJudge()
    all_results = {}

    for mode in MODES:
        mode_name = mode["name"]
        print(f"📊 Evaluando Modo: {mode_name}...")
        mode_results = []
        
        for item in EVAL_DATASET:
            print(f"  - Pregunta {item['id']}...")
            
            # 1. Obtener respuesta
            sys_res = await get_system_response(item["question"], mode["rag_mode"], mode["evrag_mode"])
            
            # 2. Evaluar todo
            metrics = await judge.evaluate_comprehensive(item["question"], sys_res["answer"], sys_res["context"])
            
            mode_results.append({
                "id": item["id"],
                **metrics
            })
        
        # Calcular promedios del modo
        avgs = {
            "info_f1": round(sum(r["info_f1"] for r in mode_results) / len(mode_results), 2),
            "cite_f1": round(sum(r["cite_f1"] for r in mode_results) / len(mode_results), 2),
            "faithfulness": round(sum(r["faithfulness"] for r in mode_results) / len(mode_results), 2),
            "correctness": round(sum(r["correctness"] for r in mode_results) / len(mode_results), 2),
            "relevancy": round(sum(r["relevancy"] for r in mode_results) / len(mode_results), 2)
        }
        
        all_results[mode_name] = {
            **avgs,
            "details": mode_results
        }
        print(f"    ✅ InfoF1: {avgs['info_f1']} | CiteF1: {avgs['cite_f1']} | Faith: {avgs['faithfulness']}")

    # --- GENERAR TABLA FINAL ---
    print(f"\n{'='*95}")
    print("🏆 RESULTADOS COMPARATIVOS FINALES (RAGAS + MiRAGE + CCRS)")
    print(f"{'='*95}")
    header = f"{'Modo':<30} | {'InfoF1':<6} | {'CiteF1':<6} | {'Faith':<6} | {'Corr':<6} | {'Rel':<6} | {'Total':<6}"
    print(header)
    print("-" * 95)
    
    for mode_name, m in all_results.items():
        total = round((m["info_f1"] + m["cite_f1"] + m["faithfulness"] + m["correctness"] + m["relevancy"]) / 5, 2)
        row = f"{mode_name:<30} | {m['info_f1']:<6} | {m['cite_f1']:<6} | {m['faithfulness']:<6} | {m['correctness']:<6} | {m['relevancy']:<6} | {total:<6}"
        print(row)
    
    print(f"{'='*70}\n")
    
    # Guardar reporte JSON
    report_path = Path("evaluation_results_advanced.json")
    report_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Reporte detallado guardado en: {report_path.name}")

if __name__ == "__main__":
    asyncio.run(run_advanced_evaluation())
