# Framework de Evaluación RAG para ArchIA

## Visión General

Este módulo proporciona un sistema completo de evaluación para el sistema RAG de ArchIA, combinando las mejores prácticas de **RAGAS** y **CCRS** en un enfoque híbrido.

### Características Principales

- **Generación automática de datasets** estilo MiRAGE (multi-agente)
- **10 QA pairs por documento** con distribución balanceada
- **Verificación adversarial** para detectar alucinaciones
- **9 métricas híbridas** (4 de RAGAS + 5 de CCRS)
- **Pipeline automatizado** con reportes en JSON y Markdown
- **Cache inteligente** que solo regenera si el documento cambió

---

## Estructura del Módulo

```
back/eval/
├── __init__.py           # exports principales
├── __main__.py           # CLI para ejecutar desde consola
├── config.py             # configuración centralizada
├── pipeline.py           # pipeline de evaluación
├── generators/
│   ├── __init__.py
│   └── dataset_generator.py   # generador de QA pairs (2 agentes)
├── metrics/
│   ├── __init__.py
│   └── hybrid_evaluator.py    # evaluador híbrido CCRS+RAGAS
└── datasets/             # datasets generados (auto)
└── reports/              # reportes de evaluación (auto)
```

---

## Instalación

Las dependencias ya están instaladas vía Poetry. Solo necesitas:

1. Tener `OPENAI_API_KEY` configurada en `.env`
2. Tener los PDFs en `back/docs/`

---

## Uso Rápido

### Desde la línea de comandos

```bash
# Evaluar capa 1 (libros actuales) con mock
poetry run python -m back.eval --layer layer1_books --mock

# Evaluar capa 1 con RAG real (requiere tener el RAG configurado)
poetry run python -m back.eval --layer layer1_books

# Forzar regeneración de datasets
poetry run python -m back.eval --layer layer1_books --regenerate

# Ver estadísticas
poetry run python -m back.eval --stats
```

### Desde Python

```python
from back.eval import evaluate_layer_1_books

# Opción 1: Con mock (para testing)
def mock_rag_func(question: str, session_id: str) -> dict:
    return {
        "retrieved_context": "Contexto de prueba...",
        "generated_answer": f"Respuesta a: {question}",
    }

report = evaluate_layer_1_books(
    rag_invoke_func=mock_rag_func,
    force_regenerate=False,
)

# Ver reporte en Markdown
print(report.to_markdown())

# Opción 2: Con tu RAG real
def mi_rag_func(question: str, session_id: str) -> dict:
    # Aquí va tu lógica para invocar el RAG
    # Debe retornar retrieved_context y generated_answer
    ...

report = evaluate_layer_1_books(rag_invoke_func=mi_rag_func)
```

---

## Configuración

El archivo `back/eval/config.py` contiene toda la configuración:

```python
EVAL_CONFIG = {
    # Dataset Generation
    "qa_pairs_per_doc": 10,
    "question_types": {
        "factual": 4,       # 40% - Retrieval básico
        "multi_hop": 4,     # 40% - Razonamiento multi-hop
        "synthesis": 2,     # 20% - Comprensión global
    },

    # Modelos LLM
    "generation_model": "gpt-4o-mini",  # Generar QA (barato)
    "evaluation_model": "gpt-4o",       # Evaluar (calidad)

    # Verificación
    "verifier_strictness": "high",
    "verification_attempts": 2,

    # Métricas habilitadas
    "metrics": {
        # RAGAS
        "faithfulness": True,
        "answer_relevance": True,
        "context_precision": True,
        "context_recall": True,

        # CCRS
        "contextual_coherence": True,
        "question_relevance": True,
        "information_density": True,
        "answer_correctness": True,
        "information_recall": True,
    },

    # Cache
    "cache_enabled": True,
    "cache_ttl_days": 30,
}
```

---

## Métricas de Evaluación

### Métricas RAGAS

| Métrica               | Descripción                                            | Rango |
| --------------------- | ------------------------------------------------------ | ----- |
| **Faithfulness**      | ¿La respuesta se infiere solo del contexto recuperado? | 0-1   |
| **Answer Relevance**  | ¿La respuesta es relevante a la pregunta?              | 0-1   |
| **Context Precision** | ¿El contexto recuperado es preciso?                    | 0-1   |
| **Context Recall**    | ¿Se recuperó todo el contexto necesario?               | 0-1   |

### Métricas CCRS

| Métrica                  | Descripción                                       | Rango |
| ------------------------ | ------------------------------------------------- | ----- |
| **Contextual Coherence** | ¿La respuesta tiene flujo lógico?                 | 0-1   |
| **Question Relevance**   | ¿La pregunta es relevante al dominio?             | 0-1   |
| **Information Density**  | ¿La respuesta es concisa sin filler?              | 0-1   |
| **Answer Correctness**   | ¿La respuesta es factualmente correcta?           | 0-1   |
| **Information Recall**   | ¿Cuánta información del ground truth se recuerda? | 0-1   |

---

## Estructura de Reportes

### JSON Report

```json
{
  "report_id": "layer1_books_20260322_143022",
  "evaluated_at": "2026-03-22T14:30:22",
  "layer_name": "layer1_books",
  "document_results": [...],
  "aggregate_metrics": {
    "faithfulness": 0.8523,
    "answer_relevance": 0.9012,
    "overall": 0.8734
  },
  "comparison_with_previous": {...}
}
```

### Markdown Report

# RAG Evaluation Report

**Report ID:** layer1_books_20260322_143022
**Evaluated At:** 2026-03-22T14:30:22
**Layer:** layer1_books

## Aggregate Metrics

| Metric           | Score  |
| ---------------- | ------ |
| faithfulness     | 0.8523 |
| answer_relevance | 0.9012 |
| overall          | 0.8734 |

## Per-Document Results

### Software_Architecture_in_practice.pdf

- **QA Pairs:** 10
- **Overall Score:** 0.8821
  ...

---

## Flujo de Evaluación

# PIPELINE DE EVALUACIÓN

1. **DETECCIÓN DE DOCUMENTOS**
   - Escanear `back/docs/*.pdf`

2. **GENERACIÓN DE DATASETS** _(si no existen en caché)_
   - **Extractor de texto:** PyMuPDF
   - **Generator Agent:** 10 QA pairs por documento
   - **Verifier Agent:** Validación adversarial

3. **EJECUCIÓN DEL RAG**
   - Invocar RAG para cada par de preguntas y respuestas (QA)
   - Capturar `retrieved_context` + `generated_answer`

4. **CÁLCULO DE MÉTRICAS**
   - **RAGAS:** 4 métricas
   - **CCRS:** 5 métricas
   - **Overall score:** Promedio final

5. **GENERACIÓN DE REPORTES**
   - **JSON:** `back/eval/reports/`
   - **Markdown:** `back/eval/reports/`

---

## Costos Estimados

| Concepto                                | Costo (GPT-4o + GPT-4o-mini) |
| --------------------------------------- | ---------------------------- |
| **Generación dataset (6 docs × 10 QA)** | ~$3.00 (una vez)             |
| **Evaluación completa**                 | ~$1.50                       |
| **Total inicial**                       | **~$4.50**                   |
| **Por doc nuevo**                       | ~$0.50                       |
| **Re-evaluación mensual**               | ~$1.50                       |

---

## Tests

```bash
# Ejecutar tests del framework
poetry run pytest back/tests/test_eval_framework.py -v

# Con coverage
poetry run pytest back/tests/test_eval_framework.py --cov=back/eval
```

---

## Arquitectura de 3 Capas

El sistema está diseñado para evaluar en 3 capas:

### Capa 1: Libros Actuales

- PDFs en `back/docs/`
- Ya implementada y funcional

### Capa 2: Nuevos Documentos

- Misma lógica que Capa 1

### Capa 3: Videos (EVRAG)

- Videos en `back/videos/raw/`

---

## Comandos Útiles

```bash
# Evaluar con mock (testing)
poetry run python -m back.eval --layer layer1_books --mock

# Evaluar con RAG real
poetry run python -m back.eval --layer layer1_books

# Forzar regeneración
poetry run python -m back.eval --layer layer1_books --regenerate

# Ver estadísticas
poetry run python -m back.eval --stats

# Ejecutar tests
poetry run pytest back/tests/test_eval_framework.py -v
```

---

## Solución de Problemas

### Error: "OPENAI_API_KEY not found"

Asegúrate de tener `.env` en `back/src/` con:

```
OPENAI_API_KEY=sk-...
```

### Error: "No PDF files found"

Verifica que hay PDFs en `back/docs/`:

```bash
ls back/docs/*.pdf
```

### Error: "RAG invoke function not configured"

Debes proporcionar una función `rag_invoke_func` al pipeline:

```python
def mi_rag_func(question, session_id):
    # Tu lógica aquí
    return {"retrieved_context": "...", "generated_answer": "..."}

pipeline = RAGEvaluationPipeline(rag_invoke_func=mi_rag_func)
```
