"""Estimación pura de dificultad de un reto generado (F5-T4).

Función testeable sin LLM ni Store. La usa `validate_difficulty_node` para
decidir si la propuesta del LLM es coherente con el perfil del usuario y, si
no lo es, regenerar con menos conceptos (loop limitado a 2 reintentos en el
nodo, no aquí).

Señales consideradas:
- `llm_difficulty`: la dificultad que el LLM se asignó a sí mismo (1..5).
- `num_concepts`: cuántos conceptos involucra el reto. >5 sube la dificultad.
- `snippet_length`: longitud del snippet de RAG inverso. Snippets largos
  cargan al usuario con más contexto a leer. >800 chars sube la dificultad.
- `current_mastery`: NO se usa para ajustar la dificultad aquí; la decisión
  de regenerar (cuando difficulty supera mastery+offset) vive en el nodo
  `validate_difficulty`. Esta función estima la dificultad real, no la
  "apropiada" para el usuario.
"""
from __future__ import annotations


def _clamp_int(value: int, lo: int, hi: int) -> int:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _safe_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def estimate_difficulty(
    *,
    num_concepts: int,
    current_mastery: float,
    snippet_length: int,
    llm_difficulty: int,
) -> int:
    """Combina señales objetivas para estimar la dificultad final del reto.

    Retorna un entero clamp-eado a [1, 5]. La función es determinista y pura
    (no hace I/O, no usa LLM).

    Reglas:
    - base = `llm_difficulty` (clamp-eado a [1,5] de entrada para tolerar
      respuestas fuera de rango del LLM).
    - +1 si `num_concepts > 5` (el reto pide cubrir muchos temas).
    - +1 si `snippet_length > 800` (snippet largo añade carga cognitiva).
    - clamp final a [1, 5].

    El parámetro `current_mastery` se acepta por simetría con la firma del
    nodo (que la pasa para futuras heurísticas). Hoy no afecta el cómputo:
    la decisión de "regenerar porque es muy difícil para este usuario" la
    toma `validate_difficulty_node` comparando la salida de esta función
    contra `current_mastery * 5 + 2`.
    """
    base = _clamp_int(_safe_int(llm_difficulty, 1), 1, 5)
    bonus = 0
    if _safe_int(num_concepts, 0) > 5:
        bonus += 1
    if _safe_int(snippet_length, 0) > 800:
        bonus += 1

    # current_mastery se acepta pero no se usa en este cálculo (ver docstring).
    _ = float(current_mastery) if current_mastery is not None else 0.0

    return _clamp_int(base + bonus, 1, 5)
