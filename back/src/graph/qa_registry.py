"""Utilities para manejo centralizado de atributos de calidad (QA).

Objetivo:
- Evitar heurísticas duplicadas en varios nodos.
- Mantener una única fuente de verdad para normalización de QA.
- Facilitar alta de nuevos QA desde `config/indices.json`.
"""

from __future__ import annotations

from functools import lru_cache
import re

from src.graph.index_resolver import get_available_indices


# Alias de naming para nodos por compatibilidad histórica.
# Se mantienen los nombres solicitados por producto: latency/scalability.
_NODE_SUFFIX_ALIAS = {
    "latencia": "latency",
    "escalabilidad": "scalability",
}


def _slug(text: str) -> str:
    """Convierte texto libre a un sufijo seguro para nombres de nodo."""
    out = (text or "").strip().lower()
    out = out.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    out = re.sub(r"[^a-z0-9_]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out or "general"


@lru_cache(maxsize=1)
def _qa_catalog() -> dict[str, set[str]]:
    """Carga catálogo QA desde config y arma mapa de sinónimos por QA id.

    Retorna un diccionario:
    {
      "latencia": {"latencia", "latency", "response time", ...},
      "escalabilidad": {...},
      ...
    }
    """
    catalog: dict[str, set[str]] = {}
    for qa in get_available_indices() or []:
        qa_id = (qa.get("id") or "").strip().lower()
        if not qa_id:
            continue
        terms = {qa_id}
        terms.update((qa.get("keywords_en") or []))
        terms.update((qa.get("keywords_es") or []))
        catalog[qa_id] = {str(t).strip().lower() for t in terms if str(t).strip()}
    return catalog


def supported_qas() -> list[str]:
    """Lista de QA ids soportados desde config (sin incluir `general`)."""
    return list(_qa_catalog().keys())


def normalize_qa(value: str) -> str:
    """Normaliza cualquier valor/sinónimo al QA id canónico.

    Si no hay match claro, retorna `general`.
    """
    low = (value or "").strip().lower()
    if not low:
        return "general"

    catalog = _qa_catalog()
    for qa_id, terms in catalog.items():
        if low == qa_id or low in terms:
            return qa_id

    # Match por substring para entradas tipo "low latency".
    for qa_id, terms in catalog.items():
        if any(term and term in low for term in terms):
            return qa_id

    return "general"


def detect_explicit_qa(text: str) -> str:
    """Detecta QA explícito en texto del usuario usando keywords del catálogo."""
    return normalize_qa(text)


def qa_to_focus_label(qa_id: str, default: str = "performance") -> str:
    """Convierte un QA canónico a etiqueta de foco para prompts en inglés.

    - Si existe `keywords_en` en config para ese QA, usa la primera keyword.
    - Si no existe, usa `display_name` o el id como fallback.
    - Si el QA no es reconocible, retorna `default`.

    Esto evita hardcodear ternarios por QA dentro de nodos.
    """
    qa = normalize_qa(qa_id)
    if qa == "general":
        return default

    for item in get_available_indices() or []:
        item_id = (item.get("id") or "").strip().lower()
        if item_id != qa:
            continue

        kws_en = [str(k).strip() for k in (item.get("keywords_en") or []) if str(k).strip()]
        if kws_en:
            return kws_en[0]

        display_name = str(item.get("display_name") or "").strip()
        if display_name:
            return display_name.lower()

        return qa

    return default


def qa_to_node_suffix(qa_id: str) -> str:
    """Mapea QA id a sufijo de nodo (con alias de compatibilidad cuando aplica)."""
    qa = normalize_qa(qa_id)
    if qa == "general":
        return "general"
    return _NODE_SUFFIX_ALIAS.get(qa, _slug(qa))


def style_node_name_for_qa(qa_id: str) -> str:
    """Devuelve nombre de nodo style para un QA canónico."""
    qa = normalize_qa(qa_id)
    if qa == "general":
        return "style"
    return f"style_{qa_to_node_suffix(qa)}"


def tactics_node_name_for_qa(qa_id: str) -> str:
    """Devuelve nombre de nodo tactics para un QA canónico."""
    qa = normalize_qa(qa_id)
    if qa == "general":
        return "tactics"
    return f"tactics_{qa_to_node_suffix(qa)}"
