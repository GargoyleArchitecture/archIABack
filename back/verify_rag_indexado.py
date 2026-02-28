"""
Verificación del RAG indexado en dos partes:
  1. Chroma — confirma que los metadatos quality_attribute/content_type están presentes
  2. Retriever — prueba que los filtros devuelven chunks distintos según índice
  3. Index resolver — verifica que el LLM clasifica correctamente las preguntas
"""
from __future__ import annotations
import os, json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

SEP  = "=" * 60
SEP2 = "-" * 60

# ══════════════════════════════════════════════════════════════
# PARTE 1 — Distribución de metadatos en Chroma
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PARTE 1: Metadatos en Chroma")
print(SEP)

from src.rag_agent import create_or_load_vectorstore

vdb = create_or_load_vectorstore()
col = vdb._collection
all_docs = col.get(include=["documents", "metadatas"])
texts  = all_docs.get("documents") or []
metas  = all_docs.get("metadatas") or []

print(f"Total chunks: {len(texts)}")

# Verificar que los campos existen
with_qa = sum(1 for m in metas if m and "quality_attribute" in m)
with_ct = sum(1 for m in metas if m and "content_type" in m)
print(f"Chunks con 'quality_attribute': {with_qa}/{len(metas)}")
print(f"Chunks con 'content_type':      {with_ct}/{len(metas)}")

# Distribución
from collections import Counter
qa_ct_counter: Counter = Counter()
for m in metas:
    qa = (m or {}).get("quality_attribute", "NO-TAG")
    ct = (m or {}).get("content_type", "NO-TAG")
    qa_ct_counter[f"{qa}/{ct}"] += 1

print(f"\nDistribución quality_attribute/content_type:")
for key, cnt in qa_ct_counter.most_common():
    bar = "█" * (cnt // 5)
    print(f"  {key:<30} {cnt:>5}  {bar}")

# ══════════════════════════════════════════════════════════════
# PARTE 2 — Prueba de filtros en el retriever
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PARTE 2: Retriever con filtros")
print(SEP)

from src.rag_agent import get_indexed_retriever

QUERY = "performance tactics to reduce latency"

tests = [
    ("Sin filtros (comportamiento base)",     None,          None),
    ("quality_attribute=latencia",            "latencia",    None),
    ("quality_attribute=escalabilidad",       "escalabilidad", None),
    ("content_type=tacticas",                 None,          "tacticas"),
    ("latencia + tacticas",                   "latencia",    "tacticas"),
    ("escalabilidad + tacticas",              "escalabilidad","tacticas"),
    ("latencia + asr",                        "latencia",    "asr"),
]

for label, qa, ct in tests:
    retriever = get_indexed_retriever(quality_attribute=qa, content_type=ct, k=4)
    try:
        docs = retriever.invoke(QUERY)
    except Exception as e:
        print(f"\n  [{label}]  ERROR: {e}")
        continue

    print(f"\n  [{label}]  → {len(docs)} docs recuperados")
    for i, d in enumerate(docs[:3], 1):
        m = d.metadata or {}
        qa_tag = m.get("quality_attribute", "?")
        ct_tag = m.get("content_type", "?")
        src    = Path(m.get("source_path", "?")).name
        page   = m.get("page", "?")
        snip   = d.page_content.replace("\n", " ")[:80]
        print(f"    [{i}] {qa_tag}/{ct_tag}  {src} p.{page}")
        print(f"        \"{snip}…\"")

# ══════════════════════════════════════════════════════════════
# PARTE 3 — Index resolver (LLM clasifica preguntas)
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("PARTE 3: Index resolver (LLM)")
print(SEP)

from src.services.llm_factory import get_chat_model
from src.graph.index_resolver import resolve_quality_attribute

llm = get_chat_model(temperature=0.0)

test_questions = [
    ("¿Cómo diseño un sistema escalable para un e-commerce?",   "escalabilidad"),
    ("Necesito reducir la latencia de mi API REST a menos de 200ms", "latencia"),
    ("¿Qué es ADD 3.0?",                                         "general"),
    ("What tactics reduce response time in a microservice?",    "latencia"),
    ("How do I scale my application horizontally?",              "escalabilidad"),
]

all_correct = True
for question, expected in test_questions:
    result = resolve_quality_attribute(question, llm)
    status = "✅" if result == expected else "❌"
    if result != expected:
        all_correct = False
    print(f"  {status}  resolved='{result}' (esperado='{expected}')")
    print(f"       Q: \"{question[:60]}\"")

print(f"\n{'✅ TODOS correctos' if all_correct else '❌ Hay discrepancias — revisa el prompt del index_resolver'}")

# ══════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════
print(f"\n{SEP}")
print("RESUMEN")
print(SEP)
total_tacticas = sum(c for k, c in qa_ct_counter.items() if "/tacticas" in k)
total_asr      = sum(c for k, c in qa_ct_counter.items() if "/asr" in k)
total_estilos  = sum(c for k, c in qa_ct_counter.items() if "/estilos" in k)
total_general  = sum(c for k, c in qa_ct_counter.items() if "/general" in k)

print(f"  tacticas : {total_tacticas:>5} chunks  ({100*total_tacticas/len(metas):.1f}%)")
print(f"  asr      : {total_asr:>5} chunks  ({100*total_asr/len(metas):.1f}%)")
print(f"  estilos  : {total_estilos:>5} chunks  ({100*total_estilos/len(metas):.1f}%)")
print(f"  general  : {total_general:>5} chunks  ({100*total_general/len(metas):.1f}%)")
print(f"\n  metadata presentes: {'✅ OK' if with_qa == len(metas) and with_ct == len(metas) else '❌ FALTAN tags'}")
