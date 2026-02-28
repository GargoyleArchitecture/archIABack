"""
Analiza el corpus de Chroma para encontrar qué términos de tácticas aparecen
más frecuentemente en los chunks actuales.
"""
from __future__ import annotations
import os, re
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

BASE_DIR = Path(__file__).resolve().parent

# ── Cargamos el vectorstore existente ─────────────────────────────────────────
from src.rag_agent import create_or_load_vectorstore

print("[analyze] Cargando vectorstore…")
vdb = create_or_load_vectorstore()

# Extraemos todos los documentos de la colección
col = vdb._collection
all_docs = col.get(include=["documents", "metadatas"])
texts    = all_docs.get("documents") or []
metas    = all_docs.get("metadatas") or []
print(f"[analyze] Total chunks: {len(texts)}")

# ── Términos candidatos para 'tacticas' ───────────────────────────────────────
# Basados en Software Architecture in Practice (Bass, Clements, Kazman)
tactic_candidates = [
    # Performance tactics (cap 8)
    "manage resources", "manage resource demand", "control resource demand",
    "limit event response", "prioritize events", "reduce overhead",
    "bound execution times", "increase resource efficiency",
    "increase resources", "introduce concurrency", "maintain multiple copies",
    "reduce contention", "bound queue sizes", "schedule resources",
    "arbitrate", "arbitration",
    # Availability tactics (cap 5)
    "detect faults", "recover from faults", "prevent faults",
    "active redundancy", "passive redundancy", "spare",
    "exception handling", "rollback", "retry", "ignore faulty behavior",
    "degradation", "graceful degradation", "shadow", "state resynchronization",
    "escalating restart", "non-stop forwarding",
    # Security tactics (cap 9)
    "authenticate actors", "authorize actors", "limit access",
    "limit exposure", "encrypt data", "separate entities",
    "change default settings", "detect intrusion", "detect service denial",
    "verify message integrity", "detect message delay",
    # Modifiability tactics (cap 7)
    "increase semantic coherence", "encapsulate", "use an intermediary",
    "restrict dependencies", "defer binding", "abstract common services",
    "split module", "redistribute responsibilities",
    "prevent ripple effects", "defer binding time",
    # Interoperability tactics
    "orchestrate", "tailor interface", "manage interaction",
    # Performance/scalability specific
    "caching", "cache", "load balancing", "load balance", "load shedding",
    "circuit breaker", "throttling", "throttle", "rate limiting",
    "backpressure", "bulkhead", "timeout", "retry", "failover",
    "replication", "sharding", "partitioning",
    "queue", "queuing", "pipeline", "batch",
    "connection pool", "thread pool", "resource pool",
    "asynchronous", "non-blocking", "event-driven",
    # Generic
    "architectural tactic", "tactic", "tactics",
    "performance tactic", "availability tactic",
    "scalability tactic", "security tactic",
]

# ── Contamos apariciones en todos los chunks ──────────────────────────────────
counter: Counter = Counter()
for text in texts:
    low = text.lower()
    for kw in tactic_candidates:
        if kw.lower() in low:
            counter[kw] += 1

print("\n[analyze] Top 50 términos de tácticas más frecuentes en el corpus:")
print(f"{'Término':<40} {'Chunks con el término':>22}")
print("-" * 65)
for term, cnt in counter.most_common(50):
    print(f"{term:<40} {cnt:>22}")

# ── Distribución actual de tags ───────────────────────────────────────────────
print("\n[analyze] Distribución actual de content_type:")
ct_counter: Counter = Counter()
for meta in metas:
    ct = (meta or {}).get("content_type", "no-tag")
    ct_counter[ct] += 1
for ct, cnt in ct_counter.most_common():
    print(f"  {ct}: {cnt} chunks")

print("\n[analyze] Distribución actual de quality_attribute:")
qa_counter: Counter = Counter()
for meta in metas:
    qa = (meta or {}).get("quality_attribute", "no-tag")
    qa_counter[qa] += 1
for qa, cnt in qa_counter.most_common():
    print(f"  {qa}: {cnt} chunks")
