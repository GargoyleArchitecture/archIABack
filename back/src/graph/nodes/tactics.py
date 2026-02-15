
import re
import os
import json
from langchain_core.messages import AIMessage

from src.graph.state import GraphState
from src.graph.resources import llm, retriever, log
from src.utils.json_helpers import (
    extract_json_array,
    strip_first_json_fence,
    normalize_tactics_json,
    build_json_from_markdown,
)
from src.graph.utils import (
    _dedupe_snippets,
    _clip_text,
    _push_turn,
    _json_only_repair_pass,
    _structured_tactics_fallback
)
from src.graph.consts import TACTICS_JSON_EXAMPLE

def _guess_quality_attribute(text: str) -> str:
    low = (text or "").lower()
    if "latenc" in low or "response time" in low: return "latency"
    if "scalab" in low or "throughput" in low:    return "scalability"
    if "availab" in low or "uptime" in low:       return "availability"
    if "secur" in low:                             return "security"
    if "modifiab" in low or "change" in low:       return "modifiability"
    if "reliab" in low or "fault" in low:          return "reliability"
    return "performance"

def tactics_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    directive = "Answer in English." if lang == "en" else "Responde en español."
    doc_only = bool(state.get("doc_only"))
    ctx_doc = (state.get("doc_context") or "").strip()
    ctx_add = (state.get("add_context") or "").strip()
    ctx = (ctx_doc if (doc_only and ctx_doc) else ctx_add)[:2000]

    # 1) Tomamos el ASR actual (o lo inferimos del mensaje)
    asr_text = state.get("asr_text") or state.get("last_asr") or ""
    if not asr_text:
        uq = state.get("userQuestion", "") or ""
        m = re.search(r"(?:^|\n)\s*ASR\s*:?\s*(.+)$", uq, flags=re.I | re.S)
        asr_text = (m.group(1).strip() if m else uq.strip())

    # 2) Deducimos el atributo de calidad
    qa = state.get("quality_attribute") or _guess_quality_attribute(asr_text)
    # Estilo (si lo trae el flujo de ESTILOS)
    style_text = state.get("style") or state.get("selected_style") or state.get("last_style") or ""

    # 3) Contexto para grounding: DOC-ONLY → sin RAG; otro caso → RAG normal
    docs_list = []
    if doc_only and ctx_doc:
        book_snippets = f"[DOC] {ctx_doc[:2000]}"
    else:
        try:
            queries = [
                f"{qa} architectural tactics",
                f"{qa} tactics performance scalability latency availability security modifiability",
                "Bass Clements Kazman performance and scalability tactics",
                "quality attribute tactics list"
            ]
            seen = set()
            gathered = []
            for q in queries:
                for d in retriever.invoke(q):
                    key = (d.metadata.get("source_path"), d.metadata.get("page"))
                    if key in seen:
                        continue
                    seen.add(key)
                    gathered.append(d)
                    if len(gathered) >= 6:
                        break
                if len(gathered) >= 6:
                    break
            docs_list = gathered
        except Exception:
            docs_list = []
        book_snippets = _dedupe_snippets(docs_list, max_items=5, max_chars=600)
    JSON_EXAMPLE = TACTICS_JSON_EXAMPLE
    # 4) Prompt: pedimos Markdown + JSON
    prompt = f"""{directive}
You are an expert software architect applying Attribute-Driven Design 3.0 (ADD 3.0).

We ALREADY HAVE an ASR / Quality Attribute Scenario. That ASR is an ADD 3.0 architectural driver.
Your job now is to continue the ADD 3.0 process by selecting architectural tactics.

PROJECT CONTEXT (if any)
{ctx or "None"}

ASR (driver to satisfy):
{asr_text or "(none provided)"}

Primary quality attribute (guessed):
{qa}
Selected architecture style (if any):
{style_text or "(none)"}


GROUNDING (use ONLY this context; if DOC-ONLY, this is the exclusive source):
{book_snippets or "(none)"}

If DOC-ONLY is ON, do not rely on knowledge beyond the PROJECT DOCUMENT even if you “know” typical tactics. If the document does not support a tactic, state “not supported by the document”.

You MUST output THREE sections, in EXACT order:

(0) Which is the ASR and it´s style (if any):
- 3–5 concise lines.
- Explicitly link back to the ASR's Source, Stimulus, Artifact, Environment and Response Measure. Also its architectonic style. Example: "The external clients and bots, when a 10x traffic burst during product drop, (checkout API) in a normal operation and one region, the system must keep throughput and protect downstreams with a response measure of p95 < 200ms and error rate < 0.5% using microservices with API gateway and Redis cache."

(1) TACTICS (TOP-3 with highest success probability):
Select EXACTLY THREE architectural tactics that maximally satisfy this ASR GIVEN the selected style.
For EACH tactic include:
- Name — canonical tactic name (e.g., "Elastic Horizontal Scaling", "Cache-Aside + TTL", "Circuit Breaker").
- Rationale — why THIS tactic directly satisfies THIS ASR's Response & Response Measure in THIS style.
- Consequences / Trade-offs — realistic costs/risks (cost, complexity, ops burden, coupling, failure modes).
- When to use — explicit runtime trigger/guard (e.g., "if p95 > 200ms during 10x burst for 1 minute, trigger X").
- Why it ranks in TOP-3 — short argument grounded on ASR + style fit.
- Sucess probability — numeric estimate [0,1] of success in production.

(2) JSON:
Return ONE code fence starting with ```json and ending with ``` that contains ONLY a JSON array with EXACTLY 3 objects.
- Use dot as decimal separator (e.g., 0.82), never commas.
- Do not use percent signs, just 0..1 floats for success_probability.
- Do not add any prose or markdown outside the JSON fence.

Example shape (values are illustrative — adjust to your tactics):
{JSON_EXAMPLE}


STRICT RULES:
- You MUST behave like ADD 3.0: tactics are chosen BECAUSE OF the ASR's Response and Response Measure, not randomly.
- Every tactic MUST explicitly tie back to the ASR driver.
- DO NOT invent product names or vendor SKUs. Stay pattern-level.
- Keep output concise, production-realistic, and auditable.
- Output EXACTLY 3 tactics — do not list more than 3.
- Provide a numeric "success_probability" in [0,1] and a unique "rank" (1..3) consistent with the markdown ranking.
"""
    resp = llm.invoke(prompt)
    raw = getattr(resp, "content", str(resp)).strip()

    # LOG opcional (útil para depurar)
    log.debug("tactics raw (first 400): %s", raw[:400].replace("\n"," "))
    log.debug("has ```json fence? %s", bool(re.search(r"```json", raw, re.I)))

    # 5) Parseo + reparación en cascada (solo helpers existentes)
    struct = extract_json_array(raw) or []

    if not (isinstance(struct, list) and struct):
        struct = _json_only_repair_pass(
            llm, asr_text=asr_text, qa=qa, style_text=style_text, md_preview=raw
        ) or []

    if not (isinstance(struct, list) and struct):
        struct = build_json_from_markdown(raw, top_n=3)

    # Normaliza a TOP-3 + shape final
    struct = normalize_tactics_json(struct, top_n=3)
    log.info(
        "tactics_struct.len=%s names=%s",
        len(struct) if isinstance(struct, list) else 0,
        [it.get("name") for it in (struct or []) if isinstance(it, dict)]
    )

        
    # Markdown a mostrar (remueve el primer bloque ```json del modelo si vino)
    md_only = strip_first_json_fence(raw)

    show_json = os.getenv("SHOW_TACTICS_JSON", "0") == "1"
    if show_json:
        md_only = f"{md_only}\n\n```json\n{json.dumps(struct, ensure_ascii=False, indent=2)}\n```"
    else:
        # si ocultas el JSON, borra el encabezado "(2) JSON:" que queda colgando
        md_only = re.sub(r"\n?\(?2\)?\s*JSON\s*:?\s*$", "", md_only, flags=re.I|re.M).rstrip()

    # Fallback visual si por alguna razón no hay markdown
    if (not md_only) and isinstance(struct, list) and struct:
        md_only = "\n".join(
            f"- {it.get('name','')}: {it.get('rationale','')}"
            for it in struct if isinstance(it, dict)
        )

    # 6) Fuentes
    src_lines = []
    for d in (docs_list or []):
        md = d.metadata or {}
        title = md.get("source_title") or md.get("title") or "doc"
        page  = md.get("page_label") or md.get("page")
        path  = md.get("source_path") or md.get("source") or ""
        page_str = f" (p.{page})" if page is not None else ""
        src_lines.append(f"- {title}{page_str} — {path}")
    if src_lines:
        src_lines = list(dict.fromkeys([_clip_text(s, 60) for s in src_lines]))[:6]
    src_block = "SOURCES:\n" + ("\n".join(src_lines) if src_lines else "- (no local sources)")

    # 7) Traza y memoria
    _push_turn(state, role="system", name="tactics_system", content=prompt)
    _push_turn(state, role="assistant", name="tactics_advisor", content=md_only)
    _push_turn(state, role="assistant", name="tactics_sources", content=src_block)

    msgs = [
        AIMessage(content=md_only, name="tactics_advisor"),
        AIMessage(content=src_block, name="tactics_sources")
    ]

    # 8) Persistimos en el estado
    state["tactics_md"] = md_only
    state["tactics_struct"] = struct if isinstance(struct, list) else []
    state["tactics_list"] = [ (it.get("name") or "").strip() for it in (struct or []) if isinstance(it, dict) and it.get("name") ]


    #Marca etapa ADD 3.0
    state["arch_stage"] = "TACTICS"        # ahora estamos en la fase de selección de tácticas ADD 3.0
    state["quality_attribute"] = qa        # refuerza cuál atributo estamos atacando
    state["current_asr"] = asr_text        # guarda el ASR que estas tácticas satisfacen

    # Señales para cortar en unifier
    state["endMessage"] = md_only
    state["intent"] = "tactics"
    state["nextNode"] = "unifier"
    prev_msgs = state.get("messages", [])
    return {**state, "messages": prev_msgs + msgs}
