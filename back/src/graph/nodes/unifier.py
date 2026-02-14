
import re
from langchain_core.messages import AIMessage

from src.graph.state import GraphState
from src.graph.resources import llm
from src.graph.utils import _push_turn, _strip_tactics_sections, _resolve_response_language, _sync_contract_from_legacy, _sync_legacy_from_contract, _build_context_summary

def _last_ai_by(state: GraphState, name: str) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and getattr(m, "name", None) == name and m.content:
            return m.content
    return ""

def _strip_all_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"^\s*#.*$", "", text, flags=re.M)
    text = text.replace("**", "")
    out = []
    for ln in text.splitlines():
        if re.search(r"^\s*(graph\s+(LR|TB)|flowchart|sequenceDiagram|classDiagram)\b", ln, re.I):
            continue
        if re.match(r"^\s*[A-Za-z0-9_-]+\s*--?[>-]", ln):
            continue
        out.append(ln)
    return "\n".join(out).strip()

def _extract_rag_sources_from(text: str) -> str:
    m = re.search(r"SOURCES:\s*(.+)$", text, flags=re.S | re.I)
    if not m:
        return ""
    raw = m.group(1)
    lines = []
    for ln in raw.splitlines():
        ln = ln.strip(" -\t")
        if ln:
            lines.append(ln)
    return "\n".join(lines[:8])

def _split_sections(text: str) -> dict:
    sections = {"Answer": "", "References": "", "Next": ""}
    current = None
    for ln in text.splitlines():
        if re.match(r"^Answer:", ln, re.I):
            current = "Answer"; sections[current] = ln.split(":", 1)[1].strip(); continue
        if re.match(r"^References:", ln, re.I):
            current = "References"; sections[current] = ln.split(":", 1)[1].strip(); continue
        if re.match(r"^Next:", ln, re.I):
            current = "Next"; sections[current] = ln.split(":", 1)[1].strip(); continue
        if current:
            sections[current] += ("\n" + ln)
    for k in sections:
        sections[k] = sections[k].strip()
    return sections

def unifier_node(state: GraphState) -> GraphState:
    _sync_contract_from_legacy(state)
    lang = _resolve_response_language(state)
    intent = state.get("intent", "general")
    state["response_language"] = lang
    state["language"] = lang

    # Guardrail: if pending context questions exist, ask for the missing data first.
    pending = state.get("pending_questions", []) or []
    if pending:
        if lang == "es":
            end_text = "Para continuar con precision, necesito este dato:\n- " + "\n- ".join(pending[:2])
        else:
            end_text = "To continue accurately, I need this input:\n- " + "\n- ".join(pending[:2])
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        return {**state, "endMessage": end_text}

    turn_ctx = state.get("turn_context") or {}
    requested = turn_ctx.get("requested_outputs") or []
    diagram_requested = bool(state.get("intent") == "diagram" or (isinstance(requested, list) and "diagram" in requested))

    # 🟣 Caso especial SOLO cuando el turno pide diagrama
    mermaid = (state.get("mermaidCode") or "").strip()
    if mermaid and diagram_requested:
        if lang == "es":
            head = (
                "Aquí tienes el diagrama solicitado, generado a partir de tu "
                "escenario de calidad (ASR), el estilo arquitectónico seleccionado "
                "y las tácticas priorizadas.\n\n"
                "Puedes copiar y pegar este script Mermaid en tu editor preferido "
                "(por ejemplo, mermaid.live o un plugin de VS Code):\n"
            )
            footer = ""
            suggestions = [
                "Formular un nuevo ASR para otro escenario de calidad.",
                "Generar un diagrama de componentes a partir de este sistema.",
                "Generar un diagrama de despliegue para este mismo sistema.",
            ]
        else:
            head = (
                "Here is the requested diagram, generated from your quality "
                "scenario (ASR), the selected architectural style and the "
                "prioritized tactics.\n\n"
                "You can copy & paste this Mermaid script into your favorite "
                "editor (for example, mermaid.live or a VS Code Mermaid plugin):\n"
            )
            footer = ""
            suggestions = [
                "Generate a new ASR for another quality scenario.",
                "Generate a component diagram from this system.",
                "Generate a deployment diagram for this same system.",
            ]
        
        # Dentro del unifier, rama "if intent == 'diagram' and state.get('mermaidCode')"
        mermaid = state.get("mermaidCode") or ""
        if lang == "es":
            head = "Aquí tienes el diagrama generado a partir del ASR, el estilo y las tácticas seleccionadas."
            footer = ""
            suggestions = [
                "Formular un nuevo ASR para otro escenario de calidad.",
                "Generar un diagrama de componentes a partir de este sistema.",
                "Generar un diagrama de despliegue para este mismo sistema.",
            ]
        else:
            head = (
                "Here is the diagram generated from your quality scenario (ASR), "
                "the selected architecture style and the prioritized tactics."
            )
            footer = ""
            suggestions = [
                "Generate a new ASR for another quality scenario.",
                "Generate a component diagram from this system.",
                "Generate a deployment diagram for this same system.",
            ]

        end_text = head  # 👈 ya NO incluimos el código mermaid en el texto

        state["suggestions"] = suggestions
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        return {**state, "endMessage": end_text, "intent": "diagram"}

    # 0) Mostrar el diagrama si existe (intención "diagram") - LÓGICA ANTIGUA, LA MANTENEMOS
    d = state.get("diagram") or {}
    if d.get("ok") and d.get("svg_b64"):
        data_url = f'data:image/svg+xml;base64,{d["svg_b64"]}'
        if lang == "es":
            head = "Aquí tienes el diagrama solicitado:"
            footer = "¿Qué te gustaría hacer ahora con este diagrama?"
            tips = [
                "Generar un diagrama de componentes a partir de este sistema.",
                "Generar un diagrama de despliegue para este mismo sistema.",
                "Formular un nuevo ASR basado en este sistema.",
            ]
        else:
            head = "Here is your requested diagram:"
            footer = "What would you like to do next with this diagram?"
            tips = [
                "Generate a component diagram from this system.",
                "Generate a deployment diagram for this same system.",
                "Define a new ASR based on this system.",
            ]

        end_text = f"""{head}
![diagram]({data_url})

{footer}
"""
        state["suggestions"] = tips
        return {**state, "endMessage": end_text, "intent": "diagram"}

    # 🔴 Caso especial para ESTILOS
    if intent == "style":
        style_txt = (
            _last_ai_by(state, "style_recommender")
            or state.get("endMessage")
            or "No style content."
        )

        if lang == "es":
            followups = state.get("suggestions") or [
                "Diseña tácticas concretas para este ASR usando el estilo recomendado.",
                "Compárame más a fondo estos dos estilos para este ASR.",
            ]
        else:
            followups = state.get("suggestions") or [
                "Explain concrete tactics for this ASR using the recommended style.",
                "Compare these two styles in more depth for this ASR.",
            ]

        state["suggestions"] = followups
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": style_txt}
        ]
        return {**state, "endMessage": style_txt}

    # 🔴 Caso especial para TÁCTICAS
    if intent == "tactics":
        has_asr = bool((state.get("active_decisions") or {}).get("current_asr") or state.get("current_asr") or state.get("last_asr"))
        if not has_asr:
            end_text = (
                "Necesito el ASR actual para proponer tacticas trazables. Puedes pegarlo o pedirme que lo redacte primero."
                if lang == "es"
                else "I need the current ASR to propose traceable tactics. You can paste it or ask me to draft it first."
            )
            state["turn_messages"] = state.get("turn_messages", []) + [
                {"role": "assistant", "name": "unifier", "content": end_text}
            ]
            return {**state, "endMessage": end_text}

        tactics_md = (
            state.get("tactics_md")
            or _last_ai_by(state, "tactics_advisor")
            or "No tactics content."
        )
        src_txt = _last_ai_by(state, "tactics_sources")
        refs_block = _extract_rag_sources_from(src_txt) if src_txt else "None"

        if lang == "es":
            followups = [
                "Genera un diagrama de componentes aplicando estas tácticas.",
                "Genera un diagrama de despliegue alineado con estas tácticas.",
            ]
            refs_label = "Referencias"
        else:
            followups = [
                "Generate a component diagram applying these tactics.",
                "Generate a deployment diagram aligned with these tactics.",
            ]
            refs_label = "References"

        end_text = f"{tactics_md}\n\n{refs_label}:\n{refs_block}"

        state["suggestions"] = followups
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        return {**state, "endMessage": end_text}

    # 🔴 Caso especial para ASR
    if intent == "asr" or intent == "ASR":
        raw_asr = (
            _last_ai_by(state, "asr_recommender")
            or state.get("endMessage")
            or "No ASR content found for this turn."
        )
        # si el LLM coló tácticas, las quitamos del ASR
        last_asr = _strip_tactics_sections(raw_asr)

        asr_src_txt = _last_ai_by(state, "asr_sources")
        refs_block = _extract_rag_sources_from(asr_src_txt) if asr_src_txt else "None"

        if lang == "es":
            followups = [
                "Propón estilos arquitectónicos para este ASR.",
                "Refina este ASR con métricas y escenarios más específicos.",
            ]
            refs_label = "Referencias"
        else:
            followups = [
                "Propose architecture styles for this ASR.",
                "Refine this ASR with more specific metrics and scenarios.",
            ]
            refs_label = "References"

        end_text = f"{last_asr}\n\n{refs_label}:\n{refs_block}"

        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        state["suggestions"] = followups
        return {**state, "endMessage": end_text}

    # 🔴 Caso especial: saludo / smalltalk
    if intent in ("greeting", "smalltalk"):
        if lang == "es":
            hello = "¡Hola! ¿Sobre qué tema de arquitectura quieres profundizar?"
            nexts = [
                "Formular un ASR (requerimiento de calidad) para mi sistema.",
                "Revisar un ASR que ya tengo.",
            ]
            footer = (
                "Si quieres, podemos empezar el ciclo ADD 3.0 formulando "
                "un ASR (por ejemplo de latencia, disponibilidad o seguridad)."
            )
        else:
            hello = "Hi! What software-architecture topic would you like to explore?"
            nexts = [
                "Define an ASR (quality attribute requirement) for my system.",
                "Review an ASR I already have.",
            ]
            footer = (
                "If you want, we can start the ADD 3.0 cycle by defining "
                "an ASR (for example latency, availability or security)."
            )

        end_text = hello + "\n\n" + footer
        state["suggestions"] = nexts
        return {**state, "endMessage": end_text}

    # 🔵 Caso por defecto: síntesis de investigador / evaluador / etc.
    researcher_txt = _last_ai_by(state, "researcher")
    evaluator_txt = _last_ai_by(state, "evaluator")
    creator_txt = _last_ai_by(state, "creator")
    asr_src_txt = _last_ai_by(state, "asr_sources")

    rag_refs = ""
    if researcher_txt:
        rag_refs = _extract_rag_sources_from(researcher_txt) or ""

    state["context_summary"] = _build_context_summary(state)
    memory_hint = state.get("context_summary") or state.get("memory_text", "")

    buckets = []
    if researcher_txt:
        buckets.append(f"researcher:\n{researcher_txt}")
    if evaluator_txt:
        buckets.append(f"evaluator:\n{evaluator_txt}")
    if creator_txt and intent == "diagram":
        buckets.append(f"creator:\n{creator_txt}")
    if asr_src_txt:
        buckets.append(f"asr_sources:\n{asr_src_txt}")

    synthesis_source = (
        "User question:\n"
        + (state.get("userQuestion", ""))
        + "\n\n"
        + "\n\n".join(buckets)
    )

    directive = "Responde en español." if lang == "es" else "Answer in English."
    prompt = f"""{directive}
You are writing the FINAL chat reply.

- Give a complete, direct solution tailored to the question and context.
- Use 6–12 concise lines (bullets or short sentences). No code fences, no mermaid.
- If useful, at the end include a short 'References:' block listing 3–6 items from RAG_SOURCES (one per line). If not useful, you may omit it.

Constraints:
- Use the user's language.
- Do not invent sources outside RAG_SOURCES.
- Keep it clean: no '#', no '**', no code blocks.

Conversation memory (for continuity): {memory_hint}

RAG_SOURCES:
{rag_refs}

SOURCE:
{synthesis_source}
"""

    resp = llm.invoke(prompt)
    final_text = getattr(resp, "content", str(resp))
    final_text = _strip_all_markdown(final_text)

    secs = _split_sections(final_text)
    chips = []
    if secs.get("Next"):
        for ln in secs["Next"].splitlines():
            ln = ln.strip(" -•\t")
            if ln:
                chips.append(ln)
    state["suggestions"] = chips[:6] if chips else []

    _push_turn(state, role="system", name="unifier_system", content=prompt)
    _push_turn(state, role="assistant", name="unifier", content=final_text)
    _sync_legacy_from_contract(state)

    return {**state, "endMessage": final_text}
