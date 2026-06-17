# -*- coding: utf-8 -*-

import re
from langchain_core.messages import AIMessage

from src.graph.state import GraphState
from src.graph.resources import llm
from src.graph.consts import MARKDOWN_FORMAT_DIRECTIVE
from src.graph.prompts.mode_prompts import apply_mode_prompt
from src.graph.nodes.profile_shadow import fire_shadow_eval, should_trigger
from src.graph.utils import _push_turn, _strip_tactics_sections


def _length_directive(style_hint: str) -> str:
    """F13-T1: deriva la regla de longitud de la respuesta desde el token de
    verbosidad embebido en `user_style_hint` por `format_user_style_hint`
    ('LOW verbosity' / 'MEDIUM verbosity' / 'HIGH verbosity'). Sin preferencia
    o MEDIUM devuelve el texto histórico EXACTO, de modo que el comportamiento
    no cambia para la mayoría de usuarios sin preferencias configuradas."""
    h = (style_hint or "").upper()
    if "HIGH VERBOSITY" in h:
        return ("- Be expansive and thorough: elaborate with examples, context "
                "and rationale. There is NO length cap; prioritize completeness.")
    if "LOW VERBOSITY" in h:
        return "- Be very brief: key points only, minimal elaboration (about 3-5 lines)."
    # MEDIUM verbosity OR no preference -> unchanged historical text.
    return "- Keep it concise (6-12 lines of content)."


def _finalize_turn(state: GraphState, final_text: str) -> GraphState:
    """F3-T2: dispara shadow eval (fire-and-forget) si la cadencia se cumple
    y resetea `turn_count_since_eval`. Devuelve el state actualizado para
    que el caller lo incluya en el dict de retorno."""
    if should_trigger(state.get("turn_count_since_eval", 0) or 0):
        fire_shadow_eval(state, final_text=final_text)
        return {**state, "turn_count_since_eval": 0}
    return state

def _last_ai_by(state: GraphState, name: str) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and getattr(m, "name", None) == name and m.content:
            return m.content
    return ""

def _last_turn_by(state: GraphState, name: str) -> str:
    for m in reversed(state.get("turn_messages", [])):
        if isinstance(m, dict) and m.get("name") == name and m.get("content"):
            return str(m.get("content"))
    return ""

def _strip_mermaid_artifacts(text: str) -> str:
    """Remove accidental Mermaid diagram syntax that the LLM might produce."""
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


def _ensure_section(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    if re.match(r"^##\s+", body):
        return body
    return f"{title}\n\n{body}"


def _render_diagram_block(state: GraphState, lang: str) -> str:
    """Build the diagram Markdown block for both single- and multi-intent paths.

    Returns an embedded SVG image block when diagram data is available, or a
    clear failure message so the user is never left with a false promise.
    """
    d = state.get("diagram") or {}
    if not (d.get("ok") and d.get("svg_b64")):
        return (
            "## Diagrama\n\nNo se pudo renderizar el diagrama esta vez."
            if lang == "es"
            else "## Diagram\n\nThe diagram could not be rendered this time."
        )
    data_url = f'data:image/svg+xml;base64,{d["svg_b64"]}'
    head = "## Diagrama" if lang == "es" else "## Diagram"
    return f"{head}\n\n![diagram]({data_url})"

async def unifier_node(state: GraphState) -> GraphState:
    lang = state.get("language", "es")
    intent = state.get("intent", "general")
    style_hint = (state.get("user_style_hint") or "").strip()
    proj_ctx = (state.get("project_context_text") or "").strip()

    requested_nodes = list(state.get("requested_nodes", []) or [])
    requested_set = set(requested_nodes)

    # Caso compuesto: si el usuario pidió varias salidas explícitas, consolidarlas juntas.
    # F13-T1 (trade-off documentado): esta rama NO invoca al LLM — ensambla
    # bloques ya generados por los nodos worker (asr/style/tactics/tech), que
    # SÍ inyectan `user_style_hint` en su propio directive al generarlos. Por
    # eso la preferencia ya está reflejada en el contenido. No se añade una
    # pasada LLM extra aquí por costo/latencia y para no alterar el formato
    # multi-output determinista; bloques cacheados de turnos previos pueden
    # reflejar el estilo del turno en que se generaron (limitación aceptada).
    if len(requested_set) >= 2:
        asr_txt = (
            _last_ai_by(state, "asr_recommender")
            or state.get("last_asr")
            or state.get("current_asr")
            or ""
        ).strip()
        asr_txt = _strip_tactics_sections(asr_txt) if asr_txt else ""

        style_txt = (
            _last_ai_by(state, "style_recommender")
            or _last_turn_by(state, "style_recommender")
            or ""
        ).strip()

        tactics_txt = (
            state.get("tactics_md")
            or _last_ai_by(state, "tactics_advisor")
            or _last_turn_by(state, "tactics_advisor")
            or ""
        ).strip()

        tech_txt = (
            _last_ai_by(state, "tech_advisor")
            or _last_turn_by(state, "tech_advisor")
            or ""
        ).strip()

        blocks = []
        if lang == "es":
            if asr_txt and "asr" in requested_set:
                blocks.append(_ensure_section("## ASR", asr_txt))
            if style_txt and "style" in requested_set:
                blocks.append(_ensure_section("## Estilos Arquitectónicos", style_txt))
            if tactics_txt and "tactics" in requested_set:
                blocks.append(_ensure_section("## Tácticas", tactics_txt))
            if tech_txt and "tech" in requested_set:
                blocks.append(_ensure_section("## Tecnologías", tech_txt))
            if "diagram_agent" in requested_set:
                blocks.append(_render_diagram_block(state, lang))
            followups = [
                "Refinar el ASR con métricas más estrictas.",
                "Aterrizar estas tácticas en un plan de implementación por fases.",
            ]
        else:
            if asr_txt and "asr" in requested_set:
                blocks.append(_ensure_section("## ASR", asr_txt))
            if style_txt and "style" in requested_set:
                blocks.append(_ensure_section("## Architecture Styles", style_txt))
            if tactics_txt and "tactics" in requested_set:
                blocks.append(_ensure_section("## Tactics", tactics_txt))
            if tech_txt and "tech" in requested_set:
                blocks.append(_ensure_section("## Technologies", tech_txt))
            if "diagram_agent" in requested_set:
                blocks.append(_render_diagram_block(state, lang))
            followups = [
                "Refine the ASR with stricter metrics.",
                "Turn these tactics into a phased implementation plan.",
            ]

        if blocks:
            end_text = "\n\n".join(blocks).strip()
            state["suggestions"] = followups
            state["turn_messages"] = state.get("turn_messages", []) + [
                {"role": "assistant", "name": "unifier", "content": end_text}
            ]
            state = _finalize_turn(state, end_text)
            return {**state, "endMessage": end_text, "intent": ("diagram" if "diagram_agent" in requested_set else intent)}

    # 0) Show rendered diagram if available
    # 0) Mostrar el diagrama si existe (intención "diagram") - LÓGICA ANTIGUA, LA MANTENEMOS
    d = state.get("diagram") or {}
    if d.get("ok") and d.get("svg_b64"):
        data_url = f'data:image/svg+xml;base64,{d["svg_b64"]}'
        if lang == "es":
            head = "## Diagrama"
            footer = "El diagrama ha sido generado. ¿Qué deseas hacer ahora?"
            tips = [
                "Regenerar el diagrama con ajustes.",
                "Continuar a la siguiente fase (tácticas arquitectónicas).",
                "Hacer una pregunta sobre el diagrama.",
            ]
        else:
            head = "## Diagram"
            footer = "The diagram has been generated. What would you like to do now?"
            tips = [
                "Regenerate the diagram with adjustments.",
                "Continue to the next phase (architectural tactics).",
                "Ask a question about the diagram.",
            ]

        end_text = f"""{head}

![diagram]({data_url})

{footer}
"""
        state["suggestions"] = tips
        state = _finalize_turn(state, end_text)
        return {**state, "endMessage": end_text, "intent": "diagram"}

    if intent == "intake":
        state = _finalize_turn(state, state.get("endMessage") or "")
        return {**state, "endMessage": state.get("endMessage") or ""}

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
        state = _finalize_turn(state, style_txt)
        return {**state, "endMessage": style_txt}

    # ðŸ"´ Caso especial para TÁCTICAS
    if intent == "tactics":
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
            refs_label = "### Referencias"
        else:
            followups = [
                "Generate a component diagram applying these tactics.",
                "Generate a deployment diagram aligned with these tactics.",
            ]
            refs_label = "### References"

        end_text = f"{tactics_md}\n\n---\n\n{refs_label}\n\n{refs_block}"

        state["suggestions"] = followups
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        state = _finalize_turn(state, end_text)
        return {**state, "endMessage": end_text}

    # 🔴 Caso especial para TECNOLOGÍAS
    if intent == "tech":
        advisor_md = _last_ai_by(state, "tech_advisor")
        end_msg    = state.get("endMessage") or ""
        tech_md    = advisor_md or end_msg

        src_txt = _last_ai_by(state, "tech_sources")
        refs_extracted = _extract_rag_sources_from(src_txt) if src_txt else ""
        has_refs = bool(refs_extracted and refs_extracted.strip() and refs_extracted.strip().lower() != "none")

        if lang == "es":
            followups = [
                "Genera un diagrama de componentes con estas tecnologías.",
                "Genera un diagrama de despliegue alineado con estas tecnologías.",
            ]
            refs_label = "### Referencias"
        else:
            followups = [
                "Generate a component diagram with these technologies.",
                "Generate a deployment diagram aligned with these technologies.",
            ]
            refs_label = "### References"

        # F20-T1: el caso degenerado (tech_advisor vacío Y endMessage vacío)
        # antes producía un mensaje literal "No technology content." + bloque
        # "Referencias / None" que el usuario veía como respuesta del agente.
        # Eso ocurre típicamente cuando el supervisor enruta una pregunta
        # educativa a intent=tech pero el nodo `tech_advisor` no aportó
        # contenido (e.g. faltan tácticas confirmadas). Ahora producimos un
        # mensaje útil que redirige la conversación.
        if not tech_md.strip():
            if lang == "es":
                tech_md = (
                    "Necesito un poco más de contexto para recomendarte "
                    "tecnologías concretas.\n\n"
                    "Si quieres una explicación conceptual (qué es y un "
                    "ejemplo mínimo), dímelo y lo abordamos como pregunta "
                    "educativa.\n\n"
                    "Si lo que buscas es una decisión de stack, primero "
                    "definimos el atributo de calidad y las tácticas que "
                    "guían la elección."
                )
                followups = [
                    "Explícame el concepto y dame un ejemplo mínimo.",
                    "Definamos primero el atributo de calidad y las tácticas.",
                ]
            else:
                tech_md = (
                    "I need a bit more context before recommending concrete "
                    "technologies.\n\n"
                    "If you want a conceptual explanation (what it is + a "
                    "minimal example), tell me and I'll treat this as an "
                    "educational question.\n\n"
                    "If you want a stack decision, we first define the "
                    "quality attribute and the tactics that drive the choice."
                )
                followups = [
                    "Explain the concept and give me a minimal example.",
                    "Let's first define the quality attribute and tactics.",
                ]

        # F20-T1: omitir el bloque "Referencias" cuando no hay fuentes — antes
        # se emitía "### Referencias\n\nNone" que es ruido visual.
        if has_refs:
            end_text = f"{tech_md}\n\n---\n\n{refs_label}\n\n{refs_extracted}"
        else:
            end_text = tech_md

        state["suggestions"] = followups
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        state = _finalize_turn(state, end_text)
        return {**state, "endMessage": end_text}

    if intent == "asr_confirm":
        end_text = state.get("endMessage") or ""
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        state["suggestions"] = state.get("suggestions") or []
        state = _finalize_turn(state, end_text)
        return {**state, "endMessage": end_text}

    # BUG-054 / BUG-055: same pass-through as asr_confirm — the confirmation
    # node already produced the user-facing message in `endMessage`.
    if intent == "style_confirm":
        end_text = state.get("endMessage") or ""
        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        state["suggestions"] = state.get("suggestions") or []
        state = _finalize_turn(state, end_text)
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
            refs_label = "### Referencias"
        else:
            followups = [
                "Propose architecture styles for this ASR.",
                "Refine this ASR with more specific metrics and scenarios.",
            ]
            refs_label = "### References"

        end_text = f"{last_asr}\n\n---\n\n{refs_label}\n\n{refs_block}"

        state["turn_messages"] = state.get("turn_messages", []) + [
            {"role": "assistant", "name": "unifier", "content": end_text}
        ]
        state["suggestions"] = followups
        state = _finalize_turn(state, end_text)
        return {**state, "endMessage": end_text}

    # ðŸ"´ Caso especial: saludo / smalltalk
    if intent in ("greeting", "smalltalk"):
        if lang == "es":
            hello = "## Bienvenido a ArchIA\n\n¡Hola! ¿Sobre qué tema de arquitectura quieres profundizar?"
            nexts = [
                "Formular un ASR (requerimiento de calidad) para mi sistema.",
                "Revisar un ASR que ya tengo.",
            ]
            footer = (
                "> Si quieres, podemos empezar el ciclo **ADD 3.0** formulando "
                "un ASR (por ejemplo de *latencia*, *disponibilidad* o *seguridad*)."
            )
        else:
            hello = "## Welcome to ArchIA\n\nHi! What software-architecture topic would you like to explore?"
            nexts = [
                "Define an ASR (quality attribute requirement) for my system.",
                "Review an ASR I already have.",
            ]
            footer = (
                "> If you want, we can start the **ADD 3.0** cycle by defining "
                "an ASR (for example *latency*, *availability* or *security*)."
            )

        end_text = hello + "\n\n" + footer
        redirect = (state.get("phase_redirect_hint") or "").strip()
        if redirect:
            end_text = end_text + "\n\n" + redirect
        state["suggestions"] = nexts
        state = _finalize_turn(state, end_text)
        return {**state, "endMessage": end_text}

    # ðŸ"µ Caso por defecto: síntesis de investigador / evaluador / etc.
    researcher_txt = _last_ai_by(state, "researcher")
    evaluator_txt = _last_ai_by(state, "evaluator")
    asr_src_txt = _last_ai_by(state, "asr_sources")

    rag_refs = ""
    if researcher_txt:
        rag_refs = _extract_rag_sources_from(researcher_txt) or ""

    memory_hint = state.get("memory_text", "")

    buckets = []
    if researcher_txt:
        buckets.append(f"researcher:\n{researcher_txt}")
    if evaluator_txt:
        buckets.append(f"evaluator:\n{evaluator_txt}")
    if asr_src_txt:
        buckets.append(f"asr_sources:\n{asr_src_txt}")

    synthesis_source = (
        "User question:\n"
        + (state.get("userQuestion", ""))
        + "\n\n"
        + "\n\n".join(buckets)
    )

    if lang == "es":
        directive = (
            "IDIOMA OBLIGATORIO: español.\n"
            "Tu respuesta COMPLETA debe estar en español. No mezcles idiomas."
        )
    else:
        directive = (
            "MANDATORY LANGUAGE: English.\n"
            "Your ENTIRE response MUST be in English. Do not mix languages."
        )
    # F13-T1: la verbosidad gobierna la longitud (antes hardcodeada a 6-12
    # líneas, lo que anulaba verbosity=HIGH). Sin preferencia → texto histórico.
    length_rule = _length_directive(style_hint)

    # F13-T1: el estilo de comunicación es una preferencia EXPLÍCITA del
    # usuario; se enmarca como bloque OBLIGATORIO (espejo de PROJECT CONTEXT),
    # no como una línea suelta anexada al directive que el LLM desprioriza.
    style_block = ""
    if style_hint:
        if lang == "es":
            style_block = (
                "\n=== ESTILO DE COMUNICACIÓN (OBLIGATORIO) ===\n"
                "Preferencia explícita del usuario. Cúmplela en TODA la "
                "respuesta; tiene prioridad sobre el estilo por defecto:\n"
                f"{style_hint}\n"
                "=== FIN ESTILO DE COMUNICACIÓN ===\n"
            )
        else:
            style_block = (
                "\n=== COMMUNICATION STYLE (MANDATORY) ===\n"
                "Explicit user preference. Honor it across the ENTIRE "
                "response; it takes precedence over the default style:\n"
                f"{style_hint}\n"
                "=== END COMMUNICATION STYLE ===\n"
            )

    proj_ctx_block = ""
    if proj_ctx:
        proj_ctx_block = (
            f"\nPROJECT CONTEXT (use this to ground your answer to the actual tech stack and business rules):\n"
            f"{proj_ctx}\n"
        )

    prompt = f"""{directive}
You are writing the FINAL chat reply.

- Give a complete, direct solution tailored to the question and context.
- Use Markdown formatting: ## for sections, **bold** for key terms, - for lists.
{length_rule}
- If useful, at the end include a '### References' section listing 3-6 items from RAG_SOURCES (one per line). If not useful, you may omit it.

Constraints:
- Use the user's language.
- Do not invent sources outside RAG_SOURCES.
- If a PROJECT CONTEXT is provided, your answer MUST reference the specific technologies and respect the business rules.
{MARKDOWN_FORMAT_DIRECTIVE}

Conversation memory (for continuity): {memory_hint}
{proj_ctx_block}{style_block}
RAG_SOURCES:
{rag_refs}

SOURCE:
{synthesis_source}

{"RECORDATORIO FINAL: toda tu respuesta debe estar en español." if lang == "es" else "FINAL REMINDER: your entire response must be in English."}
"""

    resp = await llm.ainvoke(apply_mode_prompt(state, prompt))
    final_text = getattr(resp, "content", str(resp))
    final_text = _strip_mermaid_artifacts(final_text)
    redirect = (state.get("phase_redirect_hint") or "").strip()
    if redirect:
        final_text = final_text + "\n\n" + redirect

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

    state = _finalize_turn(state, final_text)
    return {**state, "endMessage": final_text}

