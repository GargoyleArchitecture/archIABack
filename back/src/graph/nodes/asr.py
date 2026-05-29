# -*- coding: utf-8 -*-
import logging
import re
from datetime import datetime, timezone

from langchain_core.messages import AIMessage
from src.graph.resources import llm, rag_trace_record
from src.graph.state import GraphState
from src.graph.consts import MARKDOWN_FORMAT_DIRECTIVE
from src.graph.prompts.mode_prompts import apply_mode_prompt
from src.graph.utils import (
    _clip_text,
    _dedupe_snippets,
    is_explicit_asr_request,
    is_asr_regenerate_request,
    _sanitize_response,
    _strip_tactics_sections,
)
from src.rag_agent import get_indexed_retriever
from src.graph.qa_registry import normalize_qa, qa_to_focus_label
from src.ledger import (
    append_decision,
    load_ledger,
    transition_phase,
    LedgerValidationError,
    LedgerConcurrencyError,
)
from src.ledger.types import Phase, PhaseTransition
from src.graph.nodes._ledger_helpers import _refresh_ledger_state

log = logging.getLogger("asr_node")


# ---------------------------------------------------------------------------
# Compile-time regexes (Step 1 — P3)
# ---------------------------------------------------------------------------

_HISTORY_HEADING_RE = re.compile(
    r"^##\s+(?:History\s+\(superseded\s*/\s*rejected\)|"
    r"Historial\s+\(reemplazadas\s*/\s*rechazadas\))\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_ASR_HEADING_RE = re.compile(
    r"^##\s*ASR(?:\s+\d+)?\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_ASR_SUMMARY_RE = re.compile(r"\*\*ASR\s+complete\s*:\*\*\s*(.+)", re.IGNORECASE)

_ASR_FIELD_RE: dict[str, re.Pattern] = {
    "source":           re.compile(r"-\s*\*\*Source\s*:\*\*\s*(.+)",             re.IGNORECASE),
    "stimulus":         re.compile(r"-\s*\*\*Stimulus\s*:\*\*\s*(.+)",           re.IGNORECASE),
    "environment":      re.compile(r"-\s*\*\*Environment\s*:\*\*\s*(.+)",        re.IGNORECASE),
    "artifact":         re.compile(r"-\s*\*\*Artifact\s*:\*\*\s*(.+)",           re.IGNORECASE),
    "response":         re.compile(r"-\s*\*\*Response\s*:\*\*\s*(.+)",           re.IGNORECASE),
    "response_measure": re.compile(r"-\s*\*\*Response\s+Measure\s*:\*\*\s*(.+)", re.IGNORECASE),
}

_NONE_MARKER_RE = re.compile(r"_\((?:ninguna aún|none yet)\)_", re.IGNORECASE)

_RESPONSE_MEASURE_RE = re.compile(
    r"-\s*\*\*Response\s+Measure\s*:\*\*\s*(.+)", re.IGNORECASE
)

_RM_METRIC_RE = re.compile(
    r"(?P<metric>latenci[ay]|latency|p\d{1,2}|throughput|rps|tps|"
    r"disponibilidad|availability|error\s*rate|tasa\s*de\s*error|uptime|"
    r"usuarios?|users?|concurrent|concurrentes?|requests?)"
    r"\s*(?:[<>]=?|[=:≤≥])\s*"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|ms|s|rps|tps|k|m)?",
    re.IGNORECASE,
)

_RM_BARE_METRIC_RE = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>ms|rps|tps|%)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Baseline validation helpers (Punto 4 — ADD 3.0)
# ---------------------------------------------------------------------------

def _parse_response_measure_metrics(content: str) -> list[dict]:
    """Extract numeric metrics from the Response Measure field of an ASR."""
    m = _RESPONSE_MEASURE_RE.search(content)
    if not m:
        return []
    rm_text = m.group(1)
    metrics: list[dict] = []
    for match in _RM_METRIC_RE.finditer(rm_text):
        val_str = match.group("value").replace(",", ".")
        metrics.append({
            "metric": match.group("metric").lower().strip(),
            "value": float(val_str),
            "unit": (match.group("unit") or "").lower(),
        })
    if not metrics:
        for match in _RM_BARE_METRIC_RE.finditer(rm_text):
            val_str = match.group("value").replace(",", ".")
            unit = match.group("unit").lower()
            metric = "latency" if unit in ("ms", "s") else "throughput" if unit in ("rps", "tps") else "percent"
            metrics.append({"metric": metric, "value": float(val_str), "unit": unit})
    return metrics


def _is_within_normal_operation(asr_metrics: list[dict], baseline: dict) -> bool:
    """Return True if the ASR's response measure falls within normal operation.

    Logic: for latency-like metrics (lower is more demanding), the ASR is trivial
    if its threshold is >= the baseline (less demanding or equal).
    For throughput-like metrics (higher is more demanding), the ASR is trivial
    if its threshold is <= the baseline.
    """
    normal_load = baseline.get("normal_load") or []
    if not normal_load or not asr_metrics:
        return False

    _LATENCY_TERMS = {"latency", "latencia", "latenci", "p50", "p90", "p95", "p99", "p999"}
    _THROUGHPUT_TERMS = {"throughput", "rps", "tps", "requests", "request", "concurrent", "concurrentes", "concurrente", "usuarios", "usuario", "users", "user"}

    matched_any = False
    all_within = True

    for asr_m in asr_metrics:
        asr_metric = asr_m["metric"]
        asr_value = asr_m["value"]
        asr_unit = asr_m["unit"]

        for bl_m in normal_load:
            bl_metric = bl_m["metric"]
            bl_unit = bl_m["unit"]
            bl_value = bl_m["value"]

            if asr_unit and bl_unit and asr_unit != bl_unit:
                if asr_unit == "s" and bl_unit == "ms":
                    asr_value = asr_value * 1000
                elif asr_unit == "ms" and bl_unit == "s":
                    asr_value = asr_value / 1000
                else:
                    continue

            same_family = False
            if asr_metric in _LATENCY_TERMS and bl_metric in _LATENCY_TERMS:
                same_family = True
            elif asr_metric in _THROUGHPUT_TERMS and bl_metric in _THROUGHPUT_TERMS:
                same_family = True
            elif asr_metric == bl_metric:
                same_family = True

            if not same_family:
                continue

            matched_any = True
            if asr_metric in _LATENCY_TERMS or asr_unit in ("ms", "s"):
                if asr_value < bl_value:
                    all_within = False
            elif asr_metric in _THROUGHPUT_TERMS or asr_unit in ("rps", "tps"):
                if asr_value > bl_value:
                    all_within = False
            else:
                if asr_value > bl_value:
                    all_within = False

    return matched_any and all_within


def _format_baseline_for_prompt(baseline: dict, lang: str) -> str:
    """Format the baseline dict as a readable prompt section."""
    if not baseline.get("parsed"):
        return ""
    normal = baseline.get("normal_load") or []
    if not normal:
        return ""

    lines = []
    for m in normal:
        op = m.get("operator", "<")
        lines.append(f"  {m['metric']} {op} {m['value']}{m.get('unit', '')}")

    overload = baseline.get("overload") or []
    ov_lines = []
    for m in overload:
        op = m.get("operator", "<")
        ov_lines.append(f"  {m['metric']} {op} {m['value']}{m.get('unit', '')}")

    if lang == "en":
        header = "NORMAL OPERATION BASELINE (from architect's diagnosis):"
        section = f"\n{header}\n" + "\n".join(lines)
        if ov_lines:
            section += "\nOverload envelope:\n" + "\n".join(ov_lines)
        section += (
            "\n\nIMPORTANT: Do NOT propose ASRs whose Response Measure falls within "
            "the normal operation envelope above. An ASR must describe behavior BEYOND "
            "normal operation — stress conditions, failure modes, or peak scenarios.\n"
        )
    else:
        header = "BASELINE DE OPERACIÓN NORMAL (del diagnóstico del arquitecto):"
        section = f"\n{header}\n" + "\n".join(lines)
        if ov_lines:
            section += "\nEnvolvente de sobrecarga:\n" + "\n".join(ov_lines)
        section += (
            "\n\nIMPORTANTE: NO propongas ASRs cuya Medida de Respuesta caiga dentro "
            "de la envolvente de operación normal de arriba. Un ASR debe describir "
            "comportamiento FUERA de operación normal — condiciones de estrés, modos "
            "de fallo, o escenarios pico.\n"
        )
    return section


# ---------------------------------------------------------------------------
# Pure helpers (Step 1 — P3, not yet called by asr_node)
# ---------------------------------------------------------------------------

def _extract_dossier_history(dossier_md: str) -> str:
    """Return the history section from design_dossier_md, or '' if absent/empty."""
    if not dossier_md:
        return ""
    m = _HISTORY_HEADING_RE.search(dossier_md)
    if not m:
        return ""
    section = dossier_md[m.start():].strip()
    content_lines = [
        ln for ln in section.splitlines()
        if ln.strip()
        and not ln.strip().startswith("##")
        and not _NONE_MARKER_RE.fullmatch(ln.strip())
    ]
    return section if content_lines else ""


def _build_asr_payload(content: str, domain: str) -> dict:
    """Parse structured ASR markdown into a ledger payload dict."""
    m = _ASR_SUMMARY_RE.search(content)
    summary = m.group(1).strip() if m else _clip_text(content.strip(), 300)

    payload: dict = {
        "summary":          summary,
        "source":           "",
        "stimulus":         "",
        "environment":      "",
        "artifact":         "",
        "response":         "",
        "response_measure": "",
        "domain":           domain or "",
    }
    for field_key, pattern in _ASR_FIELD_RE.items():
        fm = pattern.search(content)
        if fm:
            payload[field_key] = fm.group(1).strip()
    return payload


def _build_sources_from_docs(docs_list: list) -> list[dict]:
    """Convert RAG Document objects to ledger source dicts (title, page, path)."""
    seen: set = set()
    result: list = []
    for d in docs_list or []:
        md = d.metadata or {}
        title = (md.get("source_title") or md.get("title") or "doc").strip()
        page  = md.get("page_label") or md.get("page")
        path  = (md.get("source_path") or md.get("source") or "").strip()
        key   = (title, str(page), path)
        if key in seen:
            continue
        seen.add(key)
        entry: dict = {"title": title, "path": path}
        if page is not None:
            entry["page"] = page
        result.append(entry)
        if len(result) >= 4:
            break
    return result


def _coerce_single_asr_markdown(content: str) -> str:
    """Normalize ASR output to a single ASR block.

    If the model emits `## ASR 1`, `## ASR 2`, etc., we keep only the first
    block and normalize its heading back to `## ASR`.
    """
    text = (content or "").strip()
    if not text:
        return text

    matches = list(_ASR_HEADING_RE.finditer(text))
    if matches:
        first = matches[0]
        end = matches[1].start() if len(matches) > 1 else len(text)
        text = text[first.start():end].strip()
        text = _ASR_HEADING_RE.sub("## ASR", text, count=1)
    return text


def _expand_asr_to_six_part(payload: dict, lang: str) -> tuple[dict, str]:
    """Expand a candidate ASR row into the canonical 6-part Markdown block.

    Returns (expanded_payload, markdown). The payload is the input merged with
    parsed source/stimulus/environment/artifact/response/response_measure so
    that downstream nodes (style, tactics) read structured fields from the
    ledger instead of just a one-line scenario.
    """
    candidate_id = str(payload.get("candidate_id") or "").upper() or "A?"
    qa           = str(payload.get("qa") or "").strip()
    scenario     = str(payload.get("scenario") or payload.get("summary") or "").strip()
    business     = str(payload.get("business") or "M").upper()[:1]
    risk         = str(payload.get("risk") or "M").upper()[:1]

    if lang == "es":
        prompt = (
            f"Expande el siguiente ASR a sus 6 partes canónicas (Bass/Clements/Kazman). "
            f"NO inventes información no implícita en el escenario; si un campo no se "
            f"puede derivar, escribe '(a confirmar)'. Preserva las medidas numéricas.\n\n"
            f"Cabecera: ASR {candidate_id} — {qa} (Importancia: {business}, Riesgo: {risk})\n"
            f"Escenario base: {scenario}\n\n"
            f"Produce EXACTAMENTE esta estructura Markdown (sin texto adicional):\n\n"
            f"## ASR {candidate_id} — {qa} (Importancia: {business}, Riesgo: {risk})\n\n"
            f"- **Source:** <quién/qué genera el estímulo>\n"
            f"- **Stimulus:** <el evento o condición>\n"
            f"- **Environment:** <condiciones — carga, fallo, mantenimiento>\n"
            f"- **Artifact:** <componente del sistema que responde>\n"
            f"- **Response:** <qué hace el sistema>\n"
            f"- **Response Measure:** <umbral medible del escenario base>\n\n"
            f"_Siguiente paso: seleccionar el **estilo arquitectónico** que mejor soporte este ASR._"
        )
        fallback_md = (
            f"## ASR {candidate_id} — {qa} (Importancia: {business}, Riesgo: {risk})\n\n"
            f"**Escenario:** {scenario}\n\n"
            f"_Siguiente paso: seleccionar el **estilo arquitectónico** que mejor soporte este ASR._"
        )
    else:
        prompt = (
            f"Expand the following ASR into its canonical 6 parts (Bass/Clements/Kazman). "
            f"DO NOT invent information not implied by the scenario; if a field cannot be "
            f"derived, write '(to be confirmed)'. Preserve numeric thresholds.\n\n"
            f"Header: ASR {candidate_id} — {qa} (Importance: {business}, Risk: {risk})\n"
            f"Base scenario: {scenario}\n\n"
            f"Produce EXACTLY this Markdown structure (no extra text):\n\n"
            f"## ASR {candidate_id} — {qa} (Importance: {business}, Risk: {risk})\n\n"
            f"- **Source:** <who/what triggers the stimulus>\n"
            f"- **Stimulus:** <the event or condition>\n"
            f"- **Environment:** <conditions — load, failure, maintenance>\n"
            f"- **Artifact:** <which system component responds>\n"
            f"- **Response:** <what the system does>\n"
            f"- **Response Measure:** <measurable threshold from the base scenario>\n\n"
            f"_Next step: select the **architecture style** that best supports this ASR._"
        )
        fallback_md = (
            f"## ASR {candidate_id} — {qa} (Importance: {business}, Risk: {risk})\n\n"
            f"**Scenario:** {scenario}\n\n"
            f"_Next step: select the **architecture style** that best supports this ASR._"
        )

    try:
        result = llm.invoke(prompt)
        raw = getattr(result, "content", str(result))
        md = _sanitize_response(raw) or fallback_md
    except Exception as exc:
        log.warning("asr expand: LLM call failed (%s); using deterministic fallback", exc)
        md = fallback_md

    expanded_payload = dict(payload)
    for field_key, pattern in _ASR_FIELD_RE.items():
        fm = pattern.search(md)
        if fm:
            val = fm.group(1).strip()
            if val and not _NONE_MARKER_RE.search(val):
                expanded_payload[field_key] = val
    if scenario:
        expanded_payload["summary"] = scenario
    return expanded_payload, md


# Candidate-table parser.
# A row looks like:
#   | A1 | Latencia | p95 ≤ 800ms con 600 CCU | H | H |
_ASR_TABLE_ROW_RE = re.compile(
    r"^\s*\|\s*(?P<id>A\d+)\s*\|\s*(?P<qa>[^|]+?)\s*\|\s*(?P<scenario>[^|]+?)\s*"
    r"\|\s*(?P<business>[HMLhml]|alta|media|baja|high|med|low)\s*"
    r"\|\s*(?P<risk>[HMLhml]|alta|media|baja|high|med|low)\s*\|?\s*$",
    re.MULTILINE,
)


def _parse_asr_table_rows(content: str) -> list[dict]:
    """Extract candidate rows from the ASR table the LLM produced.

    Returns a list of dicts: {id, qa, scenario, business, risk}.
    Rows are deduplicated by ID; the order returned matches the LLM output.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for m in _ASR_TABLE_ROW_RE.finditer(content or ""):
        rid = (m.group("id") or "").strip().upper()
        if not rid or rid in seen:
            continue
        seen.add(rid)
        rows.append({
            "id":       rid,
            "qa":       m.group("qa").strip(),
            "scenario": m.group("scenario").strip(),
            "business": m.group("business").strip().upper()[:1] or "M",
            "risk":     m.group("risk").strip().upper()[:1] or "M",
        })
    return rows


def asr_node(state: GraphState) -> GraphState:
    """Genera ASR y deja QA coherente para nodos siguientes (style/tactics)."""
    lang = state.get("language", "es")
    uq = state.get("userQuestion", "") or ""
    doc_only = bool(state.get("doc_only"))
    ctx_doc = (state.get("doc_context") or "").strip()
    existing_asr = (state.get("current_asr") or state.get("last_asr") or "").strip()
    explicit_asr_request = is_explicit_asr_request(uq)

    if existing_asr and not explicit_asr_request:
        log.info("asr_node: re-rendering existing ASR (no explicit request to change)")
        requested_nodes = [n for n in (state.get("requested_nodes") or []) if n != "asr"]
        pending_nodes = [n for n in (state.get("pending_nodes") or []) if n != "asr"]
        _done = list(state.get("completed_nodes") or [])
        if "asr" not in _done:
            _done.append("asr")
        return {
            **state,
            "requested_nodes": requested_nodes,
            "pending_nodes": pending_nodes,
            "endMessage": existing_asr,
            "nextNode": "unifier",
            "hasVisitedASR": True,
            "completed_nodes": _done,
        }

    # ── Regeneration: clear downstream state (P8) ─────────────────────────
    _is_regenerate = is_asr_regenerate_request(uq)
    if _is_regenerate:
        log.info("asr_node: regeneration requested — clearing downstream state")
        state["asr_candidates"] = []
        state["selected_asrs"] = []
        state["style_candidates"] = []
        state["selected_style"] = ""
        state["selected_tactics"] = []
        state["tactics_candidates"] = []
        state["tech_candidates"] = []
        state["current_phase"] = "asr_table"

    # ── Early context reads for precondition + domain derivation ─────────
    _proj_ctx_early = (state.get("project_context_text") or "").strip()
    _intake_v1_early = (
        (state.get("ledger") or {}).get("project_context", {}).get("intake_v1") or {}
    )

    # Hard precondition: refuse ASR generation when no domain context exists.
    # Both sources must be empty to trigger; either one is enough to proceed.
    if not _proj_ctx_early and not _intake_v1_early:
        _no_ctx_msg = (
            "Necesito completar el diagnóstico antes de generar ASRs. "
            "Por favor responde las preguntas del diagnóstico para que pueda "
            "generar ASRs relevantes para tu sistema."
            if lang == "es" else
            "I need to complete the diagnostic before generating ASRs. "
            "Please answer the diagnostic questions so I can generate "
            "ASRs relevant to your system."
        )
        log.warning("asr_node: no domain context — refusing to generate ASR")
        return {
            **state,
            "endMessage": _no_ctx_msg,
            "nextNode": "unifier",
            "current_phase": "diagnosis",
        }

    # Heurística del atributo
    concern = (
        "scalability"
        if re.search(r"scalab", uq, re.I)
        else "latency"
        if re.search(r"latenc", uq, re.I)
        else "performance"
    )

    # QA operativo para el pipeline: prioriza classifier (resolved_index) cuando exista.
    qa_from_classifier = normalize_qa(state.get("resolved_index", ""))
    qa_from_text = normalize_qa(concern)
    qa_pipeline = qa_from_classifier if qa_from_classifier != "general" else qa_from_text
    qa_focus = qa_to_focus_label(qa_pipeline, default=concern)

    # Domain derived from intake/project context — avoids hard-coded domain bias.
    # Priority: intake main requirement > project context header > uq keywords > neutral fallback.
    _intake_req = _intake_v1_early.get("campo_0_requerimiento", "").strip()
    low = uq.lower()
    if _intake_req:
        domain = _intake_req[:120]
    elif _proj_ctx_early:
        domain = _proj_ctx_early.split("\n")[0][:120]
    elif any(k in low for k in ["e-comm", "commerce", "shop", "checkout"]):
        domain = "e-commerce platform"
    elif "api" in low:
        domain = "public REST API"
    elif any(k in low for k in ["stream", "kafka"]):
        domain = "event streaming pipeline"
    else:
        domain = "general software system"

    # === RAG (saltable) ===
    docs_list = []
    if state.get("force_rag", False) and not doc_only:
        try:
            query = f"{qa_focus} quality attribute scenario latency measure stimulus environment artifact response response measure"
            _retriever = get_indexed_retriever(
                quality_attribute=(state.get("resolved_index") or qa_pipeline),
                content_type="asr",
                k=6,
            )
            docs_raw = list(_retriever.invoke(query))
            docs_list = docs_raw[:6]
        except Exception:
            docs_list = []

    book_snippets = _dedupe_snippets(docs_list, max_items=6, max_chars=800)

    if lang == "en":
        directive = (
            "MANDATORY LANGUAGE: English.\n"
            "Your ENTIRE response MUST be in English. Do not mix languages."
        )
    else:
        directive = (
            "IDIOMA OBLIGATORIO: español.\n"
            "Tu respuesta COMPLETA debe estar en español. No mezcles idiomas."
        )
    style_hint = (state.get("user_style_hint") or "").strip()
    if style_hint:
        directive = f"{directive}\n{style_hint}"

    ctx = (
        ctx_doc if (doc_only and ctx_doc) else (state.get("add_context") or "")
    ).strip()[:2000]
    proj_ctx = (state.get("project_context_text") or "").strip()
    # Mirror intake main requirement + components into proj_ctx when no project context exists.
    if not proj_ctx and _intake_v1_early:
        _mirror_parts = []
        for _mk in ("campo_0_requerimiento", "campo_1_alcance_funcional"):
            _mv = _intake_v1_early.get(_mk, "").strip()
            if _mv:
                _mirror_parts.append(_mv)
        if _mirror_parts:
            proj_ctx = "\n".join(_mirror_parts)[:500]

    # ── Intake context injection ───────────────────────────────────────────
    _intake_v1 = (state.get("ledger") or {}).get("project_context", {}).get("intake_v1") or {}
    if _intake_v1:
        _INTAKE_LABELS = {
            "campo_0_requerimiento": ("Requerimiento principal",      "Main requirement"),
            "campo_1_alcance_funcional": ("Alcance funcional del sistema", "Functional scope of the system"),
            "campo_2_fuente":        ("Fuente del estímulo",          "Stimulus source"),
            "campo_3_estimulo":      ("Estímulo / trigger",           "Stimulus / trigger"),
            "campo_4_ambientes":     ("Ambientes y métricas",         "Environments and metrics"),
            "campo_5_prioridad_qa":  ("Prioridad de atributos QA",    "QA attribute priorities"),
            "campo_6_restricciones": ("Restricciones técnicas",       "Technical constraints"),
            "campo_7_decisiones":    ("Decisiones de diseño previas", "Prior design decisions"),
        }
        label_idx = 1 if lang == "en" else 0
        lines = []
        for key, labels in _INTAKE_LABELS.items():
            val = _intake_v1.get(key, "").strip()
            if val:
                lines.append(f"- **{labels[label_idx]}:** {val}")
        if lines:
            if lang == "en":
                intake_context_section = (
                    f'\n{"=" * 60}\n'
                    f'INTAKE CONTEXT — ARCHITECT-PROVIDED REQUIREMENTS:\n'
                    + "\n".join(lines) + "\n"
                    f'\nIMPORTANT: The ASR MUST be grounded in this context. Use the real\n'
                    f'requirement, components, source, stimulus, environments and constraints\n'
                    f'above. Do NOT invent a generic domain.\n'
                    f'{"=" * 60}\n'
                )
            else:
                intake_context_section = (
                    f'\n{"=" * 60}\n'
                    f'CONTEXTO DEL INTAKE — REQUERIMIENTOS PROVISTOS POR EL ARQUITECTO:\n'
                    + "\n".join(lines) + "\n"
                    f'\nIMPORTANTE: El ASR DEBE estar fundamentado en este contexto. Usa el\n'
                    f'requerimiento, componentes, fuente, estímulo, ambientes y restricciones\n'
                    f'reales de arriba. NO inventes un dominio genérico.\n'
                    f'{"=" * 60}\n'
                )
        else:
            intake_context_section = ""
    else:
        intake_context_section = ""

    # ── Baseline prompt section (Punto 4) ────────────────────────────────────
    _baseline = state.get("normal_operation_baseline") or {}
    baseline_prompt_section = _format_baseline_for_prompt(_baseline, lang)

    # ── Dossier history injection (P3) ─────────────────────────────────────
    history_block = _extract_dossier_history(
        (state.get("design_dossier_md") or "").strip()
    )
    history_clipped = _clip_text(history_block, 1500) if history_block else ""
    if history_clipped:
        if lang == "en":
            prior_asr_section = (
                f'\n{"=" * 60}\n'
                f'PRIOR ASR HISTORY — DO NOT REPEAT THESE DESIGNS:\n'
                f'{history_clipped}\n\n'
                f'IMPORTANT: Your new ASR MUST be meaningfully different '
                f'in at least its Response Measure or Stimulus.\n'
                f'{"=" * 60}\n'
            )
        else:
            prior_asr_section = (
                f'\n{"=" * 60}\n'
                f'HISTORIAL DE ASR PREVIOS — NO REPITAS ESTOS DISEÑOS:\n'
                f'{history_clipped}\n\n'
                f'IMPORTANTE: Tu nuevo ASR DEBE ser notablemente distinto '
                f'en al menos su Medida de Respuesta o Estímulo.\n'
                f'{"=" * 60}\n'
            )
    else:
        prior_asr_section = ""

    _asr_glossary = (
        "GLOSARIO: En este contexto, ASR = Architecturally Significant Requirement (ADD 3.0). "
        "NUNCA interpretes ASR como reconocimiento de voz ni como Automatic Speech Recognition."
        if lang == "es" else
        "GLOSSARY: In this context, ASR = Architecturally Significant Requirement (ADD 3.0). "
        "NEVER interpret ASR as Automatic Speech Recognition or any voice/audio technology."
    )

    # Produce a prioritized CANDIDATE TABLE (one row per QA), not a single 6-part ASR.
    # The 6-part detail is reserved for AFTER the user selects an ID. The Response
    # cell must describe WHAT the system must do (the measurable target) — never
    # tactics like backpressure / circuit breaker / autoscaling.
    if lang == "es":
        _col_header = "| ID | Atributo de calidad | Descripción del escenario | Importancia negocio | Riesgo técnico |"
        _table_sep  = "|----|--------------------|----------------------------|---------------------|----------------|"
        _row_hint   = "| A1 | Latencia           | p95 ≤ 800ms con 600 CCU    | H                   | H              |"
        _select_q   = "Escribe el ID (A1, A2…) del ASR que quieras profundizar."
        _final_rmd  = "RECORDATORIO FINAL: responde completamente en español."
    else:
        _col_header = "| ID | Quality attribute | Scenario description       | Business importance | Technical risk |"
        _table_sep  = "|----|-------------------|----------------------------|---------------------|----------------|"
        _row_hint   = "| A1 | Latency           | p95 ≤ 800ms with 600 CCU   | H                   | H              |"
        _select_q   = "Type the ID (A1, A2…) of the ASR you want to expand."
        _final_rmd  = "FINAL REMINDER: answer entirely in English."

    prompt = f"""{directive}
{_asr_glossary}

You are an expert software architect following Attribute-Driven Design 3.0 (ADD 3.0).

Your job is to produce a PRIORITIZED TABLE of candidate Architecture Significant
Requirements (ASRs). The architect will pick one ID to expand later — do NOT
expand them now.

Generate between 6 and 8 candidate rows covering the quality attributes
(latency, scalability, availability, security, modifiability, etc.) that are
problematic for THIS system. Read the intake context below and infer which QAs
are at stake. Each row must be traceable to a concrete stakeholder problem or a
specific system metric from the intake context; if a QA cannot be justified
with intake evidence, omit it. The same QA may legitimately drive more than one
ASR when there are distinct scenarios (e.g. read-path latency vs. write-path
latency, or steady-state availability vs. failover availability) — in that case
emit one row per scenario, each with its own measurable threshold. Use IDs A1
through A8 in order (stop when you run out of justified scenarios).

{"=" * 60}
PROJECT CONTEXT — YOU MUST RESPECT THESE CONSTRAINTS:
{proj_ctx if proj_ctx else "(none — no project configured)"}

IMPORTANT: If a tech stack is listed above, scenario descriptions MUST reference
those specific technologies. If business rules are listed, scenarios MUST be
coherent with them.
{"=" * 60}
{intake_context_section}{baseline_prompt_section}{prior_asr_section}
Relevant domain or workload (you must stay coherent with this):
{domain}

Primary quality attribute inferred from the user message (use it for at least one row):
{qa_focus}

User input to ground these candidates:
{uq}

Additional session context (if any):
{ctx or "None"}

OPTIONAL BOOK CONTEXT (only if not in DOC-ONLY mode):
{book_snippets or "None"}

OUTPUT FORMAT (MANDATORY) — output ONLY the Markdown table below, then the
selection question, and nothing else. No prose before or after.

{_col_header}
{_table_sep}
{_row_hint}

Hard rules:
- Each row's "Scenario description" cell MUST contain a single sentence ≤ 120 characters
  with a concrete, measurable threshold (p95/p99, RPS, error rate, availability %, etc.).
- "Business importance" and "Technical risk" are EXACTLY one of: H, M, L.
- The Scenario describes WHAT the system must do under stress — NOT how. NEVER mention
  tactics: backpressure, circuit breaker, bulkhead, load shedding, retries, timeouts,
  autoscaling, replication, caching, queueing, throttling, rate limiting, sharding.
  These belong to a later phase.
- NEVER emit code blocks, YAML, JSON, k6 scripts, Go snippets, checklists, or any
  section headings (## …). ONLY the table and the selection question.
- The same QA may appear in more than one row only if each scenario is genuinely
  distinct (different stimulus, environment, or response measure); never duplicate
  scenarios.
- Answer entirely in the requested language.
{MARKDOWN_FORMAT_DIRECTIVE}

After the table, on a new line, write EXACTLY this selection prompt:
{_select_q}

{_final_rmd}
"""

    result = llm.invoke(apply_mode_prompt(state, prompt))
    content_raw = getattr(result, "content", str(result))
    content = _sanitize_response(content_raw)
    content = _strip_tactics_sections(content)

    # Parse the candidate table. If parsing fails (no rows detected) fall back
    # to the single-ASR coercion to keep degraded mode.
    _table_rows = _parse_asr_table_rows(content)
    _is_candidate_table = len(_table_rows) >= 1
    _asr_discarded = False
    if not _is_candidate_table:
        content = _coerce_single_asr_markdown(content)
        # Single-ASR fallback path: validate against baseline as before.
        if _baseline.get("parsed"):
            asr_metrics = _parse_response_measure_metrics(content)
            if _is_within_normal_operation(asr_metrics, _baseline):
                _asr_discarded = True
                _summary = _clip_text(content.strip().split("\n")[0], 120)
                _bl_raw = _baseline.get("raw", "")
                state["add_assumptions"] = (state.get("add_assumptions") or []) + [
                    f"ASR descartado: '{_summary}' — cae dentro de operación normal (baseline: {_bl_raw})"
                ]
                if lang == "es":
                    content = (
                        "Con el contexto proporcionado, el escenario descrito cae dentro de tu "
                        f"operación normal ({_bl_raw}). No identifiqué un requerimiento "
                        "arquitectónicamente significativo.\n\n"
                        "¿Puedes describir condiciones de estrés, picos de carga, o restricciones "
                        "críticas que excedan la operación normal?"
                    )
                else:
                    content = (
                        "Based on the context provided, the described scenario falls within your "
                        f"normal operation ({_bl_raw}). I did not identify an architecturally "
                        "significant requirement.\n\n"
                        "Can you describe stress conditions, load spikes, or critical constraints "
                        "that exceed normal operation?"
                    )
                log.info("asr_node: ASR discarded — within normal operation baseline")
        elif not _baseline.get("parsed") and _baseline.get("raw"):
            state["add_assumptions"] = (state.get("add_assumptions") or []) + [
                "Baseline no numérico — validación de operación normal omitida."
            ]

    # === Fuentes (si hubo RAG) ===
    # Never expose server filesystem paths in references.
    src_lines = []
    for d in docs_list or []:
        md = d.metadata or {}
        title = md.get("source_title") or md.get("title") or "doc"
        page = md.get("page_label") or md.get("page")
        page_str = f" (p.{page})" if page is not None else ""
        src_lines.append(f"- {title}{page_str}")
    if src_lines:
        src_lines = [_clip_text(s, 60) for s in src_lines]
        src_lines = list(dict.fromkeys(src_lines))[:4]

    src_block = "SOURCES:\n" + (
        "\n".join(src_lines) if src_lines else "- (no local sources)"
    )

    # Traza + memoria de turno
    state["turn_messages"] = state.get("turn_messages", []) + [
        {"role": "system", "name": "asr_system", "content": prompt},
        {"role": "assistant", "name": "asr_recommender", "content": content},
        {"role": "assistant", "name": "asr_sources", "content": src_block},
    ]
    state["messages"] = state["messages"] + [
        AIMessage(content=content, name="asr_recommender"),
        AIMessage(content=src_block, name="asr_sources"),
    ]

    # Memoria viva del chat.
    # When ASR was discarded (within normal operation), `content` is a
    # user-facing rejection message — NOT a real ASR. Writing it to current_asr/
    # last_asr/memory_text pollutes the session and makes every subsequent turn
    # think an ASR exists, blocking re-generation.
    refs_list = [
        ln.lstrip("- ").strip()
        for ln in src_block.splitlines()
        if ln.strip() and not ln.lower().startswith("sources")
    ]
    state["asr_sources_list"] = refs_list

    # Metadatos
    state["quality_attribute"] = qa_pipeline

    if not _asr_discarded:
        state["last_asr"] = content
        state["current_asr"] = content
        prev_mem = state.get("memory_text", "") or ""
        state["memory_text"] = (prev_mem + f"\n\n[LAST_ASR]\n{content}\n").strip()

    # ── Ledger write-back (P3) — skip if ASR was discarded ──────────────────
    _user_id    = (state.get("user_id_for_prefs") or "").strip()
    _project_id = (state.get("project_id") or "").strip() or None

    if _user_id and not _asr_discarded:
        try:
            _sources = _build_sources_from_docs(docs_list)
            _candidates_for_state: list[dict] = []

            if _is_candidate_table:
                # Persist ONE ledger decision per candidate row so downstream
                # nodes (style, tactics) can resolve any selected ID via
                # get_all_active_asrs / selected_asrs.
                _last_saved_id = ""
                for _row in _table_rows:
                    _qa_row = normalize_qa(_row.get("qa", "")) or qa_pipeline
                    _payload = {
                        "candidate_id":   _row["id"],
                        "summary":        _row["scenario"],
                        "scenario":       _row["scenario"],
                        "qa":             _row["qa"],
                        "business":       _row["business"],
                        "risk":           _row["risk"],
                        "domain":         domain or "",
                        # Six-part fields stay empty until the user selects an
                        # ID and the post-selection detail pass populates them.
                        "source":           "",
                        "stimulus":         "",
                        "environment":      "",
                        "artifact":         "",
                        "response":         "",
                        "response_measure": "",
                    }
                    _new_decision: dict = {
                        "id":               "",
                        "kind":             "asr",
                        "phase":            Phase.ASR_TABLE.value,
                        "iteration":        0,
                        "qa":               _qa_row,
                        "parents":          [],
                        "payload":          _payload,
                        "rationale":        "",
                        "sources":          _sources,
                        "status":           "active",
                        "parent_status":    "ok",
                        "superseded_by":    None,
                        "rejection_reason": None,
                        "created_at":       "",
                        "created_by_node":  "asr_node",
                    }
                    _saved = append_decision(_user_id, _project_id, _new_decision)
                    _last_saved_id = _saved["id"]
                    _candidates_for_state.append({
                        "id":            _saved["id"],
                        "candidate_id": _row["id"],
                        "qa":            _row["qa"],
                        "scenario":      _row["scenario"],
                        "business":      _row["business"],
                        "risk":          _row["risk"],
                        "payload":       _payload,
                    })
                log.info(
                    "asr_node: ledger ok rows=%d last_id=%s project=%s",
                    len(_table_rows), _last_saved_id, _project_id,
                )
            else:
                # Legacy single-ASR fallback: keep behaviour intact.
                _asr_payload = _build_asr_payload(content, domain)
                _new_decision_one: dict = {
                    "id":               "",
                    "kind":             "asr",
                    "phase":            Phase.ASR_TABLE.value,
                    "iteration":        0,
                    "qa":               qa_pipeline,
                    "parents":          [],
                    "payload":          _asr_payload,
                    "rationale":        "",
                    "sources":          _sources,
                    "status":           "active",
                    "parent_status":    "ok",
                    "superseded_by":    None,
                    "rejection_reason": None,
                    "created_at":       "",
                    "created_by_node":  "asr_node",
                }
                _saved = append_decision(_user_id, _project_id, _new_decision_one)
                _candidates_for_state.append({
                    "id":       _saved["id"],
                    "qa":       qa_pipeline,
                    "scenario": _clip_text(content.strip().split("\n")[0], 200),
                    "payload":  _asr_payload,
                })
                log.info(
                    "asr_node: ledger ok (single) id=%s qa=%s project=%s",
                    _saved["id"], qa_pipeline, _project_id,
                )

            _refresh_ledger_state(state, _user_id, _project_id, lang)

            # ── Fix A: self-heal phase if intake's transition_phase failed ─
            # intake_node transitions "diagnosis"→"asr_table" before routing
            # here, but that call can fail silently. If the ledger is still at
            # "diagnosis" after our append_decision, we own the transition now.
            if state.get("current_phase") == "diagnosis":
                try:
                    _ph_ledger = load_ledger(_user_id, _project_id, auto_migrate=False)
                    if _ph_ledger.get("current_phase") == "diagnosis":
                        transition_phase(_user_id, _project_id, PhaseTransition(
                            from_phase="diagnosis",
                            to_phase="asr_table",
                            iteration=_ph_ledger["current_iteration"] + 1,
                            triggered_by="asr_node_self_heal",
                            user_message=uq,
                            skipped_phases=[],
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        ))
                        log.info("asr_node: self-healed phase diagnosis→asr_table")
                    else:
                        # ledger already advanced; just sync state
                        state["current_phase"] = _ph_ledger.get("current_phase") or "asr_table"
                    _refresh_ledger_state(state, _user_id, _project_id, lang)
                except (LedgerValidationError, LedgerConcurrencyError, Exception) as _ph_exc:
                    log.warning("asr_node: self-heal transition failed (nonfatal): %s", _ph_exc)
                    state["current_phase"] = "asr_table"
            # ─────────────────────────────────────────────────────────────

            # ── Populate asr_candidates (P8) ───────────────────────────────
            if _is_regenerate or _is_candidate_table:
                state["asr_candidates"] = _candidates_for_state
            else:
                state["asr_candidates"] = (state.get("asr_candidates") or []) + _candidates_for_state

        except LedgerValidationError as _exc:
            log.warning("asr_node: ledger validation error (nonfatal): %s", _exc)
        except LedgerConcurrencyError as _exc:
            log.warning("asr_node: ledger concurrency error (nonfatal): %s", _exc)
        except Exception as _exc:
            log.warning("asr_node: unexpected ledger error (nonfatal): %s", _exc)

    # Señales de fin de turno
    state["endMessage"] = content
    state["hasVisitedASR"] = True
    state["force_rag"] = False
    state["nextNode"] = "unifier"

    # Persist completed_nodes and routing_phase so boot_node does not reset them
    # on the next turn and the supervisor does not re-run ASR — but only when a
    # real ASR was produced. On discard, leave routing_phase at "intake" so the
    # next turn can legitimately retry ASR generation with new context.
    _done = list(state.get("completed_nodes") or [])
    if "asr" not in _done:
        _done.append("asr")
    state["completed_nodes"] = _done
    if not _asr_discarded:
        state["routing_phase"] = "asr"

    return state
