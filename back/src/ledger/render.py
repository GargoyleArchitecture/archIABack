from __future__ import annotations

from typing import Any

from src.ledger.types import DesignLedger, Phase

# ---------------------------------------------------------------------------
# Token-budget constants (characters, not tokens; ~4 chars/token estimate).
# ---------------------------------------------------------------------------
_MAX_RATIONALE   = 1200
_MAX_SUMMARY     = 600
_MAX_PAYLOAD_STR = 800
_CLIP_MARKER     = "… [truncado]"

# ---------------------------------------------------------------------------
# Localisation table
# ---------------------------------------------------------------------------
_T: dict[str, dict[str, str]] = {
    "es": {
        "title":            "Design Dossier",
        "phase":            "Fase",
        "iteration":        "Iteración",
        "none_yet":         "ninguna aún",
        "constraints":      "Restricciones del proyecto",
        "active_decisions": "Decisiones activas",
        "asr_section":      "ASR",
        "style_section":    "Estilo",
        "tactics_section":  "Tácticas",
        "diagram_section":  "Diagrama",
        "phase_history":    "Historial de fase",
        "history":          "Historial (reemplazadas / rechazadas)",
        "superseded":       "Reemplazadas",
        "rejected":         "Rechazadas",
        "summary":          "Resumen",
        "chosen":           "Elegido",
        "rationale":        "Justificación",
        "candidates":       "Candidatos considerados",
        "sources":          "Fuentes",
        "on_date":          "el",
        "derived_from":     "derivado de ASR",
        "traces_to":        "traza a ASR",
        "via_style":        "vía estilo",
        "pending_advance":  "Avance pendiente",
    },
    "en": {
        "title":            "Design Dossier",
        "phase":            "Phase",
        "iteration":        "Iteration",
        "none_yet":         "none yet",
        "constraints":      "Project constraints",
        "active_decisions": "Active decisions",
        "asr_section":      "ASR",
        "style_section":    "Style",
        "tactics_section":  "Tactics",
        "diagram_section":  "Diagram",
        "phase_history":    "Phase history",
        "history":          "History (superseded / rejected)",
        "superseded":       "Superseded",
        "rejected":         "Rejected",
        "summary":          "Summary",
        "chosen":           "Chosen",
        "rationale":        "Rationale",
        "candidates":       "Candidates considered",
        "sources":          "Sources",
        "on_date":          "on",
        "derived_from":     "derived from ASR",
        "traces_to":        "traces to ASR",
        "via_style":        "via style",
        "pending_advance":  "Pending advance",
    },
}


def _clip_text(s: str, max_chars: int = _MAX_RATIONALE) -> str:
    if not s or len(s) <= max_chars:
        return s
    return s[:max_chars] + _CLIP_MARKER


def _active_view(ledger: DesignLedger) -> dict[str, Any]:
    """Inline active-view so render.py doesn't import store.py (avoids any latent cycle)."""
    active: dict[str, Any] = {}
    for d in ledger["decisions"]:
        if d["status"] == "active":
            active[d["kind"]] = d
    return active


def _fmt_sources(sources: list[dict]) -> str:
    parts = []
    for s in sources:
        title = s.get("title", "")
        page  = s.get("page")
        parts.append(f"{title} (p.{page})" if page else title)
    return ", ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------

def render_dossier(
    ledger: DesignLedger,
    *,
    lang: str = "es",
    focus: str | None = None,
    include_history: bool = True,
) -> str:
    """Render a full Markdown dossier from the ledger."""
    T = _T.get(lang, _T["es"])
    none_yet = f"_({T['none_yet']})_"
    active   = _active_view(ledger)

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    project = ledger.get("project_id") or "unknown"
    lines += [
        f"# {T['title']} — {project}",
        "",
        f"**{T['phase']}:** {ledger['current_phase']}  ·  **{T['iteration']}:** {ledger['current_iteration']}",
        "",
    ]

    # ── Constraints ───────────────────────────────────────────────────────
    if focus is None or focus == "constraint":
        lines.append(f"## {T['constraints']}")
        constraint = active.get("constraint")
        if constraint:
            payload = constraint.get("payload") or {}
            tech  = payload.get("tech_stack", [])
            rules = payload.get("business_rules", [])
            if isinstance(tech, list) and tech:
                lines.append(f"- Tech stack: {', '.join(str(t) for t in tech)}")
            if isinstance(rules, list):
                for rule in rules:
                    lines.append(f"- {rule}")
            elif isinstance(rules, str) and rules:
                lines.append(f"- {rules}")
        else:
            lines.append(none_yet)
        lines.append("")

    # ── Active decisions ───────────────────────────────────────────────────
    lines += [f"## {T['active_decisions']}", ""]

    # ASR
    if focus is None or focus == "asr":
        asr = active.get("asr")
        if asr:
            lines.append(
                f"### {T['asr_section']}  (id: {asr['id']}, QA: {asr['qa']}, fase: {asr['phase']})"
            )
            payload = asr.get("payload") or {}
            lines.append(f"**{T['summary']}:** {_clip_text(payload.get('summary', ''), _MAX_SUMMARY)}")
            lines.append("")
            for field in ("source", "stimulus", "environment", "artifact", "response", "response_measure"):
                val = payload.get(field, "")
                if val:
                    lines.append(f"- **{field.replace('_', ' ').title()}:** {val}")
            domain = payload.get("domain", "")
            if domain:
                lines.append(f"- **Dominio:** {domain}")
            rationale = asr.get("rationale", "")
            if rationale:
                lines += ["", f"**{T['rationale']}:** {_clip_text(rationale)}"]
            sources = asr.get("sources") or []
            if sources:
                lines += ["", f"**{T['sources']}:** {_fmt_sources(sources)}"]
        else:
            lines += [f"### {T['asr_section']}", none_yet]
        lines.append("")

    # Style
    if focus is None or focus == "style":
        style = active.get("style")
        if style:
            asr_ref  = active.get("asr")
            asr_note = f", {T['derived_from']} {asr_ref['id']}" if asr_ref else ""
            lines.append(f"### {T['style_section']}  (id: {style['id']}{asr_note})")
            payload = style.get("payload") or {}
            lines.append(f"**{T['chosen']}:** {payload.get('chosen', '')}")
            rationale = style.get("rationale", "")
            if rationale:
                lines.append(f"**{T['rationale']}:** {_clip_text(rationale)}")
            candidates = payload.get("candidates") or []
            if candidates:
                lines.append(f"**{T['candidates']}:**")
                for c in candidates:
                    lines.append(f"- {c.get('name', '')}: {c.get('impact', '')}")
            tradeoffs = payload.get("tradeoffs", "")
            if tradeoffs:
                lines.append(f"Tradeoffs: {_clip_text(str(tradeoffs), _MAX_PAYLOAD_STR)}")
        else:
            lines += [f"### {T['style_section']}", none_yet]
        lines.append("")

    # Tactics
    if focus is None or focus == "tactic":
        tactic = active.get("tactic")
        if tactic:
            asr_ref   = active.get("asr")
            style_ref = active.get("style")
            refs = ""
            if asr_ref and style_ref:
                refs = (
                    f", {T['traces_to']} {asr_ref['id']} "
                    f"{T['via_style']} {style_ref['id']}"
                )
            lines.append(f"### {T['tactics_section']}  (id: {tactic['id']}{refs})")
            items = (tactic.get("payload") or {}).get("items") or []
            if isinstance(items, list):
                for i, item in enumerate(items, 1):
                    name    = item.get("name", "")
                    purpose = item.get("purpose", "")
                    prob    = item.get("success_probability", "")
                    rank    = item.get("rank", "")
                    prob_str = f"  (success {prob}, rank {rank})" if (prob or rank) else ""
                    lines.append(f"{i}. {name} — {purpose}{prob_str}")
                    trades = item.get("tradeoffs", "")
                    if trades:
                        lines.append(f"   - Tradeoffs: {trades}")
            else:
                lines.append(str(items))
        else:
            lines += [f"### {T['tactics_section']}", none_yet]
        lines.append("")

    # Diagram
    if focus is None or focus == "diagram":
        diagram = active.get("diagram")
        if diagram:
            payload = diagram.get("payload") or {}
            level   = payload.get("level", "?")
            lines.append(f"### {T['diagram_section']}  (level {level}, id: {diagram['id']})")
        else:
            lines += [f"### {T['diagram_section']}", none_yet]
        lines.append("")

    # ── Phase history ──────────────────────────────────────────────────────
    if include_history:
        lines.append(f"## {T['phase_history']}")
        phase_history = ledger.get("phase_history") or []
        if phase_history:
            for t in phase_history:
                date = (t.get("timestamp") or "")[:10]
                lines.append(
                    f"- {T['iteration']} {t['iteration']}: {t['from_phase']} → {t['to_phase']}"
                    f" ({t['triggered_by']}) {T['on_date']} {date}"
                )
        else:
            lines.append(none_yet)
        lines.append("")

        # ── Superseded / Rejected ──────────────────────────────────────────
        lines += [f"## {T['history']}", ""]

        superseded_list = [d for d in ledger["decisions"] if d["status"] == "superseded"]
        lines.append(f"### {T['superseded']}")
        if superseded_list:
            for d in superseded_list:
                sup_by = d.get("superseded_by") or "?"
                date   = (d.get("created_at") or "")[:10]
                lines.append(f"- {d['kind'].upper()} {d['id']} → reemplazado por {sup_by} el {date}")
        else:
            lines.append(none_yet)
        lines.append("")

        rejected_list = [d for d in ledger["decisions"] if d["status"] in ("rejected", "orphaned")]
        lines.append(f"### {T['rejected']}")
        if rejected_list:
            for d in rejected_list:
                reason  = d.get("rejection_reason") or ""
                status  = d["status"]
                payload = d.get("payload") or {}
                name    = (
                    payload.get("chosen") or payload.get("summary") or d["id"]
                )
                lines.append(
                    f"- {d['kind'].upper()} **{name}**"
                    f" (id: {d['id']}, iteración {d['iteration']}) — {status}: \"{reason}\""
                )
        else:
            lines.append(none_yet)

    return "\n".join(lines) + "\n"


def render_dossier_compact(ledger: DesignLedger, *, lang: str = "es") -> str:
    """Compact one-liner summary per active decision — for the supervisor prompt."""
    T      = _T.get(lang, _T["es"])
    active = _active_view(ledger)
    lines: list[str] = []

    lines.append(
        f"**{T['phase']}:** {ledger['current_phase']}  |  "
        f"**{T['iteration']}:** {ledger['current_iteration']}"
    )

    asr = active.get("asr")
    if asr:
        summary = _clip_text((asr.get("payload") or {}).get("summary", ""), 200)
        lines.append(f"**ASR** ({asr['id']}, QA:{asr['qa']}): {summary}")

    style = active.get("style")
    if style:
        chosen = (style.get("payload") or {}).get("chosen", "")
        lines.append(f"**{T['style_section']}** ({style['id']}): {chosen}")

    tactic = active.get("tactic")
    if tactic:
        items = (tactic.get("payload") or {}).get("items") or []
        n = len(items) if isinstance(items, list) else "?"
        lines.append(f"**{T['tactics_section']}** ({tactic['id']}): {n} items")

    diagram = active.get("diagram")
    if diagram:
        level = (diagram.get("payload") or {}).get("level", "?")
        lines.append(f"**{T['diagram_section']}** ({diagram['id']}): level {level}")

    pending = ledger.get("pending_advance")
    if pending:
        lines.append(
            f"⏳ {T['pending_advance']}: "
            f"{pending['from_phase']} → {pending['to_phase']}"
        )

    return "\n".join(lines) + "\n"


def render_phase_prompt(ledger: DesignLedger, *, lang: str = "es") -> str:
    """Next-step guidance footer the unifier appends at end-of-phase."""
    phase = ledger.get("current_phase", Phase.INTAKE.value)

    _MSGS: dict[str, dict[str, str]] = {
        "es": {
            Phase.INTAKE.value:   "**Próximo paso:** Define el ASR — el requisito arquitectónicamente significativo que guiará el diseño.",
            Phase.ASR.value:      "**Próximo paso:** Elige un estilo arquitectónico que satisfaga el atributo de calidad del ASR.",
            Phase.STYLE.value:    "**Próximo paso:** Define tácticas que realicen el estilo con trazabilidad al ASR.",
            Phase.TACTICS.value:  "**Próximo paso:** Genera el diagrama de arquitectura basado en el estilo y las tácticas.",
            Phase.DIAGRAM.value:  "**Próximo paso:** Evalúa el diseño — identifica fortalezas, debilidades y sugerencias.",
            Phase.ANALYSIS.value: "**Próximo paso:** Revisa el análisis y confirma si el diseño está listo.",
        },
        "en": {
            Phase.INTAKE.value:   "**Next step:** Define the ASR — the architecturally significant requirement that will guide the design.",
            Phase.ASR.value:      "**Next step:** Choose an architectural style that satisfies the ASR's quality attribute.",
            Phase.STYLE.value:    "**Next step:** Define tactics that realize the style and trace back to the ASR.",
            Phase.TACTICS.value:  "**Next step:** Generate the architecture diagram based on the chosen style and tactics.",
            Phase.DIAGRAM.value:  "**Next step:** Evaluate the design — identify strengths, weaknesses, and improvement suggestions.",
            Phase.ANALYSIS.value: "**Next step:** Review the analysis and confirm if the design is ready.",
        },
    }

    msgs = _MSGS.get(lang, _MSGS["es"])
    msg  = msgs.get(phase, "")
    return msg + "\n" if msg else ""
