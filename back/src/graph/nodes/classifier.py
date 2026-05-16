import os
import re
from functools import lru_cache
from src.services.llm_factory import get_chat_model
from src.graph.state import GraphState, ClassifyOut
from src.graph.index_resolver import resolve_quality_attribute
from src.graph.qa_registry import detect_explicit_qa, normalize_qa, supported_qas
from src.graph.utils import is_asr_regenerate_request

# Fast regex-based language detector — used during INTAKE to skip the LLM call
# while still updating language on every turn.
_ES_MARKERS = re.compile(
    r"[áéíóúüñ¿¡]"
    r"|\b(hola|gracias|cómo|como|qué|que|está|tienes|puedes|"
    r"necesito|quiero|tengo|los|las|con|para|sí|hay|bien|mal|"
    r"mi|tu|su|una|ninguna|ningún|"
    r"el|la|del|al|debe|deben|usuario|usuarios|sistema|"
    r"por|pero|también|cuando|donde|quien|"
    r"escalar|procesar|diseñar|implementar|manejar|"
    r"genera|crea|muestra|hazme|dime|ahora|vamos|este|ese)\b",
    re.IGNORECASE,
)


def _detect_lang_fast(msg: str) -> str:
    return "es" if _ES_MARKERS.search(msg) else "en"

llm = get_chat_model(temperature=0.0)


# ===== F2-T4: Heuristica de auto-routing de modo =====
# Triggers bilingues (ES/EN). Cada match suma 0.5 al score (cap 1.0).
TUTOR_TRIGGERS = (
    r"\bexpl[ií]came\b", r"\bense[ñn]ame\b", r"\bno\s+entiendo\b",
    r"\bqu[eé]\s+es\b", r"\bqu[eé]\s+significa\b",
    r"\bay[uú]dame\s+a\s+entender\b", r"\bduda\b", r"\bconcept[oa]\b",
    r"\bexplain\b", r"\bteach\s+me\b", r"\bwhat\s+is\b",
    r"\bdon'?t\s+understand\b", r"\bdoubt\b",
)
PROFESSIONAL_TRIGGERS = (
    r"\bimplementa(?:me|lo)?\b", r"\bdame\s+(?:el\s+)?c[oó]digo\b",
    r"\bmu[eé]strame\b", r"\bdiagrama\b", r"\bdise[ñn]a\b",
    r"\boptimiza\b", r"\bbenchmark\b", r"\barquitectura\s+de\b",
    r"\bimplement\b", r"\bgive\s+me\s+the\s+code\b", r"\bdesign\s+the\b",
    r"\bbuild\b", r"\boptimize\b", r"\bcompare\s+latency\b",
)

_MODE_SUGGESTION_THRESHOLD = float(os.getenv("MODE_SUGGESTION_THRESHOLD", "0.7"))


def _score_triggers(text: str, patterns: tuple) -> float:
    """Confianza en [0,1]: 1 match = 0.8, 2+ saturan a 1.0.

    Calibrado para que un unico verbo de intencion claro (ej. 'explicame',
    'dame el codigo') ya supere el umbral default 0.7, y multiples matches
    converjan rapido a 1.0.
    """
    matches = sum(1 for p in patterns if re.search(p, text or "", re.IGNORECASE))
    return 0.0 if matches == 0 else min(1.0, 0.8 + 0.2 * (matches - 1))


def suggest_mode(text: str, current_mode: str):
    """Devuelve `tutor` / `professional` si la confianza supera el umbral
    Y el modo sugerido difiere del actual. None si no aplica."""
    tutor_conf = _score_triggers(text, TUTOR_TRIGGERS)
    pro_conf = _score_triggers(text, PROFESSIONAL_TRIGGERS)
    if tutor_conf >= _MODE_SUGGESTION_THRESHOLD and tutor_conf > pro_conf:
        return "tutor" if current_mode != "tutor" else None
    if pro_conf >= _MODE_SUGGESTION_THRESHOLD and pro_conf > tutor_conf:
        return "professional" if current_mode != "professional" else None
    return None

FOLLOWUP_PATTERNS = [
    ("explain_tactics", r"\b(tactics?|tácticas?).*(explain|describe|detalla|explica)|explica.*tácticas"),
    ("make_asr",        r"\b(asr|architecture significant requirement).*(make|create|example|ejemplo)|ejemplo.*asr"),
    ("component_view",  r"\b(component|diagrama de componentes|component diagram)"),
    ("deployment_view", r"\b(deployment|despliegue|deployment view)"),
    ("functional_view", r"\b(functional view|vista funcional)"),
    ("compare",         r"\b(compare|comparar).*?(latency|scalability|availability)"),
    ("checklist",       r"\b(checklist|lista de verificación|lista de verificacion)"),
]

@lru_cache(maxsize=256)
def _classify_cached(msg: str, qa_opts_str: str) -> tuple:
    """Returns (language, intent, use_rag, quality_attribute). Cached by (msg, qa_opts_str)."""
    prompt = f"""
Classify the user's last message. Return JSON with:
- language: "en" or "es"
- intent: one of ["greeting","smalltalk","architecture","diagram","asr","tactics","style","tech","other"]
  Use "tech" when the user asks about concrete technologies, frameworks, libraries, tools, or tech stack to implement the architecture.
- use_rag: true if this is a software-architecture question (ADD, tactics, latency, scalability,
  quality attributes, views, styles, diagrams, ASR, technologies, frameworks), else false.
- quality_attribute: one of [{qa_opts_str}].
  Use "general" only if no clear quality attribute is requested.

User message:
{msg}
"""
    out = llm.with_structured_output(ClassifyOut).invoke(prompt)
    return (out["language"], out["intent"], bool(out["use_rag"]), out.get("quality_attribute", "general"))


@lru_cache(maxsize=128)
def _resolve_qa_cached(msg: str) -> str:
    """Cached wrapper around resolve_quality_attribute (deterministic, temperature=0.0)."""
    return resolve_quality_attribute(msg, llm)


def classifier_node(state: GraphState) -> GraphState:
    """Clasifica intención/idioma y fija QA operativo para el turno.

    Además de "resolved_index" (para RAG), este nodo propaga
    "quality_attribute" para que supervisor/router puedan decidir nodos
    específicos por QA (p. ej. style_latency vs style_scalability).
    """
    # During DIAGNOSIS the supervisor routes to intake_node regardless of intent.
    # Skip the LLM call to preserve intent="intake" and avoid spurious QA overrides,
    # but still detect language so intake_node responds in the user's language.
    # Gate: diagnosis phase only applies in professional mode.
    #
    # BUG-020: if the message contains an explicit phase-advancing keyword
    # (style / tactics / tech / diagram), fall through into the full classifier
    # so the intent is set correctly. The supervisor's phase gate will still
    # block routing when current_phase doesn't permit it, but the user will get
    # the correct "we must finish phase X first" message instead of being
    # silently misrouted with a stale unifier response.
    if (state.get("current_phase") or "") in ("intro", "diagnosis") and (state.get("mode") or "professional") != "tutor":
        msg = state.get("userQuestion", "") or ""
        low = msg.lower()
        _phase_advancing_kw = (
            "estilo", "estilos", "style", "styles",
            "táctica", "tácticas", "tactica", "tacticas", "tactic", "tactics",
            "tecnología", "tecnologías", "tecnologias", "technology", "tech stack",
            "diagrama", "diagram",
        )
        if not any(k in low for k in _phase_advancing_kw):
            prior_lang = state.get("language") or "es"
            # BUG-014: too few words → keep the prior language; only switch when
            # signal is strong enough (> 3 tokens) or the fast detector is sure.
            if prior_lang and len(msg.split()) <= 3:
                lang = prior_lang
            else:
                detected = _detect_lang_fast(msg)
                lang = detected if (detected == "es" or len(msg.split()) > 2) else prior_lang
            return {**state, "language": lang}
        # else: fall through into the full classifier so keyword overrides run

    msg = state.get("userQuestion", "") or ""
    qa_ids = supported_qas()
    qa_opts = qa_ids + ["general"]
    qa_opts_str = ", ".join(f'"{q}"' for q in qa_opts)
    lang_raw, intent_raw, use_rag, qa_attr = _classify_cached(msg, qa_opts_str)

    low = msg.lower()
    intent = intent_raw

    #disparadores de estilo arquitectónico
    style_triggers = [
        "style", "styles",
        "architecture style", "architectural style",
        "estilo", "estilos", "estilo arquitectónico", "estilos arquitectónicos"
    ]
    if any(k in low for k in style_triggers):
        intent = "style"


    tactics_triggers = [
        "tactic", "táctica", "tactica", "tácticas", "tactics", "tactcias",
        "strategy","estrategia",
        "cómo cumplir","como cumplir","how to meet","how to satisfy","how to achieve"
    ]
    if any(k in low for k in tactics_triggers):
        intent = "tactics"

    tech_triggers = [
        "tecnología", "tecnologias", "tecnologías", "technology", "tech stack",
        "framework", "library", "librería", "herramienta", "tool", "tools",
        "implementación", "implementacion", "implementation",
        "qué usar", "que usar", "what to use", "which library", "which framework",
        "propón tecnologías", "propón tecnologias", "propose technologies",
        "stack tecnológico", "stack tecnologico",
    ]
    if any(k in low for k in tech_triggers) and intent not in ("asr", "style", "tactics"):
        intent = "tech"

    diagram_keywords = [
        "component diagram", "diagram", "diagrama", "diagrama de componentes",
        "diagrama de despliegue", "deployment diagram",
        "uml", "plantuml", "c4", "bpmn", "despliegue", "deployment", "graphviz", "dot"
    ]
    # Evita enrutar a diagrama por frases tipo "ese ASR" sin pedir diagrama explícito.
    if any(k in low for k in diagram_keywords) and intent not in ("asr", "style", "tactics"):
        intent = "diagram"

    # Confirmación / rechazo de ASR: solo aplican si ya hay un ASR vigente y la
    # fase actual del ledger es asr_table. Sin esa precondición, "confirmo" no
    # tiene referente y debe seguir el flujo normal.
    _has_existing_asr = bool(state.get("current_asr") or state.get("last_asr"))
    _in_asr_phase = (state.get("current_phase") or "") == "asr_table"
    if _has_existing_asr and _in_asr_phase:
        asr_confirm_triggers = [
            "confirmo", "lo confirmo", "acepto este asr", "acepto ese asr",
            "ese asr está bien", "ese asr esta bien", "está bien ese asr", "esta bien ese asr",
            "ese asr me sirve", "me sirve ese asr",
            "confirm", "i confirm", "approve",
            "looks good", "yes that asr", "sí ese asr", "si ese asr",
            # BUG-045: natural-language selection phrases the architect uses when
            # picking an ASR by ID from the candidate table.
            "tomo ese", "tomo el asr", "me quedo con", "elijo", "voy con",
            "ese me sirve", "perfecto ese", "ok ese", "de acuerdo", "dale",
            "vale ese", "i'll take", "let's go with", "pick", "i pick",
            "i choose", "go with",
        ]
        asr_reject_triggers = [
            "rechazo", "ese no", "no me convence", "otro asr", "otro distinto",
            "reject", "another asr", "different asr", "not that one",
        ]
        # BUG-045: ID-pattern matcher — when the user types just "a1", "A2",
        # "tomo A1", "voy con a3", we extract the matched IDs and route to
        # asr_confirm so the supervisor can advance the M1 gate to STYLE_TABLE.
        _asr_id_matches = re.findall(r"\b[Aa](\d+)\b", msg)
        _bare_id = re.match(r"^\s*[Aa]\d+\s*$", msg)
        if any(k in low for k in asr_confirm_triggers) or _asr_id_matches or _bare_id:
            intent = "asr_confirm"
            if _asr_id_matches:
                _selected_ids = [f"A{n}" for n in _asr_id_matches]
                # Preserve dedup + ordering for downstream filters.
                state["selected_asrs"] = list(dict.fromkeys(_selected_ids))
        elif any(k in low for k in asr_reject_triggers) or is_asr_regenerate_request(msg):
            intent = "asr_reject"

    # asr_detail: user requests the 6-part detail of an already-confirmed ASR.
    # Fires across all phases so the phase guard does not block retrospective
    # inspection once the design loop has advanced past asr_table.
    if intent not in ("asr_confirm", "asr_reject"):
        _detail_asr_id_matches = re.findall(r"\b[Aa](\d+)\b", msg)
        _asr_detail_explicit = re.search(
            r"\b(?:detalle|detail)\b.*\b[Aa]\d+\b"
            r"|\bmuéstrame\s+el\s+detalle\b"
            r"|\bmuestrame\s+el\s+detalle\b"
            r"|\bshow\s+me\s+(?:the\s+)?detail\b",
            msg, re.IGNORECASE
        )
        if _asr_detail_explicit and _detail_asr_id_matches:
            _confirmed_ids = {
                str(x).strip().upper() for x in (state.get("selected_asrs") or [])
                if re.match(r"^A\d+$", str(x).strip().upper())
            }
            _requested_ids = [f"A{n}" for n in _detail_asr_id_matches]
            if any(rid in _confirmed_ids for rid in _requested_ids):
                intent = "asr_detail"
                state["asr_detail_ids"] = [rid for rid in _requested_ids if rid in _confirmed_ids]

    # BUG-054 / BUG-055: style selection — mirror of the asr_confirm block.
    # When the user is in style_table and types `S1`, `Selecciono el estilo S2`,
    # etc., classify the intent as `style_confirm` and seed selected_style.
    _in_style_phase = (state.get("current_phase") or "") == "style_table"
    _has_style_candidates = bool(state.get("style_candidates") or [])
    if _in_style_phase and _has_style_candidates:
        style_confirm_triggers = [
            "selecciono el estilo", "selecciono este estilo", "selecciono ese estilo",
            "elijo el estilo", "elijo este estilo", "voy con el estilo",
            "me quedo con el estilo", "confirmo el estilo", "confirmo ese estilo",
            "ese estilo me sirve", "perfecto ese estilo", "ok ese estilo",
            "i pick", "i choose", "go with", "let's go with", "lets go with",
            "select the style", "i'll take the style", "ill take the style",
        ]
        _style_id_matches = re.findall(r"\b[Ss](\d+)\b", msg)
        _bare_sid = re.match(r"^\s*[Ss]\d+\s*$", msg)
        if any(k in low for k in style_confirm_triggers) or _style_id_matches or _bare_sid:
            intent = "style_confirm"
            if _style_id_matches:
                state["selected_style"] = f"S{_style_id_matches[0]}"

    # BUG-012/007/013: tactics selection — mirror of the style_confirm block.
    # When the user is in tactics_table and types `T1`, `acepto las tácticas`, etc.
    # classify as `tactics_confirm` so the supervisor routes to tactics_confirm_node
    # instead of firing "Bienvenido de vuelta".
    _in_tactics_phase_cand = (state.get("current_phase") or "") in ("tactics_table",)
    _has_tactics_candidates = bool(state.get("tactics_candidates") or [])
    if _in_tactics_phase_cand and _has_tactics_candidates:
        tactics_confirm_triggers = [
            "acepto las tácticas", "acepto esas tácticas", "acepto esos tácticas",
            "confirmo las tácticas", "confirmo esas tácticas",
            "me quedo con esas tácticas", "voy con esas tácticas",
            "ok con las tácticas", "perfecto con las tácticas",
            "esas tácticas me sirven", "de acuerdo con las tácticas",
            "acepto", "confirmo", "i accept", "i confirm", "accept tactics",
            "confirm tactics", "approved", "looks good", "go ahead",
            "let's go with the tactics", "lets go with the tactics",
            "i'll take the tactics", "ill take the tactics",
        ]
        _tactics_id_matches = re.findall(r"\b[Tt](\d+)\b", msg)
        _bare_tid = re.match(r"^\s*[Tt]\d+\s*$", msg)
        if any(k in low for k in tactics_confirm_triggers) or _tactics_id_matches or _bare_tid:
            intent = "tactics_confirm"
            if _tactics_id_matches:
                state["selected_tactics"] = [f"T{n}" for n in _tactics_id_matches]

    # BUG-047: when the architect picked a style and is now in TACTICS_TABLE,
    # natural continuation verbs ("Continuemos", "adelante", "ok") should route
    # to tactics generation instead of looping on a confirmation question.
    _in_tactics_phase = (state.get("current_phase") or "") == "tactics_table"
    _has_style = bool(
        state.get("selected_style") or state.get("style") or state.get("last_style")
    )
    _no_tactics = not (state.get("selected_tactics") or [])
    if _in_tactics_phase and _has_style and _no_tactics:
        _continue_triggers = [
            "continuemos", "continúa", "continua", "sigue", "adelante",
            "go ahead", "let's continue", "lets continue", "next", "proceed",
            "siguiente", "ok", "okay", "vale", "dale", "vamos",
        ]
        if any(k in low for k in _continue_triggers):
            intent = "tactics"

    # BUG-014: language is sticky for short, low-signal messages (e.g. "S2", "ok").
    # Only switch when there are enough tokens to classify reliably, OR the user
    # explicitly typed something that triggers the opposite-language detector.
    prior_lang = state.get("language")
    msg_word_count = len(msg.split())
    if prior_lang and msg_word_count <= 3:
        lang = prior_lang
    else:
        lang = lang_raw or prior_lang or "es"

    # QA primario clasificado junto al intent (misma invocación del classifier).
    qa_from_classifier = normalize_qa(qa_attr)
    explicit_qa_in_msg = detect_explicit_qa(msg)
    prev_qa = normalize_qa(state.get("quality_attribute", ""))
    has_existing_asr = bool(state.get("current_asr") or state.get("last_asr"))

    # Resolución del índice QA para RAG.
    # Regla: usar QA del classifier primero; si no, resolver por fallback.
    resolved_index = "general"
    if use_rag:
        if qa_from_classifier != "general":
            resolved_index = qa_from_classifier
        else:
            resolved_index = _resolve_qa_cached(msg)

    # QA operativo del turno (prioridad):
    # 1) QA clasificado junto al intent,
    # 2) índice resuelto para RAG,
    # 3) valor previo del estado (continuidad).
    resolved_qa = normalize_qa(resolved_index)

    preserve_followup_qa = (
        intent in ("style", "tactics", "diagram")
        and has_existing_asr
        and explicit_qa_in_msg == "general"
        and prev_qa != "general"
    )

    if preserve_followup_qa:
        quality_attribute = prev_qa
        resolved_index = prev_qa
    elif qa_from_classifier != "general":
        quality_attribute = qa_from_classifier
    elif resolved_qa != "general":
        quality_attribute = resolved_qa
    elif prev_qa != "general":
        quality_attribute = prev_qa
    else:
        quality_attribute = "general"

    # F2-T4: heuristica de auto-routing de modo. Si el usuario muestra
    # intencion de pedagogia/operatividad y el modo actual no coincide,
    # exponemos `mode_suggestion` para que el Frontend ofrezca el cambio.
    mode_suggestion = suggest_mode(msg, state.get("mode") or "professional")

    # Defense-in-depth: if QA lock-in has not been reached yet, do not let an
    # incidental mention of a quality attribute (e.g. "latencia") during a
    # non-intake turn override the state before the user has explicitly chosen
    # an ASR (BUG-006).  context_loader sets qa_locked_in=True once the phase
    # advances past diagnosis, so this guard is a no-op in normal post-intake
    # flow and only fires in edge cases where context_loader did not run.
    if not state.get("qa_locked_in", True):
        quality_attribute = "general"

    return {
        **state,
        "language": lang,
        "intent": intent if intent in [
        "greeting",
        "smalltalk",
        "architecture",
        "diagram",
        "asr",
        "asr_confirm",
        "asr_reject",
        "asr_detail",
        "tactics",
        "tactics_confirm",
        "style",
        "style_confirm",
        "tech",
    ] else "general",

        "force_rag": bool(use_rag),
        "resolved_index": resolved_index,
        "quality_attribute": quality_attribute,
        "mode_suggestion": mode_suggestion,
    }
