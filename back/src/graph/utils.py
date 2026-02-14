
import re
import json
import logging
import tiktoken
from typing import Any
from src.utils.json_helpers import extract_json_array
from src.graph.consts import TACTICS_HEADINGS
from src.graph.state import GraphState, TACTICS_ARRAY_SCHEMA

log = logging.getLogger("graph")

# ========== Helpers ==========

def _push_turn(state: GraphState, role: str, name: str, content: str) -> None:
    line = {"role": role, "name": name, "content": content}
    state["turn_messages"] = state.get("turn_messages", []) + [line]


def _detect_lang_from_text(text: str) -> str:
    t = (text or "").lower()
    es_hits = sum(w in t for w in ["que", "como", "por que", "cual", "hola", "tactica", "vista", "despliegue", "arquitectura"])
    en_hits = sum(w in t for w in ["what", "how", "why", "which", "hello", "tactic", "view", "deployment", "architecture"])
    if es_hits > en_hits:
        return "es"
    if en_hits > es_hits:
        return "en"
    return "es"


def _resolve_response_language(state, user_text: str = "") -> str:
    lang = state.get("response_language") or state.get("language")
    if lang in ("es", "en"):
        return lang
    return _detect_lang_from_text(user_text or state.get("userQuestion", ""))


def _merge_context_text(base: str, incoming: str, max_chars: int = 4000) -> str:
    b = (base or "").strip()
    i = (incoming or "").strip()
    if not i:
        return b
    if not b:
        return i[:max_chars]
    if i in b:
        return b[:max_chars]
    merged = (b + "\n\n" + i).strip()
    return merged[-max_chars:]


def _sync_contract_from_legacy(state) -> None:
    state["context_contract_version"] = "v1"

    facts = dict(state.get("context_facts") or {})
    if "domain" not in facts:
        facts["domain"] = ""
    if "constraints" not in facts:
        facts["constraints"] = []
    if "nfr_focus" not in facts:
        facts["nfr_focus"] = []
    if "open_assumptions" not in facts:
        facts["open_assumptions"] = []
    state["context_facts"] = facts

    active = dict(state.get("active_decisions") or {})
    active["arch_stage"] = state.get("arch_stage", active.get("arch_stage", ""))
    active["current_asr"] = state.get("current_asr", active.get("current_asr", ""))
    active["quality_attribute"] = state.get("quality_attribute", active.get("quality_attribute", ""))
    active["style"] = state.get("style", active.get("style", ""))
    if "tactics" not in active:
        active["tactics"] = list(state.get("tactics_list", []) or [])
    if "decision_log" not in active:
        active["decision_log"] = []
    state["active_decisions"] = active

    if "turn_context" not in state:
        state["turn_context"] = {}
    if "pending_questions" not in state:
        state["pending_questions"] = []


def _sync_legacy_from_contract(state) -> None:
    active = dict(state.get("active_decisions") or {})
    state["arch_stage"] = active.get("arch_stage", state.get("arch_stage", ""))
    state["current_asr"] = active.get("current_asr", state.get("current_asr", ""))
    state["quality_attribute"] = active.get("quality_attribute", state.get("quality_attribute", ""))
    state["style"] = active.get("style", state.get("style", ""))
    if isinstance(active.get("tactics"), list):
        state["tactics_list"] = active.get("tactics", state.get("tactics_list", []))


def _build_context_summary(state, max_chars: int = 1400) -> str:
    _sync_contract_from_legacy(state)
    active = state.get("active_decisions") or {}
    facts = state.get("context_facts") or {}
    lines = [
        f"Stage: {active.get('arch_stage','')}",
        f"Quality Attribute: {active.get('quality_attribute','')}",
        f"Current ASR: {active.get('current_asr','')}",
        f"Architecture style: {active.get('style','')}",
        f"Tactics: {active.get('tactics', [])}",
        f"Domain: {facts.get('domain','')}",
        f"Constraints: {facts.get('constraints', [])}",
        f"NFR focus: {facts.get('nfr_focus', [])}",
        f"Business / Context: {state.get('add_context','')}",
    ]
    txt = "\n".join(lines).strip()
    out = txt[:max_chars]
    state["context_summary"] = out
    return out


# ========== Token utils (soft) ==========
try:
    _enc = tiktoken.encoding_for_model("gpt-4o")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text or ""))
except Exception:
    def _count_tokens(text: str) -> int:
        # aproximación si no hay tiktoken
        return max(1, int(len(text or "") / 3))

def _clip_text(text: str, max_tokens: int) -> str:
    if _count_tokens(text) <= max_tokens:
        return text
    target_chars = max(100, int(max_tokens * 3))  # 3 chars/token aprox
    return (text or "")[:target_chars] + "…"

def _clip_lines(lines: list[str], max_tokens: int) -> list[str]:
    out, total = [], 0
    for ln in lines:
        t = _count_tokens(ln)
        if total + t > max_tokens: break
        out.append(ln)
        total += t
    return out

def _last_k_messages(msgs, k=6):
    # Mantén solo los últimos K mensajes de usuario/asistente (sin repetir system)
    core = [m for m in msgs if getattr(m, "type", "") != "system"]
    return core[-k:]

# ========== JSON / Sanitization Utils ==========

def _coerce_json_array(raw: str):
    """Intenta extraer un JSON array venga como venga (code-fence, texto suelto, etc.)."""
    if not raw:
        return None
    # 1) helper principal
    try:
        from src.utils.json_helpers import extract_json_array
        arr = extract_json_array(raw)
        if isinstance(arr, list) and arr:
            return arr
    except Exception:
        pass

    # 2) fence ```json ... ```
    m = re.search(r"```json\s*(\[.*?\])\s*```", raw, flags=re.I | re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # 3) cualquier ``` ... ```
    m = re.search(r"```\s*(\{[\s\S]*\}|\[[\s\S]*\])\s*```", raw, flags=re.I | re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            return obj if isinstance(obj, list) else None
        except Exception:
            pass

    # 4) primer array balanceado “best-effort”
    m = re.search(r"\[([\s\S]*)\]", raw)
    if m:
        txt = "[" + m.group(1) + "]"
        try:
            obj = json.loads(txt)
            return obj if isinstance(obj, list) else None
        except Exception:
            pass

    return None

def _structured_tactics_fallback(llm, asr_text: str, qa: str, style_text: str):
    """Si el modelo no devolvió JSON, fuerza un array JSON válido de 3 tácticas."""
    prompt = f"Return exactly THREE tactics that best satisfy this ASR.\nOutput ONLY JSON (no prose).\n\nASR:\n{asr_text}\n\nPrimary quality attribute: {qa}\nSelected style: {style_text or '(none)'}"
    try:
        arr = llm.with_structured_output(TACTICS_ARRAY_SCHEMA).invoke(prompt)
        if isinstance(arr, list) and len(arr) == 3:
            return arr
    except Exception as e:
        log.warning("structured tactics fallback failed: %s", e)
    return None

def _json_only_repair_pass(llm, *, asr_text: str, qa: str, style_text: str, md_preview: str):
    """Second-chance: ask the LLM to emit ONLY a JSON array of 3 tactics,
    using the markdown preview it already produced as context."""
    prompt = (
        "The following text should contain a JSON array of EXACTLY 3 architecture tactics "
        "but the JSON could not be parsed.\n\n"
        f"--- ORIGINAL TEXT (first 1500 chars) ---\n{md_preview[:1500]}\n---\n\n"
        f"ASR: {asr_text}\nQuality attribute: {qa}\nStyle: {style_text or '(none)'}\n\n"
        "Re-emit ONLY a valid JSON array (no prose, no code fence) with exactly 3 objects. "
        "Each object must have at minimum: name, rationale, categories (array), "
        "success_probability (float 0-1), rank (int 1-3)."
    )
    try:
        result = llm.invoke(prompt)
        raw = getattr(result, "content", str(result)).strip()
        arr = json.loads(raw)
        if isinstance(arr, list) and len(arr) >= 1:
            return arr
    except Exception as e:
        log.warning("_json_only_repair_pass failed: %s", e)
    return None

def _strip_tactics_sections(md: str) -> str:
    if not md:
        return md
    text = md
    for h in TACTICS_HEADINGS:
        text = re.sub(rf"(?is)\n+\s*{h}\s*:?.*$", "\n", text)
    return text.strip()

def _sanitize_plain_text(txt: str) -> str:
    txt = re.sub(r"```.*?```", "", txt, flags=re.S)
    txt = txt.replace("**", "")
    txt = re.sub(r"^\s*#\s*", "", txt, flags=re.M)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def _dedupe_snippets(docs_list, max_items=3, max_chars=400) -> str:
    """Toma documentos del retriever y arma texto sin duplicados/ruido."""
    seen, out = set(), []
    for d in docs_list:
        t = (d.page_content or "").strip().replace("\n", " ")
        t = t[:max_chars]
        if t and t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_items:
            break
    return "\n\n".join(out)

def _sanitize_mermaid(code: str) -> str:
    if not code:
        return ""

    code = code.replace("\r\n", "\n")

    # Recortar cualquier texto antes del primer "graph" o "flowchart"
    m = re.search(r"(graph\s+(?:LR|TD|BT|RL)[\s\S]*$)", code, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(flowchart[\s\S]*$)", code, flags=re.IGNORECASE)
    if m:
        code = m.group(1)
    else:
        # Si NO hay 'graph' ni 'flowchart', asumimos que son solo nodos/edges
        # y les anteponemos una cabecera por defecto.
        stripped = code.lstrip()
        if not stripped.lower().startswith(("graph ", "flowchart ")):
            code = "graph LR\n" + code

    # Reemplazar secuencias literales "\n" por espacios
    code = code.replace(r"\\n", " ")

    # Normalizar algunos caracteres unicode problemáticos
    replacements = {
        "≤": "<=",
        "≥": ">=",
        "→": "->",
        "⇒": "->",
        "↔": "<->",
        "—": "-",
        "–": "-",
        "\u00A0": " ",
        "“": '"',
        "”": '"',
        "’": "'",
    }
    for bad, good in replacements.items():
        code = code.replace(bad, good)

    lines = code.split("\n")
    new_nodes = []

    # 1) Patrones del tipo:  edge_cache ---|implements| "texto"
    edge_to_string = re.compile(
        r"^(\s*"               # indent + source id
        r"[A-Za-z_]\w*"        # id origen
        r"\s*)"
        r"(-{1,3}<?(?:>|)?)"   # operador de arista: --, -->, --- etc.
        r"\s*\|([^|]+)\|\s*"   # label
        r'"([^"]+)"\s*$'       # "texto" como destino
    )

    # 2) Patrones del tipo:  edge_cache --|MISS| cb["Circuit Breaker Proxy"]
    edge_with_inline_node = re.compile(
        r"^(\s*"               # indent + source id
        r"[A-Za-z_]\w*"
        r"\s*)"
        r"(-{1,3}<?(?:>|)?)"   # operador
        r"\s*\|([^|]+)\|\s*"   # label
        r"([A-Za-z_]\w*)\["    # id de nodo destino
        r"\"([^\"]+)\"\]\s*$"  # "texto" dentro del nodo
    )

    # Conjunto de nodos ya definidos (para no duplicar)
    defined_nodes = set()
    for line in lines:
        m_node = re.match(r"\s*([A-Za-z_]\w*)\s*\[", line)
        if m_node:
            defined_nodes.add(m_node.group(1))

    tactic_idx = 1

    for i, line in enumerate(lines):
        # Caso 1:  A ---|label| "texto"
        m1 = edge_to_string.match(line)
        if m1:
            indent, op, label, text = m1.groups()
            base_id = re.sub(r"\W+", "_", label.strip().lower()) or "note"
            node_id = f"tactic_{base_id}_{tactic_idx}"
            tactic_idx += 1

            # Definimos el nodo nuevo (nota/táctica)
            new_nodes.append(f'  {node_id}["{text}"]')
            defined_nodes.add(node_id)

            # Reemplazamos la línea original para que apunte al nodo
            lines[i] = f"{indent}{op} |{label.strip()}| {node_id}"
            continue

        # Caso 2:  A --|label| B["texto"]
        m2 = edge_with_inline_node.match(line)
        if m2:
            indent, op, label, node_id, text = m2.groups()

            # Reemplazamos la línea por edge hacia el id del nodo
            lines[i] = f"{indent}{op} |{label.strip()}| {node_id}"

            # Añadimos la definición del nodo si aún no existe
            if node_id not in defined_nodes:
                new_nodes.append(f'  {node_id}["{text}"]')
                defined_nodes.add(node_id)
            continue

    if new_nodes:
        lines.append("")
        lines.append("  %% Auto-generated tactic/note nodes")
        lines.extend(new_nodes)

    return "\n".join(lines).strip()
