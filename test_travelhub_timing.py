"""
Test de timing — prompt TravelHub (ASR + estilos + tácticas + diagrama)

El endpoint ahora devuelve SSE (text/event-stream). Este script mide:
  - t_first : tiempo hasta el primer evento `partial` (lo que el usuario ve primero)
  - t_complete : tiempo hasta el evento `complete` (toda la información disponible)
  - t_done : tiempo hasta el sentinel `[DONE]` (cierre limpio del stream)

Uso:
    python test_travelhub_timing.py [--url http://localhost:8000] [--session mi-session]
"""

import argparse
import json
import sys
import time
import uuid

try:
    import requests
except ImportError:
    sys.exit("Instala requests: pip install requests")

PROMPT = """Para TravelHub, plataforma de reservas hoteleras en 6 países de Latinoamérica \
que actualmente tiene búsquedas de 3-5 segundos, colapso en temporadas altas \
(85% CPU, rechazo de 40% de peticiones) y 98.5% de uptime con toda la infraestructura \
en una sola región AWS, necesito lo siguiente en una sola respuesta:

1. Un ASR de escalabilidad (el sistema debe pasar de 150 a 800 TPM con autoescalado horizontal)
2. Los estilos arquitectónicos recomendados para ese ASR
3. Las tácticas de escalabilidad correspondientes
4. Un diagrama de arquitectura de alto nivel que refleje el estilo y las tácticas seleccionadas

El sistema conecta con 15+ proveedores PMS distintos y debe soportar picos de hasta \
3,600 usuarios concurrentes simultáneos entre los 6 países."""


def separator(char="─", width=72):
    print(char * width)


def main():
    parser = argparse.ArgumentParser(description="Timing test — TravelHub prompt")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL del backend")
    parser.add_argument("--session", default=None, help="Session ID (genera uno aleatorio si se omite)")
    args = parser.parse_args()

    session_id = args.session or f"timing-test-{uuid.uuid4().hex[:8]}"
    endpoint = f"{args.url}/message"

    separator("═")
    print("  ARCHIA — TEST DE TIMING  |  TravelHub full pipeline")
    separator("═")
    print(f"  Endpoint   : {endpoint}")
    print(f"  Session ID : {session_id}")
    print(f"  Prompt     : {len(PROMPT)} caracteres")
    separator()

    payload = {
        "message": PROMPT,
        "session_id": session_id,
    }

    print("  Enviando request...\n")
    t_start = time.perf_counter()

    try:
        response = requests.post(endpoint, data=payload, timeout=300, stream=True)
    except requests.exceptions.ConnectionError:
        print(f"  ERROR: No se pudo conectar a {endpoint}")
        print("  Asegurate de que el servidor está corriendo (poetry run uvicorn src.main:app --reload)")
        sys.exit(1)
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - t_start
        print(f"  TIMEOUT después de {elapsed:.1f} s — el servidor no respondió en 5 min")
        sys.exit(1)

    if response.status_code != 200:
        print(f"  HTTP {response.status_code} — respuesta inesperada")
        print(f"  Body: {response.text[:500]}")
        sys.exit(1)

    # ── Parse SSE stream ──────────────────────────────────────────────────
    t_first: float | None = None
    t_complete: float | None = None
    t_done: float | None = None
    final_data: dict = {}
    chunks_seen: list[str] = []

    for raw_line in response.iter_lines():
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data:"):
            continue
        payload_str = line[5:].strip()

        if payload_str == "[DONE]":
            t_done = time.perf_counter()
            break

        try:
            event = json.loads(payload_str)
        except json.JSONDecodeError:
            continue

        etype = event.get("type")
        now = time.perf_counter()

        if etype == "partial":
            node = event.get("node", "?")
            chunks_seen.append(node)
            elapsed = now - t_start
            print(f"  ↳ partial [{node:20s}] : {elapsed:.2f} s", flush=True)
            if t_first is None:
                t_first = now

        elif etype == "complete":
            t_complete = now
            final_data = event

        elif etype == "error":
            print(f"  ERROR del servidor: {event.get('message', '?')}")
            sys.exit(1)

    t_end = t_done or t_complete or time.perf_counter()

    separator()
    print(f"  Tiempo hasta primer chunk  : {(t_first - t_start):.2f} s" if t_first else "  Primer chunk             : (no recibido)")
    print(f"  Tiempo hasta 'complete'    : {(t_complete - t_start):.2f} s" if t_complete else "  complete                  : (no recibido)")
    print(f"  Tiempo total (→ [DONE])    : {(t_end - t_start):.2f} s")
    separator()

    if not final_data:
        print("  ERROR: no se recibió evento 'complete'")
        sys.exit(1)

    end_message = final_data.get("endMessage", "")
    diagram = final_data.get("diagram") or {}
    suggestions = final_data.get("suggestions", [])
    turn_messages = final_data.get("messages", [])

    print(f"  endMessage  : {len(end_message)} caracteres")
    print(f"  turn_msgs   : {len(turn_messages)} mensaje(s) de turno")
    print(f"  suggestions : {len(suggestions)}")
    print(f"  diagram.ok  : {diagram.get('ok', False)}")
    if diagram.get("ok"):
        svg_b64 = diagram.get("svg_b64") or ""
        dot = diagram.get("dot") or ""
        print(f"  diagram.svg : {len(svg_b64)} bytes (base64)")
        print(f"  diagram.dot : {len(dot)} chars")

    separator()
    print("  PREVIEW — endMessage (primeros 600 chars):")
    separator("·")
    print(end_message[:600].strip() or "(vacío)")
    separator("·")

    if suggestions:
        print("  SUGERENCIAS:")
        for s in suggestions[:3]:
            print(f"    • {str(s)[:100]}")
        separator()

    total = t_end - t_start
    print(f"\n  Resultado final: {'OK' if end_message else 'WARNING: endMessage vacío'}")
    print(f"  Tiempo total  : {total:.2f} s\n")
    separator("═")


if __name__ == "__main__":
    main()
