"""F11-T2: Carga del Shadow Agent.

Simula N sesiones de usuario concurrentes enviando T turnos secuenciales
al endpoint ``POST /message`` del Backend IA y mide la latencia del turno
principal. El objetivo es validar que activar el Shadow Agent (F3-T2) NO
degrade la SLO de turno principal: **p95 < 4 s**.

Diseño:
    - ``asyncio`` + ``httpx.AsyncClient`` para concurrencia real.
    - Cada sesión usa un ``user_id`` único (uuid4) y un ``session_id``
      derivado. Los 6 turnos por defecto exceden la cadencia del Shadow
      Agent (default ``every_n_turns=4``) para garantizar que al menos
      una evaluación shadow se dispare en background.
    - Modo ``--dry-run`` simula latencias plausibles (lognormal centrada
      en 2.5 s, mu=0.9, sigma=0.3) sin hacer red real. Útil en CI / dev.
    - Modo ``live`` apunta a un Backend IA real corriendo localmente.
    - Reporte HTML autocontenido (sin Jinja, sin deps extra) con
      percentiles globales y por turno + tabla de errores.

CLI:
    python -m tests.load.shadow_agent_load \\
        --users 50 --turns 6 \\
        --target http://localhost:8000 \\
        --report report.html

Exit codes:
    0 si p95 < 4 s (SLO cumplido)
    1 si SLO excedido
    2 si hubo errores fatales (> 50% requests fallidas)
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Optional

# httpx es dependencia ya presente en pyproject.toml (F3-T5).
import httpx


# ============================================================
#  Modelo de resultados
# ============================================================


@dataclass
class TurnResult:
    user_id: str
    turn: int
    latency_s: Optional[float]
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.latency_s is not None


@dataclass
class LoadResults:
    turns: list[TurnResult] = field(default_factory=list)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    @property
    def duration_s(self) -> float:
        if self.started_at is None or self.finished_at is None:
            return 0.0
        return self.finished_at - self.started_at


# ============================================================
#  Funciones puras (testeables)
# ============================================================


def percentile(values: list[float], pct: float) -> float:
    """Calcula el percentil (0-100) por interpolación lineal.

    Sin numpy/scipy. Retorna 0.0 si la lista está vacía. Compatible con
    el algoritmo "linear" estándar (mismo que numpy.percentile).
    """
    if not values:
        return 0.0
    if pct <= 0:
        return min(values)
    if pct >= 100:
        return max(values)
    sorted_vals = sorted(values)
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_vals[int(rank)]
    fraction = rank - low
    return sorted_vals[low] + (sorted_vals[high] - sorted_vals[low]) * fraction


def aggregate_results(results: LoadResults) -> dict:
    """Calcula métricas agregadas a partir de la lista de turnos.

    Retorna un dict con: total/successful/failed; p50/p90/p95/p99 global y
    por turno; lista (truncada) de errores.
    """
    successful = [t for t in results.turns if t.ok and t.latency_s is not None]
    failed = [t for t in results.turns if not t.ok]
    latencies = [t.latency_s for t in successful if t.latency_s is not None]

    by_turn: dict[int, dict] = {}
    if successful:
        turn_numbers = sorted({t.turn for t in successful})
        for turn_n in turn_numbers:
            t_lats = [
                t.latency_s for t in successful if t.turn == turn_n and t.latency_s is not None
            ]
            if t_lats:
                by_turn[turn_n] = {
                    "count": len(t_lats),
                    "p50":   percentile(t_lats, 50),
                    "p95":   percentile(t_lats, 95),
                    "p99":   percentile(t_lats, 99),
                    "mean":  statistics.mean(t_lats),
                    "max":   max(t_lats),
                }

    # Solo mostramos los primeros 20 errores en el reporte (suficiente).
    error_samples = [
        {"user_id": t.user_id, "turn": t.turn, "error": t.error}
        for t in failed[:20]
    ]

    return {
        "total":      len(results.turns),
        "successful": len(successful),
        "failed":     len(failed),
        "p50":  percentile(latencies, 50),
        "p90":  percentile(latencies, 90),
        "p95":  percentile(latencies, 95),
        "p99":  percentile(latencies, 99),
        "mean": statistics.mean(latencies) if latencies else 0.0,
        "max":  max(latencies) if latencies else 0.0,
        "duration_s": results.duration_s,
        "by_turn":    by_turn,
        "error_samples": error_samples,
    }


def make_html(report: dict, *, slo_seconds: float = 4.0) -> str:
    """Genera el reporte HTML como string autocontenido.

    Sin Jinja2 (cero deps extra). Diseño minimalista: tabla de métricas
    + breakdown por turno + tabla de errores. Verde si SLO cumplido,
    rojo si no.
    """
    slo_passed = report["p95"] < slo_seconds
    slo_color = "#16a34a" if slo_passed else "#dc2626"
    slo_label = "PASS" if slo_passed else "FAIL"

    by_turn_rows = ""
    for turn_n in sorted(report["by_turn"].keys()):
        t = report["by_turn"][turn_n]
        by_turn_rows += (
            f"<tr>"
            f"<td>{turn_n}</td>"
            f"<td>{t['count']}</td>"
            f"<td>{t['mean']:.3f}</td>"
            f"<td>{t['p50']:.3f}</td>"
            f"<td>{t['p95']:.3f}</td>"
            f"<td>{t['p99']:.3f}</td>"
            f"<td>{t['max']:.3f}</td>"
            f"</tr>"
        )

    if report["error_samples"]:
        error_rows = "".join(
            f"<tr><td>{escape(e['user_id'])}</td><td>{e['turn']}</td>"
            f"<td><code>{escape(e['error'])}</code></td></tr>"
            for e in report["error_samples"]
        )
        error_section = (
            "<h2>Errores (muestra de los primeros 20)</h2>"
            f"<table><thead><tr><th>user_id</th><th>turn</th><th>error</th></tr>"
            f"</thead><tbody>{error_rows}</tbody></table>"
        )
    else:
        error_section = "<p><em>Sin errores.</em></p>"

    return f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Shadow Agent — Load Report</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1f2937; }}
  h1 {{ margin-bottom: 0.25rem; }}
  .subtitle {{ color: #6b7280; margin-top: 0; }}
  .slo {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; color: white; background: {slo_color}; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #f3f4f6; font-weight: 600; }}
  code {{ background: #f3f4f6; padding: 0.125rem 0.375rem; border-radius: 0.25rem; font-size: 0.875em; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin: 1rem 0; }}
  .metric {{ background: #f9fafb; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e5e7eb; }}
  .metric-label {{ font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric-value {{ font-size: 1.5rem; font-weight: 700; color: #111827; }}
</style>
</head>
<body>
<h1>Shadow Agent — Load Report</h1>
<p class="subtitle">F11-T2 · SLO p95 &lt; {slo_seconds:.1f}s · <span class="slo">{slo_label}</span></p>

<div class="metric-grid">
  <div class="metric"><div class="metric-label">Total requests</div><div class="metric-value">{report['total']}</div></div>
  <div class="metric"><div class="metric-label">Successful</div><div class="metric-value">{report['successful']}</div></div>
  <div class="metric"><div class="metric-label">Failed</div><div class="metric-value">{report['failed']}</div></div>
  <div class="metric"><div class="metric-label">Duration</div><div class="metric-value">{report['duration_s']:.2f}s</div></div>
</div>

<h2>Latency (global, segundos)</h2>
<table>
  <thead><tr><th>mean</th><th>p50</th><th>p90</th><th>p95</th><th>p99</th><th>max</th></tr></thead>
  <tbody><tr>
    <td>{report['mean']:.3f}</td>
    <td>{report['p50']:.3f}</td>
    <td>{report['p90']:.3f}</td>
    <td>{report['p95']:.3f}</td>
    <td>{report['p99']:.3f}</td>
    <td>{report['max']:.3f}</td>
  </tr></tbody>
</table>

<h2>Por turno</h2>
<table>
  <thead><tr><th>Turn</th><th>n</th><th>mean</th><th>p50</th><th>p95</th><th>p99</th><th>max</th></tr></thead>
  <tbody>{by_turn_rows}</tbody>
</table>

{error_section}
</body></html>
"""


# ============================================================
#  Async runner
# ============================================================


async def _send_turn(
    client: httpx.AsyncClient,
    target: str,
    *,
    user_id: str,
    session_id: str,
    turn: int,
    text: str,
) -> TurnResult:
    """Envía un turno y mide la latencia. Captura todos los errores."""
    started = time.perf_counter()
    try:
        # POST /message usa form-data (ver back/src/main.py F2-T2).
        resp = await client.post(
            f"{target.rstrip('/')}/message",
            data={
                "message": text,
                "session_id": session_id,
                "user_id": user_id,
                "mode": "professional",
            },
            timeout=30.0,
        )
        elapsed = time.perf_counter() - started
        if resp.status_code >= 400:
            return TurnResult(
                user_id=user_id, turn=turn,
                latency_s=None, error=f"HTTP {resp.status_code}",
            )
        return TurnResult(user_id=user_id, turn=turn, latency_s=elapsed)
    except httpx.RequestError as exc:
        return TurnResult(
            user_id=user_id, turn=turn,
            latency_s=None, error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # pragma: no cover (defensive)
        return TurnResult(
            user_id=user_id, turn=turn,
            latency_s=None, error=f"{type(exc).__name__}: {exc}",
        )


async def _simulate_dry(
    *, user_id: str, turn: int, rng: random.Random,
) -> TurnResult:
    """Simula una latencia plausible sin hacer red. Lognormal mu=0.9 sigma=0.3
    produce p50 ≈ 2.46 s, p95 ≈ 4.0 s (en el borde del SLO).
    """
    # Pequeño await para mantener la concurrencia real.
    latency = rng.lognormvariate(0.9, 0.3)
    await asyncio.sleep(min(latency, 0.05))  # acelera tiempo de simulación
    return TurnResult(user_id=user_id, turn=turn, latency_s=latency)


async def _simulate_session(
    client: Optional[httpx.AsyncClient],
    target: str,
    *,
    user_id: str,
    turns: int,
    dry_run: bool,
    rng: random.Random,
) -> list[TurnResult]:
    """Ejecuta T turnos secuenciales para un user_id."""
    session_id = f"load-{user_id[:8]}"
    out: list[TurnResult] = []
    for turn in range(1, turns + 1):
        text = f"Pregunta de carga turno {turn} para {user_id[:8]}"
        if dry_run:
            res = await _simulate_dry(user_id=user_id, turn=turn, rng=rng)
        else:
            assert client is not None
            res = await _send_turn(
                client, target,
                user_id=user_id, session_id=session_id, turn=turn, text=text,
            )
        out.append(res)
    return out


async def run_load(
    *,
    users: int,
    turns: int,
    target: str,
    dry_run: bool = False,
    seed: Optional[int] = None,
) -> LoadResults:
    """Orquesta la carga: N usuarios concurrentes × T turnos secuenciales."""
    rng = random.Random(seed)
    results = LoadResults()
    results.started_at = time.perf_counter()

    if dry_run:
        tasks = [
            _simulate_session(
                None, target,
                user_id=str(uuid.uuid4()), turns=turns,
                dry_run=True, rng=rng,
            )
            for _ in range(users)
        ]
        per_user = await asyncio.gather(*tasks)
    else:
        # Reusa una sola AsyncClient — httpx maneja el pool internamente.
        async with httpx.AsyncClient() as client:
            tasks = [
                _simulate_session(
                    client, target,
                    user_id=str(uuid.uuid4()), turns=turns,
                    dry_run=False, rng=rng,
                )
                for _ in range(users)
            ]
            per_user = await asyncio.gather(*tasks)

    for batch in per_user:
        results.turns.extend(batch)
    results.finished_at = time.perf_counter()
    return results


# ============================================================
#  CLI entry point
# ============================================================


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="F11-T2: Carga del Shadow Agent",
    )
    parser.add_argument("--users", type=int, default=50,
                        help="Número de sesiones concurrentes (default: 50)")
    parser.add_argument("--turns", type=int, default=6,
                        help="Número de turnos por sesión (default: 6; supera la cadencia del Shadow Agent en 4)")
    parser.add_argument("--target", type=str, default="http://localhost:8000",
                        help="URL base del Backend IA (default: http://localhost:8000)")
    parser.add_argument("--report", type=str, default="shadow_load_report.html",
                        help="Ruta del reporte HTML (default: shadow_load_report.html)")
    parser.add_argument("--slo-seconds", type=float, default=4.0,
                        help="Umbral de SLO p95 en segundos (default: 4.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simula latencias lognormales sin hacer red.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla para --dry-run (determinismo en CI).")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    print(
        f"[load] Iniciando: users={args.users} turns={args.turns} "
        f"target={args.target} dry_run={args.dry_run}",
        file=sys.stderr,
    )

    results = asyncio.run(run_load(
        users=args.users, turns=args.turns, target=args.target,
        dry_run=args.dry_run, seed=args.seed,
    ))

    report = aggregate_results(results)
    html = make_html(report, slo_seconds=args.slo_seconds)

    report_path = Path(args.report)
    report_path.write_text(html, encoding="utf-8")
    print(f"[load] Reporte HTML escrito en: {report_path.resolve()}", file=sys.stderr)

    # Resumen ASCII compacto.
    print(
        f"[load] total={report['total']} ok={report['successful']} "
        f"fail={report['failed']} "
        f"p50={report['p50']:.3f}s p95={report['p95']:.3f}s "
        f"p99={report['p99']:.3f}s max={report['max']:.3f}s",
        file=sys.stderr,
    )

    # Exit codes según SLO.
    failure_rate = report["failed"] / report["total"] if report["total"] else 1.0
    if failure_rate > 0.5:
        print("[load] FATAL: >50% requests fallaron.", file=sys.stderr)
        return 2
    if report["p95"] >= args.slo_seconds:
        print(
            f"[load] SLO FAIL: p95={report['p95']:.3f}s >= {args.slo_seconds:.1f}s",
            file=sys.stderr,
        )
        return 1
    print(
        f"[load] SLO PASS: p95={report['p95']:.3f}s < {args.slo_seconds:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
