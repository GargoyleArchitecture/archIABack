"""Tests puros de F11-T2: percentile, aggregate, make_html.

NO ejecuta carga real ni levanta el servidor — sólo verifica las
funciones que no tocan red.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Path setup para ejecutar desde la raíz del repo.
BACK_DIR = Path(__file__).resolve().parents[2]
if str(BACK_DIR) not in sys.path:
    sys.path.insert(0, str(BACK_DIR))

from tests.load.shadow_agent_load import (  # noqa: E402
    LoadResults,
    TurnResult,
    aggregate_results,
    make_html,
    percentile,
)


# ============================================================
#  percentile
# ============================================================


def test_percentile_lista_vacia_retorna_cero():
    assert percentile([], 50) == 0.0
    assert percentile([], 99) == 0.0


def test_percentile_un_solo_valor():
    assert percentile([1.5], 50) == 1.5
    assert percentile([1.5], 0) == 1.5
    assert percentile([1.5], 100) == 1.5


def test_percentile_p50_es_mediana():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(vals, 50) == pytest.approx(3.0)


def test_percentile_p95_es_casi_max():
    vals = list(range(1, 101))  # 1..100
    # Algoritmo "linear" (numpy default): p95 de 1..100 = 95.05
    assert percentile(vals, 95) == pytest.approx(95.05)


def test_percentile_pct_0_y_100_son_min_max():
    vals = [3.0, 1.0, 4.0, 1.0, 5.0]
    assert percentile(vals, 0) == 1.0
    assert percentile(vals, 100) == 5.0


# ============================================================
#  aggregate_results
# ============================================================


def test_aggregate_separa_ok_y_failed():
    r = LoadResults()
    r.started_at = 0.0
    r.finished_at = 10.0
    r.turns = [
        TurnResult(user_id="u1", turn=1, latency_s=1.0),
        TurnResult(user_id="u1", turn=2, latency_s=2.0),
        TurnResult(user_id="u2", turn=1, latency_s=None, error="HTTP 500"),
        TurnResult(user_id="u3", turn=1, latency_s=3.0),
    ]
    agg = aggregate_results(r)
    assert agg["total"] == 4
    assert agg["successful"] == 3
    assert agg["failed"] == 1
    assert agg["duration_s"] == 10.0
    assert agg["error_samples"][0]["error"] == "HTTP 500"


def test_aggregate_breakdown_por_turno():
    r = LoadResults()
    r.started_at = 0.0
    r.finished_at = 1.0
    r.turns = [
        TurnResult(user_id="u1", turn=1, latency_s=1.0),
        TurnResult(user_id="u2", turn=1, latency_s=3.0),
        TurnResult(user_id="u1", turn=2, latency_s=2.0),
    ]
    agg = aggregate_results(r)
    assert 1 in agg["by_turn"]
    assert 2 in agg["by_turn"]
    assert agg["by_turn"][1]["count"] == 2
    assert agg["by_turn"][2]["count"] == 1


def test_aggregate_sin_resultados_no_crashea():
    r = LoadResults()
    r.started_at = 0.0
    r.finished_at = 0.0
    agg = aggregate_results(r)
    assert agg["total"] == 0
    assert agg["p95"] == 0.0
    assert agg["mean"] == 0.0
    assert agg["by_turn"] == {}


# ============================================================
#  make_html
# ============================================================


def test_make_html_contiene_pass_cuando_p95_bajo_slo():
    agg = {
        "total": 10, "successful": 10, "failed": 0,
        "p50": 1.0, "p90": 2.0, "p95": 3.5, "p99": 3.8, "mean": 1.5, "max": 3.9,
        "duration_s": 5.0, "by_turn": {}, "error_samples": [],
    }
    html = make_html(agg, slo_seconds=4.0)
    assert "PASS" in html
    assert "<!doctype html>" in html
    assert "3.500" in html  # p95 formateado


def test_make_html_contiene_fail_cuando_p95_excede_slo():
    agg = {
        "total": 10, "successful": 10, "failed": 0,
        "p50": 3.0, "p90": 4.5, "p95": 5.0, "p99": 6.0, "mean": 3.2, "max": 6.5,
        "duration_s": 5.0, "by_turn": {}, "error_samples": [],
    }
    html = make_html(agg, slo_seconds=4.0)
    assert "FAIL" in html


def test_make_html_renderiza_errores_con_escapado_xss():
    agg = {
        "total": 1, "successful": 0, "failed": 1,
        "p50": 0, "p90": 0, "p95": 0, "p99": 0, "mean": 0, "max": 0,
        "duration_s": 0.0, "by_turn": {},
        "error_samples": [
            {"user_id": "<script>alert(1)</script>", "turn": 1, "error": "<bad>"},
        ],
    }
    html = make_html(agg)
    # El payload XSS no debe aparecer como tag real.
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
