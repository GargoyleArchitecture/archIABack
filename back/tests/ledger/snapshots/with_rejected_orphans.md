# Design Dossier — test-project

**Fase:** STYLE  ·  **Iteración:** 2

## Restricciones del proyecto
_(ninguna aún)_

## Decisiones activas

### ASR  (id: 01FIXTURE000000000ASR00001, QA: latencia, fase: ASR)
**Resumen:** Sistema de pagos con latencia inferior a 200ms

- **Fuente:** usuario final
- **Estímulo:** solicitud de pago
- **Entorno:** producción
- **Artefacto:** servicio de pagos
- **Respuesta:** procesar pago
- **Medida de Respuesta:** p95 < 200ms
- **Dominio:** pagos

**Justificación:** SLA de negocio.

### Estilo
_(ninguna aún)_

### Tácticas  (id: 01FIXTURE000000000TAC00001)
1. Cache local — Reducir latencia  (success 0.80, rank 1)

### Diagrama
_(ninguna aún)_

## Historial de fase
- Iteración 1: INTAKE → ASR (user_request) el 2024-01-15
- Iteración 2: ASR → STYLE (agent_suggestion_accepted) el 2024-01-15

## Historial (reemplazadas / rechazadas)

### Reemplazadas
_(ninguna aún)_

### Rechazadas
- STYLE **Arquitectura por capas** (id: 01FIXTURE000000000STY00001, iteración 2) — rejected: "No encaja con el stack de Kafka y procesamiento reactivo"
