# Design Dossier — test-project

**Fase:** ASR  ·  **Iteración:** 1

## Restricciones del proyecto
_(ninguna aún)_

## Decisiones activas

### ASR  (id: 01FIXTURE000000000ASR00001, QA: latencia, fase: ASR)
**Resumen:** Sistema de pagos con latencia inferior a 200ms bajo carga normal

- **Fuente:** usuario final
- **Estímulo:** solicitud de pago
- **Entorno:** producción bajo carga normal
- **Artefacto:** servicio de pagos
- **Respuesta:** procesar pago y retornar confirmación
- **Medida de Respuesta:** p95 < 200ms @ 5k RPS
- **Dominio:** pagos financieros

**Justificación:** El equipo requiere alta velocidad en transacciones de pago para cumplir el SLA.

**Fuentes:** ADD 3.0 Patterns (p.42)

### Estilo
_(ninguna aún)_

### Tácticas
_(ninguna aún)_

### Diagrama
_(ninguna aún)_

## Historial de fase
- Iteración 1: INTAKE → ASR (user_request) el 2024-01-15

## Historial (reemplazadas / rechazadas)

### Reemplazadas
_(ninguna aún)_

### Rechazadas
_(ninguna aún)_
