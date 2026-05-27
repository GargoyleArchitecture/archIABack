# Documentación del Backend - ArchIA

Esta carpeta contiene toda la documentación técnica relacionada con el backend del sistema ArchIA.

## 📚 Contenido

### 1. [Arquitectura.md](./Arquitectura.md)

**Especificación Técnica de Arquitectura Completa (v4.0)**

Incluye:

- Arquitectura del sistema
- Grafo de estados (LangGraph)
- Módulos principales (RAG, Diagramas, Recomendación)
- Modelo de persistencia
- Flujo de datos
- Jerarquía de archivos

**Audiencia**: Desarrolladores, arquitectos, mantenedores del sistema

---

### 2. [info_interfaz.md](./info_interfaz.md)

**Documentación de la Interfaz Web ChromaDB Explorer**

Incluye:

- Qué es y por qué se creó
- Arquitectura de la interfaz
- Requisitos para ejecutarla
- Guía de uso paso a paso
- Casos de uso y troubleshooting

**Audiencia**: Desarrolladores que necesiten explorar/depurar la base de datos vectorial

---

### 3. diagram_pipeline.md

**Guía del Pipeline de Generación de Diagramas**

Incluye:

- Arquitectura del renderizado (IR)
- Niveles de detalle y expansión progresiva
- Guía para importar en draw.io

**Audiencia**: Desarrolladores trabajando en la visualización o exportación de diagramas

---

## 🎯 Propósito de esta Carpeta

Esta carpeta (`docs_back/`) centraliza toda la documentación técnica del backend para:

1. **Onboarding** de nuevos desarrolladores
2. **Referencia** durante el desarrollo
3. **Mantenimiento** y troubleshooting
4. **Documentación de decisiones** arquitectónicas

---

## 📝 Mantenimiento

**Responsable**: Equipo de desarrollo ArchIA
**Frecuencia de actualización**: Cada vez que se agregue una nueva funcionalidad o se modifique la arquitectura

### Versionado de Documentos

Los documentos incluyen metadatos de versión en su encabezado:

```markdown
> **Versión**: 4.0
> **Fecha**: Abril 2026
```

---

## 🔗 Documentación Relacionada

- [README principal del proyecto](../../README.md)
- [Documentación del frontend](../../front/README.md) (si existe)
- [Tutorial de uso](../../Tutorial_MV.txt)

---

**Última actualización**: Febrero 2026
