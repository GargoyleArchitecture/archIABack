# archIABack

## Requisitos previos

- Python 3.11.x
- pip
- Graphviz (instalado a nivel de sistema)

---

## Configuración inicial (primera vez)

### 1. Instalar Python 3.11

**Verificar versión de Python:**

```bash
python3.11 --version
```

Si no está instalado (Fedora/RHEL):

```bash
sudo dnf install python3.11
```

**Instalar pip para Python 3.11:**

```bash
python3.11 -m ensurepip --upgrade
```

### 2. Instalar Graphviz

```bash
sudo dnf install graphviz
```

**Verificar versión de Graphviz:**

```bash
dot -V
```

### 3. Instalar Poetry

```bash
python3.11 -m pip install poetry
```

**Verificar versión de Poetry:**

```bash
python3.11 -m poetry --version
```

### 4. Instalar las dependencias del proyecto

Desde la carpeta `back/`:

```bash
cd back
python3.11 -m poetry env use python3.11
python3.11 -m poetry install
```

### 5. Configurar el archivo `.env`

Dentro de la carpeta `back/`, crear el archivo `.env` con  API Key:

```
OPENAI_API_KEY="Tu API Key va aquí"
```

### 6. Correr el servidor

```bash
python3.11 -m poetry run uvicorn src.main:app --port 8000
```

El servidor queda disponible en **http://localhost:8000**. La documentación interactiva de la API está en **http://localhost:8000/docs**.

---

## Administración del Módulo RAG (ChromaDB)

El sistema integra un motor de Generación Aumentada por Recuperación (RAG) para fundamentar las recomendaciones arquitectónicas en documentación técnica de referencia.

### 1. Gestión de la Base de Conocimientos (Vector Store)

Para inicializar o actualizar el almacén de datos vectoriales con nuevos documentos:

1. Ubicar los archivos PDF en el directorio `back/docs/`.
2. Ejecutar el proceso de indexación:
   ```bash
   cd back
   python3.11 -m poetry run python build_vectorstore.py
   ```

### 2. Auditoría y Visualización (ChromaDB Explorer)

Se proporciona una herramienta de exploración para validar la carga de datos y el comportamiento de la búsqueda semántica.

**Linux/Mac:**

1. Iniciar el servicio de exploración:
   ```bash
   cd back
   python3.11 -m poetry run python chroma_web.py
   ```
2. Acceder a la consola web en: [**http://localhost:8001**](http://localhost:8001)

**Windows:**

- Ejecutar el script de acceso directo: `back/start_chroma_web.bat`

### Capacidades del Explorador

- **Métricas de Estado**: Visualización del volumen de fragmentos (chunks) y fuentes registradas.
- **Pruebas de Recuperación**: Motor de búsqueda semántica para verificar la relevancia de los resultados.
- **Inspección de Metadatos**: Validación de la trazabilidad (fuente y página) de los segmentos almacenados.
