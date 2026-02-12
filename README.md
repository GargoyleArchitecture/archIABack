# archIABack

## Pasos para correr el back:

**Se requiere tener python 3.11.x y pip instalados**

1. Instalar Poetry:

pip install Poetry

2. Clonar el repositorio y abrir una terminal dentro del repo

3. Instalar las dependecias:

Poetry install

4. Dentro del back, crear el archivo ".env" y colocar la API key:

OPENAI_API_KEY="Tu API Key va aquí"

5. Entrar a la carpeta back:

cd back/

6. Correr el programa:

```bash
poetry run uvicorn src.main:app --port 8000
```

---

## Administración del Módulo RAG (ChromaDB)

El sistema integra un motor de Generación Aumentada por Recuperación (RAG) para fundamentar las recomendaciones arquitectónicas en documentación técnica de referencia.

### 1. Gestión de la Base de Conocimientos (Vector Store)

Para inicializar o actualizar el almacén de datos vectoriales con nuevos documentos:

1.  Ubicar los archivos PDF en el directorio `back/docs/`.
2.  Ejecutar el proceso de indexación:
    ```bash
    cd back
    poetry run python build_vectorstore.py
    ```

### 2. Auditoría y Visualización (ChromaDB Explorer)

Se proporciona una herramienta de exploración para validar la carga de datos y el comportamiento de la búsqueda semántica.

**Windows:**

- Ejecutar el script de acceso directo: [`back/start_chroma_web.bat`](file:///c:/Users/mgs05/Documents/GargoyleArchitecture/NewArchIA/archIABack/back/start_chroma_web.bat)

**Otros sistemas operativos:**

1.  Iniciar el servicio de exploración:
    ```bash
    cd back
    poetry run python chroma_web.py
    ```
2.  Acceder a la consola web en: [**http://localhost:8001**](http://localhost:8001)

### Capacidades del Explorador

- **Métricas de Estado**: Visualización del volumen de fragmentos (chunks) y fuentes registradas.
- **Pruebas de Recuperación**: Motor de búsqueda semántica para verificar la relevancia de los resultados.
- **Inspección de Metadatos**: Validación de la trazabilidad (fuente y página) de los segmentos almacenados.
