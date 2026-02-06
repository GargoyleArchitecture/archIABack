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

poetry run uvicorn src.main:app --port 8000
