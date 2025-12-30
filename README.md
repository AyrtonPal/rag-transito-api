# Rag Transito API
Este proyecto consiste en una API desarrollada con FastAPI que implementa un sistema RAG (Retrieval Augmented Generation) para responder preguntas relacionadas con normativa de tránsito y seguridad vial.
La API utiliza documentos PDF como fuente de conocimiento. Estos documentos se procesan, se dividen en fragmentos (chunks), se transforman en embeddings mediante Cohere y se almacenan en una base vectorial persistente.
Las respuestas generadas por la API se basan exclusivamente en la información contenida en los documentos cargados.

## Pasos para levantar el proyecto

### Archivo de variable de entorno
- Se debe crear un archivo llamado .env en la raíz del proyecto con la siguiente estructura: ```COHERE_API_KEY=""```. Para obtener esta clave es necesario crear una cuenta en Cohere y generar una API Key desde su panel.

### Instalacion de dependencias
- Se deben instalar las dependencias del proyecto ejecutando: ```pip install -r requirements.txt```

### Generar y almacenar embeddings
- Ejecutar el siguiente comando: ```python store_vectors.py```

Este script realiza las siguientes tareas:

- Lee los chunks desde el archivo data/chunks.json
- Genera embeddings utilizando Cohere o SentenceTransformer
- Guarda los vectores en una base vectorial persistente ubicada en la carpeta vector_db

## Levantar la API
- Una vez fin finalizados todos los pasos anteriores, la API ya está lista para ser utilizada. Solo necesitas correr el siguiente comando:

```python -m uvicorn app.api:app --reload --host 127.0.0.1 --port 8000```

## Acceso a la documentacion interactiva
- Con la API en ejecución, se puede acceder a la documentacion interactiva de Swagger desde el siguiente enlace: http://127.0.0.1:8000/docs
