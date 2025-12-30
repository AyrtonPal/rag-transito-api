# Rag Transito API
Este proyecto consiste en una API desarrollada con FastAPI que implementa un sistema RAG (Retrieval Augmented Generation) para responder preguntas relacionadas con normativa de tránsito y seguridad vial.
La API utiliza documentos PDF como fuente de conocimiento. Estos documentos se procesan, se dividen en fragmentos (chunks), se transforman en embeddings mediante Cohere y se almacenan en una base vectorial persistente.
Las respuestas generadas por la API se basan exclusivamente en la información contenida en los documentos cargados.

## Pasos para levantar el proyecto

### Archivo de variable de entorno (Obligatorio)
- Se debe crear un archivo llamado ```.env``` en la raíz del proyecto con la siguiente estructura: ```COHERE_API_KEY=""```.

Para obtener esta clave es necesario crear una cuenta en Cohere y generar una API Key desde su panel.

### Instalacion de dependencias (Obligatorio)
- Se deben instalar las dependencias del proyecto ejecutando: ```pip install -r requirements.txt```

### Agregar documentacion adicional (Opcional)
Si desea agregar o modificar la documentacion tiene que seguir los siguientes pasos
- Debe agregar los documentos en la carpeta con la siguiente ruta ```data\pdfs```
- Transformar PDFs a texto: ```python pdf_to_text.py```
- Generar chunks: ```python chunking.py```

### Generar y almacenar embeddings (Obligatorio)
- Ejecutar el siguiente comando: ```python store_vectors.py```

## Levantar la API
- Una vez finalizados todos los pasos anteriores, la API ya está lista para ser utilizada. Solo necesitas correr el siguiente comando:

```python -m uvicorn app.api:app --reload --host 127.0.0.1 --port 8000```

## Acceso a la documentacion interactiva
- Con la API en ejecución, se puede acceder a la documentacion interactiva de Swagger desde el siguiente enlace: http://127.0.0.1:8000/docs
