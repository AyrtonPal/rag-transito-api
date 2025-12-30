from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import query_rag, co
import numpy as np

# Creamos la app de FastAPI
app = FastAPI(
    title="API tránsito y seguridad vial - RAG",
    description="API para consultar documentos usando RAG",
    version="1.0"
)

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    answer: str

# Historial semántico: lista de tuplas (embedding, respuesta)
semantic_history = []

# Función para calcular similitud coseno
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/ask", response_model=QuestionResponse)
def ask_question(data: QuestionRequest):
    """
    Recibe una pregunta, consulta el sistema RAG
    y devuelve la respuesta junto con las fuentes.
    """

    # Generamos embedding de la nueva pregunta
    embed_response = co.embed(
        texts=[data.question],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    question_embedding = np.array(embed_response.embeddings.float[0])

    # Buscamos en el historial semántico si hay una pregunta similar
    for hist_embedding, hist_answer in semantic_history:
        similarity = cosine_similarity(question_embedding, hist_embedding)
        if similarity > 0.8:
            # Pregunta equivalente encontrada, devolvemos la respuesta guardada
            return {"answer": hist_answer}

    # Si no hay coincidencia, llamamos a query_rag
    answer = query_rag(data.question)

    # Guardamos el embedding y la respuesta en el historial
    semantic_history.append((question_embedding, answer))

    return {"answer": answer}
