from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import query_rag, co, query_rag_from_known_chunk
import numpy as np

# Creamos la app de FastAPI
app = FastAPI(
    title="API tránsito y seguridad vial - RAG",
    description="API para consultar documentos usando RAG",
    version="1.0"
)

class QuestionRequest(BaseModel):
    question: str

class Source(BaseModel):
    document: str
    chunk_id: int

class QuestionResponse(BaseModel):
    answer: str
    source: Source

# Historial semántico: lista de tuplas (embedding, respuesta)
semantic_history = []

# Función para calcular similitud coseno
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/ask", response_model=QuestionResponse)
def ask_question(data: QuestionRequest):

    if not data.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía")

    # Generamos embedding de la nueva pregunta
    embed_question = co.embed(
        texts=[data.question],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    embedding_vector = np.array(embed_question.embeddings.float[0])

    # Buscar en historial semántico
    for item in semantic_history:
        similarity = cosine_similarity(embedding_vector, item["embedding"])
        if similarity > 0.85:
            return query_rag_from_known_chunk(
            chunk_text=item["chunk_text"],
            source=item["source"],
            question=data.question
        )

    # Si no hay coincidencia
    embedding_query = embed_question.embeddings.float[0]
    rag_result = query_rag(embedding_query , data.question)

    # Guardamos el embedding y la respuesta
    semantic_history.append({
        "embedding": embedding_vector,
        "chunk_text": rag_result["chunk_text"],
        "source": rag_result["source"]
    })

    return rag_result
