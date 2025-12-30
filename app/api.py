from fastapi import FastAPI
from pydantic import BaseModel
from app.rag import query_rag

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

@app.post("/ask", response_model=QuestionResponse)
def ask_question(data: QuestionRequest):
    """
    Recibe una pregunta, consulta el sistema RAG
    y devuelve la respuesta junto con las fuentes.
    """

    answer = query_rag(data.question)
    return {"answer": answer}
