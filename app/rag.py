import os
from dotenv import load_dotenv
import cohere
import chromadb

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

co = cohere.ClientV2(
    api_key=os.getenv("CHROMA_COHERE_API_KEY")
)

chroma_client = chromadb.PersistentClient(
    path="vector_db"
)

collection = chroma_client.get_collection(name="legal_documents")


# Función única para construir el prompt
def build_prompt(context: str, question: str) -> str:
    return f"""
    SISTEMA:
    Sos un asistente experto en normativa de tránsito y seguridad vial.

    REGLAS DE RESPUESTA (OBLIGATORIAS):
    - Respondé UNICAMENTE usando el contexto proporcionado.
    - Buscá que la respuesta sea concisa y directa.
    - Está TERMINANTEMENTE PROHIBIDO inferir o asumir información.
    - SIEMPRE mantené un tono formal y profesional.
    - No utilizar emojis.
    - La respuesta SIEMPRE tiene que ser en español.

    Contexto:
    {context}

    Pregunta:
    {question}
    """


# Función única de generación
def generate_answer(prompt: str) -> str:
    response = co.chat(
        model="command-r-08-2024",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return response.message.content[0].text

# RAG optimizado con fuente conocida
def query_rag_from_known_chunk(chunk_text: str, source: dict, question: str):
    """
    RAG individual: busca solo en el chunk que le pases por parametro
    """
    print("Pasó por query_rag_from_known_chunk")
    prompt = build_prompt(chunk_text, question)
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "source": source
    }

def query_rag(query_embedding, question: str):
    """
    RAG completo: busca en toda la base vectorial
    """
    print("Pasó por query_rag")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "metadatas"]
    )

    context_chunks = results["documents"][0]
    context_metadatas = results["metadatas"][0]

    rerank_response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=question,
        documents=context_chunks,
        top_n=3
    )

    top_index = rerank_response.results[0].index
    source_metadata = context_metadatas[top_index]

    reranked_chunks = [
        context_chunks[result.index]
        for result in rerank_response.results
    ]

    context = "\n\n".join(reranked_chunks)
    prompt = build_prompt(context, question)
    answer = generate_answer(prompt)

    return {
        "answer": answer,
        "source": {
            "document": source_metadata.get("document"),
            "chunk_id": source_metadata.get("chunk_id")
        },
        "chunk_text": context_chunks[top_index]
    }
