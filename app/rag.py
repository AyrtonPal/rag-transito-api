import os
from dotenv import load_dotenv
import cohere
import chromadb

# Cargamos las variables de entorno (.env)
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Inicializamos Cohere
co = cohere.ClientV2(
    api_key=os.getenv("CHROMA_COHERE_API_KEY")
)

# Conectamos a la base vectorial persistente
chroma_client = chromadb.PersistentClient(
    path="vector_db"
)

# Obtenemos la colección donde están los chunks
collection = chroma_client.get_collection(name="legal_documents")


def query_rag(question: str, top_k: int = 3) -> str:
    """
    Recibe una pregunta del usuario y devuelve
    una respuesta basada únicamente en los documentos.
    """

    # Genera el embedding de la pregunta
    response = co.embed(
        texts=[question],
        model="embed-multilingual-v3.0",
        input_type="search_query",
    )

    query_embedding = response.embeddings.float[0]



    # Busca los chunks más relevantes en la base vectorial
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    context_chunks = results["documents"][0]
    context = "\n\n".join(context_chunks)

    prompt = f"""
        SISTEMA:
        Sos un asistente experto en normativa de tránsito y seguridad vial.

        REGLAS DE RESPUESTA (OBLIGATORIAS):
        - Respondé UNICAMENTE usando el contexto proporcionado a continuación.
        - Si la respuesta NO se encuentra en el contexto, la salida DEBE SER EXACTAMENTE: No puedo brindarte esa información.
        - Está TERMINANTEMENTE PROHIBIDO inferir o asumir información.
        - SIEMPRE mantené un tono formal y profesional.
        - No utilizar emojis.
        - La respuesta SIEMPRE tiene que ser en español.

        Contexto:
        {context}

        Pregunta:
        {question}
        """
    
    print(context)

    response = co.chat(
        model="command-r-08-2024",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    answer = response.message.content[0].text
    return answer