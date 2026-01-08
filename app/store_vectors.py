import os
import json
import time
import cohere
import chromadb
from dotenv import load_dotenv

# Cargamos las variables de entorno
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

CHUNKS_PATH = "data/chunks.json"

# Inicializamos ChromaDB
chroma_client = chromadb.PersistentClient(
    path="vector_db"
)

collection = chroma_client.get_or_create_collection(
    name="legal_documents"
)

# Tamaño de batch: cantidad de chunks que se envían juntos a Cohere
BATCH_SIZE = 10
# Tiempo de espera entre batches
SLEEP_TIME = 3


def load_chunks():
    """Carga los chunks del archivo JSON"""
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def store_vectors():
    """Genera embeddings en batches y los guarda en la base vectorial"""
    chunks = load_chunks()

    print("Generando embeddings...")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i: i + BATCH_SIZE]

        batch_texts = [chunk["content"] for chunk in batch_chunks]
        batch_metadatas = [{"document": chunk["document"], "chunk_id": chunk["chunk_id"]}
                           for chunk in batch_chunks]
        batch_ids = [f"{chunk['document']}_{chunk['chunk_id']}_{i + idx}"
                     for idx, chunk in enumerate(batch_chunks)]

        # Genera embeddings del batch completo
        response = co.embed(
            texts=batch_texts,
            model="embed-multilingual-v3.0",
            input_type="search_document"
        )
        embeddings = response.embeddings

        # Guardamos en Chroma
        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

        print(f"Batch generado y guardado: {i + len(batch_chunks)}/{len(chunks)}")

        # Espera para no superar límite de tokens/minuto
        time.sleep(SLEEP_TIME)

    print("Base vectorial creada y persistida")


if __name__ == "__main__":
    store_vectors()
