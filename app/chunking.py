import os
import re
import json

TXT_DIR = "data/txts"
OUTPUT_PATH = "data/chunks.json"
MAX_CHARS = 2500


def split_large_text(text: str):
    """
    Divide un texto largo en fragmentos más chicos
    cuando supera el tamaño máximo permitido.
    """
    chunks = []
    start = 0

    # Mientras no hayamos recorrido todo el texto
    while start < len(text):
        end = start + MAX_CHARS
        chunks.append(text[start:end])
        start = end

    return chunks


def chunk_by_articles(text: str):
    """
    Intenta dividir el texto usando la estructura de artículos legales.
    """

    # Expresión regular para detectar artículos
    pattern = re.compile(
        r"(art[ií]culo\s+\d+|art\.\s*\d+)",
        re.IGNORECASE
    )

    # Busca todas las coincidencias en el texto
    matches = list(pattern.finditer(text))

    if not matches:
        return None

    chunks = []

    # Recorremos cada artículo detectado
    for i, match in enumerate(matches):
        start = match.start()

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        chunk = text[start:end].strip()
        chunks.append(chunk)

    return chunks


def chunk_text(text: str):
    """
    Función principal de chunking.
    Decide cómo dividir el texto según su estructura.
    """

    # Primer intento: dividir por artículos
    article_chunks = chunk_by_articles(text)

    if article_chunks:
        final_chunks = []

        # Recorremos cada artículo detectado
        for chunk in article_chunks:
            # Si el artículo es muy largo, lo subdividimos
            if len(chunk) > MAX_CHARS:
                final_chunks.extend(split_large_text(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    # Si no hay artículos, dividir por párrafos
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    final_chunks = []

    for p in paragraphs:
        if len(p) > MAX_CHARS:
            final_chunks.extend(split_large_text(p))
        else:
            final_chunks.append(p)

    return final_chunks


def process_files():
    """
    Recorre todos los archivos .txt,
    aplica chunking y devuelve una lista
    con todos los chunks y su metadata.
    """
    all_chunks = []

    for file_name in os.listdir(TXT_DIR):
        
        if not file_name.endswith(".txt"):
            continue

        path = os.path.join(TXT_DIR, file_name)

        # Leemos el contenido completo del archivo
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Aplicamos el chunking al texto
        chunks = chunk_text(text)

        # Guardamos cada chunk con metadata
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "document": file_name,  # archivo de origen
                "chunk_id": i,          # índice del chunk
                "content": chunk        # texto del chunk
            })

    return all_chunks

if __name__ == "__main__":
    chunks = process_files()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks guardados en {OUTPUT_PATH}")
    print(f"Total de chunks: {len(chunks)}")
