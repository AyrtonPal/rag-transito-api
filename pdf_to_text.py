import os
from pypdf import PdfReader

PDF_DIR = "data/pdfs"
OUTPUT_DIR = "data/txts"


def pdf_to_text(pdf_path: str) -> str:
    """
    Convierte un archivo PDF en texto plano.
    Lee página por página el PDF y concatena el texto extraído.
    Retorna todo el contenido como un único string.
    """

    # Lector de PDF a partir del archivo indicado
    reader = PdfReader(pdf_path)

    full_text = []

    # Se recorren todas las páginas del PDF
    for page in reader.pages:
        text = page.extract_text()

        # Valida si la pagina tiene texto antes de agregarlo
        if text:
            full_text.append(text)

    # Se unen todos los textos de las páginas en un solo string,
    # separándolos con saltos de línea
    return "\n".join(full_text)


def main():
    """
    Recorre todos los archivos PDF dentro de la carpeta PDF_DIR,
    los convierte a texto y genera un archivo .txt por cada PDF
    en la carpeta OUTPUT_DIR.
    """

    # Se recorren todos los archivos dentro de la carpeta de PDFs
    for file_name in os.listdir(PDF_DIR):

        # Se ignoran los archivos que no tengan extensión .pdf
        if not file_name.lower().endswith(".pdf"):
            continue

        # Se construye la ruta completa al archivo PDF
        pdf_path = os.path.join(PDF_DIR, file_name)

        # Se genera el nombre del archivo .txt reemplazando la extensión
        txt_name = file_name.replace(".pdf", ".txt")

        # Se construye la ruta completa donde se guardará el .txt
        txt_path = os.path.join(OUTPUT_DIR, txt_name)

        print(f"Procesando: {file_name}")

        text = pdf_to_text(pdf_path)

        # Se abre (o crea) el archivo .txt en modo escritura
        # Si el archivo ya existe, su contenido se sobrescribe
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✔ Guardado: {txt_name}\n")

if __name__ == "__main__":
    main()
