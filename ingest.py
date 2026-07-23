"""
ingest.py
Lee uno o varios PDFs, los divide en fragmentos (chunks) y crea una base de
datos vectorial local con Chroma, usando embeddings de Google Gemini.

Uso:
    python src/ingest.py --pdf data/mi_documento.pdf
    python src/ingest.py --pdf-dir data/            # procesa todos los PDFs de la carpeta
"""
import argparse
import glob
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = "chroma_db"


def ingest_pdf(pdf_path):
    """Acepta la ruta a un único PDF (str) o una lista de rutas."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Falta GOOGLE_API_KEY. Copia .env.example a .env y coloca tu clave "
            "(consíguela gratis en https://aistudio.google.com/apikey)."
        )

    pdf_paths = [pdf_path] if isinstance(pdf_path, str) else list(pdf_path)

    documents = []
    for path in pdf_paths:
        print(f"📄 Cargando PDF: {path}")
        loader = PyPDFLoader(path)
        docs = loader.load()
        print(f"   -> {len(docs)} páginas cargadas")
        documents.extend(docs)

    print("✂️  Dividiendo el documento en fragmentos...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"   -> {len(chunks)} fragmentos generados")

    print("🧠 Generando embeddings y guardando en Chroma...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Si ya existe una base previa, la eliminamos para reconstruirla limpia
    if os.path.exists(CHROMA_DIR):
        import shutil

        shutil.rmtree(CHROMA_DIR)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"✅ Base vectorial creada en ./{CHROMA_DIR}")
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingesta de PDF(s) para el agente")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Ruta a un único archivo PDF a procesar")
    group.add_argument(
        "--pdf-dir", help="Carpeta con uno o más PDFs a procesar (ej. data/)"
    )
    args = parser.parse_args()

    if args.pdf_dir:
        pdfs = sorted(glob.glob(os.path.join(args.pdf_dir, "*.pdf")))
        if not pdfs:
            raise SystemExit(f"No se encontraron PDFs en {args.pdf_dir}")
        ingest_pdf(pdfs)
    else:
        ingest_pdf(args.pdf)
