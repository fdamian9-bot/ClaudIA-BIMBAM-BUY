"""
agent.py
Define el agente RAG: recibe una pregunta, busca los fragmentos más
relevantes del documento en Chroma, y usa Gemini para redactar la respuesta
citando solo la información encontrada.
"""
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DIR = "chroma_db"

PROMPT_TEMPLATE = """Eres un asistente que trabaja en BIMBAM buy, en el departamento de servicio al cliente. solo debes responder en relacion a los documentos cargados, orientando a los clientes en las policas de compra y preguntas frecuentes; no inventes datos.

Contexto:
{context}

Pregunta: {question}

Respuesta clara y directa:"""


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_agent():
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "Falta GOOGLE_API_KEY. Copia .env.example a .env y coloca tu clave."
        )
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            "No existe la base vectorial. Corre primero: python src/ingest.py --pdf data/tu_archivo.pdf"
        )

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask(question: str) -> str:
    chain = build_agent()
    return chain.invoke(question)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python src/agent.py 'tu pregunta aquí'")
        sys.exit(1)
    pregunta = " ".join(sys.argv[1:])
    print(ask(pregunta))
