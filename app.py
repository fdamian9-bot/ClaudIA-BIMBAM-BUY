"""
app.py
Interfaz web (Streamlit) para conversar con el agente sobre el PDF cargado.

Ejecutar localmente:
    streamlit run app.py
"""
import os
import sys
from dotenv import load_dotenv
load_dotenv()

 


import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from agent import build_agent, CHROMA_DIR  # noqa: E402
from ingest import ingest_pdf  # noqa: E402

st.set_page_config(page_title="Alura Agente", page_icon="🤖")
st.title("🤖 claudIA- BIMBAM BUY ")
st.caption("Hola Me llamo ClaudIA, y estoy aca para responder a todas las dudas que tengas sobre nuestras politicas y procedimientos de compras")

# En Streamlit Cloud los "secrets" no siempre llegan como variables de
# entorno automáticamente, así que los copiamos si hace falta.
if not os.getenv("GOOGLE_API_KEY") and "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


if not os.getenv("GOOGLE_API_KEY"):
    st.error(
        "Falta la variable de entorno GOOGLE_API_KEY. "
        "Configúrala en tu archivo .env o en los secretos de la plataforma de deploy."
    )
    st.stop()

# Si la base vectorial todavía no existe (primer arranque en la nube),
# la generamos automáticamente a partir del PDF en data/.
if not os.path.exists(CHROMA_DIR):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    pdfs = sorted(
        [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
        if os.path.isdir(data_dir) else []
    )
    if pdfs:
        with st.spinner(f"Procesando {len(pdfs)} documento(s) por primera vez..."):
            ingest_pdf([os.path.join(data_dir, f) for f in pdfs])
    else:
        st.error(
            "No se encontró ningún PDF en la carpeta data/. "
            "Sube un PDF a esa carpeta y vuelve a desplegar."
        )
        st.stop()

if "chain" not in st.session_state:
    try:
        st.session_state.chain = build_agent()
    except Exception as e:
        st.error(f"No se pudo iniciar el agente: {e}")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

pregunta = st.chat_input("Escribe tu pregunta sobre el documento...")

if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el documento..."):
            respuesta = st.session_state.chain.invoke(pregunta)
            st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})
