"""
The Study — a RAG chat assistant with a personal-library aesthetic.
Upload PDFs, they're embedded into a Chroma store, and you chat with
a Mistral model that answers strictly from retrieved context.
"""

import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Streamlit Community Cloud has no .env file — the key must come from
# st.secrets there. Bridge it into os.environ so langchain_mistralai
# (which reads os.environ internally) can find it either way.
if not os.environ.get("MISTRAL_API_KEY"):
    if "MISTRAL_API_KEY" in st.secrets:
        os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"].strip()

if not os.environ.get("MISTRAL_API_KEY"):
    st.error(
        "MISTRAL_API_KEY not found. Add it in Streamlit Cloud → "
        "Settings → Secrets, then reboot the app."
    )
    st.stop()

PERSIST_DIR = "project_2_store"

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="The Study — RAG Assistant",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Design tokens (see palette rationale in accompanying notes)
#   ink-900  #14172A   deep study-room navy, main background
#   ink-800  #1D2138   card / panel surface
#   ink-700  #2A2F4C   raised surface / hover
#   lamp     #D9A441   warm brass lamplight accent
#   lamp-dim #8A6A2E   muted brass for borders
#   paper    #ECE6D8   warm off-white text
#   mist     #8891AA   secondary/muted text
#   sage     #6FA787   success / grounded-answer accent
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #1D2138 0%, #14172A 55%);
        color: #ECE6D8;
    }

    /* ---- Hero ---- */
    .study-hero {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 28px 32px;
        border-radius: 14px;
        background: linear-gradient(135deg, #1D2138 0%, #191C30 100%);
        border: 1px solid #2A2F4C;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
        animation: fadeInDown 0.6s ease;
    }
    .study-hero::before {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(217,164,65,0.30) 0%, rgba(217,164,65,0) 70%);
        animation: lampPulse 4s ease-in-out infinite;
    }
    .study-hero h1 {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 2.1rem;
        margin: 0;
        color: #ECE6D8;
        letter-spacing: 0.2px;
    }
    .study-hero p {
        margin: 4px 0 0 0;
        color: #8891AA;
        font-size: 0.95rem;
    }
    .lamp-dot {
        width: 10px; height: 10px; border-radius: 50%;
        background: #D9A441;
        box-shadow: 0 0 12px 3px rgba(217,164,65,0.7);
        display: inline-block;
        animation: lampFlicker 2.6s ease-in-out infinite;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes lampPulse {
        0%, 100% { opacity: 0.6; transform: scale(1); }
        50%      { opacity: 1;   transform: scale(1.15); }
    }
    @keyframes lampFlicker {
        0%, 100% { opacity: 1; }
        45%      { opacity: 0.55; }
        50%      { opacity: 1; }
    }

    /* ---- Chat bubbles ---- */
    [data-testid="stChatMessage"] {
        animation: fadeInUp 0.35s ease;
        border-radius: 12px;
        border: 1px solid #2A2F4C;
    }

    /* ---- Source chips ---- */
    .source-chip {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #14172A;
        background: #D9A441;
        padding: 2px 9px;
        border-radius: 999px;
        margin: 2px 6px 2px 0;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #171A2C;
        border-right: 1px solid #2A2F4C;
    }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        font-family: 'Fraunces', serif;
        color: #ECE6D8;
    }

    .shelf-item {
        padding: 8px 10px;
        border-radius: 8px;
        background: #1D2138;
        border: 1px solid #2A2F4C;
        margin-bottom: 6px;
        font-size: 0.85rem;
        color: #ECE6D8;
        animation: fadeInUp 0.3s ease;
    }

    .status-pill {
        display: inline-block;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 999px;
        border: 1px solid #6FA787;
        color: #6FA787;
        margin-top: 6px;
    }

    /* Buttons */
    .stButton > button {
        background: #D9A441;
        color: #14172A;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(217,164,65,0.35);
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: #1D2138 !important;
        border: 1px solid #2A2F4C !important;
        color: #ECE6D8 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="study-hero">
        <div style="font-size:2.4rem;">📖</div>
        <div>
            <h1>The Study <span class="lamp-dot"></span></h1>
            <p>Ask questions. Answers are grounded only in the documents on your shelf.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return MistralAIEmbeddings()


@st.cache_resource(show_spinner=False)
def get_vectorstore(_embedding_model):
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_embedding_model)


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")


embedding_model = get_embedding_model()
vectorstore = get_vectorstore(embedding_model)
llm = get_llm()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a knowledgeable and precise assistant that answers questions using ONLY the information provided in the "Context" section below, retrieved from a knowledge base. Follow these rules carefully:

1. GROUNDING
   - Base your answer strictly on the provided context.
   - Do not use outside knowledge or make assumptions beyond what is stated.
   - If the context does not contain enough information to answer the question, say so clearly instead of guessing.

2. ACCURACY
   - Quote or reference specific parts of the context when relevant (e.g., "According to Document 2...").
   - If different context chunks contradict each other, point out the discrepancy rather than picking one silently.

3. CLARITY & STRUCTURE
   - Give a direct answer first, then supporting details.
   - Use bullet points or numbered lists for multi-part answers.
   - Keep the language clear and avoid unnecessary jargon unless the question requires it.

4. HONESTY ABOUT LIMITATIONS
   - If the answer is partially supported by the context, explain what is known and what is missing.
   - If no relevant information exists in the context, respond with: "The provided context does not contain information to answer this question."

5. FORMAT OF RESPONSE
   - Answer: [direct response]
   - Explanation: [reasoning based on context]
   - Source(s): [which context chunk(s) the answer came from, e.g., "Chunk 3, Chunk 5"]

---
Context:
{retrieved_context}
---
""",
        ),
        ("human", "{user_question}"),
    ]
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "shelf" not in st.session_state:
    st.session_state.shelf = []

# ----------------------------------------------------------------------
# Sidebar — the "shelf" (upload + ingest)
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Your Shelf")
    st.caption("Add PDFs to grow the knowledge base your assistant can cite from.")

    uploaded_files = st.file_uploader(
        "Add a PDF", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
    )

    if uploaded_files and st.button("➕ Add to shelf", use_container_width=True):
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
        progress = st.progress(0, text="Warming up the lamp...")

        for i, file in enumerate(uploaded_files):
            progress.progress(
                (i) / len(uploaded_files), text=f"Reading {file.name}..."
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            raw_docs = loader.load()
            chunks = splitter.split_documents(raw_docs)

            for c in chunks:
                c.metadata["source"] = file.name

            vectorstore.add_documents(chunks)
            os.unlink(tmp_path)

            if file.name not in st.session_state.shelf:
                st.session_state.shelf.append(file.name)

            progress.progress(
                (i + 1) / len(uploaded_files), text=f"Shelved {file.name}"
            )

        time.sleep(0.4)
        progress.empty()
        st.success(f"Added {len(uploaded_files)} document(s) to the shelf.")

    st.markdown("#### On the shelf")
    if st.session_state.shelf:
        for name in st.session_state.shelf:
            st.markdown(f'<div class="shelf-item">📄 {name}</div>', unsafe_allow_html=True)
        st.markdown('<span class="status-pill">● knowledge base ready</span>', unsafe_allow_html=True)
    else:
        st.caption("Nothing here yet — upload a PDF to begin.")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------
# Chat history
# ----------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🕯️" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            chips = "".join(f'<span class="source-chip">{s}</span>' for s in msg["sources"])
            st.markdown(chips, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------
question = st.chat_input("Ask something about your shelf...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🕯️"):
        placeholder = st.empty()
        placeholder.markdown("_Turning pages..._")

        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})

        final_prompt = prompt.invoke(
            {"retrieved_context": context, "user_question": question}
        )
        response = llm.invoke(final_prompt)

        placeholder.markdown(response.content)
        if sources:
            chips = "".join(f'<span class="source-chip">{s}</span>' for s in sources)
            st.markdown(chips, unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.content, "sources": sources}
    )