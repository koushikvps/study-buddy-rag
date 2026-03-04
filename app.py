# app.py — Study Buddy RAG Chatbot (Redesigned UI)
import streamlit as st
import chromadb
import os
import tempfile
import time
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from load_pdf import load_pdf
from chunk_text import split_into_chunks

load_dotenv()

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Buddy AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* ── Main content ── */
[data-testid="stMain"] {
    background: #0d1117 !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #58a6ff !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 4px 8px !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
}
[data-testid="stChatInput"] textarea {
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 8px 0 !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] button:hover {
    background: #30363d !important;
    border-color: #58a6ff !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #8b949e !important;
    font-size: 13px !important;
}

/* ── Divider ── */
hr { border-color: #21262d !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #58a6ff; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { color: #58a6ff !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; }

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #58a6ff !important; }

/* ── Toast / success / error ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedding_model, groq_client

embedding_model, groq_client = load_models()

# ── Helper functions ───────────────────────────────────────────
def build_vector_store(chunks):
    chunks = [c for c in chunks if c and c.strip()]
    if not chunks:
        st.error("❌ This PDF appears to be scanned or image-based. No text could be extracted.")
        st.stop()
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("study_buddy")
    except:
        pass
    collection = client.create_collection(
        name="study_buddy",
        metadata={"hnsw:space": "cosine"}
    )
    embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection

def get_answer_with_sources(question, collection):
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    context_chunks = results["documents"][0]
    context = "\n\n---\n\n".join(context_chunks)
    prompt = f"""You are Study Buddy, a helpful AI assistant that answers questions strictly based on the provided document context.

Rules:
- Answer ONLY using the context below
- Be concise but complete
- If the answer is not in the context, say "I couldn't find that in the document."
- Format your answer clearly with line breaks where helpful

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are Study Buddy, a precise document Q&A assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content, context_chunks

def stream_text(text):
    """Simulate typing animation by streaming words."""
    words = text.split(" ")
    result = ""
    placeholder = st.empty()
    for i, word in enumerate(words):
        result += word + " "
        placeholder.markdown(f"""
        <div class="assistant-bubble">
            <div class="bubble-content">{result}▋</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.015)
    placeholder.markdown(f"""
    <div class="assistant-bubble">
        <div class="bubble-content">{result.strip()}</div>
    </div>
    """, unsafe_allow_html=True)
    return placeholder

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo & Branding
    st.markdown("""
    <div style="padding: 28px 20px 20px; border-bottom: 1px solid #21262d; margin-bottom: 20px;">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
            <div style="
                width:40px; height:40px;
                background: linear-gradient(135deg, #58a6ff, #a371f7);
                border-radius:10px;
                display:flex; align-items:center; justify-content:center;
                font-size:20px;
            ">🧠</div>
            <div>
                <div style="font-family:'Syne',sans-serif; font-weight:800; font-size:18px; color:#e6edf3; line-height:1;">Study Buddy</div>
                <div style="font-size:11px; color:#58a6ff; font-weight:600; letter-spacing:1px; text-transform:uppercase;">AI · RAG Chatbot</div>
            </div>
        </div>
        <div style="font-size:12px; color:#8b949e; line-height:1.5;">
            Ask anything about your PDF. Powered by Llama 3.3 70B.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown("""
    <div style="padding: 0 12px; margin-bottom: 8px;">
        <div style="font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#8b949e; margin-bottom:8px;">Navigation</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🏠  Home", "💬  Chat", "📊  About"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 0 0 16px;'>", unsafe_allow_html=True)

    # PDF Upload Section
    st.markdown("""
    <div style="padding: 0 4px; margin-bottom: 12px;">
        <div style="font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#8b949e; margin-bottom:12px;">Document</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf",
        label_visibility="collapsed",
        help="Upload any text-based PDF"
    )

    if uploaded_file:
        # Process PDF
        if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
            with st.spinner("Indexing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                text = load_pdf(tmp_path)
                chunks = split_into_chunks(text)
                collection = build_vector_store(chunks)
                st.session_state.collection = collection
                st.session_state.processed_file = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                st.session_state.messages = []
            st.success("Document ready!")

        # File stats
        st.markdown(f"""
        <div style="
            background: #161b22;
            border: 1px solid #21262d;
            border-radius: 10px;
            padding: 14px 16px;
            margin-top: 12px;
        ">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                <div style="width:8px; height:8px; border-radius:50%; background:#3fb950;"></div>
                <div style="font-size:13px; font-weight:600; color:#e6edf3; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{uploaded_file.name}</div>
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                <div style="background:#0d1117; border-radius:6px; padding:8px 10px; text-align:center;">
                    <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:800; color:#58a6ff;">{st.session_state.get('chunk_count', 0)}</div>
                    <div style="font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:1px;">Chunks</div>
                </div>
                <div style="background:#0d1117; border-radius:6px; padding:8px 10px; text-align:center;">
                    <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:800; color:#3fb950;">{len(st.session_state.get('messages', [])) // 2}</div>
                    <div style="font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:1px;">Questions</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Clear chat button
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("🗑️  Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.markdown("""
        <div style="
            background: #161b22;
            border: 1px dashed #30363d;
            border-radius: 10px;
            padding: 20px 16px;
            text-align: center;
            color: #8b949e;
            font-size: 13px;
            line-height: 1.6;
        ">
            <div style="font-size:24px; margin-bottom:8px;">📄</div>
            Upload a PDF to get started
        </div>
        """, unsafe_allow_html=True)

    # Bottom - Tech stack
    st.markdown("""
    <div style="
        position: fixed;
        bottom: 0;
        width: 240px;
        padding: 16px 20px;
        border-top: 1px solid #21262d;
        background: #0d1117;
    ">
        <div style="font-size:11px; color:#8b949e; margin-bottom:8px; font-weight:600; letter-spacing:1px; text-transform:uppercase;">Powered by</div>
        <div style="display:flex; flex-wrap:wrap; gap:6px;">
            <span style="background:#21262d; border-radius:4px; padding:3px 8px; font-size:11px; color:#e6edf3; font-family:'JetBrains Mono',monospace;">Llama 3.3</span>
            <span style="background:#21262d; border-radius:4px; padding:3px 8px; font-size:11px; color:#e6edf3; font-family:'JetBrains Mono',monospace;">ChromaDB</span>
            <span style="background:#21262d; border-radius:4px; padding:3px 8px; font-size:11px; color:#e6edf3; font-family:'JetBrains Mono',monospace;">MiniLM</span>
            <span style="background:#21262d; border-radius:4px; padding:3px 8px; font-size:11px; color:#e6edf3; font-family:'JetBrains Mono',monospace;">Groq</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <style>
    .hero-section {
        padding: 80px 64px 48px;
        max-width: 860px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(88,166,255,0.1);
        border: 1px solid rgba(88,166,255,0.25);
        color: #58a6ff;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 28px;
    }
    .hero-title {
        font-family: 'Syne', sans-serif;
        font-size: 64px;
        font-weight: 800;
        line-height: 1.05;
        color: #e6edf3;
        margin-bottom: 16px;
    }
    .hero-title span {
        background: linear-gradient(135deg, #58a6ff 0%, #a371f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 18px;
        color: #8b949e;
        line-height: 1.7;
        margin-bottom: 48px;
        max-width: 560px;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 48px;
    }
    .feature-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 24px 20px;
        transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: #58a6ff; }
    .feature-icon { font-size: 28px; margin-bottom: 12px; }
    .feature-title {
        font-family: 'Syne', sans-serif;
        font-size: 16px;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 6px;
    }
    .feature-desc { font-size: 13px; color: #8b949e; line-height: 1.5; }
    .steps-section { margin-bottom: 48px; }
    .steps-title {
        font-family: 'Syne', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 20px;
    }
    .step-row {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 16px;
    }
    .step-num {
        width: 32px; height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #58a6ff, #a371f7);
        color: white;
        font-weight: 700;
        font-size: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .step-content h4 {
        font-size: 15px;
        font-weight: 600;
        color: #e6edf3;
        margin-bottom: 2px;
    }
    .step-content p { font-size: 13px; color: #8b949e; line-height: 1.5; }
    </style>

    <div class="hero-section">
        <div class="hero-badge">✦ AI-Powered Document Assistant</div>
        <div class="hero-title">Chat with your<br><span>PDF documents.</span></div>
        <div class="hero-sub">Upload any PDF and ask questions in plain English. Study Buddy finds the most relevant sections and gives you precise, grounded answers instantly.</div>

        <div class="feature-grid">
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <div class="feature-title">Lightning Fast</div>
                <div class="feature-desc">Powered by Groq's ultra-fast inference. Answers in under 2 seconds.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Grounded Answers</div>
                <div class="feature-desc">Responses based strictly on your document. No hallucinations.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🔍</div>
                <div class="feature-title">Source Transparency</div>
                <div class="feature-desc">See exactly which sections of your PDF were used to answer.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🧠</div>
                <div class="feature-title">Llama 3.3 70B</div>
                <div class="feature-desc">State-of-the-art open-source LLM by Meta. Enterprise-grade intelligence.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">💾</div>
                <div class="feature-title">Vector Search</div>
                <div class="feature-desc">ChromaDB finds semantically similar content, not just keywords.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">🆓</div>
                <div class="feature-title">100% Free</div>
                <div class="feature-desc">Entirely open-source stack. No API costs, no subscriptions.</div>
            </div>
        </div>

        <div class="steps-section">
            <div class="steps-title">How it works</div>
            <div class="step-row">
                <div class="step-num">1</div>
                <div class="step-content">
                    <h4>Upload your PDF</h4>
                    <p>Use the sidebar to upload any text-based PDF — textbooks, research papers, reports, or documentation.</p>
                </div>
            </div>
            <div class="step-row">
                <div class="step-num">2</div>
                <div class="step-content">
                    <h4>Document gets indexed</h4>
                    <p>Your PDF is split into chunks, converted to embeddings, and stored in a vector database for instant search.</p>
                </div>
            </div>
            <div class="step-row">
                <div class="step-num">3</div>
                <div class="step-content">
                    <h4>Ask anything</h4>
                    <p>Switch to the Chat page and ask questions. The AI finds the most relevant sections and answers precisely.</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE: CHAT
# ══════════════════════════════════════════════════════════════
elif page == "💬  Chat":
    st.markdown("""
    <style>
    .chat-wrapper {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 80px);
        padding: 0;
    }
    .chat-header {
        padding: 24px 40px 16px;
        border-bottom: 1px solid #21262d;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .chat-title {
        font-family: 'Syne', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: #e6edf3;
    }
    .chat-doc-badge {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 12px;
        color: #8b949e;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .chat-doc-badge .dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #3fb950;
    }
    .messages-area {
        flex: 1;
        overflow-y: auto;
        padding: 28px 40px;
    }
    .user-bubble {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    .user-bubble-inner {
        background: linear-gradient(135deg, #1f6feb, #388bfd);
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        font-size: 15px;
        line-height: 1.6;
        font-weight: 400;
        box-shadow: 0 4px 16px rgba(31,111,235,0.25);
    }
    .assistant-bubble {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 8px;
    }
    .assistant-bubble-inner {
        display: flex;
        gap: 12px;
        max-width: 80%;
    }
    .assistant-avatar {
        width: 32px; height: 32px;
        border-radius: 8px;
        background: linear-gradient(135deg, #58a6ff, #a371f7);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
        margin-top: 2px;
    }
    .bubble-content {
        background: #161b22;
        border: 1px solid #21262d;
        color: #e6edf3;
        padding: 14px 18px;
        border-radius: 4px 18px 18px 18px;
        font-size: 15px;
        line-height: 1.7;
    }
    .source-section {
        margin-top: 8px;
        margin-left: 44px;
        margin-bottom: 20px;
    }
    .source-toggle {
        font-size: 12px;
        color: #8b949e;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .source-chunks {
        margin-top: 8px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .source-chunk {
        background: #0d1117;
        border: 1px solid #21262d;
        border-left: 3px solid #58a6ff;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #8b949e;
        line-height: 1.5;
    }
    .empty-chat {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        text-align: center;
        color: #8b949e;
    }
    .empty-chat .icon { font-size: 48px; margin-bottom: 16px; }
    .empty-chat h3 {
        font-family: 'Syne', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 8px;
    }
    .empty-chat p { font-size: 14px; line-height: 1.6; max-width: 360px; }
    .suggestion-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-top: 20px;
    }
    .chip {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 13px;
        color: #e6edf3;
        cursor: pointer;
        transition: all 0.2s;
    }
    .chip:hover { border-color: #58a6ff; color: #58a6ff; }
    .chat-input-area {
        padding: 16px 40px 24px;
        border-top: 1px solid #21262d;
    }
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat header
    doc_name = st.session_state.get("processed_file", None)
    st.markdown(f"""
    <div class="chat-header">
        <div class="chat-title">💬 Chat</div>
        {'<div class="chat-doc-badge"><div class="dot"></div>' + doc_name + '</div>' if doc_name else '<div class="chat-doc-badge">No document loaded</div>'}
    </div>
    """, unsafe_allow_html=True)

    # Messages area
    if not st.session_state.messages:
        if "collection" in st.session_state:
            st.markdown("""
            <div class="empty-chat" style="padding:60px 40px;">
                <div class="icon">💬</div>
                <h3>Ready to answer!</h3>
                <p>Your document is indexed. Ask me anything about it below.</p>
                <div class="suggestion-chips">
                    <div class="chip">What is this document about?</div>
                    <div class="chip">Summarize the key points</div>
                    <div class="chip">What are the main topics?</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-chat" style="padding:80px 40px;">
                <div class="icon">📄</div>
                <h3>No document loaded</h3>
                <p>Upload a PDF using the sidebar to start chatting with your document.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Render messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="user-bubble">
                    <div class="user-bubble-inner">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-bubble">
                    <div class="assistant-bubble-inner">
                        <div class="assistant-avatar">🧠</div>
                        <div class="bubble-content">{msg["content"]}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                # Show sources
                if "sources" in msg and msg["sources"]:
                    with st.expander("📎 View source chunks used", expanded=False):
                        for i, chunk in enumerate(msg["sources"]):
                            st.markdown(f"""
                            <div class="source-chunk">
                                <span style="color:#58a6ff; font-weight:600;">Source {i+1}</span><br>
                                {chunk[:280].replace('<', '&lt;').replace('>', '&gt;')}{'...' if len(chunk) > 280 else ''}
                            </div>
                            """, unsafe_allow_html=True)

    # Chat input
    if "collection" in st.session_state:
        if question := st.chat_input("Ask anything about your document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})

            # Get answer
            with st.spinner(""):
                answer, sources = get_answer_with_sources(question, st.session_state.collection)

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "📊  About":
    st.markdown("""
    <style>
    .about-wrapper { padding: 48px 64px; max-width: 860px; }
    .about-title {
        font-family: 'Syne', sans-serif;
        font-size: 40px;
        font-weight: 800;
        color: #e6edf3;
        margin-bottom: 8px;
    }
    .about-sub { font-size: 16px; color: #8b949e; margin-bottom: 40px; }
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 20px;
        font-weight: 700;
        color: #e6edf3;
        margin-bottom: 16px;
        margin-top: 36px;
    }
    .arch-flow {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
        margin-bottom: 32px;
    }
    .arch-node {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 8px 14px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: #e6edf3;
    }
    .arch-arrow { color: #58a6ff; font-size: 16px; }
    .stack-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 32px;
    }
    .stack-table th {
        text-align: left;
        padding: 10px 16px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #8b949e;
        border-bottom: 1px solid #21262d;
    }
    .stack-table td {
        padding: 12px 16px;
        font-size: 14px;
        color: #e6edf3;
        border-bottom: 1px solid #161b22;
    }
    .stack-table tr:hover td { background: #161b22; }
    .mono { font-family: 'JetBrains Mono', monospace; color: #58a6ff; }
    .free-tag {
        background: rgba(63,185,80,0.1);
        border: 1px solid rgba(63,185,80,0.3);
        color: #3fb950;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 10px;
    }
    .built-by-card {
        background: linear-gradient(135deg, #161b22, #1c2128);
        border: 1px solid #21262d;
        border-radius: 16px;
        padding: 28px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-top: 16px;
    }
    .avatar {
        width: 56px; height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #58a6ff, #a371f7);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        flex-shrink: 0;
    }
    .builder-name {
        font-family: 'Syne', sans-serif;
        font-size: 20px;
        font-weight: 800;
        color: #e6edf3;
    }
    .builder-sub { font-size: 14px; color: #8b949e; margin-top: 2px; }
    </style>

    <div class="about-wrapper">
        <div class="about-title">About Study Buddy</div>
        <div class="about-sub">A Retrieval-Augmented Generation chatbot built from scratch in 5 weeks.</div>

        <div class="section-title">Architecture</div>
        <div class="arch-flow">
            <div class="arch-node">📄 PDF</div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">✂️ Chunks</div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">🔢 Embeddings</div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">🗄️ ChromaDB</div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">🔍 Search</div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">🦙 Llama 3.3</div>
            <div class="arch-arrow">→</div>
            <div class="arch-node">💬 Answer</div>
        </div>

        <div class="section-title">Tech Stack</div>
        <table class="stack-table">
            <tr>
                <th>Component</th>
                <th>Tool</th>
                <th>Purpose</th>
                <th>Cost</th>
            </tr>
            <tr>
                <td>LLM</td>
                <td><span class="mono">Llama 3.3 70B</span></td>
                <td>Answer generation</td>
                <td><span class="free-tag">FREE</span></td>
            </tr>
            <tr>
                <td>Inference</td>
                <td><span class="mono">Groq API</span></td>
                <td>Ultra-fast LLM serving</td>
                <td><span class="free-tag">FREE</span></td>
            </tr>
            <tr>
                <td>Embeddings</td>
                <td><span class="mono">MiniLM-L6-v2</span></td>
                <td>Text → vectors</td>
                <td><span class="free-tag">FREE · Local</span></td>
            </tr>
            <tr>
                <td>Vector DB</td>
                <td><span class="mono">ChromaDB</span></td>
                <td>Similarity search</td>
                <td><span class="free-tag">FREE · Local</span></td>
            </tr>
            <tr>
                <td>PDF Parser</td>
                <td><span class="mono">PyPDF</span></td>
                <td>Text extraction</td>
                <td><span class="free-tag">FREE</span></td>
            </tr>
            <tr>
                <td>UI Framework</td>
                <td><span class="mono">Streamlit</span></td>
                <td>Web interface</td>
                <td><span class="free-tag">FREE</span></td>
            </tr>
        </table>

        <div class="section-title">Built by</div>
        <div class="built-by-card">
            <div class="avatar">👨‍💻</div>
            <div>
                <div class="builder-name">Koushik</div>
                <div class="builder-sub">Built this RAG chatbot from scratch as part of an AI portfolio project · 2026</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)