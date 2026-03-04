# app.py
import streamlit as st
import chromadb
import os
import tempfile
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
from load_pdf import load_pdf
from chunk_text import split_into_chunks

load_dotenv()

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Study Buddy",
    page_icon="🤖",
    layout="centered"
)

# ── Load models once (cached so they don't reload every time) ──
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return embedding_model, groq_client

embedding_model, groq_client = load_models()

# ── Helper functions ──────────────────────────────────────
def build_vector_store(chunks):
    """Store chunks in ChromaDB."""
    
    # Filter out empty or whitespace-only chunks
    chunks = [c for c in chunks if c and c.strip()]
    
    # Safety check
    if not chunks:
        st.error("❌ This PDF appears to be scanned or image-based. No text could be extracted. Please try a different PDF with selectable text.")
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

def get_answer(question, collection):
    """RAG pipeline: question → search → LLM → answer."""
    # Search for relevant chunks
    query_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )
    context_chunks = results["documents"][0]
    context = "\n\n".join(context_chunks)
    
    # Build prompt
    prompt = f"""You are a helpful study assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the document to answer that."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    # Ask Llama
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content

# ── UI ────────────────────────────────────────────────────
st.title("🤖 Study Buddy")
st.caption("Upload a PDF and ask questions about it!")
st.divider()

# PDF Upload section
uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type="pdf",
    help="Upload any PDF document to chat with it"
)

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    # Process PDF (only when new file uploaded)
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        with st.spinner("📖 Reading and indexing your PDF..."):
            text = load_pdf(tmp_path)
            chunks = split_into_chunks(text)
            collection = build_vector_store(chunks)
            st.session_state.collection = collection
            st.session_state.processed_file = uploaded_file.name
            st.session_state.messages = []  # reset chat
        st.success(f"✅ PDF indexed! Created {len(chunks)} searchable chunks.")
    
    st.divider()

    # Chat section
    st.subheader("💬 Ask a Question")

    # Display chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask anything about your PDF..."):
        # Show user question
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Get and show answer
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                answer = get_answer(question, st.session_state.collection)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    # Show instructions when no PDF uploaded
    st.info("👆 Upload a PDF above to get started!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "Llama 3.3 70B")
    with col2:
        st.metric("Vector DB", "ChromaDB")
    with col3:
        st.metric("Embeddings", "MiniLM-L6")