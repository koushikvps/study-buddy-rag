cat > README.md << 'EOF'
# 🤖 Study Buddy — RAG Chatbot

An AI-powered chatbot that reads any PDF and answers questions about it using Retrieval-Augmented Generation (RAG).

## 🎯 Features
- Upload any PDF document
- Ask questions in natural language
- Get accurate answers grounded in your document
- Persistent chat history

## 🛠️ Tech Stack
| Component | Tool |
|---|---|
| LLM | Llama 3.3 70B via Groq |
| Embeddings | Sentence Transformers (MiniLM-L6) |
| Vector DB | ChromaDB |
| PDF Parser | PyPDF |
| UI | Streamlit |

## 🆓 100% Free Stack — No credit card needed

## 🚀 Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/study-buddy-rag
cd study-buddy-rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```
```bash
streamlit run app.py
```

## 🏗️ Architecture
PDF → Chunks → Embeddings → ChromaDB → Similarity Search → Llama 3.3 → Answer
EOF