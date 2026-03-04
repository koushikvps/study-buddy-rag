# rag_engine.py
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Load models
print("⏳ Loading models...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print("✅ Models ready!")

def get_collection():
    """Connect to existing ChromaDB collection."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(
        name="study_buddy",
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def search_context(query, collection, top_k=3):
    """Find most relevant chunks for the query."""
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]

def build_prompt(query, context_chunks):
    """
    Build the prompt that combines context + question.
    This is the heart of RAG!
    """
    context = "\n\n".join(context_chunks)
    
    prompt = f"""You are a helpful study assistant. 
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information in the document to answer that."

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""
    return prompt

def ask_question(query, collection):
    """
    Full RAG pipeline:
    question → search → build prompt → LLM → answer
    """
    print(f"\n❓ Question: {query}")
    
    # Step 1: Find relevant chunks
    print("🔍 Searching document...")
    context_chunks = search_context(query, collection)
    print(f"✅ Found {len(context_chunks)} relevant sections")
    
    # Step 2: Build prompt
    prompt = build_prompt(query, context_chunks)
    
    # Step 3: Send to LLM
    print("⏳ Asking Llama...")
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on provided document context."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1  # low temperature = more factual, less creative
    )
    
    answer = response.choices[0].message.content
    return answer

# Test it with sample questions
if __name__ == "__main__":
    collection = get_collection()
    
    # Test questions — these will be answered from YOUR PDF!
    test_questions = [
        "What is this document about?",
        "What are the main topics covered?",
        "Give me a brief summary of the key points."
        "who is the director of this movie"
    ]
    
    print("\n" + "="*60)
    print("🤖 STUDY BUDDY RAG - TEST RUN")
    print("="*60)
    
    for question in test_questions:
        answer = ask_question(question, collection)
        print(f"\n💬 Answer:\n{answer}")
        print("-"*60)
    
    print("\n✅ RAG Engine working! Ready for Week 5 (UI)!")