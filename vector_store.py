# vector_store.py
import chromadb
from sentence_transformers import SentenceTransformer
from load_pdf import load_pdf
from chunk_text import split_into_chunks

# Load embedding model
print("⏳ Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model ready!")

def build_vector_store(pdf_path):
    """
    Full pipeline:
    PDF → chunks → embeddings → stored in ChromaDB
    """

    # Step 1: Load PDF
    text = load_pdf(pdf_path)

    # Step 2: Split into chunks
    chunks = split_into_chunks(text)
    print(f"\n✅ Created {len(chunks)} chunks")

    # Step 3: Create ChromaDB client (stores data locally in ./chroma_db folder)
    client = chromadb.PersistentClient(path="./chroma_db")

    # Step 4: Create a collection (like a table in a database)
    # get_or_create means it won't fail if it already exists
    collection = client.get_or_create_collection(
        name="study_buddy",
        metadata={"hnsw:space": "cosine"}  # use cosine similarity for search
    )

    # Step 5: Create embeddings for all chunks
    print(f"\n⏳ Creating embeddings for {len(chunks)} chunks...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=True).tolist()
    print("✅ Embeddings created!")

    # Step 6: Store chunks + embeddings in ChromaDB
    print("\n⏳ Storing in ChromaDB...")
    collection.add(
        documents=chunks,                                    # original text
        embeddings=embeddings,                               # vector numbers
        ids=[f"chunk_{i}" for i in range(len(chunks))]      # unique ID for each
    )
    print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")
    return collection

def search_vector_store(query, collection, top_k=3):
    """
    Search the vector store for chunks most relevant to the query.
    top_k = how many results to return
    """
    print(f"\n🔍 Searching for: '{query}'")

    # Convert query to embedding
    query_embedding = embedding_model.encode([query]).tolist()

    # Search ChromaDB for most similar chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results["documents"][0]  # return the matching chunks

# Test it
if __name__ == "__main__":
    # Build the vector store from your PDF
    collection = build_vector_store("sample.pdf")

    # Test a search
    query = "What is this document about?"
    matching_chunks = search_vector_store(query, collection)

    print(f"\n📋 Top {len(matching_chunks)} matching chunks for your query:")
    for i, chunk in enumerate(matching_chunks):
        print(f"\n--- Match {i+1} ---")
        print(chunk[:300])  # show first 300 chars of each match