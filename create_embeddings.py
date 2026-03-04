# create_embeddings.py
from sentence_transformers import SentenceTransformer
from chunk_text import split_into_chunks
from load_pdf import load_pdf

# Load a free, local embedding model (downloads once, ~90MB)
print("⏳ Loading embedding model (first time takes 1-2 mins to download)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded!")

def create_embeddings(chunks):
    """Convert text chunks into numerical vectors."""
    print(f"\n⏳ Creating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print(f"✅ Embeddings created!")
    print(f"📊 Each chunk is now a vector of {len(embeddings[0])} numbers")
    return embeddings

# Test it
if __name__ == "__main__":
    text = load_pdf("sample.pdf")
    chunks = split_into_chunks(text)
    
    # Just test with first 5 chunks to save time
    sample_chunks = chunks[:5]
    embeddings = create_embeddings(sample_chunks)
    
    print(f"\n🔍 First embedding (first 10 numbers):")
    print(embeddings[0][:10])
    print("\n✅ Embeddings working perfectly!")