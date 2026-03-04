# chunk_text.py
from load_pdf import load_pdf

def split_into_chunks(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.
    
    chunk_size = how many characters per chunk
    overlap    = how many characters to repeat between chunks
                 (so we don't lose meaning at boundaries)
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap ensures continuity
    
    return chunks

# Test it
if __name__ == "__main__":
    text = load_pdf("sample.pdf")
    chunks = split_into_chunks(text)
    
    print(f"\n✅ Total chunks created: {len(chunks)}")
    print(f"\n🔍 Sample Chunk #1:")
    print("-" * 50)
    print(chunks[0])
    print(f"\n🔍 Sample Chunk #2:")
    print("-" * 50)
    print(chunks[1])