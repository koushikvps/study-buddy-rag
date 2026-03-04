# load_pdf.py
import os
from pypdf import PdfReader

def load_pdf(filepath):
    """Load a PDF and extract all text from it."""
    print(f"📄 Loading PDF: {filepath}")
    
    reader = PdfReader(filepath)
    full_text = ""
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text
        print(f"  ✅ Page {i+1} loaded")
    
    print(f"\n📊 Total pages: {len(reader.pages)}")
    print(f"📊 Total characters: {len(full_text)}")
    return full_text

# Test it
if __name__ == "__main__":
    text = load_pdf("sample.pdf")
    print("\n🔍 First 500 characters of your PDF:")
    print("-" * 50)
    print(text[:500])