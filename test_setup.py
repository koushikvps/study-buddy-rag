# test_setup.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ ERROR: API key not found. Check your .env file.")
else:
    print("✅ API key loaded successfully!")

client = Groq(api_key=api_key)

print("⏳ Sending test message to Llama 3.3 via Groq...")

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": "Say hello, tell me your name, and confirm you are ready to help build a RAG chatbot!"
        }
    ]
)

print("\n🤖 Llama says:")
print(response.choices[0].message.content)
print("\n✅ Week 1 is COMPLETE! Ready for Week 2! 🚀")