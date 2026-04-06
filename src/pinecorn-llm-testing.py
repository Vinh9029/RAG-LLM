import os
import pinecone
import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

# Check Pinecone
print("=== Pinecone Test ===")
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    indexes = pc.list_indexes().names()
    print(f"Pinecone indexes: {indexes}")
    if os.getenv("PINECONE_INDEX") in indexes:
        print(f"Index '{os.getenv('PINECONE_INDEX')}' exists and Pinecone is working!")
    else:
        print(f"Index '{os.getenv('PINECONE_INDEX')}' NOT FOUND! Please check your Pinecone dashboard.")
except Exception as e:
    print(f"Pinecone error: {e}")

# Check Gemini (by listing available models, then checking the using model)
print("\n=== Gemini Model Test ===")
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    models = list(genai.list_models())
    print("Available Gemini models:")
    for m in models:
        print(f"- {m.name} (methods: {m.supported_generation_methods})")
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    found = any(model_name in m.name for m in models)
    if found:
        print(f"Model '{model_name}' is available!")
    else:
        print(f"Model '{model_name}' is NOT available for your API key!")
except Exception as e:
    print(f"Gemini error: {e}")
