import os
import pinecone
from openai import OpenAI
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

# Check Local LLM (LM Studio)
print("\n=== LM Studio Local LLM Test ===")
try:
    base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    models = client.models.list()
    print(f"Connected successfully to LM Studio at {base_url}!")
    print("Available loaded models:")
    for m in models.data:
        print(f"- {m.id}")
except Exception as e:
    print(f"LM Studio connection error: {e}")
    print("Please ensure LM Studio is running, the model is loaded, and the Local Server is started on port 1234.")
