import os
import pinecone
import google.generativeai as genai
import requests
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

# Check Local LLM (LM Studio)
print("\n=== Local LLM (LM Studio) Test ===")
try:
    lm_studio_url = os.getenv("LM_STUDIO_API_URL", "http://127.0.0.1:1234/v1")
    lm_studio_model = os.getenv("LM_STUDIO_MODEL", "gpt-oss-20b")
    
    # Ensure URL has /v1 suffix
    if not lm_studio_url.endswith("/v1"):
        lm_studio_url = lm_studio_url.rstrip("/") + "/v1"
    
    # Test connection to LM Studio
    test_url = f"{lm_studio_url}/models"
    response = requests.get(test_url, timeout=5)
    
    if response.status_code == 200:
        available_models = response.json().get("data", [])
        print(f"LM Studio connected! Available models:")
        for model in available_models:
            print(f"- {model.get('id', 'Unknown')}")
        
        # Check if configured model is available
        model_ids = [m.get("id") for m in available_models]
        if lm_studio_model in model_ids:
            print(f"[OK] Model '{lm_studio_model}' is loaded!")
        else:
            print(f"[WARNING] Model '{lm_studio_model}' NOT found in LM Studio!")
            print(f"Please load the model in LM Studio or update LM_STUDIO_MODEL in .env")
    else:
        print(f"LM Studio returned status {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print(f"[ERROR] Could not connect to LM Studio at {lm_studio_url}")
    print("Make sure LM Studio is running with the server enabled")
except requests.exceptions.Timeout:
    print(f"[ERROR] Connection timeout to LM Studio at {lm_studio_url}")
except Exception as e:
    print(f"Local LLM error: {e}")

# Summary
print("\n=== Configuration Summary ===")
print(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'gemini')}")
print(f"Pinecone Index: {os.getenv('PINECONE_INDEX')}")
print(f"Temperature: {os.getenv('TEMPERATURE', '0.3')}")
