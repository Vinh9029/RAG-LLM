import os
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
    local_base_url = os.getenv("LOCAL_LM_BASE_URL", "http://127.0.0.1:1234/v1")
    local_model = os.getenv("LOCAL_MODEL_NAME", "qwen/qwen3.5-9b")
    
    # Check if LM Studio is running by testing the models endpoint
    response = requests.get(f"{local_base_url}/models", timeout=5)
    if response.status_code == 200:
        models_data = response.json()
        available_models = [model["id"] for model in models_data.get("data", [])]
        print(f"LM Studio is running at {local_base_url}")
        print(f"Available models: {available_models}")
        
        if any(local_model in model or model in local_model for model in available_models):
            print(f"Model '{local_model}' is available!")
        else:
            if available_models:
                print(f"Model '{local_model}' is NOT currently loaded.")
                print(f"  Loaded models: {available_models}")
                print(f"  Load '{local_model}' in LM Studio and ensure it's running.")
            else:
                print(f"No models loaded in LM Studio")
                print(f"  Load '{local_model}' in LM Studio.")
    else:
        print(f"Failed to connect to LM Studio at {local_base_url}")
except requests.ConnectionError:
    print(f"LM Studio is NOT running at {os.getenv('LOCAL_LM_BASE_URL', 'http://127.0.0.1:1234/v1')}")
    print("  Start LM Studio and load the qwen model.")
except Exception as e:
    print(f"Local LLM error: {e}")
