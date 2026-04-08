import os
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

class AppConfig:
    """Centralized configuration management for the entire application."""
    def __init__(self):
        # Enable cache to optimize response time and reduce API costs
        set_llm_cache(InMemoryCache())
        
        # LLM Provider Selection
        self.llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
        self.temperature = float(os.getenv("TEMPERATURE", 0.3))
        self.cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Document Chunking Configuration
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        
        # Retrieval Configuration
        self.search_k = int(os.getenv("SEARCH_K", 5))
        self.top_n = int(os.getenv("TOP_N", 3))
        
        # PDF Directory
        self.pdf_directory = os.getenv("PDF_DIRECTORY", "./mental_health_docs")
        
        # Initialize LLM based on provider
        if self.llm_provider == "lm_studio":
            from langchain_openai import ChatOpenAI
            # Ensure base_url ends with /v1 for OpenAI-compatible endpoint
            lm_studio_url = os.getenv("LM_STUDIO_API_URL", "http://127.0.0.1:1234/v1")
            if not lm_studio_url.endswith("/v1"):
                lm_studio_url = lm_studio_url.rstrip("/") + "/v1"
            lm_studio_model = os.getenv("LM_STUDIO_MODEL", "gpt-oss-20b")
            # Timeout set to 300s (5 min) for local inference - increase if needed
            lm_studio_timeout = float(os.getenv("LM_STUDIO_TIMEOUT", 300))
            self.llm = ChatOpenAI(
                base_url=lm_studio_url,
                api_key="sk-not-needed",  # LM Studio doesn't require actual key
                model=lm_studio_model,
                temperature=self.temperature,
                timeout=lm_studio_timeout,
                max_retries=2  # Retry up to 2 times on timeout
            )
            print(f"[OK] LLM Provider: Local LM Studio ({lm_studio_model})")
            print(f"     Endpoint: {lm_studio_url}/chat/completions")
            print(f"     Timeout: {lm_studio_timeout}s | Max Retries: 2")
        else:
            # Default to Gemini
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            self.llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=self.temperature)
            print(f"[OK] LLM Provider: Google Gemini ({gemini_model})")

        # Embeddings Configuration - FULLY LOCAL (No API calls)
        # Use HuggingFace sentence-transformers locally (384 dims)
        # This matches the existing Pinecone index dimensions
        # Using all-MiniLM-L6-v2 (384 dims)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Fast, lightweight, 384 dims (matches Pinecone index)
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.embedding_dimension = 384
        print("[OK] Embeddings: HuggingFace all-MiniLM-L6-v2 (FULLY LOCAL - NO API - 384 dims)")

config = AppConfig()