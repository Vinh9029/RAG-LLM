import os
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

class AppConfig:
    """Centralized configuration management for the entire application."""
    def __init__(self):
        # Enable cache to optimize response time and reduce API costs
        set_llm_cache(InMemoryCache())
        
        # LLM Configuration (read from .env)
        self.llm_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.temperature = float(os.getenv("TEMPERATURE", 0.3))
        self.cross_encoder_model = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Document Chunking Configuration
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        
        # Retrieval Configuration
        self.search_k = int(os.getenv("SEARCH_K", 10))
        self.top_n = int(os.getenv("TOP_N", 3))
        
        # PDF Directory
        self.pdf_directory = os.getenv("PDF_DIRECTORY", "./mental_health_docs")
        
        # Initialize shared instances
        self.llm = ChatGoogleGenerativeAI(model=self.llm_model, temperature=self.temperature)

        # Local Embeddings: Sử dụng HuggingFace miễn phí chạy trên máy (Local)
        local_embedding_model = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embeddings = HuggingFaceEmbeddings(model_name=local_embedding_model)
        self.embedding_dimension = 384  # Dimension của all-MiniLM-L6-v2 là 384

config = AppConfig()