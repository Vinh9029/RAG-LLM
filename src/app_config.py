import os
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

class AppConfig:
    """Centralized configuration management for the entire application."""
    def __init__(self):
        # Enable cache to optimize response time and reduce API costs
        set_llm_cache(InMemoryCache())
        
        # LLM Type Selection (GEMINI or LOCAL)
        self.lm_type = os.getenv("LM_TYPE", "GEMINI").upper()
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
        
        # Initialize LLM based on type
        if self.lm_type == "LOCAL":
            self.llm = self._initialize_local_llm()
            print(f"✓ Initialized Local LLM: {os.getenv('LOCAL_MODEL_NAME', 'qwen/qwen3.5-9b')}")
        else:
            self.llm = self._initialize_gemini_llm()
            print(f"✓ Initialized Gemini LLM: {os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')}")
        
        # Initialize Embeddings: Using local sentence-transformers model (384-dim)
        from langchain_huggingface import HuggingFaceEmbeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_dimension = 384
    
    def _initialize_gemini_llm(self):
        """Initialize Google Gemini LLM."""
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        return ChatGoogleGenerativeAI(model=gemini_model, temperature=self.temperature)
    
    def _initialize_local_llm(self):
        """Initialize Local LLM using LM Studio (OpenAI-compatible API)."""
        from langchain_openai import ChatOpenAI
        local_model = os.getenv("LOCAL_MODEL_NAME", "qwen/qwen3.5-9b")
        local_base_url = os.getenv("LOCAL_LM_BASE_URL", "http://127.0.0.1:1234/v1")
        return ChatOpenAI(
            model=local_model,
            base_url=local_base_url,
            api_key="not-needed",
            temperature=self.temperature
        )

config = AppConfig()

config = AppConfig()