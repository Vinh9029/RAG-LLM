from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

class AppConfig:
    """Lưu trữ tập trung các cấu hình của toàn bộ ứng dụng."""
    def __init__(self):
        # Bật Cache để tối ưu thời gian phản hồi và API
        set_llm_cache(InMemoryCache())
        
        # Cấu hình Model
        self.llm_model = "gpt-4o-mini"
        self.temperature = 0.3
        self.cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        # Cấu hình Document Chunking
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Khởi tạo các Shared Instance
        self.llm = ChatOpenAI(model=self.llm_model, temperature=self.temperature)
        self.embeddings = OpenAIEmbeddings()

config = AppConfig()
