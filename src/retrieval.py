from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_pinecone import PineconeVectorStore

class AdvancedRetriever:
    """Chịu trách nhiệm tìm kiếm Vector và tinh chỉnh hạng (Rerank) trên Pinecone.
    
    Dimensions:
    - Embedding: 384 dims (all-MiniLM-L6-v2) - matches Pinecone index
    - Cross-Encoder: Local HuggingFace model
    """
    def __init__(self, vectorstore, model_name: str, search_k: int = 10, top_n: int = 3):
        self.vectorstore = vectorstore
        self.search_k = search_k
        self.top_n = top_n
        self.model_name = model_name

    def get_retriever(self) -> ContextualCompressionRetriever:
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.search_k})
        model = HuggingFaceCrossEncoder(model_name=self.model_name)
        compressor = CrossEncoderReranker(model=model, top_n=self.top_n)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )