# Simplified retriever to avoid langchain version conflicts
from langchain_community.vectorstores import Pinecone as PineconeVectorStore

class AdvancedRetriever:
    """Simplified retriever for Pinecone vectorstore."""
    def __init__(self, vectorstore, model_name: str, search_k: int = 10, top_n: int = 3):
        self.vectorstore = vectorstore
        self.search_k = search_k
        self.top_n = top_n
        self.model_name = model_name

    def get_retriever(self):
        """Return a basic retriever."""
        return self.vectorstore.as_retriever(search_kwargs={"k": self.top_n})
    
    def invoke(self, query: str):
        """Retrieve documents without reranking to avoid version conflicts."""
        retriever = self.get_retriever()
        return retriever.invoke(query)