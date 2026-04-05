from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
import os
from dotenv import load_dotenv

class DocumentIngestor:
    """Chịu trách nhiệm Load tài liệu và tạo Vector Database trên Pinecone."""
    def __init__(self, embeddings, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        load_dotenv()
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index = os.getenv("PINECONE_INDEX")
        pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)

    def process_pdf_directory_to_pinecone(self, pdf_directory: str, namespace: str = "cbt") -> PineconeVectorStore:
        loader = PyPDFDirectoryLoader(pdf_directory)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        # Create index if not exists
        if self.pinecone_index not in pinecone.list_indexes():
            pinecone.create_index(self.pinecone_index, dimension=1536, metric="cosine")
        vectorstore = PineconeVectorStore.from_documents(
            chunks,
            embedding=self.embeddings,
            index_name=self.pinecone_index,
            namespace=namespace
        )
        print(f"[Ingestion] Ingested {len(chunks)} chunks to Pinecone index '{self.pinecone_index}' (namespace: {namespace})")
        return vectorstore

    def get_pinecone_vectorstore(self, namespace: str = "cbt") -> PineconeVectorStore:
        return PineconeVectorStore(
            index_name=self.pinecone_index,
            embedding=self.embeddings,
            namespace=namespace
        )