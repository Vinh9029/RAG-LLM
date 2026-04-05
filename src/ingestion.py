from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


class DocumentIngestor:
    """Chịu trách nhiệm Load tài liệu và tạo Vector Database."""
    def __init__(self, embeddings, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

    def process_pdf_directory(self, pdf_directory: str) -> FAISS:
        loader = PyPDFDirectoryLoader(pdf_directory)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        return FAISS.from_documents(chunks, self.embeddings)
    
    def process_texts(self, texts: list) -> FAISS:
        """
        Encode a list of plain texts (each as a document) and create a FAISS vectorstore.
        Useful for ingesting external data sources that are not PDF.
        """
        docs = [Document(page_content=txt) for txt in texts]
        chunks = self.text_splitter.split_documents(docs)
        print(f"[Ingestion] Number of chunks created from texts: {len(chunks)}")
        return FAISS.from_documents(chunks, self.embeddings)


    def create_mock_vectorstore(self, mock_docs) -> FAISS:
        """Hàm phụ trợ cho Demo."""
        return FAISS.from_documents(mock_docs, self.embeddings)
