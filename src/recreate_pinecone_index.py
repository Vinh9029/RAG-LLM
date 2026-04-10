#!/usr/bin/env python3
"""
Script to recreate Pinecone index with new 768-dimensional embeddings.
This deletes the old index and creates a new one from your PDF documents.
"""

import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load env vars FIRST
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_config import config
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

def recreate_index():
    """Delete old index and recreate with new embeddings."""
    
    # Read config from env
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX", "mental-health-cbt")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    pdf_directory_env = os.getenv("PDF_DIRECTORY", "./src/mental_health_docs")
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    
    print(f"🔍 Current embeddings dimension: {config.embedding_dimension}")
    print(f"📦 Index name: {index_name}")
    print(f"📁 PDF directory: {pdf_directory_env}")
    
    # Resolve relative path
    if pdf_directory_env.startswith("./"):
        pdf_directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), pdf_directory_env.replace("./", ""))
    else:
        pdf_directory = pdf_directory_env
    
    # Check if PDF directory exists
    if not os.path.exists(pdf_directory):
        print(f"❌ PDF directory not found: {pdf_directory}")
        print(f"   Absolute path tried: {os.path.abspath(pdf_directory)}")
        return False
    
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    print(f"   Found {len(pdf_files)} PDF files: {[f.name for f in pdf_files]}")
    
    if not pdf_files:
        print(f"❌ No PDF files found in {pdf_directory}")
        return False
    
    # Step 1: Delete existing index
    print(f"\n🗑️  Deleting existing index '{index_name}'...")
    existing_indexes = pc.list_indexes().names()
    
    if index_name in existing_indexes:
        pc.delete_index(index_name)
        print(f"✅ Index '{index_name}' deleted successfully")
        time.sleep(3)  # Wait for deletion to complete
    else:
        print(f"⚠️  Index '{index_name}' does not exist (no deletion needed)")
    
    # Step 2: Recreate index and ingest documents
    print(f"\n📒 Creating new index and ingesting documents...")
    print(f"   - Dimension: {config.embedding_dimension}")
    print(f"   - Metric: cosine")
    
    # Create the index manually
    print(f"   Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=config.embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=pinecone_env)
    )
    print(f"   ✅ Index created")
    
    time.sleep(3)  # Wait for index creation
    
    # Now ingest documents manually
    print(f"   📄 Loading PDFs from {pdf_directory}...")
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()
    print(f"   Loaded {len(documents)} documents")
    
    print(f"   🔪 Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")
    
    print(f"   🔀 Embedding and uploading chunks...")
    # Use add_texts approach instead of from_documents
    vectorstore = PineconeVectorStore(
        index=pc.Index(index_name),
        embedding=config.embeddings,
        namespace="cbt",
        text_key="page_content"
    )
    
    # Add documents in batches to avoid issues
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata for doc in chunks]
    
    # Add in batches of 100
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)
        print(f"   Uploaded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
    
    print(f"\n✅ Index recreation completed successfully!")
    print(f"   - Index: {index_name}")
    print(f"   - Dimension: {config.embedding_dimension}")
    print(f"   - Namespace: cbt")
    print(f"   - Total chunks: {len(chunks)}")
    
    return True

if __name__ == "__main__":
    try:
        print("=" * 60)
        print("PINECONE INDEX RECREATION TOOL")
        print("=" * 60)
        
        success = recreate_index()
        
        if success:
            print("\n" + "=" * 60)
            print("✨ Ready to use your app with new embeddings!")
            print("=" * 60)
            sys.exit(0)
        else:
            print("\n" + "=" * 60)
            print("❌ Recreation failed!")
            print("=" * 60)
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error during index recreation: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
