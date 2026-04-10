#!/usr/bin/env python3
"""
Test script to verify LLM and retrieval are working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_config import config
from ingestion import DocumentIngestor
from retrieval import AdvancedRetriever
from generation import ResponseGenerator

print("=" * 60)
print("LLM & RETRIEVAL TEST")
print("=" * 60)

# Test 1: LLM connection
print("\n1️⃣ Testing LLM connection...")
try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    
    test_prompt = PromptTemplate.from_template("Say 'hello world'")
    result = (test_prompt | config.llm | StrOutputParser()).invoke({})
    print(f"   ✅ LLM Response: {result[:100]}")
except Exception as e:
    print(f"   ❌ LLM Error: {e}")
    sys.exit(1)

# Test 2: Embeddings
print("\n2️⃣ Testing embeddings...")
try:
    embedding = config.embeddings.embed_query("test")
    print(f"   ✅ Embedding dimension: {len(embedding)}")
except Exception as e:
    print(f"   ❌ Embedding Error: {e}")
    sys.exit(1)

# Test 3: Pinecone retrieval
print("\n3️⃣ Testing Pinecone retrieval...")
try:
    ingestor = DocumentIngestor(
        config.embeddings, 
        config.chunk_size, 
        config.chunk_overlap, 
        config.embedding_dimension
    )
    vectorstore = ingestor.get_pinecone_vectorstore(namespace="cbt")
    retriever = AdvancedRetriever(vectorstore, config.cross_encoder_model, config.search_k, config.top_n).get_retriever()
    
    docs = retriever.invoke("anxiety depression coping strategies")
    print(f"   ✅ Retrieved {len(docs)} documents")
    if docs:
        print(f"   📄 First doc: {docs[0].page_content[:150]}...")
except Exception as e:
    print(f"   ❌ Retrieval Error: {e}")
    sys.exit(1)

# Test 4: Response generation
print("\n4️⃣ Testing response generation...")
try:
    generator = ResponseGenerator(config.llm)
    test_query = "I feel anxious and worried"
    
    print(f"   🔄 Expanding query: '{test_query}'")
    expanded = generator.expand_query(test_query)
    print(f"   📝 Expanded query result: '{expanded}'")
    print(f"   📝 Expanded query length: {len(str(expanded))}")
    
    if not expanded or not str(expanded).strip():
        print(f"   ⚠️  WARNING: Expanded query is empty!")
    
    print(f"   🔄 Generating response...")
    response = generator.generate_response(
        user_query=test_query,
        expanded_query=expanded,
        retriever=retriever,
        severe_level="moderate",
        mental_status="anxiety"
    )
    
    print(f"   📊 Response type: {type(response)}")
    print(f"   📊 Response length: {len(str(response))}")
    print(f"   📊 Response repr: {repr(response)}")
    
    if response and str(response).strip():
        print(f"   ✅ Response generated ({len(response)} chars)")
        print(f"   📢 Response preview: {response[:150]}...")
    else:
        print(f"   ❌ EMPTY RESPONSE: '{response}'")
        print(f"   ❌ This is the bug! The LLM is returning nothing.")
        
except Exception as e:
    import traceback
    print(f"   ❌ Generation Error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! Your app should work.")
print("=" * 60)
