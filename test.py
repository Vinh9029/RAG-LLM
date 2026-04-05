import time
from langdetect import detect
from src.config import config
from src.ingestion import DocumentIngestor
from src.retrieval import AdvancedRetriever
from src.generation import ResponseGenerator
from opening_questions import get_opening_questions

# Mock data for quick test (bạn có thể thay bằng PDF thật)
MOCK_DOCS = [
    "The 4-7-8 breathing exercise helps reduce anxiety. Inhale for 4s, hold for 7s, exhale for 8s.",
    "If someone has suicidal thoughts, listen non-judgmentally and advise them to call the national suicide prevention hotline 111 or emergency 115.",
    "Cognitive Behavioral Therapy (CBT) is effective for depression and anxiety disorders."
]

def translate_with_llm(llm, text, src_lang, tgt_lang):
    prompt = f"Translate this to {tgt_lang}: {text}"
    return llm.invoke(prompt)

def main():
    print("=== Mental Health RAG Pipeline Test ===\n")
    ingest = DocumentIngestor(config.embeddings, config.chunk_size, config.chunk_overlap)
    vectorstore = ingest.process_texts(MOCK_DOCS)
    retriever = AdvancedRetriever(vectorstore, config.cross_encoder_model, config.search_k, config.top_n).get_retriever()
    generator = ResponseGenerator(config.llm)
    #once asking question
    severe_level = "moderate"
    mental_status = "anxiety"
    print("Question from pooling:\n")
    questions = get_opening_questions(severe_level, mental_status)

    while True:
        user_query = input("\nYour question (type 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
        # Detect language
        lang = detect(user_query)
        print(f"[Log] Detected language: {lang}")
        translated_query = user_query
        if lang == 'vi':
            translated_query = translate_with_llm(config.llm, user_query, 'Vietnamese', 'English')
            print(f"[Log] Translated query: {translated_query}")
        else:
            print("[Log] No translation needed.")
        # Query expansion
        expanded_query = generator.expand_query(translated_query)
        print(f"[Log] Expanded query: {expanded_query}")
        # Timing vector search
        t0 = time.time()
        docs = retriever.invoke(expanded_query)
        t1 = time.time()
        print(f"[Log] Vector DB search time: {t1-t0:.3f} seconds")
        context = "\n\n".join([doc.page_content for doc in docs])
        # Compose prompt
        prompt = f"System prompt: Virtual Assistant for Mental Health Support.\nContext: {context}\nUser: {user_query}"
        print(f"[Log] Final prompt to LLM:\n{prompt}")
        # Generate answer
        answer = generator.generate_response(user_query, expanded_query, retriever)
        print(f"\n[Official Answer]:\n{answer}")

if __name__ == "__main__":
    main()
