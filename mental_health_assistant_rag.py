import os
from langchain.docstore.document import Document
from pipeline import MentalHealthPipeline

# Langchain Core & Globals
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Memory
from langchain.memory import ConversationBufferWindowMemory

# Document Loaders & Splitters
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector Store & Embeddings
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI # Bạn có thể thay bằng LLM của bạn sau
from dotenv import load_dotenv
# Retrievers & Rerankers
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ---------------------------------------------------------
# 1. CẤU HÌNH BAN ĐẦU & CACHING
# ---------------------------------------------------------
# Bật Prompt Caching trên bộ nhớ RAM để tăng tốc độ phản hồi và giảm chi phí API
set_llm_cache(InMemoryCache())

# Thiết lập LLM (Ở đây dùng OpenAI làm ví dụ, bạn có thể thay bằng Claude, Ollama, DeepSeek...)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings()

# Bộ nhớ cho hội thoại (Lưu 5 lượt gần nhất để tránh tràn context window)
memory = ConversationBufferWindowMemory(
    k=5, 
    return_messages=True, 
    memory_key="chat_history"
)

# ---------------------------------------------------------
# 2. TEXT SPLITTING & CHUNKING (Tối ưu cho tài liệu Mental Health)
# ---------------------------------------------------------
def process_pdf_documents_to_pinecone(pdf_directory: str, namespace: str = "cbt"):
    loader = PyPDFDirectoryLoader(pdf_directory)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    load_dotenv()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
    pinecone_index = os.getenv("PINECONE_INDEX")
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
    if pinecone_index not in pinecone.list_indexes():
        pinecone.create_index(pinecone_index, dimension=1536, metric="cosine")
    vectorstore = PineconeVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        index_name=pinecone_index,
        namespace=namespace
    )
    print(f"[Ingestion] Ingested {len(chunks)} chunks to Pinecone index '{pinecone_index}' (namespace: {namespace})")
    return vectorstore

# ---------------------------------------------------------
# 3. QUERY REWRITE / EXPANSION
# ---------------------------------------------------------
def expand_query(user_query: str) -> str:
    """
    Dùng LLM để viết lại và mở rộng câu hỏi của user.
    Bổ sung các từ khóa, triệu chứng liên quan đến Depression, Anxiety, Bipolar, v.v.
    giúp search vector DB chính xác hơn.
    """
    rewrite_prompt = PromptTemplate.from_template(
        """Bạn là một chuyên gia tâm lý học. Nhiệm vụ của bạn là mở rộng và làm rõ câu hỏi của người dùng 
        để tối ưu hóa việc tìm kiếm thông tin trong cơ sở dữ liệu y khoa/tâm lý học. 
        Hãy xác định xem người dùng có thể đang ở trạng thái nào (Depression, Anxiety, Normal, Personality Disorder, Bipolar, Suicidal) 
        để thêm các từ khóa chuyên ngành tương ứng (ví dụ: "buồn chán" -> "trầm cảm, mất động lực, suy nghĩ tiêu cực, depression").
        Chỉ trả về câu query đã được mở rộng, không cần giải thích thêm.
        
        Câu hỏi gốc của người dùng: {query}
        Câu query mở rộng:"""
    )
    
    chain = rewrite_prompt | llm | StrOutputParser()
    expanded_query = chain.invoke({"query": user_query})
    return expanded_query

# ---------------------------------------------------------
# 4. EMBEDDING + RETRIEVAL + RERANK
# ---------------------------------------------------------
def get_reranked_retriever(vectorstore):
    """
    Truy xuất (Retrieval) sau đó Rerank lại các kết quả.
    - Đầu tiên lấy top 10 văn bản có tính tương đồng cao nhất (Vector Search).
    - Sau đó dùng CrossEncoder để chấm điểm (Rerank) và chọn ra top 3 phù hợp nhất với ngữ cảnh.
    """
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # Sử dụng Cross Encoder mã nguồn mở (HuggingFace) để rerank (Không tốn phí API thêm)
    # Bạn có thể thay thế bằng CohereRerank nếu có API key của Cohere
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever

# ---------------------------------------------------------
# 5. CONTEXT COMBINATION & FINAL LLM GENERATION
# ---------------------------------------------------------
def generate_final_response(user_query: str, expanded_query: str, retriever):
    """
    Kết hợp Context + Query + Memory để tạo câu trả lời cuối cùng.
    Chú trọng vào prompt engineering cho trợ lý tâm lý học.
    """
    # Lấy tài liệu (Dùng câu query đã mở rộng để search)
    docs = retriever.invoke(expanded_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Lấy lịch sử trò chuyện từ bộ nhớ
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # Prompt hệ thống được thiết kế đặc biệt cho Sức khỏe tinh thần
    system_prompt = """Bạn là một Trợ lý ảo Hỗ trợ Sức khỏe Tinh thần (Virtual Assistant for Mental Health Support) đầy thấu cảm, tử tế và không phán xét.
    Dựa vào các ngữ cảnh (Context) được cung cấp, hãy đưa ra lời khuyên, bài tập hoặc phương pháp hỗ trợ phù hợp.
    
    QUY TẮC AN TOÀN QUAN TRỌNG:
    1. Nếu người dùng có biểu hiện của nhãn "Suicidal" (Ý định tự tử) hoặc tự hại, BẮT BUỘC phải cung cấp đường dây nóng hỗ trợ khủng hoảng (ví dụ: số điện thoại cấp cứu, tổng đài 111 ở Việt Nam) trước khi nói bất cứ điều gì khác.
    2. Nhắc nhở người dùng rằng bạn là Trợ lý AI, không thể thay thế bác sĩ tâm lý chuyên nghiệp.
    3. Trả lời một cách nhẹ nhàng, tích cực. Nếu ngữ cảnh không có thông tin, hãy nói bạn không có đủ thông tin và khuyên họ tìm chuyên gia.
    
    Ngữ cảnh tham khảo (Context):
    {context}
    """
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Người dùng: {query}")
    ])
    
    # Khởi tạo Chain
    generation_chain = prompt_template | llm | StrOutputParser()
    
    # Tạo câu trả lời
    response = generation_chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "query": user_query # Trả lời dựa trên câu hỏi gốc của user để giữ tính tự nhiên
    })
    
    # Cập nhật Memory sau khi có phản hồi
    memory.save_context({"input": user_query}, {"output": response})
    
    return response

# ---------------------------------------------------------
# 6. ĐÓNG GÓI THÀNH PIPELINE CHÍNH (MAIN WORKFLOW)
# ---------------------------------------------------------
def mental_health_rag_pipeline(user_query: str, vectorstore):
    print("\n" + "="*50)
    print(f"User Query: {user_query}")
    
    # Bước 1: Query Rewrite / Expansion
    expanded_query = expand_query(user_query)
    print(f"[Log] Expanded Query: {expanded_query}")
    
    # Bước 2 & 3: Retrieval & Rerank
    retriever = get_reranked_retriever(vectorstore)
    
    # Bước 4 & 5: Context + LLM Generation
    final_answer = generate_final_response(user_query, expanded_query, retriever)
    
    return final_answer

# ---------------------------------------------------------
# DEMO SỬ DỤNG
# ---------------------------------------------------------
if __name__ == "__main__":
    # CHÚ Ý: Đảm bảo bạn có folder 'mental_health_docs' chứa các file PDF tài liệu của bạn
    pdf_folder_path = "./mental_health_docs"
    
    # Tạo thư mục tạm để chạy demo nếu chưa có
    os.makedirs(pdf_folder_path, exist_ok=True)
    
    # Khởi tạo RAG Pipeline bằng OOP
    assistant = MentalHealthPipeline()

    # Khởi tạo Vector Database (Chạy 1 lần hoặc lưu lại dùng dần)
    print("Đang xử lý tài liệu PDF và xây dựng Vector DB...")
    # Bỏ comment dòng dưới khi có file PDF
    # vectorstore = process_pdf_documents(pdf_folder_path)
    
    # Dùng tài liệu giả lập cho Demo nếu thư mục trống
    from langchain.docstore.document import Document
    # Dùng tài liệu giả lập cho Demo
    mock_docs = [
        Document(page_content="Bài tập thở 4-7-8 giúp giảm lo âu (Anxiety). Cách làm: Hít vào 4s, giữ 7s, thở ra 8s."),
        Document(page_content="Khi ai đó có suy nghĩ tự tử (Suicidal), hãy lắng nghe họ không phán xét và khuyên họ gọi ngay cho tổng đài phòng chống tự tử quốc gia 111 hoặc đường dây nóng cấp cứu 115.")
    ]
    vectorstore = FAISS.from_documents(mock_docs, embeddings)
    assistant.load_documents(mock_docs=mock_docs)
    # Thực tế bạn sẽ bỏ comment dòng dưới khi có PDF:
    # assistant.load_documents(pdf_directory=pdf_folder_path)

    print("Đã tải xong Vector DB!")
    
    # Mô phỏng quá trình chat
    while True:
        query = input("\nBạn đang cảm thấy thế nào? (Nhập 'quit' để thoát): ")
        if query.lower() == 'quit':
            break
            
        answer = mental_health_rag_pipeline(query, vectorstore)
        answer = assistant.chat(query)
        print("\nTrợ lý ảo Mental Health:")
        print(answer)