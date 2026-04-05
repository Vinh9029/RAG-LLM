from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory

class ResponseGenerator:
    """Chịu trách nhiệm tạo Prompts, quản lý Memory và gọi LLM."""
    def __init__(self, llm):
        self.llm = llm
        self.memory = ConversationBufferWindowMemory(
            k=5, return_messages=True, memory_key="chat_history"
        )

    def expand_query(self, user_query: str) -> str:
        rewrite_prompt = PromptTemplate.from_template(
            """Bạn là một chuyên gia tâm lý học. Nhiệm vụ của bạn là mở rộng và làm rõ câu hỏi của người dùng 
            để tối ưu hóa việc tìm kiếm thông tin trong cơ sở dữ liệu y khoa/tâm lý học. 
            Hãy xác định xem người dùng có thể đang ở trạng thái nào (Depression, Anxiety, Normal, Personality Disorder, Bipolar, Suicidal) 
            để thêm các từ khóa chuyên ngành tương ứng (ví dụ: "buồn chán" -> "trầm cảm, mất động lực, suy nghĩ tiêu cực, depression").
            Chỉ trả về câu query đã được mở rộng, không cần giải thích thêm.
            
            Câu hỏi gốc của người dùng: {query}
            Câu query mở rộng:"""
        )
        chain = rewrite_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": user_query})

    def generate_response(self, user_query: str, expanded_query: str, retriever) -> str:
        docs = retriever.invoke(expanded_query)
        context = "\n\n".join([doc.page_content for doc in docs])
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        system_prompt = """Bạn là một Trợ lý ảo Hỗ trợ Sức khỏe Tinh thần (Virtual Assistant for Mental Health Support) đầy thấu cảm, tử tế và không phán xét.
        Dựa vào các ngữ cảnh (Context) được cung cấp, hãy đưa ra lời khuyên, bài tập hoặc phương pháp hỗ trợ phù hợp.
        
        QUY TẮC AN TOÀN QUAN TRỌNG:
        1. Nếu người dùng có biểu hiện của nhãn "Suicidal" (Ý định tự tử) hoặc tự hại, BẮT BUỘC phải cung cấp đường dây nóng hỗ trợ khủng hoảng (ví dụ: số điện thoại cấp cứu, tổng đài 111 ở Việt Nam) trước khi nói bất cứ điều gì khác.
        2. Nhắc nhở người dùng rằng bạn là Trợ lý AI, không thể thay thế bác sĩ tâm lý chuyên nghiệp.
        3. Trả lời một cách nhẹ nhàng, tích cực. Nếu ngữ cảnh không có thông tin, hãy nói bạn không có đủ thông tin và khuyên họ tìm chuyên gia.
        
        Ngữ cảnh tham khảo (Context):
        {context}"""
        
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Người dùng: {query}")
        ])
        
        response = (prompt_template | self.llm | StrOutputParser()).invoke({"context": context, "chat_history": chat_history, "query": user_query})
        self.memory.save_context({"input": user_query}, {"output": response})
        return response
