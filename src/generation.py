from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
import langdetect
from langdetect import detect, LangDetectException

class ResponseGenerator:
    """Responsible for prompt creation, memory management, and LLM calls."""
    # Initialize with LLM and memory for conversation history. Memory keeps the last 5 interactions for context.
    def __init__(self, llm):
        self.llm = llm
        self.memory = ConversationBufferWindowMemory(
            k=2, return_messages=True, memory_key="chat_history"
        )

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        Returns 'vi' for Vietnamese, 'en' for English, or defaults to 'en'.
        """
        try:
            detected_lang = detect(text)
            return detected_lang
        except LangDetectException:
            return 'en'  # Default to English if detection fails

    def expand_query(self, user_query: str) -> str:
        """
        Expand and clarify the user's query using LLM.
        Identifies mental health conditions (Depression, Anxiety, Bipolar, Suicidal, etc.)
        and adds relevant psychological keywords to optimize vector search.
        """
        rewrite_prompt = PromptTemplate.from_template(
            """You are an expert mental health professional and psychologist. 
            Your task is to analyze the user's question (which may be in any language), identify potential mental health conditions, and generate relevant psychological keywords in ENGLISH to optimize searching in a medical database.
            
            
            Examples:
            - "Tôi cảm thấy rất buồn và tuyệt vọng" -> "depression, low mood, hopelessness, anhedonia, negative thoughts, persistent sadness"
            - "I'm worried about everything" -> "anxiety, generalized anxiety disorder, worry, nervousness, panic, stress"
            
            Return ONLY the expanded ENGLISH keywords without explanations.
            
            Original user query: {query}
            Expanded English query:"""
        )
        
        chain = rewrite_prompt | self.llm | StrOutputParser()
        expanded_query = chain.invoke({"query": user_query})
        return expanded_query

    def generate_response(self, user_query: str, expanded_query: str, docs: list, severe_level: str, mental_status: str):
        """
        Generate the final response using retrieved context and conversation history.
        Converts response to original language if user queried in Vietnamese.
        """
        context = "\n\n".join([doc.page_content for doc in docs])
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        
        system_prompt = """You are a compassionate, empathetic, and non-judgmental Virtual Assistant for Mental Health Support.
        Your role is to provide evidence-based advice, therapeutic exercises, and coping strategies based on the provided context.

        User's severe level: {severe_level}
        User's mental health status: {mental_status}
        
        CRITICAL SAFETY RULES:
        1. If the user shows signs of suicidal ideation or self-harm, IMMEDIATELY provide crisis helpline numbers (e.g., National Suicide Prevention Lifeline: 988 in the US, or equivalent in the user's country) BEFORE any other response.
        2. Always remind users that you are an AI assistant and cannot replace professional mental health care from licensed therapists or psychiatrists.
        3. Respond with warmth, positivity, and hope. If the context lacks relevant information, acknowledge this and recommend consulting a mental health professional.
        4. Respect cultural and individual differences in mental health experiences.
        5. Never provide medical diagnoses; instead, suggest symptoms to discuss with a healthcare provider.
        6. IMPORTANT: You MUST respond in the SAME LANGUAGE as the user's original query. For example, if the user asks in Vietnamese, your entire response must be in Vietnamese, interpreting the English context appropriately.
        
        Retrieved context for reference:
        {context}"""
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "User: {query}")
        ])
        
        prompt_value = prompt_template.invoke({
            "context": context,
            "chat_history": chat_history,
            "query": expanded_query,  # Use expanded query for better semantic understanding
            "severe_level": severe_level,
            "mental_status": mental_status
        })
        
        formatted_prompt = prompt_value.to_string()
        response = (self.llm | StrOutputParser()).invoke(prompt_value)
        
        # Save original user query to memory for consistency
        self.memory.save_context({"input": user_query}, {"output": response})   
        
        return response, formatted_prompt
