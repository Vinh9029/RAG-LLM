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
            k=5, return_messages=True, memory_key="chat_history"
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

    def translate_query_to_english(self, query: str) -> str:
        """
        If query is in Vietnamese, translate it to English.
        Uses LLM as translator to maintain context awareness.
        """
        lang = self.detect_language(query)
        
        if lang == 'vi':
            translation_prompt = PromptTemplate.from_template(
                """You are a professional translator. 
                Translate the following Vietnamese text to English while preserving the original meaning and medical/psychological context.
                
                Vietnamese text: {query}
                English translation:"""
            )
            chain = translation_prompt | self.llm | StrOutputParser()
            translated_query = chain.invoke({"query": query})
            return translated_query
        
        # Already in English, return as is
        return query

    def expand_query(self, user_query: str) -> str:
        """
        Expand and clarify the user's query using LLM.
        Identifies mental health conditions (Depression, Anxiety, Bipolar, Suicidal, etc.)
        and adds relevant psychological keywords to optimize vector search.
        
        This uses the translated (English) version for consistency with English PDF documents.
        """
        # First, translate Vietnamese query to English
        english_query = self.translate_query_to_english(user_query)
        
        rewrite_prompt = PromptTemplate.from_template(
            """You are an expert mental health professional and psychologist. 
            Your task is to expand and clarify the user's question to optimize searching in a medical/psychological database.
            
            Identify which mental health condition the user might be experiencing (e.g., Depression, Anxiety, PTSD, Bipolar Disorder, Personality Disorder, Suicidal Ideation, Stress, etc.)
            and add relevant psychological keywords accordingly.
            
            Examples:
            - "I feel sad and hopeless" -> "depression, low mood, hopelessness, anhedonia, negative thoughts, persistent sadness"
            - "I'm worried about everything" -> "anxiety, generalized anxiety disorder, worry, nervousness, panic, stress"
            - "I want to hurt myself" -> "suicidal ideation, self-harm, self-injury, suicide risk, mental health crisis"
            
            Return ONLY the expanded query without explanations.
            
            Original user query: {query}
            Expanded query:"""
        )
        
        chain = rewrite_prompt | self.llm | StrOutputParser()
        expanded_query = chain.invoke({"query": english_query})
        return expanded_query

    def generate_response(self, user_query: str, expanded_query: str, retriever) -> str:
        """
        Generate the final response using retrieved context and conversation history.
        Converts response to original language if user queried in Vietnamese.
        """
        docs = retriever.invoke(expanded_query)
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
        
        Retrieved context for reference:
        {context}"""
        
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "User: {query}")
        ])
        
        response = (prompt_template | self.llm | StrOutputParser()).invoke({
            "context": context,
            "chat_history": chat_history,
            "query": expanded_query  # Use expanded query for better semantic understanding
        })
        
        # Save original user query to memory for consistency
        self.memory.save_context({"input": user_query}, {"output": response})
        
        # Translate response back to Vietnamese if user queried in Vietnamese
        user_lang = self.detect_language(user_query)
        if user_lang == 'vi':
            response = self.translate_response_to_vietnamese(response)
        
        return response

    def translate_response_to_vietnamese(self, response: str) -> str:
        """
        Translate English response back to Vietnamese for Vietnamese-speaking users.
        """
        translation_prompt = PromptTemplate.from_template(
            """You are a professional translator specializing in medical and psychological terminology.
            Translate the following English mental health support response to Vietnamese.
            Preserve all medical/psychological terms accurately and maintain the tone of empathy and support.
            
            English text:
            {response}
            
            Vietnamese translation:"""
        )
        chain = translation_prompt | self.llm | StrOutputParser()
        translated_response = chain.invoke({"response": response})
        return translated_response
